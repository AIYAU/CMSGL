from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cc_sgcl import CCSGCLConfig, CCSGCLModel
from cc_sgcl.data import build_loader, infer_modal_channels, load_dataset, stratified_known_train_val_split
from cc_sgcl.inference import build_open_set_ground_truth, split_hsi_lidar_patches, to_nchw
from cc_sgcl.metrics import compute_additional_metrics
from cc_sgcl.utils import seed_torch, to_categorical
from third_party.hyliosr_compat import make_sample, rscls

DATASET_LABELS = {1: 'Trento', 2: 'Houston', 3: 'MUUFL'}
DATASET_PREFIX = {1: 'trento', 2: 'houston', 3: 'muufl'}
DEFAULT_QUANTILES = [0.80, 0.85, 0.90, 0.92, 0.95]
DEFAULT_ALPHAS = [0.25, 0.5, 1.0, 1.5, 2.0]
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Offline score-calibration diagnostic for final S2G-CCSGCL checkpoints')
    parser.add_argument('--input_root', type=str, default=str(PROJECT_ROOT / 'outputs' / 'final_candidate_h2l_eta10_lcf000_3datasets_seed5'))
    parser.add_argument('--output_root', type=str, default=str(PROJECT_ROOT / 'outputs' / 'score_calibration_s2g_3datasets'))
    parser.add_argument('--datasets', type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--quantiles', type=float, nargs='+', default=DEFAULT_QUANTILES)
    parser.add_argument('--alphas', type=float, nargs='+', default=DEFAULT_ALPHAS)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_per_sample', type=str, default='true')
    return parser.parse_args()


def str2bool(value: str) -> bool:
    return str(value).strip().lower() in {'1', 'true', 'yes', 'y'}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f'No rows to write for {path}')
    with path.open('w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def latest_run_dir(data_dir: Path, data_id: int, seed: int) -> Path:
    candidates = sorted(data_dir.glob(f'data{data_id}_seed{seed}_*'), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f'No run dir found for data={data_id}, seed={seed} under {data_dir}')
    return candidates[-1]


def labels_to_indices(y: torch.Tensor) -> torch.Tensor:
    if y.ndim == 2:
        return torch.argmax(y, dim=1).long()
    if y.ndim == 1:
        return y.long()
    raise ValueError(f'Unsupported label shape: {tuple(y.shape)}')


def gather_predicted_components(outputs: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    pred_idx = outputs['energies'].argmin(dim=1)
    gather_idx = pred_idx.view(-1, 1)
    return {
        'pred_known_class': pred_idx.detach().cpu().numpy().astype(np.int64) + 1,
        'total_energy_score': outputs['E_total'].gather(1, gather_idx).detach().cpu().numpy().reshape(-1),
        'compatibility_score': outputs['E_compat'].gather(1, gather_idx).detach().cpu().numpy().reshape(-1),
        'hsi_score': outputs['E_h'].gather(1, gather_idx).detach().cpu().numpy().reshape(-1),
        'lidar_score': outputs['E_l'].gather(1, gather_idx).detach().cpu().numpy().reshape(-1),
    }


def collect_val_known_details(model: torch.nn.Module, val_loader, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    collected: dict[str, list[np.ndarray]] = {
        'gt_known_label': [],
        'pred_known_class': [],
        'total_energy_score': [],
        'compatibility_score': [],
        'hsi_score': [],
        'lidar_score': [],
    }
    with torch.no_grad():
        for x_h, x_l, y in val_loader:
            x_h = x_h.float().to(device)
            x_l = x_l.float().to(device)
            gt = labels_to_indices(y).cpu().numpy().astype(np.int64) + 1
            outputs = model(x_h, x_l)
            comps = gather_predicted_components(outputs)
            collected['gt_known_label'].append(gt)
            for key in ('pred_known_class', 'total_energy_score', 'compatibility_score', 'hsi_score', 'lidar_score'):
                collected[key].append(comps[key])
    return {key: np.concatenate(value, axis=0) for key, value in collected.items()}


def collect_test_details(model: torch.nn.Module, sampler, data_id: int, device: torch.device, gt_eval: np.ndarray, unknown_label: int) -> dict[str, np.ndarray]:
    model.eval()
    row_count, _ = sampler.gt.shape
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    pred_known_rows: list[np.ndarray] = []
    total_rows: list[np.ndarray] = []
    compat_rows: list[np.ndarray] = []
    hsi_rows: list[np.ndarray] = []
    lidar_rows: list[np.ndarray] = []

    with torch.no_grad():
        for r in range(row_count):
            row_samples = sampler.all_sample_row(r)
            x_h_np, x_l_np = split_hsi_lidar_patches(row_samples, data_id=data_id)
            x_h = torch.from_numpy(to_nchw(x_h_np)).to(device)
            x_l = torch.from_numpy(to_nchw(x_l_np)).to(device)
            outputs = model(x_h, x_l)
            comps = gather_predicted_components(outputs)
            n = comps['pred_known_class'].shape[0]
            rows.append(np.full(n, r, dtype=np.int64))
            cols.append(np.arange(n, dtype=np.int64))
            pred_known_rows.append(comps['pred_known_class'])
            total_rows.append(comps['total_energy_score'])
            compat_rows.append(comps['compatibility_score'])
            hsi_rows.append(comps['hsi_score'])
            lidar_rows.append(comps['lidar_score'])

    row_full = np.stack(rows, axis=0)
    col_full = np.stack(cols, axis=0)
    pred_known_full = np.stack(pred_known_rows, axis=0)
    total_full = np.stack(total_rows, axis=0)
    compat_full = np.stack(compat_rows, axis=0)
    hsi_full = np.stack(hsi_rows, axis=0)
    lidar_full = np.stack(lidar_rows, axis=0)
    labeled_mask = gt_eval != 0

    return {
        'row_labeled': row_full[labeled_mask].astype(np.int64),
        'col_labeled': col_full[labeled_mask].astype(np.int64),
        'gt_label_labeled': gt_eval[labeled_mask].astype(np.int64),
        'gt_known_unknown_labeled': np.where(gt_eval[labeled_mask] == unknown_label, 'unknown', 'known'),
        'pred_known_class_labeled': pred_known_full[labeled_mask].astype(np.int64),
        'total_energy_score_labeled': total_full[labeled_mask],
        'compatibility_score_labeled': compat_full[labeled_mask],
        'hsi_score_labeled': hsi_full[labeled_mask],
        'lidar_score_labeled': lidar_full[labeled_mask],
        'pred_known_class_full': pred_known_full.astype(np.int64),
        'total_energy_score_full': total_full,
        'compatibility_score_full': compat_full,
        'hsi_score_full': hsi_full,
        'lidar_score_full': lidar_full,
    }


def zscore(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    scale = std if std > EPS else 1.0
    return (values - mean) / scale


def build_score_variants(test_details: dict[str, np.ndarray], val_details: dict[str, np.ndarray], alphas: Iterable[float]) -> dict[tuple[str, str], dict[str, np.ndarray | float]]:
    test_total_full = test_details['total_energy_score_full']
    test_compat_full = test_details['compatibility_score_full']
    val_total = val_details['total_energy_score']
    val_compat = val_details['compatibility_score']

    total_mean = float(np.mean(val_total))
    total_std = float(np.std(val_total))
    compat_mean = float(np.mean(val_compat))
    compat_std = float(np.std(val_compat))

    variants: dict[tuple[str, str], dict[str, np.ndarray | float]] = {
        ('total_score', ''): {
            'test_score_full': test_total_full,
            'val_score': val_total,
        },
        ('compat_score', ''): {
            'test_score_full': test_compat_full,
            'val_score': val_compat,
        },
    }

    z_test_total_full = zscore(test_total_full, total_mean, total_std)
    z_val_total = zscore(val_total, total_mean, total_std)
    z_test_compat_full = zscore(test_compat_full, compat_mean, compat_std)
    z_val_compat = zscore(val_compat, compat_mean, compat_std)

    for alpha in alphas:
        alpha_key = f'{alpha:g}'
        variants[('total_plus_compat', alpha_key)] = {
            'test_score_full': test_total_full + alpha * test_compat_full,
            'val_score': val_total + alpha * val_compat,
        }
        variants[('zscore_total_plus_compat', alpha_key)] = {
            'test_score_full': z_test_total_full + alpha * z_test_compat_full,
            'val_score': z_val_total + alpha * z_val_compat,
        }

    stats = {
        'val_total_mean': total_mean,
        'val_total_std': total_std,
        'val_compat_mean': compat_mean,
        'val_compat_std': compat_std,
        'val_hsi_mean': float(np.mean(val_details['hsi_score'])),
        'val_hsi_std': float(np.std(val_details['hsi_score'])),
        'val_lidar_mean': float(np.mean(val_details['lidar_score'])),
        'val_lidar_std': float(np.std(val_details['lidar_score'])),
    }
    for payload in variants.values():
        payload.update(stats)
    return variants


def evaluate_run(run_dir: Path, quantiles: list[float], alphas: list[float], device_name: str, detail_root: Path, save_per_sample: bool) -> list[dict]:
    checkpoint_path = run_dir / 'best_model.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config_dict = checkpoint['config']
    valid_fields = {f.name for f in fields(CCSGCLConfig) if f.init}
    kwargs = {k: v for k, v in config_dict.items() if k in valid_fields}
    config = CCSGCLConfig(**kwargs)
    config._resolve_dataset_paths()

    hsi, gt_known, gt_full = load_dataset(config)
    config.num_classes = int(gt_known.max())
    seed_torch(config.seed)
    sampler = rscls(hsi, gt_known, cls=config.num_classes)
    sampler.padding(config.patch_size)
    x_known, y_known = sampler.train_sample(config.num_train)
    x_train_base, y_train_base, x_val, y_val = stratified_known_train_val_split(
        x_known, y_known, val_ratio=config.val_ratio, seed=config.seed
    )
    x_train, _ = make_sample(x_train_base, y_train_base)
    config.hsi_channels, config.lidar_channels = infer_modal_channels(x_train, data_id=config.data)
    val_loader = build_loader(x_val, y_val, config.num_classes, config.data, config.batch_size, False, to_categorical)

    device = torch.device('cuda' if device_name == 'cuda' and torch.cuda.is_available() else 'cpu')
    model = CCSGCLModel(
        num_classes=config.num_classes,
        hsi_channels=config.hsi_channels,
        lidar_channels=config.lidar_channels,
        feat_dim=config.feat_dim,
        eta=config.eta,
        method_variant=getattr(config, 'method_variant', 'cosgl'),
        num_positive_prototypes=getattr(config, 'num_positive_prototypes', 3),
        num_negative_prototypes=getattr(config, 'num_negative_prototypes', 8),
        np_score_weight=getattr(config, 'np_score_weight', 1.0),
        margin_score_weight=getattr(config, 'margin_score_weight', 0.5),
        uf_score_weight=getattr(config, 'uf_score_weight', 1.0),
        compat_mode=config.compat_mode,
        compat_head_type=getattr(config, 'compat_head_type', 'deterministic'),
        hsi_encoder_type=getattr(config, 'hsi_encoder_type', 'cnn'),
        normalize_features=config.normalize_features,
        energy_mode=getattr(config, 'energy_mode', 'full'),
        logvar_min=getattr(config, 'logvar_min', -5.0),
        logvar_max=getattr(config, 'logvar_max', 3.0),
    ).to(device)
    model.load_state_dict(checkpoint['model_state'], strict=False)
    model.eval()

    gt_eval = build_open_set_ground_truth(gt_known=gt_known, gt_full=gt_full, unknown_label=config.unknown_label)
    val_details = collect_val_known_details(model, val_loader, device=device)
    test_details = collect_test_details(model, sampler, config.data, device=device, gt_eval=gt_eval, unknown_label=config.unknown_label)
    variants = build_score_variants(test_details, val_details, alphas=alphas)

    if save_per_sample:
        detail_dir = detail_root / f'data{config.data}_seed{config.seed}'
        detail_dir.mkdir(parents=True, exist_ok=True)

        test_rows = []
        count = int(test_details['gt_label_labeled'].shape[0])
        for idx in range(count):
            test_rows.append(
                {
                    'data': config.data,
                    'dataset': DATASET_LABELS[config.data],
                    'seed': config.seed,
                    'run_dir': str(run_dir),
                    'row': int(test_details['row_labeled'][idx]),
                    'col': int(test_details['col_labeled'][idx]),
                    'gt_label': int(test_details['gt_label_labeled'][idx]),
                    'gt_known_unknown': str(test_details['gt_known_unknown_labeled'][idx]),
                    'pred_known_class': int(test_details['pred_known_class_labeled'][idx]),
                    'total_energy_score': float(test_details['total_energy_score_labeled'][idx]),
                    'compatibility_score': float(test_details['compatibility_score_labeled'][idx]),
                    'hsi_score': float(test_details['hsi_score_labeled'][idx]),
                    'lidar_score': float(test_details['lidar_score_labeled'][idx]),
                }
            )
        write_csv(detail_dir / 'test_scores.csv', test_rows)

        val_rows = []
        val_count = int(val_details['gt_known_label'].shape[0])
        for idx in range(val_count):
            val_rows.append(
                {
                    'data': config.data,
                    'dataset': DATASET_LABELS[config.data],
                    'seed': config.seed,
                    'run_dir': str(run_dir),
                    'val_index': idx,
                    'gt_known_label': int(val_details['gt_known_label'][idx]),
                    'pred_known_class': int(val_details['pred_known_class'][idx]),
                    'total_energy_score': float(val_details['total_energy_score'][idx]),
                    'compatibility_score': float(val_details['compatibility_score'][idx]),
                    'hsi_score': float(val_details['hsi_score'][idx]),
                    'lidar_score': float(val_details['lidar_score'][idx]),
                }
            )
        write_csv(detail_dir / 'val_known_scores.csv', val_rows)

        metadata = {
            'data': config.data,
            'dataset': DATASET_LABELS[config.data],
            'seed': config.seed,
            'run_dir': str(run_dir),
            'checkpoint': str(checkpoint_path),
            'compat_mode': config.compat_mode,
            'eta': config.eta,
            'normalize_features': bool(config.normalize_features),
            'threshold_source': 'known_validation_only',
            'val_known_count': val_count,
            'test_labeled_count': count,
        }
        (detail_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')

    rows: list[dict] = []
    labeled_mask = gt_eval != 0
    num_known = int(np.sum(gt_eval[labeled_mask] != config.unknown_label))
    num_unknown = int(np.sum(gt_eval[labeled_mask] == config.unknown_label))

    pred_known_full = test_details['pred_known_class_full']
    for (score_name, alpha_key), payload in variants.items():
        test_score_full = np.asarray(payload['test_score_full'])
        val_score = np.asarray(payload['val_score'])
        for q in quantiles:
            threshold = float(np.quantile(val_score, q))
            pred_open = pred_known_full.copy()
            pred_open[test_score_full > threshold] = config.unknown_label
            extras = compute_additional_metrics(pred_open, pred_known_full, test_score_full, gt_eval, config.unknown_label)
            rows.append(
                {
                    'data': config.data,
                    'dataset': DATASET_LABELS[config.data],
                    'seed': config.seed,
                    'run_dir': str(run_dir),
                    'checkpoint': str(checkpoint_path),
                    'score_name': score_name,
                    'alpha': alpha_key,
                    'threshold_quantile': float(q),
                    'threshold': threshold,
                    'known_accuracy': float(extras['known_accuracy']),
                    'unknown_accuracy': float(extras['unknown_accuracy']),
                    'macro_f1': float(extras['macro_f1']),
                    'AUROC': float(extras['auroc']),
                    'OSCR': float(extras['oscr']),
                    'known_rejected_as_unknown': float(extras['known_rejected_as_unknown']),
                    'unknown_accepted_as_known': float(extras['unknown_accepted_as_known']),
                    'num_eval_samples': int(np.sum(labeled_mask)),
                    'num_known_eval_samples': num_known,
                    'num_unknown_eval_samples': num_unknown,
                    'num_val_known_samples': int(val_details['gt_known_label'].shape[0]),
                    'val_total_mean': float(payload['val_total_mean']),
                    'val_total_std': float(payload['val_total_std']),
                    'val_compat_mean': float(payload['val_compat_mean']),
                    'val_compat_std': float(payload['val_compat_std']),
                    'val_hsi_mean': float(payload['val_hsi_mean']),
                    'val_hsi_std': float(payload['val_hsi_std']),
                    'val_lidar_mean': float(payload['val_lidar_mean']),
                    'val_lidar_std': float(payload['val_lidar_std']),
                }
            )
    return rows


def metric_mean_std(rows: list[dict], key: str) -> tuple[float, float]:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    return float(np.mean(values)), float(np.std(values))


def aggregate_dataset_means(summary_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[int, str, str, float], list[dict]] = {}
    for row in summary_rows:
        key = (int(row['data']), str(row['score_name']), str(row['alpha']), float(row['threshold_quantile']))
        grouped.setdefault(key, []).append(row)

    aggregated: list[dict] = []
    for (data_id, score_name, alpha_key, q), rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2], item[0][3])):
        out: dict[str, object] = {
            'data': data_id,
            'dataset': DATASET_LABELS[data_id],
            'score_name': score_name,
            'alpha': alpha_key,
            'threshold_quantile': q,
            'num_seeds': len(rows),
        }
        for key in ('threshold', 'known_accuracy', 'unknown_accuracy', 'macro_f1', 'AUROC', 'OSCR', 'known_rejected_as_unknown', 'unknown_accepted_as_known'):
            mean, std = metric_mean_std(rows, key)
            out[f'{key}_mean'] = mean
            out[f'{key}_std'] = std
        aggregated.append(out)

    baseline = {
        (int(row['data']), float(row['threshold_quantile'])): row
        for row in aggregated
        if row['score_name'] == 'total_score' and row['alpha'] == ''
    }
    for row in aggregated:
        base = baseline[(int(row['data']), float(row['threshold_quantile']))]
        for key in ('macro_f1', 'AUROC', 'OSCR'):
            row[f'delta_vs_total_{key}_mean'] = float(row[f'{key}_mean']) - float(base[f'{key}_mean'])
    return aggregated


def aggregate_avg_rows(dataset_mean_rows: list[dict], active_data_ids: list[int]) -> list[dict]:
    grouped: dict[tuple[str, str, float], list[dict]] = {}
    for row in dataset_mean_rows:
        key = (str(row['score_name']), str(row['alpha']), float(row['threshold_quantile']))
        grouped.setdefault(key, []).append(row)

    aggregated: list[dict] = []
    for (score_name, alpha_key, q), rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
        out: dict[str, object] = {
            'score_name': score_name,
            'alpha': alpha_key,
            'threshold_quantile': q,
            'num_datasets': len(rows),
        }
        for metric in ('known_accuracy', 'unknown_accuracy', 'macro_f1', 'AUROC', 'OSCR'):
            values = np.asarray([float(row[f'{metric}_mean']) for row in rows], dtype=np.float64)
            out[f'avg_{metric}'] = float(np.mean(values))

        by_data = {int(row['data']): row for row in rows}
        for data_id in active_data_ids:
            prefix = DATASET_PREFIX[data_id]
            row = by_data[data_id]
            out[f'{prefix}_macro_f1_mean'] = float(row['macro_f1_mean'])
            out[f'{prefix}_AUROC_mean'] = float(row['AUROC_mean'])
            out[f'{prefix}_OSCR_mean'] = float(row['OSCR_mean'])
            out[f'{prefix}_delta_vs_total_macro_f1_mean'] = float(row['delta_vs_total_macro_f1_mean'])
            out[f'{prefix}_delta_vs_total_AUROC_mean'] = float(row['delta_vs_total_AUROC_mean'])
            out[f'{prefix}_delta_vs_total_OSCR_mean'] = float(row['delta_vs_total_OSCR_mean'])
        aggregated.append(out)

    baseline = {
        float(row['threshold_quantile']): row
        for row in aggregated
        if row['score_name'] == 'total_score' and row['alpha'] == ''
    }
    for row in aggregated:
        base = baseline[float(row['threshold_quantile'])]
        for key in ('macro_f1', 'AUROC', 'OSCR'):
            row[f'delta_vs_total_avg_{key}'] = float(row[f'avg_{key}']) - float(base[f'avg_{key}'])
    return aggregated


def best_rows(avg_rows: list[dict], metric_key: str, active_data_ids: list[int]) -> list[dict]:
    selectors = [('avg_over_datasets', None, f'avg_{metric_key}')]
    for data_id in active_data_ids:
        prefix = DATASET_PREFIX[data_id]
        selectors.append((DATASET_LABELS[data_id], prefix, f'{prefix}_{metric_key}_mean'))

    rows: list[dict] = []
    for scope, prefix, primary in selectors:
        def sort_key(row: dict) -> tuple[float, float, float]:
            if prefix is None:
                return (float(row[primary]), float(row['avg_OSCR']), float(row['avg_macro_f1']))
            return (float(row[primary]), float(row[f'{prefix}_OSCR_mean']), float(row[f'{prefix}_macro_f1_mean']))

        best = max(avg_rows, key=sort_key)
        out = dict(best)
        out['selection_scope'] = scope
        out['selection_metric'] = metric_key
        rows.append(out)
    return rows


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    detail_root = output_root / 'per_sample'
    detail_root.mkdir(parents=True, exist_ok=True)
    save_per_sample = str2bool(args.save_per_sample)

    summary_rows: list[dict] = []
    for data_id in args.datasets:
        data_dir = input_root / f'data{data_id}'
        for seed in args.seeds:
            run_dir = latest_run_dir(data_dir, data_id, seed)
            summary_rows.extend(
                evaluate_run(
                    run_dir=run_dir,
                    quantiles=[float(q) for q in args.quantiles],
                    alphas=[float(a) for a in args.alphas],
                    device_name=args.device,
                    detail_root=detail_root,
                    save_per_sample=save_per_sample,
                )
            )

    write_csv(output_root / 'score_calibration_summary.csv', summary_rows)
    active_data_ids = sorted({int(row['data']) for row in summary_rows})
    dataset_mean_rows = aggregate_dataset_means(summary_rows)
    avg_rows = aggregate_avg_rows(dataset_mean_rows, active_data_ids)
    q085_rows = [row for row in avg_rows if abs(float(row['threshold_quantile']) - 0.85) < 1e-9]
    q085_rows = sorted(
        q085_rows,
        key=lambda row: (
            float(row['delta_vs_total_avg_OSCR']),
            float(row['delta_vs_total_avg_AUROC']),
            float(row['delta_vs_total_avg_macro_f1']),
        ),
        reverse=True,
    )

    write_csv(output_root / 'score_calibration_avg.csv', avg_rows)
    write_csv(output_root / 'score_calibration_q085.csv', q085_rows)
    write_csv(output_root / 'score_calibration_best_by_auroc.csv', best_rows(avg_rows, 'AUROC', active_data_ids))
    write_csv(output_root / 'score_calibration_best_by_oscr.csv', best_rows(avg_rows, 'OSCR', active_data_ids))

    metadata = {
        'input_root': str(input_root),
        'output_root': str(output_root),
        'datasets': args.datasets,
        'seeds': args.seeds,
        'quantiles': [float(q) for q in args.quantiles],
        'alphas': [float(a) for a in args.alphas],
        'device': args.device,
        'threshold_source': 'known_validation_only',
        'closed_set_prediction': 'fixed_total_energy_argmin',
        'score_variants': ['total_score', 'compat_score', 'total_plus_compat', 'zscore_total_plus_compat'],
    }
    (output_root / 'score_calibration_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')

    print(f'Saved {output_root / "score_calibration_summary.csv"}')
    print(f'Saved {output_root / "score_calibration_best_by_auroc.csv"}')
    print(f'Saved {output_root / "score_calibration_best_by_oscr.csv"}')
    print(f'Saved {output_root / "score_calibration_q085.csv"}')
    print(f'Saved {output_root / "score_calibration_avg.csv"}')


if __name__ == '__main__':
    main()
