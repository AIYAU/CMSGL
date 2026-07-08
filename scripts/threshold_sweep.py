from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import fields
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cc_sgcl import CCSGCLConfig, CCSGCLModel
from cc_sgcl.data import (
    build_loader,
    build_unlabeled_loader,
    infer_modal_channels,
    load_dataset,
    sample_unlabeled_known_zero_patches,
    stratified_known_train_val_split,
)
from cc_sgcl.inference import (
    SCORE_COMPONENT_KEYS,
    build_open_set_ground_truth,
    collect_known_validation_components,
    collect_known_validation_scores,
    default_score_fusion_weights,
    estimate_adaptive_unlabeled_fusion,
    estimate_score_fusion_stats,
    estimate_threshold,
    run_full_image_inference,
)
from cc_sgcl.metrics import compute_additional_metrics
from cc_sgcl.utils import seed_torch, to_categorical
from third_party.hyliosr_compat import make_sample, rscls


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Threshold-only diagnostic sweep for a trained CC-SGCL checkpoint')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--quantiles', type=float, nargs='+', default=[0.80, 0.85, 0.90, 0.92, 0.95])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    out_dir = checkpoint_path.parent
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
    x_train, y_train = make_sample(x_train_base, y_train_base)
    config.hsi_channels, config.lidar_channels = infer_modal_channels(x_train, data_id=config.data)

    val_loader = build_loader(x_val, y_val, config.num_classes, config.data, config.batch_size, False, to_categorical)

    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
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
        energy_mode=config.energy_mode,
        logvar_min=getattr(config, 'logvar_min', -5.0),
        logvar_max=getattr(config, 'logvar_max', 3.0),
    ).to(device)
    model.load_state_dict(checkpoint['model_state'], strict=False)
    model.eval()
    score_fusion_mode = getattr(config, 'score_fusion_mode', 'model')
    score_fusion_stats = checkpoint.get('score_fusion_stats')
    score_fusion_weights = checkpoint.get('score_fusion_weights')
    score_fusion_diagnostics = checkpoint.get('score_fusion_diagnostics')
    if score_fusion_mode in {"zscore_components", "adaptive_unlabeled_zscore"}:
        missing_components = (
            score_fusion_stats is None
            or any(key not in score_fusion_stats for key in SCORE_COMPONENT_KEYS)
        )
        if missing_components:
            val_components = collect_known_validation_components(model, val_loader, device=device)
            if score_fusion_mode == "adaptive_unlabeled_zscore":
                x_unlabeled = sample_unlabeled_known_zero_patches(
                    sampler,
                    gt_known,
                    num_samples=config.unlabeled_samples,
                    seed=config.seed + 1009,
                )
                unlabeled_components = None
                if x_unlabeled.ndim == 4 and len(x_unlabeled) > 0:
                    unlabeled_loader = build_unlabeled_loader(
                        x_unlabeled,
                        data_id=config.data,
                        batch_size=config.batch_size,
                        shuffle=False,
                    )
                    unlabeled_components = collect_known_validation_components(model, unlabeled_loader, device=device)
                base_weights = {
                    **default_score_fusion_weights(),
                    "energy": getattr(config, "z_energy_weight", 1.0),
                    "negative": getattr(config, "z_neg_weight", 0.5),
                    "margin": getattr(config, "z_margin_weight", 0.0),
                    "uf": getattr(config, "z_uf_weight", 0.0),
                    "nc": getattr(config, "z_nc_weight", 0.0),
                }
                score_fusion_stats, score_fusion_weights, score_fusion_diagnostics = estimate_adaptive_unlabeled_fusion(
                    known_components=val_components,
                    unlabeled_components=unlabeled_components,
                    base_weights=base_weights,
                )
                print("Recomputed adaptive score fusion weights:", score_fusion_weights)
                print("Recomputed adaptive score fusion diagnostics:", score_fusion_diagnostics)
            else:
                score_fusion_stats = estimate_score_fusion_stats(val_components)
                print("Recomputed score fusion stats:", score_fusion_stats)

    val_known_scores = collect_known_validation_scores(
        model,
        val_loader,
        device=device,
        score_fusion_mode=score_fusion_mode,
        score_fusion_stats=score_fusion_stats,
        score_fusion_weights=score_fusion_weights,
    )
    gt_eval = build_open_set_ground_truth(gt_known=gt_known, gt_full=gt_full, unknown_label=config.unknown_label)
    inference = run_full_image_inference(
        model,
        sampler,
        config.data,
        device,
        threshold=1e9,
        unknown_label=config.unknown_label,
        score_fusion_mode=score_fusion_mode,
        score_fusion_stats=score_fusion_stats,
        score_fusion_weights=score_fusion_weights,
    )
    pred_known = inference['pred_known']
    open_score = inference['open_score']

    rows = []
    for q in args.quantiles:
        threshold = estimate_threshold(val_known_scores, quantile=q)
        pred_open = pred_known.copy()
        pred_open[open_score > threshold] = config.unknown_label
        extras = compute_additional_metrics(pred_open, pred_known, open_score, gt_eval, config.unknown_label)
        row = {
            'threshold_quantile': q,
            'threshold': float(threshold),
            'oa': float(extras['oa']),
            'aa': float(extras['aa']),
            'kappa': float(extras['kappa']),
            'known_accuracy': float(extras['known_accuracy']),
            'unknown_accuracy': float(extras['unknown_accuracy']),
            'macro_f1': float(extras['macro_f1']),
            'AUROC': float(extras['auroc']),
            'OSCR': float(extras['oscr']),
            'known_rejected_as_unknown': float(extras['known_rejected_as_unknown']),
            'unknown_accepted_as_known': float(extras['unknown_accepted_as_known']),
        }
        rows.append(row)

    csv_path = out_dir / 'threshold_sweep.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = out_dir / 'threshold_sweep.json'
    json_path.write_text(
        json.dumps(
            {
                'checkpoint': str(checkpoint_path),
                'score_fusion_mode': score_fusion_mode,
                'score_fusion_stats': score_fusion_stats,
                'score_fusion_weights': score_fusion_weights,
                'score_fusion_diagnostics': score_fusion_diagnostics,
                'rows': rows,
            },
            indent=2,
        ),
        encoding='utf-8',
    )

    print('Saved:', csv_path)
    print('Saved:', json_path)
    for row in rows:
        print(row)


if __name__ == '__main__':
    main()
