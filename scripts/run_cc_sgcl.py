from __future__ import annotations

import argparse
import io
import sys
from contextlib import redirect_stdout
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import yaml
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cc_sgcl import CCSGCLConfig, CCSGCLModel, CCSGCLTrainer
from cc_sgcl.data import (
    build_loader,
    build_unlabeled_loader,
    count_eval_labels,
    count_labels_1based,
    count_original_unknown_labels,
    format_count_dict,
    get_known_unknown_class_lists,
    infer_modal_channels,
    load_dataset,
    sample_unlabeled_known_zero_patches,
    stratified_known_train_val_split,
    validate_expected_open_set_protocol,
)
from cc_sgcl.inference import (
    build_open_set_ground_truth,
    collect_known_validation_components,
    collect_known_validation_scores,
    default_score_fusion_weights,
    estimate_adaptive_unlabeled_fusion,
    estimate_threshold,
    estimate_score_fusion_stats,
    run_full_image_inference,
)
from cc_sgcl.metrics import compute_additional_metrics, compute_component_summary, compute_score_summary
from cc_sgcl.utils import save_json, save_metrics_csv, seed_torch, to_categorical
from cc_sgcl.visualization import imgDraw
from third_party.hyliosr_compat import gtcfm, make_sample, rscls


class RunLogger:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = path.open('w', encoding='utf-8')

    def log(self, msg: str = '') -> None:
        print(msg)
        self._fp.write(msg + "\n")
        self._fp.flush()

    def close(self) -> None:
        self._fp.close()


def str2bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {'true', '1', 'yes', 'y'}:
        return True
    if text in {'false', '0', 'no', 'n'}:
        return False
    raise argparse.ArgumentTypeError(f'Invalid boolean value: {value}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run standalone CC-SGCL method')
    parser.add_argument('--config', type=str, default=str(PROJECT_ROOT / 'configs' / 'default.yaml'))
    parser.add_argument('--data_root', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--data', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--eval_only', type=str2bool, default=None)
    parser.add_argument('--numTrain', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--threshold_quantile', type=float, default=None)
    parser.add_argument('--eta', type=float, default=None)
    parser.add_argument('--method_variant', type=str, choices=['cosgl', 'mp', 'np', 'ds', 'uf', 'mp_np', 'mp_ds', 'mp_uf', 'np_ds', 'np_uf', 'all', 'manr'], default=None)
    parser.add_argument('--lambda_cf', type=float, default=None)
    parser.add_argument('--num_positive_prototypes', type=int, default=None)
    parser.add_argument('--num_negative_prototypes', type=int, default=None)
    parser.add_argument('--lambda_np', type=float, default=None)
    parser.add_argument('--np_margin', type=float, default=None)
    parser.add_argument('--np_score_weight', type=float, default=None)
    parser.add_argument('--margin_score_weight', type=float, default=None)
    parser.add_argument('--score_fusion_mode', type=str, choices=['model', 'zscore_components', 'adaptive_unlabeled_zscore'], default=None)
    parser.add_argument('--z_energy_weight', type=float, default=None)
    parser.add_argument('--z_neg_weight', type=float, default=None)
    parser.add_argument('--z_margin_weight', type=float, default=None)
    parser.add_argument('--z_uf_weight', type=float, default=None)
    parser.add_argument('--z_nc_weight', type=float, default=None)
    parser.add_argument('--lambda_ds', type=float, default=None)
    parser.add_argument('--ds_margin', type=float, default=None)
    parser.add_argument('--lambda_uf', type=float, default=None)
    parser.add_argument('--lambda_tail', type=float, default=None)
    parser.add_argument('--lambda_po', type=float, default=None)
    parser.add_argument('--lambda_nc', type=float, default=None)
    parser.add_argument('--nc_temperature', type=float, default=None)
    parser.add_argument('--nc_margin', type=float, default=None)
    parser.add_argument('--nc_simplex_weight', type=float, default=None)
    parser.add_argument('--lambda_aux', type=float, default=None)
    parser.add_argument('--aux_margin', type=float, default=None)
    parser.add_argument('--aux_scale', type=float, default=None)
    parser.add_argument('--lambda_ttr', type=float, default=None)
    parser.add_argument('--ttr_temperature', type=float, default=None)
    parser.add_argument('--ttr_margin', type=float, default=None)
    parser.add_argument('--ttr_compact_weight', type=float, default=None)
    parser.add_argument('--lambda_rp', type=float, default=None)
    parser.add_argument('--rp_temperature', type=float, default=None)
    parser.add_argument('--rp_margin', type=float, default=None)
    parser.add_argument('--rp_compact_weight', type=float, default=None)
    parser.add_argument('--rp_unlabeled_weight', type=float, default=None)
    parser.add_argument('--tail_fraction', type=float, default=None)
    parser.add_argument('--pseudo_outlier_fraction', type=float, default=None)
    parser.add_argument('--pseudo_outlier_margin', type=float, default=None)
    parser.add_argument('--selection_metric', type=str, choices=['known_ce', 'unlabeled_youden'], default=None)
    parser.add_argument('--selection_unlabeled_weight', type=float, default=None)
    parser.add_argument('--uf_score_weight', type=float, default=None)
    parser.add_argument('--unlabeled_samples', type=int, default=None)
    parser.add_argument('--compat_mode', type=str, choices=['bidirectional', 'h2l_only'], default=None)
    parser.add_argument('--compat_head_type', type=str, choices=['deterministic', 'probabilistic'], default=None)
    parser.add_argument('--hsi_encoder_type', type=str, choices=['cnn', 'spectral_rwkv', 'spectral_vrwkv', 'spectral_vrwkv_official', 'spectral_vrwkv6', 'spectral_vrwkv6_cuda'], default=None)
    parser.add_argument('--logvar_min', type=float, default=None)
    parser.add_argument('--logvar_max', type=float, default=None)
    parser.add_argument('--energy_mode', type=str, choices=['full', 'no_compat', 'hsi_only', 'lidar_only'], default=None)
    parser.add_argument('--normalize_features', type=str2bool, default=None)
    parser.add_argument('--warmup_epochs', type=int, default=None)
    parser.add_argument('--save_model', type=str2bool, default=None)
    parser.add_argument('--save_maps', type=str2bool, default=None)
    parser.add_argument('--run_final_eval', type=str2bool, default=None)
    parser.add_argument('--require_unknown_eval', type=str2bool, default=None)
    parser.add_argument('--enforce_dataset_protocol', type=str2bool, default=None)
    parser.add_argument('--device', type=str, default=None)
    return parser.parse_args()


def load_yaml(path: Path) -> Dict:
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    default_path = PROJECT_ROOT / 'configs' / 'default.yaml'
    if path.resolve() == default_path.resolve():
        return config
    defaults = yaml.safe_load(default_path.read_text(encoding='utf-8'))
    defaults.update(config)
    return defaults


def apply_overrides(cfg: Dict, args: argparse.Namespace) -> Dict:
    mapping = {
        'data_root': args.data_root,
        'output_root': args.output_dir,
        'data': args.data,
        'seed': args.seed,
        'checkpoint': args.checkpoint or None,
        'eval_only': args.eval_only,
        'num_train': args.numTrain,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'threshold_quantile': args.threshold_quantile,
        'eta': args.eta,
        'method_variant': args.method_variant,
        'lambda_cf': args.lambda_cf,
        'num_positive_prototypes': args.num_positive_prototypes,
        'num_negative_prototypes': args.num_negative_prototypes,
        'lambda_np': args.lambda_np,
        'np_margin': args.np_margin,
        'np_score_weight': args.np_score_weight,
        'margin_score_weight': args.margin_score_weight,
        'score_fusion_mode': args.score_fusion_mode,
        'z_energy_weight': args.z_energy_weight,
        'z_neg_weight': args.z_neg_weight,
        'z_margin_weight': args.z_margin_weight,
        'z_uf_weight': args.z_uf_weight,
        'z_nc_weight': args.z_nc_weight,
        'lambda_ds': args.lambda_ds,
        'ds_margin': args.ds_margin,
        'lambda_uf': args.lambda_uf,
        'lambda_tail': args.lambda_tail,
        'lambda_po': args.lambda_po,
        'lambda_nc': args.lambda_nc,
        'nc_temperature': args.nc_temperature,
        'nc_margin': args.nc_margin,
        'nc_simplex_weight': args.nc_simplex_weight,
        'lambda_aux': args.lambda_aux,
        'aux_margin': args.aux_margin,
        'aux_scale': args.aux_scale,
        'lambda_ttr': args.lambda_ttr,
        'ttr_temperature': args.ttr_temperature,
        'ttr_margin': args.ttr_margin,
        'ttr_compact_weight': args.ttr_compact_weight,
        'lambda_rp': args.lambda_rp,
        'rp_temperature': args.rp_temperature,
        'rp_margin': args.rp_margin,
        'rp_compact_weight': args.rp_compact_weight,
        'rp_unlabeled_weight': args.rp_unlabeled_weight,
        'tail_fraction': args.tail_fraction,
        'pseudo_outlier_fraction': args.pseudo_outlier_fraction,
        'pseudo_outlier_margin': args.pseudo_outlier_margin,
        'selection_metric': args.selection_metric,
        'selection_unlabeled_weight': args.selection_unlabeled_weight,
        'uf_score_weight': args.uf_score_weight,
        'unlabeled_samples': args.unlabeled_samples,
        'compat_mode': args.compat_mode,
        'compat_head_type': args.compat_head_type,
        'hsi_encoder_type': args.hsi_encoder_type,
        'logvar_min': args.logvar_min,
        'logvar_max': args.logvar_max,
        'energy_mode': args.energy_mode,
        'normalize_features': args.normalize_features,
        'warmup_epochs': args.warmup_epochs,
        'save_model': args.save_model,
        'save_maps': args.save_maps,
        'run_final_eval': args.run_final_eval,
        'require_unknown_eval': args.require_unknown_eval,
        'enforce_dataset_protocol': args.enforce_dataset_protocol,
        'device': args.device,
    }
    for key, value in mapping.items():
        if value not in (None, ''):
            cfg[key] = value
    return cfg


def make_run_name(data_id: int, seed: int) -> str:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f'data{data_id}_seed{seed}_{timestamp}'


def config_to_dict(config: CCSGCLConfig) -> Dict:
    payload = {}
    for key, value in config.__dict__.items():
        if isinstance(value, Path):
            payload[key] = str(value)
        else:
            payload[key] = value
    payload['output_root_dir'] = str(config.output_root_dir)
    payload['run_dir'] = str(config.run_dir)
    return payload


def summarize_test_scores(open_score: np.ndarray, gt_eval: np.ndarray, unknown_label: int) -> Dict[str, Dict[str, float]]:
    labeled_mask = gt_eval != 0
    known_mask = np.logical_and(labeled_mask, gt_eval != unknown_label)
    unknown_mask = gt_eval == unknown_label
    return {
        'test_known_open_score': compute_score_summary(open_score[known_mask]),
        'test_unknown_open_score': compute_score_summary(open_score[unknown_mask]),
        'known_mask': known_mask,
        'unknown_mask': unknown_mask,
    }


def maybe_override_config_from_checkpoint(config: CCSGCLConfig, checkpoint_path: Path) -> Dict:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    snapshot = checkpoint.get('config', {})
    valid_fields = {f.name for f in fields(CCSGCLConfig) if f.init}
    for key, value in snapshot.items():
        if key in valid_fields and key not in {'project_root', 'output_root', 'eval_only', 'checkpoint', 'save_model', 'save_maps', 'device', 'data_root'}:
            setattr(config, key, value)
    config._resolve_dataset_paths()
    return checkpoint


def flatten_component_means(component_summary: Dict[str, Dict[str, float]], prefix: str) -> Dict[str, float]:
    flat = {}
    for key, stats in component_summary.items():
        flat[f'{prefix}_{key}_mean'] = stats['mean']
        flat[f'{prefix}_{key}_std'] = stats['std']
    return flat


def main() -> None:
    args = parse_args()
    cfg = load_yaml(Path(args.config))
    cfg = apply_overrides(cfg, args)

    data_root = cfg.get('data_root') or (PROJECT_ROOT / 'data')
    config = CCSGCLConfig(
        project_root=PROJECT_ROOT,
        data_root=Path(data_root),
        data=cfg['data'],
        num_train=cfg['num_train'],
        batch_size=cfg['batch_size'],
        epochs=cfg['epochs'],
        lr=cfg['lr'],
        weight_decay=cfg['weight_decay'],
        feat_dim=cfg['feat_dim'],
        eta=cfg['eta'],
        method_variant=cfg.get('method_variant', 'cosgl'),
        compat_mode=cfg.get('compat_mode', 'h2l_only'),
        compat_head_type=cfg.get('compat_head_type', 'deterministic'),
        hsi_encoder_type=cfg.get('hsi_encoder_type', 'cnn'),
        logvar_min=cfg.get('logvar_min', -5.0),
        logvar_max=cfg.get('logvar_max', 3.0),
        energy_mode=cfg.get('energy_mode', 'full'),
        margin=cfg['margin'],
        unknown_margin=cfg['unknown_margin'],
        lambda_margin=cfg['lambda_margin'],
        lambda_cf=cfg['lambda_cf'],
        num_positive_prototypes=cfg.get('num_positive_prototypes', 3),
        num_negative_prototypes=cfg.get('num_negative_prototypes', 8),
        lambda_np=cfg.get('lambda_np', 0.1),
        np_margin=cfg.get('np_margin', 0.5),
        np_score_weight=cfg.get('np_score_weight', 1.0),
        margin_score_weight=cfg.get('margin_score_weight', 0.5),
        score_fusion_mode=cfg.get('score_fusion_mode', 'model'),
        z_energy_weight=cfg.get('z_energy_weight', 1.0),
        z_neg_weight=cfg.get('z_neg_weight', 0.5),
        z_margin_weight=cfg.get('z_margin_weight', 0.5),
        z_uf_weight=cfg.get('z_uf_weight', 0.5),
        z_nc_weight=cfg.get('z_nc_weight', 0.0),
        lambda_ds=cfg.get('lambda_ds', 0.05),
        ds_margin=cfg.get('ds_margin', 1.0),
        lambda_uf=cfg.get('lambda_uf', 0.1),
        lambda_tail=cfg.get('lambda_tail', 0.0),
        lambda_po=cfg.get('lambda_po', 0.0),
        lambda_nc=cfg.get('lambda_nc', 0.0),
        nc_temperature=cfg.get('nc_temperature', 0.1),
        nc_margin=cfg.get('nc_margin', 0.2),
        nc_simplex_weight=cfg.get('nc_simplex_weight', 0.1),
        lambda_aux=cfg.get('lambda_aux', 0.0),
        aux_margin=cfg.get('aux_margin', 0.2),
        aux_scale=cfg.get('aux_scale', 16.0),
        lambda_ttr=cfg.get('lambda_ttr', 0.0),
        ttr_temperature=cfg.get('ttr_temperature', 0.2),
        ttr_margin=cfg.get('ttr_margin', 0.2),
        ttr_compact_weight=cfg.get('ttr_compact_weight', 0.1),
        lambda_rp=cfg.get('lambda_rp', 0.0),
        rp_temperature=cfg.get('rp_temperature', 0.15),
        rp_margin=cfg.get('rp_margin', 0.15),
        rp_compact_weight=cfg.get('rp_compact_weight', 0.1),
        rp_unlabeled_weight=cfg.get('rp_unlabeled_weight', 0.5),
        tail_fraction=cfg.get('tail_fraction', 0.30),
        pseudo_outlier_fraction=cfg.get('pseudo_outlier_fraction', 0.30),
        pseudo_outlier_margin=cfg.get('pseudo_outlier_margin', 0.5),
        selection_metric=cfg.get('selection_metric', 'known_ce'),
        selection_unlabeled_weight=cfg.get('selection_unlabeled_weight', 0.2),
        uf_score_weight=cfg.get('uf_score_weight', 1.0),
        unlabeled_samples=cfg.get('unlabeled_samples', 512),
        patch_size=cfg['patch_size'],
        val_ratio=cfg['val_ratio'],
        threshold_quantile=cfg['threshold_quantile'],
        warmup_epochs=cfg.get('warmup_epochs', 0),
        normalize_features=cfg.get('normalize_features', True),
        device=cfg['device'],
        output_root=cfg['output_root'],
        seed=cfg['seed'],
        save_model=cfg['save_model'],
        save_maps=cfg['save_maps'],
        run_final_eval=cfg.get('run_final_eval', True),
        eval_only=cfg['eval_only'],
        require_unknown_eval=cfg.get('require_unknown_eval', True),
        enforce_dataset_protocol=cfg.get('enforce_dataset_protocol', True),
        checkpoint=cfg.get('checkpoint'),
    )

    checkpoint = None
    if config.eval_only:
        if not config.checkpoint:
            raise ValueError('--eval_only requires --checkpoint')
        checkpoint = maybe_override_config_from_checkpoint(config, Path(config.checkpoint))

    config.run_name = make_run_name(config.data, config.seed)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(config.run_dir / 'train_log.txt')

    try:
        logger.log('=' * 80)
        logger.log('CC-SGCL Experiment')
        logger.log('=' * 80)
        logger.log(f'run_dir: {config.run_dir}')
        logger.log(f'dataset_id: {config.data}')
        logger.log(f'random_seed: {config.seed}')
        logger.log(f'eval_only: {config.eval_only}')
        logger.log(f'warmup_epochs: {config.warmup_epochs}')
        logger.log(f'method_variant: {config.method_variant}')
        logger.log(f'normalize_features: {config.normalize_features}')
        logger.log(f'compat_mode: {config.compat_mode}')
        logger.log(f'compat_head_type: {config.compat_head_type}')
        logger.log(f'hsi_encoder_type: {config.hsi_encoder_type}')
        if config.compat_head_type == 'probabilistic':
            logger.log(f'compat_logvar_clamp: [{config.logvar_min}, {config.logvar_max}]')
        logger.log(f'energy_mode: {config.energy_mode}')
        logger.log(f'score_fusion_mode: {config.score_fusion_mode}')
        if config.checkpoint:
            logger.log(f'checkpoint: {config.checkpoint}')

        hsi, gt_known, gt_full = load_dataset(config)
        checkpoint_num_classes = None
        if checkpoint is not None:
            checkpoint_num_classes = checkpoint.get('config', {}).get('num_classes')
        config.num_classes = int(checkpoint_num_classes or gt_known.max())
        known_classes, unknown_classes = get_known_unknown_class_lists(gt_known, gt_full)
        if config.require_unknown_eval and not unknown_classes:
            raise ValueError(
                "Open-set evaluation requires at least one unknown class in FullGT. "
                f"Dataset {config.dataset_name} under {config.data_root} has unknown_class_list=[]; "
                "check data_root or set require_unknown_eval=false only for closed-set debugging."
            )
        if config.enforce_dataset_protocol:
            validate_expected_open_set_protocol(config.data, known_classes, unknown_classes)

        seed_torch(config.seed)

        sampler = rscls(hsi, gt_known, cls=config.num_classes)
        sampler.padding(config.patch_size)
        x_known, y_known = sampler.train_sample(config.num_train)
        x_train_base, y_train_base, x_val, y_val = stratified_known_train_val_split(
            x_known, y_known, val_ratio=config.val_ratio, seed=config.seed
        )
        x_train, y_train = make_sample(x_train_base, y_train_base)

        if len(x_val) == 0:
            raise ValueError('Known validation split is empty; threshold must come from known validation only.')

        config.hsi_channels, config.lidar_channels = infer_modal_channels(x_train, data_id=config.data)
        gt_eval = build_open_set_ground_truth(gt_known=gt_known, gt_full=gt_full, unknown_label=config.unknown_label)

        train_loader = build_loader(x_train, y_train, config.num_classes, config.data, config.batch_size, True, to_categorical)
        train_eval_loader = build_loader(x_train_base, y_train_base, config.num_classes, config.data, config.batch_size, False, to_categorical)
        val_loader = build_loader(x_val, y_val, config.num_classes, config.data, config.batch_size, False, to_categorical)

        class_split_info = {
            'dataset_id': config.data,
            'seed': config.seed,
            'known_class_list': known_classes,
            'unknown_class_list': unknown_classes,
            'train_samples_per_class': count_labels_1based(y_train_base),
            'val_samples_per_class': count_labels_1based(y_val),
            'test_samples_per_class': count_eval_labels(gt_eval),
            'test_original_unknown_samples_per_class': count_original_unknown_labels(gt_known, gt_full),
            'enforce_dataset_protocol': config.enforce_dataset_protocol,
            'train_augmented_total': int(len(y_train)),
            'train_base_total': int(len(y_train_base)),
            'val_base_total': int(len(y_val)),
            'test_labeled_total': int(np.sum(gt_eval != 0)),
        }

        logger.log(f'known class list: {known_classes}')
        logger.log(f'unknown class list: {unknown_classes}')
        logger.log(format_count_dict('train samples per class', class_split_info['train_samples_per_class']))
        logger.log(format_count_dict('val samples per class', class_split_info['val_samples_per_class']))
        logger.log(format_count_dict('test samples per class', class_split_info['test_samples_per_class']))
        logger.log('threshold source: known validation samples only')
        logger.log('counterfactual source: training-batch cross-class HSI-LiDAR pairings only')
        if config.method_variant in {'uf', 'mp_uf', 'np_uf', 'all', 'manr'}:
            logger.log(f'unlabeled filter source: KnownGT==0 pixels only, samples={config.unlabeled_samples}')
        if config.lambda_tail > 0 or config.lambda_po > 0:
            logger.log(
                'tail/pseudo-outlier training: '
                f'lambda_tail={config.lambda_tail}, lambda_po={config.lambda_po}, '
                f'tail_fraction={config.tail_fraction}, '
                f'pseudo_outlier_fraction={config.pseudo_outlier_fraction}, '
                f'pseudo_outlier_margin={config.pseudo_outlier_margin}'
            )
        if config.lambda_nc > 0:
            logger.log(
                'neural-collapse alignment training: '
                f'lambda_nc={config.lambda_nc}, temperature={config.nc_temperature}, '
                f'margin={config.nc_margin}, simplex_weight={config.nc_simplex_weight}'
            )
        if config.lambda_aux > 0:
            logger.log(
                'auxiliary arc-margin training: '
                f'lambda_aux={config.lambda_aux}, margin={config.aux_margin}, scale={config.aux_scale}'
            )
        if config.lambda_ttr > 0:
            logger.log(
                'TTR retrieval alignment training: '
                f'lambda_ttr={config.lambda_ttr}, temperature={config.ttr_temperature}, '
                f'margin={config.ttr_margin}, compact_weight={config.ttr_compact_weight}'
            )
        if config.lambda_rp > 0:
            logger.log(
                'retrieval-prototype open training: '
                f'lambda_rp={config.lambda_rp}, temperature={config.rp_temperature}, '
                f'margin={config.rp_margin}, compact_weight={config.rp_compact_weight}, '
                f'unlabeled_weight={config.rp_unlabeled_weight}'
            )
        logger.log(
            'model selection: '
            f'selection_metric={config.selection_metric}, '
            f'selection_unlabeled_weight={config.selection_unlabeled_weight}'
        )

        model = CCSGCLModel(
            num_classes=config.num_classes,
            hsi_channels=config.hsi_channels,
            lidar_channels=config.lidar_channels,
            feat_dim=config.feat_dim,
            eta=config.eta,
            method_variant=config.method_variant,
            num_positive_prototypes=config.num_positive_prototypes,
            num_negative_prototypes=config.num_negative_prototypes,
            np_score_weight=config.np_score_weight,
            margin_score_weight=config.margin_score_weight,
            uf_score_weight=config.uf_score_weight,
            compat_mode=config.compat_mode,
            compat_head_type=config.compat_head_type,
            hsi_encoder_type=config.hsi_encoder_type,
            normalize_features=config.normalize_features,
            energy_mode=config.energy_mode,
            logvar_min=config.logvar_min,
            logvar_max=config.logvar_max,
        ).to(torch.device(config.device if torch.cuda.is_available() else 'cpu'))
        device = next(model.parameters()).device

        history = None
        best_state_dict = None
        best_epoch = None
        best_val_accuracy = None
        best_val_loss = None
        score_fusion_stats = None
        score_fusion_diagnostics = None
        score_fusion_weights = {
            **default_score_fusion_weights(),
            'energy': config.z_energy_weight,
            'negative': config.z_neg_weight,
            'margin': config.z_margin_weight,
            'uf': config.z_uf_weight,
            'nc': config.z_nc_weight,
        }

        if config.eval_only:
            logger.log('[CC-SGCL] eval_only=True, skipping training.')
            model.load_state_dict(checkpoint['model_state'], strict=False)
            best_state_dict = checkpoint['model_state']
            best_epoch = checkpoint.get('best_epoch')
            best_val_accuracy = checkpoint.get('best_val_accuracy')
            best_val_loss = checkpoint.get('best_val_loss')
            score_fusion_stats = checkpoint.get('score_fusion_stats')
            score_fusion_diagnostics = checkpoint.get('score_fusion_diagnostics')
            score_fusion_weights = checkpoint.get('score_fusion_weights', score_fusion_weights)
            threshold = float(checkpoint['threshold'])
        else:
            trainer = CCSGCLTrainer(config=config, model=model, device=device, logger=logger.log)
            unlabeled_loader = None
            needs_unlabeled_loader = (
                config.method_variant in {'uf', 'mp_uf', 'np_uf', 'all', 'manr'}
                or config.lambda_po > 0
                or (config.lambda_rp > 0 and config.rp_unlabeled_weight > 0)
                or config.selection_metric == 'unlabeled_youden'
            )
            if needs_unlabeled_loader:
                x_unlabeled = sample_unlabeled_known_zero_patches(
                    sampler=sampler,
                    gt_known=gt_known,
                    num_samples=config.unlabeled_samples,
                    seed=config.seed + 1009,
                )
                if x_unlabeled.ndim == 4 and len(x_unlabeled) > 0:
                    unlabeled_loader = build_unlabeled_loader(
                        x_unlabeled,
                        data_id=config.data,
                        batch_size=config.batch_size,
                        shuffle=True,
                    )
                    logger.log(f'unlabeled filter samples: {len(x_unlabeled)}')
                else:
                    logger.log('unlabeled filter samples: 0 (UF loss skipped)')
            history, best_state_dict = trainer.train(
                train_loader,
                val_loader=val_loader,
                unlabeled_loader=unlabeled_loader,
            )
            best_epoch = history.best_epoch
            best_val_accuracy = history.best_val_accuracy
            best_val_loss = history.best_val_loss
            model.load_state_dict(best_state_dict)

            if config.score_fusion_mode in {'zscore_components', 'adaptive_unlabeled_zscore'}:
                val_components = collect_known_validation_components(model, val_loader, device=device)
                if config.score_fusion_mode == 'adaptive_unlabeled_zscore':
                    unlabeled_components = None
                    if unlabeled_loader is not None:
                        unlabeled_components = collect_known_validation_components(model, unlabeled_loader, device=device)
                    score_fusion_stats, score_fusion_weights, score_fusion_diagnostics = estimate_adaptive_unlabeled_fusion(
                        known_components=val_components,
                        unlabeled_components=unlabeled_components,
                        base_weights=score_fusion_weights,
                    )
                    logger.log(f'adaptive score fusion diagnostics: {score_fusion_diagnostics}')
                else:
                    score_fusion_stats = estimate_score_fusion_stats(val_components)
                logger.log(f'score fusion stats: {score_fusion_stats}')
                logger.log(f'score fusion weights: {score_fusion_weights}')
            val_open_scores = collect_known_validation_scores(
                model,
                val_loader,
                device=device,
                score_fusion_mode=config.score_fusion_mode,
                score_fusion_stats=score_fusion_stats,
                score_fusion_weights=score_fusion_weights,
            )
            threshold = estimate_threshold(val_open_scores, quantile=config.threshold_quantile)

            if config.save_model:
                checkpoint_payload = {
                    'model_state': best_state_dict,
                    'threshold': threshold,
                    'best_epoch': best_epoch,
                    'best_val_accuracy': best_val_accuracy,
                    'best_val_loss': best_val_loss,
                    'score_fusion_stats': score_fusion_stats,
                    'score_fusion_weights': score_fusion_weights,
                    'score_fusion_diagnostics': score_fusion_diagnostics,
                    'config': config_to_dict(config),
                }
                torch.save(checkpoint_payload, config.run_dir / 'best_model.pth')
                logger.log(f"saved checkpoint: {config.run_dir / 'best_model.pth'}")

            if not config.run_final_eval:
                save_json(config.run_dir / 'config.json', config_to_dict(config))
                save_json(config.run_dir / 'class_split_info.json', class_split_info)
                logger.log('[CC-SGCL] run_final_eval=False, skipping full-image inference and final metrics.')
                return

        train_known_scores = collect_known_validation_scores(
            model,
            train_eval_loader,
            device=device,
            score_fusion_mode=config.score_fusion_mode,
            score_fusion_stats=score_fusion_stats,
            score_fusion_weights=score_fusion_weights,
        )
        val_known_scores = collect_known_validation_scores(
            model,
            val_loader,
            device=device,
            score_fusion_mode=config.score_fusion_mode,
            score_fusion_stats=score_fusion_stats,
            score_fusion_weights=score_fusion_weights,
        )

        inference = run_full_image_inference(
            model,
            sampler,
            config.data,
            device,
            threshold,
            config.unknown_label,
            score_fusion_mode=config.score_fusion_mode,
            score_fusion_stats=score_fusion_stats,
            score_fusion_weights=score_fusion_weights,
        )
        pred_known = inference['pred_known']
        pred_open = inference['pred_open']
        open_score = inference['open_score']

        with torch.no_grad():
            row, col = sampler.gt.shape
            e_h_rows, e_l_rows, e_compat_rows, e_total_rows = [], [], [], []
            neg_rows, gap_rows, uf_rows = [], [], []
            logvar_mean_rows, logvar_min_rows, logvar_max_rows = [], [], []
            for r in range(row):
                row_samples = sampler.all_sample_row(r)
                from cc_sgcl.inference import split_hsi_lidar_patches, to_nchw
                row_e_h, row_e_l, row_e_compat, row_e_total = [], [], [], []
                row_neg, row_gap, row_uf = [], [], []
                row_logvar_mean, row_logvar_min, row_logvar_max = [], [], []
                for start in range(0, len(row_samples), 256):
                    chunk = row_samples[start : start + 256]
                    x_h_np, x_l_np = split_hsi_lidar_patches(chunk, data_id=config.data)
                    x_h = torch.from_numpy(to_nchw(x_h_np)).to(device)
                    x_l = torch.from_numpy(to_nchw(x_l_np)).to(device)
                    outputs = model(x_h, x_l)
                    pred_idx = outputs['energies'].argmin(dim=1)
                    row_e_h.append(outputs['E_h'].gather(1, pred_idx.view(-1, 1)).cpu().numpy().reshape(-1))
                    row_e_l.append(outputs['E_l'].gather(1, pred_idx.view(-1, 1)).cpu().numpy().reshape(-1))
                    row_e_compat.append(outputs['E_compat'].gather(1, pred_idx.view(-1, 1)).cpu().numpy().reshape(-1))
                    row_e_total.append(outputs['E_total'].gather(1, pred_idx.view(-1, 1)).cpu().numpy().reshape(-1))
                    row_neg.append(outputs['negative_relation_score'].cpu().numpy().reshape(-1))
                    row_gap.append(outputs['energy_gap'].cpu().numpy().reshape(-1))
                    row_uf.append(outputs['uf_unknown_score'].cpu().numpy().reshape(-1))
                    if 'compat_logvar_mean' in outputs:
                        row_logvar_mean.append(outputs['compat_logvar_mean'].gather(1, pred_idx.view(-1, 1)).cpu().numpy().reshape(-1))
                        row_logvar_min.append(outputs['compat_logvar_min'].gather(1, pred_idx.view(-1, 1)).cpu().numpy().reshape(-1))
                        row_logvar_max.append(outputs['compat_logvar_max'].gather(1, pred_idx.view(-1, 1)).cpu().numpy().reshape(-1))
                e_h_rows.append(np.concatenate(row_e_h, axis=0))
                e_l_rows.append(np.concatenate(row_e_l, axis=0))
                e_compat_rows.append(np.concatenate(row_e_compat, axis=0))
                e_total_rows.append(np.concatenate(row_e_total, axis=0))
                neg_rows.append(np.concatenate(row_neg, axis=0))
                gap_rows.append(np.concatenate(row_gap, axis=0))
                uf_rows.append(np.concatenate(row_uf, axis=0))
                if row_logvar_mean:
                    logvar_mean_rows.append(np.concatenate(row_logvar_mean, axis=0))
                    logvar_min_rows.append(np.concatenate(row_logvar_min, axis=0))
                    logvar_max_rows.append(np.concatenate(row_logvar_max, axis=0))
        components = {
            'E_h': np.stack(e_h_rows, axis=0),
            'E_l': np.stack(e_l_rows, axis=0),
            'E_compat': np.stack(e_compat_rows, axis=0),
            'E_total': np.stack(e_total_rows, axis=0),
            'negative_relation_score': np.stack(neg_rows, axis=0),
            'energy_gap': np.stack(gap_rows, axis=0),
            'uf_unknown_score': np.stack(uf_rows, axis=0),
        }
        if logvar_mean_rows:
            components['compat_logvar_mean'] = np.stack(logvar_mean_rows, axis=0)
            components['compat_logvar_min'] = np.stack(logvar_min_rows, axis=0)
            components['compat_logvar_max'] = np.stack(logvar_max_rows, axis=0)

        gtcfm_buffer = io.StringIO()
        with redirect_stdout(gtcfm_buffer):
            cfm, oa, aa, kappa, osr = gtcfm(pred_open, gt_eval, config.unknown_label)
        for line in gtcfm_buffer.getvalue().strip().splitlines():
            if line.strip():
                logger.log(line)

        extras = compute_additional_metrics(pred_open, pred_known, open_score, gt_eval, config.unknown_label)
        score_masks = summarize_test_scores(open_score, gt_eval, config.unknown_label)
        train_score_stats = compute_score_summary(train_known_scores)
        val_score_stats = compute_score_summary(val_known_scores)
        known_component_stats = compute_component_summary(components, score_masks['known_mask'])
        unknown_component_stats = compute_component_summary(components, score_masks['unknown_mask'])

        logger.log(f'threshold quantile: {config.threshold_quantile}')
        logger.log(f'threshold value: {threshold:.6f}')
        logger.log(f'train known open_score stats: {train_score_stats}')
        logger.log(f'val known open_score stats: {val_score_stats}')
        logger.log(f"test known open_score stats: {score_masks['test_known_open_score']}")
        logger.log(f"test unknown open_score stats: {score_masks['test_unknown_open_score']}")
        logger.log(f'known energy components: {known_component_stats}')
        logger.log(f'unknown energy components: {unknown_component_stats}')
        logger.log(f'best epoch: {best_epoch}')
        logger.log(f'best val accuracy: {best_val_accuracy}')
        logger.log(f'best val loss: {best_val_loss}')
        logger.log(f"Known rejected as unknown: {extras['known_rejected_as_unknown']}")
        logger.log(f"Unknown accepted as known: {extras['unknown_accepted_as_known']}")

        if config.save_maps:
            np.save(config.run_dir / "pred_known.npy", pred_known.astype("int64"))
            np.save(config.run_dir / "open_score.npy", open_score.astype("float32"))
            np.save(config.run_dir / "pred_open.npy", pred_open.astype("int64"))
            imgDraw(pred_known, f'{config.key}_closed', path=str(config.run_dir), show=False)
            pred_draw = pred_open.copy()
            pred_draw[pred_draw == config.unknown_label] = 0
            imgDraw(pred_draw, f'{config.key}_open', path=str(config.run_dir), show=False)
            logger.log('saved classification maps')

        threshold_info = {
            'source': 'known_validation_only',
            'threshold_quantile': config.threshold_quantile,
            'threshold_value': float(threshold),
            'num_known_val_samples': int(len(val_known_scores)),
            'train_known_open_score_stats': train_score_stats,
            'val_known_open_score_stats': val_score_stats,
            'known_component_stats': known_component_stats,
            'unknown_component_stats': unknown_component_stats,
            'score_fusion_mode': config.score_fusion_mode,
            'score_fusion_stats': score_fusion_stats,
            'score_fusion_weights': score_fusion_weights,
            'score_fusion_diagnostics': score_fusion_diagnostics,
        }

        metrics = {
            'dataset_id': config.data,
            'seed': config.seed,
            'oa': float(oa),
            'aa': float(aa),
            'kappa': float(kappa),
            'osr': float(osr),
            'threshold': float(threshold),
            **extras,
            **flatten_component_means(known_component_stats, 'known'),
            **flatten_component_means(unknown_component_stats, 'unknown'),
            'test_known_open_score_mean': score_masks['test_known_open_score']['mean'],
            'test_unknown_open_score_mean': score_masks['test_unknown_open_score']['mean'],
        }
        metrics_json = {
            'metrics': metrics,
            'score_stats': {
                'train_known_open_score': train_score_stats,
                'val_known_open_score': val_score_stats,
                'test_known_open_score': score_masks['test_known_open_score'],
                'test_unknown_open_score': score_masks['test_unknown_open_score'],
            },
            'energy_components': {
                'known': known_component_stats,
                'unknown': unknown_component_stats,
            },
            'best_epoch': best_epoch,
            'best_val_accuracy': best_val_accuracy,
            'best_val_loss': best_val_loss,
            'confusion_matrix': cfm.tolist(),
            'warmup_epochs': config.warmup_epochs,
            'method_variant': config.method_variant,
            'normalize_features': config.normalize_features,
            'compat_mode': config.compat_mode,
            'compat_head_type': config.compat_head_type,
            'hsi_encoder_type': config.hsi_encoder_type,
            'logvar_min': config.logvar_min,
            'logvar_max': config.logvar_max,
            'energy_mode': config.energy_mode,
            'lambda_nc': config.lambda_nc,
            'nc_temperature': config.nc_temperature,
            'nc_margin': config.nc_margin,
            'nc_simplex_weight': config.nc_simplex_weight,
            'lambda_aux': config.lambda_aux,
            'aux_margin': config.aux_margin,
            'aux_scale': config.aux_scale,
            'lambda_ttr': config.lambda_ttr,
            'ttr_temperature': config.ttr_temperature,
            'ttr_margin': config.ttr_margin,
            'ttr_compact_weight': config.ttr_compact_weight,
            'lambda_rp': config.lambda_rp,
            'rp_temperature': config.rp_temperature,
            'rp_margin': config.rp_margin,
            'rp_compact_weight': config.rp_compact_weight,
            'rp_unlabeled_weight': config.rp_unlabeled_weight,
            'score_fusion_mode': config.score_fusion_mode,
            'score_fusion_stats': score_fusion_stats,
            'score_fusion_weights': score_fusion_weights,
            'score_fusion_diagnostics': score_fusion_diagnostics,
        }

        save_json(config.run_dir / 'config.json', config_to_dict(config))
        save_json(config.run_dir / 'threshold_info.json', threshold_info)
        save_json(config.run_dir / 'class_split_info.json', class_split_info)
        save_json(config.run_dir / 'metrics.json', metrics_json)
        save_metrics_csv(config.run_dir / 'metrics.csv', metrics)

        logger.log('=' * 80)
        logger.log('Dataset: ' + config.dataset_name)
        logger.log('Seed: ' + str(config.seed))
        logger.log('Known OA: ' + str(extras['known_accuracy']))
        logger.log('Unknown Acc: ' + str(extras['unknown_accuracy']))
        logger.log('Macro-F1: ' + str(extras['macro_f1']))
        logger.log('AUROC: ' + str(extras['auroc']))
        logger.log('OSCR: ' + str(extras['oscr']))
        logger.log('Threshold: ' + str(threshold))
        logger.log('Known rejected as unknown: ' + str(extras['known_rejected_as_unknown']))
        logger.log('Unknown accepted as known: ' + str(extras['unknown_accepted_as_known']))
        logger.log('=' * 80)
        logger.log('[CC-SGCL] Final Metrics')
        logger.log('=' * 80)
        for key, value in metrics.items():
            logger.log(f'{key}: {value}')

    finally:
        logger.close()


if __name__ == '__main__':
    main()
