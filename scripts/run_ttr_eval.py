from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
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
    extract_score_components,
    fuse_score_components,
    split_hsi_lidar_patches,
    to_nchw,
)
from cc_sgcl.metrics import compute_additional_metrics
from cc_sgcl.utils import seed_torch, to_categorical
from third_party.hyliosr_compat import make_sample, rscls


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test-time retrieval diagnostic for CoSGL checkpoints.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_name", default="ttr_eval")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--quantiles", type=float, nargs="+", default=[0.80, 0.85, 0.88, 0.90, 0.92, 0.95])
    parser.add_argument("--ttr_weights", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.5])
    parser.add_argument("--consistency_weights", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0])
    parser.add_argument("--global_weights", type=float, nargs="+", default=[0.0, 0.25, 0.5])
    parser.add_argument("--vote_weights", type=float, nargs="+", default=[0.0])
    parser.add_argument("--test_bank", choices=["train", "train_val"], default="train_val")
    parser.add_argument("--eval_scope", choices=["labeled", "full"], default="labeled")
    parser.add_argument("--unlabeled_samples", type=int, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--include_class_conditioned", action="store_true")
    parser.add_argument("--rescue_quantiles", type=float, nargs="+", default=[0.90, 0.95, 0.99])
    parser.add_argument("--gate_quantiles", type=float, nargs="+", default=[])
    parser.add_argument(
        "--gate_modes",
        type=str,
        nargs="+",
        choices=["class_far", "inconsistent_far", "global_far", "hybrid"],
        default=[],
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_config_model_data(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    valid_fields = {f.name for f in fields(CCSGCLConfig) if f.init}
    config_values = {k: v for k, v in checkpoint["config"].items() if k in valid_fields}
    config_values["project_root"] = PROJECT_ROOT
    saved_data_root = Path(config_values.get("data_root", ""))
    if saved_data_root.exists():
        config_values["data_root"] = saved_data_root
    elif (PROJECT_ROOT.parent / "dataset").exists():
        config_values["data_root"] = PROJECT_ROOT.parent / "dataset"
    else:
        config_values["data_root"] = PROJECT_ROOT.parents[1] / "dataset"
    config = CCSGCLConfig(**config_values)
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

    train_loader = build_loader(
        x_train_base, y_train_base, config.num_classes, config.data, config.batch_size, False, to_categorical
    )
    val_loader = build_loader(x_val, y_val, config.num_classes, config.data, config.batch_size, False, to_categorical)

    model = CCSGCLModel(
        num_classes=config.num_classes,
        hsi_channels=config.hsi_channels,
        lidar_channels=config.lidar_channels,
        feat_dim=config.feat_dim,
        eta=config.eta,
        method_variant=getattr(config, "method_variant", "cosgl"),
        num_positive_prototypes=getattr(config, "num_positive_prototypes", 3),
        num_negative_prototypes=getattr(config, "num_negative_prototypes", 8),
        np_score_weight=getattr(config, "np_score_weight", 1.0),
        margin_score_weight=getattr(config, "margin_score_weight", 0.5),
        uf_score_weight=getattr(config, "uf_score_weight", 1.0),
        compat_mode=config.compat_mode,
        compat_head_type=getattr(config, "compat_head_type", "deterministic"),
        hsi_encoder_type=getattr(config, "hsi_encoder_type", "cnn"),
        normalize_features=config.normalize_features,
        energy_mode=getattr(config, "energy_mode", "full"),
        logvar_min=getattr(config, "logvar_min", -5.0),
        logvar_max=getattr(config, "logvar_max", 3.0),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    model.eval()

    return checkpoint, config, model, sampler, train_loader, val_loader, hsi, gt_known, gt_full


def labels_to_1based(y: torch.Tensor) -> np.ndarray:
    if y.ndim == 2:
        return torch.argmax(y, dim=1).cpu().numpy().astype(np.int64) + 1
    return y.cpu().numpy().astype(np.int64) + 1


def retrieval_feature(outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    z_h = torch.nn.functional.normalize(outputs["z_h"], dim=-1)
    z_l = torch.nn.functional.normalize(outputs["z_l"], dim=-1)
    fused = torch.cat([z_h, z_l, torch.abs(z_h - z_l)], dim=-1)
    return torch.nn.functional.normalize(fused, dim=-1)


def collect_bank(model, loader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    features = []
    labels = []
    with torch.no_grad():
        for x_h, x_l, y in loader:
            outputs = model(x_h.float().to(device), x_l.float().to(device))
            features.append(retrieval_feature(outputs).detach().cpu())
            labels.append(torch.from_numpy(labels_to_1based(y)))
    return torch.cat(features, dim=0).to(device), torch.cat(labels, dim=0).to(device)


def prepare_score_fusion(checkpoint, config, model, sampler, gt_known, val_loader, device: torch.device):
    score_fusion_mode = getattr(config, "score_fusion_mode", "model")
    score_fusion_stats = checkpoint.get("score_fusion_stats")
    score_fusion_weights = checkpoint.get("score_fusion_weights")
    diagnostics = checkpoint.get("score_fusion_diagnostics")
    if score_fusion_mode not in {"zscore_components", "adaptive_unlabeled_zscore"}:
        return score_fusion_mode, score_fusion_stats, score_fusion_weights, diagnostics

    missing_components = (
        score_fusion_stats is None
        or any(key not in score_fusion_stats for key in SCORE_COMPONENT_KEYS)
    )
    if not missing_components:
        return score_fusion_mode, score_fusion_stats, score_fusion_weights, diagnostics

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
        score_fusion_stats, score_fusion_weights, diagnostics = estimate_adaptive_unlabeled_fusion(
            known_components=val_components,
            unlabeled_components=unlabeled_components,
            base_weights=base_weights,
        )
    else:
        score_fusion_stats = estimate_score_fusion_stats(val_components)
    return score_fusion_mode, score_fusion_stats, score_fusion_weights, diagnostics


def model_score_from_outputs(
    outputs: Dict[str, torch.Tensor],
    score_fusion_mode: str,
    score_fusion_stats,
    score_fusion_weights,
) -> np.ndarray:
    if score_fusion_mode in {"zscore_components", "adaptive_unlabeled_zscore"}:
        components = extract_score_components(outputs)
        return fuse_score_components(components, score_fusion_stats, score_fusion_weights)
    return outputs["open_score"].detach().cpu().numpy()


def retrieval_scores(
    query: torch.Tensor,
    pred_known: np.ndarray,
    bank_features: torch.Tensor,
    bank_labels: torch.Tensor,
    topk: int,
) -> Dict[str, np.ndarray]:
    sim = query @ bank_features.t()
    k_global = min(int(topk), sim.shape[1])
    top_global = torch.topk(sim, k=k_global, dim=1)
    global_similarity = top_global.values.mean(dim=1)
    top_global_labels = bank_labels[top_global.indices]
    top_global_class = top_global_labels[:, 0].detach().cpu().numpy()

    pred_tensor = torch.from_numpy(pred_known.astype(np.int64)).to(bank_labels.device)
    vote_purity = (top_global_labels == pred_tensor[:, None]).float().mean(dim=1)
    class_similarity = torch.full((sim.shape[0],), -1.0, device=sim.device, dtype=sim.dtype)
    for cls in torch.unique(pred_tensor):
        query_mask = pred_tensor == cls
        bank_mask = bank_labels == cls
        if not bool(bank_mask.any()):
            continue
        cls_sim = sim[query_mask][:, bank_mask]
        k_cls = min(int(topk), int(cls_sim.shape[1]))
        class_similarity[query_mask] = torch.topk(cls_sim, k=k_cls, dim=1).values.mean(dim=1)
    consistency_mismatch = (top_global_class != pred_known).astype(np.float32)

    return {
        "ttr_class_distance": (1.0 - class_similarity.detach().cpu().numpy()).astype(np.float32),
        "ttr_global_distance": (1.0 - global_similarity.detach().cpu().numpy()).astype(np.float32),
        "ttr_consistency": consistency_mismatch.astype(np.float32),
        "ttr_vote_impurity": (1.0 - vote_purity.detach().cpu().numpy()).astype(np.float32),
    }


def collect_loader_scores(
    model,
    loader,
    device: torch.device,
    bank_features: torch.Tensor,
    bank_labels: torch.Tensor,
    topk: int,
    score_fusion_mode: str,
    score_fusion_stats,
    score_fusion_weights,
) -> Dict[str, np.ndarray]:
    collected = {
        "model_score": [],
        "pred_known": [],
        "label": [],
        "ttr_class_distance": [],
        "ttr_global_distance": [],
        "ttr_consistency": [],
        "ttr_vote_impurity": [],
    }
    with torch.no_grad():
        for x_h, x_l, y in loader:
            outputs = model(x_h.float().to(device), x_l.float().to(device))
            pred_known = outputs["energies"].argmin(dim=1).detach().cpu().numpy().astype(np.int64) + 1
            query = retrieval_feature(outputs)
            ttr = retrieval_scores(query, pred_known, bank_features, bank_labels, topk=topk)
            collected["model_score"].append(model_score_from_outputs(outputs, score_fusion_mode, score_fusion_stats, score_fusion_weights))
            collected["pred_known"].append(pred_known)
            collected["label"].append(labels_to_1based(y))
            for key, values in ttr.items():
                collected[key].append(values)
    return {key: np.concatenate(values, axis=0) for key, values in collected.items()}


def collect_full_image_scores(
    model,
    sampler,
    data_id: int,
    device: torch.device,
    bank_features: torch.Tensor,
    bank_labels: torch.Tensor,
    topk: int,
    score_fusion_mode: str,
    score_fusion_stats,
    score_fusion_weights,
) -> Dict[str, np.ndarray]:
    rows: Dict[str, list[np.ndarray]] = {
        "model_score": [],
        "pred_known": [],
        "ttr_class_distance": [],
        "ttr_global_distance": [],
        "ttr_consistency": [],
        "ttr_vote_impurity": [],
    }
    with torch.no_grad():
        for row in range(sampler.gt.shape[0]):
            row_samples = sampler.all_sample_row(row)
            x_h_np, x_l_np = split_hsi_lidar_patches(row_samples, data_id=data_id)
            outputs = model(
                torch.from_numpy(to_nchw(x_h_np)).to(device),
                torch.from_numpy(to_nchw(x_l_np)).to(device),
            )
            pred_known = outputs["energies"].argmin(dim=1).detach().cpu().numpy().astype(np.int64) + 1
            query = retrieval_feature(outputs)
            ttr = retrieval_scores(query, pred_known, bank_features, bank_labels, topk=topk)
            rows["model_score"].append(model_score_from_outputs(outputs, score_fusion_mode, score_fusion_stats, score_fusion_weights))
            rows["pred_known"].append(pred_known)
            for key, values in ttr.items():
                rows[key].append(values)
    return {key: np.stack(values, axis=0) for key, values in rows.items()}


def collect_labeled_scores(
    model,
    sampler,
    data_id: int,
    device: torch.device,
    bank_features: torch.Tensor,
    bank_labels: torch.Tensor,
    topk: int,
    score_fusion_mode: str,
    score_fusion_stats,
    score_fusion_weights,
    gt_eval: np.ndarray,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    output = {
        "model_score": np.zeros(gt_eval.shape, dtype=np.float32),
        "pred_known": np.zeros(gt_eval.shape, dtype=np.int64),
        "ttr_class_distance": np.zeros(gt_eval.shape, dtype=np.float32),
        "ttr_global_distance": np.zeros(gt_eval.shape, dtype=np.float32),
        "ttr_consistency": np.zeros(gt_eval.shape, dtype=np.float32),
        "ttr_vote_impurity": np.zeros(gt_eval.shape, dtype=np.float32),
    }
    coords = np.array(np.where(gt_eval != 0)).T.astype(np.int64)
    with torch.no_grad():
        for start in range(0, coords.shape[0], batch_size):
            batch_coords = coords[start : start + batch_size]
            patches = np.asarray([sampler.get_patch(xy) for xy in batch_coords], dtype=np.float32)
            x_h_np, x_l_np = split_hsi_lidar_patches(patches, data_id=data_id)
            outputs = model(
                torch.from_numpy(to_nchw(x_h_np)).to(device),
                torch.from_numpy(to_nchw(x_l_np)).to(device),
            )
            pred_known = outputs["energies"].argmin(dim=1).detach().cpu().numpy().astype(np.int64) + 1
            query = retrieval_feature(outputs)
            ttr = retrieval_scores(query, pred_known, bank_features, bank_labels, topk=topk)
            model_score = model_score_from_outputs(outputs, score_fusion_mode, score_fusion_stats, score_fusion_weights)
            rr = batch_coords[:, 0]
            cc = batch_coords[:, 1]
            output["model_score"][rr, cc] = model_score.astype(np.float32)
            output["pred_known"][rr, cc] = pred_known.astype(np.int64)
            for key, values in ttr.items():
                output[key][rr, cc] = values.astype(np.float32)
    return output


def zscore(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (values - float(mean)) / max(float(std), 1e-6)


def class_conditioned_zscore(values: np.ndarray, pred_known: np.ndarray, class_stats: Dict[int, Tuple[float, float]]) -> np.ndarray:
    output = np.zeros_like(values, dtype=np.float32)
    for cls in np.unique(pred_known.astype(np.int64)):
        mask = pred_known == cls
        mean, std = class_stats.get(int(cls), class_stats.get(0, (0.0, 1.0)))
        output[mask] = zscore(values[mask], mean, std)
    return output.astype(np.float32)


def make_ttr_score(
    score_parts: Dict[str, np.ndarray],
    val_stats: Dict[str, Tuple[float, float]],
    ttr_weight: float,
    consistency_weight: float,
    global_weight: float,
    vote_weight: float = 0.0,
    class_conditioned: bool = False,
) -> np.ndarray:
    if class_conditioned:
        pred_known = score_parts["pred_known"].astype(np.int64)
        model_z = class_conditioned_zscore(score_parts["model_score"], pred_known, val_stats["model_score_by_class"])
        class_z = class_conditioned_zscore(
            score_parts["ttr_class_distance"],
            pred_known,
            val_stats["ttr_class_distance_by_class"],
        )
        global_z = class_conditioned_zscore(
            score_parts["ttr_global_distance"],
            pred_known,
            val_stats["ttr_global_distance_by_class"],
        )
        vote_z = class_conditioned_zscore(
            score_parts["ttr_vote_impurity"],
            pred_known,
            val_stats["ttr_vote_impurity_by_class"],
        )
    else:
        model_z = zscore(score_parts["model_score"], *val_stats["model_score"])
        class_z = zscore(score_parts["ttr_class_distance"], *val_stats["ttr_class_distance"])
        global_z = zscore(score_parts["ttr_global_distance"], *val_stats["ttr_global_distance"])
        vote_z = zscore(score_parts["ttr_vote_impurity"], *val_stats["ttr_vote_impurity"])
    consistency = score_parts["ttr_consistency"].astype(np.float32)
    ttr = class_z + float(global_weight) * global_z + float(consistency_weight) * consistency + float(vote_weight) * vote_z
    return (model_z + float(ttr_weight) * ttr).astype(np.float32)


def class_distance_thresholds(
    val_scores: Dict[str, np.ndarray],
    num_classes: int,
    quantile: float,
) -> np.ndarray:
    labels = val_scores["label"].astype(np.int64)
    distances = val_scores["ttr_class_distance"].astype(np.float32)
    global_threshold = float(np.quantile(distances, quantile))
    thresholds = np.empty((num_classes,), dtype=np.float32)
    for cls in range(1, num_classes + 1):
        mask = labels == cls
        thresholds[cls - 1] = float(np.quantile(distances[mask], quantile)) if np.any(mask) else global_threshold
    return thresholds


def apply_retrieval_rescue(
    pred_open: np.ndarray,
    pred_known: np.ndarray,
    full_scores: Dict[str, np.ndarray],
    rescue_thresholds: np.ndarray,
    unknown_label: int,
) -> tuple[np.ndarray, int]:
    """Rescue high-confidence known predictions using only known-bank retrieval."""

    rescued = pred_open.copy()
    pred_index = np.clip(pred_known.astype(np.int64) - 1, 0, rescue_thresholds.shape[0] - 1)
    was_rejected = rescued == unknown_label
    same_retrieved_class = full_scores["ttr_consistency"].astype(np.float32) <= 0.0
    close_to_predicted_class = full_scores["ttr_class_distance"] <= rescue_thresholds[pred_index]
    rescue_mask = was_rejected & same_retrieved_class & close_to_predicted_class
    rescued[rescue_mask] = pred_known[rescue_mask]
    return rescued, int(np.count_nonzero(rescue_mask))


def global_distance_threshold(val_scores: Dict[str, np.ndarray], quantile: float) -> float:
    return float(np.quantile(val_scores["ttr_global_distance"].astype(np.float32), quantile))


def apply_retrieval_gate(
    pred_open: np.ndarray,
    pred_known: np.ndarray,
    full_scores: Dict[str, np.ndarray],
    class_thresholds: np.ndarray,
    global_threshold: float,
    unknown_label: int,
    mode: str,
) -> tuple[np.ndarray, int]:
    """Reject accepted known predictions only when retrieval evidence is abnormal."""

    gated = pred_open.copy()
    pred_index = np.clip(pred_known.astype(np.int64) - 1, 0, class_thresholds.shape[0] - 1)
    accepted_as_known = gated != unknown_label
    far_from_pred_class = full_scores["ttr_class_distance"] > class_thresholds[pred_index]
    far_from_global_bank = full_scores["ttr_global_distance"] > float(global_threshold)
    retrieved_mismatch = full_scores["ttr_consistency"].astype(np.float32) > 0.0

    if mode == "class_far":
        reject_mask = accepted_as_known & far_from_pred_class
    elif mode == "inconsistent_far":
        reject_mask = accepted_as_known & retrieved_mismatch & far_from_pred_class
    elif mode == "global_far":
        reject_mask = accepted_as_known & far_from_pred_class & far_from_global_bank
    elif mode == "hybrid":
        reject_mask = accepted_as_known & far_from_pred_class & (retrieved_mismatch | far_from_global_bank)
    else:
        raise ValueError(f"Unsupported retrieval gate mode: {mode}")

    gated[reject_mask] = unknown_label
    return gated, int(np.count_nonzero(reject_mask))


def stats_from_validation(val_scores: Dict[str, np.ndarray]) -> Dict[str, Tuple[float, float]]:
    stats = {
        key: (float(np.mean(val_scores[key])), float(np.std(val_scores[key])))
        for key in ("model_score", "ttr_class_distance", "ttr_global_distance", "ttr_vote_impurity")
    }
    pred_known = val_scores["pred_known"].astype(np.int64)
    for key in ("model_score", "ttr_class_distance", "ttr_global_distance", "ttr_vote_impurity"):
        class_stats = {0: stats[key]}
        for cls in np.unique(pred_known):
            mask = pred_known == cls
            if np.any(mask):
                class_stats[int(cls)] = (float(np.mean(val_scores[key][mask])), float(np.std(val_scores[key][mask])))
        stats[f"{key}_by_class"] = class_stats
    return stats


def json_ready_stats(stats: Dict) -> Dict:
    payload = {}
    for key, value in stats.items():
        if isinstance(value, dict):
            payload[key] = {str(cls): list(pair) for cls, pair in value.items()}
        else:
            payload[key] = list(value)
    return payload


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def estimate_adaptive_ttr_weight(val_scores: Dict[str, np.ndarray], unlabeled_scores: Dict[str, np.ndarray], val_stats) -> float:
    val_ttr = zscore(val_scores["ttr_class_distance"], *val_stats["ttr_class_distance"])
    unlabeled_ttr = zscore(unlabeled_scores["ttr_class_distance"], *val_stats["ttr_class_distance"])
    if val_ttr.size == 0 or unlabeled_ttr.size == 0:
        return 0.5
    gap = float(np.mean(unlabeled_ttr) - np.mean(val_ttr))
    return float(np.clip(max(0.0, gap) / 2.0, 0.0, 1.0))


def estimate_unlabeled_youden_threshold(
    known_scores: np.ndarray,
    unlabeled_scores: np.ndarray,
    pseudo_outlier_fraction: float = 0.30,
    mode: str = "youden",
    outlier_weight: float = 1.0,
) -> float:
    """Choose a threshold from known validation and high-score unlabeled pixels.

    The unlabeled pool is sampled from KnownGT==0 pixels only. It is not labeled as
    unknown; only its high-score tail is treated as a pseudo-outlier set for
    threshold calibration.
    """

    known_scores = np.asarray(known_scores, dtype=np.float32).reshape(-1)
    unlabeled_scores = np.asarray(unlabeled_scores, dtype=np.float32).reshape(-1)
    if known_scores.size == 0:
        raise ValueError("Known validation scores are empty; cannot estimate threshold.")
    if unlabeled_scores.size == 0:
        return float(np.quantile(known_scores, 0.95))

    tail_fraction = float(np.clip(pseudo_outlier_fraction, 0.05, 1.0))
    cutoff = float(np.quantile(unlabeled_scores, 1.0 - tail_fraction))
    pseudo_outliers = unlabeled_scores[unlabeled_scores >= cutoff]
    if pseudo_outliers.size == 0:
        return float(np.quantile(known_scores, 0.95))

    candidates = np.unique(np.concatenate([known_scores, pseudo_outliers]))
    if candidates.size > 1024:
        candidates = np.quantile(candidates, np.linspace(0.0, 1.0, 1024))

    best_threshold = float(np.quantile(known_scores, 0.95))
    best_score = -np.inf
    for threshold in candidates:
        known_accept = float(np.mean(known_scores <= threshold))
        outlier_reject = float(np.mean(pseudo_outliers > threshold))
        if mode == "balanced":
            score = min(known_accept, outlier_reject)
        elif mode == "weighted":
            score = known_accept + float(outlier_weight) * outlier_reject
        else:
            score = known_accept + outlier_reject
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    out_dir = checkpoint_path.parent
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    checkpoint, config, model, sampler, train_loader, val_loader, _hsi, gt_known, gt_full = load_config_model_data(
        checkpoint_path, device
    )
    if args.unlabeled_samples is not None:
        config.unlabeled_samples = int(args.unlabeled_samples)
    eval_batch_size = int(args.eval_batch_size or config.batch_size)
    rescue_quantiles = [float(q) for q in args.rescue_quantiles]
    gate_quantiles = [float(q) for q in args.gate_quantiles]
    gate_modes = list(args.gate_modes)

    score_fusion_mode, score_fusion_stats, score_fusion_weights, score_fusion_diagnostics = prepare_score_fusion(
        checkpoint, config, model, sampler, gt_known, val_loader, device
    )
    bank_features, bank_labels = collect_bank(model, train_loader, device)
    test_bank_features, test_bank_labels = bank_features, bank_labels
    if args.test_bank == "train_val":
        val_bank_features, val_bank_labels = collect_bank(model, val_loader, device)
        test_bank_features = torch.cat([bank_features, val_bank_features], dim=0)
        test_bank_labels = torch.cat([bank_labels, val_bank_labels], dim=0)
    val_scores = collect_loader_scores(
        model,
        val_loader,
        device,
        bank_features,
        bank_labels,
        args.topk,
        score_fusion_mode,
        score_fusion_stats,
        score_fusion_weights,
    )
    val_stats = stats_from_validation(val_scores)
    rescue_threshold_map = {
        float(q): class_distance_thresholds(val_scores, config.num_classes, float(q))
        for q in rescue_quantiles
    }
    gate_threshold_map = {
        float(q): (
            class_distance_thresholds(val_scores, config.num_classes, float(q)),
            global_distance_threshold(val_scores, float(q)),
        )
        for q in gate_quantiles
    }

    x_unlabeled = sample_unlabeled_known_zero_patches(
        sampler,
        gt_known,
        num_samples=config.unlabeled_samples,
        seed=config.seed + 1009,
    )
    adaptive_ttr_weight = 0.5
    unlabeled_scores = None
    if x_unlabeled.ndim == 4 and len(x_unlabeled) > 0:
        unlabeled_loader = build_unlabeled_loader(
            x_unlabeled,
            data_id=config.data,
            batch_size=config.batch_size,
            shuffle=False,
        )
        unlabeled_scores = collect_loader_scores(
            model,
            unlabeled_loader,
            device,
            bank_features,
            bank_labels,
            args.topk,
            score_fusion_mode,
            score_fusion_stats,
            score_fusion_weights,
        )
        adaptive_ttr_weight = estimate_adaptive_ttr_weight(val_scores, unlabeled_scores, val_stats)

    gt_eval = build_open_set_ground_truth(gt_known=gt_known, gt_full=gt_full, unknown_label=config.unknown_label)
    if args.eval_scope == "full":
        full_scores = collect_full_image_scores(
            model,
            sampler,
            config.data,
            device,
            test_bank_features,
            test_bank_labels,
            args.topk,
            score_fusion_mode,
            score_fusion_stats,
            score_fusion_weights,
        )
    else:
        full_scores = collect_labeled_scores(
            model,
            sampler,
            config.data,
            device,
            test_bank_features,
            test_bank_labels,
            args.topk,
            score_fusion_mode,
            score_fusion_stats,
            score_fusion_weights,
            gt_eval,
            batch_size=eval_batch_size,
        )
    pred_known = full_scores["pred_known"].astype(np.int64)

    rows: list[dict] = []
    candidates: list[tuple[str, float, float, float, float, bool]] = [
        ("adaptive_ttr", adaptive_ttr_weight, 0.5, 0.25, 0.0, False),
        ("model_only_z", 0.0, 0.0, 0.0, 0.0, False),
    ]
    if args.include_class_conditioned:
        candidates.extend(
            [
                ("class_cond_adaptive_ttr", adaptive_ttr_weight, 0.5, 0.25, 0.0, True),
                ("class_cond_model_z", 0.0, 0.0, 0.0, 0.0, True),
            ]
        )
    for ttr_weight in args.ttr_weights:
        for consistency_weight in args.consistency_weights:
            for global_weight in args.global_weights:
                for vote_weight in args.vote_weights:
                    candidates.append(
                        (
                            "grid",
                            float(ttr_weight),
                            float(consistency_weight),
                            float(global_weight),
                            float(vote_weight),
                            False,
                        )
                    )
                    if args.include_class_conditioned:
                        candidates.append(
                            (
                                "class_cond_ttr",
                                float(ttr_weight),
                                float(consistency_weight),
                                float(global_weight),
                                float(vote_weight),
                                True,
                            )
                        )

    seen = set()
    unique_candidates = []
    for candidate in candidates:
        key = (
            candidate[0],
            round(candidate[1], 6),
            round(candidate[2], 6),
            round(candidate[3], 6),
            round(candidate[4], 6),
            bool(candidate[5]),
        )
        if key not in seen:
            seen.add(key)
            unique_candidates.append(candidate)

    for mode, ttr_weight, consistency_weight, global_weight, vote_weight, class_conditioned in unique_candidates:
        val_ttr_score = make_ttr_score(
            val_scores,
            val_stats,
            ttr_weight,
            consistency_weight,
            global_weight,
            vote_weight,
            class_conditioned=class_conditioned,
        )
        full_ttr_score = make_ttr_score(
            full_scores,
            val_stats,
            ttr_weight,
            consistency_weight,
            global_weight,
            vote_weight,
            class_conditioned=class_conditioned,
        )
        unlabeled_ttr_score = (
            make_ttr_score(
                unlabeled_scores,
                val_stats,
                ttr_weight,
                consistency_weight,
                global_weight,
                vote_weight,
                class_conditioned=class_conditioned,
            )
            if unlabeled_scores is not None
            else None
        )
        val_labels = val_scores["label"].astype(np.int64)
        for q in args.quantiles:
            global_threshold = float(np.quantile(val_ttr_score, q))
            unlabeled_youden_threshold = (
                estimate_unlabeled_youden_threshold(val_ttr_score, unlabeled_ttr_score)
                if unlabeled_ttr_score is not None
                else global_threshold
            )
            unlabeled_balanced_threshold = (
                estimate_unlabeled_youden_threshold(val_ttr_score, unlabeled_ttr_score, mode="balanced")
                if unlabeled_ttr_score is not None
                else global_threshold
            )
            unlabeled_weighted_threshold = (
                estimate_unlabeled_youden_threshold(
                    val_ttr_score,
                    unlabeled_ttr_score,
                    mode="weighted",
                    outlier_weight=2.0,
                )
                if unlabeled_ttr_score is not None
                else global_threshold
            )
            class_thresholds = np.asarray(
                [
                    np.quantile(val_ttr_score[val_labels == cls], q)
                    if np.any(val_labels == cls)
                    else global_threshold
                    for cls in range(1, config.num_classes + 1)
                ],
                dtype=np.float32,
            )
            for threshold_mode in (
                "global",
                "unlabeled_youden",
                "unlabeled_balanced",
                "unlabeled_weighted",
                "class_pred",
            ):
                if threshold_mode == "global":
                    threshold = global_threshold
                    sample_threshold = threshold
                elif threshold_mode == "unlabeled_youden":
                    threshold = unlabeled_youden_threshold
                    sample_threshold = threshold
                elif threshold_mode == "unlabeled_balanced":
                    threshold = unlabeled_balanced_threshold
                    sample_threshold = threshold
                elif threshold_mode == "unlabeled_weighted":
                    threshold = unlabeled_weighted_threshold
                    sample_threshold = threshold
                else:
                    threshold = float(np.mean(class_thresholds))
                    sample_threshold = class_thresholds[pred_known - 1]
                base_pred_open = pred_known.copy()
                base_pred_open[full_ttr_score > sample_threshold] = config.unknown_label
                rescue_options: list[tuple[str, float | None, np.ndarray, int]] = [
                    ("none", None, base_pred_open, 0)
                ]
                for rescue_q, rescue_thresholds in rescue_threshold_map.items():
                    rescued_open, rescued_count = apply_retrieval_rescue(
                        base_pred_open,
                        pred_known,
                        full_scores,
                        rescue_thresholds,
                        config.unknown_label,
                    )
                    rescue_options.append(("known_val_ttr", rescue_q, rescued_open, rescued_count))

                for rescue_mode, rescue_q, rescued_pred_open, rescued_count in rescue_options:
                    combined_gate_options: list[tuple[str, float | None, np.ndarray, int]] = [
                        ("none", None, rescued_pred_open, 0)
                    ]
                    for gate_q, (gate_class_thresholds, gate_global_threshold) in gate_threshold_map.items():
                        for gate_mode in gate_modes:
                            gated_open, gated_count = apply_retrieval_gate(
                                rescued_pred_open,
                                pred_known,
                                full_scores,
                                gate_class_thresholds,
                                gate_global_threshold,
                                config.unknown_label,
                                gate_mode,
                            )
                            combined_gate_options.append((gate_mode, gate_q, gated_open, gated_count))

                    for gate_mode, gate_q, pred_open, gated_count in combined_gate_options:
                        metrics = compute_additional_metrics(pred_open, pred_known, full_ttr_score, gt_eval, config.unknown_label)
                        row = {
                            "mode": mode,
                            "data": config.data,
                            "seed": config.seed,
                            "checkpoint": str(checkpoint_path),
                            "topk": int(args.topk),
                            "eval_scope": args.eval_scope,
                            "threshold_mode": threshold_mode,
                            "threshold_quantile": float(q),
                            "threshold": threshold,
                            "global_threshold": global_threshold,
                            "class_threshold_mean": float(np.mean(class_thresholds)),
                            "class_threshold_std": float(np.std(class_thresholds)),
                            "ttr_weight": float(ttr_weight),
                            "consistency_weight": float(consistency_weight),
                            "global_weight": float(global_weight),
                            "vote_weight": float(vote_weight),
                            "class_conditioned": bool(class_conditioned),
                            "rescue_mode": rescue_mode,
                            "rescue_quantile": "" if rescue_q is None else float(rescue_q),
                            "rescued_count": int(rescued_count),
                            "gate_mode": gate_mode,
                            "gate_quantile": "" if gate_q is None else float(gate_q),
                            "gated_count": int(gated_count),
                            "oa": float(metrics["oa"]),
                            "aa": float(metrics["aa"]),
                            "kappa": float(metrics["kappa"]),
                            "known_accuracy": float(metrics["known_accuracy"]),
                            "unknown_accuracy": float(metrics["unknown_accuracy"]),
                            "macro_f1": float(metrics["macro_f1"]),
                            "auroc": float(metrics["auroc"]),
                            "oscr": float(metrics["oscr"]),
                            "known_rejected_as_unknown": float(metrics["known_rejected_as_unknown"]),
                            "unknown_accepted_as_known": float(metrics["unknown_accepted_as_known"]),
                        }
                        for cls, acc in metrics["class_accuracy"].items():
                            row[f"class_{cls}_accuracy"] = float(acc)
                        rows.append(row)

    def ranking_key(row: dict) -> tuple[float, float, float, float]:
        balanced = min(float(row["known_accuracy"]), float(row["unknown_accuracy"]))
        headline = (float(row["oa"]) + float(row["aa"]) + float(row["kappa"])) / 3.0
        return balanced, headline, float(row["unknown_accuracy"]), float(row["known_accuracy"])

    rows_sorted = sorted(rows, key=ranking_key, reverse=True)
    csv_path = out_dir / f"{args.output_name}.csv"
    json_path = out_dir / f"{args.output_name}_summary.json"
    write_rows(csv_path, rows_sorted)
    json_path.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "score_fusion_mode": score_fusion_mode,
                "score_fusion_weights": score_fusion_weights,
                "score_fusion_diagnostics": score_fusion_diagnostics,
                "val_stats": json_ready_stats(val_stats),
                "adaptive_ttr_weight": adaptive_ttr_weight,
                "test_bank": args.test_bank,
                "eval_scope": args.eval_scope,
                "bank_size": int(bank_features.shape[0]),
                "test_bank_size": int(test_bank_features.shape[0]),
                "top20": rows_sorted[:20],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {csv_path}")
    print(f"Saved {json_path}")
    print(f"adaptive_ttr_weight={adaptive_ttr_weight:.4f}")
    for row in rows_sorted[:10]:
        print(row)


if __name__ == "__main__":
    main()
