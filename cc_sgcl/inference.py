"""Inference and thresholding utilities for standalone CC-SGCL."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch


SCORE_COMPONENT_KEYS = ("energy", "negative", "margin", "uf", "nc")


def split_hsi_lidar_patches(patches: np.ndarray, data_id: int) -> Tuple[np.ndarray, np.ndarray]:
    """Split a joint HSI-LiDAR patch tensor using the working CC-SGCL convention."""

    assert patches.ndim == 4, f"Expected [N, P, P, C], got {patches.shape}"
    if data_id == 3:
        x_l = patches[:, :, :, -2:]
        x_h = patches[:, :, :, :-2]
    else:
        x_l = patches[:, :, :, -1][:, :, :, np.newaxis]
        x_h = patches[:, :, :, :-1]
    return x_h, x_l


def to_nchw(x: np.ndarray) -> np.ndarray:
    return x.transpose((0, 3, 1, 2)).astype("float32")


def default_score_fusion_weights() -> Dict[str, float]:
    return {
        "energy": 1.0,
        "negative": 0.5,
        "margin": 0.5,
        "uf": 0.5,
        "nc": 0.0,
    }


def extract_score_components(outputs: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    energies = outputs["E_total"].min(dim=1).values
    negative = outputs.get("negative_relation_score", torch.zeros_like(energies))
    margin = outputs.get("margin_uncertainty", torch.zeros_like(energies))
    uf = outputs.get("uf_unknown_score", torch.zeros_like(energies))
    nc = outputs.get("nc_unknown_score", torch.zeros_like(energies))
    return {
        "energy": energies.detach().cpu().numpy(),
        "negative": negative.detach().cpu().numpy(),
        "margin": margin.detach().cpu().numpy(),
        "uf": uf.detach().cpu().numpy(),
        "nc": nc.detach().cpu().numpy(),
    }


def estimate_score_fusion_stats(components: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for key, values in components.items():
        values = np.asarray(values, dtype=np.float32)
        mean = float(np.mean(values)) if values.size else 0.0
        std = float(np.std(values)) if values.size else 1.0
        if not np.isfinite(std) or std < 1e-6:
            std = 1.0
        stats[key] = {"mean": mean, "std": std}
    return stats


def _as_z(values: np.ndarray, stat: Dict[str, float]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return (values - float(stat["mean"])) / max(float(stat["std"]), 1e-6)


def _tail_mean(values: np.ndarray, tail: str, fraction: float = 0.30) -> float:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return 0.0
    take = max(1, int(round(values.size * fraction)))
    ordered = np.sort(values)
    if tail == "high":
        return float(np.mean(ordered[-take:]))
    if tail == "low":
        return float(np.mean(ordered[:take]))
    raise ValueError(f"Unsupported tail: {tail}")


def estimate_adaptive_unlabeled_fusion(
    known_components: Dict[str, np.ndarray],
    unlabeled_components: Optional[Dict[str, np.ndarray]],
    base_weights: Optional[Dict[str, float]] = None,
    max_abs_weight: float = 1.0,
    min_abs_weight: float = 0.15,
    pseudo_outlier_fraction: float = 0.30,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float], Dict[str, Dict[str, float]]]:
    """Estimate component weights from known validation and unlabeled KnownGT==0 pixels."""

    stats = estimate_score_fusion_stats(known_components)
    base_weights = base_weights or default_score_fusion_weights()
    weights = {key: 0.0 for key in SCORE_COMPONENT_KEYS}
    weights["energy"] = float(base_weights.get("energy", 1.0))
    diagnostics: Dict[str, Dict[str, float]] = {}

    if unlabeled_components is None or not any(np.asarray(v).size for v in unlabeled_components.values()):
        for key in ("negative", "margin", "uf", "nc"):
            weights[key] = float(base_weights.get(key, 0.0))
        return stats, weights, {"fallback": {"used": 1.0}}

    energy_unlabeled_z = _as_z(np.asarray(unlabeled_components.get("energy", []), dtype=np.float32), stats["energy"])
    if energy_unlabeled_z.size:
        cutoff = float(np.quantile(energy_unlabeled_z, max(0.0, min(1.0, 1.0 - pseudo_outlier_fraction))))
        pseudo_outlier_mask = energy_unlabeled_z >= cutoff
    else:
        pseudo_outlier_mask = np.array([], dtype=bool)

    component_max = {
        "negative": float(max_abs_weight),
        "margin": 0.0,
        "uf": float(max_abs_weight),
        "nc": float(max_abs_weight),
    }

    for key in SCORE_COMPONENT_KEYS:
        known_z = _as_z(np.asarray(known_components.get(key, []), dtype=np.float32), stats[key])
        unlabeled_z = _as_z(np.asarray(unlabeled_components.get(key, []), dtype=np.float32), stats[key])
        if known_z.size == 0 or unlabeled_z.size == 0:
            diagnostics[key] = {"weight": float(weights.get(key, 0.0)), "reason": 0.0}
            continue

        if key == "energy":
            unlabeled_ref = unlabeled_z
        elif pseudo_outlier_mask.size == unlabeled_z.size and pseudo_outlier_mask.any():
            unlabeled_ref = unlabeled_z[pseudo_outlier_mask]
        else:
            unlabeled_ref = unlabeled_z

        pos_gap = max(0.0, float(np.mean(unlabeled_ref) - np.mean(known_z)))
        neg_gap = max(0.0, float(np.mean(known_z) - np.mean(unlabeled_ref)))
        if pos_gap >= neg_gap:
            direction = 1.0
            gap = pos_gap
        else:
            direction = -1.0
            gap = neg_gap

        magnitude = min(component_max.get(key, float(max_abs_weight)), max(0.0, abs(float(gap)) / 2.0))
        if magnitude < min_abs_weight:
            magnitude = 0.0
        if key == "negative" and direction > 0.0:
            magnitude = max(magnitude, float(base_weights.get("negative", 0.0)))

        if key != "energy":
            weights[key] = direction * magnitude
        diagnostics[key] = {
            "known_z_mean": float(np.mean(known_z)),
            "unlabeled_z_mean": float(np.mean(unlabeled_z)),
            "known_z_high_tail": _tail_mean(known_z, "high"),
            "unlabeled_z_high_tail": _tail_mean(unlabeled_z, "high"),
            "known_z_low_tail": _tail_mean(known_z, "low"),
            "unlabeled_z_low_tail": _tail_mean(unlabeled_z, "low"),
            "pseudo_outlier_z_mean": float(np.mean(unlabeled_ref)),
            "pseudo_outlier_count": float(unlabeled_ref.size),
            "positive_gap": float(pos_gap),
            "negative_gap": float(neg_gap),
            "direction": float(direction),
            "weight": float(weights[key]),
        }

    return stats, weights, diagnostics


def fuse_score_components(
    components: Dict[str, np.ndarray],
    stats: Dict[str, Dict[str, float]],
    weights: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    weights = weights or default_score_fusion_weights()
    score = None
    for key in SCORE_COMPONENT_KEYS:
        values = np.asarray(components.get(key, 0.0), dtype=np.float32)
        stat = stats.get(key, {"mean": 0.0, "std": 1.0})
        z = (values - float(stat["mean"])) / max(float(stat["std"]), 1e-6)
        term = float(weights.get(key, 0.0)) * z
        score = term if score is None else score + term
    return np.asarray(score, dtype=np.float32)


def collect_known_validation_components(
    model: torch.nn.Module,
    loader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    model.eval()
    collected: Dict[str, list[np.ndarray]] = {key: [] for key in SCORE_COMPONENT_KEYS}
    with torch.no_grad():
        for x_h, x_l, _ in loader:
            x_h = x_h.float().to(device)
            x_l = x_l.float().to(device)
            outputs = model(x_h, x_l)
            components = extract_score_components(outputs)
            for key, values in components.items():
                collected[key].append(values)
    return {
        key: np.concatenate(values, axis=0) if values else np.array([], dtype=np.float32)
        for key, values in collected.items()
    }


def collect_known_validation_scores(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    score_fusion_mode: str = "model",
    score_fusion_stats: Optional[Dict[str, Dict[str, float]]] = None,
    score_fusion_weights: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    model.eval()
    scores = []
    with torch.no_grad():
        for x_h, x_l, _ in loader:
            x_h = x_h.float().to(device)
            x_l = x_l.float().to(device)
            outputs = model(x_h, x_l)
            if score_fusion_mode in {"zscore_components", "adaptive_unlabeled_zscore"}:
                if score_fusion_stats is None:
                    raise ValueError("score_fusion_stats is required for zscore_components scoring.")
                components = extract_score_components(outputs)
                scores.append(fuse_score_components(components, score_fusion_stats, score_fusion_weights))
            else:
                scores.append(outputs["open_score"].detach().cpu().numpy())
    if not scores:
        return np.array([], dtype=np.float32)
    return np.concatenate(scores, axis=0)


def estimate_threshold(open_scores_known_val: np.ndarray, quantile: float = 0.95) -> float:
    if open_scores_known_val.size == 0:
        raise ValueError("Known validation scores are empty; cannot estimate threshold.")
    return float(np.quantile(open_scores_known_val, quantile))


def run_full_image_inference(
    model: torch.nn.Module,
    sampler,
    data_id: int,
    device: torch.device,
    threshold: float,
    unknown_label: int,
    score_fusion_mode: str = "model",
    score_fusion_stats: Optional[Dict[str, Dict[str, float]]] = None,
    score_fusion_weights: Optional[Dict[str, float]] = None,
    row_batch_size: int = 256,
) -> Dict[str, np.ndarray]:
    model.eval()
    row, col = sampler.gt.shape
    pred_known_rows = []
    open_score_rows = []

    with torch.no_grad():
        for r in range(row):
            row_samples = sampler.all_sample_row(r)
            pred_chunks = []
            score_chunks = []
            for start in range(0, len(row_samples), row_batch_size):
                chunk = row_samples[start : start + row_batch_size]
                x_h_np, x_l_np = split_hsi_lidar_patches(chunk, data_id=data_id)
                x_h = torch.from_numpy(to_nchw(x_h_np)).to(device)
                x_l = torch.from_numpy(to_nchw(x_l_np)).to(device)
                outputs = model(x_h, x_l)
                energies = outputs["energies"].detach().cpu().numpy()
                pred_chunks.append(np.argmin(energies, axis=1) + 1)
                if score_fusion_mode in {"zscore_components", "adaptive_unlabeled_zscore"}:
                    if score_fusion_stats is None:
                        raise ValueError("score_fusion_stats is required for zscore_components scoring.")
                    components = extract_score_components(outputs)
                    score_chunks.append(fuse_score_components(components, score_fusion_stats, score_fusion_weights))
                else:
                    score_chunks.append(outputs["open_score"].detach().cpu().numpy())
            pred_known_rows.append(np.concatenate(pred_chunks, axis=0))
            open_score_rows.append(np.concatenate(score_chunks, axis=0))

    pred_known = np.stack(pred_known_rows, axis=0)
    open_score = np.stack(open_score_rows, axis=0)
    pred_open = pred_known.copy()
    unknown_mask = open_score > threshold
    pred_open[unknown_mask] = unknown_label
    return {
        "pred_known": pred_known,
        "pred_open": pred_open,
        "open_score": open_score,
    }


def build_open_set_ground_truth(gt_known: np.ndarray, gt_full: np.ndarray, unknown_label: int) -> np.ndarray:
    gt_eval = gt_known.copy()
    gt_eval[np.logical_and(gt_eval == 0, gt_full != 0)] = unknown_label
    return gt_eval
