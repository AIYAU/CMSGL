"""Metric helpers for standalone CC-SGCL."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def compute_score_summary(scores: np.ndarray) -> Dict[str, float]:
    if scores.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "count": 0,
        }
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "count": int(scores.size),
    }


def compute_component_summary(components: Dict[str, np.ndarray], mask: np.ndarray) -> Dict[str, Dict[str, float]]:
    summary = {}
    for key, values in components.items():
        selected = values[mask]
        summary[key] = compute_score_summary(selected)
    return summary


def compute_oscr(
    known_scores: np.ndarray,
    unknown_scores: np.ndarray,
    known_correct: np.ndarray,
) -> float:
    if known_scores.size == 0 or unknown_scores.size == 0:
        return float("nan")
    thresholds = np.sort(np.unique(np.concatenate([known_scores, unknown_scores])))[::-1]
    ccr = []
    fpr = []
    for thr in thresholds:
        ccr.append(np.mean((known_scores >= thr) & known_correct))
        fpr.append(np.mean(unknown_scores >= thr))
    ccr = np.asarray(ccr)
    fpr = np.asarray(fpr)
    order = np.argsort(fpr)
    return float(np.trapz(ccr[order], fpr[order]))


def compute_confusion_metrics(
    pred_open: np.ndarray,
    gt_eval: np.ndarray,
    num_classes: int,
) -> Dict[str, object]:
    """Compute OA, AA, Kappa, and per-class accuracy with the HyLiOSR convention."""

    pred = np.asarray(pred_open, dtype=np.int64)
    gt = np.asarray(gt_eval, dtype=np.int64)
    labeled_mask = gt != 0
    n_labeled = int(np.sum(labeled_mask))
    if n_labeled == 0:
        return {
            "oa": float("nan"),
            "aa": float("nan"),
            "kappa": float("nan"),
            "class_accuracy": {},
        }

    cf = np.zeros((num_classes, num_classes), dtype=np.float64)
    pred_labeled = pred[labeled_mask]
    gt_labeled = gt[labeled_mask]
    valid = (pred_labeled >= 1) & (pred_labeled <= num_classes) & (gt_labeled >= 1) & (gt_labeled <= num_classes)
    for p, g in zip(pred_labeled[valid], gt_labeled[valid]):
        cf[int(p) - 1, int(g) - 1] += 1.0

    expected = 0.0
    for cls in range(num_classes):
        expected += (cf[cls, :].sum() / n_labeled) * (cf[:, cls].sum() / n_labeled)

    diagonal = np.diag(cf)
    oa = float(diagonal.sum() / n_labeled)
    denom = 1.0 - expected
    kappa = float((oa - expected) / denom) if abs(denom) > 1e-12 else float("nan")

    class_acc_values = []
    class_accuracy: Dict[str, float] = {}
    for cls in range(num_classes):
        gt_count = cf[:, cls].sum()
        acc = float(diagonal[cls] / gt_count) if gt_count > 0 else float("nan")
        class_accuracy[str(cls + 1)] = acc
        class_acc_values.append(acc)
    aa = float(np.nanmean(np.asarray(class_acc_values, dtype=np.float64)))

    return {
        "oa": oa,
        "aa": aa,
        "kappa": kappa,
        "class_accuracy": class_accuracy,
    }


def compute_additional_metrics(
    pred_open: np.ndarray,
    pred_known: np.ndarray,
    open_score: np.ndarray,
    gt_eval: np.ndarray,
    unknown_label: int,
) -> Dict[str, object]:
    labeled_mask = gt_eval != 0
    gt_labeled = gt_eval[labeled_mask]
    pred_labeled = pred_open[labeled_mask]
    score_labeled = open_score[labeled_mask]

    known_mask = gt_labeled != unknown_label
    unknown_mask = gt_labeled == unknown_label

    known_acc = float(np.mean(pred_labeled[known_mask] == gt_labeled[known_mask])) if known_mask.any() else float("nan")
    unknown_acc = float(np.mean(pred_labeled[unknown_mask] == unknown_label)) if unknown_mask.any() else float("nan")
    macro_f1 = float(f1_score(gt_labeled, pred_labeled, average="macro")) if gt_labeled.size else float("nan")

    auroc = float("nan")
    if known_mask.any() and unknown_mask.any():
        y_bin = unknown_mask.astype(np.uint8)
        auroc = float(roc_auc_score(y_bin, score_labeled))

    known_scores = -score_labeled[known_mask]
    unknown_scores = -score_labeled[unknown_mask]
    known_correct = pred_known[labeled_mask][known_mask] == gt_labeled[known_mask]
    oscr = compute_oscr(known_scores, unknown_scores, known_correct)

    known_rejected = float(np.mean(pred_labeled[known_mask] == unknown_label)) if known_mask.any() else float("nan")
    unknown_accepted = float(np.mean(pred_labeled[unknown_mask] != unknown_label)) if unknown_mask.any() else float("nan")
    confusion_metrics = compute_confusion_metrics(pred_open, gt_eval, num_classes=unknown_label)

    return {
        "oa": confusion_metrics["oa"],
        "aa": confusion_metrics["aa"],
        "kappa": confusion_metrics["kappa"],
        "class_accuracy": confusion_metrics["class_accuracy"],
        "known_accuracy": known_acc,
        "unknown_accuracy": unknown_acc,
        "macro_f1": macro_f1,
        "auroc": auroc,
        "oscr": oscr,
        "known_rejected_as_unknown": known_rejected,
        "unknown_accepted_as_known": unknown_accepted,
    }
