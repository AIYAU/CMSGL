"""Data loading and compatibility wrappers for standalone CC-SGCL."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset

from .config import CCSGCLConfig
from .inference import split_hsi_lidar_patches, to_nchw


EXPECTED_OPEN_SET_PROTOCOL = {
    1: {"known": [1, 2, 3, 4, 5, 6], "unknown": [7, 8]},
    2: {"known": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], "unknown": [16]},
    3: {"known": [1, 2, 3, 4, 5, 6], "unknown": [7, 8, 9, 10, 11]},
}


class CustomDataset(Dataset):
    """Simple PyTorch dataset for HSI/LiDAR patch pairs."""

    def __init__(self, data: torch.Tensor, lidar_data: torch.Tensor, labels: torch.Tensor):
        self.data = data
        self.lidar_data = lidar_data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.lidar_data[idx], self.labels[idx]


def load_dataset(config: CCSGCLConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if config.data == 3:
        hsi = loadmat(config.dataset_path)["combinedData"].astype("float32")
        gt_known = loadmat(config.gt_known_path)["label"].astype("int64")
        gt_full = loadmat(config.gt_full_path)["label"].astype("int64")
    else:
        hsi = np.load(config.dataset_path, allow_pickle=True).astype("float32")
        gt_known = np.load(config.gt_known_path, allow_pickle=True).astype("int64")
        gt_full = np.load(config.gt_full_path, allow_pickle=True).astype("int64")
    return hsi, gt_known, gt_full


def infer_modal_channels(patches: np.ndarray, data_id: int) -> Tuple[int, int]:
    x_h, x_l = split_hsi_lidar_patches(patches[:1], data_id=data_id)
    return x_h.shape[-1], x_l.shape[-1]


def stratified_known_train_val_split(
    x_train: np.ndarray,
    y_train: np.ndarray,
    val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_indices: List[int] = []
    val_indices: List[int] = []
    for cls in np.unique(y_train):
        cls_idx = np.where(y_train == cls)[0]
        rng.shuffle(cls_idx)
        if len(cls_idx) <= 1:
            train_indices.extend(cls_idx.tolist())
            continue
        n_val = max(1, int(round(len(cls_idx) * val_ratio)))
        n_val = min(n_val, len(cls_idx) - 1)
        val_indices.extend(cls_idx[:n_val].tolist())
        train_indices.extend(cls_idx[n_val:].tolist())
    train_indices = np.asarray(train_indices, dtype=np.int64)
    val_indices = np.asarray(val_indices, dtype=np.int64)
    return x_train[train_indices], y_train[train_indices], x_train[val_indices], y_train[val_indices]


def build_loader(
    x_joint: np.ndarray,
    y_indices: np.ndarray,
    num_classes: int,
    data_id: int,
    batch_size: int,
    shuffle: bool,
    to_categorical_fn,
) -> DataLoader:
    x_h_np, x_l_np = split_hsi_lidar_patches(x_joint, data_id=data_id)
    x_h = torch.from_numpy(to_nchw(x_h_np))
    x_l = torch.from_numpy(to_nchw(x_l_np))
    y_one_hot = torch.from_numpy(to_categorical_fn(y_indices, num_classes))
    dataset = CustomDataset(x_h, x_l, y_one_hot)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def sample_unlabeled_known_zero_patches(
    sampler,
    gt_known: np.ndarray,
    num_samples: int,
    seed: int,
) -> np.ndarray:
    """Sample unlabeled image patches from pixels absent in KnownGT without using FullGT labels."""

    coords = np.array(np.where(gt_known == 0)).T
    if coords.size == 0 or num_samples <= 0:
        return np.empty((0,), dtype=np.float32)
    rng = np.random.default_rng(seed)
    take = min(int(num_samples), coords.shape[0])
    selected = coords[rng.choice(coords.shape[0], size=take, replace=False)]
    patches = [sampler.get_patch(xy) for xy in selected]
    patches = [patch for patch in patches if len(patch) != 0]
    if not patches:
        return np.empty((0,), dtype=np.float32)
    return np.asarray(patches, dtype=np.float32)


def build_unlabeled_loader(
    x_joint: np.ndarray,
    data_id: int,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    x_h_np, x_l_np = split_hsi_lidar_patches(x_joint, data_id=data_id)
    x_h = torch.from_numpy(to_nchw(x_h_np))
    x_l = torch.from_numpy(to_nchw(x_l_np))
    y_dummy = torch.zeros((x_h.shape[0],), dtype=torch.long)
    dataset = CustomDataset(x_h, x_l, y_dummy)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_known_unknown_class_lists(gt_known: np.ndarray, gt_full: np.ndarray) -> Tuple[List[int], List[int]]:
    known = sorted(np.unique(gt_known[gt_known > 0]).astype(int).tolist())
    unknown = sorted(np.unique(gt_full[np.logical_and(gt_known == 0, gt_full != 0)]).astype(int).tolist())
    return known, unknown


def validate_expected_open_set_protocol(data_id: int, known: List[int], unknown: List[int]) -> None:
    expected = EXPECTED_OPEN_SET_PROTOCOL.get(data_id)
    if expected is None:
        return
    if known != expected["known"] or unknown != expected["unknown"]:
        raise ValueError(
            "Dataset protocol mismatch: "
            f"data={data_id}, known={known}, unknown={unknown}, "
            f"expected_known={expected['known']}, expected_unknown={expected['unknown']}. "
            "Check data_root or set enforce_dataset_protocol=false only for debugging."
        )


def count_labels_1based(labels_zero_based: np.ndarray) -> Dict[str, int]:
    counts = {}
    for cls in sorted(np.unique(labels_zero_based).astype(int).tolist()):
        counts[str(cls + 1)] = int(np.sum(labels_zero_based == cls))
    return counts


def count_eval_labels(gt_eval: np.ndarray) -> Dict[str, int]:
    counts = {}
    for cls in sorted(np.unique(gt_eval[gt_eval > 0]).astype(int).tolist()):
        counts[str(cls)] = int(np.sum(gt_eval == cls))
    return counts


def count_original_unknown_labels(gt_known: np.ndarray, gt_full: np.ndarray) -> Dict[str, int]:
    mask = np.logical_and(gt_known == 0, gt_full != 0)
    counts = {}
    for cls in sorted(np.unique(gt_full[mask]).astype(int).tolist()):
        counts[str(cls)] = int(np.sum(gt_full[mask] == cls))
    return counts


def format_count_dict(name: str, counts: Dict[str, int]) -> str:
    items = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items(), key=lambda x: int(x[0])))
    return f"{name}: {items if items else 'none'}"
