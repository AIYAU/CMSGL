from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.io import loadmat


DATASETS = {
    "Trento": {
        "data_id": 1,
        "image_out": "trento_im.npy",
        "known_out": "trento_raw_gt.npy",
        "full_out": "trento_gt8.npy",
        "hsi_key": "HSI_data",
        "lidar_key": "LiDAR_data",
        "label_key": "All_Label",
    },
    "Houston": {
        "data_id": 2,
        "image_out": "houston_im.npy",
        "known_out": "houston_raw_gt.npy",
        "full_out": "houston_gt16.npy",
        "hsi_key": "HSI_data",
        "lidar_key": "LiDAR_data",
        "label_key": "All_Label",
    },
}


def parse_class_list(value: str | None, labels: np.ndarray) -> list[int]:
    present = sorted(int(v) for v in np.unique(labels) if int(v) > 0)
    if value is None or value.strip() == "":
        if not present:
            raise ValueError("Label image has no positive classes.")
        return [present[-1]]
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def load_required_mat(path: Path, key: str) -> np.ndarray:
    payload = loadmat(path)
    if key not in payload:
        available = ", ".join(k for k in payload if not k.startswith("__"))
        raise KeyError(f"{path} does not contain key {key!r}; available keys: {available}")
    return payload[key]


def prepare_dataset(source_root: Path, output_root: Path, name: str, unknown_classes: Iterable[int]) -> None:
    spec = DATASETS[name]
    source_dir = source_root / name
    output_dir = output_root / name
    output_dir.mkdir(parents=True, exist_ok=True)

    hsi = load_required_mat(source_dir / "HSI_data.mat", spec["hsi_key"]).astype("float32")
    lidar = load_required_mat(source_dir / "LiDAR_data.mat", spec["lidar_key"]).astype("float32")
    labels = load_required_mat(source_dir / "All_Label.mat", spec["label_key"]).astype("int64")

    if lidar.ndim == 2:
        lidar = lidar[:, :, np.newaxis]
    if hsi.shape[:2] != lidar.shape[:2] or hsi.shape[:2] != labels.shape:
        raise ValueError(
            f"Shape mismatch for {name}: hsi={hsi.shape}, lidar={lidar.shape}, labels={labels.shape}"
        )

    full_gt = labels
    known_gt = labels.copy()
    for cls in unknown_classes:
        known_gt[known_gt == int(cls)] = 0

    image = np.concatenate([hsi, lidar], axis=-1).astype("float32")
    np.save(output_dir / spec["image_out"], image)
    np.save(output_dir / spec["known_out"], known_gt.astype("int64"))
    np.save(output_dir / spec["full_out"], full_gt.astype("int64"))

    known = sorted(int(v) for v in np.unique(known_gt) if int(v) > 0)
    unknown = sorted(int(v) for v in np.unique(full_gt[np.logical_and(known_gt == 0, full_gt != 0)]))
    print(f"{name}: wrote {output_dir}")
    print(f"  image shape: {image.shape}")
    print(f"  known classes: {known}")
    print(f"  held-out unknown classes: {unknown}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare local .mat HSI/LiDAR data for CC-SGCL.")
    parser.add_argument("--source_root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output_root", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="Trento")
    parser.add_argument(
        "--unknown_classes",
        default=None,
        help="Comma-separated labels to hide from training/evaluation as open-set unknowns. Defaults to the largest class label.",
    )
    args = parser.parse_args()

    label_spec = DATASETS[args.dataset]
    labels = load_required_mat(args.source_root / args.dataset / "All_Label.mat", label_spec["label_key"])
    unknown_classes = parse_class_list(args.unknown_classes, labels)
    prepare_dataset(args.source_root, args.output_root, args.dataset, unknown_classes)


if __name__ == "__main__":
    main()
