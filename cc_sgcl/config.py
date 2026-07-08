"""Configuration objects for the standalone CC-SGCL project."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class CCSGCLConfig:
    """Runtime and method configuration for standalone CC-SGCL."""

    project_root: Path
    data_root: Path
    data: int = 3
    num_train: int = 20
    batch_size: int = 32
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-4
    feat_dim: int = 128
    eta: float = 1.0
    method_variant: str = "cosgl"
    compat_mode: str = "h2l_only"
    compat_head_type: str = "deterministic"
    hsi_encoder_type: str = "cnn"
    logvar_min: float = -5.0
    logvar_max: float = 3.0
    energy_mode: str = "full"
    margin: float = 1.0
    unknown_margin: float = 8.0
    lambda_margin: float = 0.5
    lambda_cf: float = 0.0
    num_positive_prototypes: int = 3
    num_negative_prototypes: int = 8
    lambda_np: float = 0.1
    np_margin: float = 0.5
    np_score_weight: float = 1.0
    margin_score_weight: float = 0.5
    score_fusion_mode: str = "model"
    z_energy_weight: float = 1.0
    z_neg_weight: float = 0.5
    z_margin_weight: float = 0.0
    z_uf_weight: float = 0.0
    z_nc_weight: float = 0.0
    lambda_ds: float = 0.05
    ds_margin: float = 1.0
    lambda_uf: float = 0.1
    lambda_tail: float = 0.0
    lambda_po: float = 0.0
    lambda_nc: float = 0.0
    nc_temperature: float = 0.1
    nc_margin: float = 0.2
    nc_simplex_weight: float = 0.1
    lambda_aux: float = 0.0
    aux_margin: float = 0.2
    aux_scale: float = 16.0
    lambda_ttr: float = 0.0
    ttr_temperature: float = 0.2
    ttr_margin: float = 0.2
    ttr_compact_weight: float = 0.1
    lambda_rp: float = 0.0
    rp_temperature: float = 0.15
    rp_margin: float = 0.15
    rp_compact_weight: float = 0.1
    rp_unlabeled_weight: float = 0.5
    tail_fraction: float = 0.30
    pseudo_outlier_fraction: float = 0.30
    pseudo_outlier_margin: float = 0.5
    selection_metric: str = "known_ce"
    selection_unlabeled_weight: float = 0.2
    uf_score_weight: float = 1.0
    unlabeled_samples: int = 512
    patch_size: int = 9
    val_ratio: float = 0.2
    threshold_quantile: float = 0.95
    warmup_epochs: int = 0
    normalize_features: bool = True
    num_workers: int = 0
    device: str = "cuda"
    output_root: str = "outputs"
    seed: int = 0
    save_model: bool = True
    save_maps: bool = True
    run_final_eval: bool = True
    eval_only: bool = False
    require_unknown_eval: bool = True
    enforce_dataset_protocol: bool = True
    checkpoint: Optional[str] = None
    run_name: Optional[str] = None

    dataset_name: str = field(init=False)
    dataset_path: Path = field(init=False)
    gt_known_path: Path = field(init=False)
    gt_full_path: Path = field(init=False)
    key: str = field(init=False)

    num_classes: Optional[int] = None
    hsi_channels: Optional[int] = None
    lidar_channels: Optional[int] = None

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root).resolve()
        self.data_root = Path(self.data_root).resolve()
        if self.method_variant not in {
            "cosgl",
            "mp",
            "np",
            "ds",
            "uf",
            "mp_np",
            "mp_ds",
            "mp_uf",
            "np_ds",
            "np_uf",
            "all",
            "manr",
        }:
            raise ValueError(f"Unsupported method_variant: {self.method_variant}")
        if self.compat_mode not in {"bidirectional", "h2l_only"}:
            raise ValueError(f"Unsupported compat_mode: {self.compat_mode}")
        if self.compat_head_type not in {"deterministic", "probabilistic"}:
            raise ValueError(f"Unsupported compat_head_type: {self.compat_head_type}")
        if self.hsi_encoder_type not in {"cnn", "spectral_rwkv", "spectral_vrwkv", "spectral_vrwkv_official", "spectral_vrwkv6", "spectral_vrwkv6_cuda"}:
            raise ValueError(f"Unsupported hsi_encoder_type: {self.hsi_encoder_type}")
        if self.compat_head_type == "probabilistic" and self.compat_mode != "h2l_only":
            raise ValueError("Probabilistic compatibility currently supports compat_mode='h2l_only' only.")
        if self.logvar_min > self.logvar_max:
            raise ValueError("logvar_min must not exceed logvar_max.")
        if self.energy_mode not in {"full", "no_compat", "hsi_only", "lidar_only"}:
            raise ValueError(f"Unsupported energy_mode: {self.energy_mode}")
        if self.score_fusion_mode not in {"model", "zscore_components", "adaptive_unlabeled_zscore"}:
            raise ValueError(f"Unsupported score_fusion_mode: {self.score_fusion_mode}")
        if self.selection_metric not in {"known_ce", "unlabeled_youden"}:
            raise ValueError(f"Unsupported selection_metric: {self.selection_metric}")
        self._resolve_dataset_paths()

    def _resolve_dataset_paths(self) -> None:
        if self.data == 1:
            self.dataset_name = "Trento"
            self.dataset_path = self.data_root / "Trento" / "trento_im.npy"
            self.gt_known_path = self.data_root / "Trento" / "trento_raw_gt.npy"
            self.gt_full_path = self.data_root / "Trento" / "trento_gt8.npy"
        elif self.data == 2:
            self.dataset_name = "Houston"
            self.dataset_path = self.data_root / "Houston" / "houston_im.npy"
            self.gt_known_path = self.data_root / "Houston" / "houston_raw_gt.npy"
            self.gt_full_path = self.data_root / "Houston" / "houston_gt16.npy"
            if not self.dataset_path.exists():
                fallback = self.data_root / "Houston_invalid_no_unknown_class16"
                self.dataset_path = fallback / "houston_im.npy"
                self.gt_known_path = fallback / "houston_raw_gt.npy"
                self.gt_full_path = fallback / "houston_gt16.npy"
        elif self.data == 3:
            self.dataset_name = "Muufl"
            self.dataset_path = self.data_root / "Muufl" / "muufl_im.mat"
            self.gt_known_path = self.data_root / "Muufl" / "muufl_raw_gt.mat"
            self.gt_full_path = self.data_root / "Muufl" / "muufl_gt12.mat"
            if not self.dataset_path.exists():
                fallback = self.project_root.parents[1] / "dataset" / "Muufl"
                self.dataset_path = fallback / "muufl_im.mat"
                self.gt_known_path = fallback / "muufl_raw_gt.mat"
                self.gt_full_path = fallback / "muufl_gt12.mat"
        else:
            raise ValueError(f"Unsupported dataset id: {self.data}")
        self.key = self.dataset_path.name.split("_")[0]

    @property
    def output_root_dir(self) -> Path:
        return self.project_root / self.output_root

    @property
    def run_dir(self) -> Path:
        if self.run_name is None:
            raise RuntimeError("run_name is not initialized yet.")
        return self.output_root_dir / self.run_name

    @property
    def unknown_label(self) -> int:
        if self.num_classes is None:
            raise RuntimeError("num_classes is not initialized yet.")
        return self.num_classes + 1
