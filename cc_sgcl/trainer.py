"""Training utilities for standalone CC-SGCL."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import CCSGCLConfig
from .losses import compute_cc_sgcl_loss, labels_to_indices


@dataclass
class TrainHistory:
    losses: List[Dict[str, float]]
    best_epoch: int
    best_val_accuracy: float
    best_val_loss: float


class CCSGCLTrainer:
    """Trainer for the standalone CC-SGCL method."""

    def __init__(
        self,
        config: CCSGCLConfig,
        model: torch.nn.Module,
        device: torch.device,
        logger: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.config = config
        self.model = model
        self.device = device
        self.log = logger or print
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

    def _move_batch(self, batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_h, x_l, y = batch
        return (
            x_h.float().to(self.device),
            x_l.float().to(self.device),
            y.to(self.device),
        )

    def _make_counterfactual_lidar(
        self, x_l: torch.Tensor, labels: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Pair HSI samples with LiDAR from different known classes."""

        labels = labels_to_indices(labels)
        batch_size = labels.shape[0]
        if batch_size < 2:
            return None

        mismatch_indices = []
        for idx in range(batch_size):
            candidates = torch.where(labels != labels[idx])[0]
            if len(candidates) == 0:
                return None
            sampled = candidates[torch.randint(len(candidates), (1,), device=labels.device)].item()
            mismatch_indices.append(sampled)
        mismatch_indices = torch.tensor(mismatch_indices, device=labels.device)
        return x_l[mismatch_indices]

    def _evaluate_known_loader(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_correct = 0
        total_samples = 0
        total_loss = 0.0
        num_steps = 0
        with torch.no_grad():
            for batch in loader:
                x_h, x_l, y = self._move_batch(batch)
                labels = labels_to_indices(y)
                outputs = self.model(x_h, x_l)
                loss = F.cross_entropy(outputs["logits"], labels)
                preds = torch.argmax(outputs["logits"], dim=1)
                total_correct += int((preds == labels).sum().item())
                total_samples += int(labels.numel())
                total_loss += float(loss.detach().cpu())
                num_steps += 1
        self.model.train()
        return {
            "accuracy": total_correct / max(total_samples, 1),
            "loss": total_loss / max(num_steps, 1),
        }

    def _collect_open_scores(self, loader: DataLoader) -> torch.Tensor:
        scores = []
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                x_h, x_l, _ = self._move_batch(batch)
                outputs = self.model(x_h, x_l)
                scores.append(outputs["open_score"].detach().cpu())
        self.model.train()
        if not scores:
            return torch.empty(0)
        return torch.cat(scores, dim=0)

    def _evaluate_unlabeled_youden_selection(
        self,
        val_loader: DataLoader,
        unlabeled_loader: Optional[DataLoader],
    ) -> Dict[str, float]:
        if unlabeled_loader is None:
            return {
                "pseudo_youden": float("-inf"),
                "selection_q": float("nan"),
                "unlabeled_reject_rate": float("nan"),
            }
        val_scores = self._collect_open_scores(val_loader)
        unlabeled_scores = self._collect_open_scores(unlabeled_loader)
        if val_scores.numel() == 0 or unlabeled_scores.numel() == 0:
            return {
                "pseudo_youden": float("-inf"),
                "selection_q": float("nan"),
                "unlabeled_reject_rate": float("nan"),
            }

        best = {
            "pseudo_youden": float("-inf"),
            "selection_q": float("nan"),
            "unlabeled_reject_rate": float("nan"),
        }
        for q in (0.70, 0.75, 0.80, 0.85, 0.90):
            threshold = torch.quantile(val_scores, q)
            reject_rate = float((unlabeled_scores > threshold).float().mean().item())
            pseudo_youden = float(q + reject_rate - 1.0)
            if pseudo_youden > best["pseudo_youden"]:
                best = {
                    "pseudo_youden": pseudo_youden,
                    "selection_q": float(q),
                    "unlabeled_reject_rate": reject_rate,
                }
        return best

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        unlabeled_loader: Optional[DataLoader] = None,
    ) -> Tuple[TrainHistory, Dict[str, torch.Tensor]]:
        history: List[Dict[str, float]] = []
        best_epoch = 0
        best_val_accuracy = float("-inf")
        best_val_loss = float("inf")
        best_selection_score = float("-inf")
        best_state_dict = copy.deepcopy(self.model.state_dict())

        self.model.train()
        unlabeled_iter = iter(unlabeled_loader) if unlabeled_loader is not None else None
        for epoch in range(self.config.epochs):
            epoch_stats = {"total": 0.0, "ce": 0.0, "margin": 0.0, "cf": 0.0, "np": 0.0, "ds": 0.0, "uf": 0.0, "tail": 0.0, "po": 0.0, "nc": 0.0, "aux": 0.0, "ttr": 0.0, "rp": 0.0}
            num_steps = 0
            cf_enabled = epoch >= self.config.warmup_epochs
            for batch in train_loader:
                x_h, x_l, y = self._move_batch(batch)
                outputs = self.model(x_h, x_l)

                x_l_cf = self._make_counterfactual_lidar(x_l, y) if cf_enabled else None
                cf_energies = None
                cf_outputs = None
                if x_l_cf is not None:
                    cf_outputs = self.model(x_h, x_l_cf)
                    cf_energies = cf_outputs["energies"]

                unlabeled_outputs = None
                if unlabeled_iter is not None:
                    try:
                        unlabeled_batch = next(unlabeled_iter)
                    except StopIteration:
                        unlabeled_iter = iter(unlabeled_loader)
                        unlabeled_batch = next(unlabeled_iter)
                    x_h_u, x_l_u, _ = self._move_batch(unlabeled_batch)
                    unlabeled_outputs = self.model(x_h_u, x_l_u)

                loss_dict = compute_cc_sgcl_loss(
                    energies=outputs["energies"],
                    labels=y,
                    margin=self.config.margin,
                    unknown_margin=self.config.unknown_margin,
                    lambda_margin=self.config.lambda_margin,
                    lambda_cf=self.config.lambda_cf if cf_enabled else 0.0,
                    method_variant=self.config.method_variant,
                    lambda_np=self.config.lambda_np,
                    np_margin=self.config.np_margin,
                    lambda_ds=self.config.lambda_ds,
                    ds_margin=self.config.ds_margin,
                    lambda_uf=self.config.lambda_uf,
                    lambda_tail=self.config.lambda_tail,
                    lambda_po=self.config.lambda_po,
                    lambda_nc=self.config.lambda_nc,
                    nc_temperature=self.config.nc_temperature,
                    nc_margin=self.config.nc_margin,
                    nc_simplex_weight=self.config.nc_simplex_weight,
                    lambda_aux=self.config.lambda_aux,
                    aux_margin=self.config.aux_margin,
                    aux_scale=self.config.aux_scale,
                    lambda_ttr=self.config.lambda_ttr,
                    ttr_temperature=self.config.ttr_temperature,
                    ttr_margin=self.config.ttr_margin,
                    ttr_compact_weight=self.config.ttr_compact_weight,
                    lambda_rp=self.config.lambda_rp,
                    rp_temperature=self.config.rp_temperature,
                    rp_margin=self.config.rp_margin,
                    rp_compact_weight=self.config.rp_compact_weight,
                    rp_unlabeled_weight=self.config.rp_unlabeled_weight,
                    tail_fraction=self.config.tail_fraction,
                    pseudo_outlier_fraction=self.config.pseudo_outlier_fraction,
                    pseudo_outlier_margin=self.config.pseudo_outlier_margin,
                    outputs=outputs,
                    cf_energies=cf_energies,
                    cf_outputs=cf_outputs,
                    unlabeled_outputs=unlabeled_outputs,
                )

                self.optimizer.zero_grad()
                loss_dict["total"].backward()
                self.optimizer.step()

                for key in epoch_stats:
                    epoch_stats[key] += float(loss_dict[key].detach().cpu())
                num_steps += 1

            for key in epoch_stats:
                epoch_stats[key] /= max(num_steps, 1)

            val_stats = {}
            if val_loader is not None:
                val_stats = self._evaluate_known_loader(val_loader)
                if self.config.selection_metric == "unlabeled_youden":
                    val_stats.update(self._evaluate_unlabeled_youden_selection(val_loader, unlabeled_loader))
                val_acc = val_stats["accuracy"]
                val_loss = val_stats["loss"]
                if self.config.selection_metric == "unlabeled_youden":
                    selection_score = (
                        val_acc
                        + self.config.selection_unlabeled_weight * float(val_stats.get("pseudo_youden", float("-inf")))
                    )
                    should_update = (
                        selection_score > best_selection_score
                        or (selection_score == best_selection_score and val_loss < best_val_loss)
                    )
                else:
                    selection_score = val_acc
                    should_update = (
                        val_acc > best_val_accuracy
                        or (val_acc == best_val_accuracy and val_loss < best_val_loss)
                    )
                val_stats["selection_score"] = float(selection_score)
                if should_update:
                    best_selection_score = float(selection_score)
                    best_val_accuracy = val_acc
                    best_val_loss = val_loss
                    best_epoch = epoch + 1
                    best_state_dict = copy.deepcopy(self.model.state_dict())
            else:
                best_epoch = epoch + 1
                best_state_dict = copy.deepcopy(self.model.state_dict())

            epoch_record = {"epoch": epoch + 1, "cf_enabled": cf_enabled, **epoch_stats, **val_stats}
            history.append(epoch_record)

            if val_loader is not None:
                self.log(
                    f"[CC-SGCL][Epoch {epoch + 1}/{self.config.epochs}] "
                    f"cf_enabled={cf_enabled} "
                    f"total={epoch_stats['total']:.4f} "
                    f"ce={epoch_stats['ce']:.4f} "
                    f"margin={epoch_stats['margin']:.4f} "
                    f"cf={epoch_stats['cf']:.4f} "
                    f"np={epoch_stats['np']:.4f} "
                    f"ds={epoch_stats['ds']:.4f} "
                    f"uf={epoch_stats['uf']:.4f} "
                    f"tail={epoch_stats['tail']:.4f} "
                    f"po={epoch_stats['po']:.4f} "
                    f"nc={epoch_stats['nc']:.4f} "
                    f"aux={epoch_stats['aux']:.4f} "
                    f"ttr={epoch_stats['ttr']:.4f} "
                    f"rp={epoch_stats['rp']:.4f} "
                    f"val_acc={val_stats['accuracy']:.4f} "
                    f"val_loss={val_stats['loss']:.4f}"
                )
            else:
                self.log(
                    f"[CC-SGCL][Epoch {epoch + 1}/{self.config.epochs}] "
                    f"cf_enabled={cf_enabled} "
                    f"total={epoch_stats['total']:.4f} "
                    f"ce={epoch_stats['ce']:.4f} "
                    f"margin={epoch_stats['margin']:.4f} "
                    f"cf={epoch_stats['cf']:.4f} "
                    f"np={epoch_stats['np']:.4f} "
                    f"ds={epoch_stats['ds']:.4f} "
                    f"uf={epoch_stats['uf']:.4f} "
                    f"tail={epoch_stats['tail']:.4f} "
                    f"po={epoch_stats['po']:.4f} "
                    f"nc={epoch_stats['nc']:.4f} "
                    f"aux={epoch_stats['aux']:.4f} "
                    f"ttr={epoch_stats['ttr']:.4f} "
                    f"rp={epoch_stats['rp']:.4f}"
                )

        self.model.load_state_dict(best_state_dict)
        if val_loader is None:
            best_val_accuracy = float("nan")
            best_val_loss = float("nan")

        history_obj = TrainHistory(
            losses=history,
            best_epoch=best_epoch,
            best_val_accuracy=best_val_accuracy,
            best_val_loss=best_val_loss,
        )
        return history_obj, best_state_dict
