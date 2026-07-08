"""Loss functions for the standalone CC-SGCL method."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def labels_to_indices(y: torch.Tensor) -> torch.Tensor:
    """Convert HyLiOSR-style labels to class indices."""

    if y.ndim == 2:
        return torch.argmax(y, dim=1).long()
    if y.ndim == 1:
        return y.long()
    raise ValueError(f"Unsupported label shape: {y.shape}")


def margin_loss(energies: torch.Tensor, labels: torch.Tensor, margin: float) -> torch.Tensor:
    """Encourage true-class energy to be lower than all other class energies."""

    batch_size, num_classes = energies.shape
    e_y = energies.gather(1, labels.view(-1, 1))
    all_margins = margin + e_y - energies
    mask = F.one_hot(labels, num_classes=num_classes).bool()
    all_margins = all_margins.masked_fill(mask, 0.0)
    return torch.clamp(all_margins, min=0.0).sum(dim=1).mean() / max(num_classes - 1, 1)


def counterfactual_loss(cf_energies: Optional[torch.Tensor], unknown_margin: float) -> torch.Tensor:
    """Push counterfactual HSI-LiDAR pairs away from all known classes."""

    if cf_energies is None or cf_energies.numel() == 0:
        return torch.tensor(0.0)
    min_energy = cf_energies.min(dim=1).values
    return torch.clamp(unknown_margin - min_energy, min=0.0).mean()


def _variant_enabled(method_variant: str, feature: str) -> bool:
    table = {
        "mp": {"mp"},
        "np": {"np"},
        "ds": {"ds"},
        "uf": {"uf"},
        "mp_np": {"mp", "np"},
        "mp_ds": {"mp", "ds"},
        "mp_uf": {"mp", "uf"},
        "np_ds": {"np", "ds"},
        "np_uf": {"np", "uf"},
        "all": {"np", "ds", "uf"},
        "manr": {"mp", "np", "ds", "uf"},
    }
    return feature in table.get(method_variant, set())


def negative_prototype_loss(
    outputs: Optional[Dict[str, torch.Tensor]],
    cf_outputs: Optional[Dict[str, torch.Tensor]],
    np_margin: float,
) -> torch.Tensor:
    """Keep real known pairs away from negative prototypes and pull mismatched pairs toward them."""

    if outputs is None or "negative_relation_score" not in outputs:
        return torch.tensor(0.0)
    known_score = outputs["negative_relation_score"]
    loss = known_score.mean()
    if cf_outputs is not None and "negative_relation_score" in cf_outputs:
        cf_score = cf_outputs["negative_relation_score"]
        loss = loss + torch.clamp(np_margin - cf_score, min=0.0).mean()
    return loss


def distribution_shaping_loss(
    outputs: Optional[Dict[str, torch.Tensor]],
    labels: torch.Tensor,
    ds_margin: float,
) -> torch.Tensor:
    """Compact known features around their prototypes while separating prototype anchors."""

    if outputs is None:
        return torch.tensor(0.0)
    labels = labels_to_indices(labels)
    e_h = outputs["E_h"].gather(1, labels.view(-1, 1)).mean()
    e_l = outputs["E_l"].gather(1, labels.view(-1, 1)).mean()
    compact = 0.5 * (e_h + e_l)

    separation_terms = []
    for key in ("prototype_c_h", "prototype_c_l", "prototype_e_k"):
        proto = outputs.get(key)
        if proto is None or proto.shape[0] <= 1:
            continue
        if proto.ndim == 3:
            proto = proto.reshape(-1, proto.shape[-1])
        dist = torch.cdist(proto, proto, p=2).pow(2)
        mask = ~torch.eye(proto.shape[0], dtype=torch.bool, device=proto.device)
        separation_terms.append(torch.clamp(ds_margin - dist[mask], min=0.0).mean())
    if separation_terms:
        separation = torch.stack(separation_terms).mean()
    else:
        separation = torch.zeros((), device=outputs["energies"].device)
    return compact + separation


def unlabeled_filter_loss(
    outputs: Optional[Dict[str, torch.Tensor]],
    labels: torch.Tensor,
    unlabeled_outputs: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    """Train per-class binary filters using known labels and unlabeled image pixels."""

    if outputs is None or outputs.get("uf_logits") is None:
        return torch.tensor(0.0)
    labels = labels_to_indices(labels)
    logits = outputs["uf_logits"]
    targets = F.one_hot(labels, num_classes=logits.shape[1]).float()
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    if unlabeled_outputs is not None and unlabeled_outputs.get("uf_logits") is not None:
        unlabeled_logits = unlabeled_outputs["uf_logits"]
        unlabeled_targets = torch.zeros_like(unlabeled_logits)
        loss = loss + F.binary_cross_entropy_with_logits(unlabeled_logits, unlabeled_targets)
    return loss


def tail_compactness_loss(
    outputs: Optional[Dict[str, torch.Tensor]],
    labels: torch.Tensor,
    tail_fraction: float,
) -> torch.Tensor:
    """Shrink the high-score tail of known samples without using unknown labels."""

    if outputs is None or "E_total" not in outputs:
        return torch.tensor(0.0)
    labels = labels_to_indices(labels)
    true_energy = outputs["E_total"].gather(1, labels.view(-1, 1)).reshape(-1)
    if true_energy.numel() <= 1:
        return torch.zeros((), device=true_energy.device)
    q = max(0.0, min(1.0, 1.0 - float(tail_fraction)))
    tail_anchor = torch.quantile(true_energy.detach(), q)
    return torch.clamp(true_energy - tail_anchor, min=0.0).pow(2).mean()


def pseudo_outlier_separation_loss(
    outputs: Optional[Dict[str, torch.Tensor]],
    labels: torch.Tensor,
    unlabeled_outputs: Optional[Dict[str, torch.Tensor]],
    tail_fraction: float,
    pseudo_outlier_fraction: float,
    margin: float,
) -> torch.Tensor:
    """Push high-risk unlabeled pixels away from the known-score tail."""

    if outputs is None or unlabeled_outputs is None:
        return torch.tensor(0.0)
    if "E_total" not in outputs or "open_score" not in unlabeled_outputs:
        return torch.tensor(0.0)
    labels = labels_to_indices(labels)
    known_scores = outputs["E_total"].gather(1, labels.view(-1, 1)).reshape(-1)
    unlabeled_scores = unlabeled_outputs["open_score"].reshape(-1)
    if known_scores.numel() <= 1 or unlabeled_scores.numel() == 0:
        return torch.zeros((), device=known_scores.device)

    known_q = max(0.0, min(1.0, 1.0 - float(tail_fraction)))
    unlabeled_q = max(0.0, min(1.0, 1.0 - float(pseudo_outlier_fraction)))
    anchor = torch.quantile(known_scores.detach(), known_q) + float(margin)
    cutoff = torch.quantile(unlabeled_scores.detach(), unlabeled_q)
    pseudo_scores = unlabeled_scores[unlabeled_scores.detach() >= cutoff]
    if pseudo_scores.numel() == 0:
        pseudo_scores = unlabeled_scores
    return torch.clamp(anchor - pseudo_scores, min=0.0).pow(2).mean()


def neural_collapse_alignment_loss(
    outputs: Optional[Dict[str, torch.Tensor]],
    labels: torch.Tensor,
    temperature: float,
    margin: float,
    simplex_weight: float,
) -> torch.Tensor:
    """Align known features with class directions and spread class prototypes."""

    if outputs is None or "prototype_similarity" not in outputs:
        return torch.tensor(0.0)
    labels = labels_to_indices(labels)
    similarity = outputs["prototype_similarity"]
    temp = max(float(temperature), 1e-6)
    ce = F.cross_entropy(similarity / temp, labels)

    true_sim = similarity.gather(1, labels.view(-1, 1)).squeeze(1)
    other_sim = similarity.masked_fill(
        F.one_hot(labels, num_classes=similarity.shape[1]).bool(),
        -1e4,
    ).max(dim=1).values
    separation = torch.clamp(float(margin) - true_sim + other_sim, min=0.0).mean()

    proto_terms = []
    for key in ("prototype_c_h", "prototype_c_l"):
        proto = outputs.get(key)
        if proto is None or proto.shape[0] <= 1:
            continue
        if proto.ndim == 3:
            proto = proto.mean(dim=1)
        proto = F.normalize(proto, dim=-1)
        gram = proto @ proto.t()
        mask = ~torch.eye(proto.shape[0], dtype=torch.bool, device=proto.device)
        target = -1.0 / max(proto.shape[0] - 1, 1)
        proto_terms.append((gram[mask] - target).pow(2).mean())
    simplex = torch.stack(proto_terms).mean() if proto_terms else torch.zeros((), device=similarity.device)
    return ce + separation + float(simplex_weight) * simplex


def auxiliary_arc_margin_loss(
    outputs: Optional[Dict[str, torch.Tensor]],
    labels: torch.Tensor,
    margin: float,
    scale: float,
) -> torch.Tensor:
    """Auxiliary ArcFace-style cosine classifier for stronger known-class separation."""

    if outputs is None or "aux_cosine" not in outputs:
        return torch.tensor(0.0)
    labels = labels_to_indices(labels)
    cosine = outputs["aux_cosine"].clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    sine = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=1e-6))
    cos_m = torch.cos(torch.tensor(float(margin), device=cosine.device, dtype=cosine.dtype))
    sin_m = torch.sin(torch.tensor(float(margin), device=cosine.device, dtype=cosine.dtype))
    target_logits = cosine * cos_m - sine * sin_m
    logits = cosine.clone()
    one_hot = F.one_hot(labels, num_classes=cosine.shape[1]).bool()
    logits = torch.where(one_hot, target_logits, logits)
    return F.cross_entropy(logits * float(scale), labels)


def ttr_retrieval_alignment_loss(
    outputs: Optional[Dict[str, torch.Tensor]],
    labels: torch.Tensor,
    temperature: float,
    margin: float,
    compact_weight: float,
) -> torch.Tensor:
    """Shape known features for test-time retrieval without using unknown labels."""

    if outputs is None or "z_h" not in outputs or "z_l" not in outputs:
        return torch.tensor(0.0)
    labels = labels_to_indices(labels)
    z_h = F.normalize(outputs["z_h"], dim=-1)
    z_l = F.normalize(outputs["z_l"], dim=-1)
    features = F.normalize(torch.cat([z_h, z_l, torch.abs(z_h - z_l)], dim=-1), dim=-1)
    batch_size = features.shape[0]
    if batch_size <= 1:
        return torch.zeros((), device=features.device)

    same = labels.view(-1, 1) == labels.view(1, -1)
    eye = torch.eye(batch_size, dtype=torch.bool, device=features.device)
    positive_mask = same & ~eye
    valid_anchor = positive_mask.any(dim=1)
    if not bool(valid_anchor.any()):
        return torch.zeros((), device=features.device)

    logits = (features @ features.t()) / max(float(temperature), 1e-6)
    logits = logits.masked_fill(eye, -1e4)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    supcon = -(log_prob * positive_mask.float()).sum(dim=1) / positive_mask.float().sum(dim=1).clamp_min(1.0)
    supcon = supcon[valid_anchor].mean()

    similarity = features @ features.t()
    pos_mean = (similarity * positive_mask.float()).sum(dim=1) / positive_mask.float().sum(dim=1).clamp_min(1.0)
    neg_similarity = similarity.masked_fill(same, -1e4)
    hardest_negative = neg_similarity.max(dim=1).values
    margin_loss = torch.clamp(float(margin) - pos_mean + hardest_negative, min=0.0)
    margin_loss = margin_loss[valid_anchor].mean()
    compact_loss = (1.0 - pos_mean[valid_anchor]).clamp_min(0.0).mean()
    return supcon + margin_loss + float(compact_weight) * compact_loss


def _retrieval_feature_from_outputs(outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    z_h = F.normalize(outputs["z_h"], dim=-1)
    z_l = F.normalize(outputs["z_l"], dim=-1)
    return F.normalize(torch.cat([z_h, z_l, torch.abs(z_h - z_l)], dim=-1), dim=-1)


def retrieval_prototype_open_loss(
    outputs: Optional[Dict[str, torch.Tensor]],
    labels: torch.Tensor,
    unlabeled_outputs: Optional[Dict[str, torch.Tensor]],
    temperature: float,
    margin: float,
    compact_weight: float,
    unlabeled_weight: float,
    pseudo_outlier_fraction: float,
) -> torch.Tensor:
    """Shape the same feature space used by test-time retrieval.

    Known samples form batch class prototypes.  High-score unlabeled pixels from
    KnownGT==0 are treated only as pseudo-outlier candidates, never as ground
    truth unknown labels.
    """

    if outputs is None or "z_h" not in outputs or "z_l" not in outputs:
        return torch.tensor(0.0)
    labels = labels_to_indices(labels)
    features = _retrieval_feature_from_outputs(outputs)
    device = features.device
    unique_labels = torch.unique(labels)
    if unique_labels.numel() <= 1:
        return torch.zeros((), device=device)

    centroids = []
    target_indices = torch.empty_like(labels)
    for idx, cls in enumerate(unique_labels):
        mask = labels == cls
        centroids.append(F.normalize(features[mask].mean(dim=0), dim=0))
        target_indices[mask] = idx
    centroids = torch.stack(centroids, dim=0)
    similarity = features @ centroids.t()
    temp = max(float(temperature), 1e-6)
    ce = F.cross_entropy(similarity / temp, target_indices)

    true_sim = similarity.gather(1, target_indices.view(-1, 1)).squeeze(1)
    other_sim = similarity.masked_fill(
        F.one_hot(target_indices, num_classes=centroids.shape[0]).bool(),
        -1e4,
    ).max(dim=1).values
    hard_negative = torch.clamp(float(margin) - true_sim + other_sim, min=0.0).mean()
    compact = (1.0 - true_sim).clamp_min(0.0).mean()

    unlabeled_loss = torch.zeros((), device=device)
    if (
        unlabeled_outputs is not None
        and "z_h" in unlabeled_outputs
        and "z_l" in unlabeled_outputs
        and "open_score" in unlabeled_outputs
    ):
        unlabeled_features = _retrieval_feature_from_outputs(unlabeled_outputs)
        unlabeled_scores = unlabeled_outputs["open_score"].reshape(-1)
        if unlabeled_features.shape[0] > 0 and unlabeled_scores.numel() > 0:
            q = max(0.0, min(1.0, 1.0 - float(pseudo_outlier_fraction)))
            cutoff = torch.quantile(unlabeled_scores.detach(), q)
            tail_mask = unlabeled_scores.detach() >= cutoff
            if not bool(tail_mask.any()):
                tail_mask = torch.ones_like(unlabeled_scores, dtype=torch.bool)
            pseudo_features = unlabeled_features[tail_mask]
            max_known_sim = (pseudo_features @ centroids.detach().t()).max(dim=1).values
            accept_anchor = true_sim.detach().mean() - float(margin)
            unlabeled_loss = torch.clamp(max_known_sim - accept_anchor, min=0.0).pow(2).mean()

    return ce + hard_negative + float(compact_weight) * compact + float(unlabeled_weight) * unlabeled_loss


def compute_cc_sgcl_loss(
    energies: torch.Tensor,
    labels: torch.Tensor,
    margin: float,
    unknown_margin: float,
    lambda_margin: float,
    lambda_cf: float,
    method_variant: str = "cosgl",
    lambda_np: float = 0.0,
    np_margin: float = 0.5,
    lambda_ds: float = 0.0,
    ds_margin: float = 1.0,
    lambda_uf: float = 0.0,
    lambda_tail: float = 0.0,
    lambda_po: float = 0.0,
    lambda_nc: float = 0.0,
    nc_temperature: float = 0.1,
    nc_margin: float = 0.2,
    nc_simplex_weight: float = 0.1,
    lambda_aux: float = 0.0,
    aux_margin: float = 0.2,
    aux_scale: float = 16.0,
    lambda_ttr: float = 0.0,
    ttr_temperature: float = 0.2,
    ttr_margin: float = 0.2,
    ttr_compact_weight: float = 0.1,
    lambda_rp: float = 0.0,
    rp_temperature: float = 0.15,
    rp_margin: float = 0.15,
    rp_compact_weight: float = 0.1,
    rp_unlabeled_weight: float = 0.5,
    tail_fraction: float = 0.30,
    pseudo_outlier_fraction: float = 0.30,
    pseudo_outlier_margin: float = 0.5,
    outputs: Optional[Dict[str, torch.Tensor]] = None,
    cf_energies: Optional[torch.Tensor] = None,
    cf_outputs: Optional[Dict[str, torch.Tensor]] = None,
    unlabeled_outputs: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """Compute the full CC-SGCL objective."""

    labels = labels_to_indices(labels)
    ce = F.cross_entropy(-energies, labels)
    m_loss = margin_loss(energies, labels, margin=margin)
    cf_loss = counterfactual_loss(cf_energies, unknown_margin=unknown_margin).to(energies.device)
    np_loss = torch.zeros((), device=energies.device)
    ds_loss = torch.zeros((), device=energies.device)
    uf_loss = torch.zeros((), device=energies.device)
    tail_loss = torch.zeros((), device=energies.device)
    po_loss = torch.zeros((), device=energies.device)
    nc_loss = torch.zeros((), device=energies.device)
    aux_loss = torch.zeros((), device=energies.device)
    ttr_loss = torch.zeros((), device=energies.device)
    rp_loss = torch.zeros((), device=energies.device)
    if _variant_enabled(method_variant, "np"):
        np_loss = negative_prototype_loss(outputs, cf_outputs, np_margin=np_margin).to(energies.device)
    if _variant_enabled(method_variant, "ds"):
        ds_loss = distribution_shaping_loss(outputs, labels, ds_margin=ds_margin).to(energies.device)
    if _variant_enabled(method_variant, "uf"):
        uf_loss = unlabeled_filter_loss(outputs, labels, unlabeled_outputs=unlabeled_outputs).to(energies.device)
    if lambda_tail > 0:
        tail_loss = tail_compactness_loss(outputs, labels, tail_fraction=tail_fraction).to(energies.device)
    if lambda_po > 0:
        po_loss = pseudo_outlier_separation_loss(
            outputs,
            labels,
            unlabeled_outputs=unlabeled_outputs,
            tail_fraction=tail_fraction,
            pseudo_outlier_fraction=pseudo_outlier_fraction,
            margin=pseudo_outlier_margin,
        ).to(energies.device)
    if lambda_nc > 0:
        nc_loss = neural_collapse_alignment_loss(
            outputs,
            labels,
            temperature=nc_temperature,
            margin=nc_margin,
            simplex_weight=nc_simplex_weight,
        ).to(energies.device)
    if lambda_aux > 0:
        aux_loss = auxiliary_arc_margin_loss(
            outputs,
            labels,
            margin=aux_margin,
            scale=aux_scale,
        ).to(energies.device)
    if lambda_ttr > 0:
        ttr_loss = ttr_retrieval_alignment_loss(
            outputs,
            labels,
            temperature=ttr_temperature,
            margin=ttr_margin,
            compact_weight=ttr_compact_weight,
        ).to(energies.device)
    if lambda_rp > 0:
        rp_loss = retrieval_prototype_open_loss(
            outputs,
            labels,
            unlabeled_outputs=unlabeled_outputs,
            temperature=rp_temperature,
            margin=rp_margin,
            compact_weight=rp_compact_weight,
            unlabeled_weight=rp_unlabeled_weight,
            pseudo_outlier_fraction=pseudo_outlier_fraction,
        ).to(energies.device)
    total = (
        ce
        + lambda_margin * m_loss
        + lambda_cf * cf_loss
        + lambda_np * np_loss
        + lambda_ds * ds_loss
        + lambda_uf * uf_loss
        + lambda_tail * tail_loss
        + lambda_po * po_loss
        + lambda_nc * nc_loss
        + lambda_aux * aux_loss
        + lambda_ttr * ttr_loss
        + lambda_rp * rp_loss
    )
    return {
        "total": total,
        "ce": ce,
        "margin": m_loss,
        "cf": cf_loss,
        "np": np_loss,
        "ds": ds_loss,
        "uf": uf_loss,
        "tail": tail_loss,
        "po": po_loss,
        "nc": nc_loss,
        "aux": aux_loss,
        "ttr": ttr_loss,
        "rp": rp_loss,
    }
