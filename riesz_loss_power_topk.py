"""Direct weighted Riesz-kernel loss."""

from __future__ import annotations

from typing import Dict, Tuple

import torch


def _weighted_pair_mean(
    distance: torch.Tensor,
    left_weight: torch.Tensor,
    right_weight: torch.Tensor,
) -> torch.Tensor:
    """Return a per-batch weighted mean with empirical-count normalization."""
    pair_weight = left_weight[:, :, None] * right_weight[:, None, :]
    return (distance * pair_weight).mean(dim=(-1, -2))


def _topk_weighted_pair_mean(
    distance: torch.Tensor,
    left_weight: torch.Tensor,
    right_weight: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """Return per-batch weighted mean over nearest top-k right particles.

    distance has shape [B, N, M]. For each left particle, this keeps the
    k smallest finite distances over the right-particle dimension and then
    applies the same empirical-count style normalization as _weighted_pair_mean,
    but over the selected [N, k] pairs.
    """
    if distance.ndim != 3:
        raise ValueError(f"distance must have shape [B, N, M], got {tuple(distance.shape)}")
    if topk <= 0:
        raise ValueError("topk must be positive when provided")

    num_right = int(distance.shape[-1])
    if num_right == 0:
        return torch.zeros(distance.shape[0], device=distance.device, dtype=distance.dtype)

    finite_counts = torch.isfinite(distance).sum(dim=-1)
    max_finite = int(finite_counts.max().item()) if finite_counts.numel() else 0
    if max_finite == 0:
        return torch.zeros(distance.shape[0], device=distance.device, dtype=distance.dtype)

    k = min(int(topk), max_finite)
    vals, idx = torch.topk(distance, k=k, dim=-1, largest=False)

    right_weight_expanded = right_weight[:, None, :].expand(-1, distance.shape[1], -1)
    selected_right_weight = torch.gather(right_weight_expanded, dim=-1, index=idx)
    selected_pair_weight = left_weight[:, :, None] * selected_right_weight

    finite_mask = torch.isfinite(vals)
    vals = torch.where(finite_mask, vals, torch.zeros_like(vals))
    selected_pair_weight = torch.where(finite_mask, selected_pair_weight, torch.zeros_like(selected_pair_weight))

    return (vals * selected_pair_weight).mean(dim=(-1, -2))


def _maybe_topk_pair_mean(
    distance: torch.Tensor,
    left_weight: torch.Tensor,
    right_weight: torch.Tensor,
    topk: int | None,
) -> torch.Tensor:
    if topk is None:
        return _weighted_pair_mean(distance, left_weight, right_weight)
    return _topk_weighted_pair_mean(distance, left_weight, right_weight, int(topk))


def _riesz_distance(
    left: torch.Tensor,
    right: torch.Tensor,
    epsilon: float,
    power: float,
) -> torch.Tensor:
    """Return powered Riesz distance with zero self-distance."""
    sq_distance = torch.cdist(left, right).pow(2)
    distance = (sq_distance + epsilon).pow(power / 2.0)
    distance = distance - float(epsilon) ** (power / 2.0)
    return distance.clamp_min(0.0)


def riesz_loss(
    gen: torch.Tensor,
    fixed_pos: torch.Tensor,
    fixed_neg: torch.Tensor | None = None,
    weight_gen: torch.Tensor | None = None,
    weight_pos: torch.Tensor | None = None,
    weight_neg: torch.Tensor | None = None,
    epsilon: float = 1e-8,
    power: float = 1.0,
    topk: int | None = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute the direct powered Riesz energy-distance loss.

    The base objective is

        2 E d_p(G, P) - E d_p(G, G') - E d_p(P, P')

    where

        d_p(x, y) = (||x-y||^2 + epsilon)^(power/2).

    Optional fixed negatives contribute

        -2 E d_p(G, N),

    with their supplied weights, so they repel generated particles.

    Inputs have shape [B, particles, features] and the returned loss has
    shape [B].
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if power <= 0:
        raise ValueError("power must be positive")
    if topk is not None and int(topk) <= 0:
        raise ValueError("topk must be positive when provided")

    if fixed_neg is None:
        fixed_neg = torch.zeros_like(gen[:, :0, :])

    if weight_gen is None:
        weight_gen = torch.ones_like(gen[:, :, 0])
    if weight_pos is None:
        weight_pos = torch.ones_like(fixed_pos[:, :, 0])
    if weight_neg is None:
        weight_neg = torch.ones_like(fixed_neg[:, :, 0])

    gen = gen.float()
    fixed_pos = fixed_pos.detach().float()
    fixed_neg = fixed_neg.detach().float()
    weight_gen = weight_gen.detach().float()
    weight_pos = weight_pos.detach().float()
    weight_neg = weight_neg.detach().float()

    # Match drift_loss: estimate a characteristic distance for this feature
    # space, then normalize each coordinate by scale/sqrt(feature_dimension).
    with torch.no_grad():
        scale_targets = torch.cat([gen.detach(), fixed_neg, fixed_pos], dim=1)
        scale_weights = torch.cat([weight_gen, weight_neg, weight_pos], dim=1)
        scale_distance = torch.cdist(gen.detach(), scale_targets)
        scale = (
            (scale_distance * scale_weights[:, None, :]).mean()
            / (scale_weights.mean() + float(epsilon))
        )
        feature_dim = gen.shape[-1]
        scale_inputs = torch.clamp(
            scale / (feature_dim ** 0.5),
            min=1e-3,
        )

    gen_scaled = gen / scale_inputs
    pos_scaled = fixed_pos / scale_inputs
    neg_scaled = fixed_neg / scale_inputs

    distance_gen_pos = _riesz_distance(
        gen_scaled,
        pos_scaled,
        epsilon=epsilon,
        power=power,
    )
    distance_gen_gen = _riesz_distance(
        gen_scaled,
        gen_scaled,
        epsilon=epsilon,
        power=power,
    )
    distance_pos_pos = _riesz_distance(
        pos_scaled,
        pos_scaled,
        epsilon=epsilon,
        power=power,
    )

    attraction = _maybe_topk_pair_mean(
        distance_gen_pos,
        weight_gen,
        weight_pos,
        topk=topk,
    )

    # For generated-generated top-k, exclude the zero diagonal self-pair so the
    # nearest neighbours are other generated samples. The original global path
    # is left unchanged when topk is None, to preserve previous experiments.
    if topk is not None:
        if distance_gen_gen.shape[-1] != distance_gen_gen.shape[-2]:
            raise ValueError("generated self-distance matrix must be square")
        diag = torch.eye(
            distance_gen_gen.shape[-1],
            device=distance_gen_gen.device,
            dtype=torch.bool,
        ).unsqueeze(0)
        distance_gen_gen_for_mean = distance_gen_gen.masked_fill(diag, float("inf"))
    else:
        distance_gen_gen_for_mean = distance_gen_gen

    self_repulsion = _maybe_topk_pair_mean(
        distance_gen_gen_for_mean,
        weight_gen,
        weight_gen,
        topk=topk,
    )
    target_repulsion = _weighted_pair_mean(
        distance_pos_pos,
        torch.ones_like(weight_pos),
        torch.ones_like(weight_pos),
    )

    if neg_scaled.shape[1] > 0:
        distance_gen_neg = _riesz_distance(
            gen_scaled,
            neg_scaled,
            epsilon=epsilon,
            power=power,
        )
        fixed_negative_repulsion = _maybe_topk_pair_mean(
            distance_gen_neg,
            weight_gen,
            weight_neg,
            topk=topk,
        )
    else:
        fixed_negative_repulsion = torch.zeros_like(attraction)

    loss = (
        2.0 * attraction
        - self_repulsion
        - target_repulsion
        - 2.0 * fixed_negative_repulsion
    )

    info = {
        "scale": scale.detach(),
        "riesz_power": torch.as_tensor(power, device=gen.device),
        "riesz_topk": torch.as_tensor(-1 if topk is None else int(topk), device=gen.device),
        "riesz_attraction": attraction.detach().mean(),
        "riesz_self_repulsion": self_repulsion.detach().mean(),
        "riesz_target_repulsion": target_repulsion.detach().mean(),
        "riesz_fixed_negative_repulsion": fixed_negative_repulsion.detach().mean(),
    }

    return loss, info