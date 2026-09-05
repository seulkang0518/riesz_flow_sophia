"""Gaussian-kernel analogue of the direct weighted Riesz loss.

The support construction, CFG weights, empirical pair means, and feature-space
normalization match the direct Riesz implementation.  Locality is controlled
softly by a Gaussian kernel instead of hard top-k selection.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch


_LENGTHSCALE_CACHE: dict[str, tuple[int, float]] = {}


def _weighted_pair_mean(
    values: torch.Tensor,
    left_weight: torch.Tensor,
    right_weight: torch.Tensor,
) -> torch.Tensor:
    pair_weight = left_weight[:, :, None] * right_weight[:, None, :]
    return (values * pair_weight).mean(dim=(-1, -2))


def _target_eff_at_step(
    target_eff_init: float,
    target_eff_final: float,
    target_eff_decay_steps: int,
    current_step: int,
) -> float:
    """Geometrically decay target K_eff from init to final."""
    k0 = float(target_eff_init)
    k1 = float(target_eff_final)
    T = int(target_eff_decay_steps)
    if k0 <= 0 or k1 <= 0:
        raise ValueError("target effective supports must be positive")
    if k0 < k1:
        raise ValueError("expected target_eff_init >= target_eff_final")
    if T <= 0:
        raise ValueError("target_eff_decay_steps must be positive")
    frac = min(max(float(current_step), 0.0) / float(T), 1.0)
    return k0 * ((k1 / k0) ** frac)


@torch.no_grad()
def _median_effective_support(
    distance_gen_pos: torch.Tensor,
    lengthscale: float,
) -> torch.Tensor:
    ell = max(float(lengthscale), 1e-12)
    logits = -0.5 * (distance_gen_pos / ell).square()
    p = torch.softmax(logits, dim=-1)
    eff = 1.0 / p.square().sum(dim=-1).clamp_min(1e-30)
    return eff.median()


@torch.no_grad()
def _solve_lengthscale_for_target_eff(
    distance_gen_pos: torch.Tensor,
    target_eff: float,
    lengthscale_min: float,
    lengthscale_max: float,
    calibration_iters: int,
) -> float:
    """Binary search in log lengthscale; no top-k/sorting is used."""
    n_pos = int(distance_gen_pos.shape[-1])
    target = min(max(float(target_eff), 1.0), float(n_pos))
    lo = float(lengthscale_min)
    hi = float(lengthscale_max)
    if lo <= 0 or hi <= lo:
        raise ValueError("need 0 < lengthscale_min < lengthscale_max")

    for _ in range(max(1, int(calibration_iters))):
        mid = math.sqrt(lo * hi)
        eff_mid = float(_median_effective_support(distance_gen_pos, mid).item())
        if eff_mid < target:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


@torch.no_grad()
def _positive_support_metrics(
    kernel_gen_pos: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Sorting-free locality diagnostics."""
    p = kernel_gen_pos / kernel_gen_pos.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    eff = 1.0 / p.square().sum(dim=-1).clamp_min(1e-30)
    entropy = -(p.clamp_min(1e-30) * p.clamp_min(1e-30).log()).sum(dim=-1)
    return {
        "gaussian_eff_pos_mean": eff.mean(),
        "gaussian_eff_pos_median": eff.median(),
        "gaussian_eff_pos_min": eff.min(),
        "gaussian_eff_pos_max": eff.max(),
        "gaussian_eff_entropy_pos_mean": entropy.exp().mean(),
        "gaussian_eff_entropy_pos_median": entropy.exp().median(),
        "gaussian_frac_eff_pos_le_2": (eff <= 2).float().mean(),
        "gaussian_frac_eff_pos_le_4": (eff <= 4).float().mean(),
        "gaussian_frac_eff_pos_le_8": (eff <= 8).float().mean(),
        "gaussian_frac_eff_pos_le_16": (eff <= 16).float().mean(),
    }


def gaussian_mmd_loss(
    gen: torch.Tensor,
    fixed_pos: torch.Tensor,
    fixed_neg: torch.Tensor | None = None,
    weight_gen: torch.Tensor | None = None,
    weight_pos: torch.Tensor | None = None,
    weight_neg: torch.Tensor | None = None,
    epsilon: float = 1e-8,
    # Fixed-lengthscale mode if target_eff_init/final are both None.
    lengthscale: float = 1.0,
    # Effective-support schedule.
    target_eff_init: float | None = 12.0,
    target_eff_final: float | None = 3.0,
    target_eff_decay_steps: int = 30000,
    calibrate_every: int = 200,
    calibration_iters: int = 16,
    lengthscale_min: float = 1e-3,
    lengthscale_max: float = 1e3,
    current_step: int | None = None,
    calibration_key: str | None = None,
    log_effective_neighbors: bool = True,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute the Gaussian-kernel analogue of the direct Riesz objective.

    Riesz energy form:
        2 E d(G,P) - E d(G,G') - E d(P,P') - 2 E d(G,N)

    With k replacing -d, the corresponding kernel form is:
        E k(G,G') - 2 E k(G,P) + E k(P,P') + 2 E k(G,N)
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
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

    # Match the direct Riesz/drifting feature normalization exactly.
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
            scale / (feature_dim ** 0.5), min=1e-3,
        )

    gen_scaled = gen / scale_inputs
    pos_scaled = fixed_pos / scale_inputs
    neg_scaled = fixed_neg / scale_inputs

    distance_gen_pos = torch.cdist(gen_scaled, pos_scaled)
    distance_gen_gen = torch.cdist(gen_scaled, gen_scaled)
    distance_pos_pos = torch.cdist(pos_scaled, pos_scaled)
    if neg_scaled.shape[1] > 0:
        distance_gen_neg = torch.cdist(gen_scaled, neg_scaled)
    else:
        distance_gen_neg = gen.new_zeros((gen.shape[0], gen.shape[1], 0))

    target_eff = None
    if target_eff_init is not None or target_eff_final is not None:
        if target_eff_init is None or target_eff_final is None:
            raise ValueError("target_eff_init and target_eff_final must both be set")
        if current_step is None:
            raise ValueError("current_step is required for effective-support scheduling")

        target_eff = _target_eff_at_step(
            float(target_eff_init),
            float(target_eff_final),
            int(target_eff_decay_steps),
            int(current_step),
        )
        target_eff = min(max(target_eff, 1.0), float(fixed_pos.shape[1]))

        key = str(calibration_key or "default")
        every = max(1, int(calibrate_every))
        cached = _LENGTHSCALE_CACHE.get(key)
        step_i = int(current_step)
        should_calibrate = (
            cached is None
            or (step_i % every == 0 and cached[0] != step_i)
            or step_i < cached[0]
        )
        if should_calibrate:
            ell = _solve_lengthscale_for_target_eff(
                distance_gen_pos.detach(),
                target_eff,
                float(lengthscale_min),
                float(lengthscale_max),
                int(calibration_iters),
            )
            _LENGTHSCALE_CACHE[key] = (step_i, float(ell))
        else:
            ell = float(cached[1])
    else:
        ell = float(lengthscale)
        if ell <= 0:
            raise ValueError("lengthscale must be positive")

    inv_two_ell_sq = 0.5 / (float(ell) ** 2)
    kernel_gen_pos = torch.exp(-distance_gen_pos.square() * inv_two_ell_sq)
    kernel_gen_gen = torch.exp(-distance_gen_gen.square() * inv_two_ell_sq)
    kernel_pos_pos = torch.exp(-distance_pos_pos.square() * inv_two_ell_sq)

    attraction = _weighted_pair_mean(kernel_gen_pos, weight_gen, weight_pos)
    self_similarity = _weighted_pair_mean(kernel_gen_gen, weight_gen, weight_gen)
    target_similarity = _weighted_pair_mean(
        kernel_pos_pos,
        torch.ones_like(weight_pos),
        torch.ones_like(weight_pos),
    )

    if neg_scaled.shape[1] > 0:
        kernel_gen_neg = torch.exp(-distance_gen_neg.square() * inv_two_ell_sq)
        fixed_negative_similarity = _weighted_pair_mean(
            kernel_gen_neg, weight_gen, weight_neg,
        )
    else:
        fixed_negative_similarity = torch.zeros_like(attraction)

    loss = (
        self_similarity
        - 2.0 * attraction
        + target_similarity
        + 2.0 * fixed_negative_similarity
    )

    info: Dict[str, torch.Tensor] = {
        "scale": scale.detach(),
        "gaussian_lengthscale": torch.as_tensor(float(ell), device=gen.device),
        "gaussian_self_similarity": self_similarity.detach().mean(),
        "gaussian_positive_similarity": attraction.detach().mean(),
        "gaussian_target_similarity": target_similarity.detach().mean(),
        "gaussian_fixed_negative_similarity": fixed_negative_similarity.detach().mean(),
    }
    if target_eff is not None:
        info["gaussian_target_eff"] = torch.as_tensor(float(target_eff), device=gen.device)
    if log_effective_neighbors:
        info.update(_positive_support_metrics(kernel_gen_pos.detach()))

    return loss, info
