"""Gaussian MMD with detached amplitude normalization.

This keeps the same Gaussian-MMD interaction and the same effective-support
lengthscale schedule as the previous experiment, but prevents useful feature
branches from being killed when all Gaussian kernel values become tiny.

The key idea is deliberately conservative:

    raw Gaussian MMD loss L
    positive row mass m = mean_i sum_j k(G_i, P_j)
    rescale a = max(1, target_row_mass / m), clipped to max_loss_rescale
    normalized loss = stopgrad(a) * L

Because ``a`` is detached and the SAME scalar multiplies G-G, G-P, P-P and
G-N for a feature branch, this does not change the instantaneous Gaussian-MMD
gradient direction inside that feature branch; it only rescales its magnitude.
It is therefore much closer to Gaussian MMD than row-normalizing each kernel
matrix separately (which would no longer be an MMD objective).
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
    """Per-batch weighted empirical pair mean, matching the Riesz code."""
    pair_weight = left_weight[:, :, None] * right_weight[:, None, :]
    return (values * pair_weight).mean(dim=(-1, -2))


def _target_eff_at_step(
    target_eff_init: float,
    target_eff_final: float,
    target_eff_decay_steps: int,
    current_step: int,
) -> float:
    """Geometrically decay target effective support from init to final."""
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
    """Numerically stable median K_eff from Gaussian logits."""
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
    """Binary-search lengthscale in log-space; no top-k/sorting."""
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
def _positive_support_metrics_from_distance(
    distance_gen_pos: torch.Tensor,
    lengthscale: float,
) -> Dict[str, torch.Tensor]:
    """Stable locality diagnostics computed from logits, not underflowed kernels."""
    ell = max(float(lengthscale), 1e-12)
    logits = -0.5 * (distance_gen_pos / ell).square()
    p = torch.softmax(logits, dim=-1)
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


@torch.no_grad()
def _offdiag_mean(x: torch.Tensor) -> torch.Tensor:
    """Mean off-diagonal entry of [B,N,N], or zero for N<=1."""
    n = int(x.shape[-1])
    if n <= 1:
        return x.new_zeros(())
    mask = ~torch.eye(n, dtype=torch.bool, device=x.device)
    return x[:, mask].mean()


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
    # Same effective-support schedule as the previous run.
    target_eff_init: float | None = 12.0,
    target_eff_final: float | None = 3.0,
    target_eff_decay_steps: int = 30000,
    calibrate_every: int = 200,
    calibration_iters: int = 16,
    lengthscale_min: float = 1e-3,
    lengthscale_max: float = 1e3,
    current_step: int | None = None,
    calibration_key: str | None = None,
    # NEW: amplitude rescue.  The target is a ROW SUM, not a kernel mean.
    # target_positive_row_mass=1 means the average generated particle has
    # total positive Gaussian mass of at least ~1 after loss rescaling.
    normalize_kernel_amplitude: bool = True,
    target_positive_row_mass: float = 1.0,
    max_loss_rescale: float = 1.0e3,
    amplitude_floor: float = 1.0e-12,
    only_scale_up: bool = True,
    log_effective_neighbors: bool = True,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Gaussian-MMD analogue of the direct weighted Riesz loss.

    Raw kernel form:
        E k(G,G') - 2 E k(G,P) + E k(P,P') + 2 E k(G,N)

    The optional amplitude normalization multiplies this ENTIRE raw loss by
    one detached scalar per feature branch.  It does NOT row-normalize the
    kernels inside individual terms, so the relative MMD terms are preserved.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if target_positive_row_mass <= 0:
        raise ValueError("target_positive_row_mass must be positive")
    if max_loss_rescale <= 0:
        raise ValueError("max_loss_rescale must be positive")
    if amplitude_floor <= 0:
        raise ValueError("amplitude_floor must be positive")

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

    # Match direct Riesz/drifting feature normalization exactly.
    with torch.no_grad():
        scale_targets = torch.cat([gen.detach(), fixed_neg, fixed_pos], dim=1)
        scale_weights = torch.cat([weight_gen, weight_neg, weight_pos], dim=1)
        scale_distance = torch.cdist(gen.detach(), scale_targets)
        scale = (
            (scale_distance * scale_weights[:, None, :]).mean()
            / (scale_weights.mean() + float(epsilon))
        )
        feature_dim = gen.shape[-1]
        scale_inputs = torch.clamp(scale / (feature_dim ** 0.5), min=1e-3)

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
        kernel_gen_neg = None
        fixed_negative_similarity = torch.zeros_like(attraction)

    raw_loss = (
        self_similarity
        - 2.0 * attraction
        + target_similarity
        + 2.0 * fixed_negative_similarity
    )

    # ------------------------------------------------------------------
    # Detached amplitude normalization.
    # ------------------------------------------------------------------
    # Use UNWEIGHTED G-P kernel mass so CFG does not define the rescue scale.
    with torch.no_grad():
        gp_row_mass_raw = kernel_gen_pos.detach().sum(dim=-1).mean()

        if normalize_kernel_amplitude:
            ratio = float(target_positive_row_mass) / gp_row_mass_raw.clamp_min(
                float(amplitude_floor)
            )
            if only_scale_up:
                ratio = torch.maximum(ratio, torch.ones_like(ratio))
            loss_rescale = torch.clamp(ratio, max=float(max_loss_rescale))
        else:
            loss_rescale = torch.ones_like(gp_row_mass_raw)

    # Same detached scalar for every MMD term in this feature branch.
    loss = raw_loss * loss_rescale.detach()

    with torch.no_grad():
        gp_kernel_mean_raw = kernel_gen_pos.mean()
        gg_offdiag_mean_raw = _offdiag_mean(kernel_gen_gen)
        pp_offdiag_mean_raw = _offdiag_mean(kernel_pos_pos)
        if kernel_gen_neg is not None and kernel_gen_neg.numel() > 0:
            gn_kernel_mean_raw = kernel_gen_neg.mean()
        else:
            gn_kernel_mean_raw = gen.new_zeros(())

    info: Dict[str, torch.Tensor] = {
        "scale": scale.detach(),
        "gaussian_lengthscale": torch.as_tensor(float(ell), device=gen.device),
        # Raw MMD terms, useful for diagnosing collapse.
        "gaussian_self_similarity_raw": self_similarity.detach().mean(),
        "gaussian_positive_similarity_raw": attraction.detach().mean(),
        "gaussian_target_similarity_raw": target_similarity.detach().mean(),
        "gaussian_fixed_negative_similarity_raw": fixed_negative_similarity.detach().mean(),
        # Amplitude diagnostics.
        "gaussian_gp_kernel_mean_raw": gp_kernel_mean_raw,
        "gaussian_gp_row_mass_raw": gp_row_mass_raw,
        "gaussian_gg_offdiag_kernel_mean_raw": gg_offdiag_mean_raw,
        "gaussian_pp_offdiag_kernel_mean_raw": pp_offdiag_mean_raw,
        "gaussian_gn_kernel_mean_raw": gn_kernel_mean_raw,
        "gaussian_loss_rescale": loss_rescale.detach(),
        "gaussian_gp_row_mass_rescaled": (gp_row_mass_raw * loss_rescale).detach(),
        "gaussian_raw_loss_mean": raw_loss.detach().mean(),
        "gaussian_rescaled_loss_mean": loss.detach().mean(),
    }
    if target_eff is not None:
        info["gaussian_target_eff"] = torch.as_tensor(float(target_eff), device=gen.device)
    if log_effective_neighbors:
        info.update(
            _positive_support_metrics_from_distance(
                distance_gen_pos.detach(), float(ell)
            )
        )

    return loss, info
