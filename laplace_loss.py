"""Normalized Laplace unit-direction field for generator training.

This is intentionally NOT vanilla Laplace MMD and NOT the original Drifting
field.  It uses Laplace softmax weights but averages unit directions, making it
a soft analogue of q=1 top-k Riesz:

    positive:      sum_j a^+_ij (y^+_j - x_i) / ||y^+_j - x_i||
    generated:     sum_j a^g_ij (x_j   - x_i) / ||x_j   - x_i||
    unconditional: sum_j a^-_ij (y^-_j - x_i) / ||y^-_j - x_i||

where a_ij is a row-normalized Laplace affinity

    a_ij propto exp(-d_ij / tau).

The generated self-pair i=j is excluded.  The field used for training is

    V = cfg * V_pos - V_gen - (cfg - 1) * V_uncond,

when the caller supplies the same weights as the Riesz trainer.  A stop-gradient
fixed-point MSE is used so gradients flow through the generator/feature map but
not through the field construction itself.

Bandwidth tau follows a configurable decreasing schedule.  Distances entering
the Laplace logits are divided by the same detached characteristic distance
``scale`` used by the existing particle losses, so tau is dimensionless and
comparable across feature branches.  This avoids the Gaussian failure mode in
which absolute kernel magnitudes vanished: here the softmax affinities always
sum to one.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch


def _scheduled_tau(
    tau_init: float,
    tau_final: float,
    tau_decay_steps: int,
    current_step: int,
    schedule: str,
) -> float:
    tau0 = float(tau_init)
    tau1 = float(tau_final)
    T = int(tau_decay_steps)
    if tau0 <= 0 or tau1 <= 0:
        raise ValueError("tau_init and tau_final must be positive")
    if tau0 < tau1:
        raise ValueError("expected a decreasing bandwidth: tau_init >= tau_final")
    if T <= 0:
        raise ValueError("tau_decay_steps must be positive")

    frac = min(max(float(current_step), 0.0) / float(T), 1.0)
    mode = str(schedule).lower()
    if mode == "geometric":
        return tau0 * ((tau1 / tau0) ** frac)
    if mode == "linear":
        return tau0 + frac * (tau1 - tau0)
    raise ValueError(f"unknown tau schedule: {schedule!r}")


@torch.no_grad()
def _laplace_unit_field(
    source: torch.Tensor,
    target: torch.Tensor,
    target_weight: torch.Tensor,
    distance_scale: torch.Tensor,
    tau: float,
    epsilon: float,
    exclude_diagonal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return normalized Laplace unit-direction field and its row affinities.

    ``source``: [B, N, D]
    ``target``: [B, M, D]
    ``target_weight``: [B, M], non-negative

    Multiplying all target weights in a row by the same constant does not alter
    the affinity distribution.  Their average magnitude is handled separately
    by the caller (this is what preserves cfg vs cfg-1 strength).
    """
    B, N, D = source.shape
    M = target.shape[1]
    if M == 0:
        return source.new_zeros((B, N, D)), source.new_zeros((B, N, 0))

    dist = torch.cdist(source, target)  # [B,N,M]
    dist_norm = dist / distance_scale.clamp_min(float(epsilon))

    # Preserve relative target weights inside the normalized kernel, while
    # factoring out their mean magnitude in the caller.
    mean_w = target_weight.mean(dim=-1, keepdim=True)
    rel_w = torch.where(
        mean_w > float(epsilon),
        target_weight / mean_w.clamp_min(float(epsilon)),
        torch.ones_like(target_weight),
    )
    logits = -dist_norm / float(tau)
    logits = logits + rel_w.clamp_min(float(epsilon)).log()[:, None, :]

    diag_mask = None
    if exclude_diagonal:
        if N != M:
            raise ValueError("exclude_diagonal requires a square source-target matrix")
        diag_mask = torch.eye(N, dtype=torch.bool, device=source.device).unsqueeze(0)
        logits = logits.masked_fill(diag_mask, float("-inf"))

    affinity = torch.softmax(logits, dim=-1)
    if diag_mask is not None:
        affinity = affinity.masked_fill(diag_mask, 0.0)

    # sum_j a_ij (y_j-x_i)/d_ij without materializing [B,N,M,D].
    inv_dist = dist.clamp_min(float(epsilon)).reciprocal()
    if diag_mask is not None:
        inv_dist = inv_dist.masked_fill(diag_mask, 0.0)
    coeff = affinity * inv_dist
    weighted_target = torch.bmm(coeff, target)
    coeff_sum = coeff.sum(dim=-1, keepdim=True)
    field = weighted_target - source * coeff_sum
    return field, affinity


@torch.no_grad()
def _support_stats(prefix: str, affinity: torch.Tensor) -> Dict[str, torch.Tensor]:
    if affinity.numel() == 0:
        z = affinity.new_zeros(())
        return {
            f"laplace_eff_{prefix}_mean": z,
            f"laplace_eff_{prefix}_median": z,
        }
    p = affinity / affinity.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    eff = 1.0 / p.square().sum(dim=-1).clamp_min(1e-30)
    entropy_eff = (-(p.clamp_min(1e-30) * p.clamp_min(1e-30).log()).sum(dim=-1)).exp()
    return {
        f"laplace_eff_{prefix}_mean": eff.mean(),
        f"laplace_eff_{prefix}_median": eff.median(),
        f"laplace_eff_entropy_{prefix}_mean": entropy_eff.mean(),
        f"laplace_eff_entropy_{prefix}_median": entropy_eff.median(),
        f"laplace_frac_eff_{prefix}_le_2": (eff <= 2).float().mean(),
        f"laplace_frac_eff_{prefix}_le_4": (eff <= 4).float().mean(),
        f"laplace_frac_eff_{prefix}_le_8": (eff <= 8).float().mean(),
        f"laplace_frac_eff_{prefix}_le_16": (eff <= 16).float().mean(),
    }


def laplace_unit_field_loss(
    gen: torch.Tensor,
    fixed_pos: torch.Tensor,
    fixed_neg: torch.Tensor | None = None,
    weight_gen: torch.Tensor | None = None,
    weight_pos: torch.Tensor | None = None,
    weight_neg: torch.Tensor | None = None,
    epsilon: float = 1e-8,
    rms_epsilon: float = 1e-8,
    tau_init: float = 0.20,
    tau_final: float = 0.05,
    tau_decay_steps: int = 30000,
    tau_schedule: str = "geometric",
    current_step: int | None = None,
    # If True, multiply by the literal 1/tau from grad log Laplace KDE.
    # False is cleaner for a locality-only ablation because tau then changes
    # the neighbour weights without also globally amplifying the field.
    include_inverse_tau: bool = False,
    field_scale: float = 1.0,
    log_effective_neighbors: bool = True,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Train toward a normalized Laplace unit-direction field.

    This function returns a per-batch loss [B], matching the existing
    particle-loss API used by the trainer.
    """
    if epsilon <= 0 or rms_epsilon <= 0:
        raise ValueError("epsilon and rms_epsilon must be positive")
    if field_scale <= 0:
        raise ValueError("field_scale must be positive")
    if current_step is None:
        raise ValueError("current_step is required for the bandwidth schedule")

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

    if (weight_gen < 0).any() or (weight_pos < 0).any() or (weight_neg < 0).any():
        raise ValueError("Laplace affinity weights must be non-negative")

    tau = _scheduled_tau(
        tau_init=tau_init,
        tau_final=tau_final,
        tau_decay_steps=tau_decay_steps,
        current_step=int(current_step),
        schedule=tau_schedule,
    )

    old_gen = gen.detach()

    # Same characteristic-distance estimator used by the Riesz code, but use
    # d/scale in the Laplace logits.  Thus a typical normalized distance is O(1)
    # rather than O(sqrt(D)), and tau has a stable interpretation across features.
    with torch.no_grad():
        scale_targets = torch.cat([old_gen, fixed_neg, fixed_pos], dim=1)
        scale_weights = torch.cat([weight_gen, weight_neg, weight_pos], dim=1)
        scale_distance = torch.cdist(old_gen, scale_targets)
        scale = (
            (scale_distance * scale_weights[:, None, :]).mean()
            / (scale_weights.mean() + float(epsilon))
        ).clamp_min(float(epsilon))

        feature_dim = int(gen.shape[-1])
        scale_inputs = torch.clamp(scale / math.sqrt(feature_dim), min=1e-3)

        pos_unit, pos_aff = _laplace_unit_field(
            old_gen,
            fixed_pos,
            weight_pos,
            distance_scale=scale,
            tau=tau,
            epsilon=epsilon,
            exclude_diagonal=False,
        )
        gen_unit_toward, gen_aff = _laplace_unit_field(
            old_gen,
            old_gen,
            weight_gen,
            distance_scale=scale,
            tau=tau,
            epsilon=epsilon,
            exclude_diagonal=True,
        )
        neg_unit, neg_aff = _laplace_unit_field(
            old_gen,
            fixed_neg,
            weight_neg,
            distance_scale=scale,
            tau=tau,
            epsilon=epsilon,
            exclude_diagonal=False,
        )

        # In the current Riesz trainer these row strengths are cfg, 1, cfg-1.
        pos_strength = weight_pos.mean(dim=-1)[:, None, None]
        gen_strength = weight_gen.mean(dim=-1)[:, None, None]
        if fixed_neg.shape[1] > 0:
            neg_strength = weight_neg.mean(dim=-1)[:, None, None]
        else:
            neg_strength = torch.zeros_like(pos_strength)

        # Positive field points toward real data.  gen_unit_toward and neg_unit
        # point toward generated/unconditional targets, so subtracting them gives
        # repulsion.
        V_pos = pos_strength * pos_unit
        V_gen_rep = -gen_strength * gen_unit_toward
        V_neg_rep = -neg_strength * neg_unit
        raw_field = V_pos + V_gen_rep + V_neg_rep

        # Keep this option for backward compatibility.  Because the complete
        # field is RMS-normalized below, a positive global 1/tau multiplier does
        # not change the normalized direction (except at the RMS floor).
        if include_inverse_tau:
            raw_field = raw_field / float(tau)

        # Match the canonical Riesz / OT fixed-target wrapper: normalize the
        # complete field by one global RMS, freeze it, then regress toward one
        # detached field step.
        force_mean_square = torch.mean(raw_field * raw_field)
        force_rms = torch.sqrt(force_mean_square)
        force_scale = torch.clamp(force_rms, min=float(rms_epsilon))
        frozen_velocity = (raw_field / force_scale).detach()

        # Optional explicit step-size multiplier.  field_scale=1.0 exactly
        # matches the canonical RMS-normalized wrapper.
        frozen_velocity = frozen_velocity * float(field_scale)

        # Unit directions are unchanged by the isotropic feature scaling used
        # for the fixed-point surrogate.
        goal_scaled = (old_gen / scale_inputs + frozen_velocity).detach()

        info: Dict[str, torch.Tensor] = {
            "scale": scale.detach(),
            "laplace_tau": torch.as_tensor(tau, device=gen.device),
            "laplace_inverse_tau_enabled": torch.as_tensor(float(include_inverse_tau), device=gen.device),
            "laplace_pos_field_rms": V_pos.square().mean().sqrt(),
            "laplace_gen_rep_field_rms": V_gen_rep.square().mean().sqrt(),
            "laplace_neg_rep_field_rms": V_neg_rep.square().mean().sqrt(),
            "laplace_raw_field_rms": force_rms.detach(),
            "laplace_force_scale": force_scale.detach(),
            "laplace_frozen_velocity_rms": frozen_velocity.square().mean().sqrt(),
            "laplace_total_field_rms": force_rms.detach(),
            "laplace_pos_strength": pos_strength.mean(),
            "laplace_neg_strength": neg_strength.mean(),
        }
        if log_effective_neighbors:
            info.update(_support_stats("pos", pos_aff))
            info.update(_support_stats("gen", gen_aff))
            if neg_aff.numel() > 0:
                info.update(_support_stats("neg", neg_aff))

    gen_scaled = gen / scale_inputs
    diff = gen_scaled - goal_scaled
    loss = diff.square().mean(dim=(-1, -2))
    return loss, {k: v.mean() if torch.is_tensor(v) and v.ndim > 0 else v for k, v in info.items()}
