"""Softmax-smoothed top-k Riesz particle-field loss."""

from __future__ import annotations

import os
from typing import Dict, Iterable, Tuple

import torch


_COMPILE = os.environ.get("DRIFT_COMPILE", "1") != "0"


def _rms(value: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(value * value))


@torch.compiler.disable
def _temperature_metrics(temperatures, per_temperature_rms):
    # Keep Python metric-name formatting outside Dynamo, which cannot trace
    # the :g format specifier for symbolic temperatures.
    return {
        f"riesz_topk_softmax_force_rms_tau_{temperature:g}": value.detach()
        for temperature, value in zip(temperatures, per_temperature_rms)
    }


def _softmax_toward(
    source: torch.Tensor,
    target: torch.Tensor,
    target_weight: torch.Tensor,
    tau: float,
    direction_epsilon: float,
    *,
    exclude_self: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the softmax-weighted unit directions toward ``target`` and ESS."""
    if target.shape[1] == 0:
        empty_ess = source.new_zeros(source.shape[:2])
        return torch.zeros_like(source), empty_ess
    if exclude_self and source.shape[1] != target.shape[1]:
        raise ValueError("self-interaction exclusion requires equal particle counts")

    available = target.shape[1] - int(exclude_self)
    if available == 0:
        empty_ess = source.new_zeros(source.shape[:2])
        return torch.zeros_like(source), empty_ess

    distance = torch.cdist(source, target)
    logits = -distance / tau
    if exclude_self:
        diagonal = torch.eye(
            source.shape[1], device=source.device, dtype=torch.bool
        ).unsqueeze(0)
        logits = logits.masked_fill(diagonal, float("-inf"))

    softmax_weight = torch.softmax(logits, dim=-1)
    coefficient = (
        softmax_weight
        * target_weight[:, None, :]
        / distance.clamp_min(float(direction_epsilon))
    )
    if exclude_self:
        coefficient = coefficient.masked_fill(diagonal, 0.0)

    # This is algebraically sum_j a_ij (target_j - source_i) / distance_ij,
    # but avoids materializing a [B, source, target, D] direction tensor.
    toward = torch.bmm(coefficient, target)
    toward.sub_(source * coefficient.sum(dim=-1, keepdim=True))
    ess = 1.0 / torch.sum(softmax_weight * softmax_weight, dim=-1).clamp_min(1e-30)
    return toward, ess


def _riesz_topk_softmax_loss_impl(
    gen: torch.Tensor,
    fixed_pos: torch.Tensor,
    fixed_neg: torch.Tensor | None = None,
    weight_gen: torch.Tensor | None = None,
    weight_pos: torch.Tensor | None = None,
    weight_neg: torch.Tensor | None = None,
    *,
    tau: float = 1.0,
    tau_list: Iterable[float] | None = None,
    epsilon: float = 1e-8,
    rms_epsilon: float = 1e-8,
    direction_epsilon: float = 1e-8,
    normalize_per_temperature: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Regress toward the detached, RMS-normalized softmax Riesz velocity.

    As in ``drift_loss``, softmax logits use distances divided by the global
    batch-derived distance scale. For every configured temperature, the field
    is ``1 / tau`` times attraction toward ``fixed_pos``, minus attraction
    toward the other generated particles, minus attraction toward
    ``fixed_neg``. Fields are summed across ``tau_list``, then the complete
    field is divided by its global RMS magnitude, matching hard top-k Riesz.
    With ``normalize_per_temperature``, each temperature field is instead
    divided by its own RMS before summation, as in Drifting. This mode uses
    Drifting's squared-RMS floor of ``1e-8`` and no final normalization.
    The legacy total-field RMS floor is applied after the square root. Target
    weights multiply the normalized softmax averages, preserving the training
    code's CFG coefficients. The generated particle itself is excluded from
    its self-interaction softmax because its unit direction is undefined.
    """
    temperatures = (
        tuple(float(value) for value in tau_list)
        if tau_list is not None
        else (float(tau),)
    )
    if not temperatures or any(value <= 0 for value in temperatures):
        raise ValueError("tau_list must contain at least one positive temperature")
    if epsilon <= 0 or rms_epsilon <= 0 or direction_epsilon <= 0:
        raise ValueError("epsilon, rms_epsilon, and direction_epsilon must be positive")
    if gen.ndim != 3 or fixed_pos.ndim != 3:
        raise ValueError("gen and fixed_pos must have shape [B, particles, D]")
    if gen.shape[0] != fixed_pos.shape[0] or gen.shape[-1] != fixed_pos.shape[-1]:
        raise ValueError("gen and fixed_pos must have matching batch and feature dimensions")
    if gen.shape[1] < 1 or fixed_pos.shape[1] < 1:
        raise ValueError("gen and fixed_pos must each contain at least one particle")

    if fixed_neg is None:
        fixed_neg = gen.new_empty(gen.shape[0], 0, gen.shape[-1])
    if fixed_neg.ndim != 3:
        raise ValueError("fixed_neg must have shape [B, particles, D]")
    if fixed_neg.shape[0] != gen.shape[0] or fixed_neg.shape[-1] != gen.shape[-1]:
        raise ValueError("fixed_neg and gen must have matching batch and feature dimensions")

    if weight_gen is None:
        weight_gen = torch.ones_like(gen[:, :, 0])
    if weight_pos is None:
        weight_pos = torch.ones_like(fixed_pos[:, :, 0])
    if weight_neg is None:
        weight_neg = torch.ones_like(fixed_neg[:, :, 0])

    expected_shapes = (
        ("weight_gen", weight_gen, gen.shape[:2]),
        ("weight_pos", weight_pos, fixed_pos.shape[:2]),
        ("weight_neg", weight_neg, fixed_neg.shape[:2]),
    )
    for name, value, shape in expected_shapes:
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {tuple(shape)}")

    gen = gen.float()
    fixed_pos = fixed_pos.detach().float()
    fixed_neg = fixed_neg.detach().float()
    weight_gen = weight_gen.detach().float()
    weight_pos = weight_pos.detach().float()
    weight_neg = weight_neg.detach().float()

    with torch.no_grad():
        old_gen = gen.detach()
        scale_targets = torch.cat([old_gen, fixed_neg, fixed_pos], dim=1)
        scale_weights = torch.cat([weight_gen, weight_neg, weight_pos], dim=1)
        scale_distance = torch.cdist(old_gen, scale_targets)
        scale = (
            (scale_distance * scale_weights[:, None, :]).mean()
            / (scale_weights.mean() + float(epsilon))
        )
        scale_inputs = torch.clamp(scale / (gen.shape[-1] ** 0.5), min=1e-3)

        old_gen_scaled = old_gen / scale_inputs
        pos_scaled = fixed_pos / scale_inputs
        neg_scaled = fixed_neg / scale_inputs

        attraction_field = torch.zeros_like(old_gen_scaled)
        self_repulsion_field = torch.zeros_like(old_gen_scaled)
        fixed_negative_field = torch.zeros_like(old_gen_scaled)
        positive_ess_values = []
        generated_ess_values = []
        negative_ess_values = []
        per_temperature_rms = []

        # Since old_gen_scaled = old_gen / (scale / sqrt(D)), using
        # sqrt(D) * tau here gives logits -||old_gen-target|| / (scale * tau),
        # exactly matching Drifting's temperature convention.
        dimension_factor = gen.shape[-1] ** 0.5
        for temperature in temperatures:
            softmax_temperature = float(temperature) * dimension_factor
            toward_pos, positive_ess = _softmax_toward(
                old_gen_scaled,
                pos_scaled,
                weight_pos,
                softmax_temperature,
                direction_epsilon,
            )
            toward_gen, generated_ess = _softmax_toward(
                old_gen_scaled,
                old_gen_scaled,
                weight_gen,
                softmax_temperature,
                direction_epsilon,
                exclude_self=True,
            )
            toward_neg, negative_ess = _softmax_toward(
                old_gen_scaled,
                neg_scaled,
                weight_neg,
                softmax_temperature,
                direction_epsilon,
            )
            temperature_attraction = toward_pos / float(temperature)
            temperature_self_repulsion = -toward_gen / float(temperature)
            temperature_fixed_negative = -toward_neg / float(temperature)
            temperature_field = (
                temperature_attraction
                + temperature_self_repulsion
                + temperature_fixed_negative
            )
            temperature_rms = _rms(temperature_field)
            if normalize_per_temperature:
                temperature_scale = torch.sqrt(
                    torch.clamp(temperature_field.square().mean(), min=1e-8)
                )
                temperature_attraction = temperature_attraction / temperature_scale
                temperature_self_repulsion = temperature_self_repulsion / temperature_scale
                temperature_fixed_negative = temperature_fixed_negative / temperature_scale
            attraction_field.add_(temperature_attraction)
            self_repulsion_field.add_(temperature_self_repulsion)
            fixed_negative_field.add_(temperature_fixed_negative)
            positive_ess_values.append(positive_ess.mean())
            generated_ess_values.append(generated_ess.mean())
            negative_ess_values.append(negative_ess.mean())
            per_temperature_rms.append(temperature_rms)

        raw_field = attraction_field + self_repulsion_field + fixed_negative_field
        force_rms = _rms(raw_field)
        force_scale = torch.clamp(force_rms, min=float(rms_epsilon))
        frozen_velocity = (
            raw_field if normalize_per_temperature else raw_field / force_scale
        ).detach()
        goal_scaled = (old_gen_scaled + frozen_velocity).detach()

    gen_scaled = gen / scale_inputs
    loss = torch.mean((gen_scaled - goal_scaled) ** 2, dim=(-1, -2))

    info: Dict[str, torch.Tensor] = {
        "scale": scale.detach(),
        "riesz_topk_softmax_force_rms": force_rms.detach(),
        "riesz_topk_softmax_tau_count": torch.tensor(
            float(len(temperatures)), device=gen.device
        ),
        "riesz_topk_softmax_velocity_rms": _rms(frozen_velocity).detach(),
        "riesz_topk_softmax_attraction_rms": _rms(attraction_field).detach(),
        "riesz_topk_softmax_self_repulsion_rms": _rms(
            self_repulsion_field
        ).detach(),
        "riesz_topk_softmax_fixed_negative_rms": _rms(
            fixed_negative_field
        ).detach(),
        "riesz_topk_softmax_positive_ess": torch.stack(
            positive_ess_values
        ).mean().detach(),
        "riesz_topk_softmax_generated_ess": torch.stack(
            generated_ess_values
        ).mean().detach(),
        "riesz_topk_softmax_negative_ess": torch.stack(
            negative_ess_values
        ).mean().detach(),
    }
    info.update(_temperature_metrics(temperatures, per_temperature_rms))
    if len(temperatures) == 1:
        info["riesz_topk_softmax_tau"] = torch.tensor(
            temperatures[0], device=gen.device
        )
    return loss, info


if _COMPILE:
    riesz_topk_softmax = torch.compile(
        _riesz_topk_softmax_loss_impl, dynamic=True
    )
else:
    riesz_topk_softmax = _riesz_topk_softmax_loss_impl

# Keep the conventional ``*_loss`` spelling available to callers as well.
riesz_topk_softmax_loss = riesz_topk_softmax
