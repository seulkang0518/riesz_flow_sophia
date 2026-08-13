"""Direct weighted full or sliced powered Riesz-kernel loss with optional top-k.

This is a drop-in replacement/variant of riesz_loss_sliced.py.

It supports:
  - full Riesz or sliced Riesz via use_sliced
  - powered distance via power
  - fixed top-k via topk
  - scheduled top-k via topk_schedule + current_step

For use with train_power_topk.py, either:
  1. replace the import to use this file for sliced_riesz_loss, or
  2. copy this file over riesz_loss_sliced.py.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F


def _weighted_pair_mean(
    distance: torch.Tensor,
    left_weight: torch.Tensor,
    right_weight: torch.Tensor,
) -> torch.Tensor:
    """Return per-batch weighted mean for distance [B, N, M]."""
    pair_weight = left_weight[:, :, None] * right_weight[:, None, :]
    return (distance * pair_weight).mean(dim=(-1, -2))


def _weighted_pair_mean_sliced(
    distance: torch.Tensor,
    left_weight: torch.Tensor,
    right_weight: torch.Tensor,
) -> torch.Tensor:
    """Return per-batch/per-projection weighted mean for distance [B, N, M, P]."""
    pair_weight = (
        left_weight[:, :, None, None]
        * right_weight[:, None, :, None]
    )
    return (distance * pair_weight).mean(dim=(1, 2))


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
    selected_pair_weight = torch.where(
        finite_mask,
        selected_pair_weight,
        torch.zeros_like(selected_pair_weight),
    )

    return (vals * selected_pair_weight).mean(dim=(-1, -2))


def _topk_weighted_pair_mean_sliced(
    distance: torch.Tensor,
    left_weight: torch.Tensor,
    right_weight: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """Return per-batch/per-projection mean over nearest top-k right particles.

    distance has shape [B, N, M, P]. For each batch, left particle, and
    projection, this keeps the k smallest finite distances over M.

    Returns shape [B, P], matching _weighted_pair_mean_sliced.
    """
    if distance.ndim != 4:
        raise ValueError(f"distance must have shape [B, N, M, P], got {tuple(distance.shape)}")
    if topk <= 0:
        raise ValueError("topk must be positive when provided")

    num_right = int(distance.shape[2])
    if num_right == 0:
        return torch.zeros(
            distance.shape[0],
            distance.shape[3],
            device=distance.device,
            dtype=distance.dtype,
        )

    # Work with [B, N, P, M] so top-k is over the final dimension.
    distance_bnpm = distance.permute(0, 1, 3, 2)
    finite_counts = torch.isfinite(distance_bnpm).sum(dim=-1)
    max_finite = int(finite_counts.max().item()) if finite_counts.numel() else 0
    if max_finite == 0:
        return torch.zeros(
            distance.shape[0],
            distance.shape[3],
            device=distance.device,
            dtype=distance.dtype,
        )

    k = min(int(topk), max_finite)
    vals, idx = torch.topk(distance_bnpm, k=k, dim=-1, largest=False)

    # right_weight: [B, M] -> [B, 1, 1, M], expanded to [B, N, P, M].
    right_weight_expanded = right_weight[:, None, None, :].expand(
        -1,
        distance.shape[1],
        distance.shape[3],
        -1,
    )
    selected_right_weight = torch.gather(right_weight_expanded, dim=-1, index=idx)
    selected_pair_weight = left_weight[:, :, None, None] * selected_right_weight

    finite_mask = torch.isfinite(vals)
    vals = torch.where(finite_mask, vals, torch.zeros_like(vals))
    selected_pair_weight = torch.where(
        finite_mask,
        selected_pair_weight,
        torch.zeros_like(selected_pair_weight),
    )

    # Mean over N and selected k, leave [B, P].
    return (vals * selected_pair_weight).mean(dim=(1, 3))


def _maybe_topk_pair_mean(
    distance: torch.Tensor,
    left_weight: torch.Tensor,
    right_weight: torch.Tensor,
    topk: int | None,
) -> torch.Tensor:
    if topk is None:
        return _weighted_pair_mean(distance, left_weight, right_weight)
    return _topk_weighted_pair_mean(distance, left_weight, right_weight, int(topk))


def _maybe_topk_pair_mean_sliced(
    distance: torch.Tensor,
    left_weight: torch.Tensor,
    right_weight: torch.Tensor,
    topk: int | None,
) -> torch.Tensor:
    if topk is None:
        return _weighted_pair_mean_sliced(distance, left_weight, right_weight)
    return _topk_weighted_pair_mean_sliced(distance, left_weight, right_weight, int(topk))


def _resolve_topk(
    topk: int | None,
    topk_schedule: Any | None,
    current_step: int | None,
) -> int | None:
    """Resolve active top-k from either a fixed value or a step schedule.

    Accepted schedule formats:

      topk_schedule:
        - [0, 60000, 20]
        - [60000, null, 15]

    or:

      topk_schedule:
        - start: 0
          end: 60000
          topk: 20
        - start: 60000
          end: null
          topk: 15

    The interval convention is start <= current_step < end. If end is None,
    the interval is open-ended.
    """
    if topk_schedule is None:
        return topk

    if current_step is None:
        raise ValueError("current_step must be provided when topk_schedule is used")

    step = int(current_step)

    for entry in topk_schedule:
        if isinstance(entry, dict):
            start = int(entry["start"])
            end = entry.get("end", None)
            k = int(entry["topk"])
        else:
            if len(entry) != 3:
                raise ValueError(
                    "Each topk_schedule entry must be [start, end, topk] "
                    f"or a dict with start/end/topk. Got: {entry!r}"
                )
            start, end, k = entry
            start = int(start)
            k = int(k)

        if k <= 0:
            raise ValueError(f"topk_schedule contains non-positive topk={k}")

        if end is None:
            if step >= start:
                return k
        else:
            if start <= step < int(end):
                return k

    raise ValueError(f"No topk_schedule entry matched current_step={step}")


def _sample_unit_directions(
    feature_dim: int,
    num_projections: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    directions = torch.randn(
        feature_dim,
        num_projections,
        device=device,
        dtype=dtype,
    )
    return F.normalize(directions, p=2, dim=0)


def _powered_distance_from_sq(
    sq_distance: torch.Tensor,
    epsilon: float,
    power: float,
) -> torch.Tensor:
    distance = (sq_distance + float(epsilon)).pow(float(power) / 2.0)
    distance = distance - float(epsilon) ** (float(power) / 2.0)
    return distance.clamp_min(0.0)


def _full_powered_riesz_terms(
    gen_scaled: torch.Tensor,
    pos_scaled: torch.Tensor,
    neg_scaled: torch.Tensor,
    weight_gen: torch.Tensor,
    weight_pos: torch.Tensor,
    weight_neg: torch.Tensor,
    epsilon: float,
    power: float,
    topk: int | None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    distance_gen_pos = _powered_distance_from_sq(
        torch.cdist(gen_scaled, pos_scaled).pow(2),
        epsilon=epsilon,
        power=power,
    )
    distance_gen_gen = _powered_distance_from_sq(
        torch.cdist(gen_scaled, gen_scaled).pow(2),
        epsilon=epsilon,
        power=power,
    )
    distance_pos_pos = _powered_distance_from_sq(
        torch.cdist(pos_scaled, pos_scaled).pow(2),
        epsilon=epsilon,
        power=power,
    )

    attraction = _maybe_topk_pair_mean(
        distance_gen_pos,
        weight_gen,
        weight_pos,
        topk=topk,
    )

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

    # Keep target-target global, as in the non-sliced top-k powered loss.
    target_repulsion = _weighted_pair_mean(
        distance_pos_pos,
        torch.ones_like(weight_pos),
        torch.ones_like(weight_pos),
    )

    if neg_scaled.shape[1] > 0:
        distance_gen_neg = _powered_distance_from_sq(
            torch.cdist(gen_scaled, neg_scaled).pow(2),
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

    return (
        attraction,
        self_repulsion,
        target_repulsion,
        fixed_negative_repulsion,
    )


def _sliced_powered_riesz_terms(
    gen_scaled: torch.Tensor,
    pos_scaled: torch.Tensor,
    neg_scaled: torch.Tensor,
    weight_gen: torch.Tensor,
    weight_pos: torch.Tensor,
    weight_neg: torch.Tensor,
    num_projections: int,
    epsilon: float,
    power: float,
    topk: int | None,
    self_support_scaled: torch.Tensor | None = None,
    weight_self: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    feature_dim = gen_scaled.shape[-1]

    directions = _sample_unit_directions(
        feature_dim=feature_dim,
        num_projections=num_projections,
        device=gen_scaled.device,
        dtype=gen_scaled.dtype,
    )

    gen_projected = torch.matmul(gen_scaled, directions)
    pos_projected = torch.matmul(pos_scaled, directions)
    neg_projected = torch.matmul(neg_scaled, directions)

    using_generated_self = self_support_scaled is None
    if using_generated_self:
        self_projected = gen_projected
        if weight_self is None:
            weight_self = weight_gen
    else:
        self_projected = torch.matmul(self_support_scaled, directions)
        if weight_self is None:
            weight_self = torch.ones(
                self_projected.shape[:2],
                device=self_projected.device,
                dtype=weight_gen.dtype,
            )

    distance_gen_pos = _powered_distance_from_sq(
        (
            gen_projected[:, :, None, :]
            - pos_projected[:, None, :, :]
        ).pow(2),
        epsilon=epsilon,
        power=power,
    )

    distance_gen_self = _powered_distance_from_sq(
        (
            gen_projected[:, :, None, :]
            - self_projected[:, None, :, :]
        ).pow(2),
        epsilon=epsilon,
        power=power,
    )

    distance_pos_pos = _powered_distance_from_sq(
        (
            pos_projected[:, :, None, :]
            - pos_projected[:, None, :, :]
        ).pow(2),
        epsilon=epsilon,
        power=power,
    )

    attraction_per_projection = _maybe_topk_pair_mean_sliced(
        distance_gen_pos,
        weight_gen,
        weight_pos,
        topk=topk,
    )

    if topk is not None and using_generated_self:
        if distance_gen_self.shape[1] != distance_gen_self.shape[2]:
            raise ValueError("generated self-distance tensor must be square over particle axes")
        diag = torch.eye(
            distance_gen_self.shape[1],
            device=distance_gen_self.device,
            dtype=torch.bool,
        ).unsqueeze(0).unsqueeze(-1)
        distance_gen_self_for_mean = distance_gen_self.masked_fill(diag, float("inf"))
    else:
        distance_gen_self_for_mean = distance_gen_self

    self_repulsion_per_projection = _maybe_topk_pair_mean_sliced(
        distance_gen_self_for_mean,
        weight_gen,
        weight_self,
        topk=topk,
    )

    # Keep target-target global, as in the non-sliced top-k powered loss.
    target_repulsion_per_projection = _weighted_pair_mean_sliced(
        distance_pos_pos,
        torch.ones_like(weight_pos),
        torch.ones_like(weight_pos),
    )

    if neg_projected.shape[1] > 0:
        distance_gen_neg = _powered_distance_from_sq(
            (
                gen_projected[:, :, None, :]
                - neg_projected[:, None, :, :]
            ).pow(2),
            epsilon=epsilon,
            power=power,
        )

        fixed_negative_repulsion_per_projection = _maybe_topk_pair_mean_sliced(
            distance_gen_neg,
            weight_gen,
            weight_neg,
            topk=topk,
        )
    else:
        fixed_negative_repulsion_per_projection = torch.zeros_like(
            attraction_per_projection
        )

    attraction = attraction_per_projection.mean(dim=-1)
    self_repulsion = self_repulsion_per_projection.mean(dim=-1)
    target_repulsion = target_repulsion_per_projection.mean(dim=-1)
    fixed_negative_repulsion = fixed_negative_repulsion_per_projection.mean(dim=-1)

    return (
        attraction,
        self_repulsion,
        target_repulsion,
        fixed_negative_repulsion,
    )


def riesz_loss(
    gen: torch.Tensor,
    fixed_pos: torch.Tensor,
    fixed_neg: torch.Tensor | None = None,
    weight_gen: torch.Tensor | None = None,
    weight_pos: torch.Tensor | None = None,
    weight_neg: torch.Tensor | None = None,
    self_support: torch.Tensor | None = None,
    epsilon: float = 1e-8,
    power: float = 1.0,
    topk: int | None = None,
    topk_schedule: Any | None = None,
    current_step: int | None = None,
    use_sliced: bool = False,
    num_projections: int = 64,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    if power <= 0:
        raise ValueError("power must be positive")

    if num_projections <= 0:
        raise ValueError("num_projections must be positive")

    topk = _resolve_topk(
        topk=topk,
        topk_schedule=topk_schedule,
        current_step=current_step,
    )

    if topk is not None and int(topk) <= 0:
        raise ValueError("topk must be positive when provided")

    if gen.ndim != 3:
        raise ValueError(
            "gen must have shape [B, particles, features], "
            f"but received {tuple(gen.shape)}"
        )

    if fixed_pos.ndim != 3:
        raise ValueError(
            "fixed_pos must have shape [B, particles, features], "
            f"but received {tuple(fixed_pos.shape)}"
        )

    if gen.shape[0] != fixed_pos.shape[0]:
        raise ValueError("gen and fixed_pos must have the same batch size")

    if gen.shape[-1] != fixed_pos.shape[-1]:
        raise ValueError(
            "gen and fixed_pos must have the same feature dimension"
        )

    if fixed_neg is None:
        fixed_neg = torch.zeros_like(gen[:, :0, :])

    if fixed_neg.ndim != 3:
        raise ValueError(
            "fixed_neg must have shape [B, particles, features], "
            f"but received {tuple(fixed_neg.shape)}"
        )

    if gen.shape[0] != fixed_neg.shape[0]:
        raise ValueError("gen and fixed_neg must have the same batch size")

    if gen.shape[-1] != fixed_neg.shape[-1]:
        raise ValueError(
            "gen and fixed_neg must have the same feature dimension"
        )

    if self_support is not None:
        if self_support.ndim != 3:
            raise ValueError(
                "self_support must have shape [B, particles, features], "
                f"but received {tuple(self_support.shape)}"
            )

        if gen.shape[0] != self_support.shape[0]:
            raise ValueError("gen and self_support must have the same batch size")

        if gen.shape[-1] != self_support.shape[-1]:
            raise ValueError(
                "gen and self_support must have the same feature dimension"
            )

    if weight_gen is None:
        weight_gen = torch.ones_like(gen[:, :, 0])

    if weight_pos is None:
        weight_pos = torch.ones_like(fixed_pos[:, :, 0])

    if weight_neg is None:
        weight_neg = torch.ones_like(fixed_neg[:, :, 0])

    gen = gen.float()

    fixed_pos = fixed_pos.detach().float()
    fixed_neg = fixed_neg.detach().float()

    if self_support is not None:
        self_support = self_support.detach().float()

    weight_gen = weight_gen.detach().float()
    weight_pos = weight_pos.detach().float()
    weight_neg = weight_neg.detach().float()

    with torch.no_grad():
        scale_targets = torch.cat(
            [gen.detach(), fixed_neg, fixed_pos],
            dim=1,
        )

        scale_weights = torch.cat(
            [weight_gen, weight_neg, weight_pos],
            dim=1,
        )

        scale_distance = torch.cdist(
            gen.detach(),
            scale_targets,
        )

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

    self_support_scaled = None
    if self_support is not None:
        self_support_scaled = self_support / scale_inputs

    if use_sliced:
        (
            attraction,
            self_repulsion,
            target_repulsion,
            fixed_negative_repulsion,
        ) = _sliced_powered_riesz_terms(
            gen_scaled=gen_scaled,
            pos_scaled=pos_scaled,
            neg_scaled=neg_scaled,
            weight_gen=weight_gen,
            weight_pos=weight_pos,
            weight_neg=weight_neg,
            num_projections=num_projections,
            epsilon=epsilon,
            power=power,
            topk=topk,
            self_support_scaled=self_support_scaled,
        )
    else:
        (
            attraction,
            self_repulsion,
            target_repulsion,
            fixed_negative_repulsion,
        ) = _full_powered_riesz_terms(
            gen_scaled=gen_scaled,
            pos_scaled=pos_scaled,
            neg_scaled=neg_scaled,
            weight_gen=weight_gen,
            weight_pos=weight_pos,
            weight_neg=weight_neg,
            epsilon=epsilon,
            power=power,
            topk=topk,
        )

    loss = (
        2.0 * attraction
        - self_repulsion
        - target_repulsion
        - 2.0 * fixed_negative_repulsion
    )

    info = {
        "scale": scale.detach(),
        "riesz_power": torch.as_tensor(power, device=loss.device),
        "riesz_topk": torch.as_tensor(-1 if topk is None else int(topk), device=loss.device),
        "riesz_current_step": torch.as_tensor(
            -1 if current_step is None else int(current_step),
            device=loss.device,
        ),
        "riesz_attraction": attraction.detach().mean(),
        "riesz_self_repulsion": self_repulsion.detach().mean(),
        "riesz_target_repulsion": target_repulsion.detach().mean(),
        "riesz_fixed_negative_repulsion": fixed_negative_repulsion.detach().mean(),
        "riesz_use_sliced": torch.tensor(
            float(use_sliced),
            device=loss.device,
        ),
        "riesz_num_projections": torch.tensor(
            float(num_projections if use_sliced else 0),
            device=loss.device,
        ),
        "riesz_use_self_support": torch.tensor(
            float(self_support is not None),
            device=loss.device,
        ),
    }

    return loss, info
