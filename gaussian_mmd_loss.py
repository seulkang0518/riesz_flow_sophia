"""Weighted Gaussian-MMD particle loss with an optional lengthscale schedule.

This module is intended as a drop-in particle-field alternative to the Riesz
losses used by W-Flow.  It keeps the all-pairs computation but removes hard
nearest-neighbour selection/sorting: locality is controlled softly by the
Gaussian kernel lengthscale.

The optimized quantity is MMD^2 up to terms that depend only on the fixed
positive/negative supports.  Those fixed-only terms have zero generator
gradient, so omitting them saves pairwise work without changing training.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch


def _weighted_pair_mean(
    values: torch.Tensor,
    left_weight: torch.Tensor,
    right_weight: torch.Tensor,
) -> torch.Tensor:
    """Per-batch weighted pairwise empirical mean."""
    pair_weight = left_weight[:, :, None] * right_weight[:, None, :]
    return (values * pair_weight).mean(dim=(-1, -2))


def _resolve_lengthscale(
    *,
    lengthscale: float,
    lengthscale_init: float | None,
    lengthscale_final: float | None,
    lengthscale_decay: float | None,
    lengthscale_decay_steps: int | None,
    current_step: int | None,
) -> float:
    """Resolve a fixed or monotonically decreasing Gaussian lengthscale.

    Supported scheduled forms
    -------------------------
    1. Paper-style exponential decay:
         ell_t = max(ell_final, ell_init * decay**t)
       by setting ``lengthscale_init``, ``lengthscale_final`` and
       ``lengthscale_decay``.

    2. Geometric interpolation over a chosen number of steps:
         ell_t = ell_init * (ell_final / ell_init)**min(t/T, 1)
       by setting ``lengthscale_decay_steps`` instead of ``lengthscale_decay``.

    If ``lengthscale_init``/``lengthscale_final`` are omitted, ``lengthscale``
    is used as a fixed value.
    """
    if lengthscale_init is None and lengthscale_final is None:
        ell = float(lengthscale)
        if ell <= 0:
            raise ValueError("lengthscale must be positive")
        return ell

    if lengthscale_init is None or lengthscale_final is None:
        raise ValueError(
            "lengthscale_init and lengthscale_final must either both be set or both be omitted"
        )
    if current_step is None:
        raise ValueError("current_step is required when using a lengthscale schedule")

    ell0 = float(lengthscale_init)
    ell_inf = float(lengthscale_final)
    if ell0 <= 0 or ell_inf <= 0:
        raise ValueError("scheduled lengthscales must be positive")
    if ell0 < ell_inf:
        raise ValueError(
            f"Expected a decreasing schedule with init >= final, got {ell0} < {ell_inf}"
        )

    step = max(0, int(current_step))

    if lengthscale_decay is not None and lengthscale_decay_steps is not None:
        raise ValueError("Set only one of lengthscale_decay or lengthscale_decay_steps")

    if lengthscale_decay is not None:
        decay = float(lengthscale_decay)
        if not (0.0 < decay <= 1.0):
            raise ValueError("lengthscale_decay must lie in (0, 1]")
        return max(ell_inf, ell0 * (decay ** step))

    if lengthscale_decay_steps is None:
        raise ValueError(
            "A scheduled lengthscale needs either lengthscale_decay or lengthscale_decay_steps"
        )

    T = int(lengthscale_decay_steps)
    if T <= 0:
        raise ValueError("lengthscale_decay_steps must be positive")
    frac = min(float(step) / float(T), 1.0)
    # Geometric interpolation is the finite-horizon analogue of exponential decay.
    return ell0 * ((ell_inf / ell0) ** frac)


def _gaussian_kernel_from_distance(
    distance: torch.Tensor,
    *,
    scale: torch.Tensor,
    lengthscale: float,
) -> torch.Tensor:
    """Gaussian kernel on distance normalized by the per-feature drift scale."""
    normalized_distance = distance / scale.clamp_min(1e-12)
    z = normalized_distance / float(lengthscale)
    return torch.exp(-0.5 * z.square())


@torch.no_grad()
def _positive_support_metrics(
    kernel_gen_pos: torch.Tensor,
    tiny: float = 1e-30,
) -> Dict[str, torch.Tensor]:
    """Effective-neighbour diagnostics for the positive Gaussian affinities."""
    row_mass = kernel_gen_pos.sum(dim=-1, keepdim=True)
    zero_rows = row_mass <= tiny
    p = kernel_gen_pos / row_mass.clamp_min(tiny)

    eff = 1.0 / p.square().sum(dim=-1).clamp_min(tiny)
    entropy = -(p.clamp_min(tiny) * p.clamp_min(tiny).log()).sum(dim=-1)
    eff_entropy = entropy.exp()

    info: Dict[str, torch.Tensor] = {
        "gaussian_eff_pos_mean": eff.mean(),
        "gaussian_eff_pos_median": eff.median(),
        "gaussian_eff_pos_min": eff.min(),
        "gaussian_eff_pos_max": eff.max(),
        "gaussian_eff_entropy_pos_mean": eff_entropy.mean(),
        "gaussian_eff_entropy_pos_median": eff_entropy.median(),
        "gaussian_frac_zero_pos_rows": zero_rows.float().mean(),
        "gaussian_pos_kernel_mean": kernel_gen_pos.mean(),
        "gaussian_pos_kernel_median": kernel_gen_pos.median(),
        "gaussian_pos_kernel_max": kernel_gen_pos.max(),
    }

    for topk in (1, 2, 4, 8, 16, 32):
        kk = min(topk, p.shape[-1])
        mass = torch.topk(p, k=kk, dim=-1, largest=True).values.sum(dim=-1)
        info[f"gaussian_top{topk}_mass_pos_mean"] = mass.mean()
        info[f"gaussian_top{topk}_mass_pos_median"] = mass.median()

    for cutoff in (2, 4, 8, 16, 32):
        info[f"gaussian_frac_eff_pos_le_{cutoff}"] = (eff <= cutoff).float().mean()

    return info


def gaussian_mmd_loss(
    gen: torch.Tensor,
    fixed_pos: torch.Tensor,
    fixed_neg: torch.Tensor | None = None,
    weight_gen: torch.Tensor | None = None,
    weight_pos: torch.Tensor | None = None,
    weight_neg: torch.Tensor | None = None,
    epsilon: float = 1e-8,
    lengthscale: float = 0.2,
    lengthscale_init: float | None = None,
    lengthscale_final: float | None = None,
    lengthscale_decay: float | None = None,
    lengthscale_decay_steps: int | None = None,
    current_step: int | None = None,
    log_effective_neighbors: bool = True,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute a CFG-weighted Gaussian-MMD generator loss.

    Parameters
    ----------
    gen, fixed_pos, fixed_neg:
        Tensors with shape ``[B, particles, features]``.
    weight_gen:
        Generated-particle weights, normally all ones.
    weight_pos:
        Positive weights.  In the current W-Flow CFG branch these are ``cfg``.
    weight_neg:
        Negative/unconditional weights.  In the current W-Flow CFG branch
        these are ``cfg - 1``.

    Notes
    -----
    The optimized generator-dependent objective is

        E[k(G,G')] - 2 E[w_pos k(G,P)] + 2 E[w_neg k(G,N)].

    This differs from the full squared RKHS norm only by terms involving
    fixed supports alone, so it has exactly the same generator gradient while
    avoiding unnecessary P-P / P-N / N-N pairwise computations.

    Distances are divided by the same characteristic scale used by drift/Riesz
    preprocessing before the Gaussian kernel is applied.  Therefore the
    lengthscale is dimensionless and comparable across feature branches:
    ``ell ~ 1`` is broad, while smaller ``ell`` is increasingly local.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if gen.ndim != 3 or fixed_pos.ndim != 3:
        raise ValueError("gen and fixed_pos must have shape [B, particles, features]")
    if gen.shape[0] != fixed_pos.shape[0] or gen.shape[-1] != fixed_pos.shape[-1]:
        raise ValueError("gen and fixed_pos must share batch size and feature dimension")

    if fixed_neg is None:
        fixed_neg = torch.zeros_like(gen[:, :0, :])
    if fixed_neg.ndim != 3:
        raise ValueError("fixed_neg must have shape [B, particles, features]")
    if gen.shape[0] != fixed_neg.shape[0] or gen.shape[-1] != fixed_neg.shape[-1]:
        raise ValueError("gen and fixed_neg must share batch size and feature dimension")

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

    ell = _resolve_lengthscale(
        lengthscale=lengthscale,
        lengthscale_init=lengthscale_init,
        lengthscale_final=lengthscale_final,
        lengthscale_decay=lengthscale_decay,
        lengthscale_decay_steps=lengthscale_decay_steps,
        current_step=current_step,
    )

    # Compute each required pairwise distance exactly once.  The detached
    # versions are also sufficient to reproduce the drift/Riesz scale.
    distance_gen_gen = torch.cdist(gen, gen)
    distance_gen_pos = torch.cdist(gen, fixed_pos)
    if fixed_neg.shape[1] > 0:
        distance_gen_neg = torch.cdist(gen, fixed_neg)
    else:
        distance_gen_neg = gen.new_zeros((gen.shape[0], gen.shape[1], 0))

    with torch.no_grad():
        scale_distances = torch.cat(
            [distance_gen_gen.detach(), distance_gen_neg.detach(), distance_gen_pos.detach()],
            dim=-1,
        )
        scale_weights = torch.cat([weight_gen, weight_neg, weight_pos], dim=1)
        scale = (
            (scale_distances * scale_weights[:, None, :]).mean()
            / (scale_weights.mean() + float(epsilon))
        )
        scale = scale.clamp_min(1e-3)

    kernel_gen_gen = _gaussian_kernel_from_distance(
        distance_gen_gen, scale=scale, lengthscale=ell
    )
    kernel_gen_pos = _gaussian_kernel_from_distance(
        distance_gen_pos, scale=scale, lengthscale=ell
    )

    self_similarity = _weighted_pair_mean(
        kernel_gen_gen, weight_gen, weight_gen
    )
    positive_similarity = _weighted_pair_mean(
        kernel_gen_pos, weight_gen, weight_pos
    )

    if fixed_neg.shape[1] > 0:
        kernel_gen_neg = _gaussian_kernel_from_distance(
            distance_gen_neg, scale=scale, lengthscale=ell
        )
        negative_similarity = _weighted_pair_mean(
            kernel_gen_neg, weight_gen, weight_neg
        )
    else:
        negative_similarity = torch.zeros_like(self_similarity)

    loss = self_similarity - 2.0 * positive_similarity + 2.0 * negative_similarity

    info: Dict[str, torch.Tensor] = {
        "scale": scale.detach(),
        "gaussian_lengthscale": torch.tensor(ell, device=gen.device),
        "gaussian_self_similarity": self_similarity.detach().mean(),
        "gaussian_positive_similarity": positive_similarity.detach().mean(),
        "gaussian_negative_similarity": negative_similarity.detach().mean(),
    }

    if log_effective_neighbors:
        info.update(_positive_support_metrics(kernel_gen_pos.detach()))

    return loss, info


# Match the naming convention of riesz_loss.py for easy swapping/imports.
riesz_loss = gaussian_mmd_loss
