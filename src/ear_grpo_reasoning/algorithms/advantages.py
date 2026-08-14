"""Group-relative and epistemically weighted advantage estimators."""

from __future__ import annotations

import torch
from torch import Tensor


def _validate_vector(name: str, value: Tensor) -> None:
    if value.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {tuple(value.shape)}")
    if value.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    if not value.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")


def compute_group_advantages(rewards: Tensor, *, eps: float = 1e-8) -> Tensor:
    """Return population-standardized rewards for one rollout group.

    A singleton or zero-variance group contains no relative information and returns
    exact zeros. This avoids amplifying numerical noise through division by epsilon.
    """

    _validate_vector("rewards", rewards)
    if eps <= 0:
        raise ValueError("eps must be positive")
    centered = rewards - rewards.mean()
    std = rewards.std(unbiased=False)
    if rewards.numel() == 1 or std <= eps:
        return torch.zeros_like(rewards)
    return centered / (std + eps)


def compute_ear_advantages(
    rewards: Tensor,
    epistemic_uncertainty: Tensor,
    *,
    gamma: float = 0.35,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor]:
    r"""Apply the repository's EAR exponential weighting to GRPO advantages.

    The submitted formulation uses ``exp(-gamma * U_i / std(U))``. When the
    uncertainty vector is constant, it has no trajectory-ranking information; the
    implementation therefore returns neutral weights instead of dividing by epsilon.
    This guard is documented as a numerical/degeneracy clarification, not a new claim.
    """

    _validate_vector("rewards", rewards)
    _validate_vector("epistemic_uncertainty", epistemic_uncertainty)
    if rewards.shape != epistemic_uncertainty.shape:
        raise ValueError("rewards and epistemic_uncertainty must have the same shape")
    if (epistemic_uncertainty < 0).any():
        raise ValueError("epistemic_uncertainty must be non-negative")
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    if eps <= 0:
        raise ValueError("eps must be positive")

    advantages = compute_group_advantages(rewards, eps=eps)
    uncertainty_std = epistemic_uncertainty.std(unbiased=False)
    if uncertainty_std <= eps or gamma == 0:
        weights = torch.ones_like(epistemic_uncertainty)
    else:
        exponent = -gamma * epistemic_uncertainty / (uncertainty_std + eps)
        weights = torch.exp(exponent)
    return advantages * weights, weights
