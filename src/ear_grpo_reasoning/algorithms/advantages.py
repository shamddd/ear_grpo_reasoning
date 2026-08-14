"""Group-relative and epistemically weighted advantage estimators."""

from __future__ import annotations

import math

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


def _validate_positive_scalar(name: str, value: float) -> None:
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _calculation_dtype(value: Tensor) -> torch.dtype:
    """Use at least float32 so small eps values remain representable."""

    return torch.float64 if value.dtype == torch.float64 else torch.float32


def compute_group_advantages(rewards: Tensor, *, eps: float = 1e-8) -> Tensor:
    """Return population-standardized rewards for one rollout group.

    A singleton or zero-variance group contains no relative information and returns
    exact zeros. This avoids amplifying numerical noise through division by epsilon.
    """

    _validate_vector("rewards", rewards)
    _validate_positive_scalar("eps", eps)
    work = rewards.to(dtype=_calculation_dtype(rewards))
    centered = work - work.mean()
    std = work.std(unbiased=False)
    if rewards.numel() == 1 or std <= eps:
        return torch.zeros_like(rewards)
    return (centered / (std + eps)).to(dtype=rewards.dtype)


def compute_ear_advantages(
    rewards: Tensor,
    epistemic_uncertainty: Tensor,
    *,
    gamma: float = 0.35,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor]:
    r"""Apply the repository's EAR exponential weighting to GRPO advantages.

    The submitted formulation uses ``exp(-gamma * U_i / (std(U) + eps))``. It is
    evaluated literally, including for constant nonzero uncertainty. Computation uses
    at least float32 so ``eps`` remains representable for lower-precision input tensors.
    """

    _validate_vector("rewards", rewards)
    _validate_vector("epistemic_uncertainty", epistemic_uncertainty)
    if rewards.shape != epistemic_uncertainty.shape:
        raise ValueError("rewards and epistemic_uncertainty must have the same shape")
    if rewards.device != epistemic_uncertainty.device:
        raise ValueError("rewards and epistemic_uncertainty must use the same device")
    if rewards.dtype != epistemic_uncertainty.dtype:
        raise TypeError("rewards and epistemic_uncertainty must use the same dtype")
    if (epistemic_uncertainty < 0).any():
        raise ValueError("epistemic_uncertainty must be non-negative")
    if isinstance(gamma, bool) or not math.isfinite(gamma) or gamma < 0:
        raise ValueError("gamma must be finite and non-negative")
    _validate_positive_scalar("eps", eps)

    advantages = compute_group_advantages(rewards, eps=eps)
    work = epistemic_uncertainty.to(dtype=_calculation_dtype(epistemic_uncertainty))
    uncertainty_std = work.std(unbiased=False)
    exponent = -gamma * work / (uncertainty_std + eps)
    weights = torch.exp(exponent).to(dtype=epistemic_uncertainty.dtype)
    return advantages * weights, weights
