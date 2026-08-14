"""Masked token-level GRPO clipped surrogate objective."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class GRPOLossOutput:
    loss: Tensor
    policy_loss: Tensor
    kl_penalty: Tensor
    mean_ratio: Tensor
    clip_fraction: Tensor
    valid_tokens: int


def _validate_inputs(
    current_log_probs: Tensor,
    old_log_probs: Tensor,
    reference_log_probs: Tensor,
    advantages: Tensor,
    completion_mask: Tensor,
) -> Tensor:
    expected_shape = current_log_probs.shape
    if current_log_probs.ndim != 2:
        raise ValueError("log-probability tensors must have shape [batch, completion_tokens]")
    if old_log_probs.shape != expected_shape or reference_log_probs.shape != expected_shape:
        raise ValueError("current, old, and reference log probabilities must have equal shapes")
    if completion_mask.shape != expected_shape:
        raise ValueError("completion_mask must match the log-probability shape")
    if advantages.shape != (expected_shape[0],):
        raise ValueError("advantages must have shape [batch]")
    tensors = (current_log_probs, old_log_probs, reference_log_probs, advantages)
    if not all(tensor.is_floating_point() for tensor in tensors):
        raise TypeError("log probabilities and advantages must be floating point")
    if not all(torch.isfinite(tensor).all() for tensor in tensors):
        raise ValueError("objective inputs must contain only finite values")
    if (
        old_log_probs.dtype != current_log_probs.dtype
        or reference_log_probs.dtype != current_log_probs.dtype
    ):
        raise TypeError("current, old, and reference log probabilities must use the same dtype")
    if any(tensor.device != current_log_probs.device for tensor in tensors[1:]):
        raise ValueError("all objective tensors must use the same device")
    if completion_mask.device != current_log_probs.device:
        raise ValueError("completion_mask must use the same device as log probabilities")
    mask = completion_mask.to(dtype=torch.bool)
    counts = mask.sum(dim=1)
    if (counts == 0).any():
        raise ValueError("each sequence must contain at least one completion token")
    return mask


def grpo_loss(
    current_log_probs: Tensor,
    old_log_probs: Tensor,
    reference_log_probs: Tensor,
    advantages: Tensor,
    completion_mask: Tensor,
    *,
    clip_epsilon: float = 0.2,
    kl_beta: float = 0.04,
) -> GRPOLossOutput:
    r"""Compute sequence-balanced clipped GRPO loss.

    ``old_log_probs`` must come from the rollout policy before optimization. Padding,
    prompt tokens, and tokens after EOS must be zero in ``completion_mask``. The KL
    term uses the non-negative sampled estimator ``exp(ref-current) - (ref-current) - 1``.
    """

    if (
        isinstance(clip_epsilon, bool)
        or not math.isfinite(clip_epsilon)
        or not 0 <= clip_epsilon < 1
    ):
        raise ValueError("clip_epsilon must be finite and in [0, 1)")
    if isinstance(kl_beta, bool) or not math.isfinite(kl_beta) or kl_beta < 0:
        raise ValueError("kl_beta must be finite and non-negative")
    mask = _validate_inputs(
        current_log_probs,
        old_log_probs,
        reference_log_probs,
        advantages,
        completion_mask,
    )
    mask_float = mask.to(dtype=current_log_probs.dtype)
    token_counts = mask_float.sum(dim=1)

    # Rollout and reference tensors are fixed targets even if a caller accidentally
    # supplies tensors carrying an autograd history.
    frozen_old_log_probs = old_log_probs.detach()
    frozen_reference_log_probs = reference_log_probs.detach()
    log_ratio = torch.where(
        mask,
        current_log_probs - frozen_old_log_probs,
        torch.zeros_like(current_log_probs),
    )
    ratio = torch.exp(log_ratio)
    if not torch.isfinite(ratio).all():
        raise ValueError("policy ratio overflowed; log-probability differences are too large")
    expanded_advantages = advantages.unsqueeze(1)
    unclipped = ratio * expanded_advantages
    clipped_ratio = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon)
    clipped = clipped_ratio * expanded_advantages
    token_surrogate = torch.minimum(unclipped, clipped)
    sequence_surrogate = (token_surrogate * mask_float).sum(dim=1) / token_counts
    policy_loss = -sequence_surrogate.mean()

    ref_minus_current = torch.where(
        mask,
        frozen_reference_log_probs - current_log_probs,
        torch.zeros_like(current_log_probs),
    )
    token_kl = torch.exp(ref_minus_current) - ref_minus_current - 1.0
    if not torch.isfinite(token_kl).all():
        raise ValueError("sampled KL penalty overflowed")
    sequence_kl = (token_kl * mask_float).sum(dim=1) / token_counts
    kl_penalty = sequence_kl.mean()
    loss = policy_loss + kl_beta * kl_penalty

    masked_ratio = (ratio * mask_float).sum() / mask_float.sum()
    clipped_tokens = ((ratio - 1.0).abs() > clip_epsilon) & mask
    clip_fraction = clipped_tokens.to(current_log_probs.dtype).sum() / mask_float.sum()
    return GRPOLossOutput(
        loss=loss,
        policy_loss=policy_loss,
        kl_penalty=kl_penalty,
        mean_ratio=masked_ratio,
        clip_fraction=clip_fraction,
        valid_tokens=int(mask.sum().item()),
    )
