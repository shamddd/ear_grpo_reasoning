"""Thin model-facing trainers built on the tested GRPO objective."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from ear_grpo_reasoning.algorithms import (
    compute_ear_advantages,
    compute_group_advantages,
    grpo_loss,
)
from ear_grpo_reasoning.models.epistemic import EpistemicProbe


class GRPOTrainer:
    """Perform one update over a group of already-generated completions."""

    def __init__(
        self,
        policy: Any,
        ref_policy: Any,
        optimizer: torch.optim.Optimizer,
        *,
        clip_eps: float = 0.2,
        kl_beta: float = 0.04,
    ) -> None:
        if not 0 <= clip_eps < 1:
            raise ValueError("clip_eps must be in [0, 1)")
        if kl_beta < 0:
            raise ValueError("kl_beta must be non-negative")
        self.policy = policy
        self.ref_policy = ref_policy
        self.ref_policy.eval()
        for parameter in self.ref_policy.parameters():
            parameter.requires_grad_(False)
        self.optimizer = optimizer
        self.clip_eps = clip_eps
        self.kl_beta = kl_beta

    def _advantages(
        self, rewards: Tensor, input_ids: Tensor, prompt_lengths: Sequence[int]
    ) -> Tensor:
        del input_ids, prompt_lengths
        return compute_group_advantages(rewards)

    def _extra_metrics(self) -> dict[str, float]:
        return {}

    def train_step(
        self,
        input_ids_group: Tensor,
        prompt_lengths: Sequence[int],
        rewards_group: Tensor,
        *,
        old_token_log_probs: Tensor | None = None,
    ) -> dict[str, float]:
        """Update the policy once.

        For fresh on-policy rollouts, omitting ``old_token_log_probs`` snapshots the
        current values before the update. Multi-epoch reuse must pass the rollout-policy
        log probabilities captured before the first optimizer step.
        """

        if input_ids_group.shape[0] != rewards_group.shape[0]:
            raise ValueError("rollout and reward batch sizes differ")
        if len(prompt_lengths) != input_ids_group.shape[0]:
            raise ValueError("one prompt length is required per rollout")
        # Generation commonly returns inference tensors. Cloning outside inference
        # mode makes ordinary integer tensors safe for a grad-tracked policy forward.
        input_ids_group = input_ids_group.clone()
        self.policy.train()
        self.optimizer.zero_grad(set_to_none=True)
        advantages = self._advantages(rewards_group, input_ids_group, prompt_lengths)

        logits = self.policy(input_ids_group)
        current_log_probs, completion_mask = self.policy.completion_token_log_probs(
            input_ids_group,
            prompt_lengths,
            logits=logits,
        )
        if old_token_log_probs is None:
            old_token_log_probs = current_log_probs.detach()
        with torch.no_grad():
            reference_ids = input_ids_group.to(self.ref_policy.device)
            reference_logits = self.ref_policy(reference_ids)
            reference_log_probs, reference_mask = self.ref_policy.completion_token_log_probs(
                reference_ids,
                prompt_lengths,
                logits=reference_logits,
            )
            reference_log_probs = reference_log_probs.to(self.policy.device)
            reference_mask = reference_mask.to(self.policy.device)
        if not torch.equal(completion_mask, reference_mask):
            raise RuntimeError("policy and reference completion masks differ")

        output = grpo_loss(
            current_log_probs,
            old_token_log_probs.to(self.policy.device),
            reference_log_probs,
            advantages,
            completion_mask,
            clip_epsilon=self.clip_eps,
            kl_beta=self.kl_beta,
        )
        torch.autograd.backward(output.loss)
        self.optimizer.step()

        with torch.no_grad():
            token_probs = torch.softmax(logits[:, :-1], dim=-1)
            token_entropy = -(token_probs * torch.log_softmax(logits[:, :-1], dim=-1)).sum(dim=-1)
            mask_float = completion_mask.to(token_entropy.dtype)
            sequence_entropy = (token_entropy * mask_float).sum(dim=1) / mask_float.sum(dim=1)
            correct_count = int((rewards_group > 0).sum().item())
            group_size = int(rewards_group.numel())
            reward_std = float(rewards_group.std(unbiased=False))
            advantage_std = float(advantages.std(unbiased=False))

        metrics = {
            "loss": float(output.loss.detach()),
            "policy_loss": float(output.policy_loss.detach()),
            "kl_div": float(output.kl_penalty.detach()),
            "kl_penalty": float(output.kl_penalty.detach()),
            "mean_ratio": float(output.mean_ratio.detach()),
            "clip_fraction": float(output.clip_fraction.detach()),
            "valid_completion_tokens": float(output.valid_tokens),
            "mean_reward": float(rewards_group.mean()),
            "entropy": float(sequence_entropy.mean()),
            "positive_rollout_rate": correct_count / group_size,
            "frac_zero_correct": float(correct_count == 0),
            "frac_one_correct": float(correct_count == 1),
            "frac_multi_correct": float(correct_count > 1),
            "reward_std": reward_std,
            "advantage_std": advantage_std,
        }
        metrics.update(self._extra_metrics())
        return metrics


class EARGRPOTrainer(GRPOTrainer):
    """GRPO update with epistemic or negative-control advantage weights."""

    def __init__(
        self,
        policy: Any,
        ref_policy: Any,
        optimizer: torch.optim.Optimizer,
        epistemic_probe: EpistemicProbe,
        *,
        gamma_epistemic: float = 0.35,
        clip_eps: float = 0.2,
        kl_beta: float = 0.04,
        mode: str = "ear",
    ) -> None:
        super().__init__(
            policy,
            ref_policy,
            optimizer,
            clip_eps=clip_eps,
            kl_beta=kl_beta,
        )
        if gamma_epistemic < 0:
            raise ValueError("gamma_epistemic must be non-negative")
        if mode not in {"ear", "random_control", "permuted_control"}:
            raise ValueError(f"Unknown trainer mode: {mode}")
        self.epistemic_probe = epistemic_probe
        self.gamma_epistemic = gamma_epistemic
        self.mode = mode
        self._last_uncertainty: Tensor | None = None
        self._last_weights: Tensor | None = None

    def _advantages(
        self, rewards: Tensor, input_ids: Tensor, prompt_lengths: Sequence[int]
    ) -> Tensor:
        if self.mode == "random_control":
            uncertainty = torch.rand(rewards.shape, device=rewards.device)
        else:
            uncertainty = self.epistemic_probe.compute_epistemic_variance(
                self.policy,
                input_ids,
                prompt_lengths,
            ).to(rewards.device)
            if self.mode == "permuted_control":
                uncertainty = uncertainty[
                    torch.randperm(uncertainty.numel(), device=rewards.device)
                ]
        advantages, weights = compute_ear_advantages(
            rewards,
            uncertainty,
            gamma=self.gamma_epistemic,
        )
        self._last_uncertainty = uncertainty.detach()
        self._last_weights = weights.detach()
        return advantages

    def _extra_metrics(self) -> dict[str, float]:
        if self._last_uncertainty is None or self._last_weights is None:
            raise RuntimeError("EAR metrics requested before advantages were computed")
        return {
            "mean_epistemic_var": float(self._last_uncertainty.mean()),
            "mean_dampening": float(self._last_weights.mean()),
        }
