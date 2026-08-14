"""Monte Carlo dropout probe with degeneracy detection."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from ear_grpo_reasoning.models.policy import TransformerReasoningPolicy


class DegenerateUncertaintyError(RuntimeError):
    """Raised when MC dropout is requested for a model with no active dropout."""


class EpistemicProbe:
    def __init__(self, num_mc_samples: int = 3, *, on_degenerate: str = "raise") -> None:
        if num_mc_samples < 2:
            raise ValueError("num_mc_samples must be at least 2")
        if on_degenerate not in {"raise", "zeros"}:
            raise ValueError("on_degenerate must be raise or zeros")
        self.num_mc_samples = num_mc_samples
        self.on_degenerate = on_degenerate

    def compute_epistemic_variance(
        self,
        policy: TransformerReasoningPolicy,
        input_ids: Tensor,
        prompt_lengths: Sequence[int],
    ) -> Tensor:
        """Return variance of sequence-mean completion log probability."""

        if not policy.active_dropout_modules():
            if self.on_degenerate == "raise":
                raise DegenerateUncertaintyError(
                    "MC dropout is undefined for a model with no active dropout modules"
                )
            return torch.zeros(input_ids.shape[0], device=input_ids.device)
        samples: list[Tensor] = []
        with torch.no_grad():
            for _ in range(self.num_mc_samples):
                logits = policy(input_ids, mc_dropout=True)
                samples.append(
                    policy.compute_completion_log_probs(
                        input_ids,
                        prompt_lengths,
                        logits=logits,
                    )
                )
        return torch.stack(samples, dim=0).var(dim=0, unbiased=False)
