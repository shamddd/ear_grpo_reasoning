"""Scientifically testable GRPO and EAR primitives."""

from ear_grpo_reasoning.algorithms.advantages import (
    compute_ear_advantages,
    compute_group_advantages,
)
from ear_grpo_reasoning.algorithms.objective import GRPOLossOutput, grpo_loss

__all__ = [
    "GRPOLossOutput",
    "compute_ear_advantages",
    "compute_group_advantages",
    "grpo_loss",
]
