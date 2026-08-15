"""Deprecated compatibility imports; use :mod:`ear_grpo_reasoning.models`."""

from src.models.epistemic_probe import EpistemicUncertaintyProbe
from src.models.policy import TransformerReasoningPolicy

__all__ = ["EpistemicUncertaintyProbe", "TransformerReasoningPolicy"]
