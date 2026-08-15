"""Small, explicit evaluation metrics."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import mean, stdev


@dataclass(frozen=True)
class BinarySummary:
    mean: float
    standard_deviation: float
    sample_count: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def aggregate_binary(values: list[float]) -> BinarySummary:
    if not values:
        raise ValueError("values must not be empty")
    if any(value not in {0.0, 1.0} for value in values):
        raise ValueError("binary values must be 0.0 or 1.0")
    return BinarySummary(
        mean=float(mean(values)),
        standard_deviation=float(stdev(values)) if len(values) > 1 else 0.0,
        sample_count=len(values),
    )


def compute_pass_at_k(results: list[list[bool]], *, k: int = 1) -> float:
    """Compute the unbiased pass@k estimator over prompt-level samples."""

    if not results:
        raise ValueError("results must not be empty")
    if k < 1:
        raise ValueError("k must be positive")
    estimates: list[float] = []
    for prompt_results in results:
        n = len(prompt_results)
        if n < k:
            raise ValueError(f"each prompt needs at least k={k} samples")
        correct = sum(prompt_results)
        if correct == 0:
            estimates.append(0.0)
        elif n - correct < k:
            estimates.append(1.0)
        else:
            estimates.append(1.0 - math.comb(n - correct, k) / math.comb(n, k))
    return float(mean(estimates))
