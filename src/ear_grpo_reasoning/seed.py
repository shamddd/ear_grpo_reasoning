"""Central deterministic seeding utilities."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class SeedState:
    seed: int
    deterministic_requested: bool
    cuda_available: bool


def seed_everything(seed: int, *, deterministic: bool = True) -> SeedState:
    """Seed supported RNGs and request deterministic PyTorch kernels.

    PyTorch may still reject an operation that has no deterministic implementation.
    That failure is intentional: verification should not silently fall back to a
    nondeterministic kernel.
    """

    if seed < 0:
        raise ValueError("seed must be non-negative")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=False)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic
    return SeedState(seed, deterministic, torch.cuda.is_available())
