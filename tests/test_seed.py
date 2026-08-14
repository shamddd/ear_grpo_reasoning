import random

import numpy as np
import torch

from ear_grpo_reasoning.seed import seed_everything


def _draw() -> tuple[float, float, float]:
    return random.random(), float(np.random.random()), float(torch.rand(1))


def test_seed_reproducibility() -> None:
    seed_everything(17)
    first = _draw()
    seed_everything(17)
    second = _draw()
    assert first == second
