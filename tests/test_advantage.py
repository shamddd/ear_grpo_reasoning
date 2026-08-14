import pytest
import torch

from ear_grpo_reasoning.algorithms import compute_ear_advantages, compute_group_advantages


def test_compute_group_advantages() -> None:
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
    advantages = compute_group_advantages(rewards)

    assert advantages.shape == (4,)
    assert torch.mean(advantages).abs() < 1e-6
    assert advantages[0] > 0
    assert advantages[1] < 0


def test_identical_rewards_have_zero_advantages() -> None:
    actual = compute_group_advantages(torch.tensor([1.0, 1.0, 1.0]))
    assert torch.equal(actual, torch.zeros(3))


def test_singleton_group_has_zero_advantage() -> None:
    assert torch.equal(compute_group_advantages(torch.tensor([1.0])), torch.zeros(1))


def test_nonfinite_rewards_are_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        compute_group_advantages(torch.tensor([1.0, float("nan")]))


def test_compute_ear_advantages() -> None:
    rewards = torch.tensor([1.0, 1.0, 0.0, 0.0])
    # Trajectory 0 has low uncertainty (0.01), Trajectory 1 has high uncertainty (0.95)
    epistemic_vars = torch.tensor([0.01, 0.95, 0.05, 0.10])

    ear_adv, dampening = compute_ear_advantages(rewards, epistemic_vars, gamma=0.5)

    assert ear_adv.shape == (4,)
    assert dampening.shape == (4,)
    # High uncertainty trajectory 1 receives less advantage than low-uncertainty trajectory 0.
    assert ear_adv[0] > ear_adv[1]
    assert dampening[0] > dampening[1]


def test_constant_uncertainty_has_neutral_weights() -> None:
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
    ear, weights = compute_ear_advantages(rewards, torch.full((4,), 0.25))
    assert torch.equal(weights, torch.ones(4))
    assert torch.equal(ear, compute_group_advantages(rewards))


def test_negative_uncertainty_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        compute_ear_advantages(torch.tensor([1.0, 0.0]), torch.tensor([0.1, -0.1]))
