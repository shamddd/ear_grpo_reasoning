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
    assert torch.allclose(advantages, torch.tensor([1.0, -1.0, 1.0, -1.0]))


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


def test_constant_uncertainty_follows_documented_equation() -> None:
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
    uncertainty = torch.full((4,), 0.25)
    ear, weights = compute_ear_advantages(rewards, uncertainty, gamma=0.35, eps=0.5)
    expected_weights = torch.exp(-0.35 * uncertainty / 0.5)
    assert torch.allclose(weights, expected_weights)
    assert torch.allclose(ear, compute_group_advantages(rewards, eps=0.5) * expected_weights)


def test_zero_uncertainty_and_zero_gamma_are_neutral() -> None:
    rewards = torch.tensor([1.0, 0.0])
    ear_zero_u, weights_zero_u = compute_ear_advantages(rewards, torch.zeros(2))
    ear_zero_gamma, weights_zero_gamma = compute_ear_advantages(
        rewards, torch.tensor([0.1, 0.9]), gamma=0.0
    )
    expected = compute_group_advantages(rewards)
    assert torch.equal(weights_zero_u, torch.ones(2))
    assert torch.equal(weights_zero_gamma, torch.ones(2))
    assert torch.equal(ear_zero_u, expected)
    assert torch.equal(ear_zero_gamma, expected)


def test_negative_uncertainty_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        compute_ear_advantages(torch.tensor([1.0, 0.0]), torch.tensor([0.1, -0.1]))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_invalid_gamma_is_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="gamma"):
        compute_ear_advantages(torch.tensor([1.0, 0.0]), torch.tensor([0.1, 0.2]), gamma=value)


@pytest.mark.parametrize("value", [0.0, float("nan"), float("inf")])
def test_invalid_epsilon_is_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="eps"):
        compute_group_advantages(torch.tensor([1.0, 0.0]), eps=value)


def test_batched_groups_are_rejected_instead_of_broadcast() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        compute_ear_advantages(torch.ones((2, 2)), torch.ones((2, 2)))


def test_ear_preserves_input_dtype() -> None:
    rewards = torch.tensor([1.0, 0.0], dtype=torch.float64)
    uncertainty = torch.tensor([0.0, 0.2], dtype=torch.float64)
    ear, weights = compute_ear_advantages(rewards, uncertainty)
    assert ear.dtype == torch.float64
    assert weights.dtype == torch.float64


def test_ear_rejects_mixed_dtypes() -> None:
    with pytest.raises(TypeError, match="same dtype"):
        compute_ear_advantages(
            torch.tensor([1.0, 0.0], dtype=torch.float32),
            torch.tensor([0.1, 0.2], dtype=torch.float64),
        )
