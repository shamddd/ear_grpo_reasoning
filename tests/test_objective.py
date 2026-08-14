import pytest
import torch

from ear_grpo_reasoning.algorithms import grpo_loss


def _inputs() -> tuple[torch.Tensor, ...]:
    current = torch.tensor([[-0.8, -1.0, -2.0], [-1.2, -0.9, -3.0]], requires_grad=True)
    old = torch.tensor([[-1.0, -1.0, -2.0], [-1.0, -1.0, -3.0]])
    reference = torch.tensor([[-1.1, -1.1, -2.0], [-1.1, -1.1, -3.0]])
    advantages = torch.tensor([1.0, -1.0])
    mask = torch.tensor([[True, True, False], [True, True, False]])
    return current, old, reference, advantages, mask


def test_grpo_loss_is_finite_and_differentiable() -> None:
    current, old, reference, advantages, mask = _inputs()
    output = grpo_loss(current, old, reference, advantages, mask)
    output.loss.backward()
    assert torch.isfinite(output.loss)
    assert current.grad is not None
    assert current.grad[mask].abs().sum() > 0
    assert current.grad[~mask].abs().sum() == 0
    assert output.valid_tokens == 4
    assert output.kl_penalty >= 0


def test_sequence_balancing_is_invariant_to_padding() -> None:
    current, old, reference, advantages, mask = _inputs()
    base = grpo_loss(current, old, reference, advantages, mask)
    padding = torch.zeros((2, 2))
    padded = grpo_loss(
        torch.cat([current, padding], dim=1),
        torch.cat([old, padding], dim=1),
        torch.cat([reference, padding], dim=1),
        advantages,
        torch.cat([mask, torch.zeros((2, 2), dtype=torch.bool)], dim=1),
    )
    assert torch.allclose(base.loss, padded.loss)


def test_sequence_balancing_prevents_long_completion_dominance() -> None:
    current = torch.tensor([[torch.log(torch.tensor(2.0)), 0.0, 0.0], [0.0, 0.0, 0.0]])
    old = torch.zeros_like(current)
    reference = current.clone()
    mask = torch.tensor([[True, False, False], [True, True, True]])
    output = grpo_loss(
        current,
        old,
        reference,
        torch.tensor([1.0, -1.0]),
        mask,
        clip_epsilon=0.2,
        kl_beta=0.0,
    )
    # Per-sequence surrogates are 1.2 and -1.0, so their equal-weight mean is 0.1.
    assert float(output.policy_loss.detach()) == pytest.approx(-0.1)


def test_clipping_uses_frozen_old_log_probs() -> None:
    current, old, reference, advantages, mask = _inputs()
    current = current.detach().clone().requires_grad_(True)
    current.data[0, 0] = 0.0
    output = grpo_loss(current, old, reference, advantages, mask, clip_epsilon=0.2)
    assert output.clip_fraction > 0


@pytest.mark.parametrize(
    ("ratio", "advantage", "expected_surrogate"),
    [
        (2.0, 1.0, 1.2),
        (0.5, 1.0, 0.5),
        (2.0, -1.0, -2.0),
        (0.5, -1.0, -0.8),
    ],
)
def test_clipping_respects_advantage_sign(
    ratio: float, advantage: float, expected_surrogate: float
) -> None:
    current = torch.tensor([[ratio]]).log().requires_grad_(True)
    old = torch.zeros((1, 1))
    reference = current.detach().clone()
    output = grpo_loss(
        current,
        old,
        reference,
        torch.tensor([advantage]),
        torch.ones((1, 1), dtype=torch.bool),
        clip_epsilon=0.2,
        kl_beta=0.0,
    )
    assert float(output.policy_loss.detach()) == pytest.approx(-expected_surrogate)


def test_old_and_reference_log_probs_are_frozen_targets() -> None:
    current, old, reference, advantages, mask = _inputs()
    old.requires_grad_(True)
    reference.requires_grad_(True)
    output = grpo_loss(current, old, reference, advantages, mask)
    output.loss.backward()
    assert current.grad is not None
    assert old.grad is None
    assert reference.grad is None


def test_sampled_kl_matches_documented_integrand() -> None:
    current = torch.tensor([[torch.log(torch.tensor(0.5))]], requires_grad=True)
    old = current.detach().clone()
    reference = torch.tensor([[torch.log(torch.tensor(0.25))]])
    output = grpo_loss(
        current,
        old,
        reference,
        torch.ones(1),
        torch.ones((1, 1), dtype=torch.bool),
        kl_beta=1.0,
    )
    x = reference - current.detach()
    expected = torch.exp(x) - x - 1.0
    assert torch.allclose(output.kl_penalty, expected.squeeze())
    assert output.kl_penalty > 0


def test_equal_policy_and_reference_have_zero_sampled_kl() -> None:
    current, old, _, advantages, mask = _inputs()
    output = grpo_loss(current, old, current.detach().clone(), advantages, mask)
    assert float(output.kl_penalty.detach()) == pytest.approx(0.0)


def test_empty_completion_is_rejected() -> None:
    current, old, reference, advantages, mask = _inputs()
    mask[1] = False
    with pytest.raises(ValueError, match="at least one"):
        grpo_loss(current, old, reference, advantages, mask)


def test_nonfinite_log_probs_are_rejected() -> None:
    current, old, reference, advantages, mask = _inputs()
    current.data[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        grpo_loss(current, old, reference, advantages, mask)


def test_ratio_overflow_is_rejected() -> None:
    with pytest.raises(ValueError, match="ratio overflowed"):
        grpo_loss(
            torch.tensor([[1000.0]], requires_grad=True),
            torch.tensor([[-1000.0]]),
            torch.tensor([[0.0]]),
            torch.ones(1),
            torch.ones((1, 1), dtype=torch.bool),
        )


def test_masked_extreme_log_probs_do_not_overflow_or_affect_loss() -> None:
    current = torch.tensor([[0.0, 1000.0]], requires_grad=True)
    old = torch.tensor([[0.0, -1000.0]])
    reference = torch.zeros((1, 2))
    output = grpo_loss(
        current,
        old,
        reference,
        torch.ones(1),
        torch.tensor([[True, False]]),
    )
    output.loss.backward()
    assert torch.isfinite(output.loss)
    assert current.grad is not None
    assert current.grad[0, 1] == 0
