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


def test_clipping_uses_frozen_old_log_probs() -> None:
    current, old, reference, advantages, mask = _inputs()
    current = current.detach().clone().requires_grad_(True)
    current.data[0, 0] = 0.0
    output = grpo_loss(current, old, reference, advantages, mask, clip_epsilon=0.2)
    assert output.clip_fraction > 0


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
