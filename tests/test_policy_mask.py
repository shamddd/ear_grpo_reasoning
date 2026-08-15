import pytest
import torch

from ear_grpo_reasoning.models.policy import TransformerReasoningPolicy


def test_completion_mask_includes_eos_and_excludes_padding() -> None:
    ids = torch.tensor([[1, 2, 3, 9, 0], [1, 2, 4, 5, 6]])
    mask = TransformerReasoningPolicy.completion_mask(
        ids,
        [2, 3],
        pad_token_id=0,
        eos_token_id=9,
    )
    assert torch.equal(
        mask,
        torch.tensor([[False, True, True, False], [False, False, True, True]]),
    )


def test_invalid_completion_start_is_rejected() -> None:
    with pytest.raises(ValueError, match="invalid completion start"):
        TransformerReasoningPolicy.completion_mask(
            torch.tensor([[1, 2, 3]]),
            [3],
            pad_token_id=None,
            eos_token_id=None,
        )


def test_eos_is_included_when_it_is_also_the_pad_token() -> None:
    ids = torch.tensor([[1, 2, 3, 9, 9]])
    mask = TransformerReasoningPolicy.completion_mask(
        ids,
        [2],
        pad_token_id=9,
        eos_token_id=9,
    )
    assert torch.equal(mask, torch.tensor([[False, True, True, False]]))


def test_leading_padding_attention_preserves_later_eos() -> None:
    ids = torch.tensor([[9, 9, 1, 2, 9]])
    mask = TransformerReasoningPolicy.leading_padding_attention_mask(ids, pad_token_id=9)
    assert torch.equal(mask, torch.tensor([[0, 0, 1, 1, 1]]))
