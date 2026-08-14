import pytest

from ear_grpo_reasoning.data import MathReasoningDataset, SmokeMathDataset


def test_smoke_dataset_has_explicit_identity() -> None:
    dataset = SmokeMathDataset()
    assert dataset.name == "builtin-smoke-v1"
    assert len(dataset) == 5
    assert dataset[0]["source_id"] == "smoke-000"


def test_smoke_dataset_does_not_wrap_indices() -> None:
    with pytest.raises(IndexError):
        _ = SmokeMathDataset()[5]


def test_unknown_external_dataset_is_rejected_without_fallback() -> None:
    with pytest.raises(ValueError, match="Only gsm8k"):
        MathReasoningDataset("svamp")
