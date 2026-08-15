import sys
from types import SimpleNamespace

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


def test_gsm8k_revision_is_forwarded_and_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_load_dataset(path: str, config: str, **kwargs: str) -> list[dict[str, str]]:
        assert (path, config) == ("openai/gsm8k", "main")
        assert kwargs == {"split": "train[:1]", "revision": "dataset-commit"}
        return [{"question": "What is 1 + 1?", "answer": "1 + 1 = 2. #### 2"}]

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))
    dataset = MathReasoningDataset("gsm8k", split="train[:1]", revision="dataset-commit")
    assert dataset[0]["source_id"] == "openai/gsm8k@dataset-commit:main:train[:1]:0"
