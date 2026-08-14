"""Math-reasoning datasets without silent synthetic fallbacks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MathSample:
    question: str
    ground_truth: str
    solution_trace: str
    source_id: str

    def as_dict(self) -> dict[str, str]:
        return {
            "question": self.question,
            "ground_truth": self.ground_truth,
            "solution_trace": self.solution_trace,
            "source_id": self.source_id,
        }


_SMOKE_SAMPLES = (
    MathSample("What is 2 + 3?", "5", "2 + 3 = 5. #### 5", "smoke-000"),
    MathSample("What is 12 / 4?", "3", "12 / 4 = 3. #### 3", "smoke-001"),
    MathSample("What is 7 * 6?", "42", "7 * 6 = 42. #### 42", "smoke-002"),
    MathSample("What is 10 - 13?", "-3", "10 - 13 = -3. #### -3", "smoke-003"),
    MathSample("What is one half as a decimal?", "0.5", "1 / 2 = 0.5. #### 0.5", "smoke-004"),
)


class SmokeMathDataset:
    """Five labeled fixtures used only for deterministic software verification."""

    name = "builtin-smoke-v1"
    split = "verification"

    def __len__(self) -> int:
        return len(_SMOKE_SAMPLES)

    def __getitem__(self, index: int) -> dict[str, str]:
        return _SMOKE_SAMPLES[index].as_dict()


class MathReasoningDataset:
    """Load an explicitly named external benchmark through Hugging Face Datasets.

    The adapter never falls back to synthetic data and never wraps out-of-range
    indices. This prevents a small fixture set from being mislabeled as a benchmark.
    """

    def __init__(
        self,
        name: str = "gsm8k",
        split: str = "train",
        *,
        revision: str | None = None,
    ) -> None:
        if name != "gsm8k":
            raise ValueError("Only gsm8k is implemented; use SmokeMathDataset for offline checks")
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("External datasets require `pip install -e '.[research]'`") from exc
        self.name = name
        self.split = split
        self.revision = revision
        load_kwargs: dict[str, Any] = {"split": split}
        if revision is not None:
            load_kwargs["revision"] = revision
        self._dataset = load_dataset("openai/gsm8k", "main", **load_kwargs)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> dict[str, str]:
        row: Mapping[str, Any] = self._dataset[index]
        question = row.get("question")
        answer = row.get("answer")
        if not isinstance(question, str) or not isinstance(answer, str):
            raise ValueError(f"Malformed GSM8K row at index {index}")
        ground_truth = answer.rsplit("####", maxsplit=1)[-1].strip()
        return MathSample(
            question=question,
            ground_truth=ground_truth,
            solution_trace=answer,
            source_id=(
                f"openai/gsm8k@{self.revision or 'unresolved-revision'}:main:{self.split}:{index}"
            ),
        ).as_dict()
