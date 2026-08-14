import pytest

from ear_grpo_reasoning.evaluation import aggregate_binary, compute_pass_at_k


def test_pass_at_one() -> None:
    assert compute_pass_at_k([[True, False], [False, False]], k=1) == pytest.approx(0.25)


def test_pass_at_k_perfect_when_failures_fewer_than_k() -> None:
    assert compute_pass_at_k([[True, True, False]], k=2) == 1.0


def test_pass_at_k_validates_sample_count() -> None:
    with pytest.raises(ValueError, match="at least"):
        compute_pass_at_k([[True]], k=2)


def test_binary_summary_reports_sample_std() -> None:
    summary = aggregate_binary([1.0, 0.0, 1.0, 0.0])
    assert summary.mean == 0.5
    assert summary.standard_deviation == pytest.approx(0.577350269)
    assert summary.sample_count == 4
