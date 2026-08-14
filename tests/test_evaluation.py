from pathlib import Path

from ear_grpo_reasoning.config import load_config
from ear_grpo_reasoning.evaluate import run_evaluation


def test_evaluation_smoke_writes_structured_outputs(tmp_path: Path) -> None:
    result = run_evaluation(load_config("configs/evaluation.yaml"), tmp_path)
    assert result["verification_scope"] == "software-verification-only"
    assert result["sample_count"] == 5
    assert result["mean"] == 0.8
    assert len(result["config_sha256"]) == 64
    assert (tmp_path / "summary.json").is_file()
    assert len((tmp_path / "records.jsonl").read_text(encoding="utf-8").splitlines()) == 5
