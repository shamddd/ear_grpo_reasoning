import json
from pathlib import Path

from ear_grpo_reasoning.config import load_config
from ear_grpo_reasoning.evaluate import run_evaluation


def test_evaluation_smoke_writes_structured_outputs(tmp_path: Path) -> None:
    result = run_evaluation(load_config("configs/evaluation.yaml"), tmp_path)
    assert result["verification_scope"] == "software-verification-only"
    assert result["sample_count"] == 5
    assert result["mean"] == 0.8
    assert result["model_revision"] == "builtin-v1"
    assert result["dataset_revision"] == "builtin-v1"
    assert len(result["config_sha256"]) == 64
    assert (tmp_path / "summary.json").is_file()
    assert len((tmp_path / "records.jsonl").read_text(encoding="utf-8").splitlines()) == 5


def test_experiment_manifest_schema_requires_core_provenance() -> None:
    schema = json.loads(Path("results/experiment-manifest.schema.json").read_text(encoding="utf-8"))
    required = set(schema["required"])
    assert {
        "git_commit",
        "config_hash",
        "model_revision",
        "dataset_revision",
        "seeds",
        "python_version",
        "torch_version",
        "device",
        "number_of_samples",
        "number_of_updates",
        "metrics",
        "artifacts",
    } <= required
