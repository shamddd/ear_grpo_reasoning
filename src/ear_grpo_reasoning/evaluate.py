"""Deterministic evaluation entry point for software verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ear_grpo_reasoning.config import ProjectConfig, load_config
from ear_grpo_reasoning.data import SmokeMathDataset
from ear_grpo_reasoning.evaluation import aggregate_binary
from ear_grpo_reasoning.rewards import compute_math_reward, extract_answer
from ear_grpo_reasoning.seed import seed_everything


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Line {line_number} of {path} is not a JSON object")
        records.append(record)
    if not records:
        raise ValueError(f"Prediction file is empty: {path}")
    return records


def _smoke_records() -> list[dict[str, str]]:
    dataset = SmokeMathDataset()
    records = [dataset[index] for index in range(len(dataset))]
    # One deliberate error ensures both verifier outcomes are exercised.
    records[-1] = {**records[-1], "solution_trace": "1 / 2 = 0.4. #### 0.4"}
    return records


def run_evaluation(config: ProjectConfig, output_dir: Path) -> dict[str, Any]:
    """Evaluate JSONL predictions or the built-in offline smoke fixture."""

    seed_everything(config.experiment.seed, deterministic=config.experiment.deterministic)
    if config.evaluation.predictions_path:
        source = Path(config.evaluation.predictions_path)
        raw_records = _load_jsonl(source)
        scope = "user-supplied-predictions"
    elif config.data.name == "builtin-smoke-v1":
        raw_records = _smoke_records()
        scope = "software-verification-only"
    else:
        raise ValueError(
            "A predictions_path is required for non-smoke evaluation; model generation is explicit"
        )

    records: list[dict[str, Any]] = []
    rewards: list[float] = []
    for index, raw in enumerate(raw_records):
        completion = raw.get("completion", raw.get("solution_trace"))
        target = raw.get("ground_truth")
        if not isinstance(completion, str) or not isinstance(target, str):
            raise ValueError(f"Record {index} must contain string completion and ground_truth")
        reward = compute_math_reward(completion, target)
        rewards.append(reward)
        records.append(
            {
                "index": index,
                "source_id": raw.get("source_id", f"record-{index}"),
                "prediction": extract_answer(completion),
                "target": extract_answer(target),
                "reward": reward,
            }
        )

    summary = aggregate_binary(rewards)
    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "verification_scope": scope,
        "metric": "numeric_exact_match",
        "mean": summary.mean,
        "standard_deviation": summary.standard_deviation,
        "sample_count": summary.sample_count,
        "seed": config.experiment.seed,
        "model": config.model.name,
        "model_revision": config.model.revision,
        "dataset": config.data.name,
        "dataset_revision": config.data.revision,
        "split": config.data.split,
        "config_sha256": config.sha256,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "records.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    output_dir = args.output_dir or Path(config.evaluation.output_dir)
    result = run_evaluation(config, output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
