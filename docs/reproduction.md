# Reproduction and verification

## Level A — software reproducibility

Create an isolated Python 3.10–3.12 environment, then run:

```bash
python -m pip install -e ".[dev]"
python -m pip check
ruff check .
ruff format --check .
mypy -p ear_grpo_reasoning
pytest -q
python -c "import ear_grpo_reasoning"
python -m ear_grpo_reasoning.smoke_train --config configs/smoke.yaml
python -m ear_grpo_reasoning.smoke_train --config configs/ear_smoke.yaml
python -m ear_grpo_reasoning.evaluate --config configs/evaluation.yaml
```

`pyproject.toml` is the single dependency source of truth. The deliberately minimal
`requirements.txt` contains only `-e .` as a compatibility entry point, preventing a
second dependency list from drifting.

The final three commands create GRPO and EAR training diagnostics in their respective
`results/generated/*-smoke/` directories and an evaluation summary plus prompt-level
records in `results/generated/evaluation-smoke/`. They verify configuration,
deterministic seeding, masks, gradients, rewards, metrics, and structured output. They do
not reproduce manuscript results.

Level A also includes the explicit GSM8K adapter and optional tiny public-model
integration when network access is available. External downloads are kept outside the
fast offline CI gate.

## Level B — experimental reproduction

Experimental reproduction requires:

1. exact model and tokenizer identifiers and revisions;
2. exact dataset identifiers, revisions, configurations, and split indices;
3. exact seeds, resolved hyperparameters, number of samples, and update counts;
4. hardware, precision, Python, PyTorch, Transformers, CUDA, and package inventory;
5. checkpoints, immutable prompt-level generations, and per-seed raw logs;
6. cryptographic hashes for configs and artifacts;
7. an aggregate ledger generated solely from those raw files.

The machine-readable contract is `results/experiment-manifest.schema.json`. Unavailable
historical values must remain null or omitted as allowed by the schema; they must not be
inferred.

## Level C — paper-number reproduction

Level C requires a provenance-complete Level B bundle and a full rerun that regenerates
the submitted manuscript's reported values. It is not currently claimable: the public
repository lacks the raw Phase VII prompt-level artifact and exact external-data
provenance, and the full experiments have not been rerun.

The `research` dependency extra provides libraries needed to rehabilitate the historical
Hugging Face scripts, but installing dependencies alone does not resolve those evidence
gaps. No paper metric should be regenerated from the built-in smoke fixtures.
