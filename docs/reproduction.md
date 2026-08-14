# Reproduction and verification

## Verified software path

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

The final three commands create GRPO and EAR training diagnostics in their respective
`results/generated/*-smoke/` directories and an evaluation summary plus prompt-level
records in `results/generated/evaluation-smoke/`. They verify configuration,
deterministic seeding, masks, gradients, rewards, metrics, and structured output. They do
not reproduce manuscript results.

## Paper-result reproduction status

Full reproduction is blocked by missing raw Phase VII prompt-level artifacts and exact
external-data provenance in the original public commit. The historical scripts also
predate the strict configuration schema. Reinstatement requires:

1. exact dataset revisions and split indices;
2. exact model and tokenizer revisions;
3. immutable prompt-level generations and per-seed training logs;
4. a complete package/environment inventory;
5. an aggregate ledger generated solely from those raw files.

The `research` dependency extra provides libraries needed to rehabilitate the historical
Hugging Face scripts, but installing dependencies alone does not resolve those evidence
gaps. No paper metric should be regenerated from the built-in smoke fixtures.
