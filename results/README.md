# Results provenance

Only results with enough committed provenance to be regenerated should be described as
verified. `FINAL_CANONICAL_RESULTS.json` is therefore a status manifest, not a benchmark
table.

The files under `raw/` and `archive/` are preserved as historical research artifacts.
They are not exercised by CI and must not be combined across token budgets, dataset
splits, models, or experimental phases. In particular, the public repository does not
contain the Phase VII `causal_validation_summary.json` referenced by several historical
documents.

New local outputs belong under `results/generated/`, which is Git-ignored. Evaluation
summaries follow `schema.json` and distinguish software verification from paper-result
reproduction.
