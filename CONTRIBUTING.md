# Contributing

Use Python 3.10–3.12 and install the development extra. Before opening a change, run the
verification sequence in `docs/reproduction.md`.

Contributions that change an algorithm must include focused mathematical edge-case tests
and explain any difference from the submitted formulation. New result files must record
the dataset/model revisions, split, seed, configuration hash, sample count, and raw
prompt-level evidence. Smoke outputs and model checkpoints must not be committed.

Do not add publication, novelty, significance, or benchmark claims without committed
evidence that regenerates them.
