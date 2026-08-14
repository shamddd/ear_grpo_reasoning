# Software architecture

The maintained package is deliberately small. Historical experiment runners remain
separate because their evidence provenance is incomplete.

```mermaid
flowchart TD
    Y[YAML config] --> C[Strict config validation]
    C --> S[Central RNG seeding]
    C --> D{Data source}
    D -->|offline| F[Named smoke fixtures]
    D -->|research extra| H[Explicit Hugging Face dataset]
    F --> R[Numeric reward verifier]
    H --> G[Policy generation]
    G --> R
    G --> U[Optional MC-dropout probe]
    R --> A[Group advantage]
    U --> A
    A --> L[Masked token-level GRPO loss]
    L --> T[Optimizer step]
    R --> E[Evaluation summary and records]
```

Key boundaries:

- `algorithms/` contains pure tensor operations with no model download side effects.
- `models/` lazily imports Hugging Face dependencies and detects degenerate MC dropout.
- `data/` never substitutes synthetic fixtures for a named external benchmark.
- `evaluate.py` evaluates existing predictions; it does not hide model generation.
- `smoke_train.py` exercises gradients with a tiny local model and clearly labels its
  output as software verification.
- `results/generated/` is ignored; committed results require provenance review.
