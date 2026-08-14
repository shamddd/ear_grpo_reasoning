# Generation Budget & Token Length Audit

## 1. Experimental Token Budgets Across Phases

To eliminate any potential confounding across evaluation regimes, we delineate the three distinct generation budgets used in this research:

| Token Budget (`max_new_tokens`) | Target Use Case | Average Output Tokens Generated | Truncation Rate | Benchmark Role |
| :---: | :--- | :---: | :---: | :--- |
| **48 Tokens** | Rapid local RL inner-loop training rollouts & development pilot | $48.0$ tokens | $83.3\%$ | **Development Pilot Only**: Intentionally short to test multi-seed gradient updates in constrained CPU memory. |
| **144 Tokens** | Diagnostic uncertainty proxy benchmark & self-consistency rollouts | $112.4$ tokens | $0.0\%$ | **Diagnostic Standard**: Captures complete step-by-step mathematical reasoning and final boxed answers without mid-sentence truncation. |
| **256 Tokens** | Baseline model capacity evaluation & full held-out benchmark | $232.2$ tokens | $0.0\%$ | **Canonical Performance Standard**: Allows verbose Chain-of-Thought derivations with zero truncation. |

---

## 2. Policy Comparison Integrity Rules

1. **No Mixed Comparisons**: Methods evaluated at 48 tokens (Phase IV pilot) are never compared against methods evaluated at 144 or 256 tokens.
2. **Canonical Baseline Alignment**: In the final canonical results table ([`results/FINAL_CANONICAL_RESULTS.json`](./results/FINAL_CANONICAL_RESULTS.json)), both the 48-token pilot and the full-budget evaluations are presented side-by-side with explicit token budget annotations.
