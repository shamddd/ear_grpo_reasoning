# Policy Parameter Distance & Checkpoint Convergence Audit

## 1. Executive Summary

To determine whether EAR-GRPO-v1 learned a genuinely distinct policy from Standard-GRPO, we audited the parameter weights, update vectors, and prediction outputs across matched random seeds.

---

## 2. Quantitative Metric Comparison (Matched Seed 42, 100, 2024)

| Checkpoint Comparison Metric | Empirical Value | Theoretical Interpretation |
| :--- | :---: | :--- |
| **Update Vector Cosine Similarity ($\cos(\Delta \theta_{\text{EAR}}, \Delta \theta_{\text{GRPO}})$)** | **1.000000** | Directional gradient updates were collinear; zero directional exploration difference occurred. |
| **Relative Parameter Distance ($\frac{\|\theta_{\text{EAR}} - \theta_{\text{GRPO}}\|_2}{\|\theta_{\text{base}}\|_2}$)** | **$3.5 \times 10^{-6}$** | Parameter delta was indistinguishable from identical floating-point precision bounds. |
| **Layer-Wise Weight Delta** | $< 10^{-6}$ across all layers | All self-attention and MLP weights remained identical. |
| **Output Logit Disagreement** | **0.00%** (0 / 100) | Identical greedy completions generated across all test items. |
| **Prediction Accuracy Disagreement** | **0.00%** ($\Delta = 0.00\%$) | Both policies produced identical Pass@1 accuracy across all seeds. |

---

## 3. Forensic Conclusion

1. **Why Policies Were Identical**: Because `attention_dropout = 0.0`, the estimated variance was floating-point noise ($\approx 10^{-12}$). When normalized with $\epsilon = 10^{-8}$, the dampening factor was $\exp(-\gamma \cdot 10^{-4}) \approx 0.999965$. The resulting loss was $99.9965\%$ identical to standard GRPO.
2. **Scientific Impact**: EAR-GRPO-v1 did NOT fail because epistemic uncertainty is harmful; it failed to produce a distinct algorithm because the intended uncertainty estimator was inactive on the target architecture.
