# Phase VII Statistical Independence & Sample Size Audit

## 1. Experimental Unit & Sample Size Verification

To ensure zero pseudoreplication and formal statistical validity:

| Metric / Analysis | Sample Entity | Independent $N$ | Degrees of Freedom ($df$) | Statistical Test Used |
| :--- | :--- | :---: | :---: | :--- |
| **Phase VI Proxy Benchmark** | Unique Held-Out Question | **$N = 100$** | 98 | Pearson $r$, Spearman $r_s$, Partial $r$, Logistic Regression |
| **Self-Consistency Estimation** | Prompt-Clustered Rollouts | $K = 4$ per prompt | Aggregated to Prompt | Modal Frequency Disagreement ($U_{\text{SC}}$) |
| **Phase IV RL Matrix** | Independent Training Seed | **$N = 3$** per method | 2 | Seed-wise mean $\pm$ SD, exact paired differences |
| **Phase VII Confirmatory Set** | Untouched Held-Out Split | **$N = 100$** (indices 800–899) | 98 | Independent confirmatory test set |

---

## 2. Statistical Significance & Confidence Intervals for Phase VI

| Correlation Relationship | Sample Size ($N$) | Point Estimate ($r$) | 95% Bootstrap CI | Student's $t$-statistic | $p$-value | Significance ($\alpha = 0.05$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$r(U_{\text{SC}}, \text{Error})$** | 100 | **$+0.582$** | $[+0.435, +0.704]$ | $t = 7.05$ | $p = 2.4 \times 10^{-10}$ | **Statistically Significant** |
| **$r(H_{\text{token}}, \text{Error})$** | 100 | **$+0.214$** | $[+0.018, +0.395]$ | $t = 2.17$ | $p = 0.032$ | Marginal |
| **$r(H_{\text{token}}, \text{Length})$** | 100 | **$+0.486$** | $[+0.318, +0.627]$ | $t = 5.48$ | $p = 3.3 \times 10^{-7}$ | **Statistically Significant** |
| **Partial $r(H_{\text{token}}, \text{Correct} \mid \text{Length})$** | 100 | **$-0.092$** | $[-0.284, +0.108]$ | $t = -0.91$ | $p = 0.365$ | **NOT SIGNIFICANT (Confounded)** |
| **Partial $r(U_{\text{SC}}, \text{Correct} \mid \text{Length})$** | 100 | **$-0.569$** | $[-0.693, -0.418]$ | $t = -6.80$ | $p = 8.1 \times 10^{-10}$ | **ROBUST & SIGNIFICANT** |

---

## 3. Pseudoreplication Safeguards

1. **No Inflation of $N$**: Multiple rollouts generated from the same prompt are never counted as separate independent observations. All rollouts from prompt $i$ are collapsed into a single prompt-level consensus statistic ($U_{\text{SC}}^{(i)}$).
2. **Dataset Partitioning**:
   - Training split: indices 0–499 ($N = 500$)
   - Development/Tuning split: indices 500–699 ($N = 200$)
   - Phase VI Discovery split: indices 700–799 ($N = 100$)
   - **Phase VII Untouched Confirmatory split**: indices 800–899 ($N = 100$)
