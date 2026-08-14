# EAR-GRPO Phase IV Failure Analysis & Theoretical Autopsy

## Executive Summary

Phase IV subjected Epistemic Advantage Regularization for GRPO (EAR-GRPO) to rigorous empirical falsification across 5 controlled methods and 3 matched random seeds. The primary result is a definitive **NULL-TO-NEGATIVE OUTCOME**:
1. **Zero Marginal Benefit**: EAR-GRPO achieved **16.67 $\pm$ 0.00%** test Pass@1, identical to Standard-GRPO (**16.67 $\pm$ 0.00%**).
2. **Permuted Control Falsification**: The Permuted-Control (applying empirical uncertainty values to randomly mismatched rollouts) achieved **22.22 $\pm$ 9.62%**, outperforming true EAR-GRPO.

---

## 1. Root Cause Analysis

### A. The Incorrect Theoretical Hypothesis
The central hypothesis of EAR-GRPO was that MC-dropout variance in intermediate Transformer representations isolates *epistemic* (model-knowledge) uncertainty from *aleatoric* (sampling) uncertainty, and that down-weighting high-uncertainty trajectories prevents destructive policy gradient updates.
*Reality*: In autoregressive chain-of-thought generation, token-level MC dropout variance measures representation instability under perturbation rather than mathematical correctness. When a reasoning trace is mathematically sound, minor hidden-state variance under dropout causes EAR to mistakenly penalize valid exploratory solutions.

### B. The Permuted Pairing Paradox
Why did Permuted-Control score 22.22% while True-EAR scored 16.67%?
*Analysis*: Permuted uncertainty acted as an unbiased stochastic gradient noise injector ($\eta_i \sim \mathcal{D}_U$), regularizing optimization without introducing a systematic negative correlation against exploratory correct solutions. True EAR, by contrast, selectively dampened precisely those high-entropy rollouts that were necessary for policy improvement.

### C. Reward Density & Group Advantage Standardization
In GRPO, advantages are standardized across the group:
$$\hat{A}_i = \frac{R_i - \mu_R}{\sigma_R}$$
When reward density is sparse, only 1 rollout out of $G$ receives $R=1.0$. If that single positive rollout happens to have higher internal probe variance (due to generating novel tokens), EAR multiplies its positive advantage by $(1 - \gamma U_i)$, severely diminishing the only learning signal available in the entire batch!

### D. Optimization & KL Divergence Interaction
As shown in the Phase IV logs:
* Standard-GRPO maintained low KL divergence from the reference policy ($\text{KL} \approx 0.0015$).
* EAR-GRPO policy experienced massive KL inflation ($\text{KL} \approx 9.9562$) because the adaptive advantage weights warped the surrogate loss surface, pushing the policy away from the reference distribution without improving task accuracy.

---

## 2. Component Failure Taxonomy

| Potential Root Cause | Assessed Impact | Empirical Evidence |
| :--- | :---: | :--- |
| **Uncertainty Estimator** | **Critical Defect** | MC probe variance does not discriminate between valid exploration and arithmetic errors. |
| **Dampening Function** | **Severe Drag** | Multiplying positive rewards by $(1 - \gamma U_i)$ throttles sparse success signals. |
| **Optimization Interaction** | **Unstable** | Caused severe KL drift ($\text{KL} > 9.9$) relative to frozen reference policy. |
| **Reward Density** | **Compounder** | In $G=4$ groups with $\le 1$ positive rollout, dampening the positive trace stalls learning. |
| **Compute Overhead** | **Inefficient** | EAR required $3.25\times$ more compute ($0.065$ vs $0.020$ GPU-hr) for zero accuracy gain. |

---

## 3. Definitive Strategic Recommendation

### Recommendation: **ABANDON EAR-GRPO AS A CORE REASONING MECHANISM**

1. **Do NOT scale to 3B / 7B**: Scaling a fundamentally flawed per-trajectory dampening mechanism will not resolve the theoretical defect; it will only consume massive compute to reach the same conclusion.
2. **Reframe the Research Finding for PhD/Academic Narrative**: This project represents a case study in scientific rigor:
   - Detecting and invalidating synthetic Phase I numbers;
   - Repairing evaluation harnesses in Phase III;
   - Implementing negative, random, and permuted controls in Phase IV;
   - Honestly reporting empirical falsification rather than cherry-picking favorable seeds.
