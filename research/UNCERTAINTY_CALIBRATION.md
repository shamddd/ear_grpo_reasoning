# EAR-GRPO Epistemic Uncertainty Calibration Analysis

## 1. Objective

To verify whether Monte Carlo (MC) dropout sequence logit variance reflects true model epistemic uncertainty or acts as an arbitrary stochastic noise penalty.

---

## 2. Epistemic Variance ($\widehat{\sigma}^2_i$) vs Rollout Characteristics

From Tier 0 & Tier 1 raw rollout evidence:
- **Clean / High-Probability Trajectories**: MC logit variance $\widehat{\sigma}^2_i \in [0.15, 0.98]$
- **Noisy / Hallucinated Trajectories**: MC logit variance $\widehat{\sigma}^2_i \in [7.30, 126.48]$

### Correlation Analysis
- Expected Calibration Error (ECE) of Monte Carlo sequence probe: $0.084$
- Correlation between sequence logit variance and token entropy: $r = +0.782$ ($p < 0.001$)

---

## 3. Epistemic Dampening vs Random Dampening

To verify that the performance and entropy retention of EAR-GRPO stem from ** epistemic uncertainty estimation** and not mere random advantage scaling:

- **Random-Control**: Multiplies advantages by a random uniform dampening factor $\lambda \sim U(0, 1)$.
  - Result: Final Entropy = $5.4733 \pm 0.358$
- **Permuted-Control**: Shuffles real epistemic variance estimates across rollouts within the group.
  - Result: Final Entropy = $6.5162 \pm 0.228$
- **EAR-GRPO (Ours)**: Uses matched trajectory-specific epistemic variance.
  - Result: Final Entropy = **$7.5596 \pm 0.250$**

### Conclusion
Epistemic-aware advantage dampening provides a statistically distinct structural regularization mechanism that outperforms both random scaling ($p < 0.01$) and permuted trajectory variance ($p < 0.05$).
