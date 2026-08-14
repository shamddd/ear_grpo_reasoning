# Central Claim Freeze Document

## 1. Defensible Central Claim Classification

Based strictly and exclusively on the empirical data from the Phase IV 5-method controlled benchmark across 15 experimental runs, we classify the algorithmic claim as:

### **Outcome F: Current evidence does NOT support a positive algorithmic claim for EAR-GRPO.**

---

## 2. Frozen Empirical Ledger

1. **No Task Accuracy Advantage**:
   - $\text{Pass@1}(\text{EAR-GRPO}) = 16.67 \pm 0.00\%$
   - $\text{Pass@1}(\text{Standard-GRPO}) = 16.67 \pm 0.00\%$
   - Difference: $\Delta = 0.00\%$ ($p > 0.05$, Cohen's $d = 0.000$)

2. **Negative Control Falsification**:
   - $\text{Pass@1}(\text{Permuted-Control}) = 22.22 \pm 9.62\%$
   - $\text{Pass@1}(\text{EAR-GRPO}) = 16.67 \pm 0.00\%$
   - Permuting uncertainty values among rollouts outperforms true pairing, disproving the trajectory-specific epistemic advantage hypothesis.

3. **Compute Efficiency Inversion**:
   - Standard-GRPO requires 0.020 GPU-hours per run.
   - EAR-GRPO requires 0.065 GPU-hours per run ($3.25\times$ compute cost).
   - EAR delivers $0.00\%$ performance increase per unit FLOP.

---

## 3. Scientific Integrity Declaration

* We will **NOT** claim that EAR-GRPO improves mathematical reasoning.
* We will **NOT** claim that EAR-GRPO resolves "entropy collapse" into useful utility.
* We will **NOT** tune hyperparameters post-hoc until EAR wins.
* We publish the complete 15-seed dataset with full transparency.
