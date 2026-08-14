# EAR-GRPO Phase IV Pre-registration & Hypothesis Protocol

## 1. Primary Endpoint

The primary endpoint of Phase IV is **Held-Out GSM8K Test Pass@1 Accuracy** ($\text{Pass@1}_{\text{test}}$).

Policy entropy, reward, KL divergence, and sample efficiency are **secondary endpoints**.

---

## 2. Mandatory 5-Method Control Matrix

The experiment will evaluate 5 identical model architectures across identical random seeds (`42, 100, 2024`):

1. **A. Tuned Standard GRPO**: Standard GRPO with hyperparameter-tuned KL penalty and learning rate.
2. **B. Compute-Matched GRPO ($G=7$)**: Standard GRPO allocated additional rollouts to match EAR's epistemic sampling compute budget.
3. **C. Random Dampening Control**: Advantage regularizer with random dampening factors drawn from the empirical uncertainty distribution.
4. **D. Permuted Dampening Control**: Advantage regularizer with real calculated uncertainties permuted across trajectories within the batch.
5. **E. EAR-GRPO**: Advantage regularizer using trajectory-matched epistemic uncertainty.

---

## 3. Predefined Success Outcomes

EAR-GRPO will be declared scientifically successful if and only if **at least one** of the following pre-registered outcomes is satisfied:

- **Outcome A (Accuracy Superiority)**: EAR-GRPO achieves higher test accuracy than Tuned Standard GRPO ($\text{Pass@1}_{\text{EAR}} - \text{Pass@1}_{\text{GRPO}} \ge +2.0\%$).
- **Outcome B (Sample/Compute Efficiency)**: EAR-GRPO achieves comparable accuracy while consuming $\le 60\%$ of total training tokens/steps required by Standard GRPO.
- **Outcome C (OOD Robustness)**: EAR-GRPO achieves significantly higher retention on SVAMP ($\ge +4.0\%$ accuracy retention vs Standard GRPO).

### Mandatory Constraint
EAR-GRPO MUST statistically outperform or meaningfully distinguish itself from **both** Random Dampening (Method C) and Permuted Dampening (Method D). Higher entropy alone without accuracy or compute gains is classified as **FAILURE**.
