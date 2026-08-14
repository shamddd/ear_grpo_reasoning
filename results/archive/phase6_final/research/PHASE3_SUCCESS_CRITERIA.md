# EAR-GRPO Phase III Predeclared Success Criteria

## 1. Primary Objective

The primary goal of Phase III is **Demonstrable Task Learning & Performance Efficiency** on mathematical reasoning benchmarks using models with non-zero baseline capability (`Qwen/Qwen2.5-0.5B-Instruct` or `Qwen/Qwen2.5-1.5B-Instruct`).

Policy entropy is relegated to a **secondary diagnostic metric**.

---

## 2. Predeclared Success Criteria

EAR-GRPO will be classified as a scientifically successful post-training algorithm if and only if it satisfies **at least one** of the following primary performance criteria under a fully tuned experimental matrix:

### Criterion A: Superior Final Task Accuracy
- **Requirement**: EAR-GRPO achieves higher mean Pass@1 accuracy on GSM8K than tuned Standard GRPO:
  $$\text{Pass@1}_{\text{EAR-GRPO}} \ge \text{Pass@1}_{\text{Standard-GRPO}} + \Delta_{\text{min}}$$
  where $\Delta_{\text{min}} = +2.0\%$ with statistical significance ($p < 0.05$).

### Criterion B: Higher Sample/Compute Efficiency
- **Requirement**: EAR-GRPO reaches a target accuracy threshold (e.g. 85.0% Pass@1 on GSM8K) in $\le 60\%$ of the training steps / generated tokens required by Standard GRPO.

### Criterion C: Superior Out-of-Distribution (OOD) Generalization
- **Requirement**: Equal or comparable in-distribution (GSM8K) performance, but significantly higher out-of-distribution (SVAMP / GSM-Hard) accuracy retention ($+4.0\%$ retention over standard GRPO).

---

## 3. Mandatory Control Benchmarks

To ensure the observed benefit stems from **true epistemic advantage regularization**, EAR-GRPO must outperform all three control baselines:

1. **Compute-Matched GRPO ($G=7$)**: Ensures gains are not due to additional rollout compute.
2. **Random Dampening Control**: Ensures gains are not due to arbitrary advantage scaling.
3. **Permuted Variance Control**: Ensures gains stem from trajectory-specific epistemic uncertainty rather than overall variance magnitude.

---

## 4. Failure Classification Rules

If EAR-GRPO fails all three criteria (A, B, C) or does not outperform the control baselines:
- **Verdict**: Algorithm hypothesis is **NOT SUPPORTED**.
- **Action**: Do not submit as an algorithmic improvement paper. Reframe transparently as a diagnostic/limitation study or archive as a negative result.
