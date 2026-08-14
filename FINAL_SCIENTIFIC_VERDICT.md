# EAR-GRPO Phase II: Diagnostic Pilot Verification Record

> [!WARNING]
> **CORRECTED SCIENTIFIC CLAIM NOTICE (PHASE III AUDIT)**
> The Phase II pilot results established that under a 16-token sparse-reward pilot configuration with `gpt2`, **all methods achieved 0.00% Pass@1 accuracy**. Therefore, EAR-GRPO is **NOT** currently demonstrated to improve math reasoning accuracy, out-of-distribution reasoning, or benchmark performance over standard GRPO. The Phase II evidence establishes **only** that EAR-GRPO modified optimization dynamics by retaining higher measured policy exploration entropy ($7.5596$ vs $3.8094$ nats).

---

## 1. Executive Diagnostic Summary

Phase II of **EAR-GRPO** (*Epistemic Advantage Regularization for Group Relative Policy Optimization*) evaluated real PyTorch transformer models (`gpt2`) on GSM8K math prompts.

- **Phase I Synthetic Placeholders**: Formally invalidated and archived in [`research/EXPERIMENT_FORENSICS.md`](./research/EXPERIMENT_FORENSICS.md).
- **Phase II Real Model Pilot Archive**: Preserved in [`results/archive/phase2_verified/`](./results/archive/phase2_verified/).
- **Current Empirical Finding**: All 5 methods across 3 random seeds achieved $0.00\%$ Pass@1 accuracy.

---

## 2. Phase II Empirical Results (16-Token Pilot Setup)

| Method | Group Size ($G$) | Pass@1 Accuracy | Measured Policy Entropy (nats) | Epistemic Dampening ($\lambda_i$) |
| :--- | :---: | :---: | :---: | :---: |
| **Standard-GRPO** | $G=4$ | $0.00 \pm 0.00\%$ | $3.8094 \pm 0.040$ | N/A ($1.000$) |
| **Compute-Matched-GRPO** | $G=7$ | $0.00 \pm 0.00\%$ | $4.6252 \pm 0.139$ | N/A ($1.000$) |
| **Random-Control** | $G=4$ | $0.00 \pm 0.00\%$ | $5.4733 \pm 0.358$ | $0.652 \pm 0.054$ |
| **Permuted-Control** | $G=4$ | $0.00 \pm 0.00\%$ | $6.5162 \pm 0.228$ | $0.678 \pm 0.108$ |
| **EAR-GRPO (Ours)** | **$G=4$** | **$0.00 \pm 0.00\%$** | **$7.5596 \pm 0.250$** | **$0.597 \pm 0.142$** |

---

## 3. Allowed Claims vs Prohibited Claims

### Allowed Claim
Under the present sparse-reward pilot configuration, EAR-GRPO produced higher measured policy exploration entropy than standard GRPO.

### Prohibited Claims (Not Supported by Current Data)
- ❌ EAR-GRPO improves reasoning accuracy.
- ❌ EAR-GRPO outperforms GRPO on GSM8K.
- ❌ EAR-GRPO improves out-of-distribution reasoning.
