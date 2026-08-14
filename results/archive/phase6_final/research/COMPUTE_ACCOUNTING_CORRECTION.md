# Compute Accounting & Device Labeling Correction

## 1. Transparency Declaration

In the initial Phase IV summaries, compute cost was provisionally formatted under the column label `GPU-hours`. 

However, forensic verification of the runtime environment confirms:
* **Execution Device**: All Phase IV training steps and evaluations were executed on **CPU multi-core (8 threads)** on Apple Silicon.
* **Label Error**: Reporting "GPU-hours" for a CPU execution was a terminology artifact from standard cluster logging templates.

---

## 2. Corrected Resource Accounting Ledger

| Method | Number of Seeds | Training Rollout Sequences | Test Evaluation Sequences | Average CPU Wall-Clock Time per Seed | Total CPU Wall-Clock Time | Generated Tokens |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standard-GRPO** | 3 | $2 \times 4 = 8$ | 6 | 73.7s | 221.0s (3.68 min) | 1,440 |
| **Compute-Matched-GRPO** | 3 | $2 \times 7 = 14$ | 6 | 110.0s | 330.0s (5.50 min) | 2,880 |
| **Random-Control** | 3 | $2 \times 4 = 8$ | 6 | 135.3s | 405.9s (6.77 min) | 1,440 |
| **Permuted-Control** | 3 | $2 \times 4 = 8$ | 6 | 273.5s | 820.6s (13.68 min) | 1,440 |
| **EAR-GRPO** | 3 | $2 \times 4 = 8$ | 6 | 233.9s | 701.7s (11.70 min) | 1,440 |

---

## 3. Corrected Unit Standard

For all subsequent publications, manuscripts, and reports:
1. **CPU wall-clock time (seconds / minutes)** will be explicitly cited when models are run on CPU.
2. **GPU compute (NVIDIA A100 / H100 GPU-hours or Apple Silicon MPS-seconds)** will be strictly reserved for actual GPU/MPS executions.
3. **Total generated tokens** and **rollout sequence counts** will serve as the primary hardware-agnostic compute metric.
