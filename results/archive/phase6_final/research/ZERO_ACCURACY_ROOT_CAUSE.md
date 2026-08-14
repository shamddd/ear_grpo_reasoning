# EAR-GRPO Phase III: Zero-Accuracy Root Cause Audit

## Executive Summary

Phase II resulted in $0.00\%$ Pass@1 accuracy across all methods (`Standard-GRPO`, `Compute-Matched-GRPO`, `Random-Control`, `Permuted-Control`, `EAR-GRPO`). To restore a viable reinforcement learning signal, we conducted a systematic audit across 10 hypotheses (A through J).

---

## Systematic Hypothesis Matrix

| Hypothesis | Factor Audited | Test Method & Evidence | Verdict |
| :--- | :--- | :--- | :---: |
| **A. Model Capability** | Untuned `gpt2` (124M) vs Instruction-tuned `Qwen2.5-0.5B-Instruct` | `gpt2` scored 0.00% at all budgets. `Qwen2.5-0.5B-Instruct` scored **80.00% Pass@1** on GSM8K (4/5). | **PRIMARY ROOT CAUSE** |
| **B. Prompt Format** | Plain text vs `apply_chat_template()` | Plain text string `"Question: ...\nAnswer:"` bypassed instruction tuning. `apply_chat_template()` formatted system/user roles correctly. | **PRIMARY ROOT CAUSE** |
| **C. Generation Budget** | 16 tokens vs 256 tokens | 16-token budget truncated 100% of generations before reasoning finished. 256 tokens allowed complete reasoning chains. | **PRIMARY ROOT CAUSE** |
| **D. Answer Extraction** | Parser Regex | Discovered regex flaw ignoring negative signs (`r"[-+]?\d*\.\d+|\d+"`). Upgraded to `r"[-+]?\d+(?:\.\d+)?"` + LaTeX `\boxed{}`. | **REPAIRED (FIXED)** |
| **E. Reward Verifier** | Binary Math Verifier | Audited across 30 hand-constructed valid/invalid test cases. Passed 30/30 cases (100% accuracy). | **VERIFIED (100% CORRECT)** |
| **F. Training Setup** | Optimization steps | 2-4 steps in pilot was insufficient when baseline reward was 0. | **CONTRIBUTING FACTOR** |
| **G. Sparse Reward** | Zero-reward rollout groups | With 0% base capability, 100% of GRPO rollout groups had 0 reward variance, zeroing out policy gradients. | **PRIMARY CONSEQUENCE** |
| **H. Dataset Formatting** | Question / Answer parsing | GSM8K `MathReasoningDataset` loaded text and ground-truth values correctly. | **VERIFIED (NO ISSUE)** |
| **I. Chat Template** | `apply_chat_template` presence | Added official `tokenizer.apply_chat_template()` for instruction models. | **REPAIRED (FIXED)** |
| **J. Pass@1 Evaluation** | Batched vs sample loop | Fixed sample extraction per sequence using attention mask length. | **REPAIRED (FIXED)** |

---

## Empirical Before/After Comparison

- **Phase II Pilot (`gpt2`, 16 tokens, raw text)**: **0.00% Baseline Pass@1**
- **Phase III Repaired (`Qwen2.5-0.5B-Instruct`, 256 tokens, Chat Template)**: **80.00% Baseline Pass@1** (Evidence preserved in `research/BASE_MODEL_EVALUATION_QWEN.json`).
