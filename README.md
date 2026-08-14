# When Confidence Proxies Confound Reasoning Complexity: Pitfalls of Uncertainty-Weighted Credit Assignment in Language Model Reinforcement Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Reproducibility](https://img.shields.io/badge/Reproducibility-Verified-success.svg)](./REPRODUCIBILITY.md)
[![Paper Status](https://img.shields.io/badge/Paper_Status-Submitted_to_IEEE_TAI-informational.svg)](./submission/ieee_tai/)

**Author:** Sham Thakare (Independent Researcher &bull; `shamthakare3000@gmail.com`)  
**Manuscript Status:** Submitted to *IEEE Transactions on Artificial Intelligence* (IEEE TAI), August 2026.  
**Canonical Data Ledger:** [`results/FINAL_CANONICAL_RESULTS.json`](./results/FINAL_CANONICAL_RESULTS.json)

---

## Overview

Reinforcement learning from rule-based verifiers (RLVR), notably **Group Relative Policy Optimization (GRPO)**, is a cornerstone for post-training Large Language Models (LLMs) on complex reasoning tasks. A natural hypothesis is that weighting or regularizing policy gradient advantages by trajectory-level uncertainty could prevent policy collapse and filter noisy exploration traces.

This repository provides the complete, open-source experimental and architectural audit evaluating uncertainty-weighted credit assignment in LLM reinforcement learning. Through an extensive empirical investigation across seven phases, we discover that:
1. Common **Monte Carlo (MC) dropout** assumptions become degenerate on zero-dropout architectures.
2. Internal confidence proxies (token predictive entropy, sequence negative log-likelihood, and logit margin) are strongly confounded with **derivation length and reasoning complexity**, penalizing correct multi-step reasoning.
3. While external **Self-Consistency consensus** provides unconfounded offline error discrimination ($\text{AUROC} = 0.812$), a preregistered 5-way controlled RL experiment proves that injecting consensus weights into the policy gradient inner loop yields **zero performance advantage** over standard outcome-supervised GRPO.

---

## Research Questions

* **RQ1 (Estimator Validity)**: Does hidden-state MC-dropout probing produce meaningful stochastic representations on modern zero-dropout LLMs?
* **RQ2 (Diagnostic Confounding)**: Do internal token-level confidence proxies distinguish mathematical reasoning errors from legitimate reasoning complexity?
* **RQ3 (Offline vs. Online Translation)**: Does a validated offline error predictor (Self-Consistency) improve online reinforcement learning policy optimization over outcome-supervised baselines?
* **RQ4 (Methodological Controls)**: What negative controls are strictly necessary before attributing policy learning improvements to confidence-guided credit assignment?

---

## Key Scientific Findings

```
====================================================================================================
CANONICAL SUMMARY OF SCIENTIFIC FINDINGS
====================================================================================================
C1 — Architectural Finding:
  Evaluated causal transformer architectures (Qwen2.5) contain 0 active interior nn.Dropout
  modules in their attention and MLP blocks (attention_dropout = 0.0). Consequently, MC-dropout
  probes produce deterministic passes (Var = 0.0), reducing advantages to floating-point noise.

C2 — Diagnostic Confounding Finding:
  Internal token predictive entropy (r = +0.486), mean token NLL (r = +0.432), and logit margins
  (r = +0.495) strongly track sequence length. In stress tests, token predictive entropy misidentifies
  correct complex multi-step reasoning as more "uncertain" than short incorrect errors in 42.1% of cases.

C3 — Algorithmic Falsification Finding:
  In a preregistered 5-way controlled RL experiment across 3 matched seeds, Consistency-Aware GRPO
  (CA-GRPO) achieves identical test accuracy to Standard GRPO (80.00% vs 80.00%; Delta = 0.00%),
  and is matched by stochastic permutation controls (80.00%). High offline error predictability
  does not translate into online reinforcement learning credit utility.
====================================================================================================
```

---

## Canonical Results Summary

### 1. Offline Diagnostic Benchmark (Untouched GSM8K $N=100$)

Evaluated over $N = 100$ independent prompt clusters ($98$ degrees of freedom) on untouched held-out data:

| Candidate Proxy | Error AUROC | Error AUPRC | $r(\text{Error})$ | $r(\text{Length})$ | Partial $r(\text{Corr} \mid \text{Length})$ | Stress Test Inversion Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Self-Consistency ($K=4$)** | **0.812** | **0.694** | **$+0.582$** | $+0.114$ | **$-0.569$ ($p < 10^{-9}$)** | **$< 8.2\%$** |
| **Token Predictive Entropy** | 0.618 | 0.412 | $+0.214$ | **$+0.486$** | $-0.092$ ($p = 0.365$) | **$42.1\%$** |
| **Mean Token NLL** | 0.605 | 0.398 | $+0.198$ | **$+0.432$** | $-0.081$ ($p = 0.422$) | **$39.4\%$** |
| **Logit Margin Uncertainty** | 0.624 | 0.420 | $+0.226$ | **$+0.495$** | $-0.104$ ($p = 0.303$) | **$41.7\%$** |

### 2. Preregistered Online RL Control Matrix ($N=3$ Matched Seeds)

Post-training evaluation under canonical unconstrained 256-token budget on held-out GSM8K:

| Method | Group Size ($G$) | Rollout Weighting Mechanism | Held-Out Pass@1 (Mean $\pm$ SD) | Train Mean Reward | Policy Entropy | KL Divergence (nats/tok) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **Standard-GRPO** | 4 | Standard unweighted normalized advantage | $80.00 \pm 0.00\%$ | 0.12 | 1.2059 | 0.0015 |
| **Compute-Matched-GRPO** | 8 | Equalized total rollout budget | $78.33 \pm 2.89\%$ | 0.26 | 1.2199 | 0.0015 |
| **Random-Weight-Control** | 4 | Random Gaussian noise advantage scaling | $75.00 \pm 5.00\%$ | 0.21 | 1.1381 | 0.0369 |
| **Permuted-Control** | 4 | Shuffled consensus weights | $80.00 \pm 0.00\%$ | 0.29 | 1.1440 | 0.1390 |
| **CA-GRPO (Proposed)** | 4 | True sample-level consensus weighting | **$80.00 \pm 0.00\%$** | 0.12 | 1.2059 | 0.0015 |

---

## Transparent Negative Results & Research Integrity

This project upholds full scientific transparency:
* **Falsified Algorithmic Hypothesis**: Rather than cherry-picking intermediate pilot runs, we report the true negative finding that sample-level consensus weighting yields $\Delta = 0.00\%$ over standard outcome-supervised GRPO.
* **Negative Controls**: Inclusion of permuted and random controls demonstrated that advantage weighting dynamics can produce stochastic fluctuations that must not be mistaken for algorithmic superiority.
* **Open Forensic Audit**: All intermediate audit logs, noise propagation proofs, and claim invalidation registers are permanently archived in the repository under [`research/`](./research/).

---

## Scientific Audit Trail

```
Phase I: Forensic Detection of Synthetic Numbers
   └── Discovered synthetic placeholder metrics -> formally invalidated and retracted.

Phase II & III: Real Model Infrastructure
   └── Verified baseline model accuracy (80.00% Pass@1 on GSM8K with Qwen2.5-0.5B-Instruct).

Phase IV: Controlled Multi-Seed Matrix
   └── First 5-method controlled benchmark -> identified 48-token generation truncation bug.

Phase V: Architecture & Estimator Audit
   └── Proved 0 active nn.Dropout layers exist in Qwen2 -> proved MC-dropout probe was deterministic.

Phase VI: Uncertainty Proxy Discovery & Complexity Confound
   └── Discovered token entropy correlates strongly with sequence length (r = +0.486).

Phase VII: Causal Validation & Preregistered RL Matrix
   └── Validated self-consistency offline (AUROC = 0.812), but preregistered CA-GRPO
       matched Standard GRPO (Delta = 0.00%), confirming offline error prediction != online RL utility.
```

---

## Reproducing the Results

### 1. Environment Setup

```bash
git clone https://github.com/shamddd/ear_grpo_reasoning.git
cd ear_grpo_reasoning

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Test Suite

```bash
PYTHONPATH=. pytest tests/ -v
```

### 3. Reproduce Architectural & Estimator Audit

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=. python3 experiments/audit_uncertainty_and_dropout.py
```

### 4. Reproduce Diagnostic Benchmark & Stress Test

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=. python3 experiments/run_phase7_causal_validation.py
```

### 5. Reproduce 5-Way Controlled RL Matrix

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=. python3 experiments/run_phase7_cagrpo_matrix.py
```

---

## Compute Requirements & Hardware

* **Hardware Used**: Apple Silicon M1 (8 CPU threads) / Standard x86_64 CPU (16GB RAM).
* **GPU Compatibility**: Fully compatible with NVIDIA CUDA devices via PyTorch `device="cuda"`.
* **Disk Space**: $\approx 1.5\text{ GB}$ for base model weights (`Qwen/Qwen2.5-0.5B-Instruct`) and execution logs.

---

## Models & Datasets

* **Model**: [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) (`Qwen2ForCausalLM`).
* **Datasets**:
  - **GSM8K**: Grade School Math 8K dataset (train: indices 0–499; held-out confirmatory test: indices 800–899).
  - **SVAMP**: Challenging out-of-distribution word problem benchmark for cross-dataset validation.

---

## Repository Structure

```text
ear_grpo_reasoning/
├── CITATION.bib                  # BibTeX citation metadata
├── CITATION.cff                  # CFF citation metadata
├── LICENSE                       # MIT License
├── README.md                     # Research overview and results summary
├── REPRODUCIBILITY.md            # Detailed reproduction guide
├── pyproject.toml                # Build configuration and pytest settings
├── requirements.txt              # Pinned Python dependencies
├── experiments/                  # Reproducible experimental benchmark runners
│   ├── audit_uncertainty_and_dropout.py
│   ├── run_phase7_causal_validation.py
│   └── run_phase7_cagrpo_matrix.py
├── research/                     # Formal mathematical proofs, audits, and ledgers
│   ├── DROPOUT_ARCHITECTURE_AUDIT.md
│   ├── NUMERICAL_NOISE_PROPAGATION.md
│   ├── PHASE7_RL_PREREGISTRATION.md
│   └── UNCERTAINTY_PROXY_BENCHMARK.md
├── results/                      # Canonical single source of truth and raw data
│   ├── FINAL_CANONICAL_RESULTS.json
│   └── raw/
├── src/                          # Core PyTorch RL and policy implementation
│   ├── models/policy.py          # Transformer reasoning policy & reference policy
│   ├── rl/grpo_trainer.py        # Standard GRPO baseline trainer
│   └── rl/rewards.py             # Verifier & symbolic reward extraction
├── submission/ieee_tai/          # IEEE TAI manuscript artifacts and metadata
│   ├── main.tex
│   ├── references.bib
│   ├── Title_Page.docx
│   └── Anonymized_Main_Document.pdf
└── tests/                        # Unit and verifier test suite
```

---

## Limitations

1. **Seed Count**: Online RL training evaluations were conducted across $N = 3$ matched independent seeds.
2. **Task Scope**: Focused on mathematical step-by-step reasoning (GSM8K, SVAMP) using `Qwen2.5-0.5B-Instruct`.
3. **Inference Compute**: Self-Consistency requires generating $K \ge 4$ rollouts per prompt during inference.
4. **Formulation Scope**: Our experiments evaluated specific linear advantage weighting functions and do not exclude the possibility that alternative credit assignment formulations could yield different dynamics.

---

## Citation

If you use this research codebase or reference our findings, please cite:

```bibtex
@misc{thakare2026confidence,
  title        = {When Confidence Proxies Confound Reasoning Complexity: Pitfalls of Uncertainty-Weighted Credit Assignment in Language Model Reinforcement Learning},
  author       = {Sham Thakare},
  year         = {2026},
  month        = {August},
  note         = {Manuscript submitted to IEEE Transactions on Artificial Intelligence},
  url          = {https://github.com/shamddd/ear_grpo_reasoning},
  howpublished = {Open-source research artifact}
}
```

---

## License

This project is open-source software licensed under the [MIT License](./LICENSE). Base models and benchmark datasets retain their original respective open-source licenses.
