# EXPERIMENTAL PLAN & METRICS SPECIFICATION

## Experimental Setup

### Models
* **Tier 1 (Development & Sanity Check)**: `Qwen/Qwen2.5-Math-1.5B-Instruct` & `meta-llama/Llama-3.2-1B-Instruct`
* **Tier 2 (Main Empirical Benchmarking)**: `Qwen/Qwen2.5-Math-1.5B-Instruct` & `Qwen/Qwen2.5-7B-Instruct`

### Datasets
* **In-Distribution Training & Validation**: GSM8K (8.5K math word problems with step-by-step solutions).
* **Out-of-Distribution Validation**: SVAMP (1K adversarial math word problems) & GSM-Hard.

### Baselines
1. **GRPO (Standard)**: Group Relative Policy Optimization without epistemic advantage scaling ($\gamma = 0$).
2. **GRPO + Adaptive KL**: Standard GRPO with dynamic KL penalty adjustment ($\beta_{\text{adaptive}}$).
3. **UCAS (Token Entropy Regularized)**: GRPO with single-token logit entropy penalty.
4. **EAR-GRPO (Proposed)**: Epistemic Advantage Regularized GRPO ($\gamma \in [0.1, 0.5]$).

### Statistical Protocol
* **Seeds**: 5 independent random seeds (`[42, 100, 2024, 777, 999]`).
* **Hypothesis Testing**: Welch's two-sample t-test ($p < 0.05$) for Pass@1 accuracy and OOD Transfer Ratio.
