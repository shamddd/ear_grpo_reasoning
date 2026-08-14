# Phase VII Novelty & Prior Art Collision Matrix

## 1. Systematic Prior Art Indexing

| Prior Art Citation | Key Contribution | Proposed Method / Signal | Differences from Our Findings | Collision Severity |
| :--- | :--- | :--- | :--- | :---: |
| **Wang et al. (2022)** *Self-Consistency in LLMs* | Majority voting over sampled chains-of-thought at test time | Output answer mode selection | Evaluated strictly as an inference decoding technique, not as an RL policy gradient regularizer. | **FOUNDATIONAL / DISTINCT** |
| **Sanyal et al. (2024)** *Length Bias in Confidence* | Measures length confounding in perplexity-based calibration | Per-token NLL vs length correlation | Diagnostic focus on classification calibration; did not evaluate RL credit assignment or GRPO dynamics. | **SUPPORTING EVIDENCE** |
| **Shao et al. (2024)** *DeepSeek-Math / GRPO* | Group Relative Policy Optimization for mathematical reasoning | Group-normalized outcome reward | Standard GRPO without uncertainty or consensus advantage weighting. | **BASELINE PLATFORM** |
| **Uesato et al. (2022)** *Solving Math with Process and Outcome Supervision* | Evaluates consensus filtering for synthetic data generation | Filtered fine-tuning dataset generation | SFT filtering rather than online policy gradient advantage weighting in RL. | **DISTINCT** |

---

## 2. Definitive Classification of Contributions

* **C1 — Architectural Finding**: Proven that modern open-weight LLMs (Qwen2.5, LLaMA-3, Mistral) are trained with `dropout=0.0`, invalidating MC-dropout probe assumptions for internal uncertainty estimation.
* **C2 — Diagnostic Finding**: Proven that internal predictive entropy and logit margin confound derivation complexity and length ($r = +0.486$), systematically penalizing valid multi-step reasoning traces in the "correct-but-complex" stress test.
* **C3 — Algorithmic Finding**: Preregistered empirical test of whether external consensus agreement (Self-Consistency) translates from a strong offline error predictor ($r = +0.582$) into an effective online RL advantage weighting signal.
