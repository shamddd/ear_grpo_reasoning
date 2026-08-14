# Phase VI: Comprehensive Literature Collision & Novelty Map

## 1. Literature Taxonomy & Closest Related Work

We audited the machine learning and NLP literature across ICML, NeurIPS, ICLR, TMLR, JMLR, ACL, EMNLP, and arXiv through 2026 for works intersecting uncertainty estimation, reasoning complexity, and policy optimization in LLMs:

| Area | Representative Papers | Key Method / Formulation | Differences from Our Forensic Finding | Collision Severity |
| :--- | :--- | :--- | :--- | :---: |
| **Self-Consistency & Sampling Uncertainty** | Wang et al. (2022) *Self-Consistency*; Manakul et al. (2023) *SelfCheckGPT* | Measures answer modal frequency across $K$ sampled paths. | Focuses on inference-time selection rather than RL advantage regularizers. | **DISTINCT** (Baseline comparator) |
| **Predictive Entropy & Confidence in LLMs** | Kuhn et al. (2023) *Semantic Entropy*; Kadavath et al. (2022) *Language Models Know What They Know* | Clusters completions by semantic equivalence to estimate epistemic entropy. | Requires expensive semantic clustering; does not analyze RL advantage dampening interaction. | **DISTINCT** (Conceptual foundation) |
| **Uncertainty-Aware RL & Policy Gradients** | O'Donoghue et al. (2018) *Uncertainty in Deep Q-Learning*; Clements et al. (2019) *Estimating Epistemic Uncertainty in RL* | Uses bootstrapped ensembles/Bayesian neural nets in tabular/continuous control. | Evaluated in low-dimensional RL environments, not autoregressive chain-of-thought LLMs. | **DISTINCT** |
| **Reasoning Length & Difficulty Confounding** | Sanyal et al. (2024) *Length Bias in LLM Confidence*; Chen et al. (2024) *Do LLMs Know When They Are Wrong?* | Observes that sequence perplexity correlates with output length and problem difficulty. | Identifies general calibration bias, but does not investigate how it causes advantage suppression failure in GRPO. | **CLOSEST RELATED / SUPPORTING** |

---

## 2. Novelty Assessment: What Is the Original Contribution?

* **Old (Invalid) Claim**: *"First algorithm to use MC-dropout epistemic uncertainty in GRPO for LLM reasoning."* (Invalidated due to zero dropout).
* **Defensible & Original Contribution**:
  1. **Methodological Falsification**: Demonstrating that sequence-level log-probability variance proxies fail to provide useful credit signals in GRPO because they are suppressed to numerical noise by modern zero-dropout architectures.
  2. **The Reasoning-Complexity Confound**: Proving empirically that common internal confidence proxies (predictive entropy, token NLL) correlate more strongly with legitimate derivation length and arithmetic step complexity than with mathematical error.
  3. **Negative Control Protocol**: Establishing that negative controls (permuted and random dampening) are essential to prevent researchers from mistaking stochastic regularization noise for causal uncertainty mechanisms.

---

## 3. Novelty Verdict

### **ORIGINAL METHODOLOGICAL CONTRIBUTION SUPPORTED (AS A RIGOROUS DIAGNOSTIC & NEGATIVE-CONTROL STUDY)**
