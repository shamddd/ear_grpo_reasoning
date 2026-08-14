# Final Prior Art & Literature Audit (2026/2027)

## 1. Exhaustive Literature Mapping

| Citation & Venue | Key Formulation / Method | Overlap with Our Work | Key Distinction & Our Novelty | Threat Level |
| :--- | :--- | :--- | :--- | :---: |
| **Wang et al. (2022)** *Self-Consistency in LLMs* (ICLR) | Sample $K$ paths and select modal answer. | Utilizes answer consensus as an error proxy. | Evaluated purely as test-time decoding; does not study online policy gradient advantage weighting in RL. | **None (Foundational)** |
| **Sanyal et al. (2024)** *Length Bias in Confidence* (ACL) | Observes sequence perplexity correlates with output length. | Documents length confounding in LLM confidence. | Focuses on classification calibration; does not study reinforcement learning credit assignment or GRPO. | **None (Supporting)** |
| **Shao et al. (2024)** *DeepSeekMath / GRPO* (arXiv) | Group Relative Policy Optimization for mathematical reasoning. | Serves as our primary baseline algorithm. | Uses standard unweighted outcome rewards; does not investigate uncertainty-aware advantage regularization. | **None (Baseline)** |
| **Kadavath et al. (2022)** *Language Models Know What They Know* (arXiv) | Investigates P(True) and model self-evaluation. | Studies internal model confidence calibration. | Does not analyze interaction with policy gradient updates in RL. | **None (Conceptual)** |

---

## 2. Novelty Positioning for the Final Manuscript

* **Title**: *"When Confidence Proxies Confound Reasoning Complexity: Pitfalls of Uncertainty-Weighted Credit Assignment in Language Model Reinforcement Learning"*
* **Primary Novelty**: First comprehensive empirical and architectural audit demonstrating:
  1. Why naive hidden-state MC-dropout probes fail on zero-dropout transformer architectures.
  2. Why internal confidence proxies systematically confound mathematical derivation complexity with error.
  3. Why strong offline error predictors (e.g. self-consistency) fail to improve online reinforcement learning policy optimization over standard outcome-supervised GRPO.
