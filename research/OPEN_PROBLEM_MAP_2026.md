# OPEN PROBLEM MAP 2026: UNRESOLVED FRONTIERS IN CS & AI

## Overview
This map details current, unsolved scientific problems at the intersection of Reinforcement Learning, Foundation Models, Interactive Agents, World Models, and Machine Learning Reliability. Each problem is derived from an analysis of 2024–2026 literature and aligned with active research themes pursued by faculty at Harvard, Stanford, MIT, and peer top-tier CS PhD programs.

---

## 1. Foundation-Model Training & Data Curation

### Open Problem 1.1: Dynamic & Non-Stationary Data Composition during Post-Training
* **Problem**: Standard post-training (SFT / DPO / RLHF) relies on static data mixtures or fixed curricula. How can model learning dynamics (e.g., loss gradient norms per domain, token-level entropy trajectories) dynamically determine the optimal non-stationary mixture across math, code, tool-use, and general reasoning?
* **Why Unresolved**: Loss-proportional heuristics fail under multi-domain interference, where optimizing math reasoning degrades natural language instruction following or vice versa ("forgetting vs. transfer trade-off").
* **Closest Paper**: *EvoLM: Systematic Analysis of LLM Training Dynamics* (Kakade et al., 2025); *DoRA & Data Composition Scaling Laws* (2024).
* **Faculty Alignment**: Sham Kakade (Harvard), Alexander Rush (Cornell), Percy Liang (Stanford).
* **Evidence of Openness**: Recent 2025/2026 studies show static data mixtures waste 30-45% of post-training compute on saturated domains.
* **Difficulty**: Medium-High | **Compute**: Tier 2 (7B-8B models, 8x H100 hours) | **Dataset**: UltraFeedback + OpenMathInstruct + DeepSeek-Math-Data.
* **Novelty Opportunity**: Formulating post-training data scheduling as a non-stationary contextual bandit with domain interference constraints.
* **Possible Venue**: ICML / NeurIPS / TMLR.

---

## 2. Reinforcement Learning for Large Language Models

### Open Problem 2.1: Epistemic Uncertainty vs. Aleatoric Noise Decoupling in Trajectory Advantage Estimation
* **Problem**: Standard RL algorithms (PPO, GRPO) treat all sample rollout variance equally. When an LLM generates a correct answer through a low-confidence hallucination vs. a high-confidence logical derivation, GRPO assigns both identical group-relative advantages. How can we estimate policy epistemic uncertainty over reasoning trajectories to regularize RL updates?
* **Why Unresolved**: Existing uncertainty methods (UCAS, KRPO) either focus on single-token logit entropy or scalar reward noise, missing the structured semantic uncertainty of long multi-step reasoning chains.
* **Closest Paper**: *LLMs Can Learn to Reason Via Off-Policy RL* (Brantley et al., 2026); *Uncertainty-Aware Advantage Shaping (UCAS)* (2025); *Kalman-Enhanced GRPO (KRPO)* (2026).
* **Faculty Alignment**: Sham Kakade (Harvard), Kianté Brantley (Harvard), Emma Brunskill (Stanford).
* **Evidence of Openness**: ICML 2026 discussions highlight entropy collapse and reward hacking in DeepSeek-R1-style GRPO training as major unaddressed failure modes.
* **Difficulty**: High | **Compute**: Tier 1/Tier 2 (1B-7B models) | **Dataset**: GSM8K, MATH, SVAMP.
* **Novelty Opportunity**: Developing an advantage estimator scaled by policy epistemic divergence (MC-dropout / ensemble variance across reasoning rollouts) to prevent over-optimizing lucky guesses.
* **Possible Venue**: NeurIPS / ICLR / JMLR.

---

## Summary Matrix of Frontiers

| Category | Primary Unresolved Question | Top Harvard Alignment | Preferred Compute Tier |
| :--- | :--- | :--- | :--- |
| **RL for LLMs** | Decoupling epistemic trajectory uncertainty from reward noise in GRPO | Kakade & Brantley | Tier 1/2 (1.5B–7B models) |
| **Agents** | Cost-sensitive intervention thresholding under the Intervention Paradox | Brantley & Doshi-Velez | Tier 1 (1B–7B models) |
| **World Models** | Learning recoverability metrics inside generative diffusion world models | Du & Doshi-Velez | Tier 2 (Mid-size models) |
| **Data Composition** | Non-stationary contextual bandit scheduling for post-training mixtures | Kakade | Tier 2 (7B models) |
| **Reliability** | Early-warning precursor detection of reward hacking during RL | Kakade & Doshi-Velez | Tier 1/2 (1B–7B models) |
