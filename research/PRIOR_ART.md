# PRIOR ART SURVEY & COMPARATIVE ANALYSIS

## 1. Primary References (2024–2026)

* **Shao et al. (2024)**: *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*. Introduced Group Relative Policy Optimization (GRPO).
* **Kakade et al. (2025)**: *EvoLM: Systematic Analysis of LLM Training Dynamics*. Analyzed post-training policy shifts and entropy degradation in RLHF/RL.
* **Brantley et al. (2026)**: *LLMs Can Learn to Reason Via Off-Policy RL*. Examined off-policy importance sampling and trajectory variance in reasoning models.
* **UCAS (2025)**: *Uncertainty-aware Advantage Shaping*. Applied single-token logit entropy penalties during RLHF.
* **KRPO (2026)**: *Kalman Filter Enhanced GRPO*. Used Kalman filtering to smooth group reward noise in GRPO.
* **Bounded Log Likelihood Loss (2026)**: Mitigated loss runaway on low-probability tokens during reasoning RL.

## 2. Key Differences in EAR-GRPO
Unlike UCAS (single-token entropy) or KRPO (reward observation noise filtering), EAR-GRPO explicitly estimates **epistemic uncertainty over entire multi-step reasoning rollout chains** via Monte Carlo logit variance. It dynamically rescales the group-relative advantage $\tilde{A}_i$, dampening updates on uncalibrated lucky-guess trajectories while preserving full updates on confident, structured reasoning chains.
