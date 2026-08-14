# NOVELTY AUDIT 2.0: PRIOR ART & METHODOLOGICAL DISTINCTION

## 1. Deep 2025–2026 Literature Search

| Paper / Framework | Venue / Year | Primary Mechanism | Uncertainty Type | RL / Alignment | Trajectory-Level? | EAR-GRPO Key Difference |
|---|---|---|---|---|---|---|
| **BV-Blend** | arXiv 2026 | SEM proxy weighting for historical group baseline blending | Standard Error of Mean | GRPO | No | BV-Blend blends prompt-local baselines; EAR-GRPO regularizes rollout advantages via Monte Carlo dropout logit variance. |
| **MERCI** | NeurIPS 2025 | Count-based intrinsic reward using Coin Flipping Network | Epistemic / Pseudo-count | GRPO / Exploration | Yes | MERCI adds intrinsic rewards for exploration; EAR-GRPO dampens positive advantages on uncalibrated rollouts to prevent entropy collapse. |
| **DGPO** | ICLR 2026 | Entropy-gating scaling sequence-level advantages | Policy Logit Entropy | Policy Optimization | Yes | DGPO uses single-pass entropy; EAR-GRPO measures multi-pass Monte Carlo logit variance across dropout passes. |
| **UCAS** | ICLR 2025 | Single-token logit certainty penalty | Token Certainty | PPO / RLHF | No | Single-token penalty vs. EAR-GRPO trajectory-level MC ensemble variance. |
| **KRPO** | ICML 2026 | Kalman filtering of scalar reward observations | Reward Observation Noise | GRPO | No | Filters scalar reward noise; EAR-GRPO measures policy internal epistemic variance. |

---

## 2. Novelty & Claim Defense Statement

EAR-GRPO is distinct from existing 2025–2026 methods in three fundamental ways:
1. **Uncertainty Estimator**: Uses Monte Carlo stochastic dropout forward passes ($M \ge 3$) to measure per-token logit variance across multi-pass reasoning rollouts.
2. **Optimization Target**: Applies exponential dampening $\tilde{A}_i = A_i \exp(-\lambda u_i / \sigma_u)$ directly to group-relative advantage clipping bounds in GRPO.
3. **Failure Mode Mitigated**: Targeted specifically at preventing **entropy collapse** and verifier shortcut exploitation on correct-answer lucky guesses.
