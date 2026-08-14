# FORMAL RESEARCH HYPOTHESES (EAR-GRPO)

## Core Scientific Hypotheses

### Hypothesis H1 (Sample Efficiency & Entropy Retention)
* **Statement**: Epistemic Advantage Regularization (EAR-GRPO) will achieve a higher or equal Pass@1 accuracy on mathematical reasoning benchmarks (GSM8K) compared to standard GRPO and PPO-KL while maintaining significantly higher token-level entropy (preventing premature entropy collapse).
* **Metric**: Pass@1 Accuracy (%) vs Training Epochs; Average Trajectory Policy Entropy $H(\pi_\theta)$.
* **Falsification Threshold**: If EAR-GRPO exhibits identical or lower token entropy drop and no improvement in Pass@1 over standard GRPO with tuned KL penalty ($\beta \in [0.01, 0.05]$) across 5 random seeds, H1 is rejected.

### Hypothesis H2 (Out-of-Distribution Generalization)
* **Statement**: Policies trained with EAR-GRPO will exhibit superior out-of-distribution (OOD) reasoning performance on distribution-shifted math datasets (SVAMP, GSM-Hard) relative to in-distribution performance (GSM8K) compared to policies trained with unregularized GRPO.
* **Metric**: OOD Transfer Ratio $\text{TR} = \frac{\text{Pass@1}_{\text{SVAMP}}}{\text{Pass@1}_{\text{GSM8K}}}$.
* **Falsification Threshold**: If the transfer ratio of EAR-GRPO is not higher than standard GRPO by at least 5.0 percentage points with statistical significance ($p < 0.05$), H2 is rejected.

### Hypothesis H3 (Ablation of Epistemic Signals)
* **Statement**: Trajectory-level ensemble logit variance across stochastic rollout passes provides a superior regularizing signal compared to single-token entropy penalties (UCAS) because reasoning uncertainty is structured across multi-step chains rather than isolated to individual tokens.
* **Metric**: Validation Loss Variance and OOD Accuracy across ablation variants (EAR-GRPO vs Single-Token Entropy vs Unregularized GRPO).
