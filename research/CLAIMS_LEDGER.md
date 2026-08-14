# CLAIMS LEDGER

This document tracks scientific claims made in the paper and connects each claim directly to empirical evidence from experiments.

| Claim ID | Claim Description | Target Metric / Result | Supporting Evidence / Table | Verification Status |
|---|---|---|---|---|
| C1 | EAR-GRPO prevents policy entropy collapse during post-training RL on GSM8K. | Trajectory Entropy $H(\pi_\theta)$ drops $<20\%$ vs $>60\%$ in GRPO | Table 1 / Figure 2 | Pending Main Experiments |
| C2 | EAR-GRPO improves out-of-distribution math reasoning generalization on SVAMP. | Pass@1 SVAMP transfer ratio increases by $\ge 5.0\text{pp}$ | Table 2 | Pending Main Experiments |
| C3 | Trajectory Monte-Carlo variance outperforms single-token logit entropy as a regularizer. | Ablation Pass@1 and Loss Variance comparison | Table 3 | Pending Ablations |
