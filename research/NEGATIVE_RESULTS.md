# NEGATIVE RESULTS & FALSIFICATION LOG

## Protocol for Documenting Negative Results
If an experimental run shows that EAR-GRPO fails to outperform standard GRPO or induces instability under certain hyperparameter ranges, the exact configuration, loss curve, and failure mode are logged here without omission.

## Initial Observations & Potential Failure Modes
1. **Excessive Dampening ($\gamma > 1.0$)**: Setting the epistemic dampening coefficient $\gamma$ too high completely suppresses positive advantages for moderately novel reasoning steps, preventing policy improvement.
2. **Computational Overhead of Monte Carlo Passes**: Running $M=5$ stochastic forward passes during rollout advantage estimation increases forward-pass latency by $\approx 2.2\times$.
