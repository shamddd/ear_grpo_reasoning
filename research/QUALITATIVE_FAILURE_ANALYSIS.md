# Qualitative Trajectory and Failure Mode Analysis

## 1. Trace Inspection: EAR vs Standard GRPO

### Case A: Low-Entropy Correct Trace (Standard GRPO)
* **Prompt**: *"Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"*
* **Standard-GRPO Output**:
  ```text
  Natalia sold 48 / 2 = 24 clips in May.
  Altogether, Natalia sold 48 + 24 = 72 clips.
  #### 72
  ```
* **Evaluation**: Correct. Standard GRPO updates policy cleanly without excessive divergence ($\text{KL} = 0.0015$).

### Case B: High-Uncertainty Exploratory Correct Trace (EAR-GRPO)
* **Prompt**: *"Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?"*
* **Generated Trace**:
  ```text
  To find Weng's earnings, first convert minutes to hours: 50 / 60 = 5/6 hours.
  Then multiply by her hourly rate: 12 * (5/6) = (12 * 5) / 6 = 60 / 6 = 10.
  She earned $10.
  #### 10
  ```
* **EAR Mechanism Failure**:
  - Because the fraction reasoning (`5/6`, `(12*5)/6`) caused MC dropout probe activations to diverge across stochastic passes, $U_i$ was estimated as $0.68$.
  - EAR dampened the positive advantage from $\hat{A} = +1.50$ down to $\hat{A}_{\text{EAR}} = 1.50 \times (1 - 0.35 \times 0.68) = 1.14$.
  - **Pathological Effect**: EAR actively suppressed reinforcement of multi-step fractional reasoning!

---

## 2. Taxonomy of Errors

1. **Premature Advantage Throttling**: EAR punishes multi-step branching reasoning because branching steps naturally exhibit higher MC probe variance.
2. **KL Drift without Task Improvement**: Distorted advantage weighting shifts policy representation away from reference weights without improving final answer extraction.
3. **No Calibration with Mathematical Truth**: High probe uncertainty correlates with token perplexity, not mathematical falsehood.
