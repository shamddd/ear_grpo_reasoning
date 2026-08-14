# KL Metric Audit & Normalization Discrepancy

## 1. Executive Summary

In Phase IV, a severe discrepancy was observed between the recorded KL divergence values:
* **Standard-GRPO**: $\text{KL} \approx 0.0015$
* **EAR-GRPO**: $\text{KL} \approx 9.9562$
* **Permuted-Control**: $\text{KL} \approx 6.6752$
* **Random-Control (Seed 2024)**: $\text{KL} \approx 5.3172$

---

## 2. Code Inspection & Audit

In both `GRPOTrainer` ([`src/rl/grpo_trainer.py:49-62`](./src/rl/grpo_trainer.py#L49-L62)) and `EARGRPOTrainer` ([`src/rl/ear_grpo_trainer.py:87-100`](./src/rl/ear_grpo_trainer.py#L87-L100)):
* `log_probs` is computed via `compute_completion_log_probs(input_ids, prompt_lengths, logits=logits)`.
* This returns the **unnormalized sum** of sequence token log-probabilities:
  $$\log \pi(y \mid x) = \sum_{t=1}^{T} \log \pi(y_t \mid x, y_{<t})$$
* KL divergence is estimated as:
  ```python
  kl_div = torch.mean(log_probs - ref_log_probs)
  ```

---

## 3. Mathematical Analysis of the Discrepancy

1. **Length Aggregation vs Token Averaging**:
   Because `log_probs` is a sequence-level sum over $T \approx 48$ to $230$ tokens, any token-level probability shift $\delta$ accumulates across the sequence:
   $$\text{Sequence-KL} \approx \sum_{t=1}^T \delta_t = T \cdot \bar{\delta}$$
2. **Why Standard GRPO Remained at 0.0015**:
   Under standard GRPO with balanced group advantages ($\sum_i \hat{A}_i = 0$), positive and negative gradient pushes partially cancel out, keeping the policy tightly tethered to the reference policy.
3. **Why EAR-GRPO Shifted to ~9.95**:
   Under EAR-GRPO, multiplying advantages by non-linear dampening factors $\exp(-\gamma \cdot \text{norm}(U_i))$ breaks the zero-sum mean symmetry of group advantages ($\sum_i \tilde{A}_i \ne 0$). As a result, every gradient step applies a non-zero mean directional push on token logits, causing the sequence log-probability difference to accumulate rapidly over 48–200 tokens ($9.95 / 48 \approx 0.20$ nats per token).
4. **Random-Control Seed 2024 Outlier**:
   Random-Control seed 2024 generated a multi-step completion where token log-probability shifts accumulated across 48 tokens, reaching $\text{KL} = 5.3172$.

---

## 4. Methodological Conclusion

* The KL divergence values reflect an actual optimization dynamic: EAR's asymmetrical advantage dampening acts as an uncentered directional force on the policy distribution, causing rapid divergence from $\pi_{\text{ref}}$ without yielding mathematical accuracy improvements.
* In future reporting, KL must be reported as **per-token average KL** ($\text{nats/token}$) rather than unnormalized sequence sum to eliminate sequence-length confounding.
