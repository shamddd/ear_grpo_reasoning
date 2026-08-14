# Uncertainty Implementation Truth: Code-Level Forensic Audit

## 1. Executive Summary

Discrepancies existed between earlier prose descriptions ("Monte Carlo rollout logit variance") and subsequent reports ("MC dropout probe activations"). This audit inspects the exact codebase to establish the ground truth.

---

## 2. Code Inspection & Ground Truth

### A. Location in Source Code
* **Class**: `EpistemicUncertaintyProbe`
* **File**: [`src/models/epistemic_probe.py`](./src/models/epistemic_probe.py)
* **Method**: `compute_epistemic_variance(policy, input_ids, prompt_lengths)`

### B. Exact Implemented Formulation
1. **Stochastic Mechanism**:
   The probe executes $M = 2$ or $M = 3$ forward passes over the identical input sequence $X$ with `mc_dropout=True`:
   ```python
   for _ in range(self.num_mc_samples):
       logits = policy(input_ids, mc_dropout=True)
       log_probs = policy.compute_completion_log_probs(input_ids, prompt_lengths, logits=logits)
       mc_log_probs.append(log_probs.unsqueeze(0))
   ```
2. **Aggregated Quantity**:
   The sequence log-probability $\log P(y \mid x) = \sum_{t=1}^{T} \log \pi_\theta(y_t \mid x, y_{<t})$ is computed for each stochastic forward pass.
3. **Variance Computation**:
   ```python
   stacked = torch.cat(mc_log_probs, dim=0) # Shape: (M, batch_size)
   epistemic_variance = torch.var(stacked, dim=0, unbiased=True) # Shape: (batch_size,)
   ```

---

## 3. Structural Architectural Finding

In [`src/models/policy.py`](./src/models/policy.py):
* `mc_dropout=True` executes:
  ```python
  for m in self.model.modules():
      if isinstance(m, nn.Dropout):
          m.train()
  ```
* **Crucial Finding**: Standard modern pretrained models (including Qwen2.5 and LLaMA-3) are trained with `dropout=0.0` and have no active interior `nn.Dropout` modules inside their self-attention or MLP blocks.
* **Consequence**: When evaluating models with zero native dropout layers, the stochastic variance across identical forward passes is negligible unless synthetic dropout is injected into the attention layers.
* **Conclusion**: The phrase *"Monte Carlo Dropout probe on sequence log-probabilities"* is the technically exact description of the code, and this structural limitation explains why true uncertainty performed indistinguishably from low-amplitude baseline noise.
