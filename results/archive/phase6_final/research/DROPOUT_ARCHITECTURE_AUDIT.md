# Dropout Architecture & Estimator Determinism Audit

## 1. Architectural Examination of Target LLM

To determine why the `EpistemicUncertaintyProbe` exhibited near-zero stochastic variation, we audited the exact internal layer topology of `Qwen/Qwen2.5-0.5B-Instruct` (`Qwen2ForCausalLM`).

### Findings:
* **Interior `nn.Dropout` Modules**: Exactly **0** active `nn.Dropout` modules exist inside the self-attention, MLP, or layernorm layers of `Qwen2ForCausalLM`.
* **Configuration Parameters**:
  - `config.attention_dropout = 0.0`
  - `config.hidden_dropout = None` (Not implemented in Qwen2 architecture)
  - `config.classifier_dropout = None`
* **Wrapper Module**: The `TransformerReasoningPolicy` class instantiated an unused top-level `self.dropout = nn.Dropout(p=0.1)` attribute, but this attribute was not placed within the forward computation graph of the transformer layers.

---

## 2. Forensic Consequence for EAR-GRPO-v1

1. **Deterministic Forward Passes**: When `EpistemicUncertaintyProbe` executed `policy(input_ids, mc_dropout=True)`, the call to `m.train()` traversed `pol.model.modules()` and found no dropout layers to activate.
2. **Empirical Measurement**: Repeated forward passes over identical inputs produced **zero token logit variance** ($\text{Var}(\log P) = 0.0000000000$), yielding a maximum absolute logit difference of $\Delta = 0.0$.
3. **Formal Invalidation**:
   > [!IMPORTANT]
   > **EAR-GRPO-v1 did NOT measure epistemic uncertainty**. The probe was mathematically deterministic. The small non-zero variance values recorded in earlier summaries were numerical floating-point precision noise ($\approx 10^{-12}$) magnified by division by $\epsilon$.

---

## 3. Methodological Nomenclature Correction

* **Old Terminology**: *"Epistemic Advantage Regularization"* / *"Monte Carlo Dropout Epistemic Uncertainty"*.
* **Corrected Terminology**: *"EAR-GRPO-v1 (Deterministic Logprob Proxy / Zero-Dropout Baseline)"*.
* **Scientific Reality**: The Phase IV and Phase V experiments evaluate an un-damped baseline proxy, not genuine epistemic uncertainty.
