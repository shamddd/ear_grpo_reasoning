# Repository-Wide Invalidated Claims Register

## 1. Classification Taxonomy

* **VALID**: Supported by sound empirical evidence with verified statistical controls.
* **NEEDS QUALIFICATION**: Empirical observation holds, but theoretical interpretation requires correction.
* **INVALIDATED**: Empirically or mathematically refuted by forensic audit or negative controls.
* **HISTORICAL ONLY**: Early placeholder or pre-audit formulation preserved solely for auditability.

---

## 2. Exhaustive Claims Audit Register

| Claim / Terminology Phrase | Earliest Appearance | Current Status | Forensic Reason & Required Correction |
| :--- | :---: | :---: | :--- |
| *"EAR-GRPO achieves 78.41% vs 74.12% on GSM8K"* | Phase I | **INVALIDATED (HISTORICAL ONLY)** | Mock simulation artifacts; completely invalidated in Phase II. |
| *"Epistemic Uncertainty via MC-Dropout on Qwen2"* | Phase II–IV | **INVALIDATED** | Qwen2 has `attention_dropout=0.0` and 0 dropout layers. Passes were deterministic. |
| *"EAR-GRPO solves entropy collapse"* | Phase I–II | **INVALIDATED** | Policy entropy retention without task accuracy gains does not constitute useful exploration. |
| *"Trajectory-specific uncertainty dampens noisy gradients"* | Phase I–IV | **INVALIDATED** | True pairing performed identically to baseline ($16.67\%$) and below permuted control ($22.22\%$). |
| *"Permuted Control Paradox proves permutation is superior"* | Phase IV | **NEEDS QUALIFICATION** | $n=6$ pilot artifact; must be validated on large $\ge 100$ held-out sets before claiming algorithmic utility. |
| *"EAR-GRPO definitively refutes epistemic RL"* | Phase IV Verdict | **NEEDS QUALIFICATION** | Falsifies the specific deterministic proxy, not theoretical epistemic uncertainty in general. |
| *"Tuned Standard GRPO achieves 75.00% on GSM8K val"* | Phase IV Tuning | **VALID** | Verified real-model tuning run with `lr=1e-5, kl_beta=0.04` on frozen validation split. |
| *"Untouched Base Model achieves 80.00% on 200 GSM8K items"* | Phase III Audit | **VALID** | Replicated on 200 held-out examples with 256-token ChatML generation. |
| *"Standard-GRPO vs EAR-GRPO compute in GPU-hours"* | Phase IV Table | **INVALIDATED (CORRECTED)** | Ran on CPU (8 threads); corrected to CPU wall-clock seconds / minutes in `COMPUTE_ACCOUNTING_CORRECTION.md`. |
