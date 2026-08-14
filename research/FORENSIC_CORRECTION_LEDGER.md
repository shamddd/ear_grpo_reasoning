# Forensic Correction Ledger: Complete Chronological Audit

## 1. Overview of Forensic Audits & Methodological Corrections

This ledger preserves the chronological record of all theoretical assumptions, forensic discoveries, root-cause analyses, and empirical corrections throughout the research trajectory.

---

## 2. Chronological Correction Records

### Phase I Audit: Discovery of Synthetic & Placeholder Results
* **Original Assumption**: The initial project state claimed empirical reinforcement learning experiments showing EAR-GRPO achieving $78.41\%$ vs Standard GRPO $74.12\%$ ($p = 0.0011$) on GSM8K.
* **Contradicting Evidence**: Forensic code inspection revealed mock mathematical generation scripts simulating results rather than executing real PyTorch neural network forward/backward graphs.
* **Root Cause**: Premature placeholder script generation before real model infrastructure was assembled.
* **Correction**: Completely rejected and invalidated all Phase I numbers. Declared Phase I non-empirical.
* **Scientific Consequence**: Halted all claims of algorithmic superiority. Mandated execution with real transformer weights.
* **Historical Artifact Status**: Preserved in archive as `INVALIDATED_HISTORICAL_ONLY`.

---

### Phase II Audit: Real Model Pipeline & Zero-Accuracy Reality
* **Original Assumption**: Instantiating untuned `gpt2` with real causal LM rollouts would demonstrate learning dynamics.
* **Contradicting Evidence**: All methods across all seeds scored $0.00\%$ Pass@1 on GSM8K.
* **Root Cause**: Untuned base model combined with an arbitrary 16-token generation cutoff truncated reasoning traces before numbers could be generated.
* **Correction**: Upgraded base model to instruction-tuned `Qwen/Qwen2.5-0.5B-Instruct` and expanded token budget to 256 tokens.
* **Scientific Consequence**: Transitioned to real-model learning while prohibiting any publication claims based on 0% accuracy runs.
* **Historical Artifact Status**: Preserved in `results/raw/phase2/`.

---

### Phase III Audit: Verification Harness & Baseline Competence
* **Original Assumption**: Answer parsing was suspected to be losing valid mathematical answers.
* **Contradicting Evidence**: Verifier passed 30/30 unit tests, and the base model achieved $80.00\%$ Pass@1 ($160/200$) when allowed full 256-token ChatML generation.
* **Root Cause**: Answer extraction regex required specific formatting (`#### <answer>`), which was resolved by standardizing the system prompt.
* **Correction**: Frozen evaluation pipeline documented in `research/PHASE3_PIPELINE_FREEZE.md`.

---

### Phase IV Audit: Controlled 5-Method Matrix & Permuted Control Falsification
* **Original Assumption**: Trajectory-specific epistemic uncertainty advantage dampening would outperform standard and compute-matched GRPO.
* **Contradicting Evidence**:
  - EAR-GRPO scored $16.67 \pm 0.00\%$ Pass@1 (identical to Standard-GRPO $16.67\%$).
  - Permuted-Control (randomly shuffling uncertainty values among rollouts) scored $22.22 \pm 9.62\%$.
* **Root Cause**: Pilot evaluation was conducted with `max_new_tokens=48` on $n=6$ items (5 of 6 items truncated mid-sentence).
* **Correction**: Implemented 100-item confirmatory benchmark with full generation token budgets.

---

### Phase V Audit: MC-Dropout Estimator Determinism on Qwen2 Architecture
* **Original Assumption**: Repeated forward passes with `mc_dropout=True` estimated epistemic uncertainty via Monte Carlo dropout.
* **Contradicting Evidence**:
  - `Qwen2ForCausalLM` contains **0 active `nn.Dropout` modules** in its attention and MLP blocks (`attention_dropout = 0.0`).
  - Repeated forward passes produced mathematically identical logits ($\Delta = 0.0, \text{Var} = 0.0$).
* **Root Cause**: Modern open-weight models are pretrained without dropout; toggling `m.train()` on `nn.Dropout` did not affect the forward graph.
* **Correction**: Invalidate the claim that EAR-GRPO-v1 evaluated epistemic uncertainty. Reframe as an audit of a deterministic logprob proxy and initiate a formal uncertainty proxy discovery benchmark (Phase VI).
