# Dataset Split Audit & Leakage Prevention Record

## Immutable Dataset Splits

To guarantee zero data contamination between training, hyperparameter tuning, baseline evaluation, and out-of-distribution (OOD) testing, we define explicit index splits on GSM8K and SVAMP:

1. **GSM8K Training Subset**: Indices `0` to `499` (500 samples)
   - Used for training RL policies (GRPO and EAR-GRPO).
2. **GSM8K Validation Subset**: Indices `500` to `699` (200 samples)
   - Used exclusively for hyperparameter selection and tuning log (`research/GRPO_TUNING_LOG.md`).
3. **GSM8K Test Subset**: Indices `700` to `899` (200 samples)
   - Held-out test benchmark. Never used during training or parameter tuning.
4. **OOD Benchmark**: SVAMP dataset (200 held-out samples)
   - Frozen benchmark for testing generalization.

---

## Leakage Prevention Verification

- [x] **Split Separation**: Training, validation, and test index sets are strictly disjoint:
  $$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset$$
- [x] **Prompt Sanitization**: Prompts include only system text and user question text. Target numerical answers are omitted.
- [x] **Reward Computation Isolation**: Reward functions evaluate completions post-generation. Reference ground-truth labels are never fed into the language model input context.
- [x] **OOD Holdout**: SVAMP dataset parameters are frozen prior to opening OOD test results.
