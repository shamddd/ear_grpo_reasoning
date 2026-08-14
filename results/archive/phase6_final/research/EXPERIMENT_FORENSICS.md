# EXPERIMENT FORENSIC AUDIT REPORT

## Executive Forensic Verdict
> [!CAUTION]
> **ALL PHASE I REPORTED METRICS ARE INVALID / SYNTHETIC PLACEHOLDERS.**
> Every paper claim and benchmark percentage reported during Phase I (GSM8K 78.41%, SVAMP 69.50%, t=5.294, p=0.0011) originated from synthetic metric sampling (`np.random.normal`) and a GRU toy module (`SimulatedReasoningPolicy`), NOT from actual reinforcement-learning training of real language models.
>
> In accordance with Phase II Scientific Verification rules, **ALL PHASE I CLAIMS ARE HEREBY MARKED AS INVALID AND DISCARDED.**

---

## 1. Forensic Audit Trail

| Claim / Metric | Source Script | Model | Dataset | Actual RL Training? | Raw Artifact | Valid Status |
|---|---|---|---|---|---|---|
| GSM8K 74.12% (GRPO) | `experiments/run_main_experiments.py` | `SimulatedReasoningPolicy` (GRU) | Synthetic / Hardcoded | **NO** (Simulated `74.1 + norm`) | None | **INVALID RESULT** |
| GSM8K 78.41% (EAR-GRPO) | `experiments/run_main_experiments.py` | `SimulatedReasoningPolicy` (GRU) | Synthetic / Hardcoded | **NO** (Simulated `78.4 + norm`) | None | **INVALID RESULT** |
| SVAMP 62.23% (GRPO) | `experiments/run_main_experiments.py` | `SimulatedReasoningPolicy` (GRU) | Synthetic / Hardcoded | **NO** (Simulated `61.8 + norm`) | None | **INVALID RESULT** |
| SVAMP 69.50% (EAR-GRPO) | `experiments/run_main_experiments.py` | `SimulatedReasoningPolicy` (GRU) | Synthetic / Hardcoded | **NO** (Simulated `69.2 + norm`) | None | **INVALID RESULT** |
| Welch's t=5.294, p=0.0011 | `experiments/run_main_experiments.py` | None | Synthetic | **NO** | `results/raw_data/main_benchmark_results.json` | **INVALID RESULT** |

---

## 2. Identified Synthetic & Mock Code Snippets

1. **`experiments/run_main_experiments.py` (Lines 44-51)**:
   ```python
   # Simulated Pass@1 accuracy evaluation
   if method == "EAR-GRPO":
       pass1_gsm8k = 78.4 + np.random.normal(0, 0.8)
       pass1_svamp = 69.2 + np.random.normal(0, 1.0)
   else:
       pass1_gsm8k = 74.1 + np.random.normal(0, 1.2)
       pass1_svamp = 61.8 + np.random.normal(0, 1.4)
   ```
2. **`experiments/run_ablations.py` (Lines 20-28)**:
   ```python
   if g == 0.0: score = 74.1
   elif g == 0.35: score = 78.4
   elif g == 1.0: score = 71.5
   ```
3. **`src/models/policy.py`**: A 256-hidden-dim GRU rather than an open transformer language model.

---

## 3. Mandatory Phase II Corrective Action Plan
1. Delete all synthetic metric generation from experiment runners.
2. Upgrade `src/models/policy.py` to use real Hugging Face `AutoModelForCausalLM` and `AutoTokenizer` (e.g. `Qwen/Qwen2.5-0.5B-Instruct` / `Qwen/Qwen2.5-Math-1.5B-Instruct` or `Qwen/Qwen2.5-1.5B-Instruct`).
3. Load real GSM8K math reasoning examples (`openai/gsm8k`) and evaluate generated tokens using exact mathematical SymPy string/numerical parsing.
4. Log full generated text completions, prompt tokens, log-probabilities, MC epistemic variances, advantages, and rewards for every rollout in `results/raw/`.
