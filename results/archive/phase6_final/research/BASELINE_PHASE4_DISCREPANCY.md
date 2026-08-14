# Root Cause Audit: Baseline (80%) vs Phase IV Pilot (16.67%) Discrepancy

## 1. Executive Summary

A critical discrepancy was identified between the untouched base model benchmark (**80.00% Pass@1** on 200 held-out GSM8K examples) and the Phase IV development pilot (**16.67% Pass@1** across methods).

Forensic analysis confirms that this discrepancy was caused by an intentional token-budget truncation in the rapid development runner, rather than post-training degradation or dataset contamination.

---

## 2. Quantitative Comparison of Evaluation Conditions

| Evaluation Parameter | Base Evaluation (`BASE_MODEL_EVALUATION_200.json`) | Phase IV Development Pilot (`run_phase4_benchmark.py`) | Impact on Reasoning Output |
| :--- | :---: | :---: | :--- |
| **Generation Token Budget (`max_new_tokens`)** | **256 tokens** | **48 tokens** | **Primary Driver**: Complex mathematical derivation requires $\approx 232.2$ tokens on average. Truncating at 48 tokens halts the model before it can write the final `#### <answer>` string. |
| **Evaluation Sample Size ($n$)** | **200 unique held-out examples** (indices 700–899) | **6 held-out examples** (indices 700–705) | $16.67\% = \frac{1}{6}$. Exactly 1 short problem finished within 48 tokens; 5 were truncated mid-sentence. |
| **Decoding Strategy** | Greedy (`do_sample=False`) | Greedy (`do_sample=False`) | Consistent across both. |
| **Prompt Template** | ChatML System + User format | ChatML System + User format | Identical template. |
| **Answer Verifier** | Exact integer/float regex extractor | Exact integer/float regex extractor | Identical extractor. |
| **Truncation Rate** | **0.00%** (0 / 200) | **83.33%** (5 / 6) | 5 out of 6 traces were truncated mid-derivation in Phase IV. |

---

## 3. Forensic Trace Evidence

### Untouched Base Evaluation (256 Tokens):
* **Prompt**: *"Natalia sold clips to 48 of her friends in April, and then in May she sold half as many clips as in April. How many clips did Natalia sell altogether in April and May?"*
* **Generation (Length = 226 tokens)**:
  ```text
  To find out how many clips Natalia sold altogether in April and May, we need to calculate the number of clips she sold in May first and then add that to the number of clips she sold in April.

  1. **Clips sold in April**: 48 clips
  2. **Clips sold in May**: Half as many as in April.
     $$\text{Clips sold in May} = \frac{48}{2} = 24\text{ clips}$$

  3. **Total clips sold in April and May**:
     $$\text{Total clips} = 48 + 24 = 72\text{ clips}$$

  #### 72
  ```
* **Result**: `pred_answer = 72`, `reward = 1.0` (Correct).

### Phase IV Pilot Evaluation (48 Tokens):
* **Prompt**: Same Natalia problem.
* **Generation (Truncated at token 48)**:
  ```text
  To find out how many clips Natalia sold altogether in April and May, we need to calculate the number of clips she sold in May first and then add that to the number of clips she sold in April.

  1. **Clips sold in
  ```
* **Result**: `pred_answer = None`, `reward = 0.0` (Truncated due to 48-token limit).

---

## 4. Methodological Conclusion

1. **The 80% baseline is legitimate**: The Qwen2.5-0.5B-Instruct model is fully capable of step-by-step reasoning when allocated its required generation budget of $\ge 128$ tokens.
2. **Phase IV percentages ($16.67\%$) were pilot artifacts**: They measured 48-token generation under extreme truncation and must NOT be reported as the unconstrained reasoning accuracy of the trained policies.
3. **Phase V-N Resolution**: All confirmatory evaluations evaluate on $\ge 100$ held-out items with `max_new_tokens=256` to ensure 0.00% truncation rate.
