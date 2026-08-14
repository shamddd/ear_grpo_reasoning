# Repaired Evaluator & Trainer Pipeline Freeze Record

## Pipeline Specification

- **Model ID**: `Qwen/Qwen2.5-0.5B-Instruct`
- **Tokenizer**: `Qwen/Qwen2.5-0.5B-Instruct` (left padding for batch generation)
- **Dataset Revision**: `main` (`openai/gsm8k`, `main` split)
- **Prompt Format**: ChatML (`apply_chat_template`)
  ```jinja2
  <|im_start|>system
  You are a helpful math assistant. Solve the math problem step by step and present the final numerical answer after '#### '.<|im_end|>
  <|im_start|>user
  {question}<|im_end|>
  <|im_start|>assistant
  ```
- **Parser**: Regular expression answer extractor (`r"[-+]?\d+(?:\.\d+)?"`) with support for LaTeX `\boxed{...}` and `#### ...`.
- **Verifier**: Exact numerical string comparison (`compute_math_reward`).
- **Generation Budget**: `max_new_tokens = 256`, `do_sample = False` (greedy baseline evaluation), `temperature = 0.7` for RL rollouts.
- **Git State**: Repository tagged `phase2-real-model-pilot`. Commit SHA recorded at pipeline launch.

---

## Freezing Rules

1. No modifications to `extract_answer` or `compute_math_reward` during Phase IV benchmark runs.
2. Evaluation datasets (GSM8K Validation & Test) are strictly read-only.
3. Baseline evaluation code (`eval_qwen_baseline_200.py`) is frozen.
