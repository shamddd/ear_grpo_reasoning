# DATA CONTAMINATION & LEAKAGE AUDIT

## 1. Benchmark & Training Split Partitioning
To ensure rigorous evaluation of model reasoning rather than memorization:
* **In-Distribution Training Split**: `openai/gsm8k` (Train split: 7,473 examples).
* **In-Distribution Test Split**: `openai/gsm8k` (Test split: 1,319 examples).
* **Out-of-Distribution Benchmark**: `svamp` (Adversarial word math problems: 1,000 examples).

## 2. Pretraining Exposure Audit
* Base models (`Qwen/Qwen2.5-1.5B-Instruct` and `Qwen/Qwen2.5-0.5B-Instruct`) have encountered public internet text during pre-training.
* However, our RL evaluation compares **Standard GRPO vs EAR-GRPO post-training on the exact same base model**. Since both methods start from the identical pre-trained checkpoint, any differential gain in Pass@1 and OOD transfer ratio on SVAMP is strictly attributable to the post-training RL algorithm rather than pre-training data leakage.

## 3. Strict Prompt & Verifier Firewall
* Rewards are computed strictly on final numerical equivalence via SymPy string parsing.
* Prompts given to the model during training and evaluation contain zero solution trace hints or verifier answers.
