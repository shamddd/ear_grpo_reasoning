"""
Tier 0 Pipeline Validation: Real Transformer RL Training Step Verification
"""

import sys
import os
import json
import yaml
import torch
import torch.optim as optim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.loader import MathReasoningDataset
from src.models.policy import TransformerReasoningPolicy
from src.models.epistemic_probe import EpistemicUncertaintyProbe
from src.rl.grpo_trainer import GRPOTrainer
from src.rl.ear_grpo_trainer import EARGRPOTrainer
from src.rl.rewards import compute_math_reward
from src.utils.logger import ExperimentLogger

def main():
    print("=" * 70)
    print("TIER 0 PIPELINE VALIDATION: REAL TRANSFORMER RL TRAINING STEP")
    print("=" * 70)
    
    # 1. Dataset
    dataset = MathReasoningDataset()
    print(f"[1/5] Loaded dataset with {len(dataset)} reasoning samples.")

    # 2. Models
    model_name = "gpt2"
    policy = TransformerReasoningPolicy(model_name_or_path=model_name, device="auto")
    ref_policy = TransformerReasoningPolicy(model_name_or_path=model_name, device="auto")
    ref_policy.eval()
    
    probe = EpistemicUncertaintyProbe(num_mc_samples=3)
    optimizer = optim.Adam(policy.parameters(), lr=1e-5)
    
    print(f"[2/5] Initialized real transformer policy '{model_name}' on device: {policy.device}.")

    # 3. EAR-GRPO Trainer
    ear_trainer = EARGRPOTrainer(
        policy=policy,
        ref_policy=ref_policy,
        optimizer=optimizer,
        epistemic_probe=probe,
        gamma_epistemic=0.35,
        mode="ear"
    )
    
    output_dir = "results/raw/tier0_sanity"
    logger = ExperimentLogger(output_dir, name="ear_grpo_tier0")
    print(f"[3/5] Initialized EAR-GRPO Trainer and Logger at '{output_dir}'.")

    # 4. Run real optimization steps
    print("[4/5] Running 5 real transformer RL optimization steps on GSM8K prompts...")
    group_size = 4
    raw_rollouts = []
    
    for step in range(1, 6):
        sample = dataset[step - 1]
        prompt = f"Question: {sample['question']}\nAnswer:"
        
        # Sample G=4 rollouts from policy
        input_ids_group, completions = policy.generate_completions([prompt] * group_size, max_new_tokens=32)
        prompt_len = policy.tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
        prompt_lengths = [prompt_len] * group_size
        
        # Verify rewards against ground truth
        rewards_list = [compute_math_reward(c, sample["ground_truth"]) for c in completions]
        rewards_group = torch.tensor(rewards_list, dtype=torch.float32, device=policy.device)
        
        # Perform real PyTorch backpropagation update step
        metrics = ear_trainer.train_step(input_ids_group, prompt_lengths, rewards_group)
        logger.log_step(step, metrics)
        
        # Log raw rollout trace for forensic transparency
        rollout_record = {
            "step": step,
            "prompt": prompt,
            "ground_truth": sample["ground_truth"],
            "completions": completions,
            "rewards": rewards_list,
            "metrics": metrics
        }
        raw_rollouts.append(rollout_record)
        
        print(f"  Step {step}/5 | Question: '{sample['question'][:30]}...' -> Loss: {metrics['loss']:.4f} | "
              f"Mean Reward: {metrics['mean_reward']:.2f} | Epistemic Var: {metrics['mean_epistemic_var']:.4f} | "
              f"Dampening: {metrics['mean_dampening']:.4f} | Entropy: {metrics['entropy']:.4f}")

    # 5. Save Raw Rollout Evidence
    os.makedirs(output_dir, exist_ok=True)
    raw_file = os.path.join(output_dir, "raw_rollouts_evidence.json")
    with open(raw_file, "w") as f:
        json.dump(raw_rollouts, f, indent=2)
        
    print("=" * 70)
    print(f"TIER 0 VALIDATION COMPLETED SUCCESSFULLY!")
    print(f"Raw evidence preserved at: {raw_file}")
    print("=" * 70)

if __name__ == "__main__":
    main()
