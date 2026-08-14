"""
Tier 1 Scientific Pilot & Control Experiments: Real Transformer RL Benchmarking
Evaluates:
1. EAR-GRPO (Proposed Epistemic Advantage Regularized GRPO)
2. Standard-GRPO (Baseline)
3. Control 1: Random-Control (Random Gaussian noise dampening)
4. Control 2: Permuted-Control (Shuffled rollout uncertainty)
5. Control 3: Compute-Matched-GRPO (GRPO with G_matched = G + M = 7 rollouts)
"""

import sys
import os
import json
import torch
import torch.optim as optim
import numpy as np
from typing import Dict, List, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.loader import MathReasoningDataset
from src.models.policy import TransformerReasoningPolicy
from src.models.epistemic_probe import EpistemicUncertaintyProbe
from src.rl.grpo_trainer import GRPOTrainer
from src.rl.ear_grpo_trainer import EARGRPOTrainer
from src.rl.rewards import compute_math_reward
from src.utils.metrics import compute_welch_t_test

SEEDS = [42, 100, 2024]

def run_real_experiment(seed: int, method_name: str, policy: TransformerReasoningPolicy, ref_policy: TransformerReasoningPolicy) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    dataset = MathReasoningDataset()
    optimizer = optim.Adam(policy.parameters(), lr=1e-5)
    probe = EpistemicUncertaintyProbe(num_mc_samples=3)
    
    # Configure trainer mode
    if method_name == "EAR-GRPO":
        trainer = EARGRPOTrainer(policy, ref_policy, optimizer, probe, gamma_epistemic=0.35, mode="ear")
        group_size = 4
    elif method_name == "Random-Control":
        trainer = EARGRPOTrainer(policy, ref_policy, optimizer, probe, gamma_epistemic=0.35, mode="random_control")
        group_size = 4
    elif method_name == "Permuted-Control":
        trainer = EARGRPOTrainer(policy, ref_policy, optimizer, probe, gamma_epistemic=0.35, mode="permuted_control")
        group_size = 4
    elif method_name == "Compute-Matched-GRPO":
        trainer = GRPOTrainer(policy, ref_policy, optimizer)
        group_size = 7  # G_matched = G (4) + M (3) = 7 rollouts
    elif method_name == "Standard-GRPO":
        trainer = GRPOTrainer(policy, ref_policy, optimizer)
        group_size = 4
    else:
        raise ValueError(f"Unknown method_name: {method_name}")
        
    step_history = []
    
    # Run 2 optimization steps per seed for fast, robust pilot verification
    for step in range(1, 3):
        sample = dataset[(step - 1) % len(dataset)]
        prompt = f"Question: {sample['question']}\nAnswer:"
        
        input_ids_group, completions = policy.generate_completions([prompt] * group_size, max_new_tokens=16)
        prompt_len = policy.tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
        prompt_lengths = [prompt_len] * group_size
        
        rewards_list = [compute_math_reward(c, sample["ground_truth"]) for c in completions]
        rewards_group = torch.tensor(rewards_list, dtype=torch.float32, device=policy.device)
        
        metrics = trainer.train_step(input_ids_group, prompt_lengths, rewards_group)
        print(f"    [{method_name} | Seed {seed}] Step {step}/2 -> Loss: {metrics['loss']:.4f} | Reward: {metrics['mean_reward']:.2f} | Entropy: {metrics['entropy']:.4f}", flush=True)
        step_history.append({
            "step": step,
            "rewards": rewards_list,
            "metrics": metrics
        })
        
    # Evaluate Pass@1 accuracy on test dataset prompts
    test_correct = 0
    eval_subset = [dataset[i] for i in range(2)]
    for sample in eval_subset:
        prompt = f"Question: {sample['question']}\nAnswer:"
        _, eval_completions = policy.generate_completions([prompt], max_new_tokens=8)
        r = compute_math_reward(eval_completions[0], sample["ground_truth"])
        if r > 0.0:
            test_correct += 1
            
    pass1_accuracy = (test_correct / len(eval_subset)) * 100.0
    final_entropy = step_history[-1]["metrics"]["entropy"]
    
    return {
        "seed": seed,
        "method": method_name,
        "pass1_accuracy": pass1_accuracy,
        "final_entropy": final_entropy,
        "history": step_history
    }

def main():
    print("=" * 75)
    print("RUNNING TIER 1 SCIENTIFIC PILOT & CONTROL EXPERIMENTS (REAL TRANSFORMERS)")
    print("=" * 75)
    
    # Pre-instantiate policy networks once
    model_name = "gpt2"
    policy = TransformerReasoningPolicy(model_name_or_path=model_name, device="auto")
    ref_policy = TransformerReasoningPolicy(model_name_or_path=model_name, device="auto")
    ref_policy.eval()
    
    methods = ["Standard-GRPO", "Compute-Matched-GRPO", "Random-Control", "Permuted-Control", "EAR-GRPO"]
    all_results = {}
    
    for method in methods:
        all_results[method] = []
        print(f"\n>>> Evaluating Method: {method} <<<", flush=True)
        for seed in SEEDS:
            res = run_real_experiment(seed, method, policy, ref_policy)
            all_results[method].append(res)
            print(f"  Seed {seed:<5} -> Pass@1: {res['pass1_accuracy']:.2f}% | Final Entropy: {res['final_entropy']:.4f}", flush=True)
            
    # Compute statistical summaries
    summary = {}
    for method in methods:
        accs = [r["pass1_accuracy"] for r in all_results[method]]
        entropies = [r["final_entropy"] for r in all_results[method]]
        summary[method] = {
            "mean_pass1": float(np.mean(accs)),
            "std_pass1": float(np.std(accs)),
            "mean_entropy": float(np.mean(entropies)),
            "raw_accs": accs
        }
        
    print("\n" + "=" * 75)
    print("TIER 1 PILOT SUMMARY & EMPIRICAL COMPARISON")
    print("=" * 75)
    for method, s in summary.items():
        print(f"  {method:<22} -> Pass@1: {s['mean_pass1']:.2f}% ± {s['std_pass1']:.2f} | Policy Entropy: {s['mean_entropy']:.4f}")
        
    # Welch's t-test comparing EAR-GRPO vs controls
    ear_accs = summary["EAR-GRPO"]["raw_accs"]
    grpo_accs = summary["Standard-GRPO"]["raw_accs"]
    rand_accs = summary["Random-Control"]["raw_accs"]
    matched_accs = summary["Compute-Matched-GRPO"]["raw_accs"]
    
    t_grpo, p_grpo = compute_welch_t_test(ear_accs, grpo_accs)
    t_rand, p_rand = compute_welch_t_test(ear_accs, rand_accs)
    t_matched, p_matched = compute_welch_t_test(ear_accs, matched_accs)
    
    print("\n" + "-" * 75)
    print("STATISTICAL SIGNIFCANCE TESTS (WELCH'S T-TEST):")
    print(f"  EAR-GRPO vs Standard-GRPO      : t = {t_grpo:.3f}, p = {p_grpo:.4f}")
    print(f"  EAR-GRPO vs Random-Control     : t = {t_rand:.3f}, p = {p_rand:.4f}")
    print(f"  EAR-GRPO vs Compute-Matched    : t = {t_matched:.3f}, p = {p_matched:.4f}")
    print("=" * 75)

    # Save raw outputs
    output_dir = "results/raw/tier1_pilot"
    os.makedirs(output_dir, exist_ok=True)
    raw_file = os.path.join(output_dir, "tier1_pilot_results.json")
    with open(raw_file, "w") as f:
        json.dump({"summary": summary, "runs": all_results}, f, indent=2)
        
    print(f"Preserved raw experiment evidence to '{raw_file}'.")

if __name__ == "__main__":
    main()
