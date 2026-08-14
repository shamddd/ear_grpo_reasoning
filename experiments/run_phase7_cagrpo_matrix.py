import os
import json
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import math
from typing import Dict, List, Any

torch.set_num_threads(8)

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.models.policy import TransformerReasoningPolicy
from src.rl.grpo_trainer import GRPOTrainer
from src.data.loader import MathReasoningDataset
from src.rl.rewards import compute_math_reward, extract_answer

class CAGRPOTrainer:
    """
    Consistency-Aware Group Relative Policy Optimization (CA-GRPO) Trainer.
    Weights group advantages by sample-level consensus agreement.
    Supports experimental control variants:
    - mode="cagrpo" (True consensus agreement advantage scaling)
    - mode="random_control" (Random noise weighting)
    - mode="permuted_control" (Permuted consensus weights across rollouts)
    """
    def __init__(
        self,
        policy: nn.Module,
        ref_policy: nn.Module,
        optimizer: optim.Optimizer,
        lambda_cons: float = 0.5,
        clip_eps: float = 0.2,
        kl_beta: float = 0.04,
        mode: str = "cagrpo"
    ):
        self.policy = policy
        self.ref_policy = ref_policy
        self.optimizer = optimizer
        self.lambda_cons = lambda_cons
        self.clip_eps = clip_eps
        self.kl_beta = kl_beta
        self.mode = mode

    def train_step(
        self,
        input_ids_group: torch.Tensor,
        prompt_lengths: List[int],
        rewards_group: torch.Tensor,
        completions: List[str]
    ) -> Dict[str, float]:
        self.policy.train()
        self.optimizer.zero_grad()
        
        # 1. Compute standard GRPO advantages
        mean_r = torch.mean(rewards_group)
        std_r = torch.std(rewards_group, unbiased=False)
        std_advantages = (rewards_group - mean_r) / (std_r + 1e-8)
        
        # 2. Extract answers and compute modal consensus
        preds = [extract_answer(c) for c in completions]
        valid_preds = [p for p in preds if p is not None]
        
        if len(valid_preds) > 0:
            counts = {p: valid_preds.count(p) for p in set(valid_preds)}
            modal_ans = max(counts, key=counts.get)
            cons_weights = torch.tensor([1.0 if p == modal_ans else 0.0 for p in preds], dtype=torch.float32, device=self.policy.device)
        else:
            cons_weights = torch.zeros(len(completions), dtype=torch.float32, device=self.policy.device)
            
        if self.mode == "cagrpo":
            weights = cons_weights
        elif self.mode == "random_control":
            weights = torch.randn(len(completions), device=self.policy.device)
        elif self.mode == "permuted_control":
            perm = torch.randperm(len(completions))
            weights = cons_weights[perm]
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
            
        mean_w = torch.mean(weights)
        ca_advantages = std_advantages * (1.0 + self.lambda_cons * (weights - mean_w))
        
        # 3. Policy forward pass & reference logprobs
        logits = self.policy(input_ids_group)
        with torch.no_grad():
            ref_input_ids = input_ids_group.to(self.ref_policy.device)
            ref_logits = self.ref_policy(ref_input_ids)
            ref_log_probs = self.ref_policy.compute_completion_log_probs(ref_input_ids, prompt_lengths, logits=ref_logits).to(self.policy.device)
            
        log_probs = self.policy.compute_completion_log_probs(input_ids_group, prompt_lengths, logits=logits)
        
        # 4. Surrogate policy loss
        old_log_probs = log_probs.detach()
        ratios = torch.exp(log_probs - old_log_probs)
        surr1 = ratios * ca_advantages
        surr2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * ca_advantages
        policy_loss = -torch.mean(torch.min(surr1, surr2))
        
        # 5. KL divergence penalty
        kl_div = torch.mean(log_probs - ref_log_probs)
        loss = policy_loss + self.kl_beta * kl_div
        
        loss.backward()
        self.optimizer.step()
        
        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1)
            entropy = -torch.mean(torch.sum(probs * torch.log(probs + 1e-9), dim=-1)).item()
            num_correct = torch.sum(rewards_group > 0.0).item()
            
        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "kl_div": kl_div.item(),
            "mean_reward": torch.mean(rewards_group).item(),
            "entropy": entropy,
            "positive_rollout_rate": num_correct / rewards_group.shape[0]
        }

def evaluate_on_samples(policy, test_samples, device, max_new_tokens=144):
    policy.model.eval()
    correct = 0
    total_tokens = 0
    n = len(test_samples)
    for s in test_samples:
        messages = [
            {"role": "system", "content": "You are a helpful math assistant. Solve the math problem step by step and present the final numerical answer after '#### '."},
            {"role": "user", "content": s["question"]}
        ]
        prompt = policy.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = policy.tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            outputs = policy.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=policy.tokenizer.pad_token_id)
        gen_tokens = outputs[0, prompt_len:]
        total_tokens += len(gen_tokens)
        text = policy.tokenizer.decode(gen_tokens, skip_special_tokens=True)
        if compute_math_reward(text, s["ground_truth"]) > 0.0:
            correct += 1
    return (correct / n) * 100.0, total_tokens / n

def run_phase7_rl_matrix():
    print("=" * 90, flush=True)
    print("PHASE VII: PREREGISTERED 5-WAY CA-GRPO CONTROLLED RL EXPERIMENT", flush=True)
    print("=" * 90, flush=True)
    
    device_str = "cpu"
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Device: {device_str} (8 threads) | Model: {model_name}", flush=True)
    
    dataset = MathReasoningDataset()
    train_subset = [dataset[i] for i in range(0, 500)]
    test_untouched = [dataset[i] for i in range(800, 900)]
    
    pol = TransformerReasoningPolicy(model_name_or_path=model_name, device=device_str)
    ref_pol = TransformerReasoningPolicy(model_name_or_path=model_name, device=device_str)
    ref_pol.eval()
    base_state_dict = copy.deepcopy(pol.model.state_dict())
    
    best_config = {"lr": 1e-5, "kl_beta": 0.04}
    seeds = [42, 100, 2024]
    methods = [
        {"name": "Standard-GRPO", "type": "grpo", "mode": "grpo", "group_size": 4},
        {"name": "Compute-Matched-GRPO", "type": "grpo", "mode": "grpo", "group_size": 8},
        {"name": "Random-Weight-Control", "type": "ca", "mode": "random_control", "group_size": 4},
        {"name": "Permuted-Consistency-Control", "type": "ca", "mode": "permuted_control", "group_size": 4},
        {"name": "CA-GRPO", "type": "ca", "mode": "cagrpo", "group_size": 4}
    ]
    
    results = {}
    out_dir = "results/raw/phase7_rl"
    os.makedirs(out_dir, exist_ok=True)
    
    for m in methods:
        results[m["name"]] = []
        print(f"\nEvaluating RL Method: {m['name']} (G = {m['group_size']})...", flush=True)
        
        for seed in seeds:
            ckpt_path = f"{out_dir}/{m['name']}_seed_{seed}.json"
            if os.path.exists(ckpt_path):
                with open(ckpt_path, "r") as f:
                    rec = json.load(f)
                results[m["name"]].append(rec)
                print(f"  [LOADED] [{m['name']}][seed={seed}] -> Test Pass@1: {rec['test_pass1']:.2f}% | Mean Reward: {rec['train_mean_reward']:.2f}", flush=True)
                continue
                
            torch.manual_seed(seed)
            pol.model.load_state_dict(base_state_dict)
            opt = optim.Adam(pol.parameters(), lr=best_config['lr'])
            
            if m["type"] == "grpo":
                trainer = GRPOTrainer(policy=pol, ref_policy=ref_pol, optimizer=opt, kl_beta=best_config['kl_beta'])
            else:
                trainer = CAGRPOTrainer(policy=pol, ref_policy=ref_pol, optimizer=opt, lambda_cons=0.5, kl_beta=best_config['kl_beta'], mode=m["mode"])
                
            train_metrics = []
            for step in range(2):
                sample = train_subset[step % len(train_subset)]
                messages = [
                    {"role": "system", "content": "You are a helpful math assistant. Solve the math problem step by step and present the final numerical answer after '#### '."},
                    {"role": "user", "content": sample["question"]}
                ]
                prompt_text = pol.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                input_ids_group, completions = pol.generate_completions([prompt_text] * m["group_size"], max_new_tokens=48)
                prompt_len = pol.tokenizer(prompt_text, return_tensors="pt")["input_ids"].shape[1]
                rewards_list = [compute_math_reward(comp, sample["ground_truth"]) for comp in completions]
                rewards_group = torch.tensor(rewards_list, dtype=torch.float32, device=pol.device)
                
                if m["type"] == "grpo":
                    step_res = trainer.train_step(input_ids_group, [prompt_len] * m["group_size"], rewards_group)
                else:
                    step_res = trainer.train_step(input_ids_group, [prompt_len] * m["group_size"], rewards_group, completions)
                train_metrics.append(step_res)
                
            # Evaluate post-training model on 100 untouched test samples
            t_eval0 = time.time()
            test_pass1, avg_toks = evaluate_on_samples(pol, test_untouched, pol.device, max_new_tokens=144)
            eval_time = time.time() - t_eval0
            
            mean_rew = float(torch.tensor([x["mean_reward"] for x in train_metrics]).mean().item())
            mean_ent = float(torch.tensor([x["entropy"] for x in train_metrics]).mean().item())
            mean_kl = float(torch.tensor([x["kl_div"] for x in train_metrics]).mean().item())
            
            rec = {
                "method": m["name"],
                "seed": seed,
                "test_pass1": test_pass1,
                "train_mean_reward": mean_rew,
                "entropy": mean_ent,
                "kl_div": mean_kl,
                "avg_tokens": avg_toks,
                "eval_time_sec": eval_time
            }
            with open(ckpt_path, "w") as f:
                json.dump(rec, f, indent=2)
                
            results[m["name"]].append(rec)
            print(f"  [{m['name']}][seed={seed}] -> Test Pass@1: {test_pass1:.2f}% | Train Reward: {mean_rew:.2f} | Entropy: {mean_ent:.4f}", flush=True)
            
    with open(f"{out_dir}/phase7_rl_summary.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "=" * 90, flush=True)
    print("PHASE VII PREREGISTERED RL EXPERIMENT RESULTS (Untouched GSM8K N=100):", flush=True)
    print(f"{'Method':<28} | {'Seed 42':<10} | {'Seed 100':<10} | {'Seed 2024':<10} | {'Mean +/- SD':<14}")
    print("-" * 90, flush=True)
    
    for m_name, recs in results.items():
        s42 = recs[0]["test_pass1"]
        s100 = recs[1]["test_pass1"]
        s2024 = recs[2]["test_pass1"]
        vals = [s42, s100, s2024]
        mean_p1 = torch.tensor(vals).mean().item()
        sd_p1 = torch.tensor(vals).std().item()
        print(f"{m_name:<28} | {s42:>6.2f}%    | {s100:>6.2f}%    | {s2024:>6.2f}%    | {mean_p1:>5.2f} +/- {sd_p1:<5.2f}%", flush=True)
    print("=" * 90, flush=True)

if __name__ == "__main__":
    run_phase7_rl_matrix()
