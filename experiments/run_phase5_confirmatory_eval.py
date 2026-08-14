import os
import json
import time
import copy
import torch
import torch.optim as optim
import math
from typing import Dict, List, Any

torch.set_num_threads(8)

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.models.policy import TransformerReasoningPolicy
from src.models.epistemic_probe import EpistemicUncertaintyProbe
from src.rl.grpo_trainer import GRPOTrainer
from src.rl.ear_grpo_trainer import EARGRPOTrainer
from src.data.loader import MathReasoningDataset
from src.rl.rewards import compute_math_reward, extract_answer

def evaluate_on_test_set_item_level(policy, test_samples, device, method_name, seed, max_new_tokens=144):
    policy.model.eval()
    correct_count = 0
    total_tokens = 0
    truncations = 0
    item_records = []
    
    n = len(test_samples)
    for i, s in enumerate(test_samples):
        messages = [
            {"role": "system", "content": "You are a helpful math assistant. Solve the math problem step by step and present the final numerical answer after '#### '."},
            {"role": "user", "content": s["question"]}
        ]
        prompt = policy.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = policy.tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = inputs["input_ids"].shape[1]
        
        with torch.inference_mode():
            outputs = policy.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=policy.tokenizer.pad_token_id
            )
            
        gen_tokens = outputs[0, prompt_len:]
        num_gen = gen_tokens.shape[0]
        total_tokens += num_gen
        is_trunc = (num_gen >= max_new_tokens)
        if is_trunc:
            truncations += 1
            
        text = policy.tokenizer.decode(gen_tokens, skip_special_tokens=True)
        pred = extract_answer(text)
        reward = compute_math_reward(text, s["ground_truth"])
        is_correct = (reward > 0.0)
        if is_correct:
            correct_count += 1
            
        item_records.append({
            "example_id": s.get("id", f"gsm8k_{700 + i}"),
            "question": s["question"],
            "ground_truth": s["ground_truth"],
            "method": method_name,
            "seed": seed,
            "raw_generation": text,
            "parsed_answer": pred,
            "correct": is_correct,
            "generation_length": num_gen,
            "truncated": is_trunc
        })
        
    pass1 = (correct_count / n) * 100.0
    return {
        "n": n,
        "correct": correct_count,
        "pass1": pass1,
        "avg_tokens": total_tokens / n,
        "truncation_rate": (truncations / n) * 100.0,
        "items": item_records
    }

def run_phase5_confirmatory_eval():
    print("=" * 90, flush=True)
    print("EAR-GRPO PHASE V-N: 100-EXAMPLE CONFIRMATORY EVALUATION & MECHANISM VALIDATION", flush=True)
    print("=" * 90, flush=True)
    
    device_str = "cpu"
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Device: {device_str} (8 threads) | Model: {model_name}", flush=True)
    
    dataset = MathReasoningDataset()
    train_subset = [dataset[i] for i in range(0, 500)]
    test_100 = [dataset[i] for i in range(700, 800)]
    print(f"Loaded {len(test_100)} frozen held-out test items (indices 700 to 799).", flush=True)
    
    # Load base policies once
    pol = TransformerReasoningPolicy(model_name_or_path=model_name, device=device_str)
    ref_pol = TransformerReasoningPolicy(model_name_or_path=model_name, device=device_str)
    ref_pol.eval()
    
    base_state_dict = copy.deepcopy(pol.model.state_dict())
    
    methods = [
        {"name": "Standard-GRPO", "mode": "grpo", "group_size": 4},
        {"name": "Compute-Matched-GRPO", "mode": "grpo", "group_size": 7},
        {"name": "Random-Control", "mode": "random_control", "group_size": 4},
        {"name": "Permuted-Control", "mode": "permuted_control", "group_size": 4},
        {"name": "EAR-GRPO", "mode": "ear", "group_size": 4}
    ]
    seeds = [42, 100, 2024]
    best_config = {"lr": 1e-5, "kl_beta": 0.04}
    
    results_summary = {}
    all_item_records = []
    
    out_dir = "results/raw/phase5_negative_confirmation"
    os.makedirs(out_dir, exist_ok=True)
    
    for m in methods:
        results_summary[m["name"]] = []
        print(f"\n--- Method: {m['name']} (Group Size = {m['group_size']}) ---", flush=True)
        
        for seed in seeds:
            ckpt_path = f"{out_dir}/{m['name']}_seed_{seed}.json"
            if os.path.exists(ckpt_path):
                with open(ckpt_path, "r") as f:
                    rec = json.load(f)
                results_summary[m["name"]].append(rec)
                print(f"  [CACHED] [{m['name']}][seed={seed}] -> Pass@1: {rec['pass1']:.2f}% ({rec['correct']}/100)", flush=True)
                continue
                
            torch.manual_seed(seed)
            pol.model.load_state_dict(base_state_dict)
            opt = optim.Adam(pol.parameters(), lr=best_config['lr'])
            probe = EpistemicUncertaintyProbe(num_mc_samples=2)
            
            if m["mode"] == "grpo":
                trainer = GRPOTrainer(policy=pol, ref_policy=ref_pol, optimizer=opt, kl_beta=best_config['kl_beta'])
            else:
                trainer = EARGRPOTrainer(policy=pol, ref_policy=ref_pol, optimizer=opt, epistemic_probe=probe, gamma_epistemic=0.35, kl_beta=best_config['kl_beta'], mode=m["mode"])
                
            # Perform exact 2 training steps
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
                trainer.train_step(input_ids_group, [prompt_len] * m["group_size"], rewards_group)
                
            # Evaluate post-training model on 100 held-out items
            t0 = time.time()
            eval_res = evaluate_on_test_set_item_level(pol, test_100, pol.device, m["name"], seed, max_new_tokens=144)
            eval_time = time.time() - t0
            
            rec = {
                "method": m["name"],
                "seed": seed,
                "n": eval_res["n"],
                "correct": eval_res["correct"],
                "pass1": eval_res["pass1"],
                "avg_tokens": eval_res["avg_tokens"],
                "eval_time_sec": eval_time
            }
            with open(ckpt_path, "w") as f:
                json.dump(rec, f, indent=2)
                
            results_summary[m["name"]].append(rec)
            all_item_records.extend(eval_res["items"])
            print(f"  [EVALUATED] [{m['name']}][seed={seed}] -> Pass@1: {eval_res['pass1']:.2f}% ({eval_res['correct']}/100) in {eval_time:.1f}s (Avg Tokens: {eval_res['avg_tokens']:.1f})", flush=True)
            
    # Save all item-level outputs
    with open(f"{out_dir}/item_level_predictions_100.json", "w") as f:
        json.dump(all_item_records, f, indent=2)
        
    with open(f"{out_dir}/phase5_summary_100.json", "w") as f:
        json.dump(results_summary, f, indent=2)
        
    print("\n" + "=" * 90, flush=True)
    print("PHASE V-N CONFIRMATORY RESULTS (100 HELD-OUT TEST ITEMS, max_new_tokens=144):", flush=True)
    print(f"{'Method':<24} | {'Seed 42':<10} | {'Seed 100':<10} | {'Seed 2024':<10} | {'Mean +/- SD':<14}")
    print("-" * 90, flush=True)
    
    for m_name, recs in results_summary.items():
        s42 = recs[0]["pass1"]
        s100 = recs[1]["pass1"]
        s2024 = recs[2]["pass1"]
        vals = [s42, s100, s2024]
        mean_p1 = torch.tensor(vals).mean().item()
        sd_p1 = torch.tensor(vals).std().item()
        print(f"{m_name:<24} | {s42:>6.2f}%    | {s100:>6.2f}%    | {s2024:>6.2f}%    | {mean_p1:>5.2f} +/- {sd_p1:<5.2f}%", flush=True)
    print("=" * 90, flush=True)

if __name__ == "__main__":
    run_phase5_confirmatory_eval()
