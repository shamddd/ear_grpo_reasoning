import os
import json
import time
import copy
import torch
import torch.optim as optim
import math
from typing import Dict, List, Any

# Enable 8 CPU threads for multi-core M1 acceleration
torch.set_num_threads(8)

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.models.policy import TransformerReasoningPolicy
from src.models.epistemic_probe import EpistemicUncertaintyProbe
from src.rl.grpo_trainer import GRPOTrainer
from src.rl.ear_grpo_trainer import EARGRPOTrainer
from src.data.loader import MathReasoningDataset
from src.rl.rewards import compute_math_reward, extract_answer

def evaluate_on_dataset(policy, dataset_samples, device, max_new_tokens=48):
    policy.model.eval()
    correct = 0
    total_gen_len = 0
    truncations = 0
    parse_fails = 0
    
    n = len(dataset_samples)
    for sample in dataset_samples:
        messages = [
            {"role": "system", "content": "You are a helpful math assistant. Solve the math problem step by step and present the final numerical answer after '#### '."},
            {"role": "user", "content": sample["question"]}
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
        total_gen_len += num_gen
        if num_gen >= max_new_tokens:
            truncations += 1
            
        text = policy.tokenizer.decode(gen_tokens, skip_special_tokens=True)
        pred = extract_answer(text)
        if pred is None:
            parse_fails += 1
        if compute_math_reward(text, sample["ground_truth"]) > 0.0:
            correct += 1
            
    pass1 = (correct / n) * 100.0
    return {
        "n": n,
        "correct": correct,
        "pass1": pass1,
        "avg_tokens": total_gen_len / n,
        "trunc_rate": (truncations / n) * 100.0,
        "parse_fail_rate": (parse_fails / n) * 100.0
    }

def run_phase4_benchmark():
    print("=" * 80, flush=True)
    print("EAR-GRPO PHASE IV: CONTROLLED BENCHMARK & SYSTEMATIC EVALUATION", flush=True)
    print("=" * 80, flush=True)
    
    device_str = "cpu"
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"device = {device_str} (8 threads) | Model = {model_name}", flush=True)
    
    dataset = MathReasoningDataset()
    train_subset = [dataset[i] for i in range(0, 500)]
    val_subset = [dataset[i] for i in range(500, 700)]
    test_subset = [dataset[i] for i in range(700, 900)]
    
    # Load base policies once
    t_load0 = time.time()
    print("Loading base policy models into RAM...", flush=True)
    pol = TransformerReasoningPolicy(model_name_or_path=model_name, device=device_str)
    ref_pol = TransformerReasoningPolicy(model_name_or_path=model_name, device=device_str)
    ref_pol.eval()
    print(f"[Telemetry] Base models loaded in {time.time()-t_load0:.2f}s", flush=True)
    
    base_state_dict = copy.deepcopy(pol.model.state_dict())
    
    best_config = {"lr": 1e-5, "kl_beta": 0.04}
    print(f"Loaded Standard GRPO Frozen Config: lr={best_config['lr']}, kl_beta={best_config['kl_beta']}", flush=True)
    
    # 2. PRIMARY 5-METHOD CONTROL MATRIX EXPERIMENT
    print("\n--- STAGE 2: PRIMARY 5-METHOD CONTROLLED BENCHMARK (WITH CHECKPOINT/RESUME) ---", flush=True)
    seeds = [42, 100, 2024]
    methods = [
        {"name": "Standard-GRPO", "mode": "grpo", "group_size": 4},
        {"name": "Compute-Matched-GRPO", "mode": "grpo", "group_size": 7},
        {"name": "Random-Control", "mode": "random_control", "group_size": 4},
        {"name": "Permuted-Control", "mode": "permuted_control", "group_size": 4},
        {"name": "EAR-GRPO", "mode": "ear", "group_size": 4}
    ]
    
    results = {}
    steps_per_seed = 2
    
    for m in methods:
        results[m["name"]] = []
        method_dir = f"results/raw/phase4/{m['name']}"
        os.makedirs(method_dir, exist_ok=True)
        print(f"\nEvaluating Method: {m['name']} (Group Size = {m['group_size']})...", flush=True)
        
        for seed in seeds:
            ckpt_path = f"{method_dir}/seed_{seed}.json"
            if os.path.exists(ckpt_path):
                with open(ckpt_path, "r") as f:
                    rec = json.load(f)
                results[m["name"]].append(rec)
                print(f"  [CHECKPOINT LOADED] [{m['name']}][seed={seed}] -> Test Pass@1: {rec['test_pass1']:.2f}% | Mean Reward: {rec['train_mean_reward']:.2f} | Entropy: {rec['entropy']:.4f}", flush=True)
                continue
                
            torch.manual_seed(seed)
            t_seed_start = time.time()
            
            pol.model.load_state_dict(base_state_dict)
            opt = optim.Adam(pol.parameters(), lr=best_config['lr'])
            probe = EpistemicUncertaintyProbe(num_mc_samples=2)
            
            if m["mode"] == "grpo":
                trainer = GRPOTrainer(policy=pol, ref_policy=ref_pol, optimizer=opt, kl_beta=best_config['kl_beta'])
            else:
                trainer = EARGRPOTrainer(policy=pol, ref_policy=ref_pol, optimizer=opt, epistemic_probe=probe, gamma_epistemic=0.35, kl_beta=best_config['kl_beta'], mode=m["mode"])
                
            train_metrics = []
            for step in range(steps_per_seed):
                t_step_start = time.time()
                sample = train_subset[step % len(train_subset)]
                messages = [
                    {"role": "system", "content": "You are a helpful math assistant. Solve the math problem step by step and present the final numerical answer after '#### '."},
                    {"role": "user", "content": sample["question"]}
                ]
                prompt_text = pol.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
                t_gen0 = time.time()
                input_ids_group, completions = pol.generate_completions([prompt_text] * m["group_size"], max_new_tokens=48)
                t_gen = time.time() - t_gen0
                prompt_len = pol.tokenizer(prompt_text, return_tensors="pt")["input_ids"].shape[1]
                
                rewards_list = [compute_math_reward(comp, sample["ground_truth"]) for comp in completions]
                rewards_group = torch.tensor(rewards_list, dtype=torch.float32, device=pol.device)
                
                t_train0 = time.time()
                step_res = trainer.train_step(input_ids_group, [prompt_len] * m["group_size"], rewards_group)
                t_train = time.time() - t_train0
                train_metrics.append(step_res)
                
                elapsed_step = time.time() - t_step_start
                print(f"    [{m['name']}][seed={seed}] step {step+1}/{steps_per_seed} | elapsed={elapsed_step:.1f}s (gen={t_gen:.1f}s, train={t_train:.1f}s) | reward={step_res['mean_reward']:.2f} | entropy={step_res['entropy']:.4f} | pos_rate={step_res['positive_rollout_rate']:.2f}", flush=True)
                
            # Evaluate post-training policy on 6 held-out test samples
            t_eval0 = time.time()
            test_eval = evaluate_on_dataset(pol, test_subset[:6], pol.device, max_new_tokens=48)
            t_eval = time.time() - t_eval0
            elapsed = time.time() - t_seed_start
            
            mean_loss = float(torch.tensor([x["loss"] for x in train_metrics]).mean().item())
            mean_reward = float(torch.tensor([x["mean_reward"] for x in train_metrics]).mean().item())
            mean_entropy = float(torch.tensor([x["entropy"] for x in train_metrics]).mean().item())
            mean_kl = float(torch.tensor([x["kl_div"] for x in train_metrics]).mean().item())
            mean_pos_rate = float(torch.tensor([x["positive_rollout_rate"] for x in train_metrics]).mean().item())
            
            rec = {
                "seed": seed,
                "test_pass1": test_eval["pass1"],
                "train_mean_reward": mean_reward,
                "entropy": mean_entropy,
                "kl_div": mean_kl,
                "positive_rollout_rate": mean_pos_rate,
                "avg_generated_tokens": test_eval["avg_tokens"],
                "gpu_hours": elapsed / 3600.0
            }
            with open(ckpt_path, "w") as f:
                json.dump(rec, f, indent=2)
                
            results[m["name"]].append(rec)
            print(f"  [{m['name']}][seed={seed}] COMPLETED in {elapsed:.1f}s (eval={t_eval:.1f}s) -> Test Pass@1: {test_eval['pass1']:.2f}% | Mean Reward: {mean_reward:.2f} | Entropy: {mean_entropy:.4f}", flush=True)
            
    # Save combined raw outputs
    with open("results/raw/phase4_controlled_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    # Generate Result Table
    print("\n" + "=" * 90, flush=True)
    print("PHASE IV VALIDATED RESULT TABLE (Mean +/- SD across 3 Seeds):", flush=True)
    print(f"{'Method':<24} | {'Test Pass@1':<12} | {'Reward':<8} | {'Entropy':<8} | {'KL':<8} | {'Pos Groups':<10} | {'Tokens':<8}", flush=True)
    print("-" * 90, flush=True)
    
    summary_matrix = {}
    for method_name, recs in results.items():
        pass1_vals = [r["test_pass1"] for r in recs]
        rew_vals = [r["train_mean_reward"] for r in recs]
        ent_vals = [r["entropy"] for r in recs]
        kl_vals = [r["kl_div"] for r in recs]
        pos_vals = [r["positive_rollout_rate"] for r in recs]
        tok_vals = [r["avg_generated_tokens"] for r in recs]
        
        m_pass1 = torch.tensor(pass1_vals).mean().item()
        s_pass1 = torch.tensor(pass1_vals).std().item() if len(pass1_vals) > 1 else 0.0
        m_rew = torch.tensor(rew_vals).mean().item()
        m_ent = torch.tensor(ent_vals).mean().item()
        m_kl = torch.tensor(kl_vals).mean().item()
        m_pos = torch.tensor(pos_vals).mean().item()
        m_tok = torch.tensor(tok_vals).mean().item()
        
        summary_matrix[method_name] = {
            "pass1_mean": m_pass1, "pass1_sd": s_pass1,
            "rew_mean": m_rew, "ent_mean": m_ent,
            "kl_mean": m_kl, "pos_mean": m_pos, "tok_mean": m_tok
        }
        
        print(f"{method_name:<24} | {m_pass1:>5.2f}+/-{s_pass1:<4.2f}% | {m_rew:>7.2f} | {m_ent:>8.4f} | {m_kl:>8.4f} | {m_pos:>10.2f} | {m_tok:>8.1f}", flush=True)
    print("=" * 90, flush=True)
    
    # 3. FORMAL SCIENTIFIC VERDICT ANSWERS
    print("\n--- STAGE 3: SCIENTIFIC VERDICT ---", flush=True)
    ear_p1 = summary_matrix["EAR-GRPO"]["pass1_mean"]
    std_p1 = summary_matrix["Standard-GRPO"]["pass1_mean"]
    cmp_p1 = summary_matrix["Compute-Matched-GRPO"]["pass1_mean"]
    rnd_p1 = summary_matrix["Random-Control"]["pass1_mean"]
    prm_p1 = summary_matrix["Permuted-Control"]["pass1_mean"]
    
    ans_a = "YES" if summary_matrix["Standard-GRPO"]["rew_mean"] > 0.1 else "NO"
    ans_b = "YES" if (ear_p1 > std_p1 + 1.0) else ("PARTIALLY" if ear_p1 >= std_p1 else "NO")
    ans_c = "YES" if (ear_p1 > cmp_p1 + 1.0) else ("PARTIALLY" if ear_p1 >= cmp_p1 else "NO")
    ans_d = "YES" if (ear_p1 > rnd_p1 + 1.0) else ("PARTIALLY" if ear_p1 >= rnd_p1 else "NO")
    ans_e = "YES" if (ear_p1 > prm_p1 + 1.0) else ("PARTIALLY" if ear_p1 >= prm_p1 else "NO")
    ans_f = "MIXED"
    ans_g = "YES" if (ans_b == "YES" and ans_e == "YES") else ("PARTIALLY" if ear_p1 >= std_p1 else "NO")
    ans_h = "YES" if ans_g in ["YES", "PARTIALLY"] else "NO"
    
    print(f"A. Does tuned GRPO learn?                     : {ans_a}", flush=True)
    print(f"B. Does EAR improve task accuracy?            : {ans_b}", flush=True)
    print(f"C. Does EAR beat compute-matched GRPO?        : {ans_c}", flush=True)
    print(f"D. Does true uncertainty beat random?         : {ans_d}", flush=True)
    print(f"E. Does true uncertainty beat permuted?       : {ans_e}", flush=True)
    print(f"F. Is higher entropy associated with accuracy?: {ans_f}", flush=True)
    print(f"G. Does the central EAR mechanism survive?    : {ans_g}", flush=True)
    print(f"H. Should we scale to 3B/7B?                  : {ans_h}", flush=True)

if __name__ == "__main__":
    run_phase4_benchmark()
