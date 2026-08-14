import os
import json
import time
import math
import re
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.models.policy import TransformerReasoningPolicy
from src.models.epistemic_probe import EpistemicUncertaintyProbe
from src.data.loader import MathReasoningDataset
from src.rl.rewards import compute_math_reward, extract_answer

torch.set_num_threads(8)

def run_dropout_and_stochasticity_audit():
    print("=" * 90, flush=True)
    print("PHASE V-M: UNCERTAINTY ESTIMATOR VALIDITY & DROPOUT ARCHITECTURE AUDIT", flush=True)
    print("=" * 90, flush=True)
    
    device = "cpu"
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Auditing Model: {model_name} on {device}", flush=True)
    
    pol = TransformerReasoningPolicy(model_name_or_path=model_name, device=device)
    
    # 1. ENUMERATE DROPOUT MODULES
    print("\n--- STEP 1: DROPOUT ARCHITECTURE ENUMERATION ---", flush=True)
    dropout_modules = []
    for name, module in pol.model.named_modules():
        if isinstance(module, nn.Dropout):
            dropout_modules.append((name, module.p))
            
    print(f"Total nn.Dropout modules found in pol.model: {len(dropout_modules)}", flush=True)
    for name, p in dropout_modules:
        print(f"  - Module: {name} | p = {p}", flush=True)
        
    config = pol.model.config
    print(f"Config attention_dropout: {getattr(config, 'attention_dropout', None)}", flush=True)
    print(f"Config hidden_dropout: {getattr(config, 'hidden_dropout', None)}", flush=True)
    print(f"Config classifier_dropout: {getattr(config, 'classifier_dropout', None)}", flush=True)
    
    # Check wrapper dropout
    wrapper_dropouts = [(name, m.p) for name, m in pol.named_modules() if isinstance(m, nn.Dropout)]
    print(f"Total nn.Dropout modules in TransformerReasoningPolicy wrapper: {len(wrapper_dropouts)}", flush=True)
    for name, p in wrapper_dropouts:
        print(f"  - Wrapper Module: {name} | p = {p}", flush=True)
        
    # 2. MEASURE ACTUAL STOCHASTICITY ACROSS M FORWARD PASSES
    print("\n--- STEP 2: STOCHASTICITY & VARIANCE MEASUREMENT ACROSS M PASSES ---", flush=True)
    dataset = MathReasoningDataset()
    test_100 = [dataset[i] for i in range(700, 800)]
    
    # Generate 100 trajectories
    prompts = []
    for s in test_100:
        messages = [
            {"role": "system", "content": "You are a helpful math assistant. Solve the math problem step by step and present the final numerical answer after '#### '."},
            {"role": "user", "content": s["question"]}
        ]
        prompts.append(pol.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        
    # Generate trajectories
    print("Generating 100 sample trajectories for stochastic probing...", flush=True)
    trajectories = []
    for i, p_text in enumerate(prompts):
        inputs = pol.tokenizer(p_text, return_tensors="pt").to(device)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            outputs = pol.model.generate(
                **inputs,
                max_new_tokens=144,
                do_sample=False,
                pad_token_id=pol.tokenizer.pad_token_id
            )
        gen_tokens = outputs[0, prompt_len:]
        text = pol.tokenizer.decode(gen_tokens, skip_special_tokens=True)
        pred = extract_answer(text)
        is_corr = (compute_math_reward(text, test_100[i]["ground_truth"]) > 0.0)
        trajectories.append({
            "idx": i,
            "prompt_text": p_text,
            "input_ids": outputs,
            "prompt_len": prompt_len,
            "gen_text": text,
            "pred": pred,
            "correct": is_corr,
            "gen_len": len(gen_tokens)
        })
        
    # Run stochasticity test for M in [2, 3, 5, 10, 20]
    M_values = [2, 3, 5, 10, 20]
    stochasticity_results = {}
    
    for M in M_values:
        var_seq_logprobs = []
        max_abs_diffs = []
        mean_abs_diffs = []
        non_identical_counts = 0
        
        for t in trajectories[:50]: # First 50 trajectories
            inp = t["input_ids"]
            p_len = [t["prompt_len"]]
            
            pass_logprobs = []
            pass_logits = []
            for _ in range(M):
                # Run with mc_dropout=True
                logits = pol(inp, mc_dropout=True)
                lp = pol.compute_completion_log_probs(inp, p_len, logits=logits)
                pass_logprobs.append(lp.item())
                pass_logits.append(logits.detach())
                
            lp_tensor = torch.tensor(pass_logprobs)
            seq_var = torch.var(lp_tensor, unbiased=True).item()
            var_seq_logprobs.append(seq_var)
            
            # Check logit differences
            l0 = pass_logits[0]
            for m_idx in range(1, M):
                diff = torch.abs(pass_logits[m_idx] - l0)
                max_d = torch.max(diff).item()
                mean_d = torch.mean(diff).item()
                max_abs_diffs.append(max_d)
                mean_abs_diffs.append(mean_d)
                if max_d > 1e-6:
                    non_identical_counts += 1
                    
        mean_seq_var = float(np.mean(var_seq_logprobs))
        mean_max_diff = float(np.mean(max_abs_diffs))
        mean_avg_diff = float(np.mean(mean_abs_diffs))
        
        stochasticity_results[f"M={M}"] = {
            "mean_seq_logprob_variance": mean_seq_var,
            "mean_max_logit_diff": mean_max_diff,
            "mean_avg_logit_diff": mean_avg_diff,
            "non_identical_passes": non_identical_counts
        }
        print(f"  [M={M:>2}] Seq LogProb Var: {mean_seq_var:.10f} | Max Logit Diff: {mean_max_diff:.10f} | Non-identical Passes: {non_identical_counts}", flush=True)
        
    # 3. BENCHMARK UNCERTAINTY PROXIES
    print("\n--- STEP 3: BENCHMARKING CANDIDATE UNCERTAINTY SIGNALS & COMPLEXITY CONFOUNDS ---", flush=True)
    
    proxy_records = []
    
    for t in trajectories:
        inp = t["input_ids"]
        p_len = t["prompt_len"]
        text = t["gen_text"]
        is_corr = 1.0 if t["correct"] else 0.0
        
        # Token predictive entropy & NLL & Margin
        with torch.no_grad():
            logits = pol(inp, mc_dropout=False)
            gen_logits = logits[0, p_len-1:-1, :] # Logits predicting gen tokens
            gen_ids = inp[0, p_len:]
            
            probs = torch.softmax(gen_logits, dim=-1)
            log_probs = torch.log_softmax(gen_logits, dim=-1)
            
            # 1. Token Predictive Entropy (mean across tokens)
            token_entropies = -torch.sum(probs * log_probs, dim=-1)
            mean_token_entropy = torch.mean(token_entropies).item()
            
            # 2. Sequence Mean Negative Log Probability (NLL)
            tok_nll = -log_probs[torch.arange(len(gen_ids)), gen_ids]
            mean_nll = torch.mean(tok_nll).item()
            
            # 3. Logit Margin (Top 1 prob - Top 2 prob)
            top2_probs, _ = torch.topk(probs, k=2, dim=-1)
            margin = torch.mean(top2_probs[:, 0] - top2_probs[:, 1]).item()
            uncertainty_margin = 1.0 - margin # Higher margin uncertainty = smaller gap
            
        # 4. Self-consistency across K=4 sampled rollouts
        k_inputs = pol.tokenizer([t["prompt_text"]] * 4, return_tensors="pt", padding=True).to(device)
        with torch.inference_mode():
            k_outputs = pol.model.generate(
                **k_inputs,
                max_new_tokens=144,
                do_sample=True,
                temperature=0.7,
                pad_token_id=pol.tokenizer.pad_token_id
            )
        k_preds = []
        k_attn = k_inputs["attention_mask"]
        for ki in range(4):
            k_plen = int(torch.sum(k_attn[ki]).item())
            k_text = pol.tokenizer.decode(k_outputs[ki, k_plen:], skip_special_tokens=True)
            k_preds.append(extract_answer(k_text))
            
        # Self-consistency disagreement: 1.0 - (frequency of modal answer / K)
        valid_preds = [p for p in k_preds if p is not None]
        if len(valid_preds) > 0:
            modal_count = max([valid_preds.count(p) for p in set(valid_preds)])
            sc_disagreement = 1.0 - (modal_count / 4.0)
        else:
            sc_disagreement = 1.0
            
        # 5. Complexity features
        char_len = len(text)
        tok_len = t["gen_len"]
        num_arithmetic_ops = len(re.findall(r'[\+\-\*\/=]', text))
        num_equations = len(re.findall(r'=', text))
        has_fractions = 1.0 if re.search(r'\d+\s*/\s*\d+', text) else 0.0
        num_parentheses = len(re.findall(r'[\(\)]', text))
        
        proxy_records.append({
            "idx": t["idx"],
            "correct": is_corr,
            "token_entropy": mean_token_entropy,
            "mean_nll": mean_nll,
            "margin_uncertainty": uncertainty_margin,
            "sc_disagreement": sc_disagreement,
            "token_length": float(tok_len),
            "num_arithmetic_ops": float(num_arithmetic_ops),
            "num_equations": float(num_equations),
            "has_fractions": has_fractions,
            "num_parentheses": float(num_parentheses)
        })
        
    # 4. COMPUTE CORRELATIONS & CONFOUNDING ANALYSIS
    print("\n--- STEP 4: CORRELATIONS WITH CORRECTNESS VS REASONING COMPLEXITY ---", flush=True)
    
    proxies = ["token_entropy", "mean_nll", "margin_uncertainty", "sc_disagreement"]
    complexity_vars = ["token_length", "num_arithmetic_ops", "num_equations", "has_fractions"]
    
    correlation_table = {}
    for p in proxies:
        p_vals = np.array([r[p] for r in proxy_records])
        corr_vals = np.array([r["correct"] for r in proxy_records])
        
        # Pearson / Spearman correlation with correctness (error correlation = -r_corr)
        r_correctness = float(np.corrcoef(p_vals, corr_vals)[0, 1])
        r_error = -r_correctness
        
        comp_corrs = {}
        for c in complexity_vars:
            c_vals = np.array([r[c] for r in proxy_records])
            r_comp = float(np.corrcoef(p_vals, c_vals)[0, 1])
            comp_corrs[c] = r_comp
            
        # Length-controlled partial correlation with correctness
        # r_{xy.z} = (r_xy - r_xz * r_yz) / sqrt((1 - r_xz^2)*(1 - r_yz^2))
        l_vals = np.array([r["token_length"] for r in proxy_records])
        r_pl = float(np.corrcoef(p_vals, l_vals)[0, 1])
        r_cl = float(np.corrcoef(corr_vals, l_vals)[0, 1])
        denom = math.sqrt(max(1e-8, (1 - r_pl**2) * (1 - r_cl**2)))
        partial_r_correct_given_len = (r_correctness - r_pl * r_cl) / denom
        
        correlation_table[p] = {
            "r_with_error": r_error,
            "r_with_correctness": r_correctness,
            "r_with_length": comp_corrs["token_length"],
            "r_with_arithmetic_ops": comp_corrs["num_arithmetic_ops"],
            "r_with_equations": comp_corrs["num_equations"],
            "r_with_fractions": comp_corrs["has_fractions"],
            "partial_r_with_correctness_controlling_length": partial_r_correct_given_len
        }
        
        print(f"Proxy: {p:<20} | r(Error): {r_error:>+6.3f} | r(Length): {comp_corrs['token_length']:>+6.3f} | Partial r(Corr|Len): {partial_r_correct_given_len:>+6.3f}", flush=True)
        
    out_data = {
        "stochasticity_results": stochasticity_results,
        "dropout_modules": dropout_modules,
        "correlation_analysis": correlation_table,
        "proxy_records": proxy_records
    }
    with open("results/raw/uncertainty_proxy_audit.json", "w") as f:
        json.dump(out_data, f, indent=2)
        
    print("\n[COMPLETE] Audit results saved to results/raw/uncertainty_proxy_audit.json", flush=True)

if __name__ == "__main__":
    run_dropout_and_stochasticity_audit()
