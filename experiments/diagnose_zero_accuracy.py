import os
import json
import torch
import time
from typing import Dict, Any, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.data.loader import MathReasoningDataset
from src.rl.rewards import compute_math_reward, extract_answer

def test_model_generation(
    model_name: str,
    dataset: MathReasoningDataset,
    num_samples: int = 30,
    max_tokens_list: List[int] = [16, 64, 128, 256],
    use_chat_template: bool = True
) -> Dict[str, Any]:
    print(f"\n=========================================================================")
    print(f"DIAGNOSING MODEL: {model_name} (Chat Template: {use_chat_template})")
    print(f"=========================================================================")
    
    device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device)
    model.eval()
    
    samples = [dataset[i] for i in range(min(num_samples, len(dataset)))]
    results_by_budget = {}
    
    for max_tokens in max_tokens_list:
        t0 = time.time()
        correct_count = 0
        truncated_count = 0
        has_reasoning_count = 0
        raw_outputs = []
        
        for idx, sample in enumerate(samples):
            question = sample["question"]
            gt = sample["ground_truth"]
            
            if use_chat_template and hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant. Solve the math problem step by step and present the final numerical answer after '#### '."},
                    {"role": "user", "content": question}
                ]
                prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt_text = f"Question: {question}\nAnswer:"
                
            inputs = tokenizer(prompt_text, return_tensors="pt", padding=True).to(device)
            prompt_len = inputs["input_ids"].shape[1]
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
                
            completion_ids = outputs[0, prompt_len:]
            gen_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
            
            if len(completion_ids) >= max_tokens:
                truncated_count += 1
                
            if len(gen_text.strip().split()) > 5:
                has_reasoning_count += 1
                
            reward = compute_math_reward(gen_text, gt)
            if reward > 0.0:
                correct_count += 1
                
            if idx < 5:
                raw_outputs.append({
                    "sample_idx": idx,
                    "question": question,
                    "ground_truth": gt,
                    "prompt_used": prompt_text[:100] + "...",
                    "raw_generation": gen_text,
                    "extracted_answer": extract_answer(gen_text),
                    "reward": reward
                })
                
        latency = time.time() - t0
        pass1 = (correct_count / len(samples)) * 100.0
        truncation_rate = (truncated_count / len(samples)) * 100.0
        
        results_by_budget[f"tokens_{max_tokens}"] = {
            "max_new_tokens": max_tokens,
            "pass1_accuracy": pass1,
            "truncation_rate": truncation_rate,
            "has_reasoning_rate": (has_reasoning_count / len(samples)) * 100.0,
            "latency_seconds": latency,
            "sample_outputs": raw_outputs
        }
        
        print(f"  Budget {max_tokens:>4} tokens -> Pass@1: {pass1:>5.2f}% | Truncated: {truncation_rate:>5.2f}% | Latency: {latency:.2f}s")
        
    return results_by_budget

def main():
    dataset = MathReasoningDataset()
    
    # 1. Test GPT-2 (The Phase II pilot model)
    gpt2_results = test_model_generation("gpt2", dataset, num_samples=20, max_tokens_list=[16, 64, 128], use_chat_template=False)
    
    # 2. Test Qwen2.5-0.5B-Instruct (Real reasoning capable open model)
    qwen_results = test_model_generation("Qwen/Qwen2.5-0.5B-Instruct", dataset, num_samples=20, max_tokens_list=[16, 64, 128, 256], use_chat_template=True)
    
    report = {
        "gpt2_pilot": gpt2_results,
        "qwen2.5_0.5b_instruct": qwen_results
    }
    
    output_path = "research/ZERO_ACCURACY_DIAGNOSTIC_DATA.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
        
    print(f"\nDiagnostic evidence preserved to '{output_path}'.")

if __name__ == "__main__":
    main()
