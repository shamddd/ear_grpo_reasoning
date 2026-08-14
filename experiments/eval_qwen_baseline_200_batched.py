import json
import torch
import time
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.data.loader import MathReasoningDataset
from src.rl.rewards import compute_math_reward, extract_answer

def wilson_score_interval(k, n, confidence=0.95):
    if n == 0:
        return 0.0, 0.0
    p_hat = k / n
    z = 1.95996
    denominator = 1 + z**2 / n
    centre_adjusted_probability = (p_hat + z**2 / (2 * n)) / denominator
    adjusted_standard_deviation = math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denominator
    lower_bound = max(0.0, centre_adjusted_probability - z * adjusted_standard_deviation)
    upper_bound = min(1.0, centre_adjusted_probability + z * adjusted_standard_deviation)
    return lower_bound * 100.0, upper_bound * 100.0

def main():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {model_name} on device '{device}' (Batched Evaluation)...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device)
    model.eval()
    
    dataset = MathReasoningDataset()
    start_idx = 700
    end_idx = 900
    samples = [dataset[i] for i in range(start_idx, end_idx)]
    num_samples = len(samples)
    
    print(f"Evaluating {num_samples} held-out GSM8K test samples (Batched)...")
    batch_size = 16
    correct_count = 0
    truncation_count = 0
    parse_fail_count = 0
    total_generated_tokens = 0
    sample_records = []
    
    t0 = time.time()
    for b_start in range(0, num_samples, batch_size):
        b_end = min(b_start + batch_size, num_samples)
        batch_samples = samples[b_start:b_end]
        
        prompts = []
        for sample in batch_samples:
            messages = [
                {"role": "system", "content": "You are a helpful math assistant. Solve the math problem step by step and present the final numerical answer after '#### '."},
                {"role": "user", "content": sample["question"]}
            ]
            p_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(p_text)
            
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        attention_mask = inputs["attention_mask"]
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
            
        for i, sample in enumerate(batch_samples):
            global_idx = start_idx + b_start + i
            actual_prompt_len = int(torch.sum(attention_mask[i]).item())
            gen_tokens = outputs[i, actual_prompt_len:]
            num_gen = gen_tokens.shape[0]
            total_generated_tokens += num_gen
            
            if num_gen >= 256:
                truncation_count += 1
                
            gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            pred_ans = extract_answer(gen_text)
            
            if pred_ans is None:
                parse_fail_count += 1
                
            r = compute_math_reward(gen_text, sample["ground_truth"])
            if r > 0.0:
                correct_count += 1
                
            sample_records.append({
                "test_idx": global_idx,
                "question": sample["question"],
                "ground_truth": sample["ground_truth"],
                "pred_answer": pred_ans,
                "generated_length": num_gen,
                "reward": r
            })
            
        processed_so_far = b_end
        acc_so_far = (correct_count / processed_so_far) * 100.0
        print(f"  Processed {processed_so_far:>3}/{num_samples} -> Current Pass@1: {acc_so_far:>5.2f}%", flush=True)
        
    total_time = time.time() - t0
    final_pass1 = (correct_count / num_samples) * 100.0
    ci_low, ci_high = wilson_score_interval(correct_count, num_samples)
    trunc_rate = (truncation_count / num_samples) * 100.0
    parse_fail_rate = (parse_fail_count / num_samples) * 100.0
    avg_gen_tokens = total_generated_tokens / num_samples
    
    print("\n" + "=" * 75)
    print(f"200-EXAMPLE BASE MODEL HELD-OUT EVALUATION RESULT (BATCHED):")
    print(f"  Model Name            : {model_name}")
    print(f"  Sample Count (n)      : {num_samples}")
    print(f"  Correct Count         : {correct_count}")
    print(f"  Pass@1 Accuracy       : {final_pass1:.2f}%")
    print(f"  95% Wilson Score CI   : [{ci_low:.2f}%, {ci_high:.2f}%]")
    print(f"  Truncation Rate       : {trunc_rate:.2f}% ({truncation_count}/{num_samples})")
    print(f"  Parse Failure Rate    : {parse_fail_rate:.2f}% ({parse_fail_count}/{num_samples})")
    print(f"  Average Generated Tok : {avg_gen_tokens:.1f} tokens")
    print(f"  Total Latency         : {total_time:.2f}s ({total_time/num_samples:.2f}s/sample)")
    print("=" * 75)
    
    out_data = {
        "model_name": model_name,
        "n": num_samples,
        "correct": correct_count,
        "pass1_accuracy": final_pass1,
        "ci_95_wilson": [ci_low, ci_high],
        "truncation_rate": trunc_rate,
        "parse_failure_rate": parse_fail_rate,
        "average_generated_tokens": avg_gen_tokens,
        "total_latency_sec": total_time,
        "sample_records": sample_records
    }
    with open("research/BASE_MODEL_EVALUATION_200.json", "w") as f:
        json.dump(out_data, f, indent=2)
        
    print("Saved baseline 200-example evidence to 'research/BASE_MODEL_EVALUATION_200.json'.")

if __name__ == "__main__":
    main()
