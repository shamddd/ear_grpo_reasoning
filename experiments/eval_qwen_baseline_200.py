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
    z = 1.95996  # 95% confidence
    denominator = 1 + z**2 / n
    centre_adjusted_probability = (p_hat + z**2 / (2 * n)) / denominator
    adjusted_standard_deviation = math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denominator
    lower_bound = max(0.0, centre_adjusted_probability - z * adjusted_standard_deviation)
    upper_bound = min(1.0, centre_adjusted_probability + z * adjusted_standard_deviation)
    return lower_bound * 100.0, upper_bound * 100.0

def main():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {model_name} on device '{device}' for 200-sample test evaluation...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device)
    model.eval()
    
    dataset = MathReasoningDataset()
    start_idx = 700
    end_idx = 900
    num_samples = end_idx - start_idx
    samples = [dataset[i] for i in range(start_idx, end_idx)]
    
    print(f"Evaluating {len(samples)} held-out GSM8K test samples (indices {start_idx}..{end_idx-1})...")
    correct_count = 0
    truncation_count = 0
    parse_fail_count = 0
    total_generated_tokens = 0
    sample_records = []
    
    t0 = time.time()
    for idx, sample in enumerate(samples):
        question = sample["question"]
        gt = sample["ground_truth"]
        
        messages = [
            {"role": "system", "content": "You are a helpful math assistant. Solve the math problem step by step and present the final numerical answer after '#### '."},
            {"role": "user", "content": question}
        ]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        prompt_len = inputs["input_ids"].shape[1]
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
            
        gen_tokens = outputs[0, prompt_len:]
        num_gen = gen_tokens.shape[0]
        total_generated_tokens += num_gen
        
        if num_gen >= 256:
            truncation_count += 1
            
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        pred_ans = extract_answer(gen_text)
        
        if pred_ans is None:
            parse_fail_count += 1
            
        r = compute_math_reward(gen_text, gt)
        if r > 0.0:
            correct_count += 1
            
        record = {
            "test_idx": start_idx + idx,
            "question": question,
            "ground_truth": gt,
            "pred_answer": pred_ans,
            "generated_length": num_gen,
            "reward": r
        }
        sample_records.append(record)
        
        if (idx + 1) % 25 == 0 or (idx + 1) == len(samples):
            current_acc = (correct_count / (idx + 1)) * 100.0
            print(f"  Processed {idx+1:>3}/{len(samples)} -> Current Pass@1: {current_acc:>5.2f}%", flush=True)
            
    total_time = time.time() - t0
    final_pass1 = (correct_count / len(samples)) * 100.0
    ci_low, ci_high = wilson_score_interval(correct_count, len(samples))
    trunc_rate = (truncation_count / len(samples)) * 100.0
    parse_fail_rate = (parse_fail_count / len(samples)) * 100.0
    avg_gen_tokens = total_generated_tokens / len(samples)
    
    print("\n" + "=" * 75)
    print(f"200-EXAMPLE BASE MODEL HELD-OUT EVALUATION RESULT:")
    print(f"  Model Name            : {model_name}")
    print(f"  Sample Count (n)      : {len(samples)}")
    print(f"  Correct Count         : {correct_count}")
    print(f"  Pass@1 Accuracy       : {final_pass1:.2f}%")
    print(f"  95% Wilson Score CI   : [{ci_low:.2f}%, {ci_high:.2f}%]")
    print(f"  Truncation Rate       : {trunc_rate:.2f}% ({truncation_count}/{len(samples)})")
    print(f"  Parse Failure Rate    : {parse_fail_rate:.2f}% ({parse_fail_count}/{len(samples)})")
    print(f"  Average Generated Tok : {avg_gen_tokens:.1f} tokens")
    print(f"  Total Latency         : {total_time:.2f}s ({total_time/len(samples):.2f}s/sample)")
    print("=" * 75)
    
    out_data = {
        "model_name": model_name,
        "n": len(samples),
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
