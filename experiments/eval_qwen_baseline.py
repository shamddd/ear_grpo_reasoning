import json
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.data.loader import MathReasoningDataset
from src.rl.rewards import compute_math_reward, extract_answer

def main():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to("cpu")
    model.eval()
    
    dataset = MathReasoningDataset()
    num_samples = 50
    samples = [dataset[i] for i in range(min(num_samples, len(dataset)))]
    
    print(f"Evaluating base model accuracy across {len(samples)} GSM8K validation samples...")
    correct_count = 0
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
        inputs = tokenizer(prompt_text, return_tensors="pt").to("cpu")
        prompt_len = inputs["input_ids"].shape[1]
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
            
        gen_text = tokenizer.decode(outputs[0, prompt_len:], skip_special_tokens=True)
        pred_ans = extract_answer(gen_text)
        r = compute_math_reward(gen_text, gt)
        
        if r > 0.0:
            correct_count += 1
            
        record = {
            "idx": idx,
            "question": question,
            "ground_truth": gt,
            "pred_answer": pred_ans,
            "raw_completion": gen_text[:200],
            "reward": r
        }
        sample_records.append(record)
        
        if (idx + 1) % 10 == 0 or (idx + 1) == len(samples):
            current_acc = (correct_count / (idx + 1)) * 100.0
            print(f"  Processed {idx+1:>2}/{len(samples)} -> Current Pass@1: {current_acc:>5.2f}%", flush=True)
            
    total_time = time.time() - t0
    final_pass1 = (correct_count / len(samples)) * 100.0
    
    print("\n" + "=" * 70)
    print(f"BASE MODEL EVALUATION RESULT ({model_name}):")
    print(f"  Pass@1 Accuracy : {final_pass1:.2f}% ({correct_count}/{len(samples)})")
    print(f"  Total Latency   : {total_time:.2f}s ({total_time/len(samples):.2f}s/sample)")
    print("=" * 70)
    
    out_data = {
        "model_name": model_name,
        "num_samples": len(samples),
        "pass1_accuracy": final_pass1,
        "total_latency_sec": total_time,
        "sample_records": sample_records
    }
    with open("research/BASE_MODEL_EVALUATION_QWEN.json", "w") as f:
        json.dump(out_data, f, indent=2)
        
    print("Saved baseline evaluation evidence to 'research/BASE_MODEL_EVALUATION_QWEN.json'.")

if __name__ == "__main__":
    main()
