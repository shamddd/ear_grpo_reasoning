import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.data.loader import MathReasoningDataset
from src.rl.rewards import compute_math_reward, extract_answer

def audit_five():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to("cpu")
    model.eval()
    
    dataset = MathReasoningDataset()
    audit_records = []
    
    print("=" * 75)
    print("MANUAL AUDIT OF 5 GSM8K SANITY EXAMPLES")
    print("=" * 75)
    
    for i in range(5):
        sample = dataset[i]
        q = sample["question"]
        gt = sample["ground_truth"]
        
        messages = [
            {"role": "system", "content": "You are a helpful math assistant. Solve the math problem step by step and present the final numerical answer after '#### '."},
            {"role": "user", "content": q}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
        prompt_len = inputs["input_ids"].shape[1]
        
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            
        full_completion = tokenizer.decode(output[0, prompt_len:], skip_special_tokens=True)
        extracted = extract_answer(full_completion)
        r = compute_math_reward(full_completion, gt)
        
        print(f"\n--- EXAMPLE {i+1} ---")
        print(f"QUESTION         : {q}")
        print(f"GROUND TRUTH     : {gt}")
        print(f"FULL COMPLETION  :\n{full_completion}")
        print(f"EXTRACTED ANSWER : {extracted}")
        print(f"REWARD MATCH     : {r}")
        
        audit_records.append({
            "idx": i,
            "question": q,
            "ground_truth": gt,
            "full_completion": full_completion,
            "extracted_answer": extracted,
            "reward": r
        })
        
    with open("research/FIVE_EXAMPLE_AUDIT_RAW.json", "w") as f:
        json.dump(audit_records, f, indent=2)

if __name__ == "__main__":
    audit_five()
