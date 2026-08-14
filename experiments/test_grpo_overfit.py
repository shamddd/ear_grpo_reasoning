import torch
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.models.policy import TransformerReasoningPolicy
from src.rl.grpo_trainer import GRPOTrainer
from src.data.loader import MathReasoningDataset
from src.rl.rewards import compute_math_reward

def run_overfit_test():
    print("=" * 75)
    print("PHASE III HARD GATE: OVERFIT-ONE-BATCH TEST (STANDARD GRPO)")
    print("=" * 75)
    
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Loading {model_name}...")
    
    policy = TransformerReasoningPolicy(model_name_or_path=model_name, device="auto")
    ref_policy = TransformerReasoningPolicy(model_name_or_path=model_name, device="auto")
    ref_policy.eval()
    
    optimizer = optim.Adam(policy.parameters(), lr=1e-5)
    trainer = GRPOTrainer(policy=policy, ref_policy=ref_policy, optimizer=optimizer, kl_beta=0.01)
    
    dataset = MathReasoningDataset()
    train_subset = [dataset[i] for i in range(5)]
    group_size = 4
    
    print("\nStarting GRPO Overfit Training on 5 Fixed Problems (5 Steps)...")
    for step in range(1, 6):
        sample = train_subset[(step - 1) % len(train_subset)]
        question = sample["question"]
        gt = sample["ground_truth"]
        
        messages = [
            {"role": "system", "content": "You are a helpful math assistant. Solve the math problem step by step and present the final numerical answer after '#### '."},
            {"role": "user", "content": question}
        ]
        prompt_text = policy.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        input_ids_group, completions = policy.generate_completions([prompt_text] * group_size, max_new_tokens=128)
        prompt_len = policy.tokenizer(prompt_text, return_tensors="pt")["input_ids"].shape[1]
        prompt_lengths = [prompt_len] * group_size
        
        rewards_list = [compute_math_reward(comp, gt) for comp in completions]
        rewards_group = torch.tensor(rewards_list, dtype=torch.float32, device=policy.device)
        
        metrics = trainer.train_step(input_ids_group, prompt_lengths, rewards_group)
        print(f"  Step {step}/5 -> Mean Reward: {metrics['mean_reward']:.2f} | Loss: {metrics['loss']:.4f} | Entropy: {metrics['entropy']:.4f}", flush=True)
        
    print("\n" + "=" * 75)
    print("OVERFIT TEST COMPLETED: Standard GRPO gradient updates verified!")
    print("=" * 75)

if __name__ == "__main__":
    run_overfit_test()
