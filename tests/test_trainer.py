"""
PyTest Suite: Real Transformer Trainer Integration Tests
"""

import torch
import torch.optim as optim
import pytest
from src.models.policy import TransformerReasoningPolicy
from src.models.epistemic_probe import EpistemicUncertaintyProbe
from src.rl.grpo_trainer import GRPOTrainer
from src.rl.ear_grpo_trainer import EARGRPOTrainer

def test_real_transformer_ear_grpo_step():
    model_name = "gpt2"
    policy = TransformerReasoningPolicy(model_name_or_path=model_name, device="auto")
    ref_policy = TransformerReasoningPolicy(model_name_or_path=model_name, device="auto")
    ref_policy.eval()
    
    optimizer = optim.Adam(policy.parameters(), lr=1e-5)
    probe = EpistemicUncertaintyProbe(num_mc_samples=2)
    
    trainer = EARGRPOTrainer(
        policy=policy,
        ref_policy=ref_policy,
        optimizer=optimizer,
        epistemic_probe=probe,
        gamma_epistemic=0.35,
        mode="ear"
    )
    
    prompts = ["What is 2 + 2?", "Solve 10 * 5."]
    # Group size 4 (2 prompts x 2 rollouts per prompt)
    input_ids_group, text_completions = policy.generate_completions(prompts * 2, max_new_tokens=16)
    prompt_lengths = [10, 10, 10, 10]
    rewards_group = torch.tensor([1.0, 0.0, 1.0, 0.0], device=policy.device)
    
    metrics = trainer.train_step(input_ids_group, prompt_lengths, rewards_group)
    
    assert "loss" in metrics
    assert "mean_epistemic_var" in metrics
    assert "mean_dampening" in metrics
    assert not torch.isnan(torch.tensor(metrics["loss"]))
