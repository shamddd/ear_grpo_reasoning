"""
PyTest Suite: Real Transformer Epistemic Probe Tests
"""

import torch
import pytest
from src.models.policy import TransformerReasoningPolicy
from src.models.epistemic_probe import EpistemicUncertaintyProbe

def test_real_epistemic_probe_computation():
    model_name = "gpt2"
    policy = TransformerReasoningPolicy(model_name_or_path=model_name, device="auto")
    probe = EpistemicUncertaintyProbe(num_mc_samples=2)
    
    prompts = ["Calculate 15 + 25.", "Find the product of 4 and 8."]
    input_ids, completions = policy.generate_completions(prompts, max_new_tokens=16)
    prompt_lengths = [8, 8]
    
    epistemic_var = probe.compute_epistemic_variance(policy, input_ids, prompt_lengths)
    
    assert epistemic_var.shape == (2,)
    assert torch.all(epistemic_var >= 0.0)
