"""
PyTest Suite: Advantage & Regularization Calculation Tests
"""

import torch
import pytest
from src.rl.advantage import compute_group_advantages, compute_ear_advantages

def test_compute_group_advantages():
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
    advantages = compute_group_advantages(rewards)
    
    assert advantages.shape == (4,)
    assert torch.abs(torch.mean(advantages)) < 1e-5
    assert advantages[0] > 0
    assert advantages[1] < 0

def test_compute_ear_advantages():
    rewards = torch.tensor([1.0, 1.0, 0.0, 0.0])
    # Trajectory 0 has low uncertainty (0.01), Trajectory 1 has high uncertainty (0.95)
    epistemic_vars = torch.tensor([0.01, 0.95, 0.05, 0.10])
    
    ear_adv, dampening = compute_ear_advantages(rewards, epistemic_vars, gamma=0.5)
    
    assert ear_adv.shape == (4,)
    assert dampening.shape == (4,)
    # High uncertainty trajectory 1 should receive smaller advantage than low uncertainty trajectory 0
    assert ear_adv[0] > ear_adv[1]
    assert dampening[0] > dampening[1]
