"""
Advantage Estimation & Regularization Algorithms for GRPO and EAR-GRPO
"""

import torch
import torch.nn as nn
from typing import Tuple

def compute_group_advantages(
    rewards: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Computes standard Group Relative Policy Optimization (GRPO) advantages:
    A_i = (r_i - mean(r_group)) / (std(r_group) + eps)
    
    Args:
        rewards: Tensor of shape (group_size,)
        eps: Epsilon constant for numerical stability
        
    Returns:
        advantages: Normalized group advantages of shape (group_size,)
    """
    mean_r = torch.mean(rewards)
    std_r = torch.std(rewards, unbiased=False)
    advantages = (rewards - mean_r) / (std_r + eps)
    return advantages


def compute_ear_advantages(
    rewards: torch.Tensor,
    epistemic_variances: torch.Tensor,
    gamma: float = 0.35,
    eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Computes Epistemic Advantage Regularized GRPO (EAR-GRPO) advantages:
    \tilde{A}_i = A_i * exp( - \gamma * U_e(o_i) / (std(U_e) + eps) )
    
    Args:
        rewards: Tensor of shape (group_size,)
        epistemic_variances: Trajectory epistemic variance tensor of shape (group_size,)
        gamma: Epistemic dampening coefficient
        eps: Numerical stability constant
        
    Returns:
        ear_advantages: Epistemically adjusted advantages of shape (group_size,)
        dampening_factors: Dampening multiplier tensor of shape (group_size,)
    """
    # 1. Compute standard GRPO group advantages
    standard_advantages = compute_group_advantages(rewards, eps=eps)
    
    # 2. Normalize group epistemic uncertainty
    std_u = torch.std(epistemic_variances, unbiased=False)
    norm_u = epistemic_variances / (std_u + eps)
    
    # 3. Exponential epistemic dampening factor
    dampening_factors = torch.exp(-gamma * norm_u)
    
    # 4. Scale advantages
    ear_advantages = standard_advantages * dampening_factors
    return ear_advantages, dampening_factors
