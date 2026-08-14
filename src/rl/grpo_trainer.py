"""
Real Transformer GRPO Baseline Trainer
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List
from src.rl.advantage import compute_group_advantages

class GRPOTrainer:
    """
    Standard Group Relative Policy Optimization (GRPO) Trainer for real transformer causal LMs.
    """
    def __init__(
        self,
        policy: nn.Module,
        ref_policy: nn.Module,
        optimizer: optim.Optimizer,
        clip_eps: float = 0.2,
        kl_beta: float = 0.04
    ):
        self.policy = policy
        self.ref_policy = ref_policy
        self.optimizer = optimizer
        self.clip_eps = clip_eps
        self.kl_beta = kl_beta

    def train_step(
        self,
        input_ids_group: torch.Tensor,
        prompt_lengths: List[int],
        rewards_group: torch.Tensor
    ) -> Dict[str, float]:
        """
        Performs single GRPO policy optimization update across a rollout group.
        """
        self.policy.train()
        self.optimizer.zero_grad()
        
        # 1. Standard GRPO advantages
        advantages = compute_group_advantages(rewards_group)
        
        # 2. Forward pass policy and reference policy
        logits = self.policy(input_ids_group)
        with torch.no_grad():
            ref_input_ids = input_ids_group.to(self.ref_policy.device)
            ref_logits = self.ref_policy(ref_input_ids)
            ref_log_probs = self.ref_policy.compute_completion_log_probs(ref_input_ids, prompt_lengths, logits=ref_logits).to(self.policy.device)
            
        log_probs = self.policy.compute_completion_log_probs(input_ids_group, prompt_lengths, logits=logits)
            
        # 3. Ratio and clipped surrogate objective
        old_log_probs = log_probs.detach()
        ratios = torch.exp(log_probs - old_log_probs)
        
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        policy_loss = -torch.mean(torch.min(surr1, surr2))
        
        # 4. KL divergence penalty
        kl_div = torch.mean(log_probs - ref_log_probs)
        loss = policy_loss + self.kl_beta * kl_div
        
        # 5. Backward & optimizer step
        loss.backward()
        self.optimizer.step()
        
        # 6. Policy entropy & reward density calculations
        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1)
            entropy = -torch.mean(torch.sum(probs * torch.log(probs + 1e-9), dim=-1)).item()
            
            num_correct = torch.sum(rewards_group > 0.0).item()
            group_size = rewards_group.shape[0]
            positive_rollout_rate = num_correct / group_size
            frac_zero_correct = 1.0 if num_correct == 0 else 0.0
            frac_one_correct = 1.0 if num_correct == 1 else 0.0
            frac_multi_correct = 1.0 if num_correct > 1 else 0.0
            reward_std = torch.std(rewards_group).item() if group_size > 1 else 0.0
            advantage_std = torch.std(advantages).item() if group_size > 1 else 0.0
            
        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "kl_div": kl_div.item(),
            "mean_reward": torch.mean(rewards_group).item(),
            "entropy": entropy,
            "positive_rollout_rate": positive_rollout_rate,
            "frac_zero_correct": frac_zero_correct,
            "frac_one_correct": frac_one_correct,
            "frac_multi_correct": frac_multi_correct,
            "reward_std": reward_std,
            "advantage_std": advantage_std
        }
