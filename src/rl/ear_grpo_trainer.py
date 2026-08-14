"""
Real Transformer EAR-GRPO Trainer & Experimental Controls
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, List, Optional
from src.rl.advantage import compute_ear_advantages
from src.models.epistemic_probe import EpistemicUncertaintyProbe

class EARGRPOTrainer:
    """
    Epistemic Advantage Regularized Group Relative Policy Optimization (EAR-GRPO) Trainer
    for real transformer causal LMs. Supports experimental control variants:
    - mode="ear" (Proposed EAR-GRPO using Monte Carlo dropout logit variance)
    - mode="random_control" (Random Gaussian noise uncertainty scaling)
    - mode="permuted_control" (Permuted/shuffled epistemic uncertainty across rollouts)
    """
    def __init__(
        self,
        policy: nn.Module,
        ref_policy: nn.Module,
        optimizer: optim.Optimizer,
        epistemic_probe: EpistemicUncertaintyProbe,
        gamma_epistemic: float = 0.35,
        clip_eps: float = 0.2,
        kl_beta: float = 0.04,
        mode: str = "ear"
    ):
        self.policy = policy
        self.ref_policy = ref_policy
        self.optimizer = optimizer
        self.epistemic_probe = epistemic_probe
        self.gamma_epistemic = gamma_epistemic
        self.clip_eps = clip_eps
        self.kl_beta = kl_beta
        self.mode = mode

    def train_step(
        self,
        input_ids_group: torch.Tensor,
        prompt_lengths: List[int],
        rewards_group: torch.Tensor
    ) -> Dict[str, float]:
        """
        Performs single EAR-GRPO policy optimization update across a rollout group.
        """
        self.policy.train()
        self.optimizer.zero_grad()
        
        # 1. Epistemic uncertainty probing vs Control modes
        if self.mode == "ear":
            epistemic_variances = self.epistemic_probe.compute_epistemic_variance(
                policy=self.policy,
                input_ids=input_ids_group,
                prompt_lengths=prompt_lengths
            )
        elif self.mode == "random_control":
            # Control 1: Random Gaussian uncertainty
            epistemic_variances = torch.abs(torch.randn(input_ids_group.shape[0], device=input_ids_group.device))
        elif self.mode == "permuted_control":
            # Control 2: Permuted/shuffled uncertainty
            real_vars = self.epistemic_probe.compute_epistemic_variance(
                policy=self.policy,
                input_ids=input_ids_group,
                prompt_lengths=prompt_lengths
            )
            perm_indices = torch.randperm(real_vars.shape[0])
            epistemic_variances = real_vars[perm_indices]
        else:
            raise ValueError(f"Unknown trainer mode: {self.mode}")
            
        # 2. Compute Epistemic Advantage Regularized advantages
        ear_advantages, dampening_factors = compute_ear_advantages(
            rewards=rewards_group,
            epistemic_variances=epistemic_variances,
            gamma=self.gamma_epistemic
        )
        
        # 3. Policy forward pass & reference logits
        self.policy.train()
        logits = self.policy(input_ids_group)
        with torch.no_grad():
            ref_input_ids = input_ids_group.to(self.ref_policy.device)
            ref_logits = self.ref_policy(ref_input_ids)
            ref_log_probs = self.ref_policy.compute_completion_log_probs(ref_input_ids, prompt_lengths, logits=ref_logits).to(self.policy.device)
            
        log_probs = self.policy.compute_completion_log_probs(input_ids_group, prompt_lengths, logits=logits)
            
        # 4. Surrogate policy loss with EAR-advantages
        old_log_probs = log_probs.detach()
        ratios = torch.exp(log_probs - old_log_probs)
        
        surr1 = ratios * ear_advantages
        surr2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * ear_advantages
        policy_loss = -torch.mean(torch.min(surr1, surr2))
        
        # 5. KL divergence penalty
        kl_div = torch.mean(log_probs - ref_log_probs)
        loss = policy_loss + self.kl_beta * kl_div
        
        # 6. Backward pass & optimizer step
        loss.backward()
        self.optimizer.step()
        
        # 7. Diagnostics & reward density calculation
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
            advantage_std = torch.std(ear_advantages).item() if group_size > 1 else 0.0
            
        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "kl_div": kl_div.item(),
            "mean_reward": torch.mean(rewards_group).item(),
            "mean_epistemic_var": torch.mean(epistemic_variances).item(),
            "mean_dampening": torch.mean(dampening_factors).item(),
            "entropy": entropy,
            "positive_rollout_rate": positive_rollout_rate,
            "frac_zero_correct": frac_zero_correct,
            "frac_one_correct": frac_one_correct,
            "frac_multi_correct": frac_multi_correct,
            "reward_std": reward_std,
            "advantage_std": advantage_std
        }
