from pathlib import Path

import pytest
import torch
from torch import Tensor, nn

from ear_grpo_reasoning.config import load_config
from ear_grpo_reasoning.smoke_train import run_smoke_training
from ear_grpo_reasoning.training import GRPOTrainer


class TinyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(13, 8)
        self.projection = nn.Linear(8, 13)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.projection(self.embedding(input_ids))

    def completion_token_log_probs(
        self, input_ids: Tensor, starts: list[int], *, logits: Tensor
    ) -> tuple[Tensor, Tensor]:
        targets = input_ids[:, 1:]
        selected = torch.log_softmax(logits[:, :-1], dim=-1).gather(-1, targets.unsqueeze(-1))
        mask = torch.zeros_like(targets, dtype=torch.bool)
        for row, start in enumerate(starts):
            mask[row, start - 1 :] = True
        return selected.squeeze(-1), mask


def test_deterministic_training_smoke(tmp_path: Path) -> None:
    config = load_config("configs/smoke.yaml")
    first = run_smoke_training(config, tmp_path / "first")
    second = run_smoke_training(config, tmp_path / "second")
    assert first["history"] == second["history"]
    assert first["steps"] == config.algorithm.epochs
    assert first["valid_completion_tokens"] == 9
    assert (tmp_path / "first" / "smoke-training.json").is_file()


def test_ear_training_smoke_exercises_non_neutral_weights(tmp_path: Path) -> None:
    result = run_smoke_training(load_config("configs/ear_smoke.yaml"), tmp_path)
    assert result["algorithm"] == "ear"
    assert 0.0 < result["mean_ear_weight"] < 1.0


def test_model_facing_trainer_uses_masked_objective() -> None:
    torch.manual_seed(5)
    policy = TinyPolicy()
    reference = TinyPolicy()
    reference.load_state_dict(policy.state_dict())
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
    trainer = GRPOTrainer(policy, reference, optimizer)
    assert not reference.training
    assert all(not parameter.requires_grad for parameter in reference.parameters())
    with torch.inference_mode():
        inputs = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 6, 7, 8], [1, 2, 3, 4, 9], [1, 2, 6, 7, 10]])
    reference_before = {
        name: value.detach().clone() for name, value in reference.state_dict().items()
    }
    reference.train()
    metrics = trainer.train_step(inputs, [2, 2, 2, 2], torch.tensor([1.0, 0.0, 1.0, 0.0]))
    assert metrics["valid_completion_tokens"] == 12.0
    assert metrics["kl_penalty"] >= 0.0
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert not reference.training
    assert all(
        torch.equal(reference.state_dict()[name], value) for name, value in reference_before.items()
    )


def test_trainer_detaches_supplied_rollout_log_probs() -> None:
    torch.manual_seed(7)
    policy = TinyPolicy()
    reference = TinyPolicy()
    reference.load_state_dict(policy.state_dict())
    trainer = GRPOTrainer(policy, reference, torch.optim.Adam(policy.parameters(), lr=0.01))
    inputs = torch.tensor([[1, 2, 3, 4], [1, 2, 5, 6]])
    with torch.no_grad():
        logits = policy(inputs)
        captured, _ = policy.completion_token_log_probs(inputs, [2, 2], logits=logits)
    captured = captured.detach().clone().requires_grad_(True)
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.add_(0.1 * torch.randn_like(parameter))
    metrics = trainer.train_step(
        inputs,
        [2, 2],
        torch.tensor([1.0, 0.0]),
        old_token_log_probs=captured,
    )
    assert captured.grad is None
    assert abs(metrics["mean_ratio"] - 1.0) > 1e-5


def test_reference_parameters_cannot_be_in_optimizer() -> None:
    policy = TinyPolicy()
    reference = TinyPolicy()
    optimizer = torch.optim.Adam([*policy.parameters(), *reference.parameters()], lr=0.01)
    with pytest.raises(ValueError, match="must not be in the optimizer"):
        GRPOTrainer(policy, reference, optimizer)


def test_policy_and_reference_cannot_share_parameters() -> None:
    policy = TinyPolicy()
    with pytest.raises(ValueError, match="must not share"):
        GRPOTrainer(policy, policy, torch.optim.Adam(policy.parameters(), lr=0.01))
