from pathlib import Path

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
    metrics = trainer.train_step(inputs, [2, 2, 2, 2], torch.tensor([1.0, 0.0, 1.0, 0.0]))
    assert metrics["valid_completion_tokens"] == 12.0
    assert metrics["kl_penalty"] >= 0.0
    assert torch.isfinite(torch.tensor(metrics["loss"]))
