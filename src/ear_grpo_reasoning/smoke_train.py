"""CPU-only deterministic smoke training over the real GRPO objective."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn

from ear_grpo_reasoning.algorithms import (
    compute_ear_advantages,
    compute_group_advantages,
    grpo_loss,
)
from ear_grpo_reasoning.config import ProjectConfig, load_config
from ear_grpo_reasoning.seed import seed_everything


class TinyCausalLM(nn.Module):
    """Minimal causal token model used to verify gradients without external weights."""

    def __init__(self, vocab_size: int = 17, hidden_size: int = 12) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.projection = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: Tensor) -> Tensor:
        return cast(Tensor, self.projection(self.embedding(input_ids)))


def _completion_log_probs(model: nn.Module, input_ids: Tensor) -> Tensor:
    logits = cast(Tensor, model(input_ids[:, :-1]))
    targets = input_ids[:, 1:]
    selected = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return selected


def run_smoke_training(config: ProjectConfig, output_dir: Path) -> dict[str, Any]:
    """Run a tiny optimization and persist diagnostics.

    This checks implementation behavior only. It does not train a language model and
    does not reproduce any manuscript result.
    """

    seed_everything(config.experiment.seed, deterministic=config.experiment.deterministic)
    device = torch.device("cpu")
    policy = TinyCausalLM().to(device)
    old_policy = copy.deepcopy(policy).eval()
    reference_policy = copy.deepcopy(policy).eval()

    input_ids = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6],
            [1, 2, 3, 7, 8, 9],
            [1, 2, 3, 4, 5, 10],
            [1, 2, 3, 7, 11, 12],
        ],
        dtype=torch.long,
        device=device,
    )
    completion_mask = torch.tensor(
        [
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 0],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 0, 0],
        ],
        dtype=torch.bool,
        device=device,
    )
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0], device=device)
    if config.algorithm.mode == "ear":
        uncertainty = torch.tensor([0.1, 0.8, 0.2, 0.7], device=device)
        advantages, weights = compute_ear_advantages(
            rewards,
            uncertainty,
            gamma=config.algorithm.gamma_epistemic,
        )
    else:
        advantages = compute_group_advantages(rewards)
        weights = torch.ones_like(advantages)

    with torch.no_grad():
        old_log_probs = _completion_log_probs(old_policy, input_ids)
        reference_log_probs = _completion_log_probs(reference_policy, input_ids)

    optimizer = torch.optim.Adam(policy.parameters(), lr=config.algorithm.learning_rate)
    history: list[dict[str, float]] = []
    for step in range(config.algorithm.epochs):
        optimizer.zero_grad(set_to_none=True)
        current_log_probs = _completion_log_probs(policy, input_ids)
        output = grpo_loss(
            current_log_probs,
            old_log_probs,
            reference_log_probs,
            advantages,
            completion_mask,
            clip_epsilon=config.algorithm.clip_epsilon,
            kl_beta=config.algorithm.kl_beta,
        )
        torch.autograd.backward(output.loss)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0))
        if not torch.isfinite(torch.tensor(grad_norm)):
            raise RuntimeError("Smoke training produced a non-finite gradient")
        optimizer.step()
        history.append(
            {
                "step": float(step),
                "loss": float(output.loss.detach()),
                "policy_loss": float(output.policy_loss.detach()),
                "kl_penalty": float(output.kl_penalty.detach()),
                "gradient_norm": grad_norm,
            }
        )

    if not history or not all(torch.isfinite(torch.tensor(row["loss"])) for row in history):
        raise RuntimeError("Smoke training produced a non-finite loss")
    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "verification_scope": "software-verification-only",
        "algorithm": config.algorithm.mode,
        "seed": config.experiment.seed,
        "steps": len(history),
        "initial_loss": history[0]["loss"],
        "final_loss": history[-1]["loss"],
        "mean_ear_weight": float(weights.mean()),
        "valid_completion_tokens": int(completion_mask.sum()),
        "config_sha256": config.sha256,
        "history": history,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "smoke-training.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    output_dir = args.output_dir or Path(config.evaluation.output_dir)
    result = run_smoke_training(config, output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
