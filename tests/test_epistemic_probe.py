import pytest
import torch

from ear_grpo_reasoning.models import DegenerateUncertaintyError, EpistemicProbe


class NoDropoutPolicy:
    def active_dropout_modules(self) -> list[tuple[str, float]]:
        return []


class ActiveDropoutPolicy:
    def active_dropout_modules(self) -> list[tuple[str, float]]:
        return [("dropout", 0.1)]

    def __call__(self, input_ids: torch.Tensor, *, mc_dropout: bool) -> torch.Tensor:
        assert mc_dropout
        assert not torch.is_grad_enabled()
        return input_ids.to(torch.float32).unsqueeze(-1)

    def compute_completion_log_probs(
        self,
        input_ids: torch.Tensor,
        prompt_lengths: list[int],
        *,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        del input_ids, prompt_lengths
        return logits.mean(dim=(1, 2))


def test_probe_rejects_degenerate_model_by_default() -> None:
    probe = EpistemicProbe(num_mc_samples=2)
    with pytest.raises(DegenerateUncertaintyError):
        probe.compute_epistemic_variance(
            NoDropoutPolicy(),  # type: ignore[arg-type]
            torch.ones((2, 4), dtype=torch.long),
            [2, 2],
        )


def test_probe_can_report_explicit_zero_control() -> None:
    probe = EpistemicProbe(num_mc_samples=2, on_degenerate="zeros")
    result = probe.compute_epistemic_variance(
        NoDropoutPolicy(),  # type: ignore[arg-type]
        torch.ones((2, 4), dtype=torch.long),
        [2, 2],
    )
    assert torch.equal(result, torch.zeros(2))


def test_probe_disables_autograd_for_mc_passes() -> None:
    result = EpistemicProbe(2).compute_epistemic_variance(
        ActiveDropoutPolicy(),  # type: ignore[arg-type]
        torch.ones((2, 4), dtype=torch.long),
        [2, 2],
    )
    assert torch.equal(result, torch.zeros(2))
