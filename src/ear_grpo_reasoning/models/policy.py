"""Hugging Face causal-LM adapter with explicit completion masking."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any, cast

import torch
from torch import Tensor, nn


class TransformerReasoningPolicy(nn.Module):
    """Thin causal-LM wrapper used by compute-intensive research scripts.

    ``transformers`` is imported lazily so the verified offline core has no model
    download or provider dependency.
    """

    def __init__(
        self,
        model_name_or_path: str,
        *,
        revision: str | None = None,
        tokenizer_revision: str | None = None,
        device: str = "auto",
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Model adapters require `pip install -e '.[research]'`") from exc
        resolved_device = self._resolve_device(device)
        tokenizer_kwargs: dict[str, Any] = {}
        if tokenizer_revision or revision:
            tokenizer_kwargs["revision"] = tokenizer_revision or revision
        self.tokenizer: Any = AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        model_kwargs: dict[str, Any] = {}
        if revision:
            model_kwargs["revision"] = revision
        model: Any = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
        model.to(resolved_device)
        self.model: Any = model

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)

    @property
    def device(self) -> torch.device:
        return cast(torch.device, next(self.model.parameters()).device)

    @contextmanager
    def _dropout_mode(self, enabled: bool) -> Iterator[None]:
        states: list[tuple[nn.Dropout, bool]] = []
        if enabled:
            for module in self.model.modules():
                if isinstance(module, nn.Dropout) and module.p > 0:
                    states.append((module, module.training))
                    module.train(True)
        try:
            yield
        finally:
            for module, state in states:
                module.train(state)

    def active_dropout_modules(self) -> list[tuple[str, float]]:
        return [
            (name, module.p)
            for name, module in self.model.named_modules()
            if isinstance(module, nn.Dropout) and module.p > 0
        ]

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        *,
        mc_dropout: bool = False,
    ) -> Tensor:
        if attention_mask is None and self.tokenizer.pad_token_id is not None:
            attention_mask = self.leading_padding_attention_mask(
                input_ids, pad_token_id=self.tokenizer.pad_token_id
            )
        with self._dropout_mode(mc_dropout):
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return cast(Tensor, outputs.logits)

    @staticmethod
    def leading_padding_attention_mask(input_ids: Tensor, *, pad_token_id: int) -> Tensor:
        """Mask only left padding, preserving identical EOS tokens inside sequences."""

        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, tokens]")
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        for row in range(input_ids.shape[0]):
            non_pad = torch.nonzero(input_ids[row] != pad_token_id, as_tuple=False).flatten()
            if non_pad.numel() == 0:
                raise ValueError(f"input row {row} contains only padding")
            attention_mask[row, : int(non_pad[0])] = 0
        return attention_mask

    def generate_completions(
        self,
        prompts: Sequence[str],
        *,
        max_new_tokens: int,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> tuple[Tensor, list[str]]:
        if not prompts:
            raise ValueError("prompts must not be empty")
        encoded = self.tokenizer(list(prompts), return_tensors="pt", padding=True)
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        input_width = int(encoded["input_ids"].shape[1])
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if do_sample:
            generation_kwargs.update({"temperature": temperature, "top_p": top_p})
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.inference_mode():
                outputs = self.model.generate(**encoded, **generation_kwargs)
        finally:
            self.model.train(was_training)
        texts = self.tokenizer.batch_decode(outputs[:, input_width:], skip_special_tokens=True)
        return outputs, list(texts)

    @staticmethod
    def completion_mask(
        input_ids: Tensor,
        completion_starts: Sequence[int],
        *,
        pad_token_id: int | None,
        eos_token_id: int | None,
    ) -> Tensor:
        """Mask target-token positions from completion start through first EOS."""

        if input_ids.ndim != 2 or len(completion_starts) != input_ids.shape[0]:
            raise ValueError("input_ids/completion_starts batch dimensions do not match")
        mask = torch.zeros_like(input_ids[:, 1:], dtype=torch.bool)
        for row, start in enumerate(completion_starts):
            if start < 1 or start >= input_ids.shape[1]:
                raise ValueError(f"invalid completion start for row {row}: {start}")
            targets = input_ids[row, 1:]
            start_target = start - 1
            end_target = targets.numel()
            if eos_token_id is not None:
                eos_positions = torch.nonzero(
                    targets[start_target:] == eos_token_id, as_tuple=False
                ).flatten()
                if eos_positions.numel():
                    end_target = start_target + int(eos_positions[0]) + 1
            if pad_token_id is not None and pad_token_id != eos_token_id:
                pad_positions = torch.nonzero(
                    targets[start_target:end_target] == pad_token_id, as_tuple=False
                ).flatten()
                if pad_positions.numel():
                    end_target = start_target + int(pad_positions[0])
            mask[row, start_target:end_target] = True
        return mask

    def completion_token_log_probs(
        self,
        input_ids: Tensor,
        completion_starts: Sequence[int],
        *,
        logits: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if logits is None:
            logits = self(input_ids)
        if logits.shape[:2] != input_ids.shape:
            raise ValueError("logits and input_ids sequence dimensions must match")
        next_token_log_probs = torch.log_softmax(logits[:, :-1], dim=-1)
        targets = input_ids[:, 1:]
        selected = next_token_log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        mask = self.completion_mask(
            input_ids,
            completion_starts,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return selected, mask

    def compute_completion_log_probs(
        self,
        input_ids: Tensor,
        prompt_lengths: Sequence[int],
        *,
        logits: Tensor | None = None,
    ) -> Tensor:
        """Compatibility helper returning mean completion log probability per row."""

        token_log_probs, mask = self.completion_token_log_probs(
            input_ids, prompt_lengths, logits=logits
        )
        mask_float = mask.to(token_log_probs.dtype)
        return (token_log_probs * mask_float).sum(dim=1) / mask_float.sum(dim=1)
