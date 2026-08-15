"""Strict, hashable experiment configuration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when an experiment configuration is invalid."""


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_keys(data: dict[str, Any], allowed: set[str], section: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ConfigError(f"Unknown key(s) in {section}: {', '.join(unknown)}")


@dataclass(frozen=True)
class ExperimentSettings:
    name: str = "smoke"
    seed: int = 42
    deterministic: bool = True
    device: str = "cpu"

    def validate(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ConfigError("experiment.name must not be empty")
        if not _is_int(self.seed) or self.seed < 0:
            raise ConfigError("experiment.seed must be non-negative")
        if not isinstance(self.deterministic, bool):
            raise ConfigError("experiment.deterministic must be boolean")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ConfigError("experiment.device must be auto, cpu, cuda, or mps")


@dataclass(frozen=True)
class ModelSettings:
    name: str = "builtin-tiny-causal-lm"
    revision: str | None = None
    max_prompt_tokens: int = 64
    max_response_tokens: int = 32
    temperature: float = 0.7
    top_p: float = 0.95

    def validate(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ConfigError("model.name must not be empty")
        if self.revision is not None and (
            not isinstance(self.revision, str) or not self.revision.strip()
        ):
            raise ConfigError("model.revision must be a non-empty string or null")
        if not _is_int(self.max_prompt_tokens) or not _is_int(self.max_response_tokens):
            raise ConfigError("model token limits must be integers")
        if self.max_prompt_tokens < 1 or self.max_response_tokens < 1:
            raise ConfigError("model token limits must be positive")
        if not _is_number(self.temperature) or self.temperature < 0:
            raise ConfigError("model.temperature must be non-negative")
        if not _is_number(self.top_p) or not 0 < self.top_p <= 1:
            raise ConfigError("model.top_p must be in (0, 1]")


@dataclass(frozen=True)
class DataSettings:
    name: str = "builtin-smoke-v1"
    split: str = "verification"
    revision: str | None = None

    def validate(self) -> None:
        if not isinstance(self.name, str) or not isinstance(self.split, str):
            raise ConfigError("data.name and data.split must be strings")
        if not self.name.strip() or not self.split.strip():
            raise ConfigError("data.name and data.split must not be empty")
        if self.revision is not None and (
            not isinstance(self.revision, str) or not self.revision.strip()
        ):
            raise ConfigError("data.revision must be a non-empty string or null")


@dataclass(frozen=True)
class AlgorithmSettings:
    mode: str = "grpo"
    group_size: int = 4
    rollout_count: int = 4
    clip_epsilon: float = 0.2
    kl_beta: float = 0.04
    gamma_epistemic: float = 0.35
    learning_rate: float = 0.01
    batch_size: int = 4
    epochs: int = 4
    gradient_accumulation_steps: int = 1
    checkpoint_interval: int = 0
    evaluation_interval: int = 1

    def validate(self) -> None:
        if self.mode not in {"grpo", "ear"}:
            raise ConfigError("algorithm.mode must be grpo or ear")
        integer_fields = (
            "group_size",
            "rollout_count",
            "batch_size",
            "epochs",
            "gradient_accumulation_steps",
            "checkpoint_interval",
            "evaluation_interval",
        )
        for name in integer_fields:
            if not _is_int(getattr(self, name)):
                raise ConfigError(f"algorithm.{name} must be an integer")
        for name in ("group_size", "rollout_count", "batch_size", "epochs"):
            if getattr(self, name) < 1:
                raise ConfigError(f"algorithm.{name} must be positive")
        if self.rollout_count % self.group_size != 0:
            raise ConfigError("algorithm.rollout_count must be divisible by group_size")
        numeric_fields = ("clip_epsilon", "kl_beta", "gamma_epistemic", "learning_rate")
        if not all(_is_number(getattr(self, name)) for name in numeric_fields):
            raise ConfigError("algorithm floating-point settings must be numeric")
        if not 0 <= self.clip_epsilon < 1:
            raise ConfigError("algorithm.clip_epsilon must be in [0, 1)")
        if self.kl_beta < 0 or self.gamma_epistemic < 0 or self.learning_rate <= 0:
            raise ConfigError("KL, EAR, and learning-rate values must be non-negative/positive")
        if self.gradient_accumulation_steps < 1:
            raise ConfigError("algorithm.gradient_accumulation_steps must be positive")
        if self.checkpoint_interval < 0 or self.evaluation_interval < 1:
            raise ConfigError("checkpoint/evaluation intervals are invalid")


@dataclass(frozen=True)
class EvaluationSettings:
    output_dir: str = "results/generated/evaluation-smoke"
    predictions_path: str | None = None

    def validate(self) -> None:
        if not isinstance(self.output_dir, str) or not self.output_dir.strip():
            raise ConfigError("evaluation.output_dir must not be empty")
        if self.predictions_path is not None and not isinstance(self.predictions_path, str):
            raise ConfigError("evaluation.predictions_path must be a string or null")


@dataclass(frozen=True)
class ProjectConfig:
    experiment: ExperimentSettings = field(default_factory=ExperimentSettings)
    model: ModelSettings = field(default_factory=ModelSettings)
    data: DataSettings = field(default_factory=DataSettings)
    algorithm: AlgorithmSettings = field(default_factory=AlgorithmSettings)
    evaluation: EvaluationSettings = field(default_factory=EvaluationSettings)

    def validate(self) -> None:
        self.experiment.validate()
        self.model.validate()
        self.data.validate()
        self.algorithm.validate()
        self.evaluation.validate()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


_SECTIONS: dict[str, tuple[type[Any], set[str]]] = {
    "experiment": (ExperimentSettings, set(ExperimentSettings.__dataclass_fields__)),
    "model": (ModelSettings, set(ModelSettings.__dataclass_fields__)),
    "data": (DataSettings, set(DataSettings.__dataclass_fields__)),
    "algorithm": (AlgorithmSettings, set(AlgorithmSettings.__dataclass_fields__)),
    "evaluation": (EvaluationSettings, set(EvaluationSettings.__dataclass_fields__)),
}


def config_from_dict(raw: dict[str, Any]) -> ProjectConfig:
    """Create and validate a configuration, rejecting misspelled fields."""

    if not isinstance(raw, dict):
        raise ConfigError("Configuration root must be a mapping")
    _check_keys(raw, set(_SECTIONS), "root")
    values: dict[str, Any] = {}
    for section, (settings_type, allowed) in _SECTIONS.items():
        section_data = raw.get(section, {})
        if not isinstance(section_data, dict):
            raise ConfigError(f"{section} must be a mapping")
        _check_keys(section_data, allowed, section)
        try:
            values[section] = settings_type(**section_data)
        except TypeError as exc:
            raise ConfigError(f"Invalid {section} configuration: {exc}") from exc
    config = ProjectConfig(**values)
    config.validate()
    return config


def load_config(path: str | Path) -> ProjectConfig:
    """Load a UTF-8 YAML configuration from disk."""

    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"Configuration file does not exist: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {config_path}: {exc}") from exc
    return config_from_dict(raw or {})
