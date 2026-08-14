from pathlib import Path

import pytest

from ear_grpo_reasoning.config import ConfigError, config_from_dict, load_config


def test_all_committed_configs_validate() -> None:
    for path in (
        "configs/smoke.yaml",
        "configs/ear_smoke.yaml",
        "configs/baseline.yaml",
        "configs/ear_grpo.yaml",
        "configs/evaluation.yaml",
    ):
        load_config(path)


def test_config_hash_is_stable_and_sensitive_to_resolved_values() -> None:
    first = load_config("configs/smoke.yaml")
    second = load_config("configs/smoke.yaml")
    changed = config_from_dict({"experiment": {"seed": first.experiment.seed + 1}})
    assert first.sha256 == second.sha256
    assert first.sha256 != changed.sha256


def test_unknown_key_is_rejected() -> None:
    with pytest.raises(ConfigError, match="Unknown key"):
        config_from_dict({"algorithm": {"group_sze": 4}})


def test_invalid_group_relationship_is_rejected() -> None:
    with pytest.raises(ConfigError, match="divisible"):
        config_from_dict({"algorithm": {"group_size": 3, "rollout_count": 4}})


def test_wrong_scalar_type_is_rejected_cleanly() -> None:
    with pytest.raises(ConfigError, match="seed"):
        config_from_dict({"experiment": {"seed": "42"}})


def test_empty_external_revision_is_rejected() -> None:
    with pytest.raises(ConfigError, match=r"model\.revision"):
        config_from_dict({"model": {"revision": ""}})


def test_missing_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="does not exist"):
        load_config(tmp_path / "missing.yaml")
