"""Cross-config tests: every YAML/JSON config under configs/tunix/ must
validate against codex's spec.validate() AND emit overrides cleanly."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tmx.orchestration.spec import load_experiment_spec
from tmx.orchestration.tunix_config import emit_overrides

CONFIGS = sorted(Path("configs/tunix").glob("*.json"))


@pytest.mark.parametrize("config_path", CONFIGS, ids=lambda p: p.name)
def test_config_validates(config_path: Path) -> None:
    spec = load_experiment_spec(config_path)
    result = spec.validate()
    assert result.ok, f"{config_path.name}: errors={list(result.errors)}"


@pytest.mark.parametrize("config_path", CONFIGS, ids=lambda p: p.name)
def test_config_emits_overrides_cleanly(config_path: Path) -> None:
    spec = load_experiment_spec(config_path)
    if float(spec.algorithm.lambda_novelty) > 0.0:
        with pytest.raises(ValueError, match="novelty/RND"):
            emit_overrides(spec)
        return
    overrides = emit_overrides(spec)
    flat = " ".join(overrides)
    assert "model_source=maxtext" in flat
    assert "rollout_engine=" in flat
    assert "rollout_vllm_tpu_backend_type=jax" in flat
    assert "reward_functions=" in flat
    # No GCS leakage anywhere.
    assert "gs://" not in flat


@pytest.mark.parametrize("config_path", CONFIGS, ids=lambda p: p.name)
def test_config_has_no_gcs_or_internal_ip_leakage(config_path: Path) -> None:
    raw = config_path.read_text()
    assert "gs://" not in raw, f"{config_path.name} contains gs:// path"
    assert "internal-ip" not in raw.lower(), f"{config_path.name} contains internal-ip"


@pytest.mark.parametrize("config_path", CONFIGS, ids=lambda p: p.name)
def test_config_dapo_contract_invariants(config_path: Path) -> None:
    """Every Tunix config must honor the DAPO/Dr.GRPO contract."""
    data = json.loads(config_path.read_text())
    alg = data["algorithm"]
    # Tiny smoke configs may relax some knobs but the contract structure
    # (clip 0.2/0.28, KL=0, advantage_normalization=none) is universal.
    assert alg["grpo_clip_epsilon"] == 0.2
    assert alg["grpo_clip_epsilon_high"] == 0.28
    assert alg["grpo_kl_beta"] == 0.0
    assert alg["grpo_advantage_normalization"] == "none"
    assert alg["reward_correct"] == 1.0
    assert alg["reward_incorrect"] == -1.0


@pytest.mark.parametrize("config_path", CONFIGS, ids=lambda p: p.name)
def test_config_storage_is_hf_only(config_path: Path) -> None:
    data = json.loads(config_path.read_text())
    storage = data["storage"]
    assert storage["allow_gcs"] is False
    assert "hf_dataset_repo" in storage
    assert storage["hf_dataset_repo"].startswith("SaifPunjwani/")


@pytest.mark.parametrize("config_path", CONFIGS, ids=lambda p: p.name)
def test_config_network_is_external_ip_no_nat(config_path: Path) -> None:
    data = json.loads(config_path.read_text())
    net = data["network"]
    assert net["external_ips_required"] is True
    assert net["allow_nat"] is False
    assert net["allow_internal_ips"] is False
    assert net["create_tpu"] is False
