from __future__ import annotations

import base64
import contextlib
import io
import json
from pathlib import Path

import pytest

from tmx.orchestration import tunix_workload


@pytest.fixture
def resume_spec_b64() -> str:
    spec = json.loads(
        Path("configs/tunix/tmx_dapo_drgrpo_resume_step100.json").read_text()
    )
    return base64.b64encode(
        json.dumps(spec, separators=(",", ":")).encode("utf-8")
    ).decode("ascii")


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in (
        "TMX_EXPERIMENT_SPEC_JSON_B64",
        "TMX_EXPERIMENT_SPEC_JSON",
        "TMX_MAXTEXT_TUNIX_EXPERIMENTAL_ACK",
        "TMX_DRY_RUN_WORKLOAD",
        "TMX_ALLOW_EXPERIMENTAL_TUNIX_NOVELTY",
        "TMX_APPLY_TPU_RUNTIME_ENV",
        "XLA_FLAGS",
        "JAX_PLATFORMS",
    ):
        monkeypatch.delenv(k, raising=False)


def test_load_spec_from_env_round_trips_b64(
    monkeypatch: pytest.MonkeyPatch, resume_spec_b64: str
) -> None:
    monkeypatch.setenv("TMX_EXPERIMENT_SPEC_JSON_B64", resume_spec_b64)
    spec = tunix_workload._load_spec_from_env()
    assert spec.name == "dapo16k_drgrpo_novelty05_tunix_resume100"
    assert spec.algorithm.lambda_novelty == 0.5
    assert spec.algorithm.max_completion_len == 16384
    assert spec.algorithm.grpo_num_generations == 16
    assert spec.resume.source_step == "000100"


def test_load_spec_requires_env() -> None:
    with pytest.raises(RuntimeError, match="TMX_EXPERIMENT_SPEC_JSON"):
        tunix_workload._load_spec_from_env()


def test_describe_emits_command_and_run_name(
    monkeypatch: pytest.MonkeyPatch, resume_spec_b64: str
) -> None:
    monkeypatch.setenv("TMX_EXPERIMENT_SPEC_JSON_B64", resume_spec_b64)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = tunix_workload.describe()
    assert rc == 0
    payload = json.loads(buf.getvalue())
    assert payload["status"] == "ok"
    assert payload["engine"] == "tunix_maxtext_programmatic_dapo_drgrpo"
    assert payload["run_name"] == "dapo16k_drgrpo_novelty05_tunix_resume100"
    cmd = payload["command"]
    assert any("tmx.orchestration.tunix_main" in s for s in cmd)


def test_run_blocks_without_experimental_ack(
    monkeypatch: pytest.MonkeyPatch, resume_spec_b64: str
) -> None:
    monkeypatch.setenv("TMX_EXPERIMENT_SPEC_JSON_B64", resume_spec_b64)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = tunix_workload.run()
    assert rc == 4
    payload = json.loads(buf.getvalue())
    assert payload["status"] == "blocked"


def test_run_dry_run_with_ack_allows_new_architecture_novelty(
    monkeypatch: pytest.MonkeyPatch, resume_spec_b64: str
) -> None:
    monkeypatch.setenv("TMX_EXPERIMENT_SPEC_JSON_B64", resume_spec_b64)
    monkeypatch.setenv("TMX_MAXTEXT_TUNIX_EXPERIMENTAL_ACK", "1")
    monkeypatch.setenv("TMX_DRY_RUN_WORKLOAD", "1")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = tunix_workload.run()
    assert rc == 0
    payload = json.loads(buf.getvalue())
    assert payload["status"] == "starting"
    assert payload["engine"] == "tunix_maxtext_programmatic_dapo_drgrpo"


def test_run_dispatches_to_programmatic_when_lambda_positive(
    monkeypatch: pytest.MonkeyPatch, resume_spec_b64: str
) -> None:
    """The resume spec has λ=0.5, so workload must dispatch to tunix_main
    (programmatic path with custom rollout for novelty), NOT tunix.cli.grpo_main."""
    monkeypatch.setenv("TMX_EXPERIMENT_SPEC_JSON_B64", resume_spec_b64)
    monkeypatch.setenv("TMX_MAXTEXT_TUNIX_EXPERIMENTAL_ACK", "1")
    monkeypatch.setenv("TMX_ALLOW_EXPERIMENTAL_TUNIX_NOVELTY", "1")
    monkeypatch.setenv("TMX_DRY_RUN_WORKLOAD", "1")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = tunix_workload.run()
    assert rc == 0
    payload = json.loads(buf.getvalue())
    assert payload["engine"] == "tunix_maxtext_programmatic_dapo_drgrpo"
    assert payload["lambda_novelty"] == 0.5
    assert any("tmx.orchestration.tunix_main" in s for s in payload["command"])
    # Novelty env vars should now be set for the reward fn to consume.
    import os
    assert os.environ["TMX_LAMBDA_NOVELTY"] == "0.5"
    assert os.environ["TMX_INCORRECT_NOVELTY_SCALE"] == "1.0"
