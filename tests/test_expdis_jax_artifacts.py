"""Artifact transfers are optional; required handoffs must observe failures."""
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from expdis_jax import artifacts


@pytest.mark.parametrize("source,is_dir", [("local", False), ("local", True), ("gcs", False), ("gcs", True)])
def test_worker_routes_files_and_folders(monkeypatch, tmp_path, source, is_dir):
    calls = []
    api = SimpleNamespace(**{name: (lambda name: lambda **kw: calls.append((name, kw)))(name)
        for name in ("create_repo", "upload_file", "upload_folder")})
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=lambda: api))
    copies = []
    monkeypatch.setattr(artifacts.subprocess, "run", lambda cmd, **kw: copies.append((cmd, kw)))
    env = dict(EXPDIS_HF_UPLOAD_REPO_ID="test/artifacts", EXPDIS_HF_UPLOAD_REPO_TYPE="dataset",
               EXPDIS_HF_UPLOAD_PATH_IN_REPO="run/stage/checkpoint", EXPDIS_HF_UPLOAD_IS_DIR=str(int(is_dir)))
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("EXPDIS_HF_UPLOAD_GCS_URI", raising=False)
    if source == "gcs":
        monkeypatch.setenv("EXPDIS_HF_UPLOAD_GCS_URI", "gs://test-bucket/checkpoint")
    else:
        monkeypatch.setenv("EXPDIS_HF_UPLOAD_LOCAL_PATH", str(tmp_path))
    artifacts._upload_worker()
    assert calls[0][1] == dict(repo_id="test/artifacts", repo_type="dataset", private=True, exist_ok=True)
    assert calls[1][0] == ("upload_folder" if is_dir else "upload_file")
    assert calls[1][1]["path_in_repo"] == "run/stage/checkpoint"
    assert bool(copies) == (source == "gcs")
    if copies:
        assert copies[0][1]["check"] is True
        assert ("--recursive" in copies[0][0]) == is_dir


def test_force_sync_and_required_failure(monkeypatch, tmp_path):
    monkeypatch.setenv("EXPDIS_HF_ARTIFACT_REPO", "test/artifacts")
    monkeypatch.delenv("EXPDIS_HF_CHECKPOINT_REPO", raising=False)
    monkeypatch.setenv("EXPDIS_HF_MIRROR_ASYNC", "1")
    monkeypatch.setenv("EXPDIS_REQUIRE_HF_ARTIFACT_UPLOAD", "0")
    monkeypatch.setenv("EXPDIS_HF_UPLOAD_GCS_URI", "gs://stale-worker-state")
    calls = []
    def run(cmd, **kw):
        calls.append((cmd, kw))
        return SimpleNamespace(returncode=1, stderr="upload failed", stdout="")
    monkeypatch.setattr(artifacts.subprocess, "run", run)
    cfg = SimpleNamespace(output_dir=str(tmp_path))
    assert not artifacts._maybe_upload_path_to_hf(str(tmp_path), cfg, "run/model", is_dir=True, force_sync=True)
    assert calls[0][1]["env"]["EXPDIS_HF_MIRROR_ASYNC"] == "0"
    assert "EXPDIS_HF_UPLOAD_GCS_URI" not in calls[0][1]["env"]
    monkeypatch.setenv("EXPDIS_REQUIRE_HF_ARTIFACT_UPLOAD", "1")
    with pytest.raises(RuntimeError, match="HF upload failed"):
        artifacts._maybe_upload_path_to_hf(str(tmp_path), cfg, "run/model", is_dir=True)


def test_upload_child_does_not_import_jax():
    result = subprocess.run([sys.executable, "-c",
        "import runpy, sys; runpy.run_path(sys.argv[1]); assert 'jax' not in sys.modules",
        str(Path(artifacts.__file__).resolve())], check=True, capture_output=True, text=True)
    assert result.returncode == 0
