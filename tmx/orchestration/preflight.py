from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .spec import ExperimentSpec
from .tpu import production_status


def _command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def _hf_token_present() -> bool:
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True
    return Path.home().joinpath(".cache", "huggingface", "token").is_file()


def _wandb_token_present() -> bool:
    if os.environ.get("WANDB_API_KEY"):
        return True
    netrc = Path.home().joinpath(".netrc")
    if not netrc.is_file():
        return False
    try:
        text = netrc.read_text(errors="ignore")
    except OSError:
        return False
    return "api.wandb.ai" in text and "password" in text


def _gcloud_account() -> str:
    proc = subprocess.run(
        ["gcloud", "config", "get-value", "account"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if proc.returncode != 0:
        return ""
    return (proc.stdout or "").strip()


def run_preflight(spec: ExperimentSpec, *, require_tpu_ready: bool = False) -> dict[str, Any]:
    validation = spec.validate()
    status = production_status(spec)
    checks = {
        "python3": _command_exists("python3"),
        "gcloud": _command_exists("gcloud"),
        "hf_token": _hf_token_present(),
        "wandb_token": _wandb_token_present(),
        "spec_valid": validation.ok,
        "tpu_ready": bool(status.get("tpu", {}).get("ready")),
    }
    errors: list[str] = []
    if not checks["python3"]:
        errors.append("python3 is required")
    if not checks["gcloud"]:
        errors.append("gcloud is required")
    if not checks["hf_token"]:
        errors.append("HF_TOKEN or ~/.cache/huggingface/token is required")
    if not checks["wandb_token"]:
        errors.append("WANDB_API_KEY or a W&B ~/.netrc entry is required")
    if not validation.ok:
        errors.extend(validation.errors)
    if require_tpu_ready and not checks["tpu_ready"]:
        errors.append("TPU is not READY/HEALTHY yet")
    return {
        "ok": not errors,
        "errors": errors,
        "checks": checks,
        "gcloud_account": _gcloud_account() if checks["gcloud"] else "",
        "status": status,
    }
