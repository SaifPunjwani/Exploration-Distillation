#!/usr/bin/env python3
"""Shared helpers for the tmx_gpu training package.

Owns the cross-module conventions so grpo_gpu / sft_gpu / pipeline_gpu stay
consistent:

- W&B stage runs: every stage opens its OWN run via init_stage_wandb(), which
  passes reinit="create_new" so a stage run never hijacks (or finishes) the
  pipeline's run when stages execute in-process (older wandb SDKs treat the
  truthy value as the legacy reinit=True, which is also safe). Run ids are
  deterministic (sanitize_wandb_id) so a retried job resumes the same run.
- Metrics parity: stages ALSO append every metric record to a shared
  metrics.jsonl (MetricsLogger) keyed by global_step/phase/round so the
  paper-style curves (wandb_style_curves CSVs) can be rehydrated even if W&B
  is unreachable. PARITY_PHASES maps internal phase names to the CSV phase
  column values (scout_grpo / central_sft / central_grpo).
- Crash recovery: trim_jsonl_to_step / trim_metrics_stage drop rows written
  by a crashed attempt past the resume step, so restarted stages never leave
  duplicate rows for filter_pool / curve rehydration. All state rewrites
  (JSON state files via atomic_write_json, JSONL trims) go through a
  tmp-file + os.replace swap so a crash mid-write can never leave a torn
  file that bricks --resume.
- soft_overlong_penalty: the DAPO soft-overlong penalty
  -min(1, (len - expected)/cache) for len > expected (contract 13107/3277 at
  the 16384 completion budget), shared by the trainer and the harvest pass.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import time
from typing import Any, Dict, Iterable, List, Optional

# Internal phase name -> phase column value in the paper's curve CSVs.
PARITY_PHASES = {
    "scout": "scout_grpo",
    "central": "central_grpo",
    "sft": "central_sft",
}

WANDB_PROJECT_DEFAULT = "two-model-explore-gpu"
WANDB_ENTITY_DEFAULT = None  # set WANDB_ENTITY in the environment
SFT_DATASET_DIGEST_VERSION = "sft_prompt_completion_multiset_v1"
SFT_TRAINING_SEQUENCE_DIGEST_VERSION = "sft_training_sequence_v1"
TRAJECTORY_SOURCE_DIGEST_VERSION = "wandb_trajectory_sources_v1"
ARTIFACT_DOWNLOAD_MANIFEST_VERSION = "wandb_artifact_download_v1"
DIRECTORY_FILE_MANIFEST_VERSION = "directory_file_manifest_v1"

# Immutable upstream identity for every canonical Qwen3-1.7B training arm.
# Keeping the value and loader routing in one shared module prevents GRPO,
# SFT, the pipeline, and managed vLLM from resolving different moving Hub
# heads while also keeping revisions away from local stage handoffs.
CANONICAL_TRAINING_MODEL_ID = "Qwen/Qwen3-1.7B"
CANONICAL_TRAINING_MODEL_REVISION = \
    "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"


def is_local_model_reference(model_ref: str) -> bool:
    """Whether ``model_ref`` names local bytes rather than a Hub repo."""
    value = str(model_ref or "")
    return bool(
        os.path.exists(value)
        or os.path.isabs(value)
        or value.startswith("./")
        or value.startswith("../")
        or value.startswith("~")
    )


def remote_revision_kwargs(
    model_ref: str, model_revision: str | None,
) -> Dict[str, str]:
    """Return an HF revision kwarg only for a remote model reference."""
    revision = str(model_revision or "").strip()
    if not revision or is_local_model_reference(model_ref):
        return {}
    return {"revision": revision}


def canonical_json_sha256(value: Any) -> str:
    """SHA256 of deterministic, whitespace-free JSON."""
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(value: str, *, field: str = "relative_path") -> str:
    """Return a normalized, containment-safe POSIX artifact-relative path."""
    raw = str(value)
    path = pathlib.PurePosixPath(raw)
    if not raw or path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"unsafe {field}: {value!r}")
    normalized = path.as_posix()
    if normalized != raw.replace("\\", "/"):
        raise ValueError(f"non-canonical {field}: {value!r}")
    return normalized


def directory_file_manifest(
    root: str, *, exclude_relative_paths: Iterable[str] = (),
) -> Dict[str, Any]:
    """Hash every regular file below ``root`` without following it outside.

    The manifest intentionally excludes its own eventual state file when the
    caller requests that path, avoiding a circular digest while still binding
    every model/config/tokenizer/optimizer payload byte.
    """
    root_real = os.path.realpath(root)
    if not os.path.isdir(root_real):
        raise ValueError(f"directory manifest root is not a directory: {root!r}")
    excluded = {_safe_relative_path(item) for item in exclude_relative_paths}
    files: List[Dict[str, Any]] = []
    for dirpath, dirnames, filenames in os.walk(root_real, followlinks=False):
        for dirname in dirnames:
            candidate = os.path.join(dirpath, dirname)
            if os.path.islink(candidate) or not os.path.isdir(candidate):
                raise ValueError(
                    f"directory manifest rejects non-directory/symlink: {candidate}")
        dirnames[:] = sorted(dirnames)
        for filename in sorted(filenames):
            path = os.path.join(dirpath, filename)
            if os.path.islink(path) or not os.path.isfile(path):
                raise ValueError(f"directory manifest rejects non-regular file: {path}")
            relative = os.path.relpath(path, root_real).replace(os.sep, "/")
            relative = _safe_relative_path(relative)
            if relative in excluded:
                continue
            files.append({
                "relative_path": relative,
                "sha256": file_sha256(path),
                "size_bytes": os.path.getsize(path),
            })
    if not files:
        raise ValueError(f"directory manifest contains no files: {root!r}")
    normalized = {
        "version": DIRECTORY_FILE_MANIFEST_VERSION,
        "files": sorted(files, key=lambda item: item["relative_path"]),
    }
    return {
        **normalized,
        "sha256": canonical_json_sha256(normalized),
    }


def verify_directory_file_manifest(
    root: str, manifest: Dict[str, Any], *, exclude_relative_paths: Iterable[str] = (),
) -> Dict[str, Any]:
    """Re-hash ``root`` and require byte-for-byte manifest equality."""
    fresh = directory_file_manifest(root, exclude_relative_paths=exclude_relative_paths)
    if fresh != manifest:
        raise ValueError(f"directory file manifest mismatch for {root!r}")
    return fresh


def validate_artifact_download_manifest(
    manifest: Dict[str, Any], *, expected_source: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Validate exact artifact membership, local containment, and file bytes.

    Launchers build these records directly from ``wandb.Artifact.files()`` in
    a freshly cleaned digest-scoped root.  Downstream code revalidates every
    listed file and rejects any root escape, symlink, missing member, extra
    local file, or source-ref/digest mismatch.
    """
    if manifest.get("version") != ARTIFACT_DOWNLOAD_MANIFEST_VERSION:
        raise ValueError("artifact download manifest version mismatch")
    source = manifest.get("source")
    if not isinstance(source, dict) or not source.get("artifact_ref") \
            or not source.get("artifact_digest"):
        raise ValueError("artifact download manifest lacks exact source ref/digest")
    if expected_source is not None and source != expected_source:
        raise ValueError("artifact download manifest source mismatch")
    root = str(manifest.get("download_root", ""))
    root_real = os.path.realpath(root)
    if not root or not os.path.isdir(root_real):
        raise ValueError(f"artifact download root is not a directory: {root!r}")
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise ValueError("artifact download manifest has no files")
    expected_paths: set[str] = set()
    normalized_files: List[Dict[str, Any]] = []
    for item in raw_files:
        if not isinstance(item, dict):
            raise ValueError("artifact file entry must be an object")
        relative = _safe_relative_path(str(item.get("relative_path", "")))
        path = os.path.realpath(os.path.join(root_real, *pathlib.PurePosixPath(relative).parts))
        if os.path.commonpath([root_real, path]) != root_real:
            raise ValueError(f"artifact member escapes download root: {relative!r}")
        if os.path.islink(os.path.join(root_real, *pathlib.PurePosixPath(relative).parts)):
            raise ValueError(f"artifact member is a symlink: {relative!r}")
        if not os.path.isfile(path):
            raise ValueError(f"artifact member missing: {relative!r}")
        expected_sha = str(item.get("sha256", ""))
        if len(expected_sha) != 64 or file_sha256(path) != expected_sha:
            raise ValueError(f"artifact member SHA256 mismatch: {relative!r}")
        size = int(item.get("size_bytes", -1))
        if size < 0 or os.path.getsize(path) != size:
            raise ValueError(f"artifact member size mismatch: {relative!r}")
        if relative in expected_paths:
            raise ValueError(f"duplicate artifact member: {relative!r}")
        expected_paths.add(relative)
        normalized_files.append({
            "relative_path": relative,
            "sha256": expected_sha,
            "size_bytes": size,
        })
    actual_paths: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root_real, followlinks=False):
        for dirname in dirnames:
            candidate = os.path.join(dirpath, dirname)
            if os.path.islink(candidate) or not os.path.isdir(candidate):
                raise ValueError(
                    f"artifact root contains symlink/non-directory: {candidate}")
        dirnames[:] = sorted(dirnames)
        for filename in filenames:
            path = os.path.join(dirpath, filename)
            if os.path.islink(path) or not os.path.isfile(path):
                raise ValueError(f"artifact root contains non-regular file: {path}")
            actual_paths.add(
                os.path.relpath(path, root_real).replace(os.sep, "/"))
    if actual_paths != expected_paths:
        missing = sorted(expected_paths - actual_paths)
        extra = sorted(actual_paths - expected_paths)
        raise ValueError(
            f"artifact root membership mismatch (missing={missing}, extra={extra})")
    normalized = {
        "version": ARTIFACT_DOWNLOAD_MANIFEST_VERSION,
        "source": {
            "artifact_ref": str(source["artifact_ref"]),
            "artifact_digest": str(source["artifact_digest"]),
        },
        "files": sorted(normalized_files, key=lambda item: item["relative_path"]),
    }
    recorded_sha = manifest.get("file_manifest_sha256")
    fresh_sha = canonical_json_sha256(normalized)
    if recorded_sha != fresh_sha:
        raise ValueError("artifact download file-manifest digest mismatch")
    return {**normalized, "file_manifest_sha256": fresh_sha, "download_root": root_real}


def require_immutable_wandb_artifact_ref(value: str, *, field: str) -> str:
    """Require an exact W&B artifact version (``name:vN``), never an alias."""
    ref = str(value)
    if re.search(r":v[0-9]+$", ref) is None:
        raise ValueError(
            f"{field} must be an immutable W&B artifact version ending in :vN; "
            f"got {ref!r}")
    return ref


def stable_artifact_download_identity(download: Dict[str, Any]) -> Dict[str, Any]:
    """Remove the machine-local root while retaining exact artifact bytes."""
    return {
        "version": download["version"],
        "source": download["source"],
        "files": download["files"],
        "file_manifest_sha256": download["file_manifest_sha256"],
    }


def semantic_sft_dataset_digest(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Order- and metadata-independent digest of what SFT actually learns.

    Only prompt/completion text enters the SFT loss.  Sorting canonical row
    payloads makes the digest stable across local paths and harmless JSONL
    reordering while preserving duplicates (a repeated row changes the
    multiset and therefore the digest).
    """
    payloads = sorted(
        json.dumps(
            {
                "prompt_text": str(row.get("prompt_text", "")),
                "completion_text": str(row.get("completion_text", "")),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        for row in rows
    )
    envelope = {"version": SFT_DATASET_DIGEST_VERSION, "rows": payloads}
    return {
        "version": SFT_DATASET_DIGEST_VERSION,
        "row_count": len(payloads),
        "sha256": canonical_json_sha256(envelope),
    }


def semantic_sft_jsonl_digest(path: str) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"expected JSON object in {path}:{line_number}")
            if "prompt_text" not in row or "completion_text" not in row:
                raise ValueError(
                    f"SFT row in {path}:{line_number} lacks prompt_text/completion_text")
            rows.append(row)
    if not rows:
        raise ValueError(f"SFT JSONL contains no rows: {path}")
    return semantic_sft_dataset_digest(rows)


def sft_training_sequence_digest(
    rows: Iterable[Dict[str, Any]], *, epochs: int, seed: int,
) -> Dict[str, Any]:
    """Order-sensitive digest of the exact per-epoch SFT example stream.

    The semantic SFT digest deliberately ignores JSONL order, which is useful
    for identifying the training multiset but insufficient for AdamW: the
    optimizer trajectory depends on the order in which examples are applied.
    This digest mirrors :func:`tmx_gpu.sft_gpu.run_sft`'s
    ``default_rng(seed + epoch).permutation`` schedule over the input JSONL.
    Only prompt/completion text enters each row identity because those are the
    fields consumed by the loss.
    """
    import numpy as np

    row_payloads = [
        json.dumps(
            {
                "prompt_text": str(row.get("prompt_text", "")),
                "completion_text": str(row.get("completion_text", "")),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        for row in rows
    ]
    if not row_payloads:
        raise ValueError("SFT training sequence contains no rows")
    if int(epochs) <= 0:
        raise ValueError("SFT training sequence epochs must be positive")
    row_hashes = [
        hashlib.sha256(payload.encode("utf-8")).hexdigest()
        for payload in row_payloads
    ]
    sequence: List[str] = []
    for epoch in range(int(epochs)):
        order = np.random.default_rng(int(seed) + epoch).permutation(
            len(row_hashes))
        sequence.extend(row_hashes[int(index)] for index in order.tolist())
    normalized = {
        "version": SFT_TRAINING_SEQUENCE_DIGEST_VERSION,
        "seed": int(seed),
        "epochs": int(epochs),
        "input_row_count": len(row_hashes),
        "sequence_length": len(sequence),
        "sequence": sequence,
    }
    return {
        "version": SFT_TRAINING_SEQUENCE_DIGEST_VERSION,
        "seed": int(seed),
        "epochs": int(epochs),
        "input_row_count": len(row_hashes),
        "sequence_length": len(sequence),
        "sha256": canonical_json_sha256(normalized),
    }


def trajectory_source_manifest_digest(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Digest W&B trajectory identities without ephemeral download paths."""
    if manifest.get("version") != TRAJECTORY_SOURCE_DIGEST_VERSION:
        raise ValueError(
            "trajectory source manifest version mismatch: "
            f"{manifest.get('version')!r}")
    artifacts = []
    for artifact in manifest.get("artifacts", []):
        if not artifact.get("artifact_ref") or not artifact.get("artifact_digest"):
            raise ValueError("trajectory source artifact lacks exact ref/digest")
        files = [
            {
                "relative_path": str(item["relative_path"]),
                "sha256": str(item["sha256"]),
            }
            for item in artifact.get("files", [])
        ]
        if not files or any(len(item["sha256"]) != 64 for item in files):
            raise ValueError("trajectory source artifact lacks file SHA256 evidence")
        artifacts.append({
            "artifact_ref": str(artifact["artifact_ref"]),
            "artifact_digest": str(artifact["artifact_digest"]),
            "files": sorted(files, key=lambda item: (item["relative_path"], item["sha256"])),
        })
    normalized = {
        "version": TRAJECTORY_SOURCE_DIGEST_VERSION,
        "artifacts": sorted(
            artifacts, key=lambda item: (item["artifact_ref"], item["artifact_digest"])),
    }
    if not normalized["artifacts"]:
        raise ValueError("trajectory source manifest contains no artifacts")
    return {
        "version": TRAJECTORY_SOURCE_DIGEST_VERSION,
        "artifact_count": len(normalized["artifacts"]),
        "sha256": canonical_json_sha256(normalized),
    }


def verify_trajectory_source_manifest(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Verify every attributed trajectory lies in its clean artifact root."""
    # First validate the stable scientific identity and all claimed hashes.
    digest = trajectory_source_manifest_digest(manifest)
    for artifact in manifest.get("artifacts", []):
        root = str(artifact.get("download_root", ""))
        root_real = os.path.realpath(root)
        if not root or not os.path.isdir(root_real):
            raise ValueError(f"trajectory artifact root missing: {root!r}")
        attributed: set[str] = set()
        for item in artifact.get("files", []):
            relative = _safe_relative_path(str(item.get("relative_path", "")))
            expected_local = os.path.realpath(
                os.path.join(root_real, *pathlib.PurePosixPath(relative).parts))
            if os.path.commonpath([root_real, expected_local]) != root_real:
                raise ValueError(f"trajectory file escapes artifact root: {relative!r}")
            local_path = os.path.realpath(str(item.get("local_path", "")))
            if local_path != expected_local:
                raise ValueError(
                    f"trajectory local_path is not root/relative_path: {local_path!r}")
            if not os.path.isfile(local_path) or os.path.islink(local_path):
                raise ValueError(f"trajectory source file missing/unsafe: {local_path!r}")
            if file_sha256(local_path) != str(item.get("sha256", "")):
                raise ValueError(f"trajectory source SHA mismatch: {local_path!r}")
            attributed.add(local_path)
        if not attributed:
            raise ValueError("trajectory artifact has no attributed files")
    return digest


def sanitize_wandb_id(name: str) -> str:
    """Deterministic W&B run id from a run name (W&B id charset, <=120 chars)."""
    return re.sub(r"[^a-zA-Z0-9_-]", "-", str(name))[:120]


class StageRunStepOffset:
    """Wrap a RESUMED W&B run so re-logged stage steps land past the run's
    last recorded step.

    W&B silently DROPS any row logged at step <= the resumed history's last
    step. A stage rerun that restarts its local step counter (SFT reruns log
    micro-steps from 1; a GRPO stage resumed from an older checkpoint replays
    the crash window) would otherwise lose its whole rerun window from the
    W&B curves. The true global_step is always inside the logged record (and
    in metrics.jsonl), so shifting the W&B step axis never corrupts curve
    rehydration.
    """

    def __init__(self, run, step_offset: int):
        self._run = run
        self.step_offset = int(step_offset)

    def log(self, data, step=None, **kwargs):
        if step is not None:
            step = int(step) + self.step_offset
        return self._run.log(data, step=step, **kwargs)

    def __getattr__(self, name):
        return getattr(self._run, name)


def init_stage_wandb(
    *,
    name: str,
    group: str,
    config: Optional[Dict[str, Any]] = None,
    job_type: Optional[str] = None,
    run_id: Optional[str] = None,
):
    """wandb.init for one pipeline stage as its OWN run.

    reinit="create_new" (wandb >= 0.21) keeps an already-active run (the
    pipeline driver's) untouched: the stage gets a distinct run object and its
    finish() cannot finish the pipeline run. On older SDKs the string is
    truthy and falls back to legacy reinit=True semantics, which equally
    avoids returning the active run. Deterministic id + resume="allow" lets a
    retried job continue the same stage run; a resumed run comes back wrapped
    in StageRunStepOffset so re-logged steps are offset past the run's last
    step instead of being silently dropped by W&B.
    """
    import wandb

    kwargs: Dict[str, Any] = dict(
        project=os.environ.get("WANDB_PROJECT", WANDB_PROJECT_DEFAULT),
        entity=os.environ.get("WANDB_ENTITY") or WANDB_ENTITY_DEFAULT,
        name=name,
        group=group,
        id=sanitize_wandb_id(run_id or name),
        resume="allow",
        config=config or {},
    )
    if job_type:
        kwargs["job_type"] = job_type
    try:
        run = wandb.init(reinit="create_new", **kwargs)
    except TypeError:  # very old SDK: reinit must be bool
        run = wandb.init(reinit=True, **kwargs)
    try:
        last_step = int(getattr(run, "step", 0) or 0)
    except Exception:  # noqa: BLE001 — never let step introspection kill a run
        last_step = 0
    if last_step > 0:
        print(f"[wandb] stage run {name!r} resumed at step {last_step}; "
              "offsetting logged steps so re-logged rows are not dropped", flush=True)
        return StageRunStepOffset(run, last_step)
    return run


def log_dir_artifact(
    wandb_run,
    *,
    name: str,
    path: str,
    type_: str,
    metadata: Optional[Dict[str, Any]] = None,
    aliases: Optional[List[str]] = None,
):
    """Push a directory to W&B and return the logged artifact handle.

    Callers that make durability claims must call ``.wait()`` on the returned
    handle before advancing past the corresponding scientific checkpoint.
    """
    import wandb

    art = wandb.Artifact(
        re.sub(r"[^a-zA-Z0-9_.-]", "-", name), type=type_, metadata=metadata or {}
    )
    art.add_dir(path)
    return wandb_run.log_artifact(art, aliases=aliases or ["latest"])


class MetricsLogger:
    """Append-only JSONL metric log shared by all stages of one pipeline run.

    Each record carries global_step/phase/round (plus arbitrary metric keys)
    so the paper-style curves can be rebuilt offline. Independent of W&B on
    purpose: the Lepton workspace streams W&B live, but the JSONL is what the artifact
    pushes make durable.
    """

    def __init__(self, path: str):
        self.path = str(path)
        parent = os.path.dirname(os.path.abspath(self.path))
        if parent:
            os.makedirs(parent, exist_ok=True)

    def log(self, record: Dict[str, Any]) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")


def atomic_write_json(path: str, obj: Any, *, indent: int = 2) -> None:
    """Write JSON via tmp-file + os.replace so a crash mid-write can never
    leave a torn state file (pipeline_state.json / trainer_state.json torn
    writes permanently brick --resume)."""
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=indent)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_rewrite_lines(path: str, lines: List[str]) -> None:
    """Rewrite a JSONL file via tmp-file + os.replace: a crash during a
    crash-recovery trim must not destroy the durable record it is trimming."""
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        f.writelines(lines)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    """Append rows as JSON lines, dropping underscore-prefixed (private) keys."""
    rows = list(rows)
    if not rows:
        return
    with open(path, "a") as f:
        for row in rows:
            f.write(json.dumps({k: v for k, v in row.items() if not k.startswith("_")}) + "\n")


def trim_jsonl_to_step(path: str, max_step: int, *, step_key: str = "step") -> None:
    """Crash recovery: drop rows a previous attempt wrote past the resume step.

    max_step <= 0 means "starting fresh": the whole old file is rotated aside
    (kept as .stale.<ts> for forensics) instead of being appended to.
    """
    if not os.path.exists(path):
        return
    if max_step <= 0:
        os.replace(path, f"{path}.stale.{int(time.time())}")
        return
    kept: List[str] = []
    dropped = 0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                step = int(json.loads(line).get(step_key, 0) or 0)
            except Exception:  # noqa: BLE001 — malformed line from a crash mid-write
                dropped += 1
                continue
            if step <= max_step:
                kept.append(line)
            else:
                dropped += 1
    if dropped:
        _atomic_rewrite_lines(path, kept)
        print(f"[resume] trimmed {dropped} stale rows (step > {max_step}) from {path}", flush=True)


def trim_metrics_stage(
    path: str,
    *,
    phase: str,
    round_idx: int,
    keep_upto_global_step: Optional[int] = None,
) -> None:
    """Drop a restarted stage's metric rows from the shared metrics JSONL.

    Removes rows matching (phase, round) with global_step past the resume
    point (all such rows when keep_upto_global_step is None), so a rerun
    cannot leave duplicate curve points.
    """
    if not os.path.exists(path):
        return
    kept: List[str] = []
    dropped = 0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:  # noqa: BLE001
                dropped += 1
                continue
            same_stage = rec.get("phase") == phase and int(rec.get("round", -1)) == int(round_idx)
            stale = same_stage and (
                keep_upto_global_step is None
                or int(rec.get("global_step", 0) or 0) > int(keep_upto_global_step)
            )
            if stale:
                dropped += 1
            else:
                kept.append(line)
    if dropped:
        _atomic_rewrite_lines(path, kept)
        print(f"[resume] trimmed {dropped} stale metric rows for phase={phase} "
              f"round={round_idx} from {path}", flush=True)


def soft_overlong_penalty(tok_len: int, expected_len: int, cache_len: int) -> float:
    """DAPO soft-overlong penalty: -min(1, (len - expected)/cache) past expected.

    Contract geometry: expected 13107 / cache 3277 at the 16384 completion
    budget (GPU-MAX budgets split as int(0.8*N) / N - int(0.8*N)).
    """
    if expected_len <= 0 or cache_len <= 0 or tok_len <= expected_len:
        return 0.0
    return -min(1.0, float(tok_len - expected_len) / float(cache_len))
