"""Top-level pipeline driver: Explorer → trajectories → SFT → Actual GRPO → AIME eval.

Usage:
    EXPDIS_VLLM_SERVER_URLS=http://ip:8000/v1,... \
    python -m expdis_jax.pipeline --wandb-run-name pipeline_lambda0_xxx ...
"""

from __future__ import annotations

import os
import json
import subprocess
import hashlib
import shlex
import shutil
import sys
import time
import gc
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, replace
from typing import List

import numpy as np

from .artifacts import (
    _hf_artifact_repo, _maybe_upload_path_to_hf, _maybe_upload_gcs_uri_to_hf,
)
from .config import TrainConfig, parse_args, validate_contract
from .filtering import QUALITY_POOL_POLICY
from .lineage import novelty_weight_for_round, split_budget
from .checkpointing import checkpoint_path

# Script paths are resolved relative to this file so `python -m expdis_jax.pipeline`
# works from any working directory.
_EXPDIS_JAX_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_EXPDIS_JAX_DIR)



def _env_flag(name: str, default: str = "0") -> bool:
    value = (os.environ.get(name, default) or "").strip().lower()
    return value not in ("", "0", "false", "no", "off")


def _validate_experimental_mrme_gate(cfg: TrainConfig) -> None:
    """Require per-batch sampler synchronization for split-budget stages.

    Stage and round handoffs are exercised by the two-process CPU pipeline
    tests. Native K>1 runs share one process group and execute sequentially;
    parallel breadth requires separately allocated trainer/serving groups.
    """

    experimental_geometry = (
        int(cfg.scouts_per_round) != 1
        or int(cfg.num_rounds) != 1
        or bool(str(cfg.round_novelty_schedule or "").strip())
    )
    if not experimental_geometry:
        return
    reload_every = (
        os.environ.get("EXPDIS_TRAIN_VLLM_RELOAD_EVERY_STEPS", "1") or "1"
    ).strip()
    if reload_every != "1":
        raise RuntimeError(
            "TPU MultiScout/MR-ME requires the default "
            "EXPDIS_TRAIN_VLLM_RELOAD_EVERY_STEPS=1 (sampler synchronized before "
            "every rollout batch); a longer cadence can leave short split-budget "
            "stages sampling from stale weights"
        )


def _validate_multihost_stage_boundary(process_count: int, cfg: TrainConfig | None = None) -> None:
    """Distributed stage checkpoints must live on explicitly shared storage."""

    if cfg is not None and cfg.checkpoint_root:
        return

    if int(process_count) > 1 and not _env_flag(
        "EXPDIS_PIPELINE_STOP_AFTER_EXPLORER", "0"
    ):
        raise RuntimeError(
            "multihost TPU pipeline runs must set "
            "EXPDIS_PIPELINE_STOP_AFTER_EXPLORER=1; pooled filtering, SFT, and "
            "Central RL handoffs require --checkpoint-root pointing to shared "
            "storage accessible to every trainer host"
        )


def _is_source_process() -> bool:
    """Whether this process owns host-side rollout/filter I/O."""

    try:
        import jax

        return int(jax.process_index()) == 0
    except Exception:
        return True


def _broadcast_source_bytes(payload: bytes | None, *, chunk_bytes: int = 4 << 20) -> bytes:
    """Broadcast source-host bytes to every JAX process with a SHA-256 check.

    This is used only at small stage boundaries (the <=500-row SFT library and
    compact summaries), never for full raw Scout pools.  It lets multihost SFT
    consume identical local bytes even when rollout files exist only on rank 0.
    """

    try:
        import jax
        from jax.experimental import multihost_utils
    except Exception:
        if payload is None:
            raise RuntimeError("source payload is unavailable outside a JAX process group") from None
        return payload

    if int(jax.process_count()) <= 1:
        if payload is None:
            raise RuntimeError("source payload is missing")
        return payload

    is_source = int(jax.process_index()) == 0
    if is_source and payload is None:
        raise RuntimeError("source process did not provide bytes")
    source_payload = payload or b""
    header = np.asarray(
        [len(source_payload), max(1, int(chunk_bytes))],
        dtype=np.int64,
    )
    header = np.asarray(
        multihost_utils.broadcast_one_to_all(header, is_source=is_source),
        dtype=np.int64,
    )
    total_size, chunk_bytes = int(header[0]), int(header[1])
    if total_size < 0 or chunk_bytes < 1:
        raise RuntimeError("invalid distributed file header")

    expected_digest = (
        np.frombuffer(hashlib.sha256(source_payload).digest(), dtype=np.uint8)
        if is_source
        else np.zeros((32,), dtype=np.uint8)
    )
    expected_digest = bytes(
        np.asarray(
            multihost_utils.broadcast_one_to_all(
                expected_digest,
                is_source=is_source,
            ),
            dtype=np.uint8,
        ).tolist()
    )

    received = bytearray()
    for offset in range(0, total_size, chunk_bytes):
        width = min(chunk_bytes, total_size - offset)
        chunk = np.zeros((chunk_bytes,), dtype=np.uint8)
        if is_source:
            chunk[:width] = np.frombuffer(source_payload[offset : offset + width], dtype=np.uint8)
        chunk = np.asarray(
            multihost_utils.broadcast_one_to_all(chunk, is_source=is_source),
            dtype=np.uint8,
        )
        received.extend(chunk[:width].tobytes())
    result = bytes(received)
    if len(result) != total_size or hashlib.sha256(result).digest() != expected_digest:
        raise RuntimeError("distributed stage-boundary byte replication failed integrity check")
    return result


def _replicate_source_file(path: str) -> str:
    """Replicate one source-host file atomically to the same local path on all hosts."""

    source_payload = None
    if _is_source_process():
        with open(path, "rb") as handle:
            source_payload = handle.read()
    payload = _broadcast_source_bytes(source_payload)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd, temp_path = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".replica-",
        dir=os.path.dirname(os.path.abspath(path)),
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    return path


def _broadcast_source_json(payload: dict | None) -> dict:
    encoded = None
    if payload is not None:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    decoded = json.loads(_broadcast_source_bytes(encoded).decode("utf-8"))
    if not isinstance(decoded, dict):
        raise RuntimeError("distributed summary payload is not a JSON object")
    return decoded


def _gcs_artifacts_enabled() -> bool:
    backend = (os.environ.get("EXPDIS_ARTIFACT_BACKEND") or "hf").strip().lower()
    allow_gcs = _env_flag("EXPDIS_ALLOW_GCS_ARTIFACTS", "0")
    return allow_gcs and backend in ("gcs", "dual", "both", "gcs+hf")


def _pipeline_wandb_run_name(cfg: TrainConfig) -> str:
    return cfg.wandb_run_name or "jax_pipeline"


def _pipeline_wandb_run_id(cfg: TrainConfig) -> str:
    if os.environ.get("EXPDIS_WANDB_RUN_ID"):
        return os.environ["EXPDIS_WANDB_RUN_ID"].strip()
    run_name = _pipeline_wandb_run_name(cfg)
    return "jaxe2e-" + hashlib.sha1(f"{run_name}:e2e".encode("utf-8")).hexdigest()[:24]


def _scout_wandb_run_id(cfg: TrainConfig, scout_index: int) -> str:
    run_name = _pipeline_wandb_run_name(cfg)
    digest = hashlib.sha1(
        f"{run_name}:round={int(cfg.round_index)}:scout={int(scout_index)}".encode("utf-8")
    ).hexdigest()[:24]
    return f"jaxscout{digest}"


def _scout_artifact_prefix(cfg: TrainConfig, scout_index: int) -> str:
    base = _hf_run_prefix(cfg)
    return f"{base}/scout_{int(scout_index):02d}"


def _wandb_log_pipeline(cfg: TrainConfig, payload: dict, *, step: int | None = None) -> None:
    """Log pipeline-level metrics to a stable companion W&B run.

    Stage trainers own their own W&B runs. This helper mirrors the Torch/XLA
    pipeline-level metrics without keeping a W&B run open while sub-stages call
    ``wandb.init`` themselves.
    """
    if _env_flag("EXPDIS_DISABLE_WANDB_NONZERO", "1"):
        process_id = (os.environ.get("JAX_PROCESS_ID") or "0").strip()
        try:
            import jax

            if jax.distributed.is_initialized():
                process_id = str(int(jax.process_index()))
        except Exception:
            pass
        if process_id and process_id != "0":
            return
    try:
        import wandb
    except Exception as e:
        print(f"[pipeline] wandb pipeline logging disabled: {e}", flush=True)
        return

    run_name = _pipeline_wandb_run_name(cfg)
    run_id = _pipeline_wandb_run_id(cfg)
    try:
        run = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity or None,
            name=run_name,
            id=run_id,
            resume="allow",
            reinit=True,
            group=os.environ.get("EXPDIS_WANDB_GROUP") or run_name,
            job_type="pipeline",
            config=asdict(cfg),
        )
        run.log(payload, step=step)
        run.finish()
    except Exception as e:
        print(f"[pipeline] WARNING: W&B pipeline log failed: {e}", flush=True)


def _summarize_trajectory_jsonl(path: str, *, num_generations: int) -> dict:
    rows: list[dict] = []
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    if not rows:
        return {"traj/total_count": 0.0}

    def _values(key: str) -> list[float]:
        out = []
        for row in rows:
            try:
                out.append(float(row.get(key, 0.0) or 0.0))
            except Exception:
                out.append(0.0)
        return out

    def _mean(vals: list[float]) -> float:
        return float(sum(vals) / max(1, len(vals)))

    def _percentile(vals: list[float], pct: float) -> float:
        if not vals:
            return 0.0
        xs = sorted(vals)
        idx = min(len(xs) - 1, max(0, int(round((pct / 100.0) * (len(xs) - 1)))))
        return float(xs[idx])

    rewards = _values("blended_reward")
    correctness_rewards = _values("correctness_reward")
    novelty_rewards = _values("novelty_reward")
    novelty_raw = _values("r_novel_raw")
    lengths = _values("completion_token_length")
    correct = [1.0 if bool(row.get("is_correct", False)) else 0.0 for row in rows]
    clipped = [1.0 if bool(row.get("clipped", False)) else 0.0 for row in rows]
    terminated = [1.0 if bool(row.get("terminated", False)) else 0.0 for row in rows]
    valid_answer = [1.0 if bool(row.get("valid_answer", False)) else 0.0 for row in rows]
    terminated_idx = [i for i, v in enumerate(terminated) if v > 0.5]
    prompt_any = []
    if num_generations > 0:
        for i in range(0, len(correct), num_generations):
            group = correct[i:i + num_generations]
            if group:
                prompt_any.append(1.0 if max(group) > 0.5 else 0.0)

    payload = {
        "traj/total_count": float(len(rows)),
        "traj/correct_count": float(sum(correct)),
        "traj/reward_mean": _mean(rewards),
        "traj/correct_fraction": _mean(correct),
        "traj/prompt_any_correct_fraction": _mean(prompt_any),
        "traj/valid_answer_fraction": _mean(valid_answer),
        "traj/clipped_fraction": _mean(clipped),
        "traj/terminated_fraction": _mean(terminated),
        "traj/completion_token_length_mean": _mean(lengths),
        "traj/completion_token_length_max": float(max(lengths) if lengths else 0.0),
        "traj/correctness_reward_mean": _mean(correctness_rewards),
        "traj/novelty_reward_mean": _mean(novelty_rewards),
        "traj/novelty_raw_mean": _mean(novelty_raw),
        "traj/terminated_count": float(len(terminated_idx)),
    }
    if terminated_idx:
        payload.update({
            "traj/terminated_reward_mean": _mean([rewards[i] for i in terminated_idx]),
            "traj/terminated_correctness_mean": _mean([correctness_rewards[i] for i in terminated_idx]),
            "traj/terminated_novelty_mean": _mean([novelty_rewards[i] for i in terminated_idx]),
        })
    else:
        payload.update({
            "traj/terminated_reward_mean": 0.0,
            "traj/terminated_correctness_mean": 0.0,
            "traj/terminated_novelty_mean": 0.0,
        })
    return payload


def _merge_scout_trajectory_files(paths: List[str], output_path: str) -> str:
    """Merge K source-host Scout pools once, recording immutable source identity."""

    if not paths:
        raise ValueError("at least one Scout trajectory path is required")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fd, temp_path = tempfile.mkstemp(
        prefix=os.path.basename(output_path) + ".merge-",
        dir=os.path.dirname(os.path.abspath(output_path)),
    )
    row_count = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as destination:
            for scout_index, path in enumerate(paths, start=1):
                if not os.path.isfile(path):
                    raise FileNotFoundError(f"Scout {scout_index} trajectory pool is missing: {path}")
                with open(path, encoding="utf-8") as source:
                    for line_number, line in enumerate(source, start=1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise ValueError(
                                f"invalid Scout JSON at {path}:{line_number}"
                            ) from exc
                        if not isinstance(row, dict):
                            raise ValueError(f"Scout row is not an object at {path}:{line_number}")
                        row.setdefault("source_file", os.path.abspath(path))
                        row.setdefault("scout_index", scout_index)
                        destination.write(json.dumps(row, sort_keys=True) + "\n")
                        row_count += 1
            destination.flush()
            os.fsync(destination.fileno())
        if row_count < 1:
            raise RuntimeError("merged Scout trajectory pool is empty")
        os.replace(temp_path, output_path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    print(
        f"[pipeline] merged {len(paths)} Scout pools -> {output_path} ({row_count} rows)",
        flush=True,
    )
    return output_path


def _materialize_existing_trajectory(source: str, destination: str) -> str:
    """Materialize one explicit local/HF/GCS Scout pool at ``destination``."""

    source = str(source or "").strip()
    if not source:
        if not os.path.isfile(destination):
            raise FileNotFoundError(f"existing trajectory pool is missing: {destination}")
        return destination
    os.makedirs(os.path.dirname(os.path.abspath(destination)), exist_ok=True)
    if source.startswith("hf://"):
        from huggingface_hub import hf_hub_download

        raw = source[len("hf://") :].strip("/")
        parts = raw.split("/")
        if len(parts) < 3:
            raise ValueError(
                "HF trajectory URI must be hf://owner/repo/path/to/file; "
                f"got {source!r}"
            )
        downloaded = hf_hub_download(
            repo_id="/".join(parts[:2]),
            repo_type=os.environ.get("EXPDIS_HF_REPO_TYPE", "dataset") or "dataset",
            filename="/".join(parts[2:]),
        )
        shutil.copyfile(downloaded, destination)
    elif source.startswith("gs://"):
        if not _gcs_artifacts_enabled() and not _env_flag(
            "EXPDIS_ALLOW_GCS_EXISTING_TRAJECTORIES", "0"
        ):
            raise RuntimeError(
                "Refusing gs:// existing trajectories in HF-only mode. Use hf:// "
                "or set EXPDIS_ALLOW_GCS_EXISTING_TRAJECTORIES=1 explicitly."
            )
        result = subprocess.run(
            ["gsutil", "cp", source, destination],
            capture_output=True,
            text=True,
            timeout=900,
        )
        if result.returncode != 0:
            raise RuntimeError(f"failed to fetch existing trajectories: {result.stderr[:500]}")
    else:
        source_path = os.path.abspath(source)
        if source_path != os.path.abspath(destination):
            shutil.copyfile(source_path, destination)
    if not os.path.isfile(destination):
        raise FileNotFoundError(f"trajectory materialization failed: {destination}")
    return destination


def collect_accepted(
    cfg: TrainConfig,
    trajectory_paths: str | List[str],
    *,
    output_path: str | None = None,
) -> str:
    """Run the complete QualityPool contract once on one or more Scout pools."""

    from .filtering import QUALITY_POOL_POLICY, pool_trajectory_files

    paths = [trajectory_paths] if isinstance(trajectory_paths, str) else list(trajectory_paths)
    if not paths:
        raise ValueError("at least one Scout trajectory path is required")
    policy = str(cfg.accepted_selection_policy or "").strip().lower()
    if policy not in {
        "coverage_pool_c8",
        "quality_pool_mrme_v2",
        QUALITY_POOL_POLICY,
    }:
        raise ValueError(
            "the canonical JAX pipeline requires the complete one-row-per-problem "
            f"QualityPool; got accepted_selection_policy={cfg.accepted_selection_policy!r}"
        )

    max_selected = int(cfg.max_accepted_trajectories or cfg.max_distill_examples)
    max_selected = min(max_selected, int(cfg.max_distill_examples), 500)
    if max_selected < 1:
        raise ValueError("accepted trajectory cap must be in [1, 500]")
    if output_path is None:
        output_path = (
            paths[0].replace(".jsonl", ".accepted.jsonl")
            if len(paths) == 1
            else os.path.join(
                os.path.dirname(os.path.commonpath([os.path.abspath(item) for item in paths])),
                "trajectory_library.accepted.jsonl",
            )
        )
    _selected, stats = pool_trajectory_files(
        paths,
        output_path=output_path,
        max_examples=max_selected,
        min_completion_tokens=int(cfg.accepted_min_completion_tokens),
        max_completion_tokens=int(cfg.accepted_max_completion_tokens),
    )
    accepted_count = int(stats["accepted"])
    min_required = int(cfg.min_accepted_trajectories)
    if accepted_count < min_required:
        raise RuntimeError(
            f"Only {accepted_count} accepted trajectories are available, below "
            f"min_accepted_trajectories={min_required}. Continue Scout collection "
            "or preregister a lower-dose run."
        )
    stats_path = output_path + ".funnel.json"
    with open(stats_path, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, sort_keys=True, indent=2)
        handle.write("\n")
    print(
        f"[pipeline] QualityPool raw={stats['raw']} clean={stats['clean_correct']} "
        f"unique={stats['unique_clean_problems']} accepted={accepted_count} "
        f"cap={max_selected} -> {output_path}",
        flush=True,
    )
    return output_path


def _maybe_export_state_to_hf_dir(state, cfg: TrainConfig, output_dir: str) -> str | None:
    """Export a trained Flax/JAX state to an HF-style model directory.

    Every host gathers its parameter shards; only process 0 writes the bundle.
    """
    try:
        import jax
        import numpy as np
        from .train import _make_qwen3_config
        from .weights import export_flax_params_to_hf_dir
    except Exception as e:
        print(f"[pipeline] WARNING: cannot import JAX/HF export helpers: {e}", flush=True)
        return None

    if state is None or not hasattr(state, "params"):
        print("[pipeline] WARNING: no train state available for HF export.", flush=True)
        return None

    from jax.experimental import multihost_utils
    from .checkpointing import broadcast_source_object
    params_host = multihost_utils.process_allgather(state.params) if jax.process_count() > 1 else state.params
    out, error = None, None
    if jax.process_index() == 0:
        try:
            m_cfg = _make_qwen3_config(cfg.model_name)
            params_host = jax.tree_util.tree_map(lambda x: np.asarray(x), params_host)
            out = os.path.abspath(output_dir)
            export_flax_params_to_hf_dir(
                params_host, cfg.model_name, int(m_cfg.num_hidden_layers),
                bool(m_cfg.tie_word_embeddings), out,
                use_qk_norm=bool(m_cfg.use_qk_norm),
                hf_weight_prefix=str(m_cfg.hf_weight_prefix),
            )
            print(f"[pipeline] exported HF model bundle -> {out}", flush=True)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    error = broadcast_source_object(error)
    if error:
        if _env_flag("EXPDIS_DAPO_DRGRPO_FULL_CONTRACT", "1"):
            raise RuntimeError(f"required final HF export failed: {error}")
        print(f"[pipeline] WARNING: HF export failed: {error}", flush=True)
        return None
    return out


def _maybe_upload_dir_to_gcs(local_dir: str, gcs_target: str) -> str | None:
    if not _gcs_artifacts_enabled():
        print(f"[pipeline] skipped GCS dir upload; HF-only artifacts active: {gcs_target}", flush=True)
        return None
    try:
        src = shlex.quote(os.path.abspath(local_dir).rstrip("/")) + "/*"
        dst = shlex.quote(gcs_target.rstrip("/") + "/")
        ret = subprocess.run(
            ["bash", "-lc", f"gcloud storage cp --recursive {src} {dst}"],
            capture_output=True,
            text=True,
            timeout=3600,
        )
        if ret.returncode == 0:
            print(f"[pipeline] uploaded {local_dir} -> {gcs_target}", flush=True)
            return gcs_target
        print(f"[pipeline] WARNING: GCS upload failed: {ret.stderr[:300]}", flush=True)
    except Exception as e:
        print(f"[pipeline] WARNING: GCS upload error: {e}", flush=True)
    return None


def _hf_run_prefix(cfg: TrainConfig) -> str:
    return (os.environ.get("EXPDIS_HF_ARTIFACT_PREFIX") or _pipeline_wandb_run_name(cfg)).strip().strip("/")


def _maybe_upload_file_to_gcs(local_path: str, gcs_target: str) -> bool:
    if not _gcs_artifacts_enabled():
        print(f"[pipeline] skipped GCS file upload; HF-only artifacts active: {gcs_target}", flush=True)
        return False
    try:
        ret = subprocess.run(
            ["gcloud", "storage", "cp", os.path.abspath(local_path), gcs_target],
            capture_output=True,
            text=True,
            timeout=900,
        )
        if ret.returncode == 0:
            print(f"[pipeline] uploaded {local_path} -> {gcs_target}", flush=True)
            return True
        print(f"[pipeline] WARNING: file upload failed: {ret.stderr[:300]}", flush=True)
    except Exception as e:
        print(f"[pipeline] WARNING: file upload error: {e}", flush=True)
    return False


def _write_run_summary(root_output_dir: str, cfg: TrainConfig, section: str, payload: dict) -> None:
    """Persist run-level metrics/config in JSON and mirror to HF.

    This is the durable, human-readable artifact to inspect alongside W&B:
    config, official baseline, trajectory summaries, SFT/Actual completion, and
    final eval values are merged into one file as the pipeline progresses.
    """
    os.makedirs(root_output_dir, exist_ok=True)
    path = os.path.join(root_output_dir, "run_summary.json")
    summary = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                summary = json.load(f)
        except Exception:
            summary = {}
    summary.setdefault("run_name", _pipeline_wandb_run_name(cfg))
    summary.setdefault("wandb_run_id", _pipeline_wandb_run_id(cfg))
    summary.setdefault("config", asdict(cfg))
    summary.setdefault("contract", {
        "model_name": cfg.model_name,
        "dataset_name": cfg.dataset_name,
        "max_prompt_len": cfg.max_prompt_len,
        "max_completion_len": cfg.max_completion_len,
        "max_total_len": cfg.max_total_len,
        "grpo_batch_size": cfg.grpo_batch_size,
        "grpo_grad_accum": cfg.grpo_grad_accum,
        "grpo_num_generations": cfg.grpo_num_generations,
        "grpo_loss_type": cfg.grpo_loss_type,
        "grpo_loss_aggregation": cfg.grpo_loss_aggregation,
        "dr_grpo_loss_normalizer": cfg.dr_grpo_loss_normalizer,
        "optimizer": cfg.optimizer,
        "grpo_clip_epsilon": cfg.grpo_clip_epsilon,
        "grpo_clip_epsilon_high": cfg.grpo_clip_epsilon_high,
        "grpo_advantage_normalization": cfg.grpo_advantage_normalization,
        "lambda_novelty": cfg.lambda_novelty,
        "reward_correct": cfg.reward_correct,
        "reward_incorrect": cfg.reward_incorrect,
        "mask_truncated_completions": cfg.mask_truncated_completions,
        "soft_overlong_expected_len": cfg.soft_overlong_expected_len,
        "soft_overlong_cache_len": cfg.soft_overlong_cache_len,
        "novelty_zscore": cfg.novelty_zscore,
        "correctness_zscore": cfg.correctness_zscore,
        "aime24_eval_rollouts": cfg.benchmark_num_rollouts,
        "aime24_eval_completion_len": cfg.benchmark_rollout_max_completion_len,
    })
    summary.setdefault("updated_at_unix", time.time())
    summary["updated_at_unix"] = time.time()
    summary[section] = payload
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")
    gcs_base = (os.environ.get("EXPDIS_GCS_CHECKPOINT_BASE") or "").strip().rstrip("/")
    gcs_target = ""
    uploaded_to_gcs = False
    if gcs_base and _gcs_artifacts_enabled():
        gcs_target = f"{gcs_base}/run_summary.json"
        uploaded_to_gcs = _maybe_upload_file_to_gcs(path, gcs_target)
    if _env_flag("EXPDIS_HF_MIRROR_RUN_SUMMARY", "1"):
        hf_path = f"{_hf_run_prefix(cfg)}/run_summary.json"
        if uploaded_to_gcs and gcs_target:
            _maybe_upload_gcs_uri_to_hf(gcs_target, cfg, hf_path, is_dir=False)
        else:
            _maybe_upload_path_to_hf(path, cfg, hf_path, is_dir=False)


def _maybe_reload_vllm_slice(
    gcs_model_path: str | None,
    cfg: TrainConfig,
    *,
    max_model_len: int,
    max_num_seqs: int,
    stage: str,
    hf_repo_id: str = "",
    hf_path_in_repo: str = "",
    hf_repo_type: str = "dataset",
    hf_revision: str = "",
) -> str | None:
    """Reload configured TPU vLLM workers with a newly exported HF model.

    This keeps Actual GRPO and final AIME generation on the trained policy
    instead of silently continuing to sample from the base model.
    """
    tpu_name = (os.environ.get("EXPDIS_VLLM_RELOAD_TPU_NAME") or "").strip()
    zone = (os.environ.get("EXPDIS_VLLM_RELOAD_ZONE") or "").strip()
    workers = (os.environ.get("EXPDIS_VLLM_RELOAD_WORKERS") or "").strip()
    required = _env_flag("EXPDIS_EXTERNAL_VLLM_RELOAD_REQUIRED", "1")
    retry_on_fail = _env_flag("EXPDIS_EXTERNAL_VLLM_RELOAD_RETRY_ON_FAIL", "1")
    if not (tpu_name and zone and workers):
        msg = f"vLLM slice reload env not set for {stage}"
        if required:
            raise RuntimeError(msg)
        print(f"[pipeline] WARNING: {msg}; continuing with existing vLLM", flush=True)
        return None

    server_urls = (
        os.environ.get("EXPDIS_VLLM_RELOAD_SERVER_URLS")
        or os.environ.get("EXPDIS_VLLM_SERVER_URLS")
        or cfg.vllm_server_urls
    ).strip()
    mode = (
        os.environ.get("EXPDIS_PIPELINE_VLLM_RELOAD_MODE")
        or os.environ.get("EXPDIS_TRAIN_VLLM_RELOAD_MODE")
        or "direct"
    ).strip().lower()
    if mode == "external":
        return _request_external_pipeline_vllm_reload(
            gcs_model_path,
            cfg,
            stage=stage,
            max_model_len=max_model_len,
            server_urls=server_urls,
            hf_repo_id=hf_repo_id,
            hf_path_in_repo=hf_path_in_repo,
            hf_repo_type=hf_repo_type,
            hf_revision=hf_revision,
        )
    env = os.environ.copy()
    env.update(
        {
            "GCS_MODEL_PATH": gcs_model_path or "",
            "HF_REPO_ID": hf_repo_id,
            "HF_REPO_TYPE": hf_repo_type,
            "HF_PATH_IN_REPO": hf_path_in_repo,
            "HF_REVISION": hf_revision,
            "TPU_NAME": tpu_name,
            "ZONE": zone,
            "WORKERS": workers,
            "SERVED_MODEL_NAME": cfg.model_name,
            "LOCAL_MODEL_DIR_BASENAME": f"expdis_{stage}_model",
            "MAX_MODEL_LEN": str(int(max_model_len)),
            "MAX_NUM_SEQS": str(int(max_num_seqs)),
            "MAX_NUM_BATCHED_TOKENS": os.environ.get("EXPDIS_VLLM_RELOAD_MAX_BATCHED_TOKENS", "65536"),
            "VLLM_ENABLE_CHUNKED_PREFILL": os.environ.get("VLLM_ENABLE_CHUNKED_PREFILL", "1"),
            "PARALLELISM": os.environ.get("EXPDIS_VLLM_RELOAD_PARALLELISM", "8"),
        }
    )
    timeout_s = float(os.environ.get("EXPDIS_EXTERNAL_VLLM_RELOAD_TIMEOUT", "0") or "0")
    retry_s = float(os.environ.get("EXPDIS_EXTERNAL_VLLM_RELOAD_RETRY_SECONDS", "30") or "30")
    attempt_timeout_s = float(os.environ.get("EXPDIS_VLLM_RELOAD_ATTEMPT_TIMEOUT", "7200") or "7200")
    deadline = time.time() + timeout_s if timeout_s > 0 else None
    attempt = 0
    while deadline is None or time.time() < deadline:
        attempt += 1
        try:
            print(
                f"[pipeline] vLLM slice reload attempt {attempt} for {stage}: workers={workers}",
                flush=True,
            )
            ret = subprocess.run(
                ["bash", os.path.join(_EXPDIS_JAX_DIR, "scripts", "reload_vllm_direct_slice.sh")],
                capture_output=True,
                text=True,
                timeout=attempt_timeout_s,
                env=env,
            )
            if ret.returncode == 0:
                print(f"[pipeline] reloaded vLLM slice for {stage}: workers={workers}", flush=True)
                return server_urls or None
            msg = f"vLLM slice reload failed for {stage}: {(ret.stderr or ret.stdout)[-2000:]}"
        except Exception as e:
            msg = f"vLLM slice reload error for {stage}: {e}"

        if required and retry_on_fail:
            print(
                f"[pipeline] WARN {msg}; blocking until reload succeeds so Actual never uses stale/base vLLM",
                flush=True,
            )
            time.sleep(retry_s)
            continue
        if required:
            raise RuntimeError(msg)
        print(f"[pipeline] WARNING: {msg}", flush=True)
        return None

    msg = f"vLLM slice reload timed out for {stage}"
    if required:
        raise TimeoutError(msg)
    print(f"[pipeline] WARNING: {msg}", flush=True)
    return None


def _request_external_pipeline_vllm_reload(
    gcs_model_path: str | None,
    cfg: TrainConfig,
    *,
    stage: str,
    max_model_len: int,
    server_urls: str,
    hf_repo_id: str = "",
    hf_path_in_repo: str = "",
    hf_repo_type: str = "dataset",
    hf_revision: str = "",
) -> str | None:
    """Ask the local reload watcher to reload pipeline-level HF exports."""
    tpu_name = (os.environ.get("EXPDIS_VLLM_RELOAD_TPU_NAME") or "").strip()
    zone = (os.environ.get("EXPDIS_VLLM_RELOAD_ZONE") or "").strip()
    workers = (os.environ.get("EXPDIS_VLLM_RELOAD_WORKERS") or "").strip()
    required = _env_flag("EXPDIS_EXTERNAL_VLLM_RELOAD_REQUIRED", "1")
    if not (tpu_name and zone and workers):
        msg = f"external vLLM reload env not set for pipeline stage={stage}"
        if required:
            raise RuntimeError(msg)
        print(f"[pipeline] WARNING: {msg}; continuing with existing vLLM", flush=True)
        return None

    request_dir = os.path.join(cfg.output_dir, "vllm_reload_requests")
    os.makedirs(request_dir, exist_ok=True)
    stem = f"{stage}_pipeline"
    request_path = os.path.join(request_dir, f"{stem}.request.json")
    done_path = os.path.join(request_dir, f"{stem}.done")
    fail_path = os.path.join(request_dir, f"{stem}.fail")
    attempts_path = os.path.join(request_dir, f"{stem}.attempts")
    last_error_path = os.path.join(request_dir, f"{stem}.last_error")
    for path in (done_path, fail_path, attempts_path, last_error_path):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

    payload = {
        "gcs_model_path": gcs_model_path or "",
        "hf_repo_id": hf_repo_id,
        "hf_repo_type": hf_repo_type,
        "hf_path_in_repo": hf_path_in_repo,
        "hf_revision": hf_revision,
        "stage": stage,
        "step": 0,
        "model_name": cfg.model_name,
        "max_model_len": int(max_model_len),
        "tpu_name": tpu_name,
        "tpu_project": (
            os.environ.get("EXPDIS_VLLM_RELOAD_TPU_PROJECT")
            or os.environ.get("TPU_PROJECT")
            or os.environ.get("CLOUDSDK_CORE_PROJECT")
            or ""
        ),
        "zone": zone,
        "workers": workers,
        "server_urls": server_urls or cfg.vllm_server_urls,
        "request_path": request_path,
        "done_path": done_path,
        "fail_path": fail_path,
        "attempts_path": attempts_path,
        "last_error_path": last_error_path,
        "created_unix": time.time(),
    }
    tmp_path = f"{request_path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp_path, request_path)
    print(f"[pipeline] external vLLM reload request written: {request_path}", flush=True)

    if not _env_flag("EXPDIS_EXTERNAL_VLLM_RELOAD_WAIT", "1"):
        return server_urls or None
    timeout_s = float(os.environ.get("EXPDIS_EXTERNAL_VLLM_RELOAD_TIMEOUT", "0") or "0")
    poll_s = float(os.environ.get("EXPDIS_EXTERNAL_VLLM_RELOAD_POLL_SECONDS", "5") or "5")
    deadline = time.time() + timeout_s if timeout_s > 0 else None
    while deadline is None or time.time() < deadline:
        if os.path.exists(done_path):
            print(f"[pipeline] external vLLM reload acknowledged for {stage}", flush=True)
            return server_urls or None
        if os.path.exists(fail_path):
            try:
                with open(fail_path) as f:
                    detail = f.read()[-2000:]
            except Exception:
                detail = ""
            if required:
                raise RuntimeError(f"external vLLM reload failed for {stage}: {detail}")
            print(f"[pipeline] WARNING: external vLLM reload failed for {stage}: {detail}", flush=True)
            return None
        time.sleep(poll_s)
    if required:
        raise TimeoutError(f"external vLLM reload timed out for {stage}: {request_path}")
    return None


def _maybe_reload_actual_vllm(gcs_model_path: str) -> None:
    reload_url = (os.environ.get("EXPDIS_VLLM_RELOAD_WORKER_URL") or "").strip()
    reload_tpu_name = (os.environ.get("EXPDIS_VLLM_RELOAD_TPU_NAME") or "").strip()
    reload_zone = (os.environ.get("EXPDIS_VLLM_RELOAD_ZONE") or "").strip()
    if not (reload_url and reload_tpu_name and reload_zone):
        return

    reload_worker = (os.environ.get("EXPDIS_VLLM_RELOAD_WORKER") or "0").strip() or "0"
    served_model_name = (os.environ.get("EXPDIS_ACTUAL_VLLM_MODEL_NAME") or "actual_model").strip() or "actual_model"

    env = os.environ.copy()
    env.update(
        {
            "GCS_MODEL_PATH": gcs_model_path,
            "TPU_NAME": reload_tpu_name,
            "ZONE": reload_zone,
            "WORKER": reload_worker,
            "SERVED_MODEL_NAME": served_model_name,
            "LOCAL_MODEL_DIR_BASENAME": "vllm_actual_model",
        }
    )
    try:
        ret = subprocess.run(
            ["bash", os.path.join(_REPO_ROOT, "scripts", "reload_vllm_with_model.sh")],
            capture_output=True,
            text=True,
            timeout=1800,
            env=env,
        )
        if ret.returncode == 0:
            os.environ["EXPDIS_ACTUAL_VLLM_SERVER_URL"] = reload_url
            print(f"[pipeline] reloaded Actual vLLM worker -> {reload_url}", flush=True)
        else:
            print(
                f"[pipeline] WARNING: Actual vLLM reload failed: {(ret.stderr or ret.stdout)[:300]}",
                flush=True,
            )
    except Exception as e:
        print(f"[pipeline] WARNING: Actual vLLM reload error: {e}", flush=True)


def _early_init_jax_distributed_from_env() -> None:
    """Initialize JAX distributed before W&B or train imports can touch TPU.

    This intentionally imports JAX inside the function. On TPU, importing or
    querying JAX too late can leave every host with a local 4-chip backend even
    when the distributed runtime is subsequently initialized successfully.
    """
    coordinator_address = os.environ.get("JAX_COORDINATOR_ADDRESS", "").strip()
    num_processes_env = os.environ.get("JAX_NUM_PROCESSES", "").strip()
    if not coordinator_address or not num_processes_env:
        return

    import jax

    expected_processes = int(num_processes_env)
    expected_process_id = int(os.environ.get("JAX_PROCESS_ID", "0"))

    def _check_world(label: str) -> None:
        actual_processes = int(jax.process_count())
        actual_process_id = int(jax.process_index())
        print(
            f"[pipeline-dist] {label}: rank={actual_process_id} "
            f"num_hosts={actual_processes} expected_rank={expected_process_id} "
            f"expected_num_hosts={expected_processes}",
            flush=True,
        )
        # PJRT/libtpu can remap the TPU VM worker number to a topology rank.
        # The safety condition we need is the full world size; the dense rank
        # returned by JAX does not have to equal the gcloud worker index.
        if expected_processes > 1 and actual_processes != expected_processes:
            raise RuntimeError(
                "JAX distributed world mismatch after early init: "
                f"rank={actual_process_id}/{actual_processes}, "
                f"expected_num_hosts={expected_processes}. "
                "Refusing to continue with duplicated single-host training."
            )

    if jax.distributed.is_initialized():
        _check_world("already initialized")
        return

    local_device_ids = None
    local_device_ids_env = os.environ.get("JAX_LOCAL_DEVICE_IDS", "").strip()
    if local_device_ids_env:
        local_device_ids = [int(x) for x in local_device_ids_env.split(",") if x.strip()]
    elif os.environ.get("TPU_VISIBLE_CHIPS", "").strip():
        local_device_ids = [int(x) for x in os.environ["TPU_VISIBLE_CHIPS"].split(",") if x.strip()]

    init_kwargs = {
        "coordinator_address": coordinator_address,
        "num_processes": expected_processes,
        "process_id": expected_process_id,
        "initialization_timeout": int(os.environ.get("JAX_INITIALIZATION_TIMEOUT", "600")),
        "heartbeat_timeout_seconds": int(os.environ.get("JAX_HEARTBEAT_TIMEOUT_SECONDS", "600")),
    }
    if local_device_ids is not None:
        init_kwargs["local_device_ids"] = local_device_ids
    coordinator_bind_address = os.environ.get("JAX_COORDINATOR_BIND_ADDRESS", "").strip()
    if coordinator_bind_address:
        init_kwargs["coordinator_bind_address"] = coordinator_bind_address
    cluster_detection_method = os.environ.get("EXPDIS_JAX_CLUSTER_DETECTION_METHOD", "").strip()
    if cluster_detection_method:
        init_kwargs["cluster_detection_method"] = cluster_detection_method

    print(
        f"[pipeline-dist] initializing early rank={expected_process_id} "
        f"num_hosts={expected_processes} coordinator={coordinator_address}",
        flush=True,
    )
    jax.distributed.initialize(**init_kwargs)
    _check_world("initialized")


def main(cfg: TrainConfig | None = None):
    """Run a single Explorer→SFT→Actual pipeline pass.

    For multi-round campaigns (Explorer←Actual handoff across rounds), call
    ``multi_round_main()`` instead — it loops over this for cfg.num_rounds.
    The single-pass logic below is unchanged from before; only the cfg parsing
    is now optional so the multi-round dispatcher can override per-round.
    """
    if cfg is None:
        cfg = parse_args()
    if not cfg.pipeline_root_dir:
        cfg = replace(cfg, pipeline_root_dir=os.path.abspath(cfg.output_dir))
    _validate_experimental_mrme_gate(cfg)
    validate_contract(cfg, require_eval_contract=True)
    _early_init_jax_distributed_from_env()
    try:
        import jax

        _validate_multihost_stage_boundary(int(jax.process_count()), cfg)
    except ImportError:
        _validate_multihost_stage_boundary(1)
    root_output_dir = os.path.abspath(cfg.output_dir)
    pipeline_stage = (os.environ.get("EXPDIS_PIPELINE_STAGE") or "explorer_novelty").strip()
    allowed_stages = {"explorer_correctness", "explorer_novelty", "actual_sft", "actual_grpo", "eval"}
    if pipeline_stage not in allowed_stages:
        raise ValueError(f"unsupported EXPDIS_PIPELINE_STAGE={pipeline_stage!r}; expected one of {sorted(allowed_stages)}")
    if pipeline_stage == "actual_sft":
        os.environ.setdefault("EXPDIS_PIPELINE_STOP_AFTER_SFT", "1")
    if pipeline_stage in {"actual_sft", "actual_grpo"}:
        if not _env_flag("EXPDIS_PIPELINE_SKIP_EXPLORER", "0"):
            raise RuntimeError(
                f"{pipeline_stage} requires EXPDIS_PIPELINE_SKIP_EXPLORER=1 so it cannot accidentally rerun Explorer"
            )
        if not (os.environ.get("EXPDIS_EXISTING_TRAJECTORIES") or "").strip():
            raise RuntimeError(
                f"{pipeline_stage} requires EXPDIS_EXISTING_TRAJECTORIES pointing at HF/local Explorer trajectories"
            )
    if (os.environ.get("EXPDIS_WANDB_SINGLE_RUN", "1") or "").strip().lower() not in ("0", "false", "no", "off"):
        e2e_run_id = _pipeline_wandb_run_id(cfg)
        e2e_run_name = _pipeline_wandb_run_name(cfg)
        os.environ.setdefault("EXPDIS_WANDB_RUN_ID", e2e_run_id)
        os.environ.setdefault("EXPDIS_WANDB_EXPLORER_RUN_ID", e2e_run_id)
        os.environ.setdefault("EXPDIS_WANDB_DISTILL_RUN_ID", e2e_run_id)
        os.environ.setdefault("EXPDIS_WANDB_ACTUAL_RUN_ID", e2e_run_id)
        os.environ.setdefault("EXPDIS_WANDB_ACTUAL_GRPO_RUN_ID", e2e_run_id)
        os.environ.setdefault("EXPDIS_WANDB_GROUP", e2e_run_name)
    _wandb_log_pipeline(cfg, {
        "pipeline/start": 1.0,
        "pipeline/lambda_novelty": float(cfg.lambda_novelty),
        "pipeline/max_prompt_len": float(cfg.max_prompt_len),
        "pipeline/max_completion_len": float(cfg.max_completion_len),
        "pipeline/max_total_len": float(cfg.max_total_len),
        "pipeline/grpo_batch_size": float(cfg.grpo_batch_size),
        "pipeline/grpo_grad_accum": float(cfg.grpo_grad_accum),
        "pipeline/grpo_num_generations": float(cfg.grpo_num_generations),
        "pipeline/grpo_updates_per_rollout": float(cfg.grpo_updates_per_rollout),
        "pipeline/dr_grpo_loss_normalizer": float(cfg.dr_grpo_loss_normalizer),
        "pipeline/grpo_clip_epsilon": float(cfg.grpo_clip_epsilon),
        "pipeline/grpo_clip_epsilon_high": float(cfg.grpo_clip_epsilon_high),
        "pipeline/dynamic_sampling": float(bool(cfg.dynamic_sampling)),
        "pipeline/dynamic_sampling_max_attempts": float(cfg.dynamic_sampling_max_attempts),
        "pipeline/mask_truncated_completions": float(bool(cfg.mask_truncated_completions)),
        "pipeline/soft_overlong_expected_len": float(cfg.soft_overlong_expected_len),
        "pipeline/soft_overlong_cache_len": float(cfg.soft_overlong_cache_len),
        "pipeline/grpo_max_steps": float(cfg.grpo_max_steps),
        "pipeline/actual_online_grpo_max_steps": float(cfg.actual_online_grpo_max_steps),
        "pipeline/benchmark_num_rollouts": float(cfg.benchmark_num_rollouts),
        "pipeline/min_accepted_trajectories": float(cfg.min_accepted_trajectories),
        "pipeline/max_accepted_trajectories": float(cfg.max_accepted_trajectories),
    })
    _write_run_summary(root_output_dir, cfg, "pipeline_start", {
        "status": "started",
        "stage": pipeline_stage,
        "lambda_novelty": float(cfg.lambda_novelty),
        "grpo_max_steps": int(cfg.grpo_max_steps),
        "grpo_updates_per_rollout": int(cfg.grpo_updates_per_rollout),
        "dr_grpo_loss_normalizer": int(cfg.dr_grpo_loss_normalizer),
        "dynamic_sampling": bool(cfg.dynamic_sampling),
        "dynamic_sampling_max_attempts": int(cfg.dynamic_sampling_max_attempts),
        "mask_truncated_completions": bool(cfg.mask_truncated_completions),
        "actual_online_grpo_max_steps": int(cfg.actual_online_grpo_max_steps),
        "min_accepted_trajectories": int(cfg.min_accepted_trajectories),
        "max_accepted_trajectories": int(cfg.max_accepted_trajectories),
        "distill_epochs": int(cfg.distill_epochs),
        "max_distill_examples": int(cfg.max_distill_examples),
        # LR-schedule identity for every optimizer stage (constant = frozen contract).
        "grpo_lr_schedule": str(cfg.grpo_lr_schedule),
        "grpo_lr_warmup_steps": int(cfg.grpo_lr_warmup_steps),
        "grpo_lr_decay_steps": int(cfg.grpo_lr_decay_steps),
        "grpo_lr_min_ratio": float(cfg.grpo_lr_min_ratio),
        "actual_grpo_lr_schedule": str(cfg.actual_grpo_lr_schedule),
        "actual_grpo_lr_warmup_steps": int(cfg.actual_grpo_lr_warmup_steps),
        "actual_grpo_lr_decay_steps": int(cfg.actual_grpo_lr_decay_steps),
        "actual_grpo_lr_min_ratio": float(cfg.actual_grpo_lr_min_ratio),
        "distill_lr_schedule": str(cfg.distill_lr_schedule),
        "distill_lr_warmup_steps": int(cfg.distill_lr_warmup_steps),
        "distill_lr_decay_steps": int(cfg.distill_lr_decay_steps),
        "distill_lr_min_ratio": float(cfg.distill_lr_min_ratio),
    })
    scout_count = int(cfg.scouts_per_round)
    scout_step_budgets = split_budget(int(cfg.grpo_max_steps), scout_count)
    if any(steps < 1 for steps in scout_step_budgets):
        raise ValueError(
            f"round Scout budget {cfg.grpo_max_steps} cannot support "
            f"scouts_per_round={scout_count} with at least one update each"
        )
    explorer_cfgs = []
    for scout_index, scout_steps in enumerate(scout_step_budgets, start=1):
        scout_dir = (
            os.path.join(root_output_dir, "explorer")
            if scout_count == 1
            else os.path.join(root_output_dir, "explorers", f"scout_{scout_index:02d}")
        )
        explorer_cfgs.append(
            replace(
                cfg,
                output_dir=scout_dir,
                wandb_run_name=(
                    _pipeline_wandb_run_name(cfg)
                    if scout_count == 1
                    else f"{_pipeline_wandb_run_name(cfg)}_scout_{scout_index:02d}"
                ),
                seed=int(cfg.seed) + (scout_index - 1) * 10_003,
                scout_index=scout_index,
                grpo_max_steps=int(scout_steps),
                # Each Scout is a fresh stage from the same immutable parent.
                resume_checkpoint="",
                init_checkpoint="",
            )
        )
    trajectory_paths = [
        os.path.join(item.output_dir, "explorer_trajectories.jsonl")
        for item in explorer_cfgs
    ]

    from . import train as _train_mod

    skip_explorer = _env_flag("EXPDIS_PIPELINE_SKIP_EXPLORER", "0")
    existing_sources = [
        item.strip()
        for item in (os.environ.get("EXPDIS_EXISTING_TRAJECTORIES") or "").split(",")
        if item.strip()
    ]
    if skip_explorer:
        if existing_sources and len(existing_sources) != scout_count:
            raise ValueError(
                f"EXPDIS_EXISTING_TRAJECTORIES supplied {len(existing_sources)} pools, "
                f"but scouts_per_round={scout_count}"
            )
        if _is_source_process():
            for index, destination in enumerate(trajectory_paths):
                source = existing_sources[index] if existing_sources else ""
                _materialize_existing_trajectory(source, destination)
        print(
            f"[pipeline] skipping {scout_count} Scout stage(s); using explicit trajectory pools",
            flush=True,
        )
        _wandb_log_pipeline(cfg, {
            "pipeline/explorer_skipped": 1.0,
            "pipeline/resumed_from_existing_trajectories": 1.0,
        })
    else:
        # Stage 1: K independent Scouts.  They share only the parent weights;
        # optimizer/RND/counters and artifact namespaces are fresh per Scout.
        for explorer_cfg in explorer_cfgs:
            scout_index = int(explorer_cfg.scout_index)
            scout_env = {
                "EXPDIS_WANDB_EXPLORER_RUN_ID": _scout_wandb_run_id(cfg, scout_index),
                "EXPDIS_HF_ARTIFACT_PREFIX": _scout_artifact_prefix(cfg, scout_index),
                "EXPDIS_MULTIROUND_RESET_RND": "1",
            }
            with _temporary_env(scout_env):
                explorer_state = _train_mod.run_training(explorer_cfg)
                explorer_hf_dir = _maybe_export_state_to_hf_dir(
                    explorer_state,
                    explorer_cfg,
                    os.path.join(explorer_cfg.output_dir, "explorer_hf_final"),
                )
                if explorer_hf_dir is not None:
                    gcs_base = (os.environ.get("EXPDIS_GCS_CHECKPOINT_BASE") or "").strip()
                    explorer_hf_gcs = None
                    if gcs_base and _gcs_artifacts_enabled():
                        explorer_hf_gcs = _maybe_upload_dir_to_gcs(
                            explorer_hf_dir,
                            f"{gcs_base.rstrip('/')}/scout_{scout_index:02d}/explorer_hf_final",
                        )
                        if explorer_hf_gcs:
                            _wandb_log_pipeline(cfg, {"pipeline/explorer_hf_uploaded": 1.0})
                    if _env_flag("EXPDIS_HF_MIRROR_FINAL_EXPORTS", "1"):
                        hf_path = f"{_scout_artifact_prefix(cfg, scout_index)}/explorer_hf_final"
                        if explorer_hf_gcs:
                            _maybe_upload_gcs_uri_to_hf(
                                explorer_hf_gcs, explorer_cfg, hf_path, is_dir=True
                            )
                        else:
                            _maybe_upload_path_to_hf(
                                explorer_hf_dir, explorer_cfg, hf_path, is_dir=True
                            )
                del explorer_state
            gc.collect()
            try:
                import jax

                jax.clear_caches()
            except Exception:
                pass
    if _env_flag("EXPDIS_PIPELINE_STOP_AFTER_EXPLORER", "0"):
        print("[pipeline] EXPDIS_PIPELINE_STOP_AFTER_EXPLORER=1; stopping after Explorer stage", flush=True)
        _wandb_log_pipeline(cfg, {
            "pipeline/stopped_after_explorer": 1.0,
            "pipeline/sft_skipped_by_stage": 1.0,
            "pipeline/actual_grpo_skipped_by_stage": 1.0,
        })
        _write_run_summary(root_output_dir, cfg, "stopped_after_explorer", {
            "status": "stopped_after_explorer",
            "stage": os.environ.get("EXPDIS_PIPELINE_STAGE", "explorer"),
            "explorer_output_dirs": [item.output_dir for item in explorer_cfgs],
            "reason": "EXPDIS_PIPELINE_STOP_AFTER_EXPLORER",
        })
        return

    # Stage 2: merge all K raw pools once, then apply one filter.  Only rank 0
    # needs the full raw text; the compact accepted library is byte-replicated
    # to every host for distributed SFT.
    merged_path = os.path.join(root_output_dir, "trajectory_library.raw.jsonl")
    accepted = os.path.join(root_output_dir, "trajectory_library.accepted.jsonl")
    traj_payload_source = None
    if _is_source_process():
        _merge_scout_trajectory_files(trajectory_paths, merged_path)
        collect_accepted(cfg, trajectory_paths, output_path=accepted)
        traj_payload_source = _summarize_trajectory_jsonl(
            merged_path,
            num_generations=cfg.grpo_num_generations,
        )
    accepted = _replicate_source_file(accepted)
    traj_payload = _broadcast_source_json(traj_payload_source)
    accepted_payload = _summarize_trajectory_jsonl(accepted, num_generations=cfg.grpo_num_generations)
    accepted_count = float(accepted_payload.get("traj/total_count", 0.0) or 0.0)
    total_count = float(traj_payload.get("traj/total_count", 0.0) or 0.0)
    _wandb_log_pipeline(
        cfg,
        traj_payload | {
            "traj/accepted_count": accepted_count,
            "traj/accepted_fraction": accepted_count / max(1.0, total_count),
            "traj/accepted_correct_count": accepted_count,
            "actual/bootstrap_accepted_count": accepted_count,
            "actual/bootstrap_selected_count": accepted_count,
            "actual/bootstrap_min_required": float(cfg.min_accepted_trajectories),
            "actual/bootstrap_used_fallback": 0.0,
            "actual/bootstrap_oversampled_count": 0.0,
            "audit/traj_clipped_fraction": float(traj_payload.get("traj/clipped_fraction", 0.0) or 0.0),
            "audit/traj_prompt_truncated_fraction": float(traj_payload.get("traj/prompt_truncated_fraction", 0.0) or 0.0),
            "audit/traj_normalization_changed_fraction": float(traj_payload.get("traj/normalization_changed_fraction", 0.0) or 0.0),
            "audit/traj_valid_answer_fraction": float(traj_payload.get("traj/valid_answer_fraction", 0.0) or 0.0),
        },
    )
    _write_run_summary(root_output_dir, cfg, "explorer_trajectory_summary", {
        "trajectory_jsonls": trajectory_paths,
        "merged_trajectory_jsonl": merged_path,
        "accepted_jsonl": accepted,
        "trajectory_metrics": traj_payload,
        "accepted_metrics": accepted_payload,
        "accepted_count": accepted_count,
        "accepted_fraction": accepted_count / max(1.0, total_count),
        "selection_policy": QUALITY_POOL_POLICY,  # the only implemented policy; aliases map to it
        "selection_policy_alias": cfg.accepted_selection_policy,
    })

    # Stage 3: SFT distillation
    from . import distill as _distill_mod
    actual_cfg = replace(
        cfg,
        # Novelty is a Scout-only exploration bonus.  Central receives
        # correctness plus DAPO soft-overlong length control, never novelty.
        lambda_novelty=0.0,
        incorrect_novelty_scale=0.0,
        gate_novelty_by_quality=False,
        novelty_zscore="off",
        output_dir=os.path.join(root_output_dir, "actual"),
        wandb_run_name=_pipeline_wandb_run_name(cfg),
        init_weights_checkpoint="",
        resume_checkpoint="",
        init_checkpoint="",
    )
    actual_sft_state = _distill_mod.run_sft(
        actual_cfg,
        accepted,
        init_weights_checkpoint=cfg.central_parent_checkpoint,
    )
    _wandb_log_pipeline(cfg, {"pipeline/sft_complete": 1.0})
    _write_run_summary(root_output_dir, cfg, "sft", {
        "status": "complete",
        "accepted_jsonl": accepted,
        "actual_sft_checkpoint": checkpoint_path(actual_cfg, "actual_sft_final"),
        "distill_epochs": int(cfg.distill_epochs),
        "max_distill_examples": int(cfg.max_distill_examples),
        "distill_lr_schedule": str(cfg.distill_lr_schedule),
        "distill_lr_warmup_steps": int(cfg.distill_lr_warmup_steps),
    })

    actual_grpo_vllm_urls = cfg.vllm_server_urls
    sft_hf_dir = _maybe_export_state_to_hf_dir(
        actual_sft_state,
        actual_cfg,
        os.path.join(actual_cfg.output_dir, "actual_sft_hf"),
    )
    if sft_hf_dir is not None:
        gcs_base = (os.environ.get("EXPDIS_GCS_CHECKPOINT_BASE") or "").strip()
        sft_hf_gcs = None
        sft_hf_repo_id = _hf_artifact_repo()
        sft_hf_path = f"{_hf_run_prefix(cfg)}/actual_sft_hf"
        sft_hf_uploaded = False
        if gcs_base and _gcs_artifacts_enabled():
            sft_hf_gcs = _maybe_upload_dir_to_gcs(
                sft_hf_dir,
                f"{gcs_base.rstrip('/')}/actual_sft_hf",
            )
        if _env_flag("EXPDIS_HF_MIRROR_FINAL_EXPORTS", "1"):
            if sft_hf_gcs:
                sft_hf_uploaded = _maybe_upload_gcs_uri_to_hf(sft_hf_gcs, cfg, sft_hf_path, is_dir=True)
            else:
                sft_hf_uploaded = _maybe_upload_path_to_hf(
                    sft_hf_dir,
                    cfg,
                    sft_hf_path,
                    is_dir=True,
                )
        if sft_hf_gcs is not None or sft_hf_uploaded:
            reloaded_urls = _maybe_reload_vllm_slice(
                sft_hf_gcs,
                cfg,
                max_model_len=cfg.max_total_len,
                max_num_seqs=int(os.environ.get("EXPDIS_ACTUAL_GRPO_VLLM_MAX_NUM_SEQS", "32")),
                stage="actual_sft",
                hf_repo_id=sft_hf_repo_id if sft_hf_uploaded else "",
                hf_path_in_repo=sft_hf_path if sft_hf_uploaded else "",
                hf_repo_type="dataset",
            )
            if reloaded_urls:
                actual_grpo_vllm_urls = reloaded_urls

    if _env_flag("EXPDIS_PIPELINE_STOP_AFTER_SFT", "0"):
        print("[pipeline] EXPDIS_PIPELINE_STOP_AFTER_SFT=1; stopping after SFT stage", flush=True)
        _wandb_log_pipeline(cfg, {
            "pipeline/stopped_after_sft": 1.0,
            "pipeline/actual_grpo_skipped_by_stage": 1.0,
        })
        _write_run_summary(root_output_dir, cfg, "stopped_after_sft", {
            "status": "stopped_after_sft",
            "stage": pipeline_stage,
            "actual_output_dir": actual_cfg.output_dir,
            "reason": "EXPDIS_PIPELINE_STOP_AFTER_SFT",
        })
        return

    if _env_flag("EXPDIS_PIPELINE_CLEAR_MEMORY_BEFORE_ACTUAL", "1"):
        # Actual GRPO loads the SFT checkpoint and keeps a reference-policy
        # copy for KL. Drop Explorer/SFT train states before initializing it so
        # single-host v5/v6e runs do not carry stale TPU buffers into Actual.
        try:
            del explorer_state
        except UnboundLocalError:
            pass
        try:
            del actual_sft_state
        except UnboundLocalError:
            pass
        gc.collect()
        try:
            import jax
            jax.clear_caches()
            print("[pipeline] cleared JAX caches before Actual GRPO", flush=True)
        except Exception as e:
            print(f"[pipeline] WARNING: could not clear JAX caches before Actual GRPO: {e}", flush=True)

    # Stage 4: Actual online GRPO from the SFT checkpoint.
    actual_grpo_cfg = replace(
        actual_cfg,
        grpo_max_steps=cfg.actual_online_grpo_max_steps,
        grpo_lr=cfg.actual_grpo_lr,
        grpo_lr_schedule=cfg.actual_grpo_lr_schedule,
        grpo_lr_warmup_steps=cfg.actual_grpo_lr_warmup_steps,
        grpo_lr_decay_steps=cfg.actual_grpo_lr_decay_steps,
        grpo_lr_min_ratio=cfg.actual_grpo_lr_min_ratio,
        grpo_kl_beta=cfg.actual_grpo_kl_coef,
        # Central RL is a fresh stage initialized from SFT parameters.  It must
        # not inherit the SFT optimizer/counter state or use the ambiguous
        # legacy checkpoint path.
        init_weights_checkpoint=checkpoint_path(actual_cfg, "actual_sft_final"),
        resume_checkpoint="",
        init_checkpoint="",
        output_dir=os.path.join(actual_cfg.output_dir, "grpo"),
        # SFT and Actual GRPO intentionally share one W&B run. Namespaces stay
        # distinct (`distill/*`, `actual_grpo/*`) while the dashboard remains
        # readable as a single Actual-model timeline.
        wandb_run_name=actual_cfg.wandb_run_name,
        vllm_server_urls=actual_grpo_vllm_urls,
    )
    actual_state = _train_mod.run_training(actual_grpo_cfg)
    _wandb_log_pipeline(cfg, {"pipeline/actual_grpo_complete": 1.0})
    _write_run_summary(root_output_dir, cfg, "actual_grpo", {
        "status": "complete",
        "actual_grpo_dir": actual_grpo_cfg.output_dir,
        "actual_grpo_max_steps": int(cfg.actual_online_grpo_max_steps),
        "actual_grpo_lr": float(cfg.actual_grpo_lr),
        "actual_grpo_lr_schedule": str(cfg.actual_grpo_lr_schedule),
        "actual_grpo_lr_warmup_steps": int(cfg.actual_grpo_lr_warmup_steps),
        "actual_grpo_kl_coef": float(cfg.actual_grpo_kl_coef),
    })

    actual_hf_gcs = None
    actual_hf_uploaded = False
    actual_hf_repo_id = ""
    actual_hf_path = ""
    actual_source_checkpoint = None
    actual_hf_dir = _maybe_export_state_to_hf_dir(
        actual_state,
        actual_grpo_cfg,
        os.path.join(actual_cfg.output_dir, "actual_hf_final"),
    )
    if actual_hf_dir is not None:
        gcs_base = (os.environ.get("EXPDIS_GCS_CHECKPOINT_BASE") or "").strip()
        actual_hf_repo_id = _hf_artifact_repo()
        actual_hf_path = f"{_hf_run_prefix(cfg)}/actual_hf_final"
        if gcs_base and _gcs_artifacts_enabled():
            actual_hf_gcs = _maybe_upload_dir_to_gcs(
                actual_hf_dir,
                f"{gcs_base.rstrip('/')}/actual_hf_final",
            )
        if _env_flag("EXPDIS_HF_MIRROR_FINAL_EXPORTS", "1"):
            if actual_hf_gcs:
                actual_hf_uploaded = _maybe_upload_gcs_uri_to_hf(actual_hf_gcs, cfg, actual_hf_path, is_dir=True)
            else:
                actual_hf_uploaded = _maybe_upload_path_to_hf(
                    actual_hf_dir,
                    cfg,
                    actual_hf_path,
                    is_dir=True,
                )
        actual_source_checkpoint = actual_hf_gcs or (
            f"hf://{actual_hf_repo_id}/{actual_hf_path}" if actual_hf_uploaded else actual_hf_dir
        )
        if actual_hf_gcs is not None or actual_hf_uploaded:
            if _env_flag("EXPDIS_SKIP_FINAL_EVAL", "0"):
                print(
                    f"[pipeline] EXPDIS_SKIP_FINAL_EVAL=1; saved Actual HF bundle at {actual_source_checkpoint} and skipping AIME dispatch.",
                    flush=True,
                )
                _wandb_log_pipeline(
                    cfg,
                    {
                        "pipeline/actual_hf_final_saved": 1.0,
                        "pipeline/final_eval_skipped": 1.0,
                    },
                )
                _write_run_summary(root_output_dir, cfg, "actual_hf_final", {
                    "status": "saved",
                    "gcs_path": actual_hf_gcs,
                    "hf_repo_id": actual_hf_repo_id if actual_hf_uploaded else "",
                    "hf_path_in_repo": actual_hf_path if actual_hf_uploaded else "",
                    "source_checkpoint": actual_source_checkpoint,
                    "final_eval_skipped": True,
                })
                return
            final_urls = _maybe_reload_vllm_slice(
                actual_hf_gcs,
                cfg,
                max_model_len=int(os.environ.get("EXPDIS_FINAL_EVAL_MAX_MODEL_LEN", "40960")),
                max_num_seqs=int(os.environ.get("EXPDIS_FINAL_EVAL_VLLM_MAX_NUM_SEQS", "4")),
                stage="actual_final",
                hf_repo_id=actual_hf_repo_id if actual_hf_uploaded else "",
                hf_path_in_repo=actual_hf_path if actual_hf_uploaded else "",
                hf_repo_type="dataset",
            )
            if final_urls:
                os.environ["EXPDIS_ACTUAL_VLLM_SERVER_URLS"] = final_urls
            elif actual_hf_gcs:
                _maybe_reload_actual_vllm(actual_hf_gcs)

    if _env_flag("EXPDIS_SKIP_FINAL_EVAL", "0"):
        source = actual_source_checkpoint or actual_hf_dir or actual_grpo_cfg.output_dir
        print(
            f"[pipeline] EXPDIS_SKIP_FINAL_EVAL=1; Actual stage complete at {source} and skipping AIME dispatch.",
            flush=True,
        )
        _wandb_log_pipeline(
            cfg,
            {
                "pipeline/actual_hf_final_saved": 1.0 if actual_hf_dir else 0.0,
                "pipeline/final_eval_skipped": 1.0,
            },
        )
        _write_run_summary(root_output_dir, cfg, "actual_final_no_eval", {
            "status": "saved",
            "source_checkpoint": source,
            "final_eval_skipped": True,
        })
        return

    if not _is_source_process():
        return

    # Stage 5: AIME24 benchmark eval
    from .generate import parse_server_urls
    from .eval import aime24_wandb_metrics, run_eval
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True, fix_mistral_regex=True)
    actual_eval_server_urls = (
        os.environ.get("EXPDIS_ACTUAL_VLLM_SERVER_URL") or
        os.environ.get("EXPDIS_ACTUAL_VLLM_SERVER_URLS") or
        actual_grpo_cfg.vllm_server_urls
    )
    server_urls = parse_server_urls(actual_eval_server_urls)
    if actual_eval_server_urls == actual_grpo_cfg.vllm_server_urls:
        print(
            "[pipeline] WARNING: final AIME eval is using the original vLLM rollout URLs. "
            "If the served model was not reloaded to Actual weights, this benchmark will not reflect the trained Actual model.",
            flush=True,
        )
    else:
        print(
            f"[pipeline] final AIME eval using Actual-specific server URLs: {actual_eval_server_urls}",
            flush=True,
        )
    def _eval_progress(done: int, total: int, elapsed: float) -> None:
        eta = (elapsed / done) * (total - done) if done else 0.0
        _wandb_log_pipeline(cfg, {
            "eval/aime24_32k/subrequests_done": float(done),
            "eval/aime24_32k/subrequests_total": float(total),
            "eval/aime24_32k/progress_fraction": float(done / total if total else 0.0),
            "eval/aime24_32k/elapsed_seconds": float(elapsed),
            "eval/aime24_32k/eta_seconds": float(eta),
        })

    eval_temperature = float(os.environ.get("EXPDIS_FINAL_EVAL_TEMPERATURE", "0.6") or "0.6")
    eval_top_p = float(os.environ.get("EXPDIS_FINAL_EVAL_TOP_P", str(cfg.top_p)) or str(cfg.top_p))
    eval_top_k = int(os.environ.get("EXPDIS_FINAL_EVAL_TOP_K", str(cfg.top_k)) or str(cfg.top_k))
    res = run_eval(
        tok, server_urls,
        num_rollouts=cfg.benchmark_num_rollouts,
        max_tokens=cfg.benchmark_rollout_max_completion_len,
        model=cfg.model_name,
        temperature=eval_temperature,
        top_p=eval_top_p,
        top_k=eval_top_k,
        which="AIME_2024",
        enable_thinking=cfg.enable_thinking,
        progress_callback=_eval_progress,
    )
    res["protocol"] = {
        "model_name": cfg.model_name,
        "served_model_urls": server_urls,
        "dataset": "AIME_2024",
        "num_rollouts": int(cfg.benchmark_num_rollouts),
        "max_completion_len": int(cfg.benchmark_rollout_max_completion_len),
        "temperature": float(eval_temperature),
        "top_p": float(eval_top_p),
        "top_k": int(eval_top_k),
        "enable_thinking": cfg.enable_thinking,
        "source_checkpoint": actual_source_checkpoint or actual_hf_gcs or actual_hf_dir,
    }
    out = os.path.join(root_output_dir, "final_eval_aime24.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[pipeline] AIME24 avg@{cfg.benchmark_num_rollouts} = {res['avg_at_n']*100:.2f}%", flush=True)
    gcs_base = (os.environ.get("EXPDIS_GCS_CHECKPOINT_BASE") or "").strip().rstrip("/")
    gcs_target = ""
    uploaded_to_gcs = False
    if gcs_base and _gcs_artifacts_enabled():
        gcs_target = f"{gcs_base}/final_eval_aime24.json"
        uploaded_to_gcs = _maybe_upload_file_to_gcs(out, gcs_target)
    if _env_flag("EXPDIS_HF_MIRROR_FINAL_EVAL", "1"):
        hf_path = f"{_hf_run_prefix(cfg)}/final_eval_aime24.json"
        if uploaded_to_gcs and gcs_target:
            _maybe_upload_gcs_uri_to_hf(gcs_target, cfg, hf_path, is_dir=False)
        else:
            _maybe_upload_path_to_hf(out, cfg, hf_path, is_dir=False)
    avg_key = f"benchmark/aime24/avg_at_{cfg.benchmark_num_rollouts}"
    eval_payload = {
        "pipeline/final_eval_complete": 1.0,
        avg_key: float(res["avg_at_n"]),
        "benchmark/aime24/avg_at_n": float(res["avg_at_n"]),
        "benchmark/aime24/num_problems": float(res["num_problems"]),
        "benchmark/aime24/num_rollouts": float(res["num_rollouts"]),

    }
    eval_payload.update(aime24_wandb_metrics(res))
    for k in (1, 2, 4, 8, 16, 32, 64):
        key = f"pass_at_{k}"
        if key in res:
            eval_payload[f"benchmark/aime24/{key}"] = float(res[key])
    _wandb_log_pipeline(
        cfg,
        eval_payload,
    )
    _write_run_summary(root_output_dir, cfg, "final_eval_aime24", {
        "status": "complete",
        "local_path": out,
        "metrics": res,
    })


# ============================================================================
# Multi-round pipeline (Explorer ← Actual handoff across rounds)
# ============================================================================
#
# Each round runs the same Explorer→Filter→SFT→Actual sequence, but:
#   * Round 1 starts Explorer from Base (cfg.model_name).
#   * Round R+1 starts Explorer from round R's final Actual checkpoint.
#   * Each round trains on a different shuffle of the dataset (seed offset)
#     and a sub-budget of grpo_max_steps / actual_online_grpo_max_steps.
#   * RND novelty state is reset each round (multi_round_reset_rnd=True) so
#     novelty signals fresh exploration territory rather than collapsing to
#     "I've seen all training prompts before."
#   * Round outputs land at {output_dir}/round_{R}/...
#
# When pipeline_mode == "two_model" (default) this is a no-op: main() is
# called once with the user's cfg unchanged.


def _split_total_budget(total: int | None, num_rounds: int) -> List[int]:
    """Divide a total step/example budget evenly across num_rounds (excess to early rounds)."""
    if total is None:
        return []
    total_int = max(0, int(total))
    rounds_int = max(1, int(num_rounds))
    if total_int < rounds_int:
        raise ValueError(
            f"cannot split budget={total_int} across num_rounds={rounds_int}; "
            "each round needs at least one unit"
        )
    base = total_int // rounds_int
    remainder = total_int % rounds_int
    return [base + (1 if idx < remainder else 0) for idx in range(rounds_int)]


def _round_budget(total: int | None, round_idx: int, num_rounds: int, mode: str) -> int | None:
    """Return per-round budget. mode='split' divides total across rounds; 'constant' keeps full budget per round."""
    if total is None:
        return None
    mode_norm = (mode or "split").strip().lower()
    if mode_norm in ("constant", "per_round", "full_per_round"):
        return int(total)
    if mode_norm not in ("split", "constant_total", "total"):
        raise ValueError(
            f"unsupported round_budget_mode={mode!r}; expected split/constant_total or constant/full_per_round"
        )
    budgets = _split_total_budget(total, num_rounds)
    if round_idx < 1 or round_idx > len(budgets):
        raise IndexError(f"round_idx {round_idx} outside 1..{len(budgets)}")
    return budgets[round_idx - 1]


def _find_latest_actual_checkpoint(round_output_dir: str, cfg: TrainConfig | None = None) -> str | None:
    """Find the highest-numbered Actual GRPO Orbax checkpoint for handoff.

    In HF-only multihost runs, prefer the mirrored HF URI so every worker can
    restore the same checkpoint. Local paths are fine for single-host/local
    smoke tests.
    """
    actual_grpo_dir = os.path.join(round_output_dir, "actual", "grpo")
    if cfg is not None and cfg.checkpoint_root:
        from etils import epath
        stage_cfg = replace(cfg, output_dir=actual_grpo_dir)
        path = checkpoint_path(stage_cfg, f"step_{cfg.actual_online_grpo_max_steps:06d}")
        return path if epath.Path(path).exists() else None
    if not os.path.isdir(actual_grpo_dir):
        return None
    candidates = []
    for name in os.listdir(actual_grpo_dir):
        if name.startswith("step_"):
            try:
                step = int(name.split("_")[-1])
                candidates.append((step, os.path.join(actual_grpo_dir, name)))
            except ValueError:
                continue
    if not candidates:
        return None
    candidates.sort()
    _step, local_path = candidates[-1]
    if cfg is not None and _env_flag("EXPDIS_HF_MIRROR_CHECKPOINTS", "0"):
        repo_id = _hf_artifact_repo()
        if repo_id:
            hf_path = (
                f"{_hf_run_prefix(cfg)}/actual_grpo/checkpoints/"
                f"{os.path.basename(local_path)}"
            )
            return f"hf://{repo_id}/{hf_path}"
    return local_path


def _shared_handoff_required() -> bool:
    try:
        import jax

        return bool(jax.distributed.is_initialized() and jax.process_count() > 1)
    except Exception:
        return False


@contextmanager
def _temporary_env(overrides: dict[str, str], unset: list[str] | tuple[str, ...] = ()):
    old: dict[str, str | None] = {}
    keys = set(overrides) | set(unset)
    for key in keys:
        old[key] = os.environ.get(key)
    try:
        for key in unset:
            os.environ.pop(key, None)
        for key, value in overrides.items():
            os.environ[key] = str(value)
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _build_round_cfg(
    cfg: TrainConfig,
    round_idx: int,
    num_rounds: int,
    init_checkpoint: str,
    root_output_dir: str,
) -> TrainConfig:
    """Build one fixed-total round with explicit weights-only parent lineage."""
    round_dir = os.path.join(root_output_dir, f"round_{round_idx}")
    base_run_name = _pipeline_wandb_run_name(cfg)
    scout_parent = init_checkpoint if round_idx > 1 else ""
    central_parent = (
        scout_parent
        if str(cfg.central_init_policy).lower() == "carryover"
        else ""
    )
    return replace(
        cfg,
        output_dir=round_dir,
        pipeline_root_dir=cfg.pipeline_root_dir or os.path.abspath(root_output_dir),
        wandb_run_name=f"{base_run_name}_round_{round_idx:02d}",
        # Different seed per round controls stochastic sampling inside the
        # round. Disjoint dataset shards are enforced through EXPDIS_DATASET_SHARD_*
        # env vars set by multi_round_main.
        seed=int(cfg.seed) + (round_idx - 1),
        round_index=int(round_idx),
        lambda_novelty=novelty_weight_for_round(
            scalar_weight=float(cfg.lambda_novelty),
            schedule=str(cfg.round_novelty_schedule),
            rounds=int(num_rounds),
            round_index=int(round_idx),
        ),
        # Split only the RL step budgets across rounds when budget_mode="split".
        # The SFT library cap is per round (at most 500 selected rows), so it is
        # deliberately not divided by R.
        grpo_max_steps=_round_budget(cfg.grpo_max_steps, round_idx, num_rounds, cfg.round_budget_mode),
        actual_online_grpo_max_steps=_round_budget(
            cfg.actual_online_grpo_max_steps, round_idx, num_rounds, cfg.round_budget_mode
        ),
        max_distill_examples=int(cfg.max_distill_examples),
        # Round 1 uses Base.  Round r>1 initializes every Scout and (under the
        # flagship carryover policy) SFT from C_(r-1), parameters only.
        init_weights_checkpoint=scout_parent,
        central_parent_checkpoint=central_parent,
        resume_checkpoint="",
        init_checkpoint="",
    )


def multi_round_main():
    """Top-level entrypoint. Dispatches to single-round (two_model) or multi-round."""
    cfg = parse_args()
    _validate_experimental_mrme_gate(cfg)
    if (cfg.pipeline_mode or "two_model").lower() != "multi_round" or int(cfg.num_rounds) <= 1:
        # Standard single-pass Scout->Central pipeline (unchanged behavior).
        return main(cfg)

    # Multi-round: orchestrate N rounds with Explorer←Actual handoff.
    num_rounds = int(cfg.num_rounds)
    root_output_dir = os.path.abspath(cfg.output_dir)
    base_hf_prefix = (
        os.environ.get("EXPDIS_HF_ARTIFACT_PREFIX")
        or _pipeline_wandb_run_name(cfg)
    ).strip().strip("/")
    print(
        f"[multi-round] starting {num_rounds}-round pipeline; root={root_output_dir}; "
        f"budget_mode={cfg.round_budget_mode}; reset_rnd={cfg.multi_round_reset_rnd}",
        flush=True,
    )
    prev_actual_ckpt = ""
    for round_idx in range(1, num_rounds + 1):
        round_cfg = _build_round_cfg(
            cfg, round_idx, num_rounds, prev_actual_ckpt, root_output_dir
        )
        print(
            f"[multi-round] === round {round_idx}/{num_rounds} === "
            f"output_dir={round_cfg.output_dir} "
            f"scout_parent={round_cfg.init_weights_checkpoint!r} "
            f"central_parent={round_cfg.central_parent_checkpoint!r} "
            f"lambda={round_cfg.lambda_novelty:g} K={round_cfg.scouts_per_round} "
            f"grpo_max_steps={round_cfg.grpo_max_steps} actual_max_steps={round_cfg.actual_online_grpo_max_steps}",
            flush=True,
        )
        round_env = {
            # Fresh novelty each round; keep this explicit so restored policy
            # weights do not bring along stale RND predictor state.
            "EXPDIS_MULTIROUND_RESET_RND": "1" if cfg.multi_round_reset_rnd else "0",
            # Ensure a final checkpoint exists even when final_step is not on
            # the periodic save cadence.
            "EXPDIS_FORCE_FINAL_PHASE_CHECKPOINT": "1",
            # Deterministic disjoint train-data shards.
            "EXPDIS_DATASET_SHARD_INDEX": str(round_idx - 1),
            "EXPDIS_DATASET_SHARD_COUNT": str(num_rounds),
            "EXPDIS_DATASET_SHARD_POLICY": cfg.round_shard_policy,
            "EXPDIS_DATASET_SHARD_SEED": str(int(cfg.seed)),
            # Avoid checkpoint-name collisions across rounds in HF artifact
            # storage and make hf:// handoff paths unambiguous.
            "EXPDIS_HF_ARTIFACT_PREFIX": f"{base_hf_prefix}/round_{round_idx:02d}",
        }
        if _env_flag("EXPDIS_HF_MIRROR_CHECKPOINTS", "0"):
            # The next round may restore immediately; async upload is unsafe
            # for handoff checkpoints.
            round_env["EXPDIS_HF_MIRROR_ASYNC"] = "0"
        if _env_flag("EXPDIS_SKIP_FINAL_EVAL", "0") or (round_idx < num_rounds and not _env_flag("EXPDIS_MULTIROUND_EVAL_EACH_ROUND", "0")):
            round_env["EXPDIS_SKIP_FINAL_EVAL"] = "1"
        blocked_env = (
            "EXPDIS_PIPELINE_STOP_AFTER_EXPLORER",
            "EXPDIS_PIPELINE_STOP_AFTER_SFT",
            "EXPDIS_PIPELINE_SKIP_EXPLORER",
            "EXPDIS_EXISTING_TRAJECTORIES",
            "EXPDIS_INIT_CHECKPOINT_LOCAL_PATH",
            "EXPDIS_RESTORE_OPT_STATE",
            "EXPDIS_SKIP_FINAL_EVAL",
            "EXPDIS_WANDB_RUN_ID",
            "EXPDIS_WANDB_EXPLORER_RUN_ID",
            "EXPDIS_WANDB_DISTILL_RUN_ID",
            "EXPDIS_WANDB_ACTUAL_RUN_ID",
            "EXPDIS_WANDB_ACTUAL_GRPO_RUN_ID",
        )
        with _temporary_env(round_env, unset=blocked_env):
            # Run the standard single-pass body for this round's cfg.
            main(round_cfg)
            # Hand off: next round's Explorer starts from this round's Actual.
            next_ckpt = _find_latest_actual_checkpoint(round_cfg.output_dir, round_cfg)
            if next_ckpt is None:
                raise RuntimeError(
                    "multi-round handoff failed: no Actual GRPO checkpoint found under "
                    f"{round_cfg.output_dir}. Refusing to fall back to Base."
                )
            if not cfg.checkpoint_root and next_ckpt.startswith("hf://") is False and _shared_handoff_required():
                raise RuntimeError(
                    "multi-round handoff resolved to a local checkpoint in a multihost run: "
                    f"{next_ckpt}. Enable EXPDIS_HF_MIRROR_CHECKPOINTS=1 and set "
                    "EXPDIS_HF_CHECKPOINT_REPO/EXPDIS_HF_ARTIFACT_REPO so every host can restore."
                )
            print(f"[multi-round] handoff: round {round_idx + 1} init_checkpoint = {next_ckpt}", flush=True)
            prev_actual_ckpt = next_ckpt
        # Free TPU buffers between rounds so the next round starts clean.
        try:
            import jax
            jax.clear_caches()
        except Exception as _e:
            print(f"[multi-round] WARNING: jax.clear_caches() failed: {_e}", flush=True)

    print(f"[multi-round] all {num_rounds} rounds complete; final ckpt={prev_actual_ckpt!r}", flush=True)


if __name__ == "__main__":
    multi_round_main()
