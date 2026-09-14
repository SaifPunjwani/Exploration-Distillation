"""Explorer GRPO training loop (JAX / TPU) with full FSDP sharding.

Design:
  * One-time ``jax.distributed.initialize()`` for multi-host TPU (v4-32, v5e-64, v6e-64).
  * FSDP mesh over every chip; params + optimizer state sharded along the leading axis.
  * ``jax.jit`` with explicit ``in_shardings`` / ``out_shardings`` so GSPMD inserts the
    right collectives (no guesswork at first compile).
  * bf16 compute, fp32 parameters (AdamW stability).
  * vLLM HTTP for generation; ``_rescore_old_logps`` rescreens the current
    policy each round → eliminates stale-vLLM KL blow-up by construction.
  * Checkpoints via Orbax, traj JSONL persisted to `cfg.output_dir`.
"""

from __future__ import annotations

import json
import os
import gc
import hashlib
import http.server
import shlex
import shutil
import socket
import subprocess
import sys
import pickle
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from jax.experimental import multihost_utils
from jax.sharding import NamedSharding, PartitionSpec as P
from transformers import AutoTokenizer

# Persistent XLA compile cache. First training step on a fresh pod does ~5-10
# minutes of JIT/XLA compilation. Without this cache, every restart pays it
# again. With this cache, identical shapes hit disk-cached binaries.
#
# launch_jax_pipeline.sh already exports JAX_COMPILATION_CACHE_DIR and
# JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS (default lowered to 1s). This
# block re-applies them via jax.config in case the entrypoint is invoked
# directly (e.g. python -m tmx_jax.pipeline) without sourcing the launch
# script. Set TMX_DISABLE_COMPILE_CACHE=1 to opt out.
if os.environ.get("TMX_DISABLE_COMPILE_CACHE", "0").lower() not in ("1", "true", "yes", "on"):
    _COMPILE_CACHE_DIR = os.environ.get(
        "JAX_COMPILATION_CACHE_DIR",
        os.path.expanduser("~/.cache/tmx_jax_compilation_cache"),
    )
    _MIN_COMPILE_SECS = int(
        os.environ.get("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "1") or "1"
    )
    try:
        os.makedirs(_COMPILE_CACHE_DIR, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", _COMPILE_CACHE_DIR)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", _MIN_COMPILE_SECS)
        # Cache entries even when small — RND step, novelty feature_step, and
        # rescore_jit are all valuable to persist.
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    except Exception as _e:
        print(f"[tmx-jax] WARNING: could not enable persistent compile cache: {_e}", flush=True)

from .config import TrainConfig, parse_args, validate_contract
from .data import DapoExample, load_dapo, load_examples
from .generate import batched_generate, parse_server_urls
from .grpo import (
    GrpoConfig,
    chunked_per_token_logps_from_hidden,
    compute_group_advantages,
    dr_grpo_loss_from_logps,
)
from .mesh import FSDP_AXIS, build_mesh, named, shardings_for_params
from .model import Qwen3Config, Qwen3Model
from .novelty import init_rnd_map, restore_rnd_map, score_and_update_rnd_map
from .rewarding import (
    analyze_completion_text,
    completion_quality_metrics,
    extract_presented_answer,
    is_correct_completion,
    normalize_generated_completion,
    normalize_values,
    resolve_novelty_text,
)
from .weights import export_flax_params_to_hf_dir, hf_to_flax_params, load_hf_config


_BATCH_SHARD_PAYLOADS: Dict[int, bytes] = {}
_BATCH_SHARD_SERVER = None
_BATCH_SHARD_SERVED_RANKS: set[int] = set()
_BATCH_SHARD_LOCK = threading.Lock()
_BATCH_SHARD_SOURCE_HOST_CACHE: Optional[str] = None


class _BatchShardHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        try:
            rank = int(self.path.rstrip("/").split("/")[-1].split("?")[0])
            payload = _BATCH_SHARD_PAYLOADS[rank]
        except Exception:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
        with _BATCH_SHARD_LOCK:
            _BATCH_SHARD_SERVED_RANKS.add(rank)

    def log_message(self, *_args):
        return


def _ensure_batch_shard_server(port: int) -> None:
    global _BATCH_SHARD_SERVER
    if _BATCH_SHARD_SERVER is not None:
        return
    server = http.server.ThreadingHTTPServer(("0.0.0.0", int(port)), _BatchShardHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _BATCH_SHARD_SERVER = server


def _coordinator_host() -> str:
    batch_source = (os.environ.get("TMX_BATCH_SHARD_SOURCE_HOST") or "").strip()
    if batch_source:
        return batch_source.rsplit(":", 1)[0]
    coord = (os.environ.get("JAX_COORDINATOR_ADDRESS") or "").strip()
    if coord:
        return coord.rsplit(":", 1)[0]
    return "127.0.0.1"


def _local_routable_host() -> str:
    for key in ("TMX_TPU_WORKER_IP", "TMX_LOCAL_HOST_IP"):
        value = (os.environ.get(key) or "").strip()
        if value:
            return value.rsplit(":", 1)[0]
    try:
        for value in subprocess.check_output(["hostname", "-I"], text=True).split():
            if value and not value.startswith(("127.", "169.254.")):
                return value
    except Exception:
        pass
    try:
        return socket.gethostbyname(socket.gethostname())
    except Exception:
        return _coordinator_host()


def _batch_shard_source_host() -> str:
    """Return the host IP that is actually serving batch shards.

    PJRT/libtpu can remap physical TPU VM worker IDs to dense JAX process ranks,
    so rank 0 is not necessarily the worker used as JAX coordinator. Broadcast
    rank 0's routable IP before non-source ranks issue HTTP shard fetches.
    """
    explicit = (os.environ.get("TMX_BATCH_SHARD_SOURCE_HOST") or "").strip()
    if explicit:
        return explicit.rsplit(":", 1)[0]
    if jax.process_count() <= 1:
        return _local_routable_host()
    global _BATCH_SHARD_SOURCE_HOST_CACHE
    if _BATCH_SHARD_SOURCE_HOST_CACHE:
        return _BATCH_SHARD_SOURCE_HOST_CACHE
    is_source = int(jax.process_index()) == 0
    host = _local_routable_host() if is_source else ""
    encoded = host.encode("utf-8")[:255]
    payload = {
        "len": np.asarray([len(encoded)], dtype=np.int32),
        "bytes": np.pad(
            np.frombuffer(encoded, dtype=np.uint8),
            (0, max(0, 255 - len(encoded))),
            constant_values=0,
        ).astype(np.uint8),
    }
    payload = multihost_utils.broadcast_one_to_all(payload, is_source=is_source)
    n = int(np.asarray(payload["len"], dtype=np.int32).reshape(-1)[0])
    data = np.asarray(payload["bytes"], dtype=np.uint8).reshape(-1)[:n].tobytes()
    host = data.decode("utf-8") if data else _coordinator_host()
    _BATCH_SHARD_SOURCE_HOST_CACHE = host
    _batch_shard_log(f"resolved source_host={host}")
    return host


def _batch_shard_log(message: str) -> None:
    if os.environ.get("TMX_BATCH_SHARD_DEBUG", "1") in ("0", "false", "False"):
        return
    try:
        print(
            f"[batch-shard] rank={jax.process_index()}/{jax.process_count()} "
            f"host={os.uname().nodename} {message}",
            flush=True,
        )
    except Exception:
        print(f"[batch-shard] {message}", flush=True)


def _stable_wandb_run_id(cfg: TrainConfig, metric_prefix: str) -> str:
    """Stable W&B id so preempt/resume appends to the same stage run."""
    env_by_prefix = {
        "explorer_grpo": "TMX_WANDB_EXPLORER_RUN_ID",
        "actual_grpo": "TMX_WANDB_ACTUAL_GRPO_RUN_ID",
    }
    keys = []
    if metric_prefix == "actual_grpo":
        keys.append("TMX_WANDB_ACTUAL_RUN_ID")
    keys.extend([env_by_prefix.get(metric_prefix, ""), "TMX_WANDB_RUN_ID"])
    for key in keys:
        if key and os.environ.get(key):
            return os.environ[key].strip()
    run_name = cfg.wandb_run_name or f"jax_lambda{cfg.lambda_novelty}"
    if metric_prefix == "actual_grpo":
        digest = hashlib.sha1(f"{run_name}:actual".encode("utf-8")).hexdigest()[:24]
        return f"jaxactual{digest}"
    digest = hashlib.sha1(f"{run_name}:{metric_prefix}".encode("utf-8")).hexdigest()[:24]
    return f"jax{digest}"


def _env_flag(name: str, default: str = "1") -> bool:
    return (os.environ.get(name, default) or "").strip().lower() not in ("0", "false", "no", "off")


def _env_int(name: str, default: int = 0) -> int:
    try:
        return int(os.environ.get(name, str(default)) or str(default))
    except ValueError:
        return int(default)


def _step_due(step: int, every_steps: int, first_step: int = 0) -> bool:
    return (first_step > 0 and step == first_step) or (
        every_steps > 0 and step % every_steps == 0
    )


def _wandb_log_kwargs(step: int, metric_prefix: str | None = None) -> dict:
    # Prefer custom phase step metrics over W&B's global `_step`.
    #
    # When a run resumes from an earlier checkpoint into the same W&B run, W&B
    # drops logs whose global `step=` is lower than the previous max global step.
    # We already log `<phase>/step` and define `<phase>/*` against it, so leaving
    # global `_step` implicit preserves all future resume logs while keeping the
    # charts keyed by the true Explorer/Actual optimizer step.
    if _env_flag("TMX_WANDB_SINGLE_RUN", "1"):
        if not _env_flag("TMX_WANDB_EXPLICIT_STEP", "0"):
            return {}
        offsets = {
            "explorer_grpo": 0,
            "actual_grpo": 1_000_000,
        }
        return {"step": offsets.get(metric_prefix or "", 0) + int(step)}
    return {"step": step}


_WANDB_TOP_LEVEL_METRICS = {
    "reward",
    "reward_rolling_25",
    "reward_trend_25",
    "correctness",
    "correctness_rolling_25",
    "correctness_trend_25",
    "correctness_used_mean",
    "valid_answer_fraction",
    "terminated_fraction",
    "terminated_correctness_mean",
    "clipped_fraction",
    "completion_token_length_mean",
    "selected_token_probability_mean",
    "overlong_penalty_mean",
    "effective_group_fraction",
    "clip_fraction",
    "up_clip_fraction",
    "up_clipped_probability_mean",
    "kl",
    "ref_kl",
    "grad_norm",
}


def _wandb_add_metric_pair(payload: dict, phase: str, name: str, value: float | int) -> None:
    """Log phase metrics, with only decision-critical metrics duplicated top-level."""
    if name in _WANDB_TOP_LEVEL_METRICS:
        payload[name] = value
    payload[f"{phase}/{name}"] = value


_WANDB_MINIMAL_PHASE_METRICS = {
    "step",
    "reward",
    "reward_rolling_25",
    "reward_trend_25",
    "correctness",
    "correctness_rolling_25",
    "correctness_used_mean",
    "valid_answer_fraction",
    "terminated_fraction",
    "terminated_correctness_mean",
    "clipped_fraction",
    "completion_token_length_mean",
    "selected_token_probability_mean",
    "selected_token_logprob_mean",
    "overlong_penalty_mean",
    "effective_group_fraction",
    "informative_group_count",
    "all_wrong_group_fraction",
    "all_correct_group_fraction",
    "adv_std",
    "clip_fraction",
    "up_clip_fraction",
    "up_clipped_probability_mean",
    "kl",
    "ref_kl",
    "grad_norm",
    "rollout_time_s",
    "train_time_s",
    "novelty_reward_mean",
    "novelty_raw_mean",
    "novelty_used_mean",
    "terminated_novelty_mean",
}


def _metric_window_stats(values) -> Dict[str, float]:
    """Small-window diagnostics for noisy binary rewards.

    `trend` is the mean of the second half minus the mean of the first half.
    It is not used for optimization; it just makes W&B learning/collapse plots
    easier to read than raw per-step reward.
    """
    arr = np.asarray(list(values), dtype=np.float32)
    if arr.size == 0:
        return {"mean": 0.0, "trend": 0.0, "ci95": 0.0}
    mean = float(arr.mean())
    if arr.size < 2:
        return {"mean": mean, "trend": 0.0, "ci95": 0.0}
    half = max(1, arr.size // 2)
    trend = float(arr[-half:].mean() - arr[:half].mean())
    ci95 = float(1.96 * arr.std(ddof=1) / np.sqrt(arr.size))
    return {"mean": mean, "trend": trend, "ci95": ci95}


# --------------------------------------------------------------------------- #
# Multi-host init
# --------------------------------------------------------------------------- #

def init_distributed() -> None:
    """JAX distributed init for multi-host TPU slices.

    On single-host (GPU / v4-8), no-op.
    On multi-host (v5e-64 = 16 hosts etc.), reads ``JAX_COORDINATOR_ADDRESS`` +
    ``JAX_PROCESS_ID`` / ``JAX_NUM_PROCESSES`` from env (set by the TPU VM
    runtime) and initializes the distributed runtime.
    """
    expected_num_processes = None
    expected_process_id = None
    if os.environ.get("JAX_NUM_PROCESSES"):
        try:
            expected_num_processes = int(os.environ["JAX_NUM_PROCESSES"])
            expected_process_id = int(os.environ.get("JAX_PROCESS_ID", "0"))
        except ValueError:
            expected_num_processes = None
            expected_process_id = None

    def _check_expected_world(label: str) -> None:
        if expected_num_processes is None or expected_process_id is None:
            return
        actual_num_processes = int(jax.process_count())
        actual_process_id = int(jax.process_index())
        # PJRT/libtpu can remap TPU VM worker IDs to topology ranks. Require
        # the intended full world size, but do not require rank == worker ID.
        if expected_num_processes > 1 and actual_num_processes != expected_num_processes:
            raise RuntimeError(
                f"JAX distributed world mismatch after {label}: "
                f"rank={actual_process_id}/{actual_num_processes}, "
                f"expected_num_hosts={expected_num_processes}. "
                "Refusing to continue with duplicated single-host training."
            )

    if jax.distributed.is_initialized():
        print(
            f"[dist] already initialized: rank={jax.process_index()} "
            f"num_hosts={jax.process_count()} devices={len(jax.devices())}",
            flush=True,
        )
        _check_expected_world("already initialized")
        return

    if os.environ.get("JAX_COORDINATOR_ADDRESS") and os.environ.get("JAX_NUM_PROCESSES"):
        try:
            coordinator_address = os.environ["JAX_COORDINATOR_ADDRESS"]
            num_processes = int(os.environ["JAX_NUM_PROCESSES"])
            process_id = int(os.environ.get("JAX_PROCESS_ID", "0"))
            local_device_ids_env = os.environ.get("JAX_LOCAL_DEVICE_IDS", "").strip()
            local_device_ids = None
            if local_device_ids_env:
                local_device_ids = [int(x) for x in local_device_ids_env.split(",") if x.strip()]
            elif os.environ.get("TPU_VISIBLE_CHIPS", "").strip():
                local_device_ids = [
                    int(x) for x in os.environ["TPU_VISIBLE_CHIPS"].split(",") if x.strip()
                ]
            coordinator_bind_address = os.environ.get("JAX_COORDINATOR_BIND_ADDRESS")
            initialization_timeout = int(os.environ.get("JAX_INITIALIZATION_TIMEOUT", "600"))
            init_kwargs = {
                "coordinator_address": coordinator_address,
                "num_processes": num_processes,
                "process_id": process_id,
                "initialization_timeout": initialization_timeout,
                "heartbeat_timeout_seconds": int(os.environ.get("JAX_HEARTBEAT_TIMEOUT_SECONDS", "600")),
            }
            if local_device_ids is not None:
                init_kwargs["local_device_ids"] = local_device_ids
            if coordinator_bind_address:
                init_kwargs["coordinator_bind_address"] = coordinator_bind_address
            cluster_detection_method = os.environ.get("TMX_JAX_CLUSTER_DETECTION_METHOD", "").strip()
            if cluster_detection_method:
                init_kwargs["cluster_detection_method"] = cluster_detection_method
            jax.distributed.initialize(**init_kwargs)
            print(f"[dist] jax.distributed.initialize ok: rank={jax.process_index()} "
                  f"num_hosts={jax.process_count()} devices={len(jax.devices())}", flush=True)
            _check_expected_world("jax.distributed.initialize")
            return
        except Exception as e:
            print(f"[dist] init failed: {e}", flush=True)
            raise

    tpu_worker_hosts = os.environ.get("TPU_WORKER_HOSTNAMES", "").strip()
    tpu_worker_id = os.environ.get("TPU_WORKER_ID", "").strip()
    if tpu_worker_hosts and tpu_worker_id and "," in tpu_worker_hosts:
        try:
            jax.distributed.initialize()
            print(
                f"[dist] TPU auto init ok: rank={jax.process_index()} "
                f"num_hosts={jax.process_count()} devices={len(jax.devices())}",
                flush=True,
            )
            _check_expected_world("TPU auto init")
            return
        except Exception as e:
            print(f"[dist] TPU auto init failed: {e}", flush=True)
            raise
    # vLLM-on-v5e style single-host: JAX sees all local chips.


# --------------------------------------------------------------------------- #
# Config + state
# --------------------------------------------------------------------------- #

def _yarn_attention_factor(rope: Dict[str, Any]) -> float:
    if "attention_factor" in rope and rope["attention_factor"] is not None:
        return float(rope["attention_factor"])
    factor = float(rope.get("factor", 1.0) or 1.0)
    if factor <= 1.0:
        return 1.0

    def get_mscale(scale: float, mscale: float = 1.0) -> float:
        return 1.0 if scale <= 1.0 else 0.1 * float(mscale) * np.log(scale) + 1.0

    mscale = rope.get("mscale")
    mscale_all_dim = rope.get("mscale_all_dim")
    if mscale is not None and mscale_all_dim is not None:
        return float(get_mscale(factor, float(mscale)) / get_mscale(factor, float(mscale_all_dim)))
    return float(get_mscale(factor))


def _make_qwen3_config(model_name: str) -> Qwen3Config:
    c_full = load_hf_config(model_name)
    model_name_l = (model_name or "").lower()
    model_type = str(c_full.get("model_type", "")).lower()
    text_cfg = c_full.get("text_config") if isinstance(c_full.get("text_config"), dict) else None
    if text_cfg is not None and (
        "ministral" in model_name_l
        or model_type in {"mistral3", "pixtral"}
        or str(text_cfg.get("model_type", "")).lower() == "ministral3"
    ):
        c = text_cfg
        rope = c.get("rope_parameters") or c.get("rope_scaling") or {}
        rope_type = str(rope.get("rope_type", rope.get("type", "default")) or "default")
        return Qwen3Config(
            hidden_size=int(c["hidden_size"]),
            intermediate_size=int(c["intermediate_size"]),
            num_hidden_layers=int(c["num_hidden_layers"]),
            num_attention_heads=int(c["num_attention_heads"]),
            num_key_value_heads=int(c["num_key_value_heads"]),
            head_dim=int(c.get("head_dim", c["hidden_size"] // c["num_attention_heads"])),
            rope_theta=float(rope.get("rope_theta", c.get("rope_theta", 1_000_000.0))),
            rms_norm_eps=float(c.get("rms_norm_eps", 1e-5)),
            vocab_size=int(c["vocab_size"]),
            max_position_embeddings=int(c.get("max_position_embeddings", 262144)),
            tie_word_embeddings=bool(c.get("tie_word_embeddings", True)),
            use_qk_norm=False,
            hf_weight_prefix="language_model.model",
            architecture="ministral3",
            rope_type=rope_type,
            rope_factor=float(rope.get("factor", 1.0) or 1.0),
            rope_attention_factor=_yarn_attention_factor(rope) if rope_type == "yarn" else 1.0,
            rope_beta_fast=float(rope.get("beta_fast", 32.0) or 32.0),
            rope_beta_slow=float(rope.get("beta_slow", 1.0) or 1.0),
            rope_original_max_position_embeddings=int(
                rope.get("original_max_position_embeddings", c.get("max_position_embeddings", 262144))
            ),
            rope_truncate=bool(rope.get("truncate", True)),
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
        )
    if "qwen3" not in model_name_l:
        raise NotImplementedError(
            "raw-tpu-tmx-jax training currently supports Qwen3 plus text-only Ministral3 checkpoints. "
            f"Got model_name={model_name!r}. Gemma 4 E4B-it is supported for forced vLLM startup only, "
            "but is not eval-safe on this TPU stack."
        )
    c = c_full
    return Qwen3Config(
        hidden_size=int(c["hidden_size"]),
        intermediate_size=int(c["intermediate_size"]),
        num_hidden_layers=int(c["num_hidden_layers"]),
        num_attention_heads=int(c["num_attention_heads"]),
        num_key_value_heads=int(c["num_key_value_heads"]),
        head_dim=int(c.get("head_dim", c["hidden_size"] // c["num_attention_heads"])),
        rope_theta=float(c.get("rope_theta", 1_000_000.0)),
        rms_norm_eps=float(c.get("rms_norm_eps", 1e-6)),
        vocab_size=int(c["vocab_size"]),
        max_position_embeddings=int(c.get("max_position_embeddings", 40960)),
        tie_word_embeddings=bool(c.get("tie_word_embeddings", False)),
        use_qk_norm=True,
        hf_weight_prefix="model",
        architecture="qwen3",
        rope_type="default",
        dtype=jnp.bfloat16,          # compute in bf16
        param_dtype=jnp.float32,     # store params in fp32 (AdamW stability)
    )


def _adamw_tx(cfg: TrainConfig, learning_rate: float):
    if cfg.optimizer.strip().lower() != "adamw":
        raise ValueError(
            "Contract training path supports only optimizer='adamw'; "
            f"got {cfg.optimizer!r}"
        )
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=learning_rate,
            b1=cfg.adam_beta1,
            b2=cfg.adam_beta2,
            eps=cfg.adam_epsilon,
            weight_decay=cfg.weight_decay,
        ),
    )


def _put_param_on_mesh(value, sharding, mesh):
    """Place a host parameter array on a multi-host mesh using host-local input.

    `jax.device_put(full_global_array, NamedSharding(...))` is fragile on large
    multi-host TPU pods because every host tries to provide the full global
    buffer. For sharded leaves, pass each host only its process-local slice and
    let JAX assemble the global array.
    """
    if jax.process_count() <= 1:
        return jax.device_put(value, sharding)
    spec = tuple(getattr(sharding, "spec", ()) or ())
    axes = [idx for idx, axis in enumerate(spec) if axis == FSDP_AXIS]
    arr = np.asarray(value)
    if not axes:
        local_arr = arr
    elif len(axes) == 1:
        axis = axes[0]
        if arr.shape[axis] % jax.process_count() != 0:
            raise ValueError(
                f"Cannot host-shard param shape={arr.shape} spec={spec}: "
                f"axis {axis} not divisible by process_count={jax.process_count()}"
            )
        per_process = arr.shape[axis] // jax.process_count()
        start = jax.process_index() * per_process
        stop = start + per_process
        slices = [slice(None)] * arr.ndim
        slices[axis] = slice(start, stop)
        local_arr = np.ascontiguousarray(arr[tuple(slices)])
    else:
        raise ValueError(f"Unsupported multi-axis param sharding spec={spec}")
    return multihost_utils.host_local_array_to_global_array(local_arr, mesh, sharding.spec)


def _restore_param_on_mesh(value, sharding, mesh):
    """Place restored checkpoint leaves without fetching global JAX arrays.

    Orbax can restore directly into the sharding described by ``restore_args``.
    In that case each leaf is already a global ``jax.Array`` and converting it
    through ``np.asarray`` would try to fetch non-addressable remote shards on
    every host. Host arrays from older checkpoints still go through the normal
    host-local placement path.
    """
    if isinstance(value, jax.Array):
        if getattr(value, "sharding", None) == sharding:
            return value
        return jax.device_put(value, sharding)
    return _put_param_on_mesh(value, sharding, mesh)


def _zeros_param_on_mesh(shape_leaf, sharding, mesh):
    """Create a zero-valued global array for a parameter shape.

    Resume jobs only need a correctly shaped/sharded item tree so Orbax can
    restore into it. Loading base HF weights on every TPU host before restoring
    the real checkpoint is slow and can hang on large pods.
    """
    shape = tuple(int(x) for x in shape_leaf.shape)
    dtype = getattr(shape_leaf, "dtype", np.float32)
    if jax.process_count() <= 1:
        return jax.device_put(np.zeros(shape, dtype=dtype), sharding)
    spec = tuple(getattr(sharding, "spec", ()) or ())
    axes = [idx for idx, axis in enumerate(spec) if axis == FSDP_AXIS]
    local_shape = list(shape)
    if not axes:
        pass
    elif len(axes) == 1:
        axis = axes[0]
        if shape[axis] % jax.process_count() != 0:
            raise ValueError(
                f"Cannot host-shard zero param shape={shape} spec={spec}: "
                f"axis {axis} not divisible by process_count={jax.process_count()}"
            )
        local_shape[axis] = shape[axis] // jax.process_count()
    else:
        raise ValueError(f"Unsupported multi-axis param sharding spec={spec}")
    local_arr = np.zeros(tuple(local_shape), dtype=dtype)
    return multihost_utils.host_local_array_to_global_array(local_arr, mesh, sharding.spec)


def _init_state(cfg: TrainConfig, mesh):
    m_cfg = _make_qwen3_config(cfg.model_name)
    from .model import set_active_mesh
    set_active_mesh(mesh)
    model = Qwen3Model(m_cfg)
    empty_for_restore = (
        bool(cfg.init_checkpoint)
        and os.environ.get("TMX_INIT_EMPTY_FOR_RESTORE", "0") not in ("0", "false", "False")
    )
    if empty_for_restore:
        print("[train:init] building empty param shape tree for checkpoint restore", flush=True)
        dummy_ids = jnp.zeros((1, 1), dtype=jnp.int32)
        dummy_mask = jnp.ones((1, 1), dtype=jnp.int32)
        params = jax.eval_shape(
            lambda: model.init(jax.random.PRNGKey(0), dummy_ids, dummy_mask)
        )["params"]
    else:
        print("[train:init] loading base HF params", flush=True)
        params_tree = hf_to_flax_params(
            cfg.model_name,
            m_cfg.num_hidden_layers,
            m_cfg.tie_word_embeddings,
            dtype=jnp.float32,
            use_qk_norm=bool(m_cfg.use_qk_norm),
            hf_weight_prefix=str(m_cfg.hf_weight_prefix),
        )
        params = params_tree["params"]
        print("[train:init] loaded base HF params", flush=True)

    # Apply FSDP sharding to params.
    param_sh = shardings_for_params(params, mesh)
    print("[train:init] placing params on mesh", flush=True)
    if empty_for_restore:
        params = jax.tree_util.tree_map(lambda x, s: _zeros_param_on_mesh(x, s, mesh), params, param_sh)
    else:
        params = jax.tree_util.tree_map(lambda x, s: _put_param_on_mesh(x, s, mesh), params, param_sh)
    print("[train:init] params placed on mesh", flush=True)

    tx = _adamw_tx(cfg, cfg.grpo_lr)

    def init_fn(raw_params):
        return train_state.TrainState.create(apply_fn=model.apply, params=raw_params, tx=tx)

    print("[train:init] building TrainState shape", flush=True)
    state_shape = jax.eval_shape(init_fn, params)
    print("[train:init] TrainState shape built", flush=True)

    def infer_leaf_sh(leaf):
        if getattr(leaf, "ndim", 0) >= 2:
            if int(leaf.shape[0]) % int(mesh.size) == 0:
                return NamedSharding(mesh, P(FSDP_AXIS, *([None] * (leaf.ndim - 1))))
            if int(leaf.shape[-1]) % int(mesh.size) == 0:
                return NamedSharding(mesh, P(*([None] * (leaf.ndim - 1)), FSDP_AXIS))
            return NamedSharding(mesh, P())
        return NamedSharding(mesh, P())

    state_sh = jax.tree_util.tree_map(infer_leaf_sh, state_shape)
    state_sh = state_sh.replace(params=param_sh)
    jit_init = jax.jit(init_fn, out_shardings=state_sh)
    print("[train:init] compiling TrainState init", flush=True)
    state = jit_init(params)
    print("[train:init] TrainState init complete", flush=True)
    return state, model, m_cfg, param_sh


def _orbax_checkpoint_path(path: str) -> str:
    path = str(path)
    if path.startswith("hf://"):
        local_override = os.environ.get("TMX_INIT_CHECKPOINT_LOCAL_PATH", "").strip()
        if local_override:
            local_path = os.path.abspath(local_override)
            if not os.path.exists(local_path):
                raise FileNotFoundError(
                    "TMX_INIT_CHECKPOINT_LOCAL_PATH is set but does not exist: "
                    f"{local_path!r}"
                )
            print(f"[ckpt] using local checkpoint override: {local_path}", flush=True)
            return local_path
        raw = path[len("hf://") :].strip("/")
        parts = raw.split("/", 2)
        if len(parts) < 3:
            raise ValueError(
                "HF checkpoint URI must be hf://owner/repo/path/to/checkpoint, "
                f"got {path!r}"
            )
        repo_id = "/".join(parts[:2])
        path_in_repo = parts[2].rstrip("/")
        from huggingface_hub import snapshot_download

        snapshot_root = snapshot_download(
            repo_id=repo_id,
            repo_type=os.environ.get("TMX_HF_REPO_TYPE", "dataset"),
            allow_patterns=[f"{path_in_repo}/**"],
        )
        local_path = os.path.join(snapshot_root, path_in_repo)
        if not os.path.exists(local_path):
            raise FileNotFoundError(
                f"HF checkpoint URI {path!r} downloaded snapshot, but local path "
                f"{local_path!r} does not exist"
            )
        return local_path
    if path.startswith("gs://"):
        return path
    return os.path.abspath(path)


def _maybe_restore_state_from_checkpoint(state, param_sh, cfg: TrainConfig, mesh):
    if not cfg.init_checkpoint:
        return state, 0
    import orbax.checkpoint as ocp
    import orbax.checkpoint.checkpoint_utils as ocp_utils

    print(f"[ckpt] resolving init checkpoint: {cfg.init_checkpoint}", flush=True)
    restore_path = _orbax_checkpoint_path(cfg.init_checkpoint)
    print(f"[ckpt] restore path resolved: {restore_path}", flush=True)
    # Direct-GCS multihost checkpoints contain global sharded arrays. Supplying
    # an item tree and explicit restore args gives Orbax concrete destination
    # shardings instead of replaying saved sharding metadata from a different
    # TPU topology.
    restore_item = {"params": state.params, "step": np.array(0, dtype=np.int64)}
    print("[ckpt] constructing restore args", flush=True)
    restore_args = ocp_utils.construct_restore_args(restore_item, strict=False)
    print("[ckpt] starting Orbax restore", flush=True)
    ckpt = ocp.PyTreeCheckpointer().restore(
        restore_path,
        item=restore_item,
        restore_args=restore_args,
        partial_restore=True,
    )
    print("[ckpt] Orbax restore returned", flush=True)
    params = ckpt.get("params")
    if params is None:
        raise RuntimeError(f"Checkpoint missing params: {cfg.init_checkpoint}")
    print("[ckpt] placing restored params on mesh", flush=True)
    params = jax.tree_util.tree_map(lambda x, s: _restore_param_on_mesh(x, s, mesh), params, param_sh)
    print("[ckpt] restored params placed on mesh", flush=True)
    state = state.replace(params=params)
    opt_state = ckpt.get("opt_state")
    restore_opt_state = os.environ.get("TMX_RESTORE_OPT_STATE", "1") not in (
        "0",
        "false",
        "False",
    )
    if opt_state is not None and restore_opt_state:
        try:
            opt_state = jax.tree_util.tree_map(
                lambda x, ref: jax.device_put(x, ref.sharding) if hasattr(ref, "sharding") else x,
                opt_state,
                state.opt_state,
            )
            state = state.replace(opt_state=opt_state)
        except Exception as e:
            print(
                "[ckpt] WARN optimizer-state restore failed; "
                f"continuing with fresh optimizer state: {e}",
                flush=True,
            )
    elif opt_state is not None:
        print("[ckpt] skipped optimizer-state restore (TMX_RESTORE_OPT_STATE=0)", flush=True)
    restored_step = int(ckpt.get("step", 0) or 0)
    print(f"[train] restored checkpoint: {cfg.init_checkpoint} step={restored_step}", flush=True)
    return state, restored_step


def _maybe_restore_rnd_from_checkpoint(rnd_map, cfg: TrainConfig):
    if not cfg.init_checkpoint or rnd_map is None:
        return rnd_map
    # Multi-round mode: skip RND restore so each round gets a fresh novelty
    # signal. Without this, after round 1 the RND predictor has converged to
    # mimic the target on training-distribution texts, collapsing novelty to
    # ~0 for all subsequent rounds.
    if os.environ.get("TMX_MULTIROUND_RESET_RND", "0").lower() in ("1", "true", "yes", "on"):
        print("[novelty] TMX_MULTIROUND_RESET_RND=1 — skipping RND restore for fresh per-round signal", flush=True)
        return rnd_map
    try:
        import orbax.checkpoint as ocp
        import orbax.checkpoint.checkpoint_utils as ocp_utils

        restore_item = {"rnd_state": rnd_map}
        ckpt = ocp.PyTreeCheckpointer().restore(
            _orbax_checkpoint_path(cfg.init_checkpoint),
            item=restore_item,
            restore_args=ocp_utils.construct_restore_args(restore_item, strict=False),
            partial_restore=True,
        )
        restored = ckpt.get("rnd_state") if isinstance(ckpt, dict) else None
        if restored:
            print(f"[novelty] restored RND state from {cfg.init_checkpoint}", flush=True)
            return restore_rnd_map(rnd_map, restored)
    except Exception as e:
        print(f"[novelty] WARNING: failed to restore RND state: {e}", flush=True)
    return rnd_map


# --------------------------------------------------------------------------- #
# Rollout batch builder (host-side)
# --------------------------------------------------------------------------- #

def _make_novelty_feature_step(model: Qwen3Model, layers: Tuple[int, ...], pool: str):
    layers = tuple(int(x) for x in layers)
    pool = (pool or "mean").lower()

    @jax.jit
    def feature_step(params, input_ids, attention_mask):
        return model.apply(
            {"params": params},
            input_ids,
            attention_mask,
            return_pooled_layers=layers,
            layer_pool=pool,
        )

    return feature_step


def _encode_novelty_texts(tokenizer, texts: List[str], cfg: TrainConfig) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    max_len = int(cfg.novelty_feature_max_length or cfg.max_total_len)
    print(f"[novelty] encoding texts rows={len(texts)} max_len={max_len}", flush=True)
    trunc_mode = (os.environ.get("TMX_NOVELTY_TRUNCATION_MODE") or "tokenizer").lower()
    if trunc_mode == "head_tail":
        pad_id = int(getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", 0) or 0)
        encoded_rows: List[List[int]] = []
        original_lengths: List[int] = []
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=True)
            original_lengths.append(int(len(ids)))
            if len(ids) > max_len:
                head = max_len // 2
                tail = max_len - head
                ids = ids[:head] + ids[-tail:]
            encoded_rows.append([int(x) for x in ids])
        input_ids = np.full((len(encoded_rows), max_len), pad_id, dtype=np.int32)
        attention_mask = np.zeros((len(encoded_rows), max_len), dtype=np.int32)
        for i, ids in enumerate(encoded_rows):
            n = min(len(ids), max_len)
            if n:
                input_ids[i, :n] = np.asarray(ids[:n], dtype=np.int32)
                attention_mask[i, :n] = 1
        encoded_lengths = [min(n, max_len) for n in original_lengths]
        truncated_count = sum(1 for n in original_lengths if n > max_len)
    else:
        enc = tokenizer(
            texts,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="np",
        )
        input_ids = enc["input_ids"].astype(np.int32)
        attention_mask = enc["attention_mask"].astype(np.int32)
        encoded_lengths = [int(x) for x in attention_mask.sum(axis=1).tolist()]
        truncated_count = sum(1 for n in encoded_lengths if int(n) >= max_len)
    stats = {
        "count": float(len(encoded_lengths)),
        "truncated_count": float(truncated_count),
        "token_length_sum": float(sum(encoded_lengths)),
        "max_input_tokens": float(max(encoded_lengths) if encoded_lengths else 0.0),
        "max_length": float(max_len),
    }
    print(
        f"[novelty] encoding complete rows={len(texts)} "
        f"max_input_tokens={stats['max_input_tokens']} truncated={stats['truncated_count']}",
        flush=True,
    )
    return input_ids, attention_mask, stats


def _novelty_feature_stats_array(stats: Optional[Dict[str, float]] = None) -> np.ndarray:
    stats = stats or {}
    return np.asarray(
        [
            float(stats.get("count", 0.0)),
            float(stats.get("truncated_count", 0.0)),
            float(stats.get("token_length_sum", 0.0)),
            float(stats.get("max_input_tokens", 0.0)),
            float(stats.get("max_length", 0.0)),
        ],
        dtype=np.float32,
    )


def _novelty_feature_stats_dict(values: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    padded = np.zeros((5,), dtype=np.float32)
    padded[: min(5, arr.shape[0])] = arr[:5]
    return {
        "count": float(padded[0]),
        "truncated_count": float(padded[1]),
        "token_length_sum": float(padded[2]),
        "max_input_tokens": float(padded[3]),
        "max_length": float(padded[4]),
    }


def _rnd_input_dim(rnd_map: Optional[Dict[str, Any]]) -> int:
    if not rnd_map:
        return 0
    for item in rnd_map.values():
        try:
            return int(np.asarray(item["target"]["fc1"]["kernel"]).shape[0])
        except Exception:
            continue
    return 0


def _hash_text_to_feature(text: str, dim: int) -> np.ndarray:
    """Deterministic local novelty feature used when TPU hidden-state collectives fail.

    This is intentionally simple and stable: it preserves the RND online novelty
    contract without requiring a second sharded model forward through every TPU
    host. Hidden-state novelty remains the default path.
    """
    dim = int(dim)
    out = np.empty((dim,), dtype=np.float32)
    filled = 0
    block_index = 0
    text_bytes = (text or "").encode("utf-8", errors="ignore")
    while filled < dim:
        h = hashlib.blake2b(
            text_bytes + int(block_index).to_bytes(4, "little", signed=False),
            digest_size=64,
        ).digest()
        vals = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
        vals = (vals / 127.5) - 1.0
        take = min(vals.shape[0], dim - filled)
        out[filled : filled + take] = vals[:take]
        filled += take
        block_index += 1
    norm = float(np.linalg.norm(out))
    if norm > 0.0:
        out /= norm
    return out


def _text_hash_novelty_features(
    *,
    texts: List[str],
    cfg: TrainConfig,
    rnd_map: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, jnp.ndarray], Dict[str, float]]:
    dim = _rnd_input_dim(rnd_map)
    if dim <= 0:
        raise RuntimeError("text_hash novelty fallback could not infer RND input dimension")
    layers = sorted((rnd_map or {}).keys(), key=lambda x: int(str(x).split("_")[-1]))
    features = np.stack([_hash_text_to_feature(text, dim) for text in texts], axis=0)
    token_lengths = [min(int(cfg.novelty_feature_max_length), max(1, int(len(text or "") / 3.5))) for text in texts]
    stats = {
        "count": float(len(texts)),
        "truncated_count": 0.0,
        "token_length_sum": float(sum(token_lengths)),
        "max_input_tokens": float(max(token_lengths) if token_lengths else 0.0),
        "max_length": float(cfg.novelty_feature_max_length),
    }
    print(
        f"[novelty] text_hash fallback features rows={len(texts)} dim={dim} layers={len(layers)}",
        flush=True,
    )
    return {key: jnp.asarray(features) for key in layers}, stats


def _rnd_numpy_forward(params: Dict[str, Dict[str, np.ndarray]], x: np.ndarray):
    w1 = np.asarray(params["fc1"]["kernel"], dtype=np.float32)
    b1 = np.asarray(params["fc1"]["bias"], dtype=np.float32)
    w2 = np.asarray(params["fc2"]["kernel"], dtype=np.float32)
    b2 = np.asarray(params["fc2"]["bias"], dtype=np.float32)
    w3 = np.asarray(params["fc3"]["kernel"], dtype=np.float32)
    b3 = np.asarray(params["fc3"]["bias"], dtype=np.float32)
    z1 = x @ w1 + b1
    a1 = np.maximum(z1, 0.0)
    z2 = a1 @ w2 + b2
    a2 = np.maximum(z2, 0.0)
    y = a2 @ w3 + b3
    return y, (x, z1, a1, z2, a2)


def _score_and_update_rnd_map_numpy(
    rnd_map: Dict[str, Dict[str, Any]],
    features_by_layer: Dict[str, Any],
    *,
    lr: float,
    metric: str,
    clip_value: float = 0.0,
    train: bool = True,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, np.ndarray]]:
    """Host-local RND update for the text-hash fallback path.

    The normal hidden-state path uses the JAX implementation. This path avoids
    TPU compilation/collectives after text-hash feature extraction so the
    fallback cannot block the 64-host training mesh.
    """
    updated: Dict[str, Dict[str, Any]] = {}
    per_layer: Dict[str, np.ndarray] = {}
    lr = float(lr)
    for key, item in rnd_map.items():
        feats = features_by_layer.get(key)
        if feats is None:
            updated[key] = item
            continue
        x = np.asarray(feats, dtype=np.float32)
        target_params = item["target"]
        predictor_params = item["predictor"]
        target, _ = _rnd_numpy_forward(target_params, x)
        pred, cache = _rnd_numpy_forward(predictor_params, x)
        diff = pred - target
        metric_l = str(metric or "mse").lower()
        mse = np.mean(diff * diff, axis=-1)
        if metric_l == "sqrt_mse":
            values = np.sqrt(mse + 1e-8)
        elif metric_l == "l2":
            values = np.sqrt(np.sum(diff * diff, axis=-1) + 1e-8)
        elif metric_l == "l1":
            values = np.mean(np.abs(diff), axis=-1)
        else:
            values = mse
        if float(clip_value or 0.0) > 0.0:
            values = np.minimum(values, float(clip_value))
        per_layer[key] = values.astype(np.float32)

        if train and x.size:
            x0, z1, a1, z2, a2 = cache
            batch = max(1, int(x.shape[0]))
            hidden = max(1, int(pred.shape[-1]))
            dy = (2.0 / float(batch * hidden)) * diff
            w3 = np.asarray(predictor_params["fc3"]["kernel"], dtype=np.float32)
            w2 = np.asarray(predictor_params["fc2"]["kernel"], dtype=np.float32)
            grad_w3 = a2.T @ dy
            grad_b3 = dy.sum(axis=0)
            da2 = dy @ w3.T
            dz2 = da2 * (z2 > 0.0)
            grad_w2 = a1.T @ dz2
            grad_b2 = dz2.sum(axis=0)
            da1 = dz2 @ w2.T
            dz1 = da1 * (z1 > 0.0)
            grad_w1 = x0.T @ dz1
            grad_b1 = dz1.sum(axis=0)
            predictor_params = {
                "fc1": {
                    "kernel": np.asarray(predictor_params["fc1"]["kernel"], dtype=np.float32) - lr * grad_w1,
                    "bias": np.asarray(predictor_params["fc1"]["bias"], dtype=np.float32) - lr * grad_b1,
                },
                "fc2": {
                    "kernel": np.asarray(predictor_params["fc2"]["kernel"], dtype=np.float32) - lr * grad_w2,
                    "bias": np.asarray(predictor_params["fc2"]["bias"], dtype=np.float32) - lr * grad_b2,
                },
                "fc3": {
                    "kernel": np.asarray(predictor_params["fc3"]["kernel"], dtype=np.float32) - lr * grad_w3,
                    "bias": np.asarray(predictor_params["fc3"]["bias"], dtype=np.float32) - lr * grad_b3,
                },
            }
        updated[key] = {
            "target": target_params,
            "predictor": predictor_params,
            "opt_state": item.get("opt_state"),
        }
    return updated, per_layer


def _novelty_uses_multihost_feature_forward() -> bool:
    return not _env_flag("TMX_NOVELTY_TEXT_HASH_FALLBACK", "0")


def _broadcast_novelty_feature_inputs(
    *,
    input_ids: Optional[np.ndarray] = None,
    attention_mask: Optional[np.ndarray] = None,
    stats: Optional[Dict[str, float]] = None,
    is_source_host: bool,
) -> Optional[Tuple[np.ndarray, np.ndarray, Dict[str, float]]]:
    """Broadcast one novelty feature-forward request from source to all hosts.

    Hidden-state novelty uses the sharded training model. On multihost TPU runs,
    every process must enter the same model forward; otherwise the source host
    can hang or terminate while peers wait at the next collective.
    """
    if jax.process_count() == 1:
        if input_ids is None or attention_mask is None:
            return None
        return input_ids.astype(np.int32), attention_mask.astype(np.int32), stats or {}

    debug = _env_flag("TMX_NOVELTY_BROADCAST_DEBUG", "0")
    rank = int(jax.process_index())
    active = int(is_source_host and input_ids is not None and attention_mask is not None)
    rows = int(input_ids.shape[0]) if active else 0
    seq = int(input_ids.shape[1]) if active else 0
    if debug:
        print(
            f"[novelty-bcast] rank={rank} source={int(is_source_host)} "
            f"before_header active={active} rows={rows} seq={seq}",
            flush=True,
        )
    header = {
        "active": np.asarray([active], dtype=np.int32),
        "rows": np.asarray([rows], dtype=np.int32),
        "seq": np.asarray([seq], dtype=np.int32),
        "stats": _novelty_feature_stats_array(stats),
    }
    header = multihost_utils.broadcast_one_to_all(header, is_source=is_source_host)
    active = int(np.asarray(header["active"], dtype=np.int32).reshape(-1)[0])
    if debug:
        dbg_rows = int(np.asarray(header["rows"], dtype=np.int32).reshape(-1)[0])
        dbg_seq = int(np.asarray(header["seq"], dtype=np.int32).reshape(-1)[0])
        print(
            f"[novelty-bcast] rank={rank} source={int(is_source_host)} "
            f"after_header active={active} rows={dbg_rows} seq={dbg_seq}",
            flush=True,
        )
    if active == 0:
        return None

    rows = int(np.asarray(header["rows"], dtype=np.int32).reshape(-1)[0])
    seq = int(np.asarray(header["seq"], dtype=np.int32).reshape(-1)[0])
    if rows <= 0 or seq <= 0:
        return None

    if is_source_host:
        payload = {
            "input_ids": np.asarray(input_ids, dtype=np.int32),
            "attention_mask": np.asarray(attention_mask, dtype=np.int32),
        }
    else:
        payload = {
            "input_ids": np.zeros((rows, seq), dtype=np.int32),
            "attention_mask": np.zeros((rows, seq), dtype=np.int32),
        }
    if debug:
        print(
            f"[novelty-bcast] rank={rank} source={int(is_source_host)} before_payload rows={rows} seq={seq}",
            flush=True,
        )
    payload = multihost_utils.broadcast_one_to_all(payload, is_source=is_source_host)
    if debug:
        print(
            f"[novelty-bcast] rank={rank} source={int(is_source_host)} after_payload rows={rows} seq={seq}",
            flush=True,
        )
    return (
        np.asarray(payload["input_ids"], dtype=np.int32),
        np.asarray(payload["attention_mask"], dtype=np.int32),
        _novelty_feature_stats_dict(np.asarray(header["stats"], dtype=np.float32)),
    )


def _broadcast_novelty_feature_stop(*, is_source_host: bool) -> None:
    if jax.process_count() == 1 or not _novelty_uses_multihost_feature_forward():
        return
    _broadcast_novelty_feature_inputs(is_source_host=is_source_host)


def _global_or_replicated_array_to_numpy(value) -> np.ndarray:
    value = jax.block_until_ready(value)
    try:
        return np.asarray(value, dtype=np.float32)
    except Exception:
        if jax.process_count() > 1:
            gathered = multihost_utils.process_allgather(value, tiled=True)
            return np.asarray(gathered, dtype=np.float32)
        if hasattr(value, "addressable_data"):
            return np.asarray(value.addressable_data(0), dtype=np.float32)
        raise


def _extract_novelty_features_from_encoded(
    *,
    params,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    cfg: TrainConfig,
    feature_step,
) -> Dict[str, jnp.ndarray]:
    layers = tuple(int(x) for x in cfg.novelty_layers)
    batch_size = max(1, int(os.environ.get("TMX_NOVELTY_BATCH_SIZE", "1") or "1"))
    total_rows = int(input_ids.shape[0])
    progress_every = max(batch_size, int(os.environ.get("TMX_NOVELTY_PROGRESS_EVERY_ROWS", "32") or "32"))
    if int(jax.process_index()) == 0:
        print(
            f"[novelty] feature extraction rows={total_rows} seq={int(input_ids.shape[1])} "
            f"batch_size={batch_size}",
            flush=True,
        )
    by_layer: Dict[str, list[np.ndarray]] = {f"layer_{int(layer)}": [] for layer in layers}
    for start in range(0, input_ids.shape[0], batch_size):
        stop = min(input_ids.shape[0], start + batch_size)
        ids = input_ids[start:stop]
        mask = attention_mask[start:stop]
        actual_rows = ids.shape[0]
        if actual_rows < batch_size:
            pad_rows = batch_size - actual_rows
            ids = np.pad(ids, ((0, pad_rows), (0, 0)), constant_values=0)
            mask = np.pad(mask, ((0, pad_rows), (0, 0)), constant_values=0)
        feats = feature_step(params, jnp.asarray(ids), jnp.asarray(mask))
        feats_np = _global_or_replicated_array_to_numpy(feats)[:actual_rows]
        for li, layer in enumerate(layers):
            by_layer[f"layer_{int(layer)}"].append(feats_np[:, li, :])
        if int(jax.process_index()) == 0 and (stop == total_rows or stop % progress_every == 0):
            print(f"[novelty] feature rows {stop}/{total_rows}", flush=True)
    return {k: jnp.asarray(np.concatenate(v, axis=0)) for k, v in by_layer.items() if v}


def _extract_novelty_features(
    *,
    params,
    tokenizer,
    model: Qwen3Model,
    texts: List[str],
    cfg: TrainConfig,
    feature_step,
    is_source_host: bool = True,
) -> Tuple[Dict[str, jnp.ndarray], Dict[str, float]]:
    if not texts:
        return {}, {"count": 0.0, "truncated_count": 0.0, "token_length_sum": 0.0, "max_input_tokens": 0.0, "max_length": float(cfg.novelty_feature_max_length)}
    input_ids, attention_mask, stats = _encode_novelty_texts(tokenizer, texts, cfg)
    payload = _broadcast_novelty_feature_inputs(
        input_ids=input_ids,
        attention_mask=attention_mask,
        stats=stats,
        is_source_host=is_source_host,
    )
    if payload is None:
        return {}, stats
    input_ids, attention_mask, stats = payload
    return _extract_novelty_features_from_encoded(
        params=params,
        input_ids=input_ids,
        attention_mask=attention_mask,
        cfg=cfg,
        feature_step=feature_step,
    ), stats


def _novelty_feature_receiver_loop(
    *,
    params,
    cfg: TrainConfig,
    feature_step,
) -> None:
    if (
        jax.process_count() == 1
        or float(cfg.lambda_novelty or 0.0) == 0.0
        or not _novelty_uses_multihost_feature_forward()
    ):
        return
    while True:
        payload = _broadcast_novelty_feature_inputs(is_source_host=False)
        if payload is None:
            return
        input_ids, attention_mask, _stats = payload
        _extract_novelty_features_from_encoded(
            params=params,
            input_ids=input_ids,
            attention_mask=attention_mask,
            cfg=cfg,
            feature_step=feature_step,
        )


def _score_rollouts(
    *,
    tokenizer,
    examples: List[DapoExample],
    completions,
    cfg: TrainConfig,
    params=None,
    model: Optional[Qwen3Model] = None,
    feature_step=None,
    rnd_map: Optional[Dict[str, Any]] = None,
    norm_state: Optional[Dict[str, Dict[str, float]]] = None,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]], Dict[str, float]]:
    rows: List[Dict[str, Any]] = []
    novelty_texts: List[str] = []
    total_expected = sum(len(comps) for comps in completions or [])
    progress_every = max(1, int(os.environ.get("TMX_SCORE_PROGRESS_EVERY_ROWS", "1") or "1"))
    answer_tail_chars = max(512, int(os.environ.get("TMX_SCORE_ANSWER_TAIL_CHARS", "4096") or "4096"))
    scored_count = 0

    for ex, comps in zip(examples, completions):
        for c in comps:
            completion_text = getattr(c, "text", "") or ""
            scoring_text = completion_text[-answer_tail_chars:] if len(completion_text) > answer_tail_chars else completion_text
            finish_reason = str(getattr(c, "finish_reason", "") or "").lower()
            token_ids = getattr(c, "token_ids", None) or []
            if token_ids:
                completion_token_length = min(int(len(token_ids)), int(cfg.max_completion_len))
            else:
                # Avoid a duplicate full-tokenizer pass here. The train batch
                # builder tokenizes exactly once later; scoring only needs a
                # length for soft-overlong. For length-finished requests, the
                # correct value is the cap. For stopped requests, a conservative
                # character heuristic is enough to avoid blocking the hot path.
                completion_token_length = int(cfg.max_completion_len) if finish_reason == "length" else min(
                    int(cfg.max_completion_len),
                    max(1, int(len(completion_text) / 3.5)),
                )
            clipped = finish_reason == "length" or completion_token_length >= int(cfg.max_completion_len)
            terminated = finish_reason in ("stop", "eos", "eos_token", "stop_sequence") or (
                finish_reason == "" and not clipped
            )
            meta = analyze_completion_text(scoring_text)
            quality = completion_quality_metrics(meta, {"clipped": clipped, "terminated": terminated})
            if (cfg.novelty_text_contract or "full_raw").lower() == "full_raw":
                normalized = {
                    "completion_full_raw": completion_text.strip(),
                    "completion_scored": scoring_text.strip(),
                    "completion_canonical": meta.get("parsed_final_answer"),
                    "used_first_line": False,
                }
                novelty_text = completion_text.strip()
            else:
                normalized = normalize_generated_completion(scoring_text)
                novelty_text = resolve_novelty_text(
                    normalized,
                    fallback_completion=scoring_text,
                    novelty_text_contract=cfg.novelty_text_contract,
                )
            is_corr = is_correct_completion(scoring_text, ex.ground_truth)
            novelty_texts.append(novelty_text)
            rows.append({
                "problem_id": ex.problem_id,
                "prompt_text": ex.prompt_text,
                "completion_text": completion_text,
                "ground_truth": ex.ground_truth,
                "finish_reason": finish_reason,
                "completion_token_length": int(completion_token_length),
                "clipped": bool(clipped),
                "terminated": bool(terminated),
                "valid_answer": bool(meta.get("valid_answer_presentation", False)),
                "is_correct": bool(is_corr),
                "correctness_raw": float(cfg.reward_correct if is_corr else cfg.reward_incorrect),
                "format_quality": float(quality["quality"]),
                "format_penalty": float(quality["penalty"]),
                "format_penalty_breakdown": dict(quality.get("penalties", {})),
                "novelty_text": novelty_text,
                "novelty_text_contract": cfg.novelty_text_contract,
                "completion_canonical": normalized.get("completion_canonical"),
                "used_first_line": bool(normalized.get("used_first_line", False)),
                "r_novel_layers": {},
            })
            scored_count += 1
            if scored_count == total_expected or scored_count % progress_every == 0:
                print(f"[score] parsed rows {scored_count}/{total_expected}", flush=True)

    norm_state = norm_state or {
        "novelty": {"mean": 0.0, "std": 1.0, "initialized": False},
        "correctness": {"mean": 0.0, "std": 1.0, "initialized": False},
    }
    novelty_raw = [0.0 for _ in rows]
    feature_stats = {
        "count": 0.0,
        "truncated_count": 0.0,
        "token_length_sum": 0.0,
        "max_input_tokens": 0.0,
        "max_length": float(cfg.novelty_feature_max_length),
    }
    if float(cfg.lambda_novelty or 0.0) != 0.0:
        if model is None or feature_step is None or rnd_map is None or params is None:
            raise RuntimeError("lambda_novelty > 0 requires model params, novelty feature_step, and RND state.")
        novelty_text_hash_fallback = _env_flag("TMX_NOVELTY_TEXT_HASH_FALLBACK", "0")
        if novelty_text_hash_fallback:
            features_by_layer, feature_stats = _text_hash_novelty_features(
                texts=novelty_texts,
                cfg=cfg,
                rnd_map=rnd_map,
            )
        else:
            features_by_layer, feature_stats = _extract_novelty_features(
                params=params,
                tokenizer=tokenizer,
                model=model,
                texts=novelty_texts,
                cfg=cfg,
                feature_step=feature_step,
            )
        print("[novelty] RND score/update start", flush=True)
        if novelty_text_hash_fallback:
            rnd_map, per_layer = _score_and_update_rnd_map_numpy(
                rnd_map,
                features_by_layer,
                lr=float(cfg.rnd_lr),
                metric=cfg.novelty_metric,
                clip_value=float(cfg.novelty_clip or 0.0),
                train=True,
            )
        else:
            rnd_map, per_layer = score_and_update_rnd_map(
                rnd_map,
                features_by_layer,
                hidden_dim=int(cfg.rnd_hidden),
                lr=float(cfg.rnd_lr),
                metric=cfg.novelty_metric,
                clip_value=float(cfg.novelty_clip or 0.0),
                train=True,
            )
        print("[novelty] RND score/update complete", flush=True)
        if per_layer:
            keys = sorted(per_layer.keys(), key=lambda x: int(x.split("_")[-1]))
            stacked = np.stack([per_layer[k] for k in keys], axis=1)
            if (cfg.novelty_layer_agg or "mean").lower() == "mean":
                novelty_raw = [float(x) for x in stacked.mean(axis=1)]
            else:
                novelty_raw = [float(x) for x in stacked.mean(axis=1)]
            for i, row in enumerate(rows):
                row["r_novel_layers"] = {k: float(per_layer[k][i]) for k in keys}

    correctness_used, corr_center, corr_scale = normalize_values(
        [float(r["correctness_raw"]) for r in rows],
        cfg.correctness_zscore,
        norm_state["correctness"],
    )
    novelty_used, nov_center, nov_scale = normalize_values(
        novelty_raw,
        cfg.novelty_zscore,
        norm_state["novelty"],
    )
    novelty_gate_source = novelty_used
    if (cfg.novelty_zscore or "off").lower() == "off":
        mean_raw = float(sum(novelty_raw) / max(1, len(novelty_raw)))
        novelty_gate_source = [float(v - mean_raw) for v in novelty_raw]

    rewards: List[float] = []
    novelty_reward_values: List[float] = []
    novelty_gate_values: List[float] = []
    termination_penalty_values: List[float] = []
    invalid_answer_penalty_values: List[float] = []
    overlong_penalty_values: List[float] = []
    for i, row in enumerate(rows):
        novelty_reward = float(novelty_used[i])
        novelty_gate = float(novelty_gate_source[i])
        if not bool(row["is_correct"]):
            novelty_reward *= float(max(0.0, cfg.incorrect_novelty_scale))
        final_reward = float(correctness_used[i] + float(cfg.lambda_novelty) * novelty_reward)
        termination_penalty = 0.0
        invalid_answer_penalty = 0.0
        if not bool(row.get("terminated", False)):
            termination_penalty = float(cfg.nontermination_penalty)
        elif not bool(row.get("valid_answer", False)):
            invalid_answer_penalty = float(cfg.invalid_answer_penalty)
        overlong_penalty = 0.0
        expected_len = int(cfg.soft_overlong_expected_len or 0)
        cache_len = int(cfg.soft_overlong_cache_len or 0)
        if expected_len > 0 and cache_len > 0:
            length = int(row.get("completion_token_length", 0) or 0)
            if length > expected_len:
                overlong_penalty = -min(1.0, float(length - expected_len) / float(cache_len))
        final_reward += termination_penalty + invalid_answer_penalty + overlong_penalty
        if not np.isfinite(final_reward):
            final_reward = float(cfg.reward_incorrect)
        row.update({
            "correctness_reward": float(correctness_used[i]),
            "novelty_raw": float(novelty_raw[i]),
            "novelty_used": float(novelty_used[i]),
            "novelty_reward": float(novelty_reward),
            "novelty_gate": float(novelty_gate),
            "termination_penalty": float(termination_penalty),
            "invalid_answer_penalty": float(invalid_answer_penalty),
            "overlong_penalty": float(overlong_penalty),
            "soft_overlong_expected_len": int(expected_len),
            "soft_overlong_cache_len": int(cache_len),
            "blended_reward": float(final_reward),
        })
        rewards.append(float(final_reward))
        novelty_reward_values.append(float(novelty_reward))
        novelty_gate_values.append(float(novelty_gate))
        termination_penalty_values.append(float(termination_penalty))
        invalid_answer_penalty_values.append(float(invalid_answer_penalty))
        overlong_penalty_values.append(float(overlong_penalty))

    def _mean(vals: List[float]) -> float:
        return float(sum(vals) / max(1, len(vals)))

    terminated = [r for r in rows if bool(r.get("terminated", False))]
    summary = {
        "reward/correctness_raw_mean": _mean([float(r["correctness_raw"]) for r in rows]),
        "reward/correctness_used_mean": _mean([float(r["correctness_reward"]) for r in rows]),
        "reward/novelty_raw_mean": _mean(novelty_raw),
        "reward/novelty_used_mean": _mean(novelty_used),
        "reward/novelty_reward_mean": _mean(novelty_reward_values),
        "reward/novelty_gate_mean": _mean(novelty_gate_values),
        "reward/final_mean": _mean(rewards),
        "reward/correctness_center": float(corr_center),
        "reward/correctness_scale": float(corr_scale),
        "reward/novelty_center": float(nov_center),
        "reward/novelty_scale": float(nov_scale),
        "reward/incorrect_novelty_scale": float(cfg.incorrect_novelty_scale),
        "reward/nontermination_penalty_mean": _mean(termination_penalty_values),
        "reward/invalid_answer_penalty_mean": _mean(invalid_answer_penalty_values),
        "reward/overlong_penalty_mean": _mean(overlong_penalty_values),
        "reward/format_quality_mean": _mean([float(r["format_quality"]) for r in rows]),
        "reward/format_penalty_mean": _mean([float(r["format_penalty"]) for r in rows]),
        "reward/novelty_feature_truncated_fraction": float(feature_stats["truncated_count"] / max(1.0, feature_stats["count"])),
        "reward/novelty_feature_mean_input_tokens": float(feature_stats["token_length_sum"] / max(1.0, feature_stats["count"])),
        "reward/novelty_feature_max_input_tokens": float(feature_stats["max_input_tokens"]),
        "reward/novelty_feature_max_length": float(feature_stats["max_length"]),
    }
    if terminated:
        summary.update({
            "reward/terminated_final_mean": _mean([float(r["blended_reward"]) for r in terminated]),
            "reward/terminated_correctness_mean": _mean([float(r["correctness_reward"]) for r in terminated]),
            "reward/terminated_novelty_mean": _mean([float(r["novelty_reward"]) for r in terminated]),
            "reward/terminated_count": float(len(terminated)),
        })
    else:
        summary.update({
            "reward/terminated_final_mean": 0.0,
            "reward/terminated_correctness_mean": 0.0,
            "reward/terminated_novelty_mean": 0.0,
            "reward/terminated_count": 0.0,
        })
    return rows, rnd_map, summary


def _build_rollout_batch(tokenizer, examples: List[DapoExample], completions, cfg: TrainConfig, scored_rows: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Flatten prompts × num_generations into rows.

    Row layout: [prompt_pad_left | completion_pad_right] of length P+T.
    Keeps `prompt_len` for shift alignment in logit computation.
    """
    P = cfg.max_prompt_len
    T = cfg.max_completion_len
    pad_id = int(tokenizer.pad_token_id)

    rows_full_ids = []
    rows_attn = []
    rows_comp_mask = []
    rewards = []
    correct_flags = []
    completion_lengths = []
    terminated_flags = []
    clipped_flags = []
    valid_answer_flags = []
    correctness_rewards = []
    novelty_rewards = []
    novelty_raw_rewards = []
    novelty_used_rewards = []
    format_quality = []
    format_penalties = []
    ground_truths = []
    flat_idx = 0

    for ex, comps in zip(examples, completions):
        prompt_ids = tokenizer(ex.prompt_text, truncation=True, max_length=P, add_special_tokens=False)["input_ids"]
        prompt_ids = prompt_ids[-P:]
        left_pad = P - len(prompt_ids)
        prompt_row = [pad_id] * left_pad + prompt_ids
        prompt_attn = [0] * left_pad + [1] * len(prompt_ids)
        for c in comps:
            comp_ids = tokenizer(c.text, truncation=True, max_length=T, add_special_tokens=False)["input_ids"]
            comp_ids = comp_ids[:T]
            comp_pad = comp_ids + [pad_id] * (T - len(comp_ids))
            rows_full_ids.append(prompt_row + comp_pad)
            rows_attn.append(prompt_attn + [1] * T)
            row = scored_rows[flat_idx] if scored_rows is not None else None
            if row is None:
                row, _, _summary = _score_rollouts(
                    tokenizer=tokenizer,
                    examples=[ex],
                    completions=[[c]],
                    cfg=cfg,
                )
                row = row[0]
            trainable = bool(row.get("used_for_training", True))
            if bool(cfg.mask_truncated_completions) and bool(row.get("clipped", False)):
                trainable = False
            comp_mask = [1] * len(comp_ids) + [0] * (T - len(comp_ids))
            if not trainable:
                comp_mask = [0] * T
            rows_comp_mask.append(comp_mask)
            r = float(row["blended_reward"])
            _correct = bool(row["is_correct"])
            rewards.append(r)
            correct_flags.append(1.0 if _correct else 0.0)
            completion_lengths.append(float(len(comp_ids)))
            terminated_flags.append(1.0 if bool(row["terminated"]) else 0.0)
            clipped_flags.append(1.0 if bool(row["clipped"]) else 0.0)
            valid_answer_flags.append(1.0 if bool(row["valid_answer"]) else 0.0)
            row["completion_token_length"] = int(len(comp_ids))
            row["clipped"] = bool(row.get("clipped", False)) or len(comp_ids) >= int(T)
            correctness_rewards.append(float(row["correctness_reward"]))
            novelty_rewards.append(float(row["novelty_reward"]))
            novelty_raw_rewards.append(float(row["novelty_raw"]))
            novelty_used_rewards.append(float(row["novelty_used"]))
            format_quality.append(float(row["format_quality"]))
            format_penalties.append(float(row["format_penalty"]))
            ground_truths.append(ex.ground_truth)
            flat_idx += 1

    full_input_ids = np.array(rows_full_ids, dtype=np.int32)
    full_attention_mask = np.array(rows_attn, dtype=np.int32)
    completion_mask = np.array(rows_comp_mask, dtype=np.int32)
    rewards_arr = np.array(rewards, dtype=np.float32)
    correct_arr = np.array(correct_flags, dtype=np.float32)
    completion_lengths_arr = np.array(completion_lengths, dtype=np.float32)
    terminated_arr = np.array(terminated_flags, dtype=np.float32)
    clipped_arr = np.array(clipped_flags, dtype=np.float32)
    valid_answer_arr = np.array(valid_answer_flags, dtype=np.float32)
    correctness_rewards_arr = np.array(correctness_rewards, dtype=np.float32)
    novelty_rewards_arr = np.array(novelty_rewards, dtype=np.float32)
    novelty_raw_rewards_arr = np.array(novelty_raw_rewards, dtype=np.float32)
    novelty_used_rewards_arr = np.array(novelty_used_rewards, dtype=np.float32)
    format_quality_arr = np.array(format_quality, dtype=np.float32)
    format_penalties_arr = np.array(format_penalties, dtype=np.float32)
    grouped_rewards = rewards_arr.reshape(-1, int(cfg.grpo_num_generations))
    advantages = grouped_rewards - grouped_rewards.mean(axis=1, keepdims=True)
    if str(cfg.grpo_advantage_normalization or "none").lower() in ("std", "zscore", "group_std", "grpo"):
        advantages = advantages / (grouped_rewards.std(axis=1, keepdims=True, ddof=1) + 1e-4)
    advantages = advantages.reshape(-1).astype(np.float32)
    return {
        "full_input_ids": full_input_ids,
        "full_attention_mask": full_attention_mask,
        "completion_mask": completion_mask,
        "rewards": rewards_arr,
        "is_correct": correct_arr,
        "completion_lengths": completion_lengths_arr,
        "terminated": terminated_arr,
        "clipped": clipped_arr,
        "valid_answer": valid_answer_arr,
        "correctness_rewards": correctness_rewards_arr,
        "novelty_rewards": novelty_rewards_arr,
        "novelty_raw_rewards": novelty_raw_rewards_arr,
        "novelty_used_rewards": novelty_used_rewards_arr,
        "format_quality": format_quality_arr,
        "format_penalties": format_penalties_arr,
        "advantages": advantages,
        "real_row_mask": np.ones((int(advantages.shape[0]),), dtype=np.float32),
        "prompt_len": P,
    }


def _group_rollout_rows(rows: List[Dict[str, Any]], num_generations: int) -> List[List[Dict[str, Any]]]:
    n = max(1, int(num_generations))
    return [rows[i:i + n] for i in range(0, len(rows), n) if len(rows[i:i + n]) == n]


def _group_signal_stats(rows: List[Dict[str, Any]], num_generations: int) -> Dict[str, float]:
    groups = _group_rollout_rows(rows, num_generations)
    if not groups:
        return {
            "group_count": 0.0,
            "informative_group_count": 0.0,
            "effective_group_fraction": 0.0,
            "all_wrong_group_fraction": 0.0,
            "all_correct_group_fraction": 0.0,
        }
    n = float(max(1, int(num_generations)))
    counts = [sum(1 for row in group if bool(row.get("is_correct", False))) for group in groups]
    informative = sum(1 for c in counts if 0 < c < n)
    all_wrong = sum(1 for c in counts if c == 0)
    all_correct = sum(1 for c in counts if c == n)
    denom = float(len(groups))
    return {
        "group_count": denom,
        "informative_group_count": float(informative),
        "effective_group_fraction": float(informative) / denom,
        "all_wrong_group_fraction": float(all_wrong) / denom,
        "all_correct_group_fraction": float(all_correct) / denom,
    }


def _mark_group_rows(
    rows: List[Dict[str, Any]],
    *,
    used_for_training: bool,
    reason: str,
    attempt: int,
    group_index: int,
) -> None:
    for row in rows:
        row["used_for_training"] = bool(used_for_training)
        row["dynamic_sampling_reason"] = str(reason)
        row["dynamic_sampling_attempt"] = int(attempt)
        row["dynamic_sampling_group_index"] = int(group_index)


def _reward_summary_from_rows(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    def _mean(vals: List[float]) -> float:
        return float(sum(vals) / max(1, len(vals)))

    terminated = [r for r in rows if bool(r.get("terminated", False))]
    summary = {
        "reward/correctness_raw_mean": _mean([float(r.get("correctness_raw", 0.0)) for r in rows]),
        "reward/correctness_used_mean": _mean([float(r.get("correctness_reward", 0.0)) for r in rows]),
        "reward/novelty_raw_mean": _mean([float(r.get("novelty_raw", 0.0)) for r in rows]),
        "reward/novelty_used_mean": _mean([float(r.get("novelty_used", 0.0)) for r in rows]),
        "reward/novelty_reward_mean": _mean([float(r.get("novelty_reward", 0.0)) for r in rows]),
        "reward/novelty_gate_mean": _mean([float(r.get("novelty_gate", 0.0)) for r in rows]),
        "reward/final_mean": _mean([float(r.get("blended_reward", 0.0)) for r in rows]),
        "reward/format_quality_mean": _mean([float(r.get("format_quality", 0.0)) for r in rows]),
        "reward/format_penalty_mean": _mean([float(r.get("format_penalty", 0.0)) for r in rows]),
    }
    if terminated:
        summary.update({
            "reward/terminated_final_mean": _mean([float(r.get("blended_reward", 0.0)) for r in terminated]),
            "reward/terminated_correctness_mean": _mean([float(r.get("correctness_reward", 0.0)) for r in terminated]),
            "reward/terminated_novelty_mean": _mean([float(r.get("novelty_reward", 0.0)) for r in terminated]),
            "reward/terminated_count": float(len(terminated)),
        })
    else:
        summary.update({
            "reward/terminated_final_mean": 0.0,
            "reward/terminated_correctness_mean": 0.0,
            "reward/terminated_novelty_mean": 0.0,
            "reward/terminated_count": 0.0,
        })
    return summary


def _dynamic_sample_rollouts(
    *,
    rng: np.random.Generator,
    tokenizer,
    examples: List[DapoExample],
    server_urls: List[str],
    cfg: TrainConfig,
    state,
    model: Qwen3Model,
    feature_step,
    rnd_map: Optional[Dict[str, Any]],
    norm_state: Dict[str, Dict[str, float]],
    prompts_per_step: int,
) -> Tuple[List[DapoExample], List[Any], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, float], Optional[Dict[str, Any]], Dict[str, float], float, Dict[str, float]]:
    selected_prompts: List[DapoExample] = []
    selected_completions: List[Any] = []
    selected_rows: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []
    fallback_groups: List[Tuple[DapoExample, Any, List[Dict[str, Any]], str]] = []
    diversity_acc: Dict[str, float] = {}
    rollout_dt = 0.0
    groups_per_attempt = max(
        1,
        int(os.environ.get("TMX_DYNAMIC_SAMPLING_GROUPS_PER_ATTEMPT", "1") or "1"),
    )
    max_attempts = max(1, int(cfg.dynamic_sampling_max_attempts))
    # If we sample multiple prompt-groups per attempt, the total candidate
    # budget must scale with that fanout. Otherwise attempts=2, groups=2 only
    # tries two total groups, making DAPO dynamic sampling fall back too often.
    max_candidate_groups = max_attempts * max(1, int(prompts_per_step)) * groups_per_attempt
    num_gen = max(1, int(cfg.grpo_num_generations))
    candidate_groups = 0
    kept_groups = 0
    discarded_groups = 0

    attempt = 0
    while attempt < max_attempts and candidate_groups < max_candidate_groups:
        attempt += 1
        need = prompts_per_step - len(selected_prompts)
        if need <= 0:
            break
        batch_need = min(
            need * groups_per_attempt,
            max_candidate_groups - candidate_groups,
            len(examples),
        )
        idxs = rng.choice(len(examples), size=batch_need, replace=False).tolist()
        attempt_prompts = [examples[i] for i in idxs]
        t0 = time.perf_counter()
        print(
            f"[dynamic-sampling] attempt={attempt} need={need} "
            f"candidate_groups={candidate_groups}/{max_candidate_groups} "
            f"batch_groups={len(attempt_prompts)}",
            flush=True,
        )
        attempt_completions = batched_generate(
            [p.prompt_text for p in attempt_prompts],
            server_urls,
            n_per_prompt=cfg.grpo_num_generations,
            max_tokens=cfg.max_completion_len,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            enable_thinking=cfg.enable_thinking,
            model=cfg.model_name,
            concurrency=cfg.vllm_concurrency,
            fanout_per_prompt=cfg.vllm_fanout_per_prompt,
            timeout=_vllm_timeout_seconds(),
            max_retries=_vllm_max_retries(),
        )
        rollout_dt += time.perf_counter() - t0
        if _env_flag("TMX_DISABLE_ROLLOUT_DIVERSITY", "0"):
            print("[dynamic-sampling] skipping online rollout diversity metrics", flush=True)
        else:
            print("[dynamic-sampling] computing online rollout diversity metrics", flush=True)
            for key, value in _rollout_diversity_metrics(attempt_completions).items():
                diversity_acc[key] = diversity_acc.get(key, 0.0) + float(value)
            print("[dynamic-sampling] online rollout diversity metrics complete", flush=True)
        print(f"[dynamic-sampling] scoring {len(attempt_prompts) * num_gen} rollout rows", flush=True)
        scored_rows, rnd_map, _attempt_summary = _score_rollouts(
            tokenizer=tokenizer,
            examples=attempt_prompts,
            completions=attempt_completions,
            cfg=cfg,
            params=state.params,
            model=model,
            feature_step=feature_step,
            rnd_map=rnd_map,
            norm_state=norm_state,
        )
        print("[dynamic-sampling] rollout scoring complete", flush=True)
        grouped_rows = _group_rollout_rows(scored_rows, num_gen)
        for group_index, (ex, comps, group_rows) in enumerate(zip(attempt_prompts, attempt_completions, grouped_rows)):
            candidate_groups += 1
            correct_count = sum(1 for row in group_rows if bool(row.get("is_correct", False)))
            clipped_count = sum(1 for row in group_rows if bool(row.get("clipped", False)))
            reward_values = np.asarray(
                [float(row.get("blended_reward", row.get("correctness_reward", 0.0))) for row in group_rows],
                dtype=np.float32,
            )
            reward_std = float(reward_values.std()) if reward_values.size else 0.0
            accept_reward_variance = (
                _env_flag("TMX_DYNAMIC_SAMPLING_ACCEPT_REWARD_VARIANCE", "1")
                and float(cfg.lambda_novelty) > 0.0
            )
            reward_informative = accept_reward_variance and reward_std > float(
                os.environ.get("TMX_DYNAMIC_SAMPLING_MIN_REWARD_STD", "1e-6") or "1e-6"
            )
            if correct_count == 0:
                reason = "reward_informative_all_wrong" if reward_informative else "all_wrong"
            elif correct_count == num_gen:
                reason = "reward_informative_all_correct" if reward_informative else "all_correct"
            else:
                reason = "informative"
            use_group = (
                (reason == "informative" or reason.startswith("reward_informative_"))
                and len(selected_prompts) < prompts_per_step
            )
            _mark_group_rows(
                group_rows,
                used_for_training=use_group,
                reason=reason,
                attempt=attempt,
                group_index=group_index,
            )
            for row in group_rows:
                row["dynamic_sampling_correct_count"] = int(correct_count)
                row["dynamic_sampling_clipped_count"] = int(clipped_count)
                row["dynamic_sampling_num_generations"] = int(num_gen)
                row["dynamic_sampling_reward_std"] = float(reward_std)
            all_rows.extend(group_rows)
            if use_group:
                selected_prompts.append(ex)
                selected_completions.append(comps)
                selected_rows.extend(group_rows)
                kept_groups += 1
            else:
                fallback_groups.append((ex, comps, group_rows, reason))
                discarded_groups += 1
        print(
            f"[dynamic-sampling] attempt={attempt} complete "
            f"selected={len(selected_prompts)}/{prompts_per_step} "
            f"kept={kept_groups} discarded={discarded_groups} "
            f"fallback_candidates={len(fallback_groups)} "
            f"dt={time.perf_counter() - t0:.1f}s",
            flush=True,
        )

    fallback_used = 0
    while len(selected_prompts) < prompts_per_step and fallback_groups:
        ex, comps, group_rows, reason = fallback_groups.pop(0)
        _mark_group_rows(
            group_rows,
            used_for_training=True,
            reason=f"fallback_{reason}",
            attempt=int(group_rows[0].get("dynamic_sampling_attempt", 0) or 0),
            group_index=int(group_rows[0].get("dynamic_sampling_group_index", 0) or 0),
        )
        for row in group_rows:
            row["dynamic_sampling_fallback"] = True
        selected_prompts.append(ex)
        selected_completions.append(comps)
        selected_rows.extend(group_rows)
        fallback_used += 1

    if len(selected_prompts) != prompts_per_step:
        raise RuntimeError(
            f"dynamic sampling failed to assemble batch: selected={len(selected_prompts)} "
            f"needed={prompts_per_step} attempts={max_attempts}"
        )

    attempts_used = max(1, min(max_attempts, candidate_groups))
    diversity_metrics = {k: v / float(attempts_used) for k, v in diversity_acc.items()}
    dynamic_stats = _group_signal_stats(all_rows, num_gen)
    selected_stats = _group_signal_stats(selected_rows, num_gen)
    dynamic_stats.update({
        "candidate_group_count": float(candidate_groups),
        "kept_group_count": float(kept_groups),
        "discarded_group_count": float(max(0, discarded_groups - fallback_used)),
        "fallback_group_count": float(fallback_used),
        "groups_per_attempt": float(groups_per_attempt),
        "max_candidate_group_count": float(max_candidate_groups),
        "selected_effective_group_fraction": float(selected_stats["effective_group_fraction"]),
    })
    return (
        selected_prompts,
        selected_completions,
        selected_rows,
        all_rows,
        _reward_summary_from_rows(selected_rows),
        rnd_map,
        diversity_metrics,
        rollout_dt,
        dynamic_stats,
    )


def _empty_rollout_batch(cfg: TrainConfig) -> Dict[str, Any]:
    rows = int(cfg.grpo_batch_size) * int(cfg.grpo_grad_accum) * int(cfg.grpo_num_generations)
    total_len = int(cfg.max_prompt_len) + int(cfg.max_completion_len)
    return {
        "full_input_ids": np.zeros((rows, total_len), dtype=np.int32),
        "full_attention_mask": np.zeros((rows, total_len), dtype=np.int32),
        "completion_mask": np.zeros((rows, int(cfg.max_completion_len)), dtype=np.int32),
        "rewards": np.zeros((rows,), dtype=np.float32),
        "is_correct": np.zeros((rows,), dtype=np.float32),
        "completion_lengths": np.zeros((rows,), dtype=np.float32),
        "terminated": np.zeros((rows,), dtype=np.float32),
        "clipped": np.zeros((rows,), dtype=np.float32),
        "valid_answer": np.zeros((rows,), dtype=np.float32),
        "correctness_rewards": np.zeros((rows,), dtype=np.float32),
        "novelty_rewards": np.zeros((rows,), dtype=np.float32),
        "novelty_raw_rewards": np.zeros((rows,), dtype=np.float32),
        "novelty_used_rewards": np.zeros((rows,), dtype=np.float32),
        "format_quality": np.zeros((rows,), dtype=np.float32),
        "format_penalties": np.zeros((rows,), dtype=np.float32),
        "advantages": np.zeros((rows,), dtype=np.float32),
        "real_row_mask": np.zeros((rows,), dtype=np.float32),
        "prompt_len": int(cfg.max_prompt_len),
    }


def _pad_rollout_batch_rows(batch: Dict[str, Any], target_rows: int, pad_id: int) -> Dict[str, Any]:
    current_rows = int(np.asarray(batch["completion_mask"]).shape[0])
    target_rows = int(target_rows)
    if current_rows == target_rows:
        return batch
    if current_rows > target_rows:
        raise ValueError(f"rollout batch has {current_rows} rows, target_rows={target_rows}")
    pad_rows = target_rows - current_rows
    out = dict(batch)
    for key, value in batch.items():
        if key == "prompt_len":
            continue
        arr = np.asarray(value)
        if arr.ndim == 0 or arr.shape[0] != current_rows:
            out[key] = value
            continue
        pad_width = [(0, pad_rows)] + [(0, 0)] * (arr.ndim - 1)
        constant = int(pad_id) if key == "full_input_ids" else 0
        out[key] = np.asarray(np.pad(arr, pad_width, mode="constant", constant_values=constant))
    return out


def _trim_rollout_batch_tokens(batch: Dict[str, Any], prompt_len: int, multiple: int = 256) -> Dict[str, Any]:
    """Trim completion-token columns to the current batch's actual max length.

    The training contract is a 16K *cap*, not a requirement to run every update
    at exactly 16K. JAX attention still pays for padded sequence columns, so on
    long-context TPU runs this is the difference between compiling/executing the
    current batch shape and wasting attention on unused padding.
    """
    if os.environ.get("TMX_TRIM_ROLLOUT_BATCH_TOKENS", "1") in ("0", "false", "False"):
        return batch
    if "completion_mask" not in batch or "full_input_ids" not in batch:
        return batch
    completion_mask = np.asarray(batch["completion_mask"])
    if completion_mask.ndim != 2 or completion_mask.shape[1] <= 0:
        return batch
    used = int(completion_mask.sum(axis=1).max(initial=0))
    min_completion = int(os.environ.get("TMX_MIN_TRAIN_COMPLETION_COLUMNS", "512") or "512")
    used = max(1, used, min_completion)
    multiple = max(1, int(os.environ.get("TMX_TRIM_TOKEN_MULTIPLE", str(multiple)) or str(multiple)))
    used = int(min(completion_mask.shape[1], ((used + multiple - 1) // multiple) * multiple))
    if used >= completion_mask.shape[1]:
        print(
            f"[train] token trim skipped completion_cols={completion_mask.shape[1]} used={used}",
            flush=True,
        )
        return batch
    out = dict(batch)
    p = int(prompt_len)
    out["completion_mask"] = completion_mask[:, :used]
    for key in ("full_input_ids", "full_attention_mask"):
        arr = np.asarray(batch[key])
        out[key] = arr[:, : p + used]
    print(
        f"[train] token trim applied completion_cols={completion_mask.shape[1]}->{used} "
        f"total_len={p + completion_mask.shape[1]}->{p + used}",
        flush=True,
    )
    return out


def _sort_rollout_batch_rows_by_length(batch: Dict[str, Any], descending: bool = True) -> Dict[str, Any]:
    """Sort rollout rows by completion length without changing per-row advantages.

    GRPO grouping is already encoded in the scalar advantage for each row. Sorting
    rows lets host-side grad accumulation put clipped/very-long rows into the
    same few microbatches, so short rows can use shorter static sequence shapes.
    """
    if "completion_mask" not in batch:
        return batch
    completion_mask = np.asarray(batch["completion_mask"])
    if completion_mask.ndim != 2 or completion_mask.shape[0] <= 1:
        return batch
    lengths = completion_mask.sum(axis=1)
    order = np.argsort(lengths, kind="stable")
    if descending:
        order = order[::-1]
    out = dict(batch)
    rows = int(completion_mask.shape[0])
    for key, value in batch.items():
        if key == "prompt_len":
            continue
        arr = np.asarray(value)
        if arr.ndim > 0 and arr.shape[0] == rows:
            out[key] = np.ascontiguousarray(arr[order])
    return out


def _shard_batch_for_hosts(batch: Dict[str, Any], pspecs: Dict[str, P]) -> Dict[str, Any]:
    """Return this JAX process's row shard without broadcasting token arrays.

    Non-token/scalar fields remain replicated because they are small and the
    source host uses them for logging. Token arrays are sent one shard per host.
    """
    if os.environ.get("TMX_REPLICATE_ROLLOUT_BATCH", "0") not in ("0", "false", "False"):
        # On very large FSDP-only meshes (for example v5litepod-256), sharding
        # a tiny rollout batch over the FSDP axis forces us to pad 16 real rows
        # to 256 rows. That makes the update mostly fake compute. In this mode
        # every host receives the same compact rollout batch, while parameters
        # and optimizer state remain FSDP-sharded.
        if jax.process_count() <= 1:
            return batch
        is_source = jax.process_index() == 0
        port = int(os.environ.get("TMX_BATCH_SHARD_HTTP_PORT", "18777"))
        source_host = _batch_shard_source_host()
        if is_source:
            _BATCH_SHARD_PAYLOADS.clear()
            with _BATCH_SHARD_LOCK:
                _BATCH_SHARD_SERVED_RANKS.clear()
            payload = pickle.dumps(batch, protocol=pickle.HIGHEST_PROTOCOL)
            _BATCH_SHARD_PAYLOADS.update({rank: payload for rank in range(int(jax.process_count()))})
            _ensure_batch_shard_server(port)
            _batch_shard_log(
                "replicated source ready "
                f"rows={int(np.asarray(batch['completion_mask']).shape[0])} "
                f"url=http://{source_host}:{port} bytes={len(payload)}"
            )
        else:
            _batch_shard_log(f"replicated non-source will fetch url=http://{source_host}:{port}")
        if is_source:
            payload = _BATCH_SHARD_PAYLOADS[int(jax.process_index())]
            source_wait_s = float(os.environ.get("TMX_BATCH_SHARD_SOURCE_WAIT_SECONDS", "90"))
            start_wait = time.time()
            while time.time() - start_wait < source_wait_s:
                with _BATCH_SHARD_LOCK:
                    served = len(_BATCH_SHARD_SERVED_RANKS - {int(jax.process_index())})
                if served >= int(jax.process_count()) - 1:
                    break
                if served == 0 or served % 8 == 0:
                    _batch_shard_log(
                        f"replicated source waiting served={served}/{int(jax.process_count()) - 1}"
                    )
                time.sleep(1.0)
            with _BATCH_SHARD_LOCK:
                served_final = len(_BATCH_SHARD_SERVED_RANKS - {int(jax.process_index())})
            _batch_shard_log(
                f"replicated source returning local batch bytes={len(payload)} "
                f"served={served_final}/{int(jax.process_count()) - 1}"
            )
            return pickle.loads(payload)

        import urllib.request

        url = f"http://{source_host}:{port}/batch/{int(jax.process_index())}"
        timeout = float(os.environ.get("TMX_BATCH_SHARD_HTTP_TIMEOUT", "120"))
        attempts = int(os.environ.get("TMX_BATCH_SHARD_HTTP_ATTEMPTS", "240"))
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                _batch_shard_log(f"replicated fetch start attempt={attempt}/{attempts} url={url}")
                with urllib.request.urlopen(url, timeout=timeout) as resp:
                    payload = resp.read()
                _batch_shard_log(f"replicated fetch complete bytes={len(payload)}")
                return pickle.loads(payload)
            except Exception as exc:
                last_error = exc
                _batch_shard_log(f"replicated fetch failed attempt={attempt}/{attempts} error={exc!r}")
                time.sleep(min(5.0, 0.25 * attempt))
        raise RuntimeError(f"failed to fetch replicated batch from {url}") from last_error

    if jax.process_count() <= 1 or os.environ.get("TMX_BATCH_SHARD_HTTP", "1") in ("0", "false", "False"):
        _batch_shard_log("using multihost broadcast fallback")
        return multihost_utils.broadcast_one_to_all(batch, is_source=(jax.process_index() == 0))

    is_source = jax.process_index() == 0
    port = int(os.environ.get("TMX_BATCH_SHARD_HTTP_PORT", "18777"))
    process_count = int(jax.process_count())
    source_host = _batch_shard_source_host()
    rows_per_host = None
    if is_source:
        _batch_shard_log("source building shards start")
        rows = int(np.asarray(batch["completion_mask"]).shape[0])
        if rows % process_count != 0:
            raise ValueError(f"cannot shard rollout batch rows={rows} over process_count={process_count}")
        rows_per_host = rows // process_count
        shards: list[Dict[str, Any]] = []
        for rank in range(process_count):
            start = rank * rows_per_host
            stop = start + rows_per_host
            shard: Dict[str, Any] = {"_host_local_shard": True}
            for key, value in batch.items():
                if key == "prompt_len":
                    shard[key] = value
                    continue
                arr = np.asarray(value)
                pspec = pspecs.get(key)
                first_axis = pspec[0] if pspec is not None and len(pspec) > 0 else None
                if pspec is not None and first_axis == FSDP_AXIS and arr.ndim > 0 and arr.shape[0] == rows:
                    shard[key] = np.ascontiguousarray(arr[start:stop])
                else:
                    shard[key] = arr
            shards.append(shard)
        _BATCH_SHARD_PAYLOADS.clear()
        with _BATCH_SHARD_LOCK:
            _BATCH_SHARD_SERVED_RANKS.clear()
        _BATCH_SHARD_PAYLOADS.update({rank: pickle.dumps(shard, protocol=pickle.HIGHEST_PROTOCOL) for rank, shard in enumerate(shards)})
        _ensure_batch_shard_server(port)
        _batch_shard_log(
            "source ready "
            f"rows={rows} rows_per_host={rows_per_host} url=http://{source_host}:{port} "
            f"payloads={len(_BATCH_SHARD_PAYLOADS)} "
            f"rank0_bytes={len(_BATCH_SHARD_PAYLOADS.get(0, b''))}"
        )
    else:
        _batch_shard_log(f"non-source will fetch from source url=http://{source_host}:{port}")
    if is_source:
        payload = _BATCH_SHARD_PAYLOADS[int(jax.process_index())]
        source_wait_s = float(os.environ.get("TMX_BATCH_SHARD_SOURCE_WAIT_SECONDS", "90"))
        start_wait = time.time()
        # Keep Python available to the ThreadingHTTPServer until peer hosts have
        # fetched their shards. Entering the first TPU update immediately can
        # starve the daemon server thread during XLA compile on this 64-host pod.
        while time.time() - start_wait < source_wait_s:
            with _BATCH_SHARD_LOCK:
                served = len(_BATCH_SHARD_SERVED_RANKS - {int(jax.process_index())})
            if served >= process_count - 1:
                break
            if served == 0 or served % 8 == 0:
                _batch_shard_log(f"source waiting for shard fetches served={served}/{process_count - 1}")
            time.sleep(1.0)
        with _BATCH_SHARD_LOCK:
            served_final = len(_BATCH_SHARD_SERVED_RANKS - {int(jax.process_index())})
        _batch_shard_log(f"source returning local shard bytes={len(payload)} served={served_final}/{process_count - 1}")
        return pickle.loads(payload)

    import urllib.request

    url = f"http://{source_host}:{port}/batch/{int(jax.process_index())}"
    timeout = float(os.environ.get("TMX_BATCH_SHARD_HTTP_TIMEOUT", "120"))
    attempts = int(os.environ.get("TMX_BATCH_SHARD_HTTP_ATTEMPTS", "240"))
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            _batch_shard_log(f"fetch start attempt={attempt}/{attempts} url={url}")
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                payload = resp.read()
            _batch_shard_log(f"fetch complete bytes={len(payload)}")
            return pickle.loads(payload)
        except Exception as exc:
            last_error = exc
            _batch_shard_log(f"fetch failed attempt={attempt}/{attempts} error={exc!r}")
            # Non-source ranks can arrive here before the source rank finishes
            # long-context rollout/scoring and starts the HTTP shard server.
            # Treat connection-refused as "source not ready yet", not failure.
            time.sleep(min(5.0, 0.25 * attempt))
    raise RuntimeError(f"failed to fetch batch shard from {url}") from last_error


# --------------------------------------------------------------------------- #
# On-device rescoring + training step
# --------------------------------------------------------------------------- #

def _lm_head_kernel(params, tie_word_embeddings: bool):
    if tie_word_embeddings:
        return params["embed_tokens"]["embedding"].T
    return params["lm_head"]["kernel"]


def _rescore_old_logps(
    state,
    full_input_ids,
    full_attention_mask,
    prompt_len: int,
    tie_word_embeddings: bool,
    batch_chunk: int = 1,
    token_chunk: int = 256,
):
    """Rescore old logps under the current policy, chunked along batch to bound HBM."""
    apply_fn = state.apply_fn
    @jax.jit
    def _step(params, ids, mask):
        hidden = apply_fn({"params": params}, ids, mask, return_hidden=True)
        shift_hidden = hidden[:, prompt_len - 1 : -1, :]
        completion_ids = ids[:, prompt_len:]
        return chunked_per_token_logps_from_hidden(
            shift_hidden,
            completion_ids,
            _lm_head_kernel(params, tie_word_embeddings),
            chunk_tokens=token_chunk,
        )

    B = full_input_ids.shape[0]
    pieces = []
    for s in range(0, B, batch_chunk):
        e = min(s + batch_chunk, B)
        piece = _step(state.params, full_input_ids[s:e], full_attention_mask[s:e])
        pieces.append(piece)
    return jnp.concatenate(pieces, axis=0)


def make_train_step(
    grpo_cfg: GrpoConfig,
    prompt_len: int,
    tie_word_embeddings: bool,
    logprob_chunk_tokens: int,
    rows_per_microbatch: int,
    grad_accum_steps: int,
    use_inline_old_logprobs: bool,
    track_external_old_gap: bool,
    row_vector_chunk: int,
    use_ref_kl: bool,
):
    """Return a full optimizer step with prompt-group grad accumulation inside JIT."""

    def train_step(state, batch, ref_params):
        full_input_ids = batch["full_input_ids"]
        full_attention_mask = batch["full_attention_mask"]
        completion_mask = batch["completion_mask"]
        old_per_token_logps = batch["old_per_token_logps"]
        advantages = batch["advantages"]
        ref_per_token_logps = batch["ref_per_token_logps"] if use_ref_kl else None

        def micro_grad(params, micro_idx):
            row_start = micro_idx * rows_per_microbatch
            micro_full_input_ids = jax.lax.dynamic_slice_in_dim(
                full_input_ids, row_start, rows_per_microbatch, axis=0
            )
            micro_full_attention_mask = jax.lax.dynamic_slice_in_dim(
                full_attention_mask, row_start, rows_per_microbatch, axis=0
            )
            micro_completion_mask = jax.lax.dynamic_slice_in_dim(
                completion_mask, row_start, rows_per_microbatch, axis=0
            )
            micro_old_per_token_logps = jax.lax.dynamic_slice_in_dim(
                old_per_token_logps, row_start, rows_per_microbatch, axis=0
            )
            micro_advantages = jax.lax.dynamic_slice_in_dim(
                advantages, row_start, rows_per_microbatch, axis=0
            )
            if use_ref_kl:
                micro_ref_per_token_logps = jax.lax.dynamic_slice_in_dim(
                    ref_per_token_logps, row_start, rows_per_microbatch, axis=0
                )
            else:
                micro_ref_per_token_logps = None
            B = micro_full_input_ids.shape[0]

            def vector_chunk_loss(params, chunk_start, chunk_rows):
                chunk_full_input_ids = jax.lax.dynamic_slice_in_dim(
                    micro_full_input_ids, chunk_start, chunk_rows, axis=0
                )
                chunk_full_attention_mask = jax.lax.dynamic_slice_in_dim(
                    micro_full_attention_mask, chunk_start, chunk_rows, axis=0
                )
                chunk_completion_mask = jax.lax.dynamic_slice_in_dim(
                    micro_completion_mask, chunk_start, chunk_rows, axis=0
                )
                chunk_old_per_token_logps = jax.lax.dynamic_slice_in_dim(
                    micro_old_per_token_logps, chunk_start, chunk_rows, axis=0
                )
                chunk_advantages = jax.lax.dynamic_slice_in_dim(
                    micro_advantages, chunk_start, chunk_rows, axis=0
                )
                if use_ref_kl:
                    chunk_ref_per_token_logps = jax.lax.dynamic_slice_in_dim(
                        micro_ref_per_token_logps, chunk_start, chunk_rows, axis=0
                    )
                else:
                    chunk_ref_per_token_logps = None
                hidden = state.apply_fn(
                    {"params": params},
                    chunk_full_input_ids,
                    chunk_full_attention_mask,
                    return_hidden=True,
                )
                shift_hidden = hidden[:, prompt_len - 1 : -1, :]
                completion_ids = chunk_full_input_ids[:, prompt_len:]
                new_lp = chunked_per_token_logps_from_hidden(
                    shift_hidden,
                    completion_ids,
                    _lm_head_kernel(params, tie_word_embeddings),
                    chunk_tokens=logprob_chunk_tokens,
                )
                olp = chunk_old_per_token_logps
                if track_external_old_gap:
                    rescore_gap = jnp.abs(new_lp - olp) * chunk_completion_mask.astype(jnp.float32)
                if use_inline_old_logprobs:
                    # Same semantics as row-scan mode: one optimizer step per
                    # rollout batch, so old policy is the pre-update current
                    # policy. stop_gradient keeps the REINFORCE gradient while
                    # making ratio/clip metrics exact for the one-step update.
                    olp = jax.lax.stop_gradient(new_lp)
                loss, metrics = dr_grpo_loss_from_logps(
                    new_lp,
                    chunk_completion_mask,
                    olp,
                    chunk_advantages,
                    grpo_cfg,
                    ref_per_token_logps=chunk_ref_per_token_logps,
                )
                if track_external_old_gap:
                    denom = jnp.maximum(chunk_completion_mask.astype(jnp.float32).sum(), 1.0)
                    metrics["rescore_old_logprob_abs_mean"] = rescore_gap.sum() / denom
                    metrics["rescore_old_logprob_abs_max"] = rescore_gap.max()
                return loss, metrics

            vector_rows = int(row_vector_chunk)
            if vector_rows > 1 and B % vector_rows == 0:
                grad_fn = jax.value_and_grad(
                    lambda p, chunk_idx: vector_chunk_loss(p, chunk_idx * vector_rows, vector_rows),
                    has_aux=True,
                )
                (_loss0, metrics0), grads0 = grad_fn(params, 0)
                metrics0["loss"] = _loss0
                num_chunks = B // vector_rows
                if num_chunks == 1:
                    return grads0, metrics0

                def chunk_body(carry, chunk_idx):
                    acc_g, acc_m = carry
                    (_loss, metrics), grads = grad_fn(params, chunk_idx)
                    metrics["loss"] = _loss
                    acc_g = jax.tree_util.tree_map(lambda a, b: a + b, acc_g, grads)
                    acc_m = jax.tree_util.tree_map(lambda a, b: a + b, acc_m, metrics)
                    return (acc_g, acc_m), None

                (sum_g, sum_m), _ = jax.lax.scan(
                    chunk_body,
                    (grads0, metrics0),
                    jnp.arange(1, num_chunks),
                )
                inv_chunks = 1.0 / jnp.asarray(num_chunks, dtype=jnp.float32)
                avg_g = jax.tree_util.tree_map(lambda x: x * inv_chunks, sum_g)
                avg_m = jax.tree_util.tree_map(lambda x: x * inv_chunks, sum_m)
                return avg_g, avg_m

            def row_loss(params, i):
                ids = jax.lax.dynamic_slice_in_dim(micro_full_input_ids, i, 1, axis=0)
                amk = jax.lax.dynamic_slice_in_dim(micro_full_attention_mask, i, 1, axis=0)
                cmk = jax.lax.dynamic_slice_in_dim(micro_completion_mask, i, 1, axis=0)
                olp = jax.lax.dynamic_slice_in_dim(micro_old_per_token_logps, i, 1, axis=0)
                adv = jax.lax.dynamic_slice_in_dim(micro_advantages, i, 1, axis=0)
                if use_ref_kl:
                    ref_lp = jax.lax.dynamic_slice_in_dim(micro_ref_per_token_logps, i, 1, axis=0)
                else:
                    ref_lp = None
                hidden = state.apply_fn({"params": params}, ids, amk, return_hidden=True)
                shift_hidden = hidden[:, prompt_len - 1 : -1, :]
                completion_ids = ids[:, prompt_len:]
                new_lp = chunked_per_token_logps_from_hidden(
                    shift_hidden,
                    completion_ids,
                    _lm_head_kernel(params, tie_word_embeddings),
                    chunk_tokens=logprob_chunk_tokens,
                )
                if track_external_old_gap:
                    rescore_gap = jnp.abs(new_lp - olp) * cmk.astype(jnp.float32)
                if use_inline_old_logprobs:
                    # We take exactly one optimizer step per rollout batch. In that
                    # regime the PPO "old" policy is the same current-policy
                    # forward pass before the update, so anchoring old logprobs to
                    # stop_gradient(new_lp) is exact and avoids TPU/XLA boundary
                    # skew between separately materialized logprob tensors.
                    olp = jax.lax.stop_gradient(new_lp)
                loss, metrics = dr_grpo_loss_from_logps(
                    new_lp, cmk, olp, adv, grpo_cfg, ref_per_token_logps=ref_lp
                )
                if track_external_old_gap:
                    denom = jnp.maximum(cmk.astype(jnp.float32).sum(), 1.0)
                    metrics["rescore_old_logprob_abs_mean"] = rescore_gap.sum() / denom
                    metrics["rescore_old_logprob_abs_max"] = rescore_gap.max()
                return loss, metrics

            grad_fn = jax.value_and_grad(row_loss, has_aux=True)

            def body(carry, i):
                acc_g, acc_l, acc_m_sum, acc_count = carry
                (l, m), g = grad_fn(params, i)
                acc_g = jax.tree_util.tree_map(lambda a, b: a + b, acc_g, g)
                acc_l = acc_l + l
                acc_m_sum = jax.tree_util.tree_map(lambda a, b: a + b, acc_m_sum, m)
                return (acc_g, acc_l, acc_m_sum, acc_count + 1), None

            # Seed with row 0, then scan rows 1..B-1. Using row 0 only for
            # shape inference and then scanning from 0 computes one extra
            # full forward/backward pass per microbatch.
            (_l0, _m0), _g0 = grad_fn(params, 0)
            (acc_g, acc_l, acc_m, _count), _ = jax.lax.scan(
                body,
                (_g0, _l0, _m0, jnp.asarray(1, dtype=jnp.int32)),
                jnp.arange(1, B),
            )
            inv = 1.0 / jnp.asarray(B, dtype=jnp.float32)
            avg_g = jax.tree_util.tree_map(lambda a: a * inv, acc_g)
            avg_l = acc_l * inv
            avg_m = jax.tree_util.tree_map(lambda a: a * inv, acc_m)
            avg_m["loss"] = avg_l
            return avg_g, avg_m

        first_g, first_m = micro_grad(state.params, 0)
        if grad_accum_steps == 1:
            avg_g = first_g
            avg_m = first_m
        else:
            def accum_body(carry, micro_idx):
                acc_g, acc_m = carry
                g, m = micro_grad(state.params, micro_idx)
                acc_g = jax.tree_util.tree_map(lambda a, b: a + b, acc_g, g)
                acc_m = jax.tree_util.tree_map(lambda a, b: a + b, acc_m, m)
                return (acc_g, acc_m), None

            (sum_g, sum_m), _ = jax.lax.scan(
                accum_body,
                (first_g, first_m),
                jnp.arange(1, grad_accum_steps),
            )
            inv_acc = 1.0 / jnp.asarray(grad_accum_steps, dtype=jnp.float32)
            avg_g = jax.tree_util.tree_map(lambda x: x * inv_acc, sum_g)
            avg_m = jax.tree_util.tree_map(lambda x: x * inv_acc, sum_m)

        grad_norm = optax.global_norm(avg_g)
        new_state = state.apply_gradients(grads=avg_g)
        return new_state, avg_m, grad_norm

    return train_step


def make_micro_grad_step(
    apply_fn,
    grpo_cfg: GrpoConfig,
    prompt_len: int,
    tie_word_embeddings: bool,
    logprob_chunk_tokens: int,
    use_inline_old_logprobs: bool,
    track_external_old_gap: bool,
    row_vector_chunk: int,
    use_ref_kl: bool,
):
    """Return gradients for one rollout microbatch without applying them.

    This supports host-side grad accumulation with per-microbatch sequence
    trimming. It preserves the same averaged-gradient semantics as
    ``make_train_step(..., grad_accum_steps=N)`` but avoids one huge static
    16K-wide batch shape when only a small subset of rows are clipped.
    """

    def micro_grad_step(params, batch, ref_params):
        full_input_ids = batch["full_input_ids"]
        full_attention_mask = batch["full_attention_mask"]
        completion_mask = batch["completion_mask"]
        old_per_token_logps = batch["old_per_token_logps"]
        advantages = batch["advantages"]
        ref_per_token_logps = batch["ref_per_token_logps"] if use_ref_kl else None
        B = full_input_ids.shape[0]

        def vector_chunk_loss(params, chunk_start, chunk_rows):
            chunk_full_input_ids = jax.lax.dynamic_slice_in_dim(
                full_input_ids, chunk_start, chunk_rows, axis=0
            )
            chunk_full_attention_mask = jax.lax.dynamic_slice_in_dim(
                full_attention_mask, chunk_start, chunk_rows, axis=0
            )
            chunk_completion_mask = jax.lax.dynamic_slice_in_dim(
                completion_mask, chunk_start, chunk_rows, axis=0
            )
            chunk_old_per_token_logps = jax.lax.dynamic_slice_in_dim(
                old_per_token_logps, chunk_start, chunk_rows, axis=0
            )
            chunk_advantages = jax.lax.dynamic_slice_in_dim(
                advantages, chunk_start, chunk_rows, axis=0
            )
            if use_ref_kl:
                chunk_ref_per_token_logps = jax.lax.dynamic_slice_in_dim(
                    ref_per_token_logps, chunk_start, chunk_rows, axis=0
                )
            else:
                chunk_ref_per_token_logps = None
            hidden = apply_fn(
                {"params": params},
                chunk_full_input_ids,
                chunk_full_attention_mask,
                return_hidden=True,
            )
            shift_hidden = hidden[:, prompt_len - 1 : -1, :]
            completion_ids = chunk_full_input_ids[:, prompt_len:]
            new_lp = chunked_per_token_logps_from_hidden(
                shift_hidden,
                completion_ids,
                _lm_head_kernel(params, tie_word_embeddings),
                chunk_tokens=logprob_chunk_tokens,
            )
            olp = chunk_old_per_token_logps
            if track_external_old_gap:
                rescore_gap = jnp.abs(new_lp - olp) * chunk_completion_mask.astype(jnp.float32)
            if use_inline_old_logprobs:
                olp = jax.lax.stop_gradient(new_lp)
            loss, metrics = dr_grpo_loss_from_logps(
                new_lp,
                chunk_completion_mask,
                olp,
                chunk_advantages,
                grpo_cfg,
                ref_per_token_logps=chunk_ref_per_token_logps,
            )
            if track_external_old_gap:
                denom = jnp.maximum(chunk_completion_mask.astype(jnp.float32).sum(), 1.0)
                metrics["rescore_old_logprob_abs_mean"] = rescore_gap.sum() / denom
                metrics["rescore_old_logprob_abs_max"] = rescore_gap.max()
            return loss, metrics

        vector_rows = int(row_vector_chunk)
        if vector_rows > 1 and B % vector_rows == 0:
            grad_fn = jax.value_and_grad(
                lambda p, chunk_idx: vector_chunk_loss(p, chunk_idx * vector_rows, vector_rows),
                has_aux=True,
            )
            (_loss0, metrics0), grads0 = grad_fn(params, 0)
            metrics0["loss"] = _loss0
            num_chunks = B // vector_rows
            if num_chunks == 1:
                return grads0, metrics0

            def chunk_body(carry, chunk_idx):
                acc_g, acc_m = carry
                (_loss, metrics), grads = grad_fn(params, chunk_idx)
                metrics["loss"] = _loss
                acc_g = jax.tree_util.tree_map(lambda a, b: a + b, acc_g, grads)
                acc_m = jax.tree_util.tree_map(lambda a, b: a + b, acc_m, metrics)
                return (acc_g, acc_m), None

            (sum_g, sum_m), _ = jax.lax.scan(
                chunk_body,
                (grads0, metrics0),
                jnp.arange(1, num_chunks),
            )
            inv_chunks = 1.0 / jnp.asarray(num_chunks, dtype=jnp.float32)
            avg_g = jax.tree_util.tree_map(lambda x: x * inv_chunks, sum_g)
            avg_m = jax.tree_util.tree_map(lambda x: x * inv_chunks, sum_m)
            return avg_g, avg_m

        def row_loss(params, i):
            ids = jax.lax.dynamic_slice_in_dim(full_input_ids, i, 1, axis=0)
            amk = jax.lax.dynamic_slice_in_dim(full_attention_mask, i, 1, axis=0)
            cmk = jax.lax.dynamic_slice_in_dim(completion_mask, i, 1, axis=0)
            olp = jax.lax.dynamic_slice_in_dim(old_per_token_logps, i, 1, axis=0)
            adv = jax.lax.dynamic_slice_in_dim(advantages, i, 1, axis=0)
            if use_ref_kl:
                ref_lp = jax.lax.dynamic_slice_in_dim(ref_per_token_logps, i, 1, axis=0)
            else:
                ref_lp = None
            hidden = apply_fn({"params": params}, ids, amk, return_hidden=True)
            shift_hidden = hidden[:, prompt_len - 1 : -1, :]
            completion_ids = ids[:, prompt_len:]
            new_lp = chunked_per_token_logps_from_hidden(
                shift_hidden,
                completion_ids,
                _lm_head_kernel(params, tie_word_embeddings),
                chunk_tokens=logprob_chunk_tokens,
            )
            if track_external_old_gap:
                rescore_gap = jnp.abs(new_lp - olp) * cmk.astype(jnp.float32)
            if use_inline_old_logprobs:
                olp = jax.lax.stop_gradient(new_lp)
            loss, metrics = dr_grpo_loss_from_logps(
                new_lp, cmk, olp, adv, grpo_cfg, ref_per_token_logps=ref_lp
            )
            if track_external_old_gap:
                denom = jnp.maximum(cmk.astype(jnp.float32).sum(), 1.0)
                metrics["rescore_old_logprob_abs_mean"] = rescore_gap.sum() / denom
                metrics["rescore_old_logprob_abs_max"] = rescore_gap.max()
            return loss, metrics

        grad_fn = jax.value_and_grad(row_loss, has_aux=True)

        def body(carry, i):
            acc_g, acc_l, acc_m_sum, acc_count = carry
            (loss, metrics), grads = grad_fn(params, i)
            acc_g = jax.tree_util.tree_map(lambda a, b: a + b, acc_g, grads)
            acc_l = acc_l + loss
            acc_m_sum = jax.tree_util.tree_map(lambda a, b: a + b, acc_m_sum, metrics)
            return (acc_g, acc_l, acc_m_sum, acc_count + 1), None

        (_loss0, _metrics0), _grads0 = grad_fn(params, 0)
        (acc_g, acc_l, acc_m, _count), _ = jax.lax.scan(
            body,
            (_grads0, _loss0, _metrics0, jnp.asarray(1, dtype=jnp.int32)),
            jnp.arange(1, B),
        )
        inv = 1.0 / jnp.asarray(B, dtype=jnp.float32)
        avg_g = jax.tree_util.tree_map(lambda a: a * inv, acc_g)
        avg_l = acc_l * inv
        avg_m = jax.tree_util.tree_map(lambda a: a * inv, acc_m)
        avg_m["loss"] = avg_l
        return avg_g, avg_m

    return micro_grad_step


def make_apply_grads_step():
    def apply_grads_step(state, grads):
        grad_norm = optax.global_norm(grads)
        return state.apply_gradients(grads=grads), grad_norm

    return apply_grads_step


def make_trainpath_logprob_step(
    prompt_len: int,
    tie_word_embeddings: bool,
    logprob_chunk_tokens: int,
    rows_per_microbatch: int,
    grad_accum_steps: int,
):
    """Return the exact no-grad logprob path used by the train step."""

    def trainpath_logprob_step(state, batch):
        full_input_ids = batch["full_input_ids"]
        full_attention_mask = batch["full_attention_mask"]
        completion_len = batch["completion_mask"].shape[1]
        total_rows = full_input_ids.shape[0]

        def micro_body(acc_lp, micro_idx):
            row_start = micro_idx * rows_per_microbatch
            micro_full_input_ids = jax.lax.dynamic_slice_in_dim(
                full_input_ids, row_start, rows_per_microbatch, axis=0
            )
            micro_full_attention_mask = jax.lax.dynamic_slice_in_dim(
                full_attention_mask, row_start, rows_per_microbatch, axis=0
            )

            def row_body(micro_lp, i):
                ids = jax.lax.dynamic_slice_in_dim(micro_full_input_ids, i, 1, axis=0)
                amk = jax.lax.dynamic_slice_in_dim(micro_full_attention_mask, i, 1, axis=0)
                hidden = state.apply_fn({"params": state.params}, ids, amk, return_hidden=True)
                shift_hidden = hidden[:, prompt_len - 1 : -1, :]
                completion_ids = ids[:, prompt_len:]
                new_lp = chunked_per_token_logps_from_hidden(
                    shift_hidden,
                    completion_ids,
                    _lm_head_kernel(state.params, tie_word_embeddings),
                    chunk_tokens=logprob_chunk_tokens,
                )
                micro_lp = jax.lax.dynamic_update_slice(micro_lp, new_lp, (i, 0))
                return micro_lp, None

            init_micro_lp = jnp.zeros((rows_per_microbatch, completion_len), dtype=jnp.float32)
            micro_lp, _ = jax.lax.scan(row_body, init_micro_lp, jnp.arange(rows_per_microbatch))
            acc_lp = jax.lax.dynamic_update_slice(acc_lp, micro_lp, (row_start, 0))
            return acc_lp, None

        init_lp = jnp.zeros((total_rows, completion_len), dtype=jnp.float32)
        all_lp, _ = jax.lax.scan(micro_body, init_lp, jnp.arange(grad_accum_steps))
        return all_lp

    return trainpath_logprob_step


def make_rescore_step(
    prompt_len: int,
    tie_word_embeddings: bool,
    logprob_chunk_tokens: int,
    rows_per_microbatch: int,
    grad_accum_steps: int,
):
    """Return a no-grad rescoring pass on the same sharded row-scan path as training."""

    def rescore_step(state, batch):
        full_input_ids = batch["full_input_ids"]
        full_attention_mask = batch["full_attention_mask"]
        completion_len = batch["completion_mask"].shape[1]
        total_rows = full_input_ids.shape[0]

        def micro_body(acc_lp, micro_idx):
            row_start = micro_idx * rows_per_microbatch
            micro_full_input_ids = jax.lax.dynamic_slice_in_dim(
                full_input_ids, row_start, rows_per_microbatch, axis=0
            )
            micro_full_attention_mask = jax.lax.dynamic_slice_in_dim(
                full_attention_mask, row_start, rows_per_microbatch, axis=0
            )

            def row_body(micro_lp, i):
                ids = jax.lax.dynamic_slice_in_dim(micro_full_input_ids, i, 1, axis=0)
                amk = jax.lax.dynamic_slice_in_dim(micro_full_attention_mask, i, 1, axis=0)
                hidden = state.apply_fn({"params": state.params}, ids, amk, return_hidden=True)
                shift_hidden = hidden[:, prompt_len - 1 : -1, :]
                completion_ids = ids[:, prompt_len:]
                new_lp = chunked_per_token_logps_from_hidden(
                    shift_hidden,
                    completion_ids,
                    _lm_head_kernel(state.params, tie_word_embeddings),
                    chunk_tokens=logprob_chunk_tokens,
                )
                micro_lp = jax.lax.dynamic_update_slice(micro_lp, new_lp, (i, 0))
                return micro_lp, None

            init_micro_lp = jnp.zeros((rows_per_microbatch, completion_len), dtype=jnp.float32)
            micro_lp, _ = jax.lax.scan(row_body, init_micro_lp, jnp.arange(rows_per_microbatch))
            acc_lp = jax.lax.dynamic_update_slice(acc_lp, micro_lp, (row_start, 0))
            return acc_lp, None

        init_lp = jnp.zeros((total_rows, completion_len), dtype=jnp.float32)
        all_lp, _ = jax.lax.scan(micro_body, init_lp, jnp.arange(grad_accum_steps))
        return all_lp

    return rescore_step


def _slice_batch(batch: Dict[str, Any], start: int, end: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        if key == "prompt_len":
            out[key] = value
            continue
        out[key] = value[start:end]
    return out


def _select_rollout_batch_rows(batch: Dict[str, Any], indices: np.ndarray) -> Dict[str, Any]:
    rows = int(np.asarray(batch["completion_mask"]).shape[0])
    idx = np.asarray(indices, dtype=np.int64)
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        if key == "prompt_len":
            out[key] = value
            continue
        arr = np.asarray(value)
        if arr.ndim > 0 and arr.shape[0] == rows:
            out[key] = np.ascontiguousarray(arr[idx])
        else:
            out[key] = value
    return out


def _zero_train_metrics(advantages: np.ndarray | None = None) -> Dict[str, float]:
    adv = np.asarray(advantages if advantages is not None else np.zeros((1,), dtype=np.float32), dtype=np.float32)
    if adv.size == 0:
        adv = np.zeros((1,), dtype=np.float32)
    return {
        "loss": 0.0,
        "kl_term": 0.0,
        "ref_kl": 0.0,
        "clip_fraction": 0.0,
        "up_clip_fraction": 0.0,
        "up_clipped_probability_mean": 0.0,
        "approx_kl": 0.0,
        "mean_ratio": 1.0,
        "old_new_logprob_abs_mean": 0.0,
        "old_new_logprob_abs_max": 0.0,
        "selected_token_logprob_mean": 0.0,
        "selected_token_probability_mean": 0.0,
        "adv_mean": float(adv.mean()),
        "adv_std": float(adv.std()),
    }


def _to_global_batch(batch: Dict[str, Any], mesh, pspecs: Dict[str, P]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    multi_host = jax.process_count() > 1
    host_local_shard = bool(batch.get("_host_local_shard", False))
    for key, value in batch.items():
        if key not in pspecs:
            out[key] = value
            continue
        pspec = pspecs[key]
        if not multi_host:
            out[key] = jax.device_put(value, named(mesh, pspec))
            continue
        first_axis = pspec[0] if len(pspec) > 0 else None
        if first_axis == FSDP_AXIS:
            if host_local_shard:
                local_value = np.asarray(value)
            else:
                host_rows = value.shape[0] // jax.process_count()
                start = jax.process_index() * host_rows
                stop = start + host_rows
                local_value = np.asarray(value[start:stop])
        else:
            local_value = np.asarray(value)
        out[key] = multihost_utils.host_local_array_to_global_array(local_value, mesh, pspec)
    return out


def _inter_generation_distinct_n(generations: List[str], n: int) -> float:
    if not generations:
        return 0.0
    if len(generations) <= 1:
        return 1.0

    def ngrams(text: str) -> list[tuple[str, ...]]:
        toks = text.split()
        return [tuple(toks[i:i + n]) for i in range(max(0, len(toks) - n + 1))]

    all_lists = [ngrams(g) for g in generations]
    all_sets = [set(xs) for xs in all_lists]
    scores = []
    for i, xs in enumerate(all_lists):
        if not xs:
            scores.append(0.0)
            continue
        others = set()
        for j, s in enumerate(all_sets):
            if i != j:
                others.update(s)
        scores.append(len(all_sets[i] - others) / max(1, len(xs)))
    return float(sum(scores) / max(1, len(scores)))


def _rollout_diversity_metrics(completions) -> Dict[str, float]:
    """Torch/XLA-compatible DARLING dist-3/dist-4 rollout metrics."""
    dist3 = []
    dist4 = []
    trunc_dist3 = []
    trunc_dist4 = []
    for comps in completions or []:
        texts = [getattr(c, "text", "") or "" for c in comps if getattr(c, "text", "")]
        if len(texts) >= 2:
            dist3.append(_inter_generation_distinct_n(texts, 3))
            dist4.append(_inter_generation_distinct_n(texts, 4))
        trunc_texts = []
        for c in comps:
            reason = str(getattr(c, "finish_reason", "") or "").lower()
            if reason != "length" and getattr(c, "text", ""):
                trunc_texts.append(c.text)
        if len(trunc_texts) >= 2:
            trunc_dist3.append(_inter_generation_distinct_n(trunc_texts, 3))
            trunc_dist4.append(_inter_generation_distinct_n(trunc_texts, 4))

    def mean(xs: list[float]) -> float:
        return float(sum(xs) / max(1, len(xs))) if xs else 0.0

    return {
        "diversity/dist_3": mean(dist3),
        "diversity/dist_4": mean(dist4),
        "diversity/num_groups": float(len(dist3)),
        "diversity/truncated_dist_3": mean(trunc_dist3),
        "diversity/truncated_dist_4": mean(trunc_dist4),
        "diversity/truncated_num_groups": float(len(trunc_dist3)),
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def _trajectory_record(
    problem_id,
    prompt_text,
    completion_text,
    gt,
    correctness_reward,
    novelty_reward,
    blended_reward,
    is_correct_flag,
    explorer_step,
    *,
    finish_reason: str = "",
    completion_token_length: int = 0,
    clipped: bool = False,
    terminated: bool = False,
    valid_answer: bool = False,
    extra: Optional[Dict[str, Any]] = None,
):
    row = {
        "problem_id": str(problem_id),
        "prompt_text": prompt_text,
        "completion_text": completion_text,
        "ground_truth": gt,
        "correctness_reward": float(correctness_reward),
        "novelty_reward": float(novelty_reward),
        "blended_reward": float(blended_reward),
        "reward": float(blended_reward),
        "r_correct": float(correctness_reward),
        "r_novel": float(novelty_reward),
        "is_correct": bool(is_correct_flag),
        "explorer_step": int(explorer_step),
        "step": int(explorer_step),
        "finish_reason": str(finish_reason or ""),
        "completion_token_length": int(completion_token_length),
        "clipped": bool(clipped),
        "terminated": bool(terminated),
        "valid_answer": bool(valid_answer),
    }
    if extra:
        row.update(extra)
    return row


def _append_rollout_records(
    output_dir: str,
    records: List[Dict[str, Any]],
    *,
    filename: str = "explorer_trajectories.jsonl",
) -> None:
    """Append full rollout JSONL records for offline analysis."""
    if not records:
        return
    jsonl_path = os.path.join(output_dir, filename)
    with open(jsonl_path, "a") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _trajectory_gcs_dir(cfg: TrainConfig, *, run_name: Optional[str] = None) -> str:
    gcs_base = os.environ.get("TMX_GCS_CHECKPOINT_BASE", "").strip()
    if not gcs_base:
        gcs_base = "gs://two-model-exploration-checkpoints/checkpoints"
    run_name = run_name or os.path.basename(os.path.abspath(cfg.output_dir.rstrip("/")))
    gcs_base = gcs_base.rstrip("/")
    if os.path.basename(gcs_base) == run_name:
        return f"{gcs_base}/trajectories"
    return f"{gcs_base}/{run_name}/trajectories"


def _write_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _vllm_timeout_seconds() -> int:
    return max(1, _env_int("TMX_VLLM_REQUEST_TIMEOUT_SECONDS", 180))


def _vllm_max_retries() -> int:
    return max(0, _env_int("TMX_VLLM_MAX_RETRIES", 0))


def _upload_trajectory_file(cfg: TrainConfig, local_path: str, target_names: Tuple[str, ...]) -> None:
    if not os.path.exists(local_path):
        return
    if not _env_flag("TMX_ALLOW_GCS_ARTIFACTS", "0"):
        if _env_flag("TMX_HF_MIRROR_TRAJECTORIES", "0"):
            for target_name in target_names:
                _maybe_upload_path_to_hf(
                    local_path,
                    cfg,
                    f"{_hf_run_prefix(cfg)}/{_stage_name_for_output(cfg)}/trajectories/{target_name}",
                    is_dir=False,
                )
        return
    gcs_dir = _trajectory_gcs_dir(cfg)
    for target_name in target_names:
        ret = subprocess.run(
            ["gsutil", "cp", local_path, f"{gcs_dir}/{target_name}"],
            capture_output=True,
            text=True,
            timeout=900,
        )
        if ret.returncode != 0:
            print(f"[ckpt] WARN trajectory upload failed: {ret.stderr[:300]}", flush=True)
            return

def _base_policy_snapshot_enabled(cfg: TrainConfig) -> bool:
    if cfg.init_checkpoint:
        return False
    return os.environ.get("TMX_SAVE_BASE_POLICY_TRAJECTORIES", "1") not in (
        "0",
        "false",
        "False",
        "no",
    )


def _save_base_policy_trajectory_snapshot(
    *,
    cfg: TrainConfig,
    tokenizer,
    examples: List[DapoExample],
    server_urls: List[str],
    params,
    model: Qwen3Model,
    feature_step,
    rnd_map: Optional[Dict[str, Any]],
) -> None:
    """Persist base-policy rollouts before any optimizer step.

    These records are intentionally separate from explorer_trajectories.jsonl so
    SFT/Actual training does not accidentally distill base-policy outputs.
    """
    if not _base_policy_snapshot_enabled(cfg):
        if cfg.init_checkpoint:
            print("[base-policy] skipped: init_checkpoint is set, so this is not a base-policy run", flush=True)
        return
    if not examples:
        return

    prompts_per_step = int(cfg.grpo_batch_size) * int(cfg.grpo_grad_accum)
    batches = max(1, int(os.environ.get("TMX_BASE_POLICY_TRAJECTORY_BATCHES", "1") or "1"))
    seed = int(os.environ.get("TMX_BASE_POLICY_TRAJECTORY_SEED", str(cfg.seed)) or str(cfg.seed))
    rng = np.random.default_rng(seed)
    all_records: List[Dict[str, Any]] = []
    tmp_norm_state = {
        "novelty": {"mean": 0.0, "std": 1.0, "initialized": False},
        "correctness": {"mean": 0.0, "std": 1.0, "initialized": False},
    }

    base_max_tokens = int(
        os.environ.get("TMX_BASE_POLICY_MAX_COMPLETION_LEN", str(cfg.max_completion_len))
        or str(cfg.max_completion_len)
    )
    print(
        f"[base-policy] saving step-0 snapshot batches={batches} "
        f"prompts_per_batch={prompts_per_step} max_tokens={base_max_tokens}",
        flush=True,
    )
    for batch_idx in range(batches):
        idxs = rng.choice(len(examples), size=prompts_per_step, replace=False).tolist()
        prompts = [examples[i] for i in idxs]
        t0 = time.perf_counter()
        completions = batched_generate(
            [p.prompt_text for p in prompts],
            server_urls,
            n_per_prompt=cfg.grpo_num_generations,
            max_tokens=base_max_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            enable_thinking=cfg.enable_thinking,
            model=cfg.model_name,
            concurrency=cfg.vllm_concurrency,
            fanout_per_prompt=cfg.vllm_fanout_per_prompt,
            timeout=_vllm_timeout_seconds(),
            max_retries=_vllm_max_retries(),
        )
        scored_rows, _unused_rnd_map, reward_summary = _score_rollouts(
            tokenizer=tokenizer,
            examples=prompts,
            completions=completions,
            cfg=cfg,
            params=params,
            model=model,
            feature_step=feature_step,
            rnd_map=rnd_map,
            norm_state=tmp_norm_state,
        )
        for row in scored_rows:
            all_records.append(_trajectory_record(
                row["problem_id"],
                row["prompt_text"],
                row["completion_text"],
                row["ground_truth"],
                row["correctness_reward"],
                row["novelty_reward"],
                row["blended_reward"],
                row["is_correct"],
                0,
                finish_reason=row["finish_reason"],
                completion_token_length=int(row["completion_token_length"]),
                clipped=bool(row["clipped"]),
                terminated=bool(row["terminated"]),
                valid_answer=bool(row["valid_answer"]),
                extra={
                    "base_policy": True,
                    "snapshot_batch_index": int(batch_idx),
                    "r_correct_raw": float(row["correctness_raw"]),
                    "r_correct_used": float(row["correctness_reward"]),
                    "r_novel_raw": float(row["novelty_raw"]),
                    "r_novel_used": float(row["novelty_used"]),
                    "r_novel_reward": float(row["novelty_reward"]),
                    "novelty_gate": float(row["novelty_gate"]),
                    "r_novel_layers": dict(row.get("r_novel_layers", {})),
                    "termination_penalty": float(row.get("termination_penalty", 0.0)),
                    "invalid_answer_penalty": float(row.get("invalid_answer_penalty", 0.0)),
                    "format_quality": float(row["format_quality"]),
                    "format_penalty": float(row["format_penalty"]),
                    "format_penalty_weight": float(cfg.format_penalty_weight),
                    "format_penalty_breakdown": dict(row.get("format_penalty_breakdown", {})),
                    "novelty_text": str(row.get("novelty_text", "")),
                    "novelty_text_contract": str(row.get("novelty_text_contract", cfg.novelty_text_contract)),
                    "completion_canonical": row.get("completion_canonical"),
                    "used_first_line": bool(row.get("used_first_line", False)),
                },
            ))
        print(
            f"[base-policy] batch {batch_idx + 1}/{batches} rows={len(scored_rows)} "
            f"reward={reward_summary.get('reward/final_mean', 0.0):.3f} "
            f"dt={time.perf_counter() - t0:.1f}s",
            flush=True,
        )

    local_path = os.path.join(cfg.output_dir, "base_policy_trajectories_step_000000.jsonl")
    _write_jsonl(local_path, all_records)
    _upload_trajectory_file(
        cfg,
        local_path,
        (
            "base_policy_trajectories_step_000000.jsonl",
            "base_policy_trajectories_latest.jsonl",
            "explorer_trajectories_step_000000.jsonl",
            "rollouts_step_000000.jsonl",
        ),
    )
    print(f"[base-policy] saved {len(all_records)} rows -> {local_path}", flush=True)


def _stage_name_for_output(cfg: TrainConfig) -> str:
    hint = f"{cfg.wandb_run_name} {cfg.output_dir}".lower()
    if "actual_grpo" in hint or "/actual/grpo" in hint:
        return "actual_grpo"
    if "actual" in hint:
        return "actual"
    return "explorer"


def _maybe_upload_hf_dir_to_gcs(local_dir: str, gcs_target: str) -> str | None:
    """Upload HF model directory contents to exactly ``gcs_target``."""
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
            print(f"[vllm-reload] uploaded HF bundle {local_dir} -> {gcs_target}", flush=True)
            return gcs_target
        print(f"[vllm-reload] WARN upload failed: {ret.stderr[-1000:]}", flush=True)
    except Exception as e:
        print(f"[vllm-reload] WARN upload error: {e}", flush=True)
    return None


_ORBAX_CHECKPOINT_TO_HF_EXPORT_CODE = r"""
import os
import traceback

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")

import jax
import numpy as np
import orbax.checkpoint as ocp

from tmx_jax.weights import export_flax_params_to_hf_dir

ckpt_path = os.environ["TMX_EXPORT_ORBAX_CKPT"]
output_dir = os.environ["TMX_EXPORT_HF_OUTPUT_DIR"]
model_name = os.environ["TMX_EXPORT_MODEL_NAME"]
num_layers = int(os.environ["TMX_EXPORT_NUM_HIDDEN_LAYERS"])
tie_word_embeddings = os.environ.get("TMX_EXPORT_TIE_WORD_EMBEDDINGS", "1") not in ("0", "false", "False")
use_qk_norm = os.environ.get("TMX_EXPORT_USE_QK_NORM", "1") not in ("0", "false", "False")
hf_weight_prefix = os.environ.get("TMX_EXPORT_HF_WEIGHT_PREFIX", "model")

try:
    print(f"[vllm-export-child] restoring {ckpt_path}", flush=True)
    ckpt = ocp.PyTreeCheckpointer().restore(ckpt_path)
    params = ckpt.get("params", ckpt) if isinstance(ckpt, dict) else ckpt
    params = jax.tree_util.tree_map(lambda x: np.asarray(x), params)
    print(f"[vllm-export-child] exporting HF bundle -> {output_dir}", flush=True)
    export_flax_params_to_hf_dir(
        params,
        model_name,
        num_layers,
        tie_word_embeddings,
        output_dir,
        use_qk_norm=use_qk_norm,
        hf_weight_prefix=hf_weight_prefix,
    )
    print("[vllm-export-child] done", flush=True)
except Exception:
    traceback.print_exc()
    raise
"""


def _export_vllm_hf_from_gcs_checkpoint(
    gcs_checkpoint_path: str,
    cfg: TrainConfig,
    m_cfg: Qwen3Config,
    local_dir: str,
) -> bool:
    """Export a multihost Orbax checkpoint to a local HF bundle in a CPU child.

    The live train process owns globally sharded JAX arrays, so materializing
    params with ``np.asarray`` on one source host is invalid. A direct-GCS Orbax
    checkpoint is shared storage; restoring it in a fresh CPU-only process yields
    host arrays that the normal HF exporter can serialize.
    """
    env = os.environ.copy()
    for key in list(env):
        if (
            key.startswith("TPU_")
            or key.startswith("JAX_")
            or key.startswith("PJRT_")
            or key in {"CLOUD_TPU_TASK_ID", "COORDINATOR_ADDRESS", "LOCAL_RANK", "RANK", "WORLD_SIZE"}
        ):
            env.pop(key, None)
    env.update(
        {
            "JAX_PLATFORMS": "cpu",
            "XLA_FLAGS": "--xla_force_host_platform_device_count=1",
            "TMX_EXPORT_ORBAX_CKPT": gcs_checkpoint_path,
            "TMX_EXPORT_HF_OUTPUT_DIR": os.path.abspath(local_dir),
            "TMX_EXPORT_MODEL_NAME": cfg.model_name,
            "TMX_EXPORT_NUM_HIDDEN_LAYERS": str(int(m_cfg.num_hidden_layers)),
            "TMX_EXPORT_TIE_WORD_EMBEDDINGS": "1" if bool(m_cfg.tie_word_embeddings) else "0",
            "TMX_EXPORT_USE_QK_NORM": "1" if bool(m_cfg.use_qk_norm) else "0",
            "TMX_EXPORT_HF_WEIGHT_PREFIX": str(m_cfg.hf_weight_prefix),
        }
    )
    timeout = float(os.environ.get("TMX_VLLM_EXPORT_FROM_CKPT_TIMEOUT_SECONDS", "7200") or "7200")
    ret = subprocess.run(
        [sys.executable, "-c", _ORBAX_CHECKPOINT_TO_HF_EXPORT_CODE],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if ret.returncode == 0:
        out = (ret.stdout or "").strip()
        if out:
            print(out[-2000:], flush=True)
        return True
    text = ((ret.stdout or "") + "\n" + (ret.stderr or "")).strip()
    print(f"[vllm-reload] WARN checkpoint-to-HF export failed: {text[-3000:]}", flush=True)
    return False


def _export_vllm_hf_from_live_params(
    params,
    cfg: TrainConfig,
    m_cfg: Qwen3Config,
    local_dir: str,
) -> bool:
    """Export live multihost-sharded params to an HF bundle.

    All hosts must call this helper. ``process_allgather`` materializes the
    global sharded arrays as host numpy arrays; only process 0 writes the HF
    files, while the other hosts return after participating in the collective.
    """
    if jax.process_count() > 1:
        if jax.process_index() == 0:
            print("[vllm-reload] all-gathering live params for HF export", flush=True)
        params_host = multihost_utils.process_allgather(params)
    else:
        params_host = params
    params_host = jax.tree_util.tree_map(lambda x: np.asarray(x), params_host)
    if jax.process_index() != 0:
        del params_host
        gc.collect()
        return True
    try:
        export_flax_params_to_hf_dir(
            params_host,
            cfg.model_name,
            int(m_cfg.num_hidden_layers),
            bool(m_cfg.tie_word_embeddings),
            os.path.abspath(local_dir),
            use_qk_norm=bool(m_cfg.use_qk_norm),
            hf_weight_prefix=str(m_cfg.hf_weight_prefix),
        )
    except Exception:
        _remove_tree(local_dir)
        _ensure_space_for_vllm_export(cfg, local_dir)
        export_flax_params_to_hf_dir(
            params_host,
            cfg.model_name,
            int(m_cfg.num_hidden_layers),
            bool(m_cfg.tie_word_embeddings),
            os.path.abspath(local_dir),
            use_qk_norm=bool(m_cfg.use_qk_norm),
            hf_weight_prefix=str(m_cfg.hf_weight_prefix),
        )
    finally:
        del params_host
        gc.collect()
    return True


def _maybe_reload_vllm_from_gcs(
    gcs_model_path: str | None,
    cfg: TrainConfig,
    *,
    stage: str,
    hf_repo_id: str = "",
    hf_path_in_repo: str = "",
    hf_repo_type: str = "dataset",
    hf_revision: str = "",
    model_tar_url: str = "",
) -> bool:
    """Reload configured rollout workers with the exported current-policy HF bundle.

    The reload is called only at checkpoint/resume boundaries, before the next
    rollout starts, so the OpenAI-compatible endpoints are not killed mid-request.
    """
    tpu_name = (os.environ.get("TMX_VLLM_RELOAD_TPU_NAME") or "").strip()
    zone = (os.environ.get("TMX_VLLM_RELOAD_ZONE") or "").strip()
    workers = (os.environ.get("TMX_VLLM_RELOAD_WORKERS") or "").strip()
    if not (tpu_name and zone and workers):
        print(f"[vllm-reload] skipped: reload env not set for stage={stage}", flush=True)
        return False

    env = os.environ.copy()
    env.update(
        {
            "GCS_MODEL_PATH": gcs_model_path or "",
            "HF_REPO_ID": hf_repo_id,
            "HF_REPO_TYPE": hf_repo_type,
            "HF_PATH_IN_REPO": hf_path_in_repo,
            "HF_REVISION": hf_revision,
            "MODEL_TAR_URL": model_tar_url,
            "TPU_NAME": tpu_name,
            "ZONE": zone,
            "TPU_PROJECT": (
                os.environ.get("TMX_VLLM_RELOAD_TPU_PROJECT")
                or os.environ.get("TPU_PROJECT")
                or os.environ.get("CLOUDSDK_CORE_PROJECT")
                or ""
            ),
            "WORKERS": workers,
            "SERVED_MODEL_NAME": cfg.model_name,
            "LOCAL_MODEL_DIR_BASENAME": f"tmx_{stage}_model",
            "MAX_MODEL_LEN": str(int(cfg.max_total_len)),
            "MAX_NUM_SEQS": os.environ.get("TMX_TRAIN_RELOAD_VLLM_MAX_NUM_SEQS", "32"),
            "MAX_NUM_BATCHED_TOKENS": os.environ.get("TMX_VLLM_RELOAD_MAX_BATCHED_TOKENS", "65536"),
            "VLLM_ENABLE_CHUNKED_PREFILL": os.environ.get("VLLM_ENABLE_CHUNKED_PREFILL", "1"),
            "PARALLELISM": os.environ.get("TMX_VLLM_RELOAD_PARALLELISM", "8"),
        }
    )
    try:
        ret = subprocess.run(
            ["bash", "tmx_jax/scripts/reload_vllm_direct_slice.sh"],
            capture_output=True,
            text=True,
            timeout=7200,
            env=env,
        )
        if ret.returncode == 0:
            print(f"[vllm-reload] reloaded rollout workers for {stage}", flush=True)
            return True
        print(f"[vllm-reload] WARN reload failed: {(ret.stderr or ret.stdout)[-2000:]}", flush=True)
    except Exception as e:
        print(f"[vllm-reload] WARN reload error: {e}", flush=True)
    return False


def _internal_reload_host() -> str:
    explicit = (os.environ.get("TMX_VLLM_INTERNAL_RELOAD_HOST") or "").strip()
    if explicit:
        return explicit
    coord = (os.environ.get("JAX_COORDINATOR_ADDRESS") or "").strip()
    if coord:
        return coord.rsplit(":", 1)[0]
    return ""


def _start_internal_reload_tar_server(local_dir: str, cfg: TrainConfig, step: int) -> tuple[str, subprocess.Popen] | tuple[str, None]:
    """Serve a local HF-format model bundle to rollout workers over the pod VPC.

    This avoids the expensive HF/GCS round-trip for vLLM reloads while keeping
    durable checkpoints/trajectories on HF through the normal checkpoint path.
    """
    if os.environ.get("TMX_VLLM_RELOAD_TRANSPORT", "hf").strip().lower() not in (
        "internal_http",
        "internal-tar",
        "pod_http",
    ):
        return "", None
    host = _internal_reload_host()
    if not host:
        raise RuntimeError("TMX_VLLM_RELOAD_TRANSPORT=internal_http requires TMX_VLLM_INTERNAL_RELOAD_HOST or JAX_COORDINATOR_ADDRESS")
    serve_root = os.path.join(cfg.output_dir, "vllm_reload_internal")
    os.makedirs(serve_root, exist_ok=True)
    tar_name = f"step_{step:06d}.tar"
    tar_path = os.path.join(serve_root, tar_name)
    tmp_tar = f"{tar_path}.tmp"
    if os.path.exists(tmp_tar):
        os.remove(tmp_tar)
    if os.path.exists(tar_path):
        os.remove(tar_path)
    print(f"[vllm-reload] packing internal reload tar {tar_path}", flush=True)
    subprocess.run(["tar", "-C", os.path.abspath(local_dir), "-cf", tmp_tar, "."], check=True, timeout=1800)
    os.replace(tmp_tar, tar_path)
    port = int(os.environ.get("TMX_VLLM_INTERNAL_RELOAD_PORT", "18765") or "18765")
    url = f"http://{host}:{port}/{tar_name}"
    proc = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port), "--bind", "0.0.0.0", "--directory", serve_root],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(1.0)
    if proc.poll() is not None:
        raise RuntimeError(f"internal reload HTTP server exited immediately on port={port}")
    print(f"[vllm-reload] serving internal reload bundle {url}", flush=True)
    return url, proc


def _request_external_vllm_reload(
    gcs_model_path: str | None,
    cfg: TrainConfig,
    *,
    stage: str,
    step: int,
    hf_repo_id: str = "",
    hf_path_in_repo: str = "",
    hf_repo_type: str = "dataset",
    hf_revision: str = "",
) -> bool:
    """Request an out-of-process vLLM reload and wait until it is acknowledged.

    This is for TPU setups where the training VM can run JAX, but its local
    gcloud/IAP path cannot reliably SSH into rollout workers. A local watcher
    can read the request JSON, run the reload with local gcloud, then touch the
    done marker. Training waits here so it does not continue with stale rollout
    weights after a checkpoint boundary.
    """
    request_dir = os.path.join(cfg.output_dir, "vllm_reload_requests")
    os.makedirs(request_dir, exist_ok=True)
    stem = f"{stage}_step_{step:06d}"
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
        "step": int(step),
        "model_name": cfg.model_name,
        "max_model_len": int(cfg.max_total_len),
        "tpu_name": os.environ.get("TMX_VLLM_RELOAD_TPU_NAME", ""),
        "tpu_project": (
            os.environ.get("TMX_VLLM_RELOAD_TPU_PROJECT")
            or os.environ.get("TPU_PROJECT")
            or os.environ.get("CLOUDSDK_CORE_PROJECT")
            or ""
        ),
        "zone": os.environ.get("TMX_VLLM_RELOAD_ZONE", ""),
        "workers": os.environ.get("TMX_VLLM_RELOAD_WORKERS", ""),
        "server_urls": os.environ.get("TMX_VLLM_RELOAD_SERVER_URLS", cfg.vllm_server_urls),
        "multi_server": os.environ.get("TMX_VLLM_MULTI_SERVER", "0"),
        "chips": os.environ.get("TMX_VLLM_MULTI_SERVER_CHIPS", ""),
        "port_base": os.environ.get("TMX_VLLM_MULTI_SERVER_PORT_BASE", ""),
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
    print(f"[vllm-reload] external request written: {request_path}", flush=True)

    if os.environ.get("TMX_EXTERNAL_VLLM_RELOAD_WAIT", "1") in ("0", "false", "False"):
        return True

    timeout_s = float(os.environ.get("TMX_EXTERNAL_VLLM_RELOAD_TIMEOUT", "0") or "0")
    poll_s = float(os.environ.get("TMX_EXTERNAL_VLLM_RELOAD_POLL_SECONDS", "5") or "5")
    required = os.environ.get("TMX_EXTERNAL_VLLM_RELOAD_REQUIRED", "1") not in ("0", "false", "False")
    retry_on_fail = os.environ.get("TMX_EXTERNAL_VLLM_RELOAD_RETRY_ON_FAIL", "1") not in ("0", "false", "False")
    deadline = time.time() + timeout_s if timeout_s > 0 else None
    while deadline is None or time.time() < deadline:
        if os.path.exists(done_path):
            print(f"[vllm-reload] external reload acknowledged: {done_path}", flush=True)
            return True
        if os.path.exists(fail_path):
            with open(fail_path, "r", errors="ignore") as f:
                msg = f.read()[-2000:]
            text = f"[vllm-reload] external reload failed: {msg}"
            if required and retry_on_fail:
                print(f"WARN {text}; keeping training blocked until a retry succeeds", flush=True)
                try:
                    os.remove(fail_path)
                except FileNotFoundError:
                    pass
                time.sleep(poll_s)
                continue
            if required:
                raise RuntimeError(text)
            print(f"WARN {text}", flush=True)
            return False
        time.sleep(poll_s)

    text = f"[vllm-reload] external reload timed out waiting for {done_path}"
    if required:
        raise TimeoutError(text)
    print(f"WARN {text}", flush=True)
    return False


def _maybe_export_and_reload_vllm(state, cfg: TrainConfig, m_cfg: Qwen3Config, step: int, *, reason: str) -> None:
    live_allgather_export = (
        jax.process_count() > 1
        and _env_flag("TMX_VLLM_EXPORT_FROM_LIVE_ALLGATHER", "1")
        and not _env_flag("TMX_VLLM_EXPORT_FROM_GCS_CHECKPOINT", "0")
    )
    if jax.process_index() != 0 and not live_allgather_export:
        return
    gcs_base = (os.environ.get("TMX_GCS_CHECKPOINT_BASE") or "").strip()
    hf_repo_id = _hf_artifact_repo()
    mirror_to_hf = _env_flag("TMX_HF_MIRROR_VLLM_EXPORTS", "0") and bool(hf_repo_id)
    internal_reload_transport = os.environ.get("TMX_VLLM_RELOAD_TRANSPORT", "hf").strip().lower() in (
        "internal_http",
        "internal-tar",
        "pod_http",
    )
    if not gcs_base and not mirror_to_hf and not internal_reload_transport:
        if jax.process_index() == 0:
            print("[vllm-reload] skipped: no GCS base or HF artifact repo configured", flush=True)
        return
    if jax.process_index() != 0 and live_allgather_export:
        _export_vllm_hf_from_live_params(state.params, cfg, m_cfg, "")
        return
    stage = _stage_name_for_output(cfg)
    local_dir = os.path.join(cfg.output_dir, "vllm_reload_hf", f"step_{step:06d}")
    gcs_target = f"{gcs_base.rstrip('/')}/{stage}_vllm_reload_hf/step_{step:06d}" if gcs_base else ""
    hf_path_in_repo = f"{_hf_run_prefix(cfg)}/{stage}_vllm_reload_hf/step_{step:06d}"
    reload_required = os.environ.get("TMX_EXTERNAL_VLLM_RELOAD_REQUIRED", "1") not in ("0", "false", "False")
    reload_mode = os.environ.get("TMX_TRAIN_VLLM_RELOAD_MODE", "direct").strip().lower()
    external_wait = os.environ.get("TMX_EXTERNAL_VLLM_RELOAD_WAIT", "1") not in ("0", "false", "False")
    waits_for_reload = reload_mode != "external" or external_wait or reload_required
    async_hf_upload = (
        os.environ.get("TMX_HF_MIRROR_ASYNC", "1") not in ("0", "false", "False", "no")
        and os.environ.get("TMX_REQUIRE_HF_ARTIFACT_UPLOAD", "0") in ("0", "false", "False", "no")
    )
    delete_local = os.environ.get("TMX_DELETE_LOCAL_VLLM_RELOAD_AFTER_UPLOAD", "0") not in (
        "0",
        "false",
        "False",
        "no",
    )
    if mirror_to_hf and async_hf_upload:
        # A background upload keeps the path open after this function returns.
        # Deleting here caused missing reload bundles on HF at checkpoint
        # boundaries. The checkpoint janitor prunes older exports separately.
        delete_local = False
    uploaded = None
    internal_reload_url = ""
    internal_server_proc = None
    try:
        _ensure_space_for_vllm_export(cfg, local_dir)
        _remove_tree(local_dir)
        if live_allgather_export:
            if not _export_vllm_hf_from_live_params(state.params, cfg, m_cfg, local_dir):
                raise RuntimeError("live all-gather HF export failed")
        elif jax.process_count() > 1 and _env_flag("TMX_VLLM_EXPORT_FROM_GCS_CHECKPOINT", "0"):
            ckpt_dir = os.path.abspath(os.path.join(cfg.output_dir, f"step_{step:06d}"))
            ckpt_gcs = _gcs_checkpoint_target(ckpt_dir, cfg)
            print(f"[vllm-reload] exporting HF bundle from shared checkpoint {ckpt_gcs}", flush=True)
            if not _export_vllm_hf_from_gcs_checkpoint(ckpt_gcs, cfg, m_cfg, local_dir):
                raise RuntimeError(f"checkpoint-to-HF export failed for {ckpt_gcs}")
        else:
            params_host = jax.tree_util.tree_map(lambda x: np.asarray(x), state.params)
            try:
                export_flax_params_to_hf_dir(
                    params_host,
                    cfg.model_name,
                    int(m_cfg.num_hidden_layers),
                    bool(m_cfg.tie_word_embeddings),
                    os.path.abspath(local_dir),
                    use_qk_norm=bool(m_cfg.use_qk_norm),
                    hf_weight_prefix=str(m_cfg.hf_weight_prefix),
                )
            except Exception:
                _remove_tree(local_dir)
                _ensure_space_for_vllm_export(cfg, local_dir)
                export_flax_params_to_hf_dir(
                    params_host,
                    cfg.model_name,
                    int(m_cfg.num_hidden_layers),
                    bool(m_cfg.tie_word_embeddings),
                    os.path.abspath(local_dir),
                    use_qk_norm=bool(m_cfg.use_qk_norm),
                    hf_weight_prefix=str(m_cfg.hf_weight_prefix),
                )
            del params_host
        internal_reload_url, internal_server_proc = _start_internal_reload_tar_server(local_dir, cfg, step)
        if gcs_target:
            uploaded = _maybe_upload_hf_dir_to_gcs(local_dir, gcs_target)
        hf_uploaded = False
        if mirror_to_hf and not internal_reload_url:
            force_sync_hf = bool(waits_for_reload and not uploaded)
            if uploaded:
                hf_uploaded = _maybe_upload_gcs_uri_to_hf(
                    uploaded,
                    cfg,
                    hf_path_in_repo,
                    is_dir=True,
                    force_sync=False,
                )
            else:
                hf_uploaded = _maybe_upload_path_to_hf(
                    local_dir,
                    cfg,
                    hf_path_in_repo,
                    is_dir=True,
                    force_sync=force_sync_hf,
                )
        gc.collect()
        if internal_reload_url or uploaded or hf_uploaded:
            reload_stage = f"{stage}_{reason}_step_{step:06d}"
            if reload_mode == "external":
                ok = _request_external_vllm_reload(
                    uploaded,
                    cfg,
                    stage=reload_stage,
                    step=step,
                    hf_repo_id=hf_repo_id if hf_uploaded else "",
                    hf_path_in_repo=hf_path_in_repo if hf_uploaded else "",
                    hf_repo_type="dataset",
                )
            else:
                ok = _maybe_reload_vllm_from_gcs(
                    uploaded,
                    cfg,
                    stage=reload_stage,
                    hf_repo_id=hf_repo_id if hf_uploaded else "",
                    hf_path_in_repo=hf_path_in_repo if hf_uploaded else "",
                    hf_repo_type="dataset",
                    model_tar_url=internal_reload_url,
                )
            if reload_required and not ok:
                raise RuntimeError(f"vLLM reload failed for {reload_stage}")
    except Exception as e:
        if reload_required:
            raise
        print(f"[vllm-reload] WARN export/reload failed at step={step}: {e}", flush=True)
    finally:
        if internal_server_proc is not None:
            internal_server_proc.terminate()
            try:
                internal_server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                internal_server_proc.kill()
        if delete_local:
            _remove_tree(local_dir)


def _maybe_run_aime_probe(
    *,
    cfg: TrainConfig,
    tokenizer,
    server_urls: List[str],
    step: int,
    stage: str,
    use_wandb: bool,
) -> None:
    """Run a small fixed AIME24 probe against the freshly reloaded policy.

    This is a trend metric, not the final benchmark. The official comparison
    remains the full 30-problem x 32-rollout x 32k eval after Actual GRPO.
    """
    every = int(os.environ.get("TMX_AIME_PROBE_EVERY_STEPS", "0") or "0")
    if every <= 0 or step <= 0 or step % every != 0:
        return
    if not server_urls:
        print("[eval-probe] skipped: no vLLM server URLs", flush=True)
        return

    max_tokens = int(os.environ.get("TMX_AIME_PROBE_MAX_TOKENS", str(cfg.max_completion_len)) or str(cfg.max_completion_len))
    if max_tokens > int(cfg.max_completion_len):
        print(
            f"[eval-probe] clamping max_tokens={max_tokens} to training completion len={cfg.max_completion_len}",
            flush=True,
        )
        max_tokens = int(cfg.max_completion_len)
    num_problems = int(os.environ.get("TMX_AIME_PROBE_NUM_PROBLEMS", "6") or "6")
    num_rollouts = int(os.environ.get("TMX_AIME_PROBE_NUM_ROLLOUTS", "4") or "4")
    temperature = float(os.environ.get("TMX_AIME_PROBE_TEMPERATURE", "0.6") or "0.6")
    top_p = float(os.environ.get("TMX_AIME_PROBE_TOP_P", str(cfg.top_p)) or str(cfg.top_p))
    top_k = int(os.environ.get("TMX_AIME_PROBE_TOP_K", str(cfg.top_k)) or str(cfg.top_k))
    concurrency = int(os.environ.get("TMX_AIME_PROBE_CONCURRENCY", str(min(8, max(1, cfg.vllm_concurrency)))) or "8")
    fanout = int(os.environ.get("TMX_AIME_PROBE_FANOUT_PER_PROMPT", str(min(cfg.vllm_fanout_per_prompt, max(1, num_rollouts)))) or "1")
    prefix = f"eval_probe/{stage}/aime24"
    print(
        f"[eval-probe] step={step} problems={num_problems} rollouts={num_rollouts} max_tokens={max_tokens} temp={temperature}",
        flush=True,
    )

    try:
        from .eval import OFFICIAL_QWEN3_17B_AIME24, annotate_probe_trend, run_eval
        import wandb

        def _progress(done: int, total: int, elapsed: float) -> None:
            if not use_wandb or not _env_flag("TMX_WANDB_EVAL_PROGRESS", "0"):
                return
            eta = (elapsed / done) * (total - done) if done else 0.0
            wandb.log({
                f"{prefix}/step": float(step),
                f"{prefix}/progress_fraction": float(done / total if total else 0.0),
                f"{prefix}/subrequests_done": float(done),
                f"{prefix}/subrequests_total": float(total),
                f"{prefix}/elapsed_seconds": float(elapsed),
                f"{prefix}/eta_seconds": float(eta),
            }, **_wandb_log_kwargs(step, stage))

        res = run_eval(
            tokenizer,
            server_urls,
            num_rollouts=num_rollouts,
            max_tokens=max_tokens,
            model=cfg.model_name,
            temperature=temperature,
            which="AIME_2024",
            enable_thinking=cfg.enable_thinking,
            concurrency=concurrency,
            top_p=top_p,
            top_k=top_k,
            fanout_per_prompt=fanout,
            max_problems=num_problems,
            progress_callback=_progress,
        )
        from .prompting import prompt_metadata
        res["protocol"] = {
            **prompt_metadata(),
            "probe": True,
            "stage": stage,
            "step": int(step),
            "which": "AIME_2024",
            "num_problems": int(num_problems),
            "num_rollouts": int(num_rollouts),
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "official_final_baseline": OFFICIAL_QWEN3_17B_AIME24,
        }
        probe_dir = os.path.join(cfg.output_dir, "eval_probes")
        os.makedirs(probe_dir, exist_ok=True)
        probe_path = os.path.join(probe_dir, f"aime24_{stage}_step_{step:06d}.json")
        annotate_probe_trend(res, probe_dir)
        with open(probe_path, "w") as f:
            json.dump(res, f, indent=2, sort_keys=True)

        payload = {
            f"{prefix}/step": float(step),
            f"{prefix}/avg_at_{num_rollouts}": float(res.get("avg_at_n", 0.0) or 0.0),
            f"{prefix}/pass_at_1": float(res.get("pass_at_1", res.get("avg_at_n", 0.0)) or 0.0),
            f"{prefix}/pass_at_2": float(res.get("pass_at_2", res.get("avg_at_n", 0.0)) or 0.0),
            f"{prefix}/pass_at_4": float(res.get("pass_at_4", res.get("avg_at_n", 0.0)) or 0.0),
            f"{prefix}/error_fraction": float(res.get("error_fraction", 0.0) or 0.0),
            f"{prefix}/delta_vs_initial": float(res.get("delta_vs_initial_avg_at_n", 0.0) or 0.0),
            f"{prefix}/best_avg_at_n": float(res.get("best_avg_at_n_so_far", res.get("avg_at_n", 0.0)) or 0.0),
            "aime24_probe_step": float(step),
            f"aime24_probe_avg_at_{num_rollouts}": float(res.get("avg_at_n", 0.0) or 0.0),
            "aime24_probe_pass_at_4": float(res.get("pass_at_4", res.get("avg_at_n", 0.0)) or 0.0),
            "aime24_probe_delta_vs_initial": float(res.get("delta_vs_initial_avg_at_n", 0.0) or 0.0),
            "aime24_probe_best_so_far": float(res.get("best_avg_at_n_so_far", res.get("avg_at_n", 0.0)) or 0.0),
        }
        if use_wandb:
            wandb.log(payload, **_wandb_log_kwargs(step, stage))

        gcs_base = (os.environ.get("TMX_GCS_CHECKPOINT_BASE") or "").strip().rstrip("/")
        target = ""
        if gcs_base:
            target = f"{gcs_base}/eval_probes/{os.path.basename(probe_path)}"
            subprocess.run(["gcloud", "storage", "cp", probe_path, target], check=False, timeout=900)
        if _env_flag("TMX_HF_MIRROR_EVAL_PROBES", "0"):
            hf_target = f"{_hf_run_prefix(cfg)}/{_stage_name_for_output(cfg)}/eval_probes/{os.path.basename(probe_path)}"
            if target:
                _maybe_upload_gcs_uri_to_hf(target, cfg, hf_target, is_dir=False)
            else:
                _maybe_upload_path_to_hf(probe_path, cfg, hf_target, is_dir=False)
        print(f"[eval-probe] avg@{num_rollouts}={float(res.get('avg_at_n', 0.0))*100:.2f}% -> {probe_path}", flush=True)
    except Exception as e:
        if os.environ.get("TMX_AIME_PROBE_REQUIRED", "0") in ("1", "true", "True", "yes"):
            raise
        print(f"[eval-probe] WARN failed at step={step}: {e}", flush=True)


def run_training(cfg: TrainConfig):
    validate_contract(cfg, require_eval_contract=False)
    init_distributed()
    is_source_host = (jax.process_index() == 0)

    if os.environ.get("TMX_VLLM_SERVER_URLS"):
        if not cfg.vllm_server_urls:
            cfg = type(cfg)(**{**asdict(cfg), "vllm_server_urls": os.environ["TMX_VLLM_SERVER_URLS"]})
    server_urls = parse_server_urls(cfg.vllm_server_urls)
    if not server_urls:
        raise RuntimeError("Set --vllm-server-urls or TMX_VLLM_SERVER_URLS to vLLM /v1 endpoints.")

    os.makedirs(cfg.output_dir, exist_ok=True)
    traj_path = os.path.join(cfg.output_dir, "explorer_trajectories.jsonl")
    if cfg.init_checkpoint and os.path.exists(traj_path):
        print(f"[train] resume mode: appending trajectories to {traj_path}", flush=True)
    else:
        open(traj_path, "w").close()  # truncate only for fresh runs

    mesh = build_mesh()
    print(f"[train] devices={len(jax.devices())} mesh.size={mesh.size}", flush=True)
    if is_source_host:
        train_prompts_per_step = int(cfg.grpo_batch_size) * int(cfg.grpo_grad_accum)
        train_rows_per_step = train_prompts_per_step * int(cfg.grpo_num_generations)
        print(
            "[train] contract "
            f"G={int(cfg.grpo_num_generations)} "
            f"batch={int(cfg.grpo_batch_size)} "
            f"grad_accum={int(cfg.grpo_grad_accum)} "
            f"prompts_per_step={train_prompts_per_step} "
            f"trajectories_per_step={train_rows_per_step} "
            f"clip_high={float(cfg.grpo_clip_epsilon_high):.2f} "
            f"max_completion={int(cfg.max_completion_len)} "
            f"loss={cfg.grpo_loss_type} "
            f"loss_agg={cfg.grpo_loss_aggregation} "
            f"updates_per_rollout={int(cfg.grpo_updates_per_rollout)} "
            f"logprob_chunk={int(cfg.logprob_chunk_tokens)} "
            f"dynamic_sampling={int(bool(cfg.dynamic_sampling))} "
            f"soft_overlong={int(cfg.soft_overlong_expected_len)}+{int(cfg.soft_overlong_cache_len)} "
            f"KL={float(cfg.grpo_kl_beta):.3g} "
            f"vllm_endpoints={len(server_urls)} "
            f"reload_every={os.environ.get('TMX_TRAIN_VLLM_RELOAD_EVERY_STEPS', '0')} "
            f"save_every={cfg.save_every_steps}",
            flush=True,
        )

    tokenizer = None
    if is_source_host:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    stage_name = _stage_name_for_output(cfg)
    if cfg.lambda_novelty != 0.0 and stage_name != "explorer":
        raise ValueError(
            "lambda_novelty is only valid for Explorer GRPO. Actual GRPO must be correctness-only."
        )

    train_prompts_per_step = int(cfg.grpo_batch_size) * int(cfg.grpo_grad_accum)
    prompts_per_step = int(os.environ.get("TMX_EFFECTIVE_PROMPTS_PER_STEP", str(train_prompts_per_step)) or train_prompts_per_step)
    if prompts_per_step <= 0:
        raise ValueError("effective prompts per step must be > 0")
    replicate_rollout_batch = os.environ.get("TMX_REPLICATE_ROLLOUT_BATCH", "0") not in ("0", "false", "False")
    actual_rollout_rows_per_step = prompts_per_step * int(cfg.grpo_num_generations)
    train_rows_per_step = train_prompts_per_step * int(cfg.grpo_num_generations)
    if actual_rollout_rows_per_step > train_rows_per_step and not replicate_rollout_batch:
        raise ValueError(
            f"effective rollout rows={actual_rollout_rows_per_step} exceed train rows={train_rows_per_step}"
        )
    if train_rows_per_step % int(mesh.size) != 0 and not replicate_rollout_batch:
        raise ValueError(
            "GRPO rollout rows must be divisible by the FSDP mesh size before "
            "multihost sharding. Got "
            f"grpo_batch_size={cfg.grpo_batch_size}, "
            f"grpo_grad_accum={cfg.grpo_grad_accum}, "
            f"grpo_num_generations={cfg.grpo_num_generations}, "
            f"train_rows={train_rows_per_step}, mesh_size={int(mesh.size)}. "
            "Increase grpo_batch_size * grpo_grad_accum; for G=16 on a "
            "64-chip slice use at least 4 prompts per optimizer step."
        )
    if is_source_host and replicate_rollout_batch:
        print(
            f"[train] effective prompts_per_step={prompts_per_step}; "
            f"replicating rollout rows {actual_rollout_rows_per_step} across FSDP hosts",
            flush=True,
        )
    elif is_source_host and prompts_per_step != train_prompts_per_step:
        print(
            f"[train] effective prompts_per_step={prompts_per_step}; "
            f"padding rollout rows {actual_rollout_rows_per_step}->{train_rows_per_step}",
            flush=True,
        )
    updates_per_rollout = max(1, int(cfg.grpo_updates_per_rollout))
    if cfg.trajectory_batch_size != prompts_per_step:
        print(
            f"[train] overriding trajectory_batch_size={cfg.trajectory_batch_size} "
            f"to contract prompts_per_step={prompts_per_step}",
            flush=True,
        )

    with mesh:
        state, model, m_cfg, param_sh = _init_state(cfg, mesh)
        state, restored_step = _maybe_restore_state_from_checkpoint(state, param_sh, cfg, mesh)
        # The KL reference is the policy at the start of this GRPO stage.
        # For Actual GRPO, that is the restored SFT checkpoint, not base Qwen.
        ref_params = state.params if float(cfg.grpo_kl_beta) > 0.0 else None
        print(f"[train] model config: {m_cfg}", flush=True)
        if ref_params is not None and is_source_host:
            print(f"[train] reference KL enabled: beta={cfg.grpo_kl_beta}", flush=True)

        novelty_enabled = float(cfg.lambda_novelty or 0.0) != 0.0
        novelty_feature_step = None
        rnd_map = None
        reward_norm_state = {
            "novelty": {"mean": 0.0, "std": 1.0, "initialized": False},
            "correctness": {"mean": 0.0, "std": 1.0, "initialized": False},
        }
        if novelty_enabled:
            if (cfg.novelty_feature_source or "multilayer").lower() != "multilayer":
                raise ValueError("JAX novelty currently requires novelty_feature_source=multilayer")
            layers = tuple(int(x) for x in cfg.novelty_layers)
            if not layers:
                layers = (int(m_cfg.num_hidden_layers) // 4, int(m_cfg.num_hidden_layers) // 2, (3 * int(m_cfg.num_hidden_layers)) // 4)
            for layer in layers:
                if layer < 0 or layer >= int(m_cfg.num_hidden_layers):
                    raise ValueError(f"novelty layer {layer} out of range for {m_cfg.num_hidden_layers} layers")
            cfg = type(cfg)(**{**asdict(cfg), "novelty_layers": list(layers)})
            if is_source_host:
                print("[novelty] building feature extractor", flush=True)
            novelty_feature_step = _make_novelty_feature_step(model, layers, cfg.novelty_layer_pool)
            if is_source_host:
                print("[novelty] initializing RND state on all hosts", flush=True)
            # JAX/Flax initialization runs after distributed startup. Execute it
            # on every host so rank 0 does not block in a rank-local TPU
            # computation while the other hosts skip ahead to collectives.
            rnd_map = init_rnd_map(
                rng=int(cfg.seed) + 17,
                layers=layers,
                input_dim=int(m_cfg.hidden_size),
                hidden_dim=int(cfg.rnd_hidden),
                lr=float(cfg.rnd_lr),
            )
            if is_source_host:
                print("[novelty] RND init complete", flush=True)
            rnd_map = _maybe_restore_rnd_from_checkpoint(rnd_map, cfg)
            if not is_source_host:
                rnd_map = None
            if is_source_host:
                print(
                    f"[novelty] enabled: lambda={cfg.lambda_novelty} layers={list(layers)} "
                    f"pool={cfg.novelty_layer_pool} agg={cfg.novelty_layer_agg} "
                    f"metric={cfg.novelty_metric}",
                    flush=True,
                )

        grpo_cfg = GrpoConfig(
            clip_epsilon=cfg.grpo_clip_epsilon,
            clip_epsilon_high=cfg.grpo_clip_epsilon_high,
            max_completion_len=cfg.max_completion_len,
            kl_beta=cfg.grpo_kl_beta,
            loss_aggregation=cfg.grpo_loss_aggregation,
        )

        examples: List[DapoExample] = []
        if is_source_host:
            assert tokenizer is not None
            examples = load_examples(
                cfg.dataset_name,
                tokenizer,
                max_examples=cfg.max_train_examples,
                enable_thinking=cfg.enable_thinking,
            )
            print(f"[train] {cfg.dataset_name} examples loaded: {len(examples)}", flush=True)
            if int(os.environ.get("TMX_DATASET_SHARD_COUNT", "1") or "1") > 1:
                manifest_path = os.path.join(cfg.output_dir, "dataset_round_manifest.json")
                with open(manifest_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "dataset_name": cfg.dataset_name,
                            "max_train_examples": int(cfg.max_train_examples),
                            "shard_index": int(os.environ.get("TMX_DATASET_SHARD_INDEX", "0") or "0"),
                            "shard_count": int(os.environ.get("TMX_DATASET_SHARD_COUNT", "1") or "1"),
                            "shard_policy": os.environ.get("TMX_DATASET_SHARD_POLICY", "uniform"),
                            "shard_seed": int(os.environ.get("TMX_DATASET_SHARD_SEED", "0") or "0"),
                            "num_examples": len(examples),
                            "problem_ids": [ex.problem_id for ex in examples],
                        },
                        f,
                        indent=2,
                    )
                print(f"[train] wrote dataset shard manifest: {manifest_path}", flush=True)
        else:
            print("[train] non-source host skipping tokenizer + DAPO load", flush=True)

        # Shard every row-aligned training tensor across the fsdp axis. Replicating
        # PPO metadata while sharding token tensors can leave the differentiated
        # train step with row-misaligned auxiliaries at the JIT boundary even when
        # the no-grad rescoring path looks clean.
        if os.environ.get("TMX_REPLICATE_ROLLOUT_BATCH", "0") not in ("0", "false", "False"):
            step_batch_pspec = {
                "full_input_ids":        P(None, None),
                "full_attention_mask":   P(None, None),
                "completion_mask":       P(None, None),
                "old_per_token_logps":   P(None, None),
                "ref_per_token_logps":   P(None, None),
                "advantages":            P(None),
            }
            if is_source_host:
                print("[train] replicated rollout batch enabled; no row padding to mesh size", flush=True)
        else:
            step_batch_pspec = {
                "full_input_ids":        P(FSDP_AXIS, None),
                "full_attention_mask":   P(FSDP_AXIS, None),
                "completion_mask":       P(FSDP_AXIS, None),
                "old_per_token_logps":   P(FSDP_AXIS, None),
                "ref_per_token_logps":   P(FSDP_AXIS, None),
                "advantages":            P(FSDP_AXIS),
            }

        rows_per_microbatch = int(cfg.grpo_batch_size) * int(cfg.grpo_num_generations)
        host_grad_rows_per_microbatch = rows_per_microbatch
        host_grad_accum_steps = int(cfg.grpo_grad_accum)
        debug_compare_paths = os.environ.get("TMX_DEBUG_COMPARE_LOGPROB_PATHS", "0") not in ("0", "false", "False")
        use_inline_old_logprobs = os.environ.get("TMX_USE_INLINE_OLD_LOGPROBS", "1") not in ("0", "false", "False")
        if updates_per_rollout > 1:
            use_inline_old_logprobs = False
            if is_source_host:
                print(
                    f"[train] multi-update GRPO enabled: updates_per_rollout={updates_per_rollout}; "
                    "using fixed rescored old logprobs",
                    flush=True,
                )
        # This is diagnostic only when inline old-logprobs are enabled: the loss
        # uses stop_gradient(current_logps) inside the train JIT. A separate
        # rescore pass costs ~15-30s/step at 10k tokens, so keep it opt-in.
        track_external_old_gap = debug_compare_paths or (
            use_inline_old_logprobs
            and os.environ.get("TMX_TRACK_EXTERNAL_OLD_GAP", "0") not in ("0", "false", "False")
        )
        if os.environ.get("TMX_VECTORIZE_GRPO_ROWS", "0") not in ("0", "false", "False"):
            row_vector_chunk = rows_per_microbatch
        else:
            row_vector_chunk = int(os.environ.get("TMX_GRPO_ROW_VECTOR_CHUNK", "1"))
        row_vector_chunk = max(1, min(int(row_vector_chunk), int(rows_per_microbatch)))
        if is_source_host and row_vector_chunk > 1:
            print(f"[train] row vectorization enabled: chunk={row_vector_chunk}/{rows_per_microbatch}", flush=True)

        replicate_rollout_batch = os.environ.get("TMX_REPLICATE_ROLLOUT_BATCH", "0") not in (
            "0",
            "false",
            "False",
        )
        host_grad_accum = os.environ.get("TMX_HOST_GRAD_ACCUM", "0") not in ("0", "false", "False")
        compact_nonzero_adv = os.environ.get("TMX_HOST_GRAD_COMPACT_NONZERO_ADV", "0") not in (
            "0",
            "false",
            "False",
        )
        compact_adv_eps = float(os.environ.get("TMX_HOST_GRAD_COMPACT_ADV_EPS", "1e-8"))
        compact_fixed_token_shape = os.environ.get("TMX_HOST_GRAD_FIXED_TOKEN_SHAPE", "0") not in (
            "0",
            "false",
            "False",
        )
        if host_grad_accum:
            if updates_per_rollout != 1:
                raise ValueError("TMX_HOST_GRAD_ACCUM supports only grpo_updates_per_rollout=1")
            if not use_inline_old_logprobs or track_external_old_gap:
                raise ValueError(
                    "TMX_HOST_GRAD_ACCUM currently requires inline old logprobs "
                    "and TMX_TRACK_EXTERNAL_OLD_GAP=0"
                )
            if ref_params is not None:
                raise ValueError("TMX_HOST_GRAD_ACCUM currently requires KL/ref_params disabled")
            total_rows_per_step = (
                int(cfg.grpo_batch_size)
                * int(cfg.grpo_grad_accum)
                * int(cfg.grpo_num_generations)
            )
            min_global_rows = max(1, int(jax.process_count()) * int(jax.local_device_count()))
            if (not replicate_rollout_batch) and host_grad_rows_per_microbatch < min_global_rows:
                host_grad_rows_per_microbatch = min_global_rows
            # Keep microbatches aligned to the original prompt-group unit so equal
            # microbatch averaging remains equivalent to averaging all rollout rows.
            if host_grad_rows_per_microbatch % rows_per_microbatch != 0:
                host_grad_rows_per_microbatch = (
                    ((host_grad_rows_per_microbatch + rows_per_microbatch - 1) // rows_per_microbatch)
                    * rows_per_microbatch
                )
            if total_rows_per_step % host_grad_rows_per_microbatch != 0:
                raise ValueError(
                    "TMX_HOST_GRAD_ACCUM requires total rollout rows to divide host micro rows: "
                    f"total_rows={total_rows_per_step} host_micro_rows={host_grad_rows_per_microbatch}"
                )
            host_grad_accum_steps = total_rows_per_step // host_grad_rows_per_microbatch
            if is_source_host:
                print(
                    "[train] host grad accumulation enabled: "
                    f"micro_rows={host_grad_rows_per_microbatch} "
                    f"accum={host_grad_accum_steps} "
                    f"base_micro_rows={rows_per_microbatch} "
                    f"config_accum={int(cfg.grpo_grad_accum)}",
                    flush=True,
                )
                if compact_nonzero_adv:
                    print(
                        "[train] compact nonzero-advantage host grad enabled: "
                        f"eps={compact_adv_eps:g} fixed_token_shape={int(compact_fixed_token_shape)}",
                        flush=True,
                    )

        step_fn = make_train_step(
            grpo_cfg,
            int(cfg.max_prompt_len),
            bool(m_cfg.tie_word_embeddings),
            int(cfg.logprob_chunk_tokens),
            rows_per_microbatch,
            int(cfg.grpo_grad_accum),
            use_inline_old_logprobs,
            track_external_old_gap,
            row_vector_chunk,
            ref_params is not None,
        )
        micro_grad_fn = make_micro_grad_step(
            state.apply_fn,
            grpo_cfg,
            int(cfg.max_prompt_len),
            bool(m_cfg.tie_word_embeddings),
            int(cfg.logprob_chunk_tokens),
            use_inline_old_logprobs,
            track_external_old_gap,
            row_vector_chunk,
            ref_params is not None,
        )
        apply_grads_fn = make_apply_grads_step()
        rescore_fn = make_rescore_step(
            int(cfg.max_prompt_len),
            bool(m_cfg.tie_word_embeddings),
            int(cfg.logprob_chunk_tokens),
            rows_per_microbatch,
            int(cfg.grpo_grad_accum),
        )
        trainpath_logprob_fn = make_trainpath_logprob_step(
            int(cfg.max_prompt_len),
            bool(m_cfg.tie_word_embeddings),
            int(cfg.logprob_chunk_tokens),
            rows_per_microbatch,
            int(cfg.grpo_grad_accum),
        )
        # When reference KL is enabled, ref_params intentionally snapshots the
        # stage-start policy. It may alias state.params, so donating `state`
        # can make XLA see the same buffer as both donated and non-donated in a
        # single Execute() call. Keep batch donation, but do not donate state in
        # KL runs.
        if updates_per_rollout > 1:
            # Reuse the same rollout batch for several updates, so batch buffers
            # cannot be donated on the first call.
            step_donate_argnums = () if ref_params is not None else (0,)
        else:
            step_donate_argnums = (1,) if ref_params is not None else (0, 1)
        step_jit = jax.jit(step_fn, donate_argnums=step_donate_argnums)
        micro_grad_jit = jax.jit(micro_grad_fn, donate_argnums=(1,))
        apply_grads_jit = jax.jit(apply_grads_fn, donate_argnums=(0, 1))
        rescore_jit = jax.jit(
            rescore_fn,
            out_shardings=named(mesh, step_batch_pspec["old_per_token_logps"]),
        )
        trainpath_logprob_jit = jax.jit(
            trainpath_logprob_fn,
            out_shardings=named(mesh, step_batch_pspec["old_per_token_logps"]),
        )

        name_hint = f"{cfg.wandb_run_name} {cfg.output_dir}".lower()
        grpo_metric_prefix = (
            os.environ.get("TMX_WANDB_GRPO_PREFIX")
            or ("actual_grpo" if "actual_grpo" in name_hint or "/actual/grpo" in name_hint else "explorer_grpo")
        )
        train_metric_prefix = "grpo" if grpo_metric_prefix == "explorer_grpo" else grpo_metric_prefix

        # WandB
        use_wandb = False
        wandb_mod = None
        try:
            import wandb
            wandb_mod = wandb
            if is_source_host:
                if (os.environ.get("WANDB_MODE") or "").strip().lower() == "disabled":
                    os.environ.pop("WANDB_MODE", None)
                wandb_run_id = _stable_wandb_run_id(cfg, grpo_metric_prefix)
                wandb.init(
                    project=cfg.wandb_project, entity=cfg.wandb_entity or None,
                    name=cfg.wandb_run_name or f"jax_lambda{cfg.lambda_novelty}",
                    id=wandb_run_id,
                    resume="allow",
                    group=os.environ.get("TMX_WANDB_GROUP") or (cfg.wandb_run_name or f"jax_lambda{cfg.lambda_novelty}"),
                    job_type=grpo_metric_prefix,
                    config=asdict(cfg),
                )
                try:
                    wandb.define_metric(f"{grpo_metric_prefix}/step")
                    wandb.define_metric(f"{grpo_metric_prefix}/*", step_metric=f"{grpo_metric_prefix}/step")
                    wandb.define_metric(f"{train_metric_prefix}/step")
                    wandb.define_metric(f"{train_metric_prefix}/*", step_metric=f"{train_metric_prefix}/step")
                    wandb.define_metric("step")
                    for metric_name in sorted(_WANDB_TOP_LEVEL_METRICS):
                        wandb.define_metric(metric_name, step_metric="step")
                    wandb.define_metric("aime24_probe_step")
                    wandb.define_metric("aime24_probe_*", step_metric="aime24_probe_step")
                    probe_step = f"eval_probe/{grpo_metric_prefix}/aime24/step"
                    wandb.define_metric(probe_step)
                    wandb.define_metric(f"eval_probe/{grpo_metric_prefix}/aime24/*", step_metric=probe_step)
                except Exception:
                    pass
                use_wandb = True
        except Exception as e:
            if is_source_host:
                print(f"[train] wandb disabled: {e}", flush=True)

        rng = np.random.default_rng(cfg.seed)
        global_step = int(restored_step)
        if global_step == 0 and _base_policy_snapshot_enabled(cfg):
            if is_source_host:
                assert tokenizer is not None
                try:
                    _save_base_policy_trajectory_snapshot(
                        cfg=cfg,
                        tokenizer=tokenizer,
                        examples=examples,
                        server_urls=server_urls,
                        params=state.params,
                        model=model,
                        feature_step=novelty_feature_step,
                        rnd_map=rnd_map,
                    )
                finally:
                    if novelty_enabled and jax.process_count() > 1:
                        _broadcast_novelty_feature_stop(is_source_host=True)
            elif novelty_enabled:
                _novelty_feature_receiver_loop(
                    params=state.params,
                    cfg=cfg,
                    feature_step=novelty_feature_step,
                )
        if global_step == 0 and _base_policy_snapshot_enabled(cfg) and jax.process_count() > 1:
            multihost_utils.sync_global_devices("base_policy_snapshot_ready")
        if global_step > 0 and is_source_host:
            # Keep prompt sampling deterministic across checkpoint resume by
            # consuming the same RNG draws used by the completed steps.
            for _ in range(global_step):
                rng.choice(len(examples), size=prompts_per_step, replace=False)
            print(f"[train] resumed at global_step={global_step}; advanced prompt RNG", flush=True)
        reward_window = deque(maxlen=25)
        correctness_window = deque(maxlen=25)
        valid_answer_window = deque(maxlen=25)
        terminated_window = deque(maxlen=25)
        effective_group_window = deque(maxlen=25)
        reload_every_steps = _env_int("TMX_TRAIN_VLLM_RELOAD_EVERY_STEPS", 0)
        reload_first_step = _env_int("TMX_TRAIN_VLLM_RELOAD_FIRST_STEP", 0)
        local_checkpoint_every_steps = _env_int("TMX_LOCAL_CHECKPOINT_EVERY_STEPS", 0)
        gcs_checkpoint_every_steps = _env_int(
            "TMX_GCS_CHECKPOINT_EVERY_STEPS",
            int(cfg.save_every_steps),
        )
        gcs_checkpoint_first_step = _env_int("TMX_GCS_CHECKPOINT_FIRST_STEP", 0)
        if (
            global_step > 0
            and os.environ.get("TMX_TRAIN_VLLM_RELOAD_ON_RESUME", "0") not in ("0", "false", "False")
        ):
            _maybe_export_and_reload_vllm(state, cfg, m_cfg, global_step, reason="resume")
        if global_step > 0 and jax.process_count() > 1:
            multihost_utils.sync_global_devices(f"train_resume_ready_{global_step}")

        prefetch_rollouts = (
            is_source_host
            and not bool(cfg.dynamic_sampling)
            and os.environ.get("TMX_PREFETCH_ROLLOUTS", "0") not in ("0", "false", "False")
        )
        prefetch_pool: Optional[ThreadPoolExecutor] = (
            ThreadPoolExecutor(max_workers=1) if prefetch_rollouts else None
        )
        pending_rollout: Optional[Future] = None

        def _start_rollout_prefetch() -> Optional[Future]:
            if not prefetch_rollouts:
                return None
            assert tokenizer is not None
            idxs = rng.choice(len(examples), size=prompts_per_step, replace=False).tolist()
            prefetched_prompts = [examples[i] for i in idxs]

            def _job():
                t_gen = time.perf_counter()
                prefetched_completions = batched_generate(
                    [p.prompt_text for p in prefetched_prompts],
                    server_urls,
                    n_per_prompt=cfg.grpo_num_generations,
                    max_tokens=cfg.max_completion_len,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    top_k=cfg.top_k,
                    enable_thinking=cfg.enable_thinking,
                    model=cfg.model_name,
                    concurrency=cfg.vllm_concurrency,
                    fanout_per_prompt=cfg.vllm_fanout_per_prompt,
                    timeout=_vllm_timeout_seconds(),
                    max_retries=_vllm_max_retries(),
                )
                return (
                    prefetched_prompts,
                    prefetched_completions,
                    _rollout_diversity_metrics(prefetched_completions),
                    time.perf_counter() - t_gen,
                )

            assert prefetch_pool is not None
            return prefetch_pool.submit(_job)

        if prefetch_rollouts:
            pending_rollout = _start_rollout_prefetch()
            if is_source_host:
                print("[train] async rollout prefetch enabled", flush=True)

        while global_step < cfg.grpo_max_steps:
            # 1. Sample prompts
            rollout_t0 = time.perf_counter()
            prompts = None
            scored_rows = None
            all_scored_rows = None
            reward_summary: Dict[str, float] = {}
            dynamic_stats: Dict[str, float] = {}
            if is_source_host:
                if bool(cfg.dynamic_sampling):
                    assert tokenizer is not None
                    try:
                        (
                            prompts,
                            completions,
                            scored_rows,
                            all_scored_rows,
                            reward_summary,
                            rnd_map,
                            diversity_metrics,
                            rollout_dt,
                            dynamic_stats,
                        ) = _dynamic_sample_rollouts(
                            rng=rng,
                            tokenizer=tokenizer,
                            examples=examples,
                            server_urls=server_urls,
                            cfg=cfg,
                            state=state,
                            model=model,
                            feature_step=novelty_feature_step,
                            rnd_map=rnd_map,
                            norm_state=reward_norm_state,
                            prompts_per_step=prompts_per_step,
                        )
                    finally:
                        if novelty_enabled and jax.process_count() > 1:
                            _broadcast_novelty_feature_stop(is_source_host=True)
                elif pending_rollout is not None:
                    prompts, completions, diversity_metrics, rollout_dt = pending_rollout.result()
                    pending_rollout = None
                else:
                    idxs = rng.choice(len(examples), size=prompts_per_step, replace=False).tolist()
                    prompts = [examples[i] for i in idxs]

                    # 2. vLLM generate
                    completions = batched_generate(
                        [p.prompt_text for p in prompts],
                        server_urls,
                        n_per_prompt=cfg.grpo_num_generations,
                        max_tokens=cfg.max_completion_len,
                        temperature=cfg.temperature,
                        top_p=cfg.top_p,
                        top_k=cfg.top_k,
                        enable_thinking=cfg.enable_thinking,
                        model=cfg.model_name,
                        concurrency=cfg.vllm_concurrency,
                        fanout_per_prompt=cfg.vllm_fanout_per_prompt,
                        timeout=_vllm_timeout_seconds(),
                        max_retries=_vllm_max_retries(),
                    )
                    if _env_flag("TMX_DISABLE_ROLLOUT_DIVERSITY", "0"):
                        print("[train] skipping online rollout diversity metrics", flush=True)
                        diversity_metrics = {}
                    else:
                        print("[train] computing online rollout diversity metrics", flush=True)
                        diversity_metrics = _rollout_diversity_metrics(completions)
                        print("[train] online rollout diversity metrics complete", flush=True)
                    rollout_dt = time.perf_counter() - rollout_t0
            else:
                if novelty_enabled:
                    _novelty_feature_receiver_loop(
                        params=state.params,
                        cfg=cfg,
                        feature_step=novelty_feature_step,
                    )
                completions = None
                diversity_metrics = {}
                rollout_dt = time.perf_counter() - rollout_t0

            if is_source_host:
                assert prompts is not None and completions is not None
                if scored_rows is None:
                    try:
                        print(f"[train] scoring rollout rows={len(prompts) * int(cfg.grpo_num_generations)}", flush=True)
                        scored_rows, rnd_map, reward_summary = _score_rollouts(
                            tokenizer=tokenizer,
                            examples=prompts,
                            completions=completions,
                            cfg=cfg,
                            params=state.params,
                            model=model,
                            feature_step=novelty_feature_step,
                            rnd_map=rnd_map,
                            norm_state=reward_norm_state,
                        )
                        print("[train] rollout scoring complete", flush=True)
                    finally:
                        if novelty_enabled and jax.process_count() > 1:
                            _broadcast_novelty_feature_stop(is_source_host=True)
                if all_scored_rows is None:
                    all_scored_rows = scored_rows

            # 3. Build batch
            if is_source_host:
                assert prompts is not None and completions is not None
                print("[train] build_rollout_batch start", flush=True)
                batch_local = _build_rollout_batch(tokenizer, prompts, completions, cfg, scored_rows=scored_rows)
                print("[train] build_rollout_batch complete", flush=True)
                print("[train] pad_rollout_batch start", flush=True)
                batch_local = _pad_rollout_batch_rows(
                    batch_local,
                    target_rows=(
                        int(prompts_per_step) * int(cfg.grpo_num_generations)
                        if replicate_rollout_batch
                        else int(cfg.grpo_batch_size) * int(cfg.grpo_grad_accum) * int(cfg.grpo_num_generations)
                    ),
                    pad_id=int(tokenizer.pad_token_id or 0),
                )
                print("[train] pad_rollout_batch complete", flush=True)
                if host_grad_accum:
                    print("[train] sort_rollout_batch_by_length start", flush=True)
                    batch_local = _sort_rollout_batch_rows_by_length(batch_local, descending=True)
                    print("[train] sort_rollout_batch_by_length complete", flush=True)
                else:
                    print("[train] trim_rollout_batch_tokens start", flush=True)
                    batch_local = _trim_rollout_batch_tokens(
                        batch_local,
                        prompt_len=int(cfg.max_prompt_len),
                    )
                    print("[train] trim_rollout_batch_tokens complete", flush=True)
            else:
                batch_local = _empty_rollout_batch(cfg)
            if host_grad_accum:
                batch = None
                rescore_dt = 0.0
                ref_rescore_dt = 0.0
            else:
                if is_source_host:
                    print("[train] broadcast batch_local start", flush=True)
                batch_local = _shard_batch_for_hosts(batch_local, step_batch_pspec)
                if is_source_host:
                    print("[train] broadcast batch_local complete", flush=True)

                if is_source_host:
                    print("[train] to_global_batch start", flush=True)
                batch = _to_global_batch(batch_local, mesh, {
                    "full_input_ids": step_batch_pspec["full_input_ids"],
                    "full_attention_mask": step_batch_pspec["full_attention_mask"],
                    "completion_mask": step_batch_pspec["completion_mask"],
                    "advantages": step_batch_pspec["advantages"],
                })
                if is_source_host:
                    print("[train] to_global_batch complete", flush=True)

                # Rescore old logprobs on the exact same sharded row-scan path used by
                # the train step. This removes semantic skew between the rollout-policy
                # logprobs and the optimizer path.
                if use_inline_old_logprobs and not track_external_old_gap:
                    if is_source_host:
                        print("[train] old_logprobs inline-zero start", flush=True)
                    batch["old_per_token_logps"] = jnp.zeros_like(batch["completion_mask"], dtype=jnp.float32)
                    rescore_dt = 0.0
                    if is_source_host:
                        print("[train] old_logprobs inline-zero complete", flush=True)
                else:
                    if is_source_host:
                        print("[train] old_logprobs rescore start", flush=True)
                    rescore_t0 = time.perf_counter()
                    batch["old_per_token_logps"] = rescore_jit(state, batch)
                    jax.block_until_ready(batch["old_per_token_logps"])
                    if jax.process_count() == 1:
                        # On single-host runs we can materialize and re-upload to
                        # make the rescore/train JIT boundary explicit. On
                        # multi-host runs the array spans non-addressable devices;
                        # the rescore JIT already returned the requested sharding.
                        batch["old_per_token_logps"] = jax.device_put(
                            np.asarray(batch["old_per_token_logps"]),
                            named(mesh, step_batch_pspec["old_per_token_logps"]),
                        )
                        jax.block_until_ready(batch["old_per_token_logps"])
                    rescore_dt = time.perf_counter() - rescore_t0
                    if is_source_host:
                        print(f"[train] old_logprobs rescore complete dt={rescore_dt:.1f}s", flush=True)

                if debug_compare_paths:
                    debug_lp = trainpath_logprob_jit(state, batch)
                    jax.block_until_ready(debug_lp)
                    if jax.process_count() == 1:
                        debug_lp = jax.device_put(
                            np.asarray(debug_lp),
                            named(mesh, step_batch_pspec["old_per_token_logps"]),
                        )
                    debug_gap = jnp.abs(debug_lp - batch["old_per_token_logps"]) * batch["completion_mask"].astype(jnp.float32)
                    debug_denom = jnp.maximum(batch["completion_mask"].astype(jnp.float32).sum(), 1.0)
                    debug_gap_mean = float(debug_gap.sum() / debug_denom)
                    debug_gap_max = float(debug_gap.max())
                    if is_source_host:
                        print(
                            f"[debug] rescore_vs_trainpath gap_mean={debug_gap_mean:.3e} gap_max={debug_gap_max:.3e}",
                            flush=True,
                        )

                if ref_params is not None:
                    ref_t0 = time.perf_counter()
                    ref_state = state.replace(params=ref_params)
                    batch["ref_per_token_logps"] = rescore_jit(ref_state, batch)
                    jax.block_until_ready(batch["ref_per_token_logps"])
                    if jax.process_count() == 1:
                        batch["ref_per_token_logps"] = jax.device_put(
                            np.asarray(batch["ref_per_token_logps"]),
                            named(mesh, step_batch_pspec["ref_per_token_logps"]),
                        )
                        jax.block_until_ready(batch["ref_per_token_logps"])
                    ref_rescore_dt = time.perf_counter() - ref_t0
                else:
                    ref_rescore_dt = 0.0

            # 5. Train step
            will_reload_after_step = (
                _step_due(global_step + 1, reload_every_steps, reload_first_step)
            )
            if prefetch_rollouts and not will_reload_after_step and (global_step + 1) < cfg.grpo_max_steps:
                # vLLM weights are unchanged between reload checkpoints, so this
                # overlaps remote rollout generation for the next optimizer step
                # with the local TPU update for the current step. We deliberately
                # do not prefetch across a pending reload boundary.
                pending_rollout = _start_rollout_prefetch()

            rewards_np = np.asarray(batch_local["rewards"], dtype=np.float32)
            correct_np = np.asarray(batch_local["is_correct"], dtype=np.float32)
            lengths_np = np.asarray(batch_local["completion_lengths"], dtype=np.float32)
            terminated_np = np.asarray(batch_local["terminated"], dtype=np.float32)
            clipped_np = np.asarray(batch_local["clipped"], dtype=np.float32)
            valid_answer_np = np.asarray(batch_local["valid_answer"], dtype=np.float32)
            correctness_rewards_np = np.asarray(batch_local["correctness_rewards"], dtype=np.float32)
            novelty_rewards_np = np.asarray(batch_local["novelty_rewards"], dtype=np.float32)
            novelty_raw_rewards_np = np.asarray(batch_local["novelty_raw_rewards"], dtype=np.float32)
            novelty_used_rewards_np = np.asarray(batch_local["novelty_used_rewards"], dtype=np.float32)
            format_quality_np = np.asarray(batch_local["format_quality"], dtype=np.float32)
            format_penalties_np = np.asarray(batch_local["format_penalties"], dtype=np.float32)
            real_mask_np = np.asarray(batch_local.get("real_row_mask", np.ones_like(rewards_np)), dtype=np.float32) > 0.5
            if real_mask_np.shape == rewards_np.shape and real_mask_np.any():
                rewards_np = rewards_np[real_mask_np]
                correct_np = correct_np[real_mask_np]
                lengths_np = lengths_np[real_mask_np]
                terminated_np = terminated_np[real_mask_np]
                clipped_np = clipped_np[real_mask_np]
                valid_answer_np = valid_answer_np[real_mask_np]
                correctness_rewards_np = correctness_rewards_np[real_mask_np]
                novelty_rewards_np = novelty_rewards_np[real_mask_np]
                novelty_raw_rewards_np = novelty_raw_rewards_np[real_mask_np]
                novelty_used_rewards_np = novelty_used_rewards_np[real_mask_np]
                format_quality_np = format_quality_np[real_mask_np]
                format_penalties_np = format_penalties_np[real_mask_np]
            reward_mean = float(rewards_np.mean())
            reward_std = float(rewards_np.std())
            reward_min = float(rewards_np.min())
            reward_max = float(rewards_np.max())
            correct_fraction = float(correct_np.mean())
            correctness_used_mean = float(correctness_rewards_np.mean())
            novelty_reward_mean = float(novelty_rewards_np.mean())
            novelty_raw_mean = float(novelty_raw_rewards_np.mean())
            novelty_used_mean = float(novelty_used_rewards_np.mean())
            correct_count = float(correct_np.sum())
            completion_token_length_mean = float(lengths_np.mean())
            completion_token_length_max = float(lengths_np.max())
            terminated_fraction = float(terminated_np.mean())
            clipped_fraction = float(clipped_np.mean())
            valid_answer_fraction = float(valid_answer_np.mean())
            format_quality_mean = float(format_quality_np.mean())
            format_penalty_mean = float(format_penalties_np.mean())
            terminated_mask = terminated_np > 0.5
            terminated_count = float(terminated_mask.sum())
            if terminated_count > 0:
                terminated_final_mean = float(rewards_np[terminated_mask].mean())
                terminated_correctness_mean = float(correctness_rewards_np[terminated_mask].mean())
                terminated_novelty_mean = float(novelty_rewards_np[terminated_mask].mean())
            else:
                terminated_final_mean = 0.0
                terminated_correctness_mean = 0.0
                terminated_novelty_mean = 0.0
            prompt_any_correct_fraction = float(
                correct_np.reshape(-1, int(cfg.grpo_num_generations)).max(axis=1).mean()
            )
            group_correct_counts = correct_np.reshape(-1, int(cfg.grpo_num_generations)).sum(axis=1)
            group_count = float(group_correct_counts.shape[0])
            informative_group_count = float(((group_correct_counts > 0) & (group_correct_counts < int(cfg.grpo_num_generations))).sum())
            all_wrong_group_fraction = float((group_correct_counts == 0).mean()) if group_correct_counts.size else 0.0
            all_correct_group_fraction = float((group_correct_counts == int(cfg.grpo_num_generations)).mean()) if group_correct_counts.size else 0.0
            effective_group_fraction = informative_group_count / max(1.0, group_count)
            reward_window.append(reward_mean)
            correctness_window.append(correct_fraction)
            valid_answer_window.append(valid_answer_fraction)
            terminated_window.append(terminated_fraction)
            effective_group_window.append(effective_group_fraction)
            reward_stats_25 = _metric_window_stats(reward_window)
            correctness_stats_25 = _metric_window_stats(correctness_window)
            valid_answer_stats_25 = _metric_window_stats(valid_answer_window)
            terminated_stats_25 = _metric_window_stats(terminated_window)
            effective_group_stats_25 = _metric_window_stats(effective_group_window)
            reward_rolling_25 = reward_stats_25["mean"]
            reward_trend_25 = reward_stats_25["trend"]
            reward_ci95_25 = reward_stats_25["ci95"]
            correctness_rolling_25 = correctness_stats_25["mean"]
            correctness_trend_25 = correctness_stats_25["trend"]
            correctness_ci95_25 = correctness_stats_25["ci95"]
            valid_answer_rolling_25 = valid_answer_stats_25["mean"]
            terminated_rolling_25 = terminated_stats_25["mean"]
            effective_group_rolling_25 = effective_group_stats_25["mean"]
            train_batch_local = batch_local
            train_compact_rows = None
            train_scale_mode = "uniform-microbatch"
            train_scale_denominator = float(max(1, int(host_grad_accum_steps)))
            if host_grad_accum and compact_nonzero_adv:
                if is_source_host:
                    adv_all = np.asarray(batch_local["advantages"], dtype=np.float32)
                    train_rows_total = int(np.asarray(batch_local["completion_mask"]).shape[0])
                    real_mask_all = np.asarray(
                        batch_local.get("real_row_mask", np.ones((train_rows_total,), dtype=np.float32)),
                        dtype=np.float32,
                    ) > 0.5
                    nonzero_adv_idx = np.flatnonzero((np.abs(adv_all) > float(compact_adv_eps)) & real_mask_all)
                    train_compact_rows = int(nonzero_adv_idx.shape[0])
                    agg = str(cfg.grpo_loss_aggregation or "dr_grpo").lower()
                    if agg in ("token-mean", "dapo"):
                        train_scale_mode = "token-mean"
                        train_scale_denominator = float(
                            max(1.0, float(np.asarray(batch_local["completion_mask"], dtype=np.float32).sum()))
                        )
                    elif agg in ("sequence-mean-token-mean", "seq-mean-token-mean", "vanilla"):
                        train_scale_mode = "sequence-mean-token-mean"
                        row_tokens = np.asarray(batch_local["completion_mask"], dtype=np.float32).sum(axis=1)
                        train_scale_denominator = float(max(1.0, float((row_tokens > 0.0).sum())))
                    else:
                        train_scale_mode = "dr_grpo"
                        train_scale_denominator = float(max(1, train_rows_total))
                    if train_compact_rows > 0:
                        train_batch_local = _select_rollout_batch_rows(batch_local, nonzero_adv_idx)
                        train_batch_local = _sort_rollout_batch_rows_by_length(train_batch_local, descending=True)
                    else:
                        train_batch_local = None
                    print(
                        "[train] compact nonzero-advantage rows "
                        f"kept={train_compact_rows}/{train_rows_total} "
                        f"scale_mode={train_scale_mode} "
                        f"scale_denominator={train_scale_denominator:.1f}",
                        flush=True,
                    )
            train_dt = 0.0
            metrics = None
            grad_norm = None
            for update_idx in range(updates_per_rollout):
                if is_source_host:
                    print(
                        f"[train] update_start step={global_step + 1} "
                        f"update={update_idx + 1}/{updates_per_rollout}",
                        flush=True,
                    )
                t0 = time.perf_counter()
                if host_grad_accum:
                    sum_grads = None
                    sum_metrics = None
                    compact_rows_total = (
                        int(np.asarray(train_batch_local["completion_mask"]).shape[0])
                        if compact_nonzero_adv and is_source_host and train_batch_local is not None
                        else int(host_grad_accum_steps) * int(host_grad_rows_per_microbatch)
                    )
                    micro_count = (
                        int(host_grad_accum_steps)
                        if compact_nonzero_adv
                        else int(host_grad_accum_steps)
                    )
                    for micro_idx in range(micro_count):
                        row_start = int(micro_idx) * int(host_grad_rows_per_microbatch)
                        row_end = row_start + int(host_grad_rows_per_microbatch)
                        if is_source_host:
                            source_batch = train_batch_local if compact_nonzero_adv else batch_local
                            if compact_nonzero_adv and (source_batch is None or row_start >= compact_rows_total):
                                micro_local = _slice_batch(
                                    _empty_rollout_batch(cfg),
                                    0,
                                    int(host_grad_rows_per_microbatch),
                                )
                            else:
                                assert source_batch is not None
                                micro_local = _slice_batch(source_batch, row_start, min(row_end, compact_rows_total))
                            actual_micro_rows = int(np.asarray(micro_local["completion_mask"]).shape[0])
                            if compact_nonzero_adv and actual_micro_rows < int(host_grad_rows_per_microbatch):
                                micro_local = _pad_rollout_batch_rows(
                                    micro_local,
                                    target_rows=int(host_grad_rows_per_microbatch),
                                    pad_id=int(tokenizer.pad_token_id or 0),
                                )
                            if not (compact_nonzero_adv and compact_fixed_token_shape):
                                micro_local = _trim_rollout_batch_tokens(
                                    micro_local,
                                    prompt_len=int(cfg.max_prompt_len),
                                )
                            if compact_nonzero_adv:
                                micro_local["_train_scale_mode"] = train_scale_mode
                                micro_local["_train_scale_denominator"] = float(train_scale_denominator)
                            if micro_idx == 0 or (micro_idx + 1) % 8 == 0:
                                micro_cols = int(np.asarray(micro_local["completion_mask"]).shape[1])
                                micro_used = int(np.asarray(micro_local["completion_mask"]).sum(axis=1).max(initial=0))
                                print(
                                    f"[train] micro_grad start step={global_step + 1} "
                                    f"micro={micro_idx + 1}/{int(micro_count)} "
                                    f"rows={int(host_grad_rows_per_microbatch)} cols={micro_cols} used={micro_used}",
                                    flush=True,
                                )
                        else:
                            micro_local = _empty_rollout_batch(cfg)
                        micro_local = _shard_batch_for_hosts(micro_local, step_batch_pspec)
                        micro_weight = 1.0
                        if compact_nonzero_adv:
                            local_scale_mode = str(micro_local.get("_train_scale_mode", train_scale_mode))
                            local_scale_denominator = float(
                                micro_local.get("_train_scale_denominator", train_scale_denominator)
                            )
                            if local_scale_mode == "token-mean":
                                micro_weight = float(
                                    np.asarray(micro_local["completion_mask"], dtype=np.float32).sum()
                                ) / float(max(1.0, local_scale_denominator))
                            elif local_scale_mode in (
                                "sequence-mean-token-mean",
                                "seq-mean-token-mean",
                                "vanilla",
                            ):
                                micro_row_tokens = np.asarray(micro_local["completion_mask"], dtype=np.float32).sum(axis=1)
                                micro_weight = float((micro_row_tokens > 0.0).sum()) / float(
                                    max(1.0, local_scale_denominator)
                                )
                            else:
                                micro_weight = float(host_grad_rows_per_microbatch) / float(
                                    max(1.0, local_scale_denominator)
                                )
                            micro_local.pop("_train_scale_mode", None)
                            micro_local.pop("_train_scale_denominator", None)
                        micro_batch = _to_global_batch(micro_local, mesh, {
                            "full_input_ids": step_batch_pspec["full_input_ids"],
                            "full_attention_mask": step_batch_pspec["full_attention_mask"],
                            "completion_mask": step_batch_pspec["completion_mask"],
                            "advantages": step_batch_pspec["advantages"],
                        })
                        micro_batch["old_per_token_logps"] = jnp.zeros_like(
                            micro_batch["completion_mask"], dtype=jnp.float32
                        )
                        grads_i, metrics_i = micro_grad_jit(state.params, micro_batch, ref_params)
                        jax.block_until_ready(metrics_i["loss"])
                        if compact_nonzero_adv:
                            grads_i = jax.tree_util.tree_map(lambda x: x * micro_weight, grads_i)
                            metrics_i = jax.tree_util.tree_map(lambda x: x * micro_weight, metrics_i)
                        if sum_grads is None:
                            sum_grads = grads_i
                            sum_metrics = metrics_i
                        else:
                            sum_grads = jax.tree_util.tree_map(lambda a, b: a + b, sum_grads, grads_i)
                            sum_metrics = jax.tree_util.tree_map(lambda a, b: a + b, sum_metrics, metrics_i)
                    if sum_grads is not None and sum_metrics is not None:
                        if compact_nonzero_adv:
                            avg_grads = sum_grads
                            metrics = sum_metrics
                        else:
                            inv_acc = 1.0 / float(max(1, int(host_grad_accum_steps)))
                            avg_grads = jax.tree_util.tree_map(lambda x: x * inv_acc, sum_grads)
                            metrics = jax.tree_util.tree_map(lambda x: x * inv_acc, sum_metrics)
                        state, grad_norm = apply_grads_jit(state, avg_grads)
                        jax.block_until_ready(grad_norm)
                else:
                    state, metrics, grad_norm = step_jit(state, batch, ref_params)
                    jax.block_until_ready(metrics["loss"])
                update_dt = time.perf_counter() - t0
                train_dt += update_dt
                if is_source_host:
                    print(
                        f"[train] update_done step={global_step + 1} "
                        f"update={update_idx + 1}/{updates_per_rollout} "
                        f"dt={update_dt:.1f}s",
                        flush=True,
                    )
            global_step += 1
            assert metrics is not None and grad_norm is not None

            # Persist only rollouts that correspond to a completed optimizer step.
            # This keeps JSONL artifacts aligned with metrics/checkpoints and avoids
            # retaining failed-attempt rows after a crash or killed process.
            if is_source_host:
                assert scored_rows is not None
                records = []

                def _record_from_scored_row(row: Dict[str, Any]) -> Dict[str, Any]:
                    return _trajectory_record(
                        row["problem_id"],
                        row["prompt_text"],
                        row["completion_text"],
                        row["ground_truth"],
                        row["correctness_reward"],
                        row["novelty_reward"],
                        row["blended_reward"],
                        row["is_correct"],
                        global_step,
                        finish_reason=row["finish_reason"],
                        completion_token_length=int(row["completion_token_length"]),
                        clipped=bool(row["clipped"]),
                        terminated=bool(row["terminated"]),
                        valid_answer=bool(row["valid_answer"]),
                        extra={
                            "r_correct_raw": float(row["correctness_raw"]),
                            "r_correct_used": float(row["correctness_reward"]),
                            "r_novel_raw": float(row["novelty_raw"]),
                            "r_novel_used": float(row["novelty_used"]),
                            "r_novel_reward": float(row["novelty_reward"]),
                            "novelty_gate": float(row["novelty_gate"]),
                            "r_novel_layers": dict(row.get("r_novel_layers", {})),
                            "termination_penalty": float(row.get("termination_penalty", 0.0)),
                            "invalid_answer_penalty": float(row.get("invalid_answer_penalty", 0.0)),
                            "overlong_penalty": float(row.get("overlong_penalty", 0.0)),
                            "soft_overlong_expected_len": int(row.get("soft_overlong_expected_len", 0) or 0),
                            "soft_overlong_cache_len": int(row.get("soft_overlong_cache_len", 0) or 0),
                            "format_quality": float(row["format_quality"]),
                            "format_penalty": float(row["format_penalty"]),
                            "format_penalty_weight": float(cfg.format_penalty_weight),
                            "format_penalty_breakdown": dict(row.get("format_penalty_breakdown", {})),
                            "novelty_text": str(row.get("novelty_text", "")),
                            "novelty_text_contract": str(row.get("novelty_text_contract", cfg.novelty_text_contract)),
                            "completion_canonical": row.get("completion_canonical"),
                            "used_first_line": bool(row.get("used_first_line", False)),
                            "used_for_training": bool(row.get("used_for_training", True)),
                            "dynamic_sampling_reason": row.get("dynamic_sampling_reason"),
                            "dynamic_sampling_attempt": row.get("dynamic_sampling_attempt"),
                            "dynamic_sampling_group_index": row.get("dynamic_sampling_group_index"),
                            "dynamic_sampling_correct_count": row.get("dynamic_sampling_correct_count"),
                            "dynamic_sampling_clipped_count": row.get("dynamic_sampling_clipped_count"),
                            "dynamic_sampling_fallback": bool(row.get("dynamic_sampling_fallback", False)),
                        },
                    )

                for row in scored_rows:
                    records.append(_record_from_scored_row(row))
                _append_rollout_records(cfg.output_dir, records)
                if all_scored_rows is not None and all_scored_rows is not scored_rows:
                    _append_rollout_records(
                        cfg.output_dir,
                        [_record_from_scored_row(row) for row in all_scored_rows],
                        filename="rollouts_all.jsonl",
                    )

            grad_norm_value = float(grad_norm)
            if use_wandb:
                import wandb
                payload = {
                    # Compact dashboards: enough to judge learning/collapse.
                    f"{grpo_metric_prefix}/step": global_step,
                }
                compact_metrics = {
                    "step": global_step,
                    "loss": float(metrics["loss"]),
                    "reward": reward_mean,
                    "reward_rolling_25": reward_rolling_25,
                    "reward_trend_25": reward_trend_25,
                    "reward_ci95_25": reward_ci95_25,
                    "correctness": correct_fraction,
                    "correctness_rolling_25": correctness_rolling_25,
                    "correctness_trend_25": correctness_trend_25,
                    "correctness_ci95_25": correctness_ci95_25,
                    "correctness_used_mean": correctness_used_mean,
                    "valid_answer_fraction": valid_answer_fraction,
                    "valid_answer_rolling_25": valid_answer_rolling_25,
                    "terminated_fraction": terminated_fraction,
                    "terminated_rolling_25": terminated_rolling_25,
                    "terminated_correctness_mean": terminated_correctness_mean,
                    "clipped_fraction": clipped_fraction,
                    "completion_token_length_mean": completion_token_length_mean,
                    "effective_group_fraction": effective_group_fraction,
                    "effective_group_rolling_25": effective_group_rolling_25,
                    "informative_group_count": informative_group_count,
                    "all_wrong_group_fraction": all_wrong_group_fraction,
                    "all_correct_group_fraction": all_correct_group_fraction,
                    "adv_std": float(metrics["adv_std"]),
                    "clip_fraction": float(metrics["clip_fraction"]),
                    "up_clip_fraction": float(metrics.get("up_clip_fraction", 0.0)),
                    "up_clipped_probability_mean": float(metrics.get("up_clipped_probability_mean", 0.0)),
                    "kl": float(metrics["approx_kl"]),
                    "ref_kl": float(metrics.get("ref_kl", 0.0)),
                    "selected_token_probability_mean": float(metrics.get("selected_token_probability_mean", 0.0)),
                    "selected_token_logprob_mean": float(metrics.get("selected_token_logprob_mean", 0.0)),
                    "overlong_penalty_mean": float(reward_summary.get("reward/overlong_penalty_mean", 0.0)),
                    "grad_norm": grad_norm_value,
                    "rollout_time_s": rollout_dt,
                    "train_time_s": train_dt,
                    "total_time_s": rollout_dt + rescore_dt + ref_rescore_dt + train_dt,
                    "learning_rate": float(cfg.grpo_lr),
                    "updates_per_rollout": float(updates_per_rollout),
                    "optimizer_updates_total": float(global_step * updates_per_rollout),
                }
                if abs(float(cfg.lambda_novelty)) > 0.0:
                    compact_metrics.update({
                        "novelty_reward_mean": novelty_reward_mean,
                        "novelty_raw_mean": novelty_raw_mean,
                        "novelty_used_mean": novelty_used_mean,
                        "terminated_novelty_mean": terminated_novelty_mean,
                    })
                if _env_flag("TMX_WANDB_MINIMAL", "1"):
                    wandb_metrics = {
                        key: value
                        for key, value in compact_metrics.items()
                        if key in _WANDB_MINIMAL_PHASE_METRICS
                    }
                else:
                    wandb_metrics = compact_metrics
                for key, value in wandb_metrics.items():
                    _wandb_add_metric_pair(payload, grpo_metric_prefix, key, value)
                if cfg.dynamic_sampling and not _env_flag("TMX_WANDB_MINIMAL", "1"):
                    payload.update({
                        f"{grpo_metric_prefix}/dynamic/candidate_group_count": float(dynamic_stats.get("candidate_group_count", group_count)),
                        f"{grpo_metric_prefix}/dynamic/kept_group_count": float(dynamic_stats.get("kept_group_count", informative_group_count)),
                        f"{grpo_metric_prefix}/dynamic/discarded_group_count": float(dynamic_stats.get("discarded_group_count", 0.0)),
                        f"{grpo_metric_prefix}/dynamic/fallback_group_count": float(dynamic_stats.get("fallback_group_count", 0.0)),
                        f"{grpo_metric_prefix}/dynamic/selected_effective_group_fraction": float(dynamic_stats.get("selected_effective_group_fraction", effective_group_fraction)),
                    })
                if not _env_flag("TMX_WANDB_COMPACT", "1"):
                    payload.update({f"{grpo_metric_prefix}/{k}": float(v) for k, v in metrics.items()})
                    payload.update({
                        f"{grpo_metric_prefix}/reward_mean": reward_mean,
                        f"{grpo_metric_prefix}/reward_std": reward_std,
                        f"{grpo_metric_prefix}/reward_min": reward_min,
                        f"{grpo_metric_prefix}/reward_max": reward_max,
                        f"{grpo_metric_prefix}/correct_count": correct_count,
                        f"{grpo_metric_prefix}/completion_token_length_mean": completion_token_length_mean,
                        f"{grpo_metric_prefix}/completion_token_length_max": completion_token_length_max,
                        f"{grpo_metric_prefix}/terminated_count": terminated_count,
                        f"{grpo_metric_prefix}/terminated_final_mean": terminated_final_mean,
                        f"{grpo_metric_prefix}/terminated_correctness_mean": terminated_correctness_mean,
                        f"{grpo_metric_prefix}/terminated_novelty_mean": terminated_novelty_mean,
                    })
                    for key, value in reward_summary.items():
                        payload[f"{grpo_metric_prefix}/{key}"] = float(value)
                    for key, value in diversity_metrics.items():
                        payload[f"{grpo_metric_prefix}/{key}"] = float(value)
                else:
                    for key, value in diversity_metrics.items():
                        if key in ("diversity/dist_3", "diversity/dist_4"):
                            payload[f"{grpo_metric_prefix}/{key}"] = float(value)
                            payload[key.replace("diversity/", "")] = float(value)
                wandb.log(
                    payload,
                    **_wandb_log_kwargs(global_step, grpo_metric_prefix),
                )
            if is_source_host:
                _append_training_metrics_jsonl(
                    cfg,
                    {
                        "stage": grpo_metric_prefix,
                        "train_metric_prefix": train_metric_prefix,
                        "step": int(global_step),
                        "max_steps": int(cfg.grpo_max_steps),
                        "loss": float(metrics["loss"]),
                        "approx_kl": float(metrics["approx_kl"]),
                        "clip_fraction": float(metrics["clip_fraction"]),
                        "mean_ratio": float(metrics.get("mean_ratio", 0.0)),
                        "old_new_logprob_abs_mean": float(metrics.get("old_new_logprob_abs_mean", 0.0)),
                        "old_new_logprob_abs_max": float(metrics.get("old_new_logprob_abs_max", 0.0)),
                        "rescore_old_logprob_abs_mean": float(metrics.get("rescore_old_logprob_abs_mean", 0.0)),
                        "rescore_old_logprob_abs_max": float(metrics.get("rescore_old_logprob_abs_max", 0.0)),
                        "adv_mean": float(metrics.get("adv_mean", 0.0)),
                        "adv_std": float(metrics.get("adv_std", 0.0)),
                        "grad_norm": grad_norm_value,
                        "learning_rate": float(cfg.grpo_lr),
                        "rollout_time_s": float(rollout_dt),
                        "rescore_time_s": float(rescore_dt),
                        "ref_rescore_time_s": float(ref_rescore_dt),
                        "train_time_s": float(train_dt),
                        "total_time_s": float(rollout_dt + rescore_dt + ref_rescore_dt + train_dt),
                        "reward_mean": reward_mean,
                        "reward": reward_mean,
                        "reward_rolling_25": reward_rolling_25,
                        "reward_trend_25": reward_trend_25,
                        "reward_ci95_25": reward_ci95_25,
                        "reward_std": reward_std,
                        "reward_min": reward_min,
                        "reward_max": reward_max,
                        "correct_fraction": correct_fraction,
                        "correctness": correct_fraction,
                        "correctness_rolling_25": correctness_rolling_25,
                        "correctness_trend_25": correctness_trend_25,
                        "correctness_ci95_25": correctness_ci95_25,
                        "correct_count": correct_count,
                        "prompt_any_correct_fraction": prompt_any_correct_fraction,
                        "group_count": group_count,
                        "informative_group_count": informative_group_count,
                        "effective_group_fraction": effective_group_fraction,
                        "effective_group_rolling_25": effective_group_rolling_25,
                        "all_wrong_group_fraction": all_wrong_group_fraction,
                        "all_correct_group_fraction": all_correct_group_fraction,
                        "dynamic_sampling": {
                            "enabled": bool(cfg.dynamic_sampling),
                            "candidate_group_count": float(dynamic_stats.get("candidate_group_count", group_count)),
                            "kept_group_count": float(dynamic_stats.get("kept_group_count", informative_group_count)),
                            "discarded_group_count": float(dynamic_stats.get("discarded_group_count", 0.0)),
                            "fallback_group_count": float(dynamic_stats.get("fallback_group_count", 0.0)),
                            "selected_effective_group_fraction": float(dynamic_stats.get("selected_effective_group_fraction", effective_group_fraction)),
                        },
                        "valid_answer_fraction": valid_answer_fraction,
                        "valid_answer_rolling_25": valid_answer_rolling_25,
                        "terminated_fraction": terminated_fraction,
                        "terminated_rolling_25": terminated_rolling_25,
                        "terminated_count": terminated_count,
                        "terminated_final_mean": terminated_final_mean,
                        "terminated_correctness_mean": terminated_correctness_mean,
                        "terminated_novelty_mean": terminated_novelty_mean,
                        "clipped_fraction": clipped_fraction,
                        "completion_token_length_mean": completion_token_length_mean,
                        "completion_token_length_max": completion_token_length_max,
                        "reward_summary": reward_summary,
                        "diversity_metrics": diversity_metrics,
                        "updates_per_rollout": int(updates_per_rollout),
                        "optimizer_updates_total": int(global_step * updates_per_rollout),
                    },
                )
            if is_source_host:
                external_gap_mean = float(metrics.get("rescore_old_logprob_abs_mean", 0.0))
                external_gap_max = float(metrics.get("rescore_old_logprob_abs_max", 0.0))
                print(
                    f"[train] step {global_step}/{cfg.grpo_max_steps}  "
                    f"loss={float(metrics['loss']):.4f}  "
                    f"reward={reward_mean:.3f}  "
                    f"reward25={reward_rolling_25:.3f}  "
                    f"reward_trend25={reward_trend_25:+.3f}  "
                    f"eff_groups={effective_group_fraction:.2f}  "
                    f"clip={float(metrics['clip_fraction']):.3f}  "
                    f"kl={float(metrics['approx_kl']):.2e}  "
                    f"logp_gap_mean={float(metrics['old_new_logprob_abs_mean']):.3e}  "
                    f"logp_gap_max={float(metrics['old_new_logprob_abs_max']):.3e}  "
                    f"external_gap_mean={external_gap_mean:.3e}  "
                    f"external_gap_max={external_gap_max:.3e}  "
                    f"grad={grad_norm_value:.3f}  "
                    f"rollout={rollout_dt:.1f}s  "
                    f"rescore={rescore_dt:.1f}s  "
                    f"ref_rescore={ref_rescore_dt:.1f}s  "
                    f"train={train_dt:.1f}s",
                    flush=True,
                )

            should_upload_ckpt = _step_due(
                global_step,
                gcs_checkpoint_every_steps,
                gcs_checkpoint_first_step,
            )
            should_save_local_ckpt = (
                _step_due(global_step, local_checkpoint_every_steps)
                or should_upload_ckpt
            )
            if should_save_local_ckpt:
                _save_checkpoint(state, cfg, global_step, rnd_state=rnd_map, upload_to_gcs=should_upload_ckpt)
                if use_wandb:
                    import wandb
                    ckpt_payload = {
                        "checkpoint_step": float(global_step),
                        "checkpoint_uploaded_to_gcs": float(should_upload_ckpt),
                        f"{grpo_metric_prefix}/checkpoint_step": float(global_step),
                        f"{grpo_metric_prefix}/checkpoint_uploaded_to_gcs": float(should_upload_ckpt),
                    }
                    if not _env_flag("TMX_WANDB_COMPACT", "1"):
                        ckpt_payload[f"{grpo_metric_prefix}/checkpoint_local_path"] = os.path.abspath(
                            os.path.join(cfg.output_dir, f"step_{global_step:06d}")
                        )
                        ckpt_payload[f"{grpo_metric_prefix}/checkpoint_gcs_base"] = (
                            os.environ.get("TMX_GCS_CHECKPOINT_BASE") or ""
                        ).strip()
                    wandb.log(ckpt_payload, **_wandb_log_kwargs(global_step, grpo_metric_prefix))
            if _step_due(global_step, reload_every_steps, reload_first_step):
                _maybe_export_and_reload_vllm(state, cfg, m_cfg, global_step, reason="checkpoint")
                if is_source_host:
                    assert tokenizer is not None
                    _maybe_run_aime_probe(
                        cfg=cfg,
                        tokenizer=tokenizer,
                        server_urls=server_urls,
                        step=global_step,
                        stage=grpo_metric_prefix,
                        use_wandb=use_wandb,
                    )
                if jax.process_count() > 1:
                    multihost_utils.sync_global_devices(f"vllm_reload_done_{global_step}")
                if prefetch_rollouts and global_step < cfg.grpo_max_steps:
                    pending_rollout = _start_rollout_prefetch()

        if use_wandb and is_source_host and wandb_mod is not None:
            try:
                wandb_mod.finish()
            except Exception:
                pass
        if prefetch_pool is not None:
            prefetch_pool.shutdown(wait=False, cancel_futures=True)
        force_final_checkpoint = _env_flag("TMX_FORCE_FINAL_PHASE_CHECKPOINT", "0")
        if is_source_host and (
            force_final_checkpoint
            or gcs_checkpoint_every_steps <= 0
            or (global_step % gcs_checkpoint_every_steps) != 0
        ):
            # Phase-end checkpoints must be durable even if the step is not on
            # the regular GCS cadence; downstream SFT/Actual export depends on
            # this exact final state.
            old_hf_async = os.environ.get("TMX_HF_MIRROR_ASYNC")
            if force_final_checkpoint and _env_flag("TMX_HF_MIRROR_CHECKPOINTS", "0"):
                # Multi-round handoff restores immediately after this call; do
                # not leave the final checkpoint upload racing in the background.
                os.environ["TMX_HF_MIRROR_ASYNC"] = "0"
            try:
                _save_checkpoint(state, cfg, global_step, rnd_state=rnd_map, upload_to_gcs=True)
            finally:
                if old_hf_async is None:
                    os.environ.pop("TMX_HF_MIRROR_ASYNC", None)
                else:
                    os.environ["TMX_HF_MIRROR_ASYNC"] = old_hf_async
    print("[train] done.", flush=True)
    return state


def _remove_tree(path: str) -> None:
    try:
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.exists(path):
            os.remove(path)
    except Exception as e:
        print(f"[ckpt] WARN cleanup failed for {path}: {e}", flush=True)


def _json_safe(value):
    """Convert JAX/NumPy scalars and arrays into JSON-serializable values."""
    try:
        import numpy as _np
        if isinstance(value, _np.ndarray):
            if value.ndim == 0:
                return _json_safe(value.item())
            return value.tolist()
        if isinstance(value, _np.generic):
            return value.item()
    except Exception:
        pass
    try:
        import jax
        return _json_safe(jax.device_get(value))
    except Exception:
        pass
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return float(value)
    except Exception:
        return str(value)


def _append_training_metrics_jsonl(cfg: TrainConfig, record: Dict[str, Any]) -> None:
    """Persist per-step training metrics locally so W&B is not the only record."""
    try:
        path = os.path.join(cfg.output_dir, "training_metrics.jsonl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(_json_safe(record), sort_keys=True) + "\n")
    except Exception as e:
        print(f"[metrics] WARN metrics jsonl append failed: {e}", flush=True)


def _free_bytes_for(path: str) -> int:
    probe = os.path.abspath(path)
    while probe and not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    try:
        return int(shutil.disk_usage(probe or "/").free)
    except Exception:
        return 0


def _run_root_for_output(output_dir: str) -> str:
    """Return the top-level run directory for nested stage outputs."""
    out = os.path.abspath(output_dir.rstrip("/"))
    parts = out.split(os.sep)
    for idx, part in enumerate(parts):
        if part == "runs" and idx + 1 < len(parts):
            prefix = os.sep if out.startswith(os.sep) else ""
            return prefix + os.path.join(*parts[1 : idx + 2] if prefix else parts[: idx + 2])
    return out


def _cleanup_orbax_temps(root: str) -> None:
    if not root or not os.path.exists(root):
        return
    for cur, dirs, _files in os.walk(root, topdown=True):
        remove = [d for d in dirs if d.endswith(".orbax-checkpoint-tmp")]
        for name in remove:
            path = os.path.join(cur, name)
            print(f"[ckpt] removing stale temp checkpoint {path}", flush=True)
            _remove_tree(path)
        dirs[:] = [d for d in dirs if d not in remove]


def _step_number_from_name(name: str) -> Optional[int]:
    if not name.startswith("step_"):
        return None
    try:
        return int(name.split("_", 1)[1])
    except Exception:
        return None


def _prune_step_dirs_in_parent(parent: str, *, keep_last: int) -> None:
    if keep_last < 0 or not os.path.isdir(parent):
        return
    entries = []
    try:
        for name in os.listdir(parent):
            step = _step_number_from_name(name)
            if step is None:
                continue
            path = os.path.join(parent, name)
            if os.path.isdir(path):
                entries.append((step, path))
    except OSError:
        return
    entries.sort()
    stale = entries if keep_last == 0 else entries[:-keep_last]
    for _step, path in stale:
        print(f"[ckpt] pruning local step dir {path}", flush=True)
        _remove_tree(path)
        _remove_tree(path + ".orbax-checkpoint-tmp")


def _prune_step_dirs_under_run(cfg: TrainConfig, *, keep_last: int) -> None:
    """Keep only the newest local checkpoint per stage directory under a run."""
    run_root = _run_root_for_output(cfg.output_dir)
    if not os.path.isdir(run_root):
        return
    for cur, dirs, _files in os.walk(run_root, topdown=True):
        # Do not traverse hidden/tool cache directories; they are handled below.
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        if any(_step_number_from_name(d) is not None for d in dirs):
            _prune_step_dirs_in_parent(cur, keep_last=keep_last)
            dirs[:] = [d for d in dirs if _step_number_from_name(d) is None]


def _prune_vllm_reload_dirs(cfg: TrainConfig, *, keep_last: int) -> None:
    run_root = _run_root_for_output(cfg.output_dir)
    if not os.path.isdir(run_root):
        return
    for cur, dirs, _files in os.walk(run_root, topdown=True):
        if os.path.basename(cur) == "vllm_reload_hf":
            _prune_step_dirs_in_parent(cur, keep_last=keep_last)
            dirs[:] = []


def _cleanup_cache_paths(*, aggressive: bool) -> None:
    cache_paths = [
        os.path.expanduser("~/.cache/pip"),
        os.path.expanduser("~/miniforge3/pkgs"),
    ]
    if aggressive or os.environ.get("TMX_CKPT_CLEAR_WANDB_CACHE", "1") not in ("0", "false", "False", "no"):
        cache_paths.extend(
            [
                os.path.expanduser("~/wandb"),
                os.path.expanduser("~/tmx-jax/wandb"),
                os.path.expanduser("~/two-model-exploration/wandb"),
            ]
        )
    for cache in cache_paths:
        if os.path.exists(cache):
            print(f"[ckpt] clearing cache to free space: {cache}", flush=True)
            _remove_tree(cache)
            if cache.endswith(("pip", "pkgs")):
                os.makedirs(cache, exist_ok=True)
    hf_tmp_root = os.path.expanduser("~/.cache/huggingface/hub")
    if os.path.isdir(hf_tmp_root):
        for name in os.listdir(hf_tmp_root):
            if name.startswith("tmp") or name.endswith(".lock"):
                _remove_tree(os.path.join(hf_tmp_root, name))
    if aggressive and os.environ.get("TMX_CKPT_CLEAR_HF_CACHE_ON_LOW_SPACE", "0") not in (
        "0",
        "false",
        "False",
        "no",
    ):
        for path in (
            os.path.expanduser("~/.cache/huggingface/hub"),
            os.path.expanduser("~/.cache/huggingface/xet"),
        ):
            if os.path.exists(path):
                print(f"[ckpt] aggressively clearing HF cache: {path}", flush=True)
                _remove_tree(path)
    if aggressive and os.environ.get("TMX_CKPT_CLEAR_JAX_CACHE_ON_LOW_SPACE", "1") not in (
        "0",
        "false",
        "False",
        "no",
    ):
        jax_cache = os.environ.get(
            "JAX_COMPILATION_CACHE_DIR",
            os.path.expanduser("~/.cache/tmx_jax_compilation_cache"),
        )
        if os.path.exists(jax_cache):
            print(f"[ckpt] aggressively clearing JAX compilation cache: {jax_cache}", flush=True)
            _remove_tree(jax_cache)


def _cleanup_old_run_dirs(cfg: TrainConfig, min_free_bytes: int) -> None:
    if os.environ.get("TMX_CKPT_DELETE_OLD_RUNS", "0") in ("0", "false", "False", "no"):
        return
    current = _run_root_for_output(cfg.output_dir)
    runs_root = os.path.dirname(current)
    if not os.path.isdir(runs_root):
        return
    candidates = []
    for name in os.listdir(runs_root):
        path = os.path.abspath(os.path.join(runs_root, name))
        if path == current or not os.path.isdir(path):
            continue
        try:
            candidates.append((os.path.getmtime(path), path))
        except OSError:
            continue
    for _mtime, path in sorted(candidates):
        if _free_bytes_for(runs_root) >= min_free_bytes:
            break
        print(f"[ckpt] deleting old inactive run dir to free space: {path}", flush=True)
        _remove_tree(path)


def _checkpoint_space_janitor(cfg: TrainConfig, ckpt_dir: str, *, aggressive: bool = False) -> None:
    min_gb = float(os.environ.get("TMX_CKPT_MIN_FREE_GB", "24") or "24")
    min_free = int(min_gb * (1024 ** 3))
    keep_last = int(os.environ.get("TMX_LOCAL_CHECKPOINT_KEEP_LAST", "1") or "1")
    keep_reload = int(os.environ.get("TMX_LOCAL_VLLM_RELOAD_KEEP_LAST", "0") or "0")
    run_root = _run_root_for_output(cfg.output_dir)
    roots = {
        os.path.abspath(cfg.output_dir),
        os.path.dirname(os.path.abspath(cfg.output_dir)),
        run_root,
        os.path.expanduser("~/two-model-exploration/runs"),
        os.path.expanduser("~/tmx-jax/runs"),
    }
    for root in sorted(roots):
        _cleanup_orbax_temps(root)
    if aggressive or _free_bytes_for(ckpt_dir) < min_free:
        _prune_local_checkpoints(cfg, keep_last)
        _prune_step_dirs_under_run(cfg, keep_last=keep_last)
        _prune_vllm_reload_dirs(cfg, keep_last=keep_reload)
        _cleanup_cache_paths(aggressive=aggressive)
        _cleanup_old_run_dirs(cfg, min_free)
    free_gb = _free_bytes_for(ckpt_dir) / (1024 ** 3)
    print(f"[ckpt] free disk before save: {free_gb:.1f} GiB", flush=True)


def _ensure_space_for_vllm_export(cfg: TrainConfig, local_dir: str) -> None:
    _checkpoint_space_janitor(cfg, local_dir, aggressive=False)
    min_gb = float(os.environ.get("TMX_VLLM_EXPORT_MIN_FREE_GB", "28") or "28")
    if _free_bytes_for(local_dir) < int(min_gb * (1024 ** 3)):
        _checkpoint_space_janitor(cfg, local_dir, aggressive=True)


def _save_checkpoint_payload(ckpt_dir: str, payload: Dict[str, Any]) -> None:
    import orbax.checkpoint as ocp
    _remove_tree(ckpt_dir + ".orbax-checkpoint-tmp")
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(ckpt_dir, payload)
    # Orbax/TensorStore may finish file commits asynchronously. Do not let the
    # next checkpoint/janitor pass touch temporary OCDBT dirs while commits are
    # still in flight.
    if hasattr(checkpointer, "wait_until_finished"):
        checkpointer.wait_until_finished()


def _checkpoint_host_leaf(value: Any) -> Any:
    if isinstance(value, jax.Array):
        try:
            return np.asarray(jax.device_get(value))
        except Exception:
            if hasattr(value, "addressable_data"):
                return np.asarray(jax.device_get(value.addressable_data(0)))
            raise
    if isinstance(value, np.generic):
        return value.item()
    return value


def _checkpoint_safe_host_tree(tree: Any) -> Any:
    return jax.tree_util.tree_map(_checkpoint_host_leaf, tree)


def _gcs_checkpoint_target(ckpt_dir: str, cfg: TrainConfig) -> str:
    gcs_base = os.environ.get("TMX_GCS_CHECKPOINT_BASE", "").strip()
    if not gcs_base:
        gcs_base = "gs://two-model-exploration-checkpoints/checkpoints"
    run_name = os.path.basename(os.path.abspath(cfg.output_dir.rstrip("/")))
    gcs_base = gcs_base.rstrip("/")
    if os.path.basename(gcs_base) == run_name:
        return f"{gcs_base}/{os.path.basename(ckpt_dir)}"
    return f"{gcs_base}/{run_name}/{os.path.basename(ckpt_dir)}"


def _prune_local_checkpoints(cfg: TrainConfig, keep_last: int) -> None:
    if keep_last <= 0:
        return
    out = os.path.abspath(cfg.output_dir)
    try:
        entries = []
        for name in os.listdir(out):
            if not name.startswith("step_"):
                continue
            path = os.path.join(out, name)
            if not os.path.isdir(path):
                continue
            try:
                step = int(name.split("_", 1)[1])
            except Exception:
                continue
            entries.append((step, path))
        entries.sort()
        stale = entries[:-keep_last]
        for _step, path in stale:
            print(f"[ckpt] pruning old local checkpoint {path}", flush=True)
            _remove_tree(path)
            _remove_tree(path + ".orbax-checkpoint-tmp")
    except Exception as e:
        print(f"[ckpt] WARN local checkpoint prune failed: {e}", flush=True)


def _save_checkpoint(state, cfg: TrainConfig, step: int, rnd_state=None, *, upload_to_gcs: bool = True):
    ckpt_dir = os.path.abspath(os.path.join(cfg.output_dir, f"step_{step:06d}"))
    payload = {"params": state.params, "step": int(step)}
    save_opt_state = os.environ.get("TMX_SAVE_OPT_STATE", "0") not in ("0", "false", "False")
    keep_last = int(os.environ.get("TMX_LOCAL_CHECKPOINT_KEEP_LAST", "1") or "1")
    if save_opt_state:
        payload["opt_state"] = state.opt_state
    if rnd_state is not None:
        payload["rnd_state"] = _checkpoint_safe_host_tree(rnd_state)
    direct_gcs = (
        upload_to_gcs
        and _env_flag("TMX_ALLOW_GCS_ARTIFACTS", "0")
        and jax.process_count() > 1
        and _env_flag("TMX_ORBAX_DIRECT_GCS_CHECKPOINT", "0")
    )
    if jax.process_count() > 1 and not upload_to_gcs and _env_flag("TMX_SKIP_MULTIHOST_LOCAL_ONLY_CHECKPOINT", "1"):
        print(
            "[ckpt] skipping local-only multi-host checkpoint; "
            "non-shared per-worker filesystems can deadlock Orbax finalization",
            flush=True,
        )
        return
    _checkpoint_space_janitor(cfg, ckpt_dir)
    if direct_gcs:
        gcs_target = _gcs_checkpoint_target(ckpt_dir, cfg)
        try:
            print(f"[ckpt] saving shared GCS checkpoint {gcs_target}", flush=True)
            _save_checkpoint_payload(gcs_target, payload)
            print(f"[ckpt] saved shared GCS checkpoint {gcs_target}", flush=True)
            if _env_flag("TMX_HF_MIRROR_CHECKPOINTS", "0"):
                _maybe_upload_gcs_uri_to_hf(
                    gcs_target,
                    cfg,
                    f"{_hf_run_prefix(cfg)}/{_stage_name_for_output(cfg)}/checkpoints/{os.path.basename(ckpt_dir)}",
                    is_dir=True,
                )
            _maybe_upload_trajectory_snapshot(cfg, step)
            _maybe_upload_training_metrics_snapshot(cfg, step)
            _prune_local_checkpoints(cfg, keep_last)
            return
        except Exception as e:
            print(f"[ckpt] WARN shared GCS checkpoint save failed: {e}", flush=True)
            if os.environ.get("TMX_REQUIRE_GCS_CHECKPOINT_UPLOAD", "1") not in ("0", "false", "False", "no"):
                raise
            _maybe_upload_trajectory_snapshot(cfg, step)
            _maybe_upload_training_metrics_snapshot(cfg, step)
            return
    try:
        _save_checkpoint_payload(ckpt_dir, payload)
        print(f"[ckpt] saved {ckpt_dir}", flush=True)
        if upload_to_gcs and not _env_flag("TMX_ALLOW_GCS_ARTIFACTS", "0"):
            if _env_flag("TMX_HF_MIRROR_CHECKPOINTS", "0"):
                _maybe_upload_path_to_hf(
                    ckpt_dir,
                    cfg,
                    f"{_hf_run_prefix(cfg)}/{_stage_name_for_output(cfg)}/checkpoints/{os.path.basename(ckpt_dir)}",
                    is_dir=True,
                )
            _maybe_upload_trajectory_snapshot(cfg, step)
            _maybe_upload_training_metrics_snapshot(cfg, step)
            _prune_local_checkpoints(cfg, keep_last)
            return
        if upload_to_gcs:
            ok = _maybe_upload_checkpoint_to_gcs(ckpt_dir, cfg)
            if not ok and os.environ.get("TMX_REQUIRE_GCS_CHECKPOINT_UPLOAD", "1") not in ("0", "false", "False", "no"):
                raise RuntimeError(f"GCS checkpoint upload failed for {ckpt_dir}")
            _maybe_upload_trajectory_snapshot(cfg, step)
            _maybe_upload_training_metrics_snapshot(cfg, step)
        else:
            print(f"[ckpt] local-only checkpoint; GCS upload deferred for step {step}", flush=True)
        _prune_local_checkpoints(cfg, keep_last)
    except Exception as e:
        print(f"[ckpt] WARN save failed: {e}", flush=True)
        _remove_tree(ckpt_dir)
        _remove_tree(ckpt_dir + ".orbax-checkpoint-tmp")
        _checkpoint_space_janitor(cfg, ckpt_dir, aggressive=True)
        retry_payload = {"params": state.params, "step": int(step)}
        if rnd_state is not None:
            retry_payload["rnd_state"] = _checkpoint_safe_host_tree(rnd_state)
        try:
            print("[ckpt] retrying params/RND-only checkpoint after cleanup", flush=True)
            _save_checkpoint_payload(ckpt_dir, retry_payload)
            print(f"[ckpt] saved {ckpt_dir} after cleanup", flush=True)
            if upload_to_gcs:
                ok = _maybe_upload_checkpoint_to_gcs(ckpt_dir, cfg)
                if not ok and os.environ.get("TMX_REQUIRE_GCS_CHECKPOINT_UPLOAD", "1") not in ("0", "false", "False", "no"):
                    raise RuntimeError(f"GCS checkpoint upload failed for {ckpt_dir} after cleanup")
                _maybe_upload_trajectory_snapshot(cfg, step)
                _maybe_upload_training_metrics_snapshot(cfg, step)
            else:
                print(f"[ckpt] local-only checkpoint; GCS upload deferred for step {step}", flush=True)
            _prune_local_checkpoints(cfg, keep_last)
            return
        except Exception as retry_e:
            print(f"[ckpt] WARN retry save failed: {retry_e}", flush=True)
        if upload_to_gcs:
            _maybe_upload_trajectory_snapshot(cfg, step)
            _maybe_upload_training_metrics_snapshot(cfg, step)


def _maybe_upload_checkpoint_to_gcs(ckpt_dir: str, cfg: TrainConfig) -> bool:
    gcs_target = _gcs_checkpoint_target(ckpt_dir, cfg)
    max_attempts = int(os.environ.get("TMX_GCS_UPLOAD_MAX_ATTEMPTS", "5") or "5")
    retry_seconds = float(os.environ.get("TMX_GCS_UPLOAD_RETRY_SECONDS", "30") or "30")
    attempt = 0
    while max_attempts <= 0 or attempt < max_attempts:
        attempt += 1
        try:
            ret = subprocess.run(
                ["gsutil", "-m", "cp", "-r", ckpt_dir, gcs_target],
                capture_output=True,
                text=True,
                timeout=1800,
            )
        except subprocess.TimeoutExpired as exc:
            print(
                f"[ckpt] WARN gcs upload timed out attempt={attempt}/{max_attempts or 'inf'}: {exc}",
                flush=True,
            )
            time.sleep(retry_seconds)
            continue
        except Exception as exc:
            print(
                f"[ckpt] WARN gcs upload failed attempt={attempt}/{max_attempts or 'inf'}: {type(exc).__name__}: {exc}",
                flush=True,
            )
            time.sleep(retry_seconds)
            continue
        if ret.returncode == 0:
            print(f"[ckpt] uploaded {ckpt_dir} -> {gcs_target}", flush=True)
            if _env_flag("TMX_HF_MIRROR_CHECKPOINTS", "0"):
                _maybe_upload_gcs_uri_to_hf(
                    gcs_target,
                    cfg,
                    f"{_hf_run_prefix(cfg)}/{_stage_name_for_output(cfg)}/checkpoints/{os.path.basename(ckpt_dir)}",
                    is_dir=True,
                )
            return True
        print(
            f"[ckpt] WARN gcs upload failed attempt={attempt}/{max_attempts or 'inf'}: {ret.stderr[:500]}",
            flush=True,
        )
        time.sleep(retry_seconds)
    return False


def _hf_artifact_repo() -> str:
    return (
        os.environ.get("TMX_HF_CHECKPOINT_REPO")
        or os.environ.get("TMX_HF_ARTIFACT_REPO")
        or ""
    ).strip()


def _hf_run_prefix(cfg: TrainConfig) -> str:
    default = os.path.basename(os.path.abspath(cfg.output_dir.rstrip("/")))
    return (os.environ.get("TMX_HF_ARTIFACT_PREFIX") or default).strip().strip("/")


_HF_LOCAL_UPLOAD_CODE = r"""
import os
import traceback

from huggingface_hub import HfApi

repo_id = os.environ["TMX_HF_UPLOAD_REPO_ID"]
repo_type = os.environ.get("TMX_HF_UPLOAD_REPO_TYPE") or "dataset"
local_path = os.environ["TMX_HF_UPLOAD_LOCAL_PATH"]
path_in_repo = os.environ["TMX_HF_UPLOAD_PATH_IN_REPO"].strip("/")
is_dir = os.environ.get("TMX_HF_UPLOAD_IS_DIR", "0") == "1"

print(f"[hf-child] upload local={local_path} -> {repo_id}/{path_in_repo}", flush=True)
try:
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=True, exist_ok=True)
    if is_dir:
        api.upload_folder(
            repo_id=repo_id,
            repo_type=repo_type,
            folder_path=local_path,
            path_in_repo=path_in_repo,
            commit_message=f"mirror {path_in_repo}",
        )
    else:
        api.upload_file(
            repo_id=repo_id,
            repo_type=repo_type,
            path_or_fileobj=local_path,
            path_in_repo=path_in_repo,
            commit_message=f"mirror {path_in_repo}",
        )
    print("[hf-child] done", flush=True)
except Exception:
    traceback.print_exc()
    raise
"""


_HF_GCS_UPLOAD_CODE = r"""
import os
import subprocess
import tempfile
import traceback
from pathlib import Path

from huggingface_hub import HfApi

repo_id = os.environ["TMX_HF_UPLOAD_REPO_ID"]
repo_type = os.environ.get("TMX_HF_UPLOAD_REPO_TYPE") or "dataset"
gcs_uri = os.environ["TMX_HF_UPLOAD_GCS_URI"].rstrip("/")
path_in_repo = os.environ["TMX_HF_UPLOAD_PATH_IN_REPO"].strip("/")
is_dir = os.environ.get("TMX_HF_UPLOAD_IS_DIR", "0") == "1"

print(f"[hf-child] mirror gcs={gcs_uri} -> {repo_id}/{path_in_repo}", flush=True)
try:
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tmx_hf_gcs_") as tmp:
        root = Path(tmp)
        if is_dir:
            local = root / "artifact"
            local.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["gcloud", "storage", "cp", "--recursive", gcs_uri + "/*", str(local) + "/"],
                check=True,
                text=True,
            )
            api.upload_folder(
                repo_id=repo_id,
                repo_type=repo_type,
                folder_path=str(local),
                path_in_repo=path_in_repo,
                commit_message=f"mirror {path_in_repo}",
            )
        else:
            local = root / (Path(path_in_repo).name or "artifact")
            subprocess.run(
                ["gcloud", "storage", "cp", gcs_uri, str(local)],
                check=True,
                text=True,
            )
            api.upload_file(
                repo_id=repo_id,
                repo_type=repo_type,
                path_or_fileobj=str(local),
                path_in_repo=path_in_repo,
                commit_message=f"mirror {path_in_repo}",
            )
    print("[hf-child] done", flush=True)
except Exception:
    traceback.print_exc()
    raise
"""


def _hf_upload_log_path(cfg: TrainConfig, path_in_repo: str, source: str) -> str:
    log_dir = os.environ.get("TMX_HF_UPLOAD_LOG_DIR", "").strip()
    if not log_dir:
        log_dir = os.path.join(cfg.output_dir, "hf_upload_logs")
    os.makedirs(log_dir, exist_ok=True)
    basename = os.path.basename(path_in_repo.rstrip("/")) or "artifact"
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in basename)[:80]
    digest = hashlib.sha1(f"{source}:{path_in_repo}".encode("utf-8")).hexdigest()[:12]
    return os.path.join(log_dir, f"{int(time.time())}_{digest}_{safe}.log")


def _run_hf_upload_child(env: Dict[str, str], code: str, cfg: TrainConfig, path_in_repo: str, source: str) -> bool:
    require = str(env.get("TMX_REQUIRE_HF_ARTIFACT_UPLOAD", os.environ.get("TMX_REQUIRE_HF_ARTIFACT_UPLOAD", "0"))).lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    async_upload = str(env.get("TMX_HF_MIRROR_ASYNC", os.environ.get("TMX_HF_MIRROR_ASYNC", "1"))).lower() not in (
        "0",
        "false",
        "no",
        "off",
    ) and not require
    log_path = _hf_upload_log_path(cfg, path_in_repo, source)
    if async_upload:
        try:
            with open(log_path, "ab") as log_f:
                subprocess.Popen(
                    [sys.executable, "-c", code],
                    env=env,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            print(f"[hf] scheduled background upload -> {env['TMX_HF_UPLOAD_REPO_ID']}/{path_in_repo} log={log_path}", flush=True)
            return True
        except Exception as exc:
            print(f"[hf] WARN failed to schedule upload for {path_in_repo}: {type(exc).__name__}: {exc}", flush=True)
            if require:
                raise
            return False
    ret = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=float(os.environ.get("TMX_HF_UPLOAD_TIMEOUT_SECONDS", "7200") or "7200"),
    )
    if ret.returncode == 0:
        print(f"[hf] uploaded -> {env['TMX_HF_UPLOAD_REPO_ID']}/{path_in_repo}", flush=True)
        return True
    print(f"[hf] WARN upload failed for {path_in_repo}: {(ret.stderr or ret.stdout)[-1000:]}", flush=True)
    if require:
        raise RuntimeError(f"HF upload failed for {path_in_repo}")
    return False


def _maybe_upload_path_to_hf(
    local_path: str,
    cfg: TrainConfig,
    path_in_repo: str,
    *,
    is_dir: bool,
    force_sync: bool = False,
) -> bool:
    repo_id = _hf_artifact_repo()
    if not repo_id or not os.path.exists(local_path):
        return False
    env = os.environ.copy()
    env.update(
        {
            "TMX_HF_UPLOAD_REPO_ID": repo_id,
            "TMX_HF_UPLOAD_REPO_TYPE": "dataset",
            "TMX_HF_UPLOAD_LOCAL_PATH": os.path.abspath(local_path),
            "TMX_HF_UPLOAD_PATH_IN_REPO": path_in_repo.strip("/"),
            "TMX_HF_UPLOAD_IS_DIR": "1" if is_dir else "0",
        }
    )
    if force_sync:
        env["TMX_HF_MIRROR_ASYNC"] = "0"
    return _run_hf_upload_child(env, _HF_LOCAL_UPLOAD_CODE, cfg, path_in_repo, os.path.abspath(local_path))


def _maybe_upload_gcs_uri_to_hf(
    gcs_uri: str,
    cfg: TrainConfig,
    path_in_repo: str,
    *,
    is_dir: bool,
    force_sync: bool = False,
) -> bool:
    repo_id = _hf_artifact_repo()
    if not repo_id or not gcs_uri:
        return False
    env = os.environ.copy()
    env.update(
        {
            "TMX_HF_UPLOAD_REPO_ID": repo_id,
            "TMX_HF_UPLOAD_REPO_TYPE": "dataset",
            "TMX_HF_UPLOAD_GCS_URI": gcs_uri,
            "TMX_HF_UPLOAD_PATH_IN_REPO": path_in_repo.strip("/"),
            "TMX_HF_UPLOAD_IS_DIR": "1" if is_dir else "0",
        }
    )
    if force_sync:
        env["TMX_HF_MIRROR_ASYNC"] = "0"
    return _run_hf_upload_child(env, _HF_GCS_UPLOAD_CODE, cfg, path_in_repo, gcs_uri)


def _maybe_upload_trajectory_snapshot(cfg: TrainConfig, step: int) -> None:
    traj = os.path.join(cfg.output_dir, "explorer_trajectories.jsonl")
    if not os.path.exists(traj):
        return
    if not _env_flag("TMX_ALLOW_GCS_ARTIFACTS", "0"):
        if _env_flag("TMX_HF_MIRROR_TRAJECTORIES", "0"):
            target_names = (
                f"explorer_trajectories_step_{step:06d}.jsonl",
                "explorer_trajectories_latest.jsonl",
                f"rollouts_step_{step:06d}.jsonl",
                "rollouts_latest.jsonl",
            )
            for target_name in target_names:
                _maybe_upload_path_to_hf(
                    traj,
                    cfg,
                    f"{_hf_run_prefix(cfg)}/{_stage_name_for_output(cfg)}/trajectories/{target_name}",
                    is_dir=False,
                )
        return
    gcs_base = os.environ.get("TMX_GCS_CHECKPOINT_BASE", "").strip()
    if not gcs_base:
        gcs_base = "gs://two-model-exploration-checkpoints/checkpoints"
    run_name = os.path.basename(os.path.abspath(cfg.output_dir.rstrip("/")))
    gcs_base = gcs_base.rstrip("/")
    if os.path.basename(gcs_base) == run_name:
        gcs_dir = f"{gcs_base}/trajectories"
    else:
        gcs_dir = f"{gcs_base}/{run_name}/trajectories"
    try:
        target_names = (
            f"explorer_trajectories_step_{step:06d}.jsonl",
            "explorer_trajectories_latest.jsonl",
            f"rollouts_step_{step:06d}.jsonl",
            "rollouts_latest.jsonl",
        )
        for target_name in target_names:
            ret = subprocess.run(
                ["gsutil", "cp", traj, f"{gcs_dir}/{target_name}"],
                capture_output=True,
                text=True,
                timeout=900,
            )
            if ret.returncode != 0:
                print(f"[ckpt] WARN trajectory upload failed: {ret.stderr[:300]}", flush=True)
                return
            if _env_flag("TMX_HF_MIRROR_TRAJECTORIES", "0"):
                _maybe_upload_gcs_uri_to_hf(
                    f"{gcs_dir}/{target_name}",
                    cfg,
                    f"{_hf_run_prefix(cfg)}/{_stage_name_for_output(cfg)}/trajectories/{target_name}",
                    is_dir=False,
                )
        all_rollouts = os.path.join(cfg.output_dir, "rollouts_all.jsonl")
        if os.path.exists(all_rollouts):
            for target_name in (
                f"rollouts_all_step_{step:06d}.jsonl",
                "rollouts_all_latest.jsonl",
            ):
                ret = subprocess.run(
                    ["gsutil", "cp", all_rollouts, f"{gcs_dir}/{target_name}"],
                    capture_output=True,
                    text=True,
                    timeout=900,
                )
                if ret.returncode != 0:
                    print(f"[ckpt] WARN all-rollouts upload failed: {ret.stderr[:300]}", flush=True)
                    return
                if _env_flag("TMX_HF_MIRROR_TRAJECTORIES", "0"):
                    _maybe_upload_gcs_uri_to_hf(
                        f"{gcs_dir}/{target_name}",
                        cfg,
                        f"{_hf_run_prefix(cfg)}/{_stage_name_for_output(cfg)}/trajectories/{target_name}",
                        is_dir=False,
                    )
        print(f"[ckpt] uploaded trajectory snapshot -> {gcs_dir}", flush=True)
    except Exception as e:
        print(f"[ckpt] WARN trajectory upload error: {e}", flush=True)


def _maybe_upload_training_metrics_snapshot(cfg: TrainConfig, step: int) -> None:
    metrics = os.path.join(cfg.output_dir, "training_metrics.jsonl")
    if not os.path.exists(metrics):
        return
    if not _env_flag("TMX_ALLOW_GCS_ARTIFACTS", "0"):
        if _env_flag("TMX_HF_MIRROR_METRICS", "0"):
            for target_name in (
                f"training_metrics_step_{step:06d}.jsonl",
                "training_metrics_latest.jsonl",
            ):
                _maybe_upload_path_to_hf(
                    metrics,
                    cfg,
                    f"{_hf_run_prefix(cfg)}/{_stage_name_for_output(cfg)}/metrics/{target_name}",
                    is_dir=False,
                )
        return
    gcs_base = os.environ.get("TMX_GCS_CHECKPOINT_BASE", "").strip()
    if not gcs_base:
        gcs_base = "gs://two-model-exploration-checkpoints/checkpoints"
    run_name = os.path.basename(os.path.abspath(cfg.output_dir.rstrip("/")))
    gcs_base = gcs_base.rstrip("/")
    if os.path.basename(gcs_base) == run_name:
        gcs_dir = f"{gcs_base}/metrics"
    else:
        gcs_dir = f"{gcs_base}/{run_name}/metrics"
    for target_name in (
        f"training_metrics_step_{step:06d}.jsonl",
        "training_metrics_latest.jsonl",
    ):
        try:
            ret = subprocess.run(
                ["gsutil", "cp", metrics, f"{gcs_dir}/{target_name}"],
                capture_output=True,
                text=True,
                timeout=900,
            )
            if ret.returncode != 0:
                print(f"[ckpt] WARN metrics upload failed: {ret.stderr[:300]}", flush=True)
                return
            if _env_flag("TMX_HF_MIRROR_METRICS", "0"):
                _maybe_upload_gcs_uri_to_hf(
                    f"{gcs_dir}/{target_name}",
                    cfg,
                    f"{_hf_run_prefix(cfg)}/{_stage_name_for_output(cfg)}/metrics/{target_name}",
                    is_dir=False,
                )
        except Exception as e:
            print(f"[ckpt] WARN metrics upload error: {e}", flush=True)
            return
    print(f"[ckpt] uploaded training metrics snapshot -> {gcs_dir}", flush=True)


if __name__ == "__main__":
    run_training(parse_args())
