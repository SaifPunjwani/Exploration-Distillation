import os
import json
import time
import gc
import re
import subprocess
from dataclasses import replace

import torch

from .config import RunConfig
from .devices import maybe_import_xla, resolve_device
from .data import (
    build_math_dataset_bundle,
    build_math_eval_dataset_bundle,
    build_round_shards,
    build_supervised_dataset_from_trajectories,
    build_rewarded_dataset_from_trajectories,
    choose_actual_bootstrap_trajectories,
    choose_actual_policy_trajectories,
    select_accepted_trajectories,
)
from .explorer import build_novelty_scorer, train_explorer_with_grpo, generate_offline_trajectories
from .distill import train_actual_model, train_actual_with_offline_grpo
from .eval_plot import evaluate_model_on_gsm8k, evaluate_model_with_rollouts, plot_results
from .novelty_eval import run_embedding_novelty_analysis
from .prompting import (
    generation_answer_prefix,
    get_answer_token_constraints,
    render_generation_prompt,
    resolved_max_completion_len,
)
from .reward import is_correct_completion, normalize_generated_completion
from .artifacts import (
    gcs_artifacts_enabled,
    hf_artifact_prefix,
    hf_repo_id,
    hf_repo_type,
    upload_path_to_hf,
)
from .utils import (
    causal_lm_from_pretrained,
    hf_from_pretrained_kwargs,
    patch_accelerate_gather_object_for_xla,
    patch_transformers_nested_xla_mesh_reduce_for_single_worker,
    patch_transformers_trainer_logging_for_single_worker,
    patch_torch_autocast_enabled_signature,
    patch_torch_checkpoint_autocast_for_xla,
    patch_torch_xla_device_module_for_rng,
    _register_tpu_flash_attention,
    patch_torch_isin_for_xla,
    patch_transformers_attention_mask_for_xla,
    patch_transformers_isin_for_xla,
    patch_transformers_logits_processor_for_xla,
    patch_transformers_stopping_criteria_for_xla,
    set_seed,
    xla_safe_generate,
)
from .wandb_metrics import filter_wandb_payload


def _save_model_to_gcs_and_reload_vllm(model, tokenizer, run_name: str, role: str = "actual"):
    """Save model to Hugging Face and optionally reload a vLLM server to serve it.

    GCS writes are disabled by default. Set TMX_ALLOW_GCS_ARTIFACTS=1 and
    TMX_ARTIFACT_BACKEND=dual/gcs only for legacy migration or emergency use.
    """
    xm_mod, xr_mod = maybe_import_xla()
    xla_global_ordinal = 0
    xla_world_size = 1
    if os.environ.get("PJRT_DEVICE", "").upper() == "TPU" and xr_mod is not None:
        try:
            xla_global_ordinal = int(xr_mod.global_ordinal())
        except Exception:
            xla_global_ordinal = 0
        try:
            xla_world_size = int(xr_mod.world_size())
        except Exception:
            xla_world_size = 1
    xla_world_size = max(1, xla_world_size)
    is_multi_rank_tpu = xla_world_size > 1 and xm_mod is not None
    barrier_tag = f"tmx_ckpt_{_artifact_safe_name(run_name)}_{str(role).lower()}"

    def _xla_ckpt_barrier(stage: str) -> None:
        if is_multi_rank_tpu:
            xm_mod.rendezvous(f"{barrier_tag}_{stage}")

    _xla_ckpt_barrier("enter")
    if is_multi_rank_tpu and xla_global_ordinal != 0:
        print(
            f"[ckpt] Skipping {role} model export/reload on non-master TPU rank "
            f"{xla_global_ordinal}/{xla_world_size}."
        )
        _xla_ckpt_barrier("done")
        return None

    gcs_base = os.environ.get("TMX_GCS_CHECKPOINT_BASE", "").strip()
    gcs_model_path = f"{gcs_base.rstrip('/')}/{run_name}/{role}" if gcs_artifacts_enabled() and gcs_base else None
    hf_path_in_repo = f"{hf_artifact_prefix(run_name)}/{role}"
    hf_model_uri = None
    local_tmp = f"/tmp/tmx_model_save_{_artifact_safe_name(run_name)}_{str(role).lower()}_{os.getpid()}"

    try:
        print(f"[ckpt] Saving {role} model for HF artifact {hf_repo_id()}/{hf_path_in_repo}...")
        os.makedirs(local_tmp, exist_ok=True)
        model_cpu = model.to(torch.device("cpu"))
        model_cpu.eval()
        try:
            model_cpu.save_pretrained(local_tmp, safe_serialization=True)
        except TypeError:
            model_cpu.save_pretrained(local_tmp)
        tokenizer.save_pretrained(local_tmp)

        hf_model_uri = upload_path_to_hf(local_tmp, hf_path_in_repo, is_dir=True)
        if hf_model_uri:
            print(f"[ckpt] Saved {role} model to {hf_model_uri}")

        if gcs_model_path:
            try:
                ret = subprocess.run(
                    ["gsutil", "-m", "cp", "-r", f"{local_tmp}/*", f"{gcs_model_path}/"],
                    capture_output=True, text=True, timeout=300,
                )
                if ret.returncode == 0:
                    print(f"[ckpt] Saved legacy GCS copy of {role} model to {gcs_model_path}")
                else:
                    print(f"[ckpt] WARNING: legacy gsutil upload failed: {ret.stderr[:200]}")
                    gcs_model_path = None
            except Exception as e:
                print(f"[ckpt] WARNING: legacy gsutil upload error: {e}")
                gcs_model_path = None

        if not hf_model_uri and not gcs_model_path:
            print("[ckpt] WARNING: no durable artifact upload succeeded")
            return None

        common_reload_env = {}
        for env_key in (
            "PORT",
            "TP_SIZE",
            "MAX_MODEL_LEN",
            "MAX_NUM_SEQS",
            "MAX_NUM_BATCHED_TOKENS",
            "VLLM_VENV_DIR",
        ):
            value = os.environ.get(env_key, "").strip()
            if value:
                common_reload_env[env_key] = value

        if role == "explorer":
            reload_tpu_name = os.environ.get("TMX_EXPLORER_VLLM_RELOAD_TPU_NAME", "").strip()
            reload_zone = os.environ.get("TMX_EXPLORER_VLLM_RELOAD_ZONE", "").strip()
            reload_workers = os.environ.get("TMX_EXPLORER_VLLM_RELOAD_WORKERS", "all").strip() or "all"
            # Default to the client's current vLLM model name (typically the base
            # model id, e.g. Qwen/Qwen3-1.7B) so reload doesn't swap the name out
            # from under the training client and start returning 404s. Override
            # with TMX_EXPLORER_VLLM_MODEL_NAME if you deliberately want a new id.
            _client_current = os.environ.get("TMX_VLLM_MODEL_NAME", "").strip()
            served_model_name = (
                os.environ.get("TMX_EXPLORER_VLLM_MODEL_NAME", "").strip()
                or _client_current
                or "explorer_model"
            )
            reload_succeeded = False
            if reload_tpu_name and reload_zone:
                print(
                    f"[ckpt] Reloading Explorer vLLM slice on {reload_tpu_name} "
                    f"(workers={reload_workers}) from {hf_model_uri or gcs_model_path}..."
                )
                reload_env = os.environ.copy()
                reload_env.update(
                    {
                        "GCS_MODEL_PATH": gcs_model_path or "",
                        "HF_REPO_ID": hf_repo_id() if hf_model_uri else "",
                        "HF_REPO_TYPE": hf_repo_type(),
                        "HF_PATH_IN_REPO": hf_path_in_repo if hf_model_uri else "",
                        "TPU_NAME": reload_tpu_name,
                        "ZONE": reload_zone,
                        "WORKERS": reload_workers,
                        "SERVED_MODEL_NAME": served_model_name,
                        "LOCAL_MODEL_DIR_BASENAME": "vllm_explorer_model",
                    }
                )
                reload_env.update(common_reload_env)
                try:
                    ret = subprocess.run(
                        ["bash", "scripts/reload_vllm_slice_with_model.sh"],
                        capture_output=True,
                        text=True,
                        timeout=3600,
                        env=reload_env,
                    )
                    if ret.returncode == 0:
                        reload_succeeded = True
                        print("[ckpt] Reloaded Explorer vLLM slice for trajectory collection.")
                    else:
                        stderr = (ret.stderr or "").strip()
                        stdout = (ret.stdout or "").strip()
                        print(
                            "[ckpt] WARNING: Explorer vLLM slice reload failed: "
                            f"{stderr[:200] or stdout[:200]}"
                        )
                except Exception as e:
                    print(f"[ckpt] WARNING: Explorer vLLM reload error: {e}")
            else:
                print(
                    "[ckpt] Explorer vLLM slice reload is not configured; "
                    "trajectory collection will stay on the local model path."
                )

            if reload_succeeded:
                os.environ["TMX_TRAJ_USE_VLLM"] = "1"
                os.environ["TMX_VLLM_MODEL_NAME"] = served_model_name
                print(
                    f"[ckpt] Enabled TPU-backed vLLM trajectory collection with "
                    f"TMX_VLLM_MODEL_NAME={served_model_name}"
                )

        # Reload a dedicated vLLM worker if requested so Actual eval can stay on the
        # fast remote serving path instead of falling back to local CPU generation.
        vllm_reload_url = os.environ.get("TMX_VLLM_RELOAD_WORKER_URL", "").strip()
        if role == "actual" and vllm_reload_url:
            reload_tpu_name = os.environ.get("TMX_VLLM_RELOAD_TPU_NAME", "").strip()
            reload_zone = os.environ.get("TMX_VLLM_RELOAD_ZONE", "").strip()
            reload_worker = os.environ.get("TMX_VLLM_RELOAD_WORKER", "0").strip() or "0"
            served_model_name = os.environ.get("TMX_ACTUAL_VLLM_MODEL_NAME", "actual_model").strip() or "actual_model"
            reload_succeeded = False
            if reload_tpu_name and reload_zone:
                print(
                    f"[ckpt] Reloading vLLM worker {reload_worker} on {reload_tpu_name} "
                    f"from {hf_model_uri or gcs_model_path}..."
                )
                reload_env = os.environ.copy()
                reload_env.update(
                    {
                        "GCS_MODEL_PATH": gcs_model_path or "",
                        "HF_REPO_ID": hf_repo_id() if hf_model_uri else "",
                        "HF_REPO_TYPE": hf_repo_type(),
                        "HF_PATH_IN_REPO": hf_path_in_repo if hf_model_uri else "",
                        "TPU_NAME": reload_tpu_name,
                        "ZONE": reload_zone,
                        "WORKER": reload_worker,
                        "SERVED_MODEL_NAME": served_model_name,
                        "LOCAL_MODEL_DIR_BASENAME": "vllm_actual_model",
                    }
                )
                reload_env.update(common_reload_env)
                try:
                    ret = subprocess.run(
                        ["bash", "scripts/reload_vllm_with_model.sh"],
                        capture_output=True,
                        text=True,
                        timeout=900,
                        env=reload_env,
                    )
                    if ret.returncode == 0:
                        reload_succeeded = True
                        print(f"[ckpt] Reloaded Actual vLLM worker: {vllm_reload_url}")
                    else:
                        stderr = (ret.stderr or "").strip()
                        stdout = (ret.stdout or "").strip()
                        print(
                            "[ckpt] WARNING: vLLM reload failed: "
                            f"{stderr[:200] or stdout[:200]}"
                        )
                except Exception as e:
                    print(f"[ckpt] WARNING: vLLM reload error: {e}")
            else:
                print(
                    "[ckpt] WARNING: TMX_VLLM_RELOAD_WORKER_URL set without "
                    "TMX_VLLM_RELOAD_TPU_NAME/TMX_VLLM_RELOAD_ZONE; skipping remote reload."
                )

            if reload_succeeded:
                os.environ["TMX_ACTUAL_VLLM_MODEL_PATH"] = hf_model_uri or gcs_model_path or ""
                os.environ["TMX_ACTUAL_VLLM_SERVER_URL"] = vllm_reload_url
                print(f"[ckpt] Set TMX_ACTUAL_VLLM_SERVER_URL={vllm_reload_url} for Actual eval")

        return hf_model_uri or gcs_model_path
    finally:
        _xla_ckpt_barrier("done")


def _wandb_enabled_for_this_process() -> bool:
    if os.environ.get("TMX_WANDB_MASTER_ONLY", "1") == "0":
        return True
    if os.environ.get("PJRT_DEVICE", "").upper() != "TPU":
        return True
    try:
        import torch_xla.runtime as xr  # type: ignore

        ordinal = int(xr.global_ordinal())
    except Exception:
        return True
    return ordinal == 0


def init_wandb(cfg: RunConfig):
    if not cfg.use_wandb:
        return None
    if not _wandb_enabled_for_this_process():
        print("[wandb] Disabled on non-master TPU process.")
        return None
    try:
        import wandb  # type: ignore
    except Exception as e:
        print(f"[WARN] wandb not available ({e}); disabling logging for this run.")
        return None

    def _ensure_runtime_dir(env_key: str, fallback_name: str) -> None:
        fallback_path = os.path.join(os.getcwd(), fallback_name)
        candidates = [os.environ.get(env_key), fallback_path]
        for candidate in candidates:
            if not candidate:
                continue
            try:
                os.makedirs(candidate, exist_ok=True)
            except OSError:
                continue
            os.environ[env_key] = candidate
            return
        raise RuntimeError(f"Could not create a writable directory for {env_key}.")

    os.environ.setdefault("WANDB_START_METHOD", "thread")
    os.environ.setdefault("WANDB__SERVICE_WAIT", "300")
    os.environ.setdefault("WANDB_DISABLE_CODE", "true")
    _ensure_runtime_dir("WANDB_CACHE_DIR", ".wandb_cache")
    _ensure_runtime_dir("WANDB_DATA_DIR", ".wandb_data")
    wandb_mode = os.environ.get("WANDB_MODE", "").strip().lower()
    api_key = os.environ.get("WANDB_API_KEY") or getattr(getattr(wandb, "api", None), "api_key", None)
    if wandb_mode != "offline" and not api_key:
        raise RuntimeError(
            "W&B logging requested but no API key is configured. "
            "Run `wandb login` or export WANDB_API_KEY before launching tracked runs."
        )

    tags_env = os.environ.get("WANDB_TAGS")
    env_tags = [t.strip() for t in tags_env.split(",") if t.strip()] if tags_env else []
    tags = list(
        dict.fromkeys(
            env_tags
            + [
                f"dataset={cfg.dataset_name}",
                f"benchmark={cfg.benchmark_dataset_name or 'none'}",
                f"pipeline={cfg.pipeline_mode}",
                f"device={cfg.device_preference}",
                f"novelty={cfg.novelty_metric}",
                f"anchor={cfg.anchor_metric}",
                f"rounds={cfg.num_rounds}",
            ]
        )
    )
    # Only rank 0 should initialize wandb in multi-process (FSDP) mode
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank != 0:
        return None
    try:
        wandb_run_id = os.environ.get("WANDB_RUN_ID") or None
        wandb_resume = os.environ.get("WANDB_RESUME") or None
        run = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.wandb_run_name,
            id=wandb_run_id,
            resume=wandb_resume,
            group=os.environ.get("WANDB_GROUP"),
            job_type=os.environ.get("WANDB_JOB_TYPE"),
            tags=tags or None,
            notes=os.environ.get("WANDB_NOTES"),
            config={**cfg.__dict__},
            reinit=True,
        )
    except Exception as e:
        raise RuntimeError(
            "Failed to initialize W&B. Verify the API key, entity/project names, and network access. "
            f"Underlying error: {e}"
        ) from e
    if run is None:
        return None

    try:
        run.define_metric("train/step")
        run.define_metric("train/*", step_metric="train/step")
        run.define_metric("grpo/step")
        run.define_metric("grpo/*", step_metric="grpo/step")
        run.define_metric("distill/step")
        run.define_metric("distill/*", step_metric="distill/step")
        run.define_metric("actual_grpo/step")
        run.define_metric("actual_grpo/*", step_metric="actual_grpo/step")
    except Exception:
        pass

    class _SafeWandbRun:
        def __init__(self, inner):
            self._inner = inner
            self.enabled = True

        def log(self, data, step=None):
            if not self.enabled:
                return
            filtered = filter_wandb_payload(dict(data))
            if not filtered:
                return
            try:
                if step is None:
                    self._inner.log(filtered)
                else:
                    self._inner.log(filtered, step=step)
            except Exception as e:
                self.enabled = False
                print(f"[WARN] wandb logging failed; disabling logging for this run. ({e})")

        def __getattr__(self, name):
            return getattr(self._inner, name)

    return _SafeWandbRun(run)


def _env_flag(name: str, default: str = "0") -> bool:
    value = str(os.environ.get(name, default) or default).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _empty_eval_metrics(reason: str) -> dict:
    return {
        "accuracy": 0.0,
        "primary_metric_name": "accuracy",
        "novelty_mean": 0.0,
        "novelty_values": [],
        "correct": 0,
        "total": 0,
        "id_accuracy": 0.0,
        "ood_accuracy": 0.0,
        "id_total": 0,
        "ood_total": 0,
        "id_novelty_mean": 0.0,
        "ood_novelty_mean": 0.0,
        "samples": [],
        "skipped": True,
        "skip_reason": reason,
    }


def _resolve_final_eval_roles() -> set[str]:
    if _env_flag("TMX_FINAL_EVAL_ONLY_ACTUAL", "0"):
        return {"actual"}
    raw = str(os.environ.get("TMX_FINAL_EVAL_ROLES", "") or "").strip().lower()
    if not raw:
        return {"base", "explorer", "actual"}
    roles = {part.strip() for part in raw.split(",") if part.strip()}
    allowed = {"base", "explorer", "actual"}
    resolved = roles & allowed
    return resolved or {"base", "explorer", "actual"}


def _resolve_final_eval_cfg(cfg: RunConfig) -> RunConfig:
    final_cfg = cfg
    max_eval_override = str(os.environ.get("TMX_FINAL_EVAL_MAX_EXAMPLES", "") or "").strip()
    if max_eval_override:
        try:
            parsed = max(0, int(max_eval_override))
            final_cfg = replace(final_cfg, max_eval_examples=parsed)
        except Exception:
            print(f"[WARN] Ignoring invalid TMX_FINAL_EVAL_MAX_EXAMPLES={max_eval_override!r}")

    max_benchmark_override = str(os.environ.get("TMX_FINAL_BENCHMARK_MAX_EXAMPLES", "") or "").strip()
    if max_benchmark_override:
        try:
            parsed = max(0, int(max_benchmark_override))
            final_cfg = replace(final_cfg, max_benchmark_examples=parsed)
        except Exception:
            print(f"[WARN] Ignoring invalid TMX_FINAL_BENCHMARK_MAX_EXAMPLES={max_benchmark_override!r}")
    return final_cfg


def _split_total_budget(total: int | None, num_rounds: int) -> list[int]:
    if total is None:
        return []
    total_int = max(0, int(total))
    rounds_int = max(1, int(num_rounds))
    base = total_int // rounds_int
    remainder = total_int % rounds_int
    return [base + (1 if idx < remainder else 0) for idx in range(rounds_int)]


def _round_budget_value(total: int | None, round_idx: int, num_rounds: int) -> int | None:
    if total is None:
        return None
    budgets = _split_total_budget(total, num_rounds)
    if round_idx < 1 or round_idx > len(budgets):
        raise IndexError(f"Round index {round_idx} is outside 1..{len(budgets)}")
    return budgets[round_idx - 1]


def _build_round_cfg(cfg: RunConfig, round_idx: int, shard_ds) -> tuple[RunConfig, dict]:
    round_root = os.path.join(os.path.dirname(cfg.plots_dir) or ".", f"round_{round_idx}")
    round_cfg = replace(
        cfg,
        plots_dir=os.path.join(round_root, "plots"),
        explorer_traj_path=os.path.join(round_root, f"explorer_trajectories_round{round_idx}.jsonl"),
        explorer_output_dir=(f"{cfg.explorer_output_dir}_round{round_idx}" if cfg.explorer_output_dir else f"explorer_round_{round_idx}"),
        max_train_examples=len(shard_ds),
    )
    budget_info = {
        "mode": str(getattr(cfg, "round_budget_mode", "constant_total") or "constant_total"),
        "train_examples": int(len(shard_ds)),
    }
    if str(getattr(cfg, "round_budget_mode", "constant_total") or "constant_total") != "constant_total":
        return round_cfg, budget_info

    round_cfg = replace(
        round_cfg,
        grpo_max_steps=_round_budget_value(cfg.grpo_max_steps, round_idx, cfg.num_rounds),
        max_distill_examples=_round_budget_value(cfg.max_distill_examples, round_idx, cfg.num_rounds),
        max_actual_examples=_round_budget_value(cfg.max_actual_examples, round_idx, cfg.num_rounds),
        wall_clock_budget_minutes=(
            float(cfg.wall_clock_budget_minutes) / max(1, int(cfg.num_rounds))
            if cfg.wall_clock_budget_minutes is not None
            else None
        ),
    )
    budget_info.update(
        {
            "grpo_max_steps": int(round_cfg.grpo_max_steps),
            "max_distill_examples": None if round_cfg.max_distill_examples is None else int(round_cfg.max_distill_examples),
            "max_actual_examples": None if round_cfg.max_actual_examples is None else int(round_cfg.max_actual_examples),
            "wall_clock_budget_minutes": (
                None if round_cfg.wall_clock_budget_minutes is None else float(round_cfg.wall_clock_budget_minutes)
            ),
        }
    )
    return round_cfg, budget_info


def _percentile(values, pct: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    if len(vals) == 1:
        return float(vals[0])
    pos = (pct / 100.0) * (len(vals) - 1)
    low = int(pos)
    high = min(low + 1, len(vals) - 1)
    weight = pos - low
    return float(vals[low] * (1 - weight) + vals[high] * weight)


def _aggregate_round_stage_timings(round_summaries):
    aggregated = {}
    for round_summary in round_summaries:
        stage_timings = round_summary.get("stage_timings", {}) or {}
        for key, value in stage_timings.items():
            try:
                aggregated[key] = float(aggregated.get(key, 0.0)) + float(value)
            except (TypeError, ValueError):
                continue
    return aggregated


def summarize_trajectories(trajectories):
    rewards = [t["reward"] for t in trajectories]
    novelties = [t["r_novel"] for t in trajectories]
    anchors = [float(t.get("r_anchor", 0.0)) for t in trajectories]
    format_quality = [float(t.get("format_quality", t.get("reward_format_quality", 0.0)) or 0.0) for t in trajectories]
    format_penalty = [float(t.get("format_penalty", t.get("reward_format_penalty", 0.0)) or 0.0) for t in trajectories]
    valid_answers = [1.0 if bool(t.get("valid_answer_presentation", False)) else 0.0 for t in trajectories]
    terminated = [1.0 if bool(t.get("terminated", False)) else 0.0 for t in trajectories]
    clipped = [1.0 if bool(t.get("clipped", False)) else 0.0 for t in trajectories]
    prompt_truncated = [1.0 if bool(t.get("prompt_truncated", False)) else 0.0 for t in trajectories]
    normalization_changed = [1.0 if bool(t.get("normalization_changed", False)) else 0.0 for t in trajectories]
    correct = [
        1.0 if (t.get("is_correct") if "is_correct" in t else (t["r_correct"] > 0.5)) else 0.0
        for t in trajectories
    ]

    bucket_counts = {}
    layer_accum = {}
    layer_count = {}
    for t in trajectories:
        bucket = str(t.get("reward_bucket", "unknown"))
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        layer_map = t.get("r_novel_layers", {}) or {}
        if isinstance(layer_map, dict):
            for key, value in layer_map.items():
                layer_accum[key] = layer_accum.get(key, 0.0) + float(value)
                layer_count[key] = layer_count.get(key, 0) + 1
    topic_stats = {}
    for t in trajectories:
        topic = str(t.get("topic", "unknown"))
        if topic not in topic_stats:
            topic_stats[topic] = {"count": 0, "correct": 0.0, "novelty_sum": 0.0}
        topic_stats[topic]["count"] += 1
        topic_stats[topic]["correct"] += 1.0 if bool(t.get("is_correct", False)) else 0.0
        topic_stats[topic]["novelty_sum"] += float(t.get("r_novel", 0.0))

    summary = {
        "traj/count": len(trajectories),
        "traj/correct_fraction": sum(correct) / max(1, len(correct)),
        "traj/reward_mean": float(sum(rewards) / max(1, len(rewards))) if rewards else 0.0,
        "traj/reward_p50": _percentile(rewards, 50.0),
        "traj/reward_p90": _percentile(rewards, 90.0),
        "traj/novelty_mean": float(sum(novelties) / max(1, len(novelties))) if novelties else 0.0,
        "traj/novelty_p50": _percentile(novelties, 50.0),
        "traj/novelty_p90": _percentile(novelties, 90.0),
        "traj/anchor_mean": float(sum(anchors) / max(1, len(anchors))) if anchors else 0.0,
        "traj/format_quality_mean": float(sum(format_quality) / max(1, len(format_quality))) if format_quality else 0.0,
        "traj/format_penalty_mean": float(sum(format_penalty) / max(1, len(format_penalty))) if format_penalty else 0.0,
        "traj/valid_answer_fraction": float(sum(valid_answers) / max(1, len(valid_answers))) if valid_answers else 0.0,
        "traj/terminated_fraction": float(sum(terminated) / max(1, len(terminated))) if terminated else 0.0,
        "traj/clipped_fraction": float(sum(clipped) / max(1, len(clipped))) if clipped else 0.0,
        "traj/prompt_truncated_fraction": float(sum(prompt_truncated) / max(1, len(prompt_truncated))) if prompt_truncated else 0.0,
        "traj/normalization_changed_fraction": float(sum(normalization_changed) / max(1, len(normalization_changed))) if normalization_changed else 0.0,
    }
    for bucket, count in bucket_counts.items():
        summary[f"traj/reward_bucket/{bucket}"] = float(count)
    for key in sorted(layer_accum.keys()):
        summary[f"traj/layer_novelty_mean/{key}"] = layer_accum[key] / max(1, layer_count.get(key, 1))
    for topic, stat in topic_stats.items():
        c = max(1, int(stat["count"]))
        summary[f"traj/topic/{topic}/count"] = float(stat["count"])
        summary[f"traj/topic/{topic}/correct_fraction"] = float(stat["correct"] / c)
        summary[f"traj/topic/{topic}/novelty_mean"] = float(stat["novelty_sum"] / c)
    return summary


def _mean(values):
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _truncate_text(text: str, max_chars: int = 320) -> str:
    value = (text or "").strip()
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3] + "..."


def _series_summary(values):
    if not values:
        return None
    vals = [float(v) for v in values]
    return {
        "count": len(vals),
        "start": float(vals[0]),
        "end": float(vals[-1]),
        "min": float(min(vals)),
        "max": float(max(vals)),
        "mean": _mean(vals),
        "delta": float(vals[-1] - vals[0]),
    }


def _compact_completion_record(record: dict):
    return {
        "prompt": _truncate_text(record.get("prompt", "")),
        "completion": _truncate_text(record.get("completion", "")),
        "completion_raw": _truncate_text(record.get("completion_full_raw", record.get("completion_raw", ""))),
        "completion_canonical": _truncate_text(record.get("completion_canonical", "")),
        "actual_target_canonical": _truncate_text(record.get("actual_target_canonical", "")),
        "actual_target_full_trace": _truncate_text(record.get("actual_target_full_trace", "")),
        "ground_truth": str(record.get("ground_truth", "")),
        "presented_answer": record.get("presented_answer"),
        "topic": str(record.get("topic", "unknown")),
        "is_correct": bool(record.get("is_correct", False)),
        "format_bucket": str(record.get("format_bucket", "unknown")),
        "parsed_final_answer": record.get("parsed_final_answer"),
        "prompt_token_length": int(record.get("prompt_token_length", 0) or 0),
        "completion_token_length": int(record.get("completion_token_length", 0) or 0),
        "char_length": int(record.get("char_length", 0) or 0),
        "terminated": bool(record.get("terminated", False)),
        "clipped": bool(record.get("clipped", False)),
        "prompt_truncated": bool(record.get("prompt_truncated", False)),
        "normalization_changed": bool(record.get("normalization_changed", False)),
        "has_final_answer_tag": bool(record.get("has_final_answer_tag", False)),
        "final_answer_on_last_line": bool(record.get("final_answer_on_last_line", False)),
        "valid_answer_presentation": bool(record.get("valid_answer_presentation", False)),
        "has_latex": bool(record.get("has_latex", False)),
        "has_multiple_choice": bool(record.get("has_multiple_choice", False)),
        "has_markdown": bool(record.get("has_markdown", False)),
        "mentions_instruction": bool(record.get("mentions_instruction", False)),
        "starts_with_instruction_echo": bool(record.get("starts_with_instruction_echo", False)),
        "format_quality": float(record.get("format_quality", record.get("reward_format_quality", 0.0)) or 0.0),
        "format_penalty": float(record.get("format_penalty", record.get("reward_format_penalty", 0.0)) or 0.0),
        "raw_format_quality": float(record.get("raw_format_quality", 0.0) or 0.0),
        "raw_format_penalty": float(record.get("raw_format_penalty", 0.0) or 0.0),
        "novelty": float(record.get("novelty", record.get("r_novel", 0.0)) or 0.0),
        "reward": float(record.get("reward", 0.0) or 0.0),
    }


def _top_records(records, *, predicate=None, sort_key=None, reverse=False, limit=3):
    chosen = [record for record in records if predicate(record)] if predicate is not None else list(records)
    if sort_key is not None:
        chosen = sorted(chosen, key=sort_key, reverse=reverse)
    return [_compact_completion_record(record) for record in chosen[:limit]]


def _summarize_completion_records(records, *, novelty_key: str, reward_key: str | None = None):
    rows = list(records or [])
    if not rows:
        return {"count": 0, "format_buckets": {}, "examples": {}}

    correct = [1.0 if bool(row.get("is_correct", False)) else 0.0 for row in rows]
    prompt_token_lengths = [float(row.get("prompt_token_length", 0) or 0.0) for row in rows]
    token_lengths = [float(row.get("completion_token_length", 0) or 0.0) for row in rows]
    char_lengths = [float(row.get("char_length", 0) or 0.0) for row in rows]
    novelty_values = [float(row.get(novelty_key, 0.0) or 0.0) for row in rows]
    quality_values = [float(row.get("format_quality", row.get("reward_format_quality", 0.0)) or 0.0) for row in rows]
    penalty_values = [float(row.get("format_penalty", row.get("reward_format_penalty", 0.0)) or 0.0) for row in rows]
    raw_quality_values = [float(row.get("raw_format_quality", 0.0) or 0.0) for row in rows]
    raw_penalty_values = [float(row.get("raw_format_penalty", 0.0) or 0.0) for row in rows]
    reward_values = [float(row.get(reward_key, 0.0) or 0.0) for row in rows] if reward_key else []
    answer_valid_values = [1.0 if bool(row.get("valid_answer_presentation", False)) else 0.0 for row in rows]

    flag_keys = [
        "terminated",
        "clipped",
        "prompt_truncated",
        "normalization_changed",
        "has_final_answer_tag",
        "prefix_only",
        "final_answer_on_last_line",
        "has_boxed",
        "has_latex",
        "has_markdown",
        "has_multiple_choice",
        "mentions_instruction",
        "starts_with_instruction_echo",
        "used_first_line",
    ]
    summary = {
        "count": len(rows),
        "correct_fraction": _mean(correct),
        "mean_prompt_token_length": _mean(prompt_token_lengths),
        "prompt_token_length_p50": _percentile(prompt_token_lengths, 50.0),
        "prompt_token_length_p90": _percentile(prompt_token_lengths, 90.0),
        "mean_completion_token_length": _mean(token_lengths),
        "completion_token_length_p50": _percentile(token_lengths, 50.0),
        "completion_token_length_p90": _percentile(token_lengths, 90.0),
        "mean_completion_char_length": _mean(char_lengths),
        "mean_novelty": _mean(novelty_values),
        "novelty_p50": _percentile(novelty_values, 50.0),
        "novelty_p90": _percentile(novelty_values, 90.0),
        "mean_format_quality": _mean(quality_values),
        "format_quality_p50": _percentile(quality_values, 50.0),
        "format_quality_p90": _percentile(quality_values, 90.0),
        "mean_format_penalty": _mean(penalty_values),
        "mean_raw_format_quality": _mean(raw_quality_values),
        "mean_raw_format_penalty": _mean(raw_penalty_values),
        "valid_answer_fraction": _mean(answer_valid_values),
    }
    if reward_values:
        summary["mean_reward"] = _mean(reward_values)
        summary["reward_p50"] = _percentile(reward_values, 50.0)
        summary["reward_p90"] = _percentile(reward_values, 90.0)

    for flag in flag_keys:
        summary[f"{flag}_fraction"] = _mean([1.0 if bool(row.get(flag, False)) else 0.0 for row in rows])

    bucket_stats = {}
    for row in rows:
        bucket = str(row.get("format_bucket", "unknown"))
        stats = bucket_stats.setdefault(bucket, {"count": 0, "correct": 0.0, "mean_novelty_sum": 0.0})
        stats["count"] += 1
        stats["correct"] += 1.0 if bool(row.get("is_correct", False)) else 0.0
        stats["mean_novelty_sum"] += float(row.get(novelty_key, 0.0) or 0.0)
    summary["format_buckets"] = {
        bucket: {
            "count": int(stats["count"]),
            "fraction": float(stats["count"] / max(1, len(rows))),
            "correct_fraction": float(stats["correct"] / max(1, stats["count"])),
            "mean_novelty": float(stats["mean_novelty_sum"] / max(1, stats["count"])),
        }
        for bucket, stats in sorted(bucket_stats.items(), key=lambda item: (-item[1]["count"], item[0]))
    }

    summary["examples"] = {
        "correct": _top_records(rows, predicate=lambda row: bool(row.get("is_correct", False)), limit=3),
        "incorrect": _top_records(rows, predicate=lambda row: not bool(row.get("is_correct", False)), limit=3),
        "multiple_choice": _top_records(
            rows,
            predicate=lambda row: bool(row.get("has_multiple_choice", False)),
            limit=3,
        ),
        "instruction_echo": _top_records(
            rows,
            predicate=lambda row: bool(row.get("mentions_instruction", False) or row.get("starts_with_instruction_echo", False)),
            limit=3,
        ),
        "prefix_only": _top_records(
            rows,
            predicate=lambda row: bool(row.get("prefix_only", False)),
            limit=3,
        ),
        "valid_answers": _top_records(
            rows,
            predicate=lambda row: bool(row.get("valid_answer_presentation", False)),
            limit=3,
        ),
        "high_quality": _top_records(
            rows,
            sort_key=lambda row: float(row.get("format_quality", row.get("reward_format_quality", 0.0)) or 0.0),
            reverse=True,
            limit=3,
        ),
        "top_novelty": _top_records(rows, sort_key=lambda row: float(row.get(novelty_key, 0.0) or 0.0), reverse=True, limit=3),
    }
    if reward_key:
        summary["examples"]["top_reward"] = _top_records(
            rows,
            sort_key=lambda row: float(row.get(reward_key, 0.0) or 0.0),
            reverse=True,
            limit=3,
        )
    return summary


def _summarize_log_history(log_history):
    rows = [row for row in (log_history or []) if isinstance(row, dict)]
    candidate_fields = [
        "loss",
        "reward",
        "reward_mean",
        "reward_std",
        "kl",
        "grad_norm",
        "learning_rate",
        "completions/clipped_ratio",
        "completions/mean_length",
        "batch_correct_frac",
        "batch_r_correct",
        "batch_r_novel",
        "wall_time_seconds",
    ]
    summary = {"num_rows": len(rows), "series": {}}
    for field in candidate_fields:
        values = []
        for row in rows:
            value = row.get(field)
            if isinstance(value, (int, float)):
                values.append(float(value))
        field_summary = _series_summary(values)
        if field_summary is not None:
            summary["series"][field] = field_summary
    return summary


def _build_loss_diagnostics(result: dict):
    return {
        "baseline": _summarize_log_history(result.get("baseline_log_hist", [])),
        "explorer": _summarize_log_history(result.get("explorer_log_hist", [])),
        "actual": _summarize_log_history(result.get("actual_log_hist", [])),
        "distill": {
            "num_epochs": len(result.get("distill_losses", [])),
            "loss_curve": [float(v) for v in result.get("distill_losses", [])],
            "summary": _series_summary(result.get("distill_losses", [])),
        },
    }


def _resolve_actual_training_gate(actual_bootstrap_info: dict, actual_policy_info: dict | None = None):
    bootstrap = dict(actual_bootstrap_info or {})
    accepted_count = int(bootstrap.get("accepted_count", 0) or 0)
    if accepted_count <= 0:
        return {
            "skip": True,
            "reason": "no_accepted_trajectories",
            "accepted_count": accepted_count,
        }

    if actual_policy_info is not None:
        positive_count = int((actual_policy_info or {}).get("positive_count", 0) or 0)
        if positive_count <= 0:
            return {
                "skip": True,
                "reason": "no_positive_policy_trajectories",
                "accepted_count": accepted_count,
                "positive_count": positive_count,
            }

    return {
        "skip": False,
        "reason": "ok",
        "accepted_count": accepted_count,
        "positive_count": None if actual_policy_info is None else int((actual_policy_info or {}).get("positive_count", 0) or 0),
    }


def _build_trajectory_audit_payload(cfg: RunConfig, result: dict):
    return {
        "config": {
            "actual_target_contract": cfg.actual_target_contract,
            "novelty_text_contract": cfg.novelty_text_contract,
            "novelty_feature_max_length": cfg.novelty_feature_max_length or cfg.max_total_len,
            "max_prompt_len": cfg.max_prompt_len,
            "max_completion_len": cfg.max_completion_len,
            "exploration_max_completion_len": resolved_max_completion_len(cfg, phase="explore"),
            "eval_max_completion_len": resolved_max_completion_len(cfg, phase="eval"),
            "max_total_len": cfg.max_total_len,
            "exploration_response_style": cfg.exploration_response_style,
            "eval_response_style": cfg.eval_response_style,
            "exploration_answer_constraint_mode": cfg.exploration_answer_constraint_mode,
            "eval_answer_constraint_mode": cfg.eval_answer_constraint_mode,
        },
        "prompt_diagnostics": result.get("prompt_diagnostics", {}),
        "accepted_trajectories": result.get("accepted_info", {}),
        "actual_training": result.get("actual_training_info", {}),
        "trajectory_generation": _summarize_completion_records(
            result.get("trajectories", []),
            novelty_key="r_novel",
            reward_key="reward",
        ),
        "actual_datasets": {
            "distill": result.get("distill_dataset_meta"),
            "offline_grpo": result.get("actual_dataset_meta"),
        },
        "novelty_feature_stats": result.get("novelty_feature_stats", {}),
        "eval": {
            "base": _summarize_completion_records(result["base_metrics"].get("samples", []), novelty_key="novelty"),
            "explorer": _summarize_completion_records(result["explorer_metrics"].get("samples", []), novelty_key="novelty"),
            "actual": _summarize_completion_records(result["actual_metrics"].get("samples", []), novelty_key="novelty"),
        },
    }


def _write_analysis_json(path: str, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _load_train_selection_manifest(path: str):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("train selection manifest must be a JSON object")
    return payload


def _apply_train_selection_manifest(train_ds, manifest: dict, manifest_path: str):
    if "__row_id__" not in train_ds.column_names:
        raise ValueError("train dataset does not expose __row_id__ required for manifest replay")

    selected_row_ids = manifest.get("selected_row_ids") or []
    if not selected_row_ids:
        selected_count = int(manifest.get("selected_count", 0) or 0)
        candidate_count = int(manifest.get("candidate_count", 0) or 0)
        if selected_count > 0 and selected_count >= len(train_ds) and candidate_count >= len(train_ds):
            selected_row_ids = [int(v) for v in train_ds["__row_id__"]]
    if not selected_row_ids:
        raise ValueError("train selection manifest does not contain selected_row_ids")

    row_id_to_index = {int(row_id): idx for idx, row_id in enumerate(train_ds["__row_id__"])}
    selected_indices = [row_id_to_index[int(row_id)] for row_id in selected_row_ids if int(row_id) in row_id_to_index]
    if not selected_indices:
        raise ValueError("train selection manifest did not match any current train rows")

    selected_ds = train_ds.select(selected_indices)
    loaded_manifest = dict(manifest)
    loaded_manifest["manifest_path"] = str(manifest_path)
    loaded_manifest["loaded_from_path"] = True
    loaded_manifest["used_probe"] = False
    loaded_manifest["candidate_count"] = int(len(train_ds))
    loaded_manifest["selected_count"] = int(len(selected_ds))
    loaded_manifest["missing_row_ids"] = [
        int(row_id) for row_id in selected_row_ids if int(row_id) not in row_id_to_index
    ]
    return selected_ds, loaded_manifest


def _topic_eval_slice(metrics: dict):
    id_acc = float(metrics.get("id_accuracy", 0.0))
    ood_acc = float(metrics.get("ood_accuracy", 0.0))
    id_nov = float(metrics.get("id_novelty_mean", 0.0))
    ood_nov = float(metrics.get("ood_novelty_mean", 0.0))
    return {
        "id_total": int(metrics.get("id_total", 0)),
        "ood_total": int(metrics.get("ood_total", 0)),
        "id_accuracy": id_acc,
        "ood_accuracy": ood_acc,
        "id_ood_accuracy_gap": float(id_acc - ood_acc),
        "id_novelty_mean": id_nov,
        "ood_novelty_mean": ood_nov,
        "id_ood_novelty_gap": float(id_nov - ood_nov),
    }


def _analysis_dir(cfg: RunConfig) -> str:
    return os.path.join(_run_output_root(cfg), "analysis")


def _run_output_root(cfg: RunConfig) -> str:
    plots_dir = getattr(cfg, "plots_dir", None)
    if plots_dir:
        normalized_plots = os.path.normpath(str(plots_dir))
        if normalized_plots not in {".", "plots"}:
            return os.path.abspath(os.path.dirname(str(plots_dir)) or ".")

    saved_model_root = getattr(cfg, "saved_model_root", None)
    if saved_model_root:
        return os.path.abspath(os.path.dirname(str(saved_model_root)) or ".")

    explorer_traj_path = getattr(cfg, "explorer_traj_path", None)
    if explorer_traj_path:
        normalized_traj = os.path.normpath(str(explorer_traj_path))
        if normalized_traj not in {".", "explorer_trajectories.jsonl"}:
            return os.path.abspath(os.path.dirname(str(explorer_traj_path)) or ".")

    manifest_path = getattr(cfg, "train_selection_manifest_path", None)
    if manifest_path:
        return os.path.abspath(os.path.dirname(str(manifest_path)) or ".")

    return os.path.abspath(".")


def _artifact_safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "").strip())
    cleaned = cleaned.strip("-.")
    return cleaned or "artifact"


def _git_metadata() -> dict:
    def _run(*args: str) -> str | None:
        try:
            proc = subprocess.run(
                args,
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception:
            return None
        value = proc.stdout.strip()
        return value or None

    return {
        "branch": _run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "commit": _run("git", "rev-parse", "HEAD"),
        "status_short": _run("git", "status", "--short"),
    }


def _benchmark_eval_cfg(cfg: RunConfig) -> RunConfig:
    if not cfg.benchmark_dataset_name:
        return cfg
    return replace(
        cfg,
        dataset_name=cfg.benchmark_dataset_name,
        dataset_config_name=cfg.benchmark_dataset_config_name,
        dataset_split_eval=cfg.benchmark_dataset_split_eval,
        dataset_split_policy=cfg.benchmark_dataset_split_policy,
        max_eval_examples=cfg.max_benchmark_examples or cfg.max_eval_examples,
        dataset_manifest_path=cfg.benchmark_manifest_path,
    )


def _benchmark_eval_is_sampled(cfg: RunConfig) -> bool:
    mode = str(getattr(cfg, "benchmark_eval_mode", "auto") or "auto").lower()
    if mode == "sampled":
        return True
    if mode == "greedy":
        return False
    return int(getattr(cfg, "benchmark_num_rollouts", 1) or 1) > 1


def _benchmark_primary_metric(metrics: dict | None) -> tuple[str, float]:
    payload = metrics or {}
    return (
        str(payload.get("primary_metric_name") or "accuracy"),
        float(payload.get("accuracy", 0.0) or 0.0),
    )


def _cleanup_runtime_memory(is_xla: bool, aggressive: bool = False):
    for _ in range(3 if aggressive else 1):
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not is_xla:
        return
    xm_mod, _ = maybe_import_xla()
    if xm_mod is None:
        return
    try:
        xm_mod.mark_step()
    except Exception:
        pass
    wait_ops = getattr(xm_mod, "wait_device_ops", None)
    if wait_ops is not None:
        try:
            wait_ops()
        except Exception:
            pass
    if aggressive:
        # Clear XLA compilation cache to free HBM held by cached programs
        try:
            import torch_xla._XLAC as _xlac
            if hasattr(_xlac, "_xla_clear_runtime_cache"):
                _xlac._xla_clear_runtime_cache()
            elif hasattr(_xlac, "_clear_xla_fallback_cache"):
                _xlac._clear_xla_fallback_cache()
        except Exception:
            pass
        gc.collect()
        try:
            xm_mod.mark_step()
        except Exception:
            pass
        import time as _time
        _time.sleep(2)


def _rehydrate_model_for_xla_generation(model, device, is_xla: bool, label: str):
    if model is None or not is_xla:
        return model
    cpu = torch.device("cpu")
    try:
        print(f"[patch] Rehydrating {label} through CPU before XLA generation.")
        model = model.to(cpu)
        _cleanup_runtime_memory(is_xla=is_xla)
        model = model.to(device)
        _cleanup_runtime_memory(is_xla=is_xla)
    except Exception as exc:
        print(f"[WARN] Failed to rehydrate {label} before XLA generation: {exc}")
    return model


def _offload_novelty_scorer_to_cpu(novelty_scorer):
    if novelty_scorer is None:
        return None
    cpu = torch.device("cpu")
    embed_model = getattr(novelty_scorer, "embed_model", None)
    if embed_model is not None:
        embed_model.to(cpu)
        embed_model.eval()
    for rnd in (getattr(novelty_scorer, "rnd_map", {}) or {}).values():
        try:
            rnd.to(cpu)
        except Exception:
            continue
        rnd.is_xla = False
    novelty_scorer.device = cpu
    novelty_scorer.use_input_embeddings = False
    return novelty_scorer


def _select_train_subset_by_probe(train_ds, cfg: RunConfig, device, is_xla: bool, use_wandb: bool = False, wandb_run=None):
    policy = str(getattr(cfg, "train_selection_policy", "random") or "random").lower()
    desired_count = max(1, int(getattr(cfg, "max_train_examples", len(train_ds)) or len(train_ds)))
    if policy != "base_solvable" or len(train_ds) <= desired_count:
        selected_row_ids = [int(v) for v in train_ds["__row_id__"]] if "__row_id__" in train_ds.column_names else []
        return train_ds, {
            "policy": policy,
            "candidate_count": int(len(train_ds)),
            "selected_count": int(min(len(train_ds), desired_count)),
            "solvable_count": 0,
            "used_probe": False,
            "desired_count": int(desired_count),
            "candidate_row_ids": list(selected_row_ids),
            "selected_row_ids": list(selected_row_ids),
            "records": [],
        }

    from transformers import AutoModelForCausalLM, AutoTokenizer

    candidate_count = len(train_ds)
    sample_limit = max(1, int(getattr(cfg, "train_probe_samples_per_prompt", 2) or 2))
    sample_temperature = float(getattr(cfg, "train_probe_temperature", 0.8) or 0.8)
    configured_probe_batch = getattr(cfg, "train_probe_batch_size", None)
    if configured_probe_batch is None:
        configured_probe_batch = max(
            int(getattr(cfg, "trajectory_batch_size", 1) or 1),
            int(getattr(cfg, "eval_batch_size", 1) or 1),
        )
    batch_size = max(1, int(configured_probe_batch or 1))
    max_new_tokens = int(resolved_max_completion_len(cfg, phase="explore"))

    print(
        "[train-selection] Probing candidate pool for solvable prompts: "
        f"policy={policy}, candidates={candidate_count}, desired={desired_count}, "
        f"samples_per_prompt={sample_limit}, temperature={sample_temperature}, "
        f"batch_size={batch_size}."
    )

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        **hf_from_pretrained_kwargs(cfg.model_name, repo_type="model"),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = causal_lm_from_pretrained(cfg.model_name, is_xla=is_xla).to(device)
    model.eval()
    answer_constraints = get_answer_token_constraints(tokenizer, cfg, phase="explore")
    answer_prefix = generation_answer_prefix(cfg, phase="explore")

    flags = []
    probe_records = []
    eos_id = tokenizer.eos_token_id
    if isinstance(eos_id, (list, tuple)):
        eos_id = eos_id[0] if eos_id else None
    if torch.is_tensor(eos_id):
        eos_id = eos_id.flatten()[0].item() if eos_id.numel() > 0 else None

    try:
        for batch_index, start in enumerate(range(0, len(train_ds), batch_size), start=1):
            stop = min(len(train_ds), start + batch_size)
            batch = train_ds.select(range(start, stop))
            prompts = list(batch["prompt"])
            ground_truths = list(batch["ground_truth"])
            row_ids = list(batch["__row_id__"]) if "__row_id__" in batch.column_names else list(range(start, stop))
            rendered_prompts = [render_generation_prompt(tokenizer, prompt, cfg, phase="explore") for prompt in prompts]
            enc = tokenizer(
                rendered_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=cfg.max_prompt_len,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            prompt_width = int(enc["input_ids"].shape[1])

            batch_best = [
                {
                    "row_id": int(row_id),
                    "solvable": False,
                    "correct_samples": 0,
                    "best_completion": "",
                    "ground_truth": gt,
                }
                for row_id, gt in zip(row_ids, ground_truths)
            ]

            with torch.no_grad():
                for _ in range(sample_limit):
                    if answer_constraints is not None or device.type == "xla":
                        out_ids = xla_safe_generate(
                            model,
                            input_ids=enc["input_ids"],
                            attention_mask=enc["attention_mask"],
                            max_new_tokens=max_new_tokens,
                            do_sample=True,
                            temperature=sample_temperature,
                            top_p=0.9,
                            eos_token_id=eos_id,
                            pad_token_id=tokenizer.pad_token_id,
                            allowed_token_ids=(answer_constraints or {}).get("allowed_token_ids"),
                            initial_token_ids=(answer_constraints or {}).get("initial_token_ids"),
                        )
                    else:
                        out_ids = model.generate(
                            **enc,
                            max_new_tokens=max_new_tokens,
                            do_sample=True,
                            temperature=sample_temperature,
                            top_p=0.9,
                            eos_token_id=eos_id,
                            pad_token_id=tokenizer.pad_token_id,
                        )

                    decoded = [
                        tokenizer.decode(out_ids[i][prompt_width:], skip_special_tokens=True)
                        for i in range(out_ids.shape[0])
                    ]
                    for i, raw_completion in enumerate(decoded):
                        normalized = normalize_generated_completion(raw_completion, answer_prefix=answer_prefix)
                        scored = str(normalized.get("completion_scored", raw_completion))
                        is_correct = bool(is_correct_completion(scored, ground_truths[i]))
                        if is_correct:
                            batch_best[i]["solvable"] = True
                            batch_best[i]["correct_samples"] += 1
                            batch_best[i]["best_completion"] = scored
                        elif not batch_best[i]["best_completion"]:
                            batch_best[i]["best_completion"] = scored

            flags.extend(batch_best)
            if batch_index % 4 == 0 or stop >= len(train_ds):
                solved_so_far = sum(1 for row in flags if row["solvable"])
                print(
                    "[train-selection] Probe progress: "
                    f"{stop}/{len(train_ds)} prompts, solvable={solved_so_far}."
                )

        solvable = [row for row in flags if row["solvable"]]
        remaining = [row for row in flags if not row["solvable"]]
        selected_rows = solvable[:desired_count]
        if len(selected_rows) < desired_count:
            remaining_sorted = sorted(
                remaining,
                key=lambda row: (
                    len(str(row.get("best_completion", "") or "")),
                    int(row.get("row_id", 0)),
                ),
            )
            selected_rows.extend(remaining_sorted[: max(0, desired_count - len(selected_rows))])

        selected_row_ids = {int(row["row_id"]) for row in selected_rows}
        selected_indices = [
            idx
            for idx, row_id in enumerate(train_ds["__row_id__"])
            if int(row_id) in selected_row_ids
        ]
        selected_ds = train_ds.select(selected_indices)

        probe_records = sorted(
            [
                {
                    "row_id": int(row["row_id"]),
                    "solvable": bool(row["solvable"]),
                    "correct_samples": int(row["correct_samples"]),
                    "best_completion": str(row["best_completion"]),
                    "ground_truth": str(row["ground_truth"]),
                }
                for row in flags
            ],
            key=lambda row: (-int(row["solvable"]), -int(row["correct_samples"]), int(row["row_id"])),
        )

        manifest = {
            "policy": policy,
            "candidate_count": int(candidate_count),
            "selected_count": int(len(selected_ds)),
            "solvable_count": int(len(solvable)),
            "used_probe": True,
            "probe_samples_per_prompt": int(sample_limit),
            "probe_temperature": float(sample_temperature),
            "resolved_max_completion_len": int(max_new_tokens),
            "candidate_row_ids": [int(v) for v in train_ds["__row_id__"]] if "__row_id__" in train_ds.column_names else [],
            "selected_row_ids": [int(v) for v in selected_ds["__row_id__"]] if "__row_id__" in selected_ds.column_names else [],
            "records": probe_records[: max(16, desired_count)],
        }
        if use_wandb and wandb_run is not None:
            wandb_run.log(
                {
                    "train_selection/solvable_count": float(len(solvable)),
                    "train_selection/candidate_count": float(candidate_count),
                    "train_selection/selected_count": float(len(selected_ds)),
                }
            )
        return selected_ds, manifest
    finally:
        del model
        _cleanup_runtime_memory(is_xla=is_xla)


def _build_prompt_diagnostics(tokenizer, train_ds, eval_ds, cfg: RunConfig, sample_limit: int = 256):
    thresholds = sorted({96, 128, 160, 192, 256, int(cfg.max_prompt_len)})

    def _summarize(split_name: str, ds):
        if tokenizer is None or ds is None or len(ds) == 0:
            return {"split": split_name, "sample_count": 0, "thresholds": {}}

        limit = min(len(ds), int(sample_limit))
        subset = ds.select(range(limit))
        prompts = [render_generation_prompt(tokenizer, prompt, cfg) for prompt in subset["prompt"]]
        lengths = []
        longest_rows = []
        for idx, prompt in enumerate(prompts):
            token_count = len(
                tokenizer(
                    prompt,
                    add_special_tokens=True,
                    padding=False,
                    truncation=False,
                )["input_ids"]
            )
            lengths.append(int(token_count))
            longest_rows.append(
                {
                    "index": idx,
                    "token_length": int(token_count),
                    "prompt_preview": _truncate_text(prompt, max_chars=320),
                }
            )

        lengths_sorted = sorted(lengths)
        return {
            "split": split_name,
            "sample_count": int(limit),
            "max_prompt_len": int(cfg.max_prompt_len),
            "min": int(lengths_sorted[0]),
            "p50": int(lengths_sorted[len(lengths_sorted) // 2]),
            "p90": int(lengths_sorted[min(len(lengths_sorted) - 1, int(len(lengths_sorted) * 0.9))]),
            "max": int(lengths_sorted[-1]),
            "fraction_over_max_prompt_len": _mean([1.0 if length > int(cfg.max_prompt_len) else 0.0 for length in lengths]),
            "thresholds": {
                str(threshold): _mean([1.0 if length > threshold else 0.0 for length in lengths])
                for threshold in thresholds
            },
            "longest_examples": sorted(longest_rows, key=lambda row: row["token_length"], reverse=True)[:3],
        }

    return {
        "train": _summarize("train", train_ds),
        "eval": _summarize("eval", eval_ds),
    }


def _log_wandb_artifact(
    wandb_run,
    path: str | None,
    name: str,
    artifact_type: str = "analysis",
    aliases: list[str] | None = None,
    metadata: dict | None = None,
):
    if wandb_run is None or not path or not os.path.exists(path):
        return
    try:
        import wandb  # type: ignore
    except Exception:
        return
    try:
        artifact = wandb.Artifact(name=name, type=artifact_type, metadata=metadata or None)
        if os.path.isdir(path):
            artifact.add_dir(path)
        else:
            artifact.add_file(path)
        wandb_run.log_artifact(artifact, aliases=aliases or None)
    except Exception as e:
        print(f"[WARN] Failed to log W&B artifact '{name}': {e}")


def _persist_model_bundle(
    cfg: RunConfig,
    *,
    role: str,
    model,
    tokenizer,
    eval_metrics: dict | None,
    benchmark_metrics: dict | None,
    use_wandb: bool,
    wandb_run=None,
    is_xla: bool,
    promotion: dict | None = None,
) -> dict:
    save_flag_name = "save_actual_model" if str(role).lower() == "actual" else "save_base_model"
    if not getattr(cfg, save_flag_name, False):
        return {"status": "disabled", "reason": f"{save_flag_name}_false"}
    if not _wandb_enabled_for_this_process():
        return {"status": "skipped", "reason": "non_master_process"}
    if model is None or tokenizer is None:
        return {"status": "skipped", "reason": f"missing_{role}_model_or_tokenizer"}

    run_root = _run_output_root(cfg)
    save_root = os.path.abspath(getattr(cfg, "saved_model_root", None) or os.path.join(run_root, "saved_models"))
    bundle_name = _artifact_safe_name(
        getattr(wandb_run, "name", None) or getattr(cfg, "wandb_run_name", None) or os.path.basename(run_root)
    )
    model_dir_name = bundle_name if str(role).lower() == "actual" else f"{bundle_name}-{str(role).lower()}"
    model_dir = os.path.join(save_root, model_dir_name)
    os.makedirs(model_dir, exist_ok=True)

    model.to(torch.device("cpu"))
    model.eval()
    _cleanup_runtime_memory(is_xla=is_xla)

    try:
        model.save_pretrained(model_dir, safe_serialization=True)
    except TypeError:
        model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    eval_accuracy = float((eval_metrics or {}).get("accuracy", 0.0) or 0.0)
    benchmark_metric_name, benchmark_metric_score = _benchmark_primary_metric(benchmark_metrics or {})
    if not benchmark_metrics:
        benchmark_metric_name = "disabled"

    metadata = {
        "status": "saved",
        "role": str(role).lower(),
        "bundle_name": bundle_name,
        "model_dir": model_dir,
        "run_root": run_root,
        "model_name": cfg.model_name,
        "dataset_name": cfg.dataset_name,
        "dataset_config_name": cfg.dataset_config_name,
        "benchmark_dataset_name": cfg.benchmark_dataset_name,
        "benchmark_dataset_config_name": cfg.benchmark_dataset_config_name,
        "pipeline_mode": cfg.pipeline_mode,
        "actual_train_mode": cfg.actual_train_mode,
        "novelty_feature_source": cfg.novelty_feature_source,
        "novelty_metric": cfg.novelty_metric,
        "lambda_novelty": float(cfg.lambda_novelty),
        "save_time_unix": float(time.time()),
        "git": _git_metadata(),
        "metrics": {
            "eval_accuracy": eval_accuracy,
            "benchmark_metric_name": benchmark_metric_name,
            "benchmark_metric_score": benchmark_metric_score,
        },
        "wandb": {
            "entity": cfg.wandb_entity,
            "project": cfg.wandb_project,
            "run_id": getattr(wandb_run, "id", None),
            "run_name": getattr(wandb_run, "name", None) or cfg.wandb_run_name,
        },
    }
    if promotion is not None:
        metadata["promotion"] = dict(promotion)
    metadata_path = os.path.join(model_dir, "artifact_metadata.json")
    _write_analysis_json(metadata_path, metadata)

    artifact_name = f"{bundle_name}-{str(role).lower()}-model"
    aliases = ["latest"]
    if str(role).lower() == "actual" and bool((promotion or {}).get("promoted")) and getattr(cfg, "promote_best_actual_model", True):
        aliases.append("promoted")
    if use_wandb and wandb_run is not None and getattr(cfg, "log_model_artifact_to_wandb", True):
        _log_wandb_artifact(
            wandb_run,
            model_dir,
            artifact_name,
            artifact_type="model",
            aliases=aliases,
            metadata=metadata,
        )
        metadata["wandb"]["artifact_name"] = artifact_name
        metadata["wandb"]["artifact_aliases"] = aliases
        _write_analysis_json(metadata_path, metadata)

    return metadata


def _persist_base_model_bundle(
    cfg: RunConfig,
    *,
    model,
    tokenizer,
    eval_metrics: dict | None,
    benchmark_metrics: dict | None,
    use_wandb: bool,
    wandb_run=None,
    is_xla: bool,
) -> dict:
    return _persist_model_bundle(
        cfg,
        role="base",
        model=model,
        tokenizer=tokenizer,
        eval_metrics=eval_metrics,
        benchmark_metrics=benchmark_metrics,
        use_wandb=use_wandb,
        wandb_run=wandb_run,
        is_xla=is_xla,
    )


def _persist_actual_model_bundle(
    cfg: RunConfig,
    result: dict,
    *,
    use_wandb: bool,
    wandb_run=None,
    is_xla: bool,
) -> dict:
    actual_model = result.get("actual_model")
    tokenizer = result.get("tokenizer")
    if actual_model is None or tokenizer is None:
        return {"status": "skipped", "reason": "missing_actual_model_or_tokenizer"}

    base_eval_acc = float((result.get("base_metrics") or {}).get("accuracy", 0.0) or 0.0)
    explorer_eval_acc = float((result.get("explorer_metrics") or {}).get("accuracy", 0.0) or 0.0)
    actual_eval_acc = float((result.get("actual_metrics") or {}).get("accuracy", 0.0) or 0.0)
    benchmark_metrics = result.get("benchmark_metrics") or {}
    _, benchmark_base_score = _benchmark_primary_metric(benchmark_metrics.get("base"))
    _, benchmark_explorer_score = _benchmark_primary_metric(benchmark_metrics.get("explorer"))
    _, benchmark_actual_score = _benchmark_primary_metric(benchmark_metrics.get("actual"))
    benchmark_beats = (
        True
        if not benchmark_metrics
        else (
            benchmark_actual_score > benchmark_base_score
            and benchmark_actual_score > benchmark_explorer_score
        )
    )
    eval_beats = actual_eval_acc > base_eval_acc and actual_eval_acc > explorer_eval_acc
    promoted = bool(eval_beats and benchmark_beats)

    return _persist_model_bundle(
        cfg,
        role="actual",
        model=actual_model,
        tokenizer=tokenizer,
        eval_metrics=result.get("actual_metrics"),
        benchmark_metrics=benchmark_metrics.get("actual"),
        use_wandb=use_wandb,
        wandb_run=wandb_run,
        is_xla=is_xla,
        promotion={
            "eval_actual_beats_base_and_explorer": eval_beats,
            "benchmark_actual_beats_base_and_explorer": benchmark_beats,
            "promoted": promoted,
        },
    )


def _filter_holdout_topics(train_ds, holdout_topics):
    if not holdout_topics:
        return train_ds
    if "topic" not in train_ds.column_names:
        print(
            "[WARN] id_ood_holdout_topics provided but dataset has no 'topic' column. "
            "Set --topic-split-mode heuristic_math_topics."
        )
        return train_ds
    before_n = len(train_ds)
    filtered = train_ds.filter(lambda ex: ex["topic"] not in holdout_topics)
    after_n = len(filtered)
    if after_n <= 0:
        raise RuntimeError(
            f"id_ood_holdout_topics={sorted(holdout_topics)} removed all training data; "
            "choose fewer holdout topics or increase dataset size."
        )
    min_after_holdout = min(before_n, max(8, before_n // 4))
    if after_n < min_after_holdout:
        print(
            f"[id/ood] Holdout topics would shrink train data too far ({before_n} -> {after_n}, "
            f"minimum allowed {min_after_holdout}); disabling holdout filtering for this run."
        )
        return train_ds
    print(
        f"[id/ood] Holdout topics={sorted(holdout_topics)}. "
        f"Train filtered from {before_n} -> {after_n} examples."
    )
    return filtered


def _run_training_round(
    cfg: RunConfig,
    train_ds,
    eval_ds,
    benchmark_eval_ds,
    benchmark_eval_cfg,
    device,
    is_xla: bool,
    use_wandb: bool,
    wandb_run=None,
    actual_init_model=None,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    holdout_topics = set(cfg.id_ood_holdout_topics or [])
    distill_losses = []
    explorer_log_hist = []
    actual_log_hist = []
    baseline_log_hist = []
    trajectories = []
    prompt_diagnostics = {}
    stage_timings = {}
    baseline_model = None
    baseline_novelty_scorer = None
    tokenizer = None
    base_metrics = None
    explorer_metrics = None
    distill_dataset_meta = None
    actual_dataset_meta = None
    actual_training_info = {"skipped": False, "reason": "not_evaluated"}
    novelty_feature_stats = {}
    benchmark_metrics = None
    benchmark_prompt_diagnostics = {}
    accepted_info = {"total": 0, "correct": 0, "strict": 0, "relaxed": 0, "accepted": 0, "policy": "none"}
    base_model_artifact = {"status": "not_requested"}

    def _eval_benchmark_model(model, tokenizer, novelty_scorer, role: str):
        nonlocal benchmark_metrics, benchmark_prompt_diagnostics
        if benchmark_eval_ds is None or benchmark_eval_cfg is None:
            return None
        if benchmark_metrics is None:
            benchmark_metrics = {}
        print(f"\nEvaluating {role} model on benchmark dataset '{benchmark_eval_cfg.dataset_name}'...")
        if _benchmark_eval_is_sampled(benchmark_eval_cfg):
            metrics = evaluate_model_with_rollouts(
                model,
                tokenizer,
                benchmark_eval_ds,
                benchmark_eval_cfg,
                novelty_scorer,
                desc=f"Benchmark{role}",
                use_wandb=use_wandb,
                wandb_run=wandb_run,
            )
        else:
            metrics = evaluate_model_on_gsm8k(
                model,
                tokenizer,
                benchmark_eval_ds,
                benchmark_eval_cfg,
                novelty_scorer,
                desc=f"Benchmark{role}",
                use_wandb=use_wandb,
                wandb_run=wandb_run,
            )
        benchmark_metrics[str(role).lower()] = metrics
        if not benchmark_prompt_diagnostics:
            benchmark_prompt_diagnostics = _build_prompt_diagnostics(
                tokenizer,
                None,
                benchmark_eval_ds,
                benchmark_eval_cfg,
            )
        return metrics

    if actual_init_model is not None and is_xla:
        actual_init_model.to(torch.device("cpu"))
        _cleanup_runtime_memory(is_xla=is_xla)

    actual_stage_init_model = actual_init_model
    warmstart_from_baseline_grpo = (
        actual_stage_init_model is None
        and str(getattr(cfg, "actual_init_source", "base") or "base").lower() == "baseline_grpo"
        and cfg.pipeline_mode != "base_only"
    )
    needs_baseline_grpo = cfg.base_eval_mode == "grpo" or warmstart_from_baseline_grpo

    skip_intermediate_evals = str(os.environ.get("TMX_SKIP_INTERMEDIATE_EVALS", "0") or "0").strip().lower() in (
        "1",
        "true",
        "yes",
    )

    if needs_baseline_grpo:
        baseline_cfg = replace(cfg, lambda_novelty=0.0, anchor_metric="none", anchor_weight=0.0)
        if baseline_cfg.explorer_output_dir:
            baseline_cfg.explorer_output_dir = f"{baseline_cfg.explorer_output_dir}_baseline"
        else:
            baseline_cfg.explorer_output_dir = os.path.join(_run_output_root(cfg), "baseline_grpo")
        t0 = time.perf_counter()
        baseline_model, tokenizer, baseline_novelty_scorer, baseline_log_hist = train_explorer_with_grpo(
            train_ds,
            cfg=baseline_cfg,
            device=device,
            is_xla=is_xla,
            use_wandb=use_wandb,
            wandb_run=wandb_run,
        )
        baseline_model.to(device)
        stage_timings["baseline_train_seconds"] = time.perf_counter() - t0

        if cfg.pipeline_mode == "base_only":
            base_metrics = None
        elif skip_intermediate_evals:
            print("[patch] Skipping intermediate Base evals for checkpoint-fit run.")
            base_metrics = None
        else:
            t0 = time.perf_counter()
            print("\nEvaluating Base model...")
            base_metrics = evaluate_model_on_gsm8k(
                baseline_model,
                tokenizer,
                eval_ds,
                cfg,
                baseline_novelty_scorer,
                desc="Base",
                use_wandb=use_wandb,
                wandb_run=wandb_run,
            )
            stage_timings["base_eval_seconds"] = time.perf_counter() - t0
            if benchmark_eval_ds is not None and benchmark_eval_cfg is not None:
                t0 = time.perf_counter()
                _eval_benchmark_model(
                    baseline_model,
                    tokenizer,
                    baseline_novelty_scorer,
                    "Base",
                )
                stage_timings["benchmark_base_eval_seconds"] = time.perf_counter() - t0
        if warmstart_from_baseline_grpo:
            print("[actual] Warm-starting Actual from the GRPO-trained baseline model.")
            if use_wandb and wandb_run is not None:
                wandb_run.log({"actual/init_from_baseline_grpo": 1.0})
            actual_stage_init_model = baseline_model
            if actual_stage_init_model is not None and is_xla:
                actual_stage_init_model.to(torch.device("cpu"))
        if base_model_artifact.get("status") == "not_requested":
            base_model_artifact = _persist_base_model_bundle(
                cfg,
                model=baseline_model,
                tokenizer=tokenizer,
                eval_metrics=base_metrics,
                benchmark_metrics=(benchmark_metrics or {}).get("base"),
                use_wandb=use_wandb,
                wandb_run=wandb_run,
                is_xla=is_xla,
            )
        if not warmstart_from_baseline_grpo and baseline_model is not None:
            del baseline_model
        baseline_model = None
        if baseline_novelty_scorer is not None:
            del baseline_novelty_scorer
            baseline_novelty_scorer = None
        _cleanup_runtime_memory(is_xla=is_xla)

    if cfg.pipeline_mode == "base_only":
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                cfg.model_name,
                **hf_from_pretrained_kwargs(cfg.model_name, repo_type="model"),
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
        base_model = baseline_model or causal_lm_from_pretrained(cfg.model_name, is_xla=is_xla).to(device)
        base_model.config.use_cache = True
        novelty_scorer = baseline_novelty_scorer
        if novelty_scorer is None:
            novelty_scorer, _ = build_novelty_scorer(base_model, tokenizer, cfg, device, is_xla)
        t0 = time.perf_counter()
        base_metrics = evaluate_model_on_gsm8k(
            base_model,
            tokenizer,
            eval_ds,
            cfg,
            novelty_scorer,
            desc="Base",
            use_wandb=use_wandb,
            wandb_run=wandb_run,
        )
        stage_timings["eval_seconds"] = time.perf_counter() - t0
        prompt_diagnostics = _build_prompt_diagnostics(tokenizer, train_ds, eval_ds, cfg)
        if benchmark_eval_ds is not None and benchmark_eval_cfg is not None:
            t0 = time.perf_counter()
            _eval_benchmark_model(
                base_model,
                tokenizer,
                novelty_scorer,
                "Base",
            )
            stage_timings["benchmark_base_eval_seconds"] = time.perf_counter() - t0
            if novelty_scorer is not None:
                benchmark_metrics["explorer"] = dict(benchmark_metrics["base"])
                benchmark_metrics["actual"] = dict(benchmark_metrics["base"])
        if base_model_artifact.get("status") == "not_requested":
            base_model_artifact = _persist_base_model_bundle(
                cfg,
                model=base_model,
                tokenizer=tokenizer,
                eval_metrics=base_metrics,
                benchmark_metrics=(benchmark_metrics or {}).get("base"),
                use_wandb=use_wandb,
                wandb_run=wandb_run,
                is_xla=is_xla,
            )
        explorer_metrics = dict(base_metrics)
        actual_metrics = dict(base_metrics)
        actual_model = baseline_model or base_model
        traj_summary = summarize_trajectories(trajectories)
        actual_training_info = {"skipped": True, "reason": "base_only_pipeline"}
        return {
            "base_metrics": base_metrics,
            "explorer_metrics": explorer_metrics,
            "actual_metrics": actual_metrics,
            "distill_losses": distill_losses,
            "explorer_log_hist": explorer_log_hist,
            "baseline_log_hist": baseline_log_hist,
            "actual_log_hist": actual_log_hist,
            "trajectories": trajectories,
            "traj_summary": traj_summary,
            "stage_timings": stage_timings,
            "base_model_artifact": base_model_artifact,
            "actual_model": actual_model,
            "tokenizer": tokenizer,
            "prompt_diagnostics": prompt_diagnostics,
            "benchmark_metrics": benchmark_metrics,
            "benchmark_prompt_diagnostics": benchmark_prompt_diagnostics,
            "distill_dataset_meta": distill_dataset_meta,
            "actual_dataset_meta": actual_dataset_meta,
            "actual_training_info": actual_training_info,
            "novelty_feature_stats": novelty_feature_stats,
        }

    anchor_reference_model = actual_init_model
    loaded_anchor_reference = False
    if anchor_reference_model is None and cfg.anchor_metric != "none" and cfg.anchor_weight > 0:
        anchor_reference_model = causal_lm_from_pretrained(cfg.model_name, is_xla=is_xla).to(device)
        loaded_anchor_reference = True

    # Skip loading Explorer model entirely when precomputed trajectories exist
    # and Explorer GRPO is already done (resume_step >= grpo_max_steps).
    # This keeps the TPU completely clean for Actual SFT/GRPO training.
    _precomp_path = os.environ.get("TMX_PRECOMPUTED_TRAJ_PATH", "").strip()
    _resume_step = int(os.environ.get("TMX_RESUME_STEP", "0") or 0)
    _skip_explorer_load = (
        _precomp_path
        and os.path.exists(_precomp_path)
        and _resume_step >= cfg.grpo_max_steps
    )
    if _skip_explorer_load:
        print(f"[patch] Skipping Explorer model load: precomputed trajectories at {_precomp_path}, "
              f"resume_step={_resume_step} >= grpo_max_steps={cfg.grpo_max_steps}")
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name,
            **hf_from_pretrained_kwargs(cfg.model_name, repo_type="model"),
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        explorer_model = None
        novelty_scorer = None
        explorer_log_hist = []
        stage_timings["explorer_train_seconds"] = 0.0
    elif (explorer_pretrained_path := os.environ.get("TMX_EXPLORER_PRETRAINED_PATH", "").strip()):
        print(f"Loading pre-trained Explorer from {explorer_pretrained_path} (skipping GRPO).")
        tokenizer = AutoTokenizer.from_pretrained(
            explorer_pretrained_path,
            **hf_from_pretrained_kwargs(explorer_pretrained_path, repo_type="model"),
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        explorer_model = causal_lm_from_pretrained(explorer_pretrained_path, is_xla=is_xla).to(device)
        novelty_scorer, _ = build_novelty_scorer(explorer_model, tokenizer, cfg, device, is_xla)
        explorer_log_hist = []
    elif cfg.decoupling_mode == "frozen":
        print("Loading frozen Explorer (base model, no RL).")
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name,
            **hf_from_pretrained_kwargs(cfg.model_name, repo_type="model"),
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        explorer_model = causal_lm_from_pretrained(cfg.model_name, is_xla=is_xla).to(device)
        novelty_scorer, _ = build_novelty_scorer(explorer_model, tokenizer, cfg, device, is_xla)
    else:
        t0 = time.perf_counter()
        explorer_model, tokenizer, novelty_scorer, explorer_log_hist = train_explorer_with_grpo(
            train_ds,
            cfg=cfg,
            device=device,
            is_xla=is_xla,
            use_wandb=use_wandb,
            wandb_run=wandb_run,
            init_model=actual_init_model,
            anchor_model=anchor_reference_model,
        )
        explorer_model.to(device)
        stage_timings["explorer_train_seconds"] = time.perf_counter() - t0

        # Save Explorer model to GCS immediately after GRPO training
        _run_name = getattr(cfg, "wandb_run_name", None) or "unnamed"
        _save_model_to_gcs_and_reload_vllm(explorer_model, tokenizer, _run_name, role="explorer")

    if _env_flag("TMX_STOP_AFTER_EXPLORER", "0"):
        print("[patch] Stopping after Explorer GRPO by request.")
        return {
            "base_metrics": base_metrics if base_metrics is not None else _empty_eval_metrics("stopped_after_explorer"),
            "explorer_metrics": _empty_eval_metrics("stopped_after_explorer"),
            "actual_metrics": _empty_eval_metrics("stopped_after_explorer"),
            "distill_losses": distill_losses,
            "explorer_log_hist": explorer_log_hist,
            "baseline_log_hist": baseline_log_hist,
            "actual_log_hist": actual_log_hist,
            "trajectories": [],
            "traj_summary": {"traj/count": 0.0},
            "accepted_info": {"accepted": 0, "total": 0, "correct": 0, "policy": "stopped_after_explorer"},
            "stage_timings": stage_timings,
            "base_model_artifact": base_model_artifact,
            "actual_model": explorer_model,
            "tokenizer": tokenizer,
            "prompt_diagnostics": prompt_diagnostics,
            "benchmark_metrics": benchmark_metrics,
            "benchmark_prompt_diagnostics": benchmark_prompt_diagnostics,
            "distill_dataset_meta": {"skipped": True, "reason": "stopped_after_explorer"},
            "actual_dataset_meta": {"skipped": True, "reason": "stopped_after_explorer"},
            "actual_training_info": {"skipped": True, "reason": "stopped_after_explorer"},
            "novelty_feature_stats": novelty_feature_stats,
        }

    precomputed_traj_path = os.environ.get("TMX_PRECOMPUTED_TRAJ_PATH", "").strip()
    if precomputed_traj_path and os.path.exists(precomputed_traj_path):
        print(f"Loading precomputed trajectories from {precomputed_traj_path}")
        trajectories = []
        with open(precomputed_traj_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    trajectories.append(json.loads(line))
        print(f"Loaded {len(trajectories)} precomputed trajectories.")
        stage_timings["trajectory_collection_seconds"] = 0.0
    else:
        explorer_model = _rehydrate_model_for_xla_generation(
            explorer_model,
            device=device,
            is_xla=is_xla,
            label="Explorer model",
        )
        explorer_final_step = 0
        for row in reversed(explorer_log_hist or []):
            try:
                explorer_final_step = max(explorer_final_step, int(row.get("step", 0) or 0))
            except Exception:
                continue
        if explorer_final_step <= 0:
            explorer_final_step = int(getattr(cfg, "grpo_max_steps", 0) or 0)
        t0 = time.perf_counter()
        trajectories = generate_offline_trajectories(
            explorer_model,
            tokenizer,
            train_ds,
            cfg=cfg,
            novelty_scorer=novelty_scorer,
            use_wandb=use_wandb,
            wandb_run=wandb_run,
            novelty_train=cfg.traj_novelty_train,
            explorer_step=explorer_final_step,
        )
        stage_timings["trajectory_collection_seconds"] = time.perf_counter() - t0
    traj_summary = summarize_trajectories(trajectories)
    accepted_trajectories, accepted_info = select_accepted_trajectories(trajectories)
    traj_summary["traj/accepted_count"] = float(accepted_info["accepted"])
    traj_summary["traj/accepted_fraction"] = float(accepted_info["accepted"] / max(1, accepted_info["total"]))
    traj_summary["traj/accepted_correct_count"] = float(accepted_info["correct"])
    traj_summary["traj/accepted_policy"] = str(accepted_info["policy"])
    print(f"Trajectory summary: {traj_summary}")
    if use_wandb and wandb_run is not None:
        wandb_run.log(traj_summary)

    # Back up trajectories to Hugging Face. GCS remains opt-in only for legacy use.
    _run_name = getattr(cfg, "wandb_run_name", None) or "unnamed"
    _traj_local = getattr(cfg, "explorer_traj_path", None)
    if _traj_local and os.path.exists(str(_traj_local)):
        _hf_traj = f"{hf_artifact_prefix(_run_name)}/explorer_trajectories.jsonl"
        _uploaded = upload_path_to_hf(str(_traj_local), _hf_traj, is_dir=False)
        if _uploaded:
            print(f"[ckpt] Backed up trajectories to {_uploaded}")
        if gcs_artifacts_enabled():
            _gcs_base = os.environ.get("TMX_GCS_CHECKPOINT_BASE", "").strip()
            if _gcs_base:
                _gcs_traj = f"{_gcs_base.rstrip('/')}/{_run_name}/explorer_trajectories.jsonl"
                try:
                    _ret = subprocess.run(
                        ["gsutil", "cp", str(_traj_local), _gcs_traj],
                        capture_output=True, text=True, timeout=120,
                    )
                    if _ret.returncode == 0:
                        print(f"[ckpt] Backed up legacy GCS trajectories to {_gcs_traj}")
                    else:
                        print(f"[ckpt] WARNING: legacy trajectory backup failed: {_ret.stderr[:200]}")
                except Exception as _e:
                    print(f"[ckpt] WARNING: legacy trajectory backup error: {_e}")

    if _env_flag("TMX_STOP_AFTER_TRAJECTORIES", "0"):
        print("[patch] Stopping after trajectory collection by request.")
        return {
            "base_metrics": base_metrics if base_metrics is not None else _empty_eval_metrics("stopped_after_trajectories"),
            "explorer_metrics": _empty_eval_metrics("stopped_after_trajectories"),
            "actual_metrics": _empty_eval_metrics("stopped_after_trajectories"),
            "distill_losses": distill_losses,
            "explorer_log_hist": explorer_log_hist,
            "baseline_log_hist": baseline_log_hist,
            "actual_log_hist": actual_log_hist,
            "trajectories": trajectories,
            "traj_summary": traj_summary,
            "accepted_info": accepted_info,
            "stage_timings": stage_timings,
            "base_model_artifact": base_model_artifact,
            "actual_model": explorer_model,
            "tokenizer": tokenizer,
            "prompt_diagnostics": prompt_diagnostics,
            "benchmark_metrics": benchmark_metrics,
            "benchmark_prompt_diagnostics": benchmark_prompt_diagnostics,
            "distill_dataset_meta": {"skipped": True, "reason": "stopped_after_trajectories"},
            "actual_dataset_meta": {"skipped": True, "reason": "stopped_after_trajectories"},
            "actual_training_info": {"skipped": True, "reason": "stopped_after_trajectories"},
            "novelty_feature_stats": novelty_feature_stats,
        }

    if cfg.decoupling_mode == "decoupled" and (skip_intermediate_evals or explorer_model is None):
        print("[patch] Skipping intermediate Explorer evals (skip_evals=%s, model=%s)."
              % (skip_intermediate_evals, "None" if explorer_model is None else "loaded"))
        explorer_metrics = None
    elif cfg.decoupling_mode == "decoupled":
        t0 = time.perf_counter()
        print("\nEvaluating Explorer model...")
        explorer_metrics = evaluate_model_on_gsm8k(
            explorer_model,
            tokenizer,
            eval_ds,
            cfg,
            novelty_scorer,
            desc="Explorer",
            use_wandb=use_wandb,
            wandb_run=wandb_run,
        )
        stage_timings["explorer_eval_seconds"] = time.perf_counter() - t0
        if benchmark_eval_ds is not None and benchmark_eval_cfg is not None:
            t0 = time.perf_counter()
            _eval_benchmark_model(
                explorer_model,
                tokenizer,
                novelty_scorer,
                "Explorer",
            )
            stage_timings["benchmark_explorer_eval_seconds"] = time.perf_counter() - t0

    # Free Explorer model and novelty scorer before Actual training to avoid OOM
    if cfg.decoupling_mode == "decoupled":
        print("[cleanup] Freeing Explorer model and novelty scorer before Actual training...")
        # Fully destroy novelty scorer (RND networks + embed model) to free TPU memory
        if novelty_scorer is not None:
            embed_model = getattr(novelty_scorer, "embed_model", None)
            if embed_model is not None:
                embed_model.cpu()
                del embed_model
            for rnd in list((getattr(novelty_scorer, "rnd_map", {}) or {}).values()):
                try:
                    rnd.cpu()
                except Exception:
                    pass
            del novelty_scorer
        novelty_scorer = None
        if explorer_model is not None:
            explorer_model.cpu()
        del explorer_model
        explorer_model = None
        _cleanup_runtime_memory(is_xla=is_xla, aggressive=True)
        print("[cleanup] Done.")

    if cfg.decoupling_mode == "decoupled":
        min_actual_examples = max(
            int(getattr(cfg, "actual_min_accepted_trajectories", 8) or 8),
            int(cfg.actual_grpo_batch_size) * max(1, int(cfg.actual_grpo_grad_accum)),
        )
        actual_bootstrap_trajectories, actual_bootstrap_info = choose_actual_bootstrap_trajectories(
            accepted_trajectories,
            trajectories,
            min_actual_examples,
            seed=cfg.seed,
        )
        if actual_bootstrap_info["used_fallback"]:
            print(
                "[actual] Accepted explorer trajectories are below the preferred bootstrap minimum; "
                f"have {len(accepted_trajectories)}, need {min_actual_examples}. "
                "Falling back to the full trajectory set for Actual bootstrapping."
            )
        elif actual_bootstrap_info["policy"] == "accepted_oversampled":
            print(
                "[actual] Accepted explorer trajectories are sparse but usable; "
                f"have {len(accepted_trajectories)}, need {min_actual_examples}. "
                f"Oversampling accepted traces by {actual_bootstrap_info['oversampled_count']} "
                "to preserve a clean Actual bootstrap set."
            )
        if use_wandb and wandb_run is not None:
            wandb_run.log(
                {
                    "actual/bootstrap_used_fallback": float(actual_bootstrap_info["used_fallback"]),
                    "actual/bootstrap_accepted_count": float(actual_bootstrap_info["accepted_count"]),
                    "actual/bootstrap_selected_count": float(actual_bootstrap_info["selected_count"]),
                    "actual/bootstrap_min_required": float(actual_bootstrap_info["min_required"]),
                    "actual/bootstrap_oversampled_count": float(actual_bootstrap_info.get("oversampled_count", 0)),
                }
            )
        actual_gate = _resolve_actual_training_gate(actual_bootstrap_info)
        if actual_gate["skip"]:
            actual_training_info = {
                **actual_gate,
                "mode": cfg.actual_train_mode,
                "bootstrap_info": dict(actual_bootstrap_info),
            }
            print(
                "[actual] Skipping Actual training because Explorer produced no accepted trajectories. "
                f"accepted={actual_gate['accepted_count']}."
            )
            if use_wandb and wandb_run is not None:
                wandb_run.log(
                    {
                        "actual/skipped": 1.0,
                        "actual/skip_reason": actual_gate["reason"],
                        "actual/skip_accepted_count": float(actual_gate["accepted_count"]),
                    }
                )
            distill_dataset_meta = {
                "skipped": True,
                "reason": actual_gate["reason"],
                "bootstrap_info": dict(actual_bootstrap_info),
            }
            actual_dataset_meta = {
                "skipped": True,
                "reason": actual_gate["reason"],
                "bootstrap_info": dict(actual_bootstrap_info),
            }
            actual_model = actual_stage_init_model
            if actual_model is None:
                actual_model = causal_lm_from_pretrained(cfg.model_name, is_xla=is_xla).to(device)
            else:
                actual_model.to(device)
            actual_model.config.use_cache = True
        if cfg.actual_train_mode == "distill":
            if not actual_gate["skip"]:
                actual_training_info = {
                    "skipped": False,
                    "reason": "trained",
                    "mode": cfg.actual_train_mode,
                    "bootstrap_info": dict(actual_bootstrap_info),
                }
                _distill_max_len = int(os.environ.get("TMX_DISTILL_MAX_TOTAL_LEN", "0") or 0) or cfg.max_total_len
                distill_dataset, distill_dataset_meta = build_supervised_dataset_from_trajectories(
                    actual_bootstrap_trajectories,
                    tokenizer,
                    cfg,
                    max_examples=cfg.max_distill_examples,
                    max_total_len=_distill_max_len,
                    target_contract=cfg.actual_target_contract,
                )
                distill_dataset_meta = {**distill_dataset_meta, "bootstrap_info": dict(actual_bootstrap_info)}
                t0 = time.perf_counter()
                actual_model, distill_losses = train_actual_model(
                    distill_dataset,
                    tokenizer,
                    cfg=cfg,
                    device=device,
                    is_xla=is_xla,
                    use_wandb=use_wandb,
                    wandb_run=wandb_run,
                    init_model=actual_stage_init_model,
                )
                stage_timings["actual_train_seconds"] = time.perf_counter() - t0
                actual_model.to(device)
            else:
                stage_timings["actual_train_seconds"] = 0.0
        else:
            actual_policy_trajectories, actual_policy_info = choose_actual_policy_trajectories(
                accepted_trajectories,
                trajectories,
                policy=getattr(cfg, "actual_dataset_policy", "accepted_balanced"),
                max_incorrect_ratio=float(getattr(cfg, "actual_max_incorrect_ratio", 1.0) or 0.0),
                seed=cfg.seed,
            )
            print(
                "[actual] Offline dataset policy="
                f"{actual_policy_info['policy']} source={actual_policy_info['positive_source']} "
                f"selected={actual_policy_info['selected_count']} "
                f"(positives={actual_policy_info['positive_count']}, "
                f"incorrect={actual_policy_info['incorrect_selected_count']})."
            )
            if use_wandb and wandb_run is not None:
                wandb_run.log(
                    {
                        "actual/policy_selected_count": float(actual_policy_info["selected_count"]),
                        "actual/policy_positive_count": float(actual_policy_info["positive_count"]),
                        "actual/policy_incorrect_selected_count": float(actual_policy_info["incorrect_selected_count"]),
                        "actual/policy_high_quality_incorrect_available": float(
                            actual_policy_info["high_quality_incorrect_available"]
                        ),
                        "actual/policy_incorrect_available": float(actual_policy_info["incorrect_available"]),
                        "actual/policy_used_fallback": float(actual_policy_info["used_fallback"]),
                    }
                )
            actual_gate = _resolve_actual_training_gate(actual_bootstrap_info, actual_policy_info)
            if actual_gate["skip"]:
                actual_training_info = {
                    **actual_gate,
                    "mode": cfg.actual_train_mode,
                    "bootstrap_info": dict(actual_bootstrap_info),
                    "policy_info": dict(actual_policy_info),
                }
                print(
                    "[actual] Skipping Actual training because the offline policy dataset has no positive trajectories. "
                    f"accepted={actual_gate['accepted_count']} positives={actual_gate.get('positive_count', 0)}."
                )
                if use_wandb and wandb_run is not None:
                    wandb_run.log(
                        {
                            "actual/skipped": 1.0,
                            "actual/skip_reason": actual_gate["reason"],
                            "actual/skip_accepted_count": float(actual_gate["accepted_count"]),
                            "actual/skip_positive_count": float(actual_gate.get("positive_count", 0) or 0),
                        }
                    )
                distill_dataset_meta = {
                    "skipped": True,
                    "reason": actual_gate["reason"],
                    "bootstrap_info": dict(actual_bootstrap_info),
                    "policy_info": dict(actual_policy_info),
                }
                actual_dataset_meta = {
                    "skipped": True,
                    "reason": actual_gate["reason"],
                    "bootstrap_info": dict(actual_bootstrap_info),
                    "policy_info": dict(actual_policy_info),
                }
                actual_model = actual_stage_init_model
                if actual_model is None:
                    actual_model = causal_lm_from_pretrained(cfg.model_name, is_xla=is_xla).to(device)
                else:
                    actual_model.to(device)
                actual_model.config.use_cache = True
                stage_timings["actual_train_seconds"] = 0.0
            else:
                actual_training_info = {
                    "skipped": False,
                    "reason": "trained",
                    "mode": cfg.actual_train_mode,
                    "bootstrap_info": dict(actual_bootstrap_info),
                    "policy_info": dict(actual_policy_info),
                }
                for trajectory in actual_policy_trajectories:
                    if cfg.actual_reward_source == "correctness":
                        trajectory["reward_actual"] = float(trajectory.get("r_correct", 0.0))
                    else:
                        trajectory["reward_actual"] = float(trajectory.get("reward", 0.0))
                if cfg.actual_train_mode == "sft_online_grpo":
                    actual_dataset = []
                    actual_dataset_meta = {
                        "skipped": True,
                        "reason": "online_grpo_uses_dataset_prompts",
                        "bootstrap_info": dict(actual_bootstrap_info),
                        "policy_info": dict(actual_policy_info),
                    }
                    from .distill import train_actual_with_online_grpo
                    t0 = time.perf_counter()

                    # Skip SFT if resuming Online GRPO from a checkpoint
                    _actual_resume_ckpt = os.environ.get("TMX_ACTUAL_GRPO_RESUME_CHECKPOINT", "")
                    if _actual_resume_ckpt:
                        print(f"[actual] Skipping SFT: resuming Online GRPO from checkpoint {_actual_resume_ckpt}")
                        actual_model = causal_lm_from_pretrained(_actual_resume_ckpt, is_xla=is_xla).to(device)
                        distill_dataset_meta = {"skipped": True, "reason": "resume_from_checkpoint"}
                    else:
                        _distill_max_len = int(os.environ.get("TMX_DISTILL_MAX_TOTAL_LEN", "0") or 0) or cfg.max_total_len
                        distill_dataset, distill_dataset_meta = build_supervised_dataset_from_trajectories(
                            actual_bootstrap_trajectories,
                            tokenizer,
                            cfg,
                            max_examples=cfg.max_distill_examples,
                            max_total_len=_distill_max_len,
                            target_contract=cfg.actual_target_contract,
                        )
                        distill_dataset_meta = {**distill_dataset_meta, "bootstrap_info": dict(actual_bootstrap_info)}
                        actual_model, distill_losses = train_actual_model(
                            distill_dataset,
                            tokenizer,
                            cfg=cfg,
                            device=device,
                            is_xla=is_xla,
                            use_wandb=use_wandb,
                            wandb_run=wandb_run,
                            init_model=actual_stage_init_model,
                        )
                        actual_model.to(device)
                        _run_name = getattr(cfg, "wandb_run_name", None) or "unnamed"
                        print("[ckpt] Saving distill-stage Actual model before online GRPO.")
                        _save_model_to_gcs_and_reload_vllm(actual_model, tokenizer, _run_name, role="actual")
                        actual_model.to(device)

                    print("[actual] Starting online GRPO on dataset prompts (correctness reward)...")
                    actual_model, actual_log_hist = train_actual_with_online_grpo(
                        train_ds,
                        tokenizer,
                        cfg=cfg,
                        device=device,
                        is_xla=is_xla,
                        use_wandb=use_wandb,
                        wandb_run=wandb_run,
                        init_model=actual_model,
                    )
                    stage_timings["actual_train_seconds"] = time.perf_counter() - t0
                elif cfg.actual_train_mode == "sft_grpo":
                    actual_dataset, actual_dataset_meta = build_rewarded_dataset_from_trajectories(
                        actual_policy_trajectories,
                        tokenizer,
                        cfg,
                        max_examples=cfg.max_actual_examples,
                        max_total_len=cfg.max_total_len,
                        reward_key="reward_actual",
                        target_correct_fraction=cfg.actual_target_correct_fraction,
                        seed=cfg.seed,
                        target_contract=cfg.actual_target_contract,
                    )
                    actual_dataset_meta = {
                        **actual_dataset_meta,
                        "bootstrap_info": dict(actual_bootstrap_info),
                        "policy_info": dict(actual_policy_info),
                    }
                    distill_dataset, distill_dataset_meta = build_supervised_dataset_from_trajectories(
                        actual_bootstrap_trajectories,
                        tokenizer,
                        cfg,
                        max_examples=cfg.max_distill_examples,
                        max_total_len=cfg.max_total_len,
                        target_contract=cfg.actual_target_contract,
                    )
                    distill_dataset_meta = {**distill_dataset_meta, "bootstrap_info": dict(actual_bootstrap_info)}
                    t0 = time.perf_counter()
                    actual_model, distill_losses = train_actual_model(
                        distill_dataset,
                        tokenizer,
                        cfg=cfg,
                        device=device,
                        is_xla=is_xla,
                        use_wandb=use_wandb,
                        wandb_run=wandb_run,
                        init_model=actual_stage_init_model,
                    )
                    actual_model.to(device)
                    _run_name = getattr(cfg, "wandb_run_name", None) or "unnamed"
                    print("[ckpt] Saving distill-stage Actual model before offline GRPO.")
                    _save_model_to_gcs_and_reload_vllm(actual_model, tokenizer, _run_name, role="actual")
                    actual_model.to(device)
                    actual_model, actual_log_hist = train_actual_with_offline_grpo(
                        actual_dataset,
                        tokenizer,
                        cfg=cfg,
                        device=device,
                        is_xla=is_xla,
                        use_wandb=use_wandb,
                        wandb_run=wandb_run,
                        init_model=actual_model,
                    )
                    stage_timings["actual_train_seconds"] = time.perf_counter() - t0
                else:
                    actual_dataset, actual_dataset_meta = build_rewarded_dataset_from_trajectories(
                        actual_policy_trajectories,
                        tokenizer,
                        cfg,
                        max_examples=cfg.max_actual_examples,
                        max_total_len=cfg.max_total_len,
                        reward_key="reward_actual",
                        target_correct_fraction=cfg.actual_target_correct_fraction,
                        seed=cfg.seed,
                        target_contract=cfg.actual_target_contract,
                    )
                    actual_dataset_meta = {
                        **actual_dataset_meta,
                        "bootstrap_info": dict(actual_bootstrap_info),
                        "policy_info": dict(actual_policy_info),
                    }
                    t0 = time.perf_counter()
                    actual_model, actual_log_hist = train_actual_with_offline_grpo(
                        actual_dataset,
                        tokenizer,
                        cfg=cfg,
                        device=device,
                        is_xla=is_xla,
                        use_wandb=use_wandb,
                        wandb_run=wandb_run,
                        init_model=actual_stage_init_model,
                    )
                    stage_timings["actual_train_seconds"] = time.perf_counter() - t0
                    actual_model.to(device)
    else:
        actual_model = explorer_model
        actual_training_info = {"skipped": False, "reason": "shared_explorer_model"}

    if baseline_model is not None:
        base_model = baseline_model
    elif cfg.decoupling_mode == "frozen":
        base_model = explorer_model
    elif base_metrics is None:
        base_model = causal_lm_from_pretrained(cfg.model_name, is_xla=is_xla).to(device)
    else:
        base_model = None

    final_eval_cfg = _resolve_final_eval_cfg(cfg)
    final_eval_roles = _resolve_final_eval_roles()
    skip_final_benchmark_evals = _env_flag("TMX_SKIP_FINAL_BENCHMARK_EVALS", "0")
    if final_eval_cfg.max_eval_examples != cfg.max_eval_examples:
        print(
            "[patch] Final eval max examples overridden: "
            f"{cfg.max_eval_examples} -> {final_eval_cfg.max_eval_examples}."
        )
    if final_eval_roles != {"base", "explorer", "actual"}:
        print(f"[patch] Final eval roles limited to: {sorted(final_eval_roles)}")
    if skip_final_benchmark_evals:
        print("[patch] Skipping final benchmark evals.")

    t0 = time.perf_counter()
    if base_model is not None and base_metrics is None:
        if "base" in final_eval_roles:
            base_model.config.use_cache = True
            print("\nEvaluating Base model...")
            base_metrics = evaluate_model_on_gsm8k(
                base_model,
                tokenizer,
                eval_ds,
                final_eval_cfg,
                novelty_scorer,
                desc="Base",
                use_wandb=use_wandb,
                wandb_run=wandb_run,
            )
        else:
            print("[patch] Skipping final Base eval.")
            base_metrics = _empty_eval_metrics("skipped_final_base_eval")
    if explorer_metrics is None and explorer_model is not None:
        if "explorer" in final_eval_roles:
            print("\nEvaluating Explorer model...")
            explorer_metrics = evaluate_model_on_gsm8k(
                explorer_model,
                tokenizer,
                eval_ds,
                final_eval_cfg,
                novelty_scorer,
                desc="Explorer",
                use_wandb=use_wandb,
                wandb_run=wandb_run,
            )
            if (
                not skip_final_benchmark_evals
                and benchmark_eval_ds is not None
                and benchmark_eval_cfg is not None
            ):
                t0_bench = time.perf_counter()
                _eval_benchmark_model(
                    explorer_model,
                    tokenizer,
                    novelty_scorer,
                    "Explorer",
                )
                stage_timings["benchmark_explorer_eval_seconds"] = stage_timings.get("benchmark_explorer_eval_seconds", 0.0) + (
                    time.perf_counter() - t0_bench
                )
        else:
            print("[patch] Skipping final Explorer eval.")
            explorer_metrics = _empty_eval_metrics("skipped_final_explorer_eval")
    # Save Actual model to GCS for checkpointing and potential vLLM eval
    _run_name = getattr(cfg, "wandb_run_name", None) or "unnamed"
    _save_model_to_gcs_and_reload_vllm(actual_model, tokenizer, _run_name, role="actual")
    if actual_model is not None:
        actual_model.to(device)

    # Also save Explorer model
    if explorer_model is not None:
        _save_model_to_gcs_and_reload_vllm(explorer_model, tokenizer, _run_name, role="explorer")

    if "actual" in final_eval_roles:
        print("\nEvaluating Actual model...")
        actual_metrics = evaluate_model_on_gsm8k(
            actual_model,
            tokenizer,
            eval_ds,
            final_eval_cfg,
            novelty_scorer,
            desc="Actual",
            use_wandb=use_wandb,
            wandb_run=wandb_run,
        )
    else:
        print("[patch] Skipping final Actual eval.")
        actual_metrics = _empty_eval_metrics("skipped_final_actual_eval")
    stage_timings["eval_seconds"] = time.perf_counter() - t0
    prompt_diagnostics = _build_prompt_diagnostics(tokenizer, train_ds, eval_ds, cfg)
    if not skip_final_benchmark_evals and benchmark_eval_ds is not None and benchmark_eval_cfg is not None:
        if (benchmark_metrics is None or "base" not in benchmark_metrics) and "base" in final_eval_roles:
            t0_bench = time.perf_counter()
            _eval_benchmark_model(
                base_model or actual_model,
                tokenizer,
                novelty_scorer,
                "Base",
            )
            stage_timings["benchmark_base_eval_seconds"] = stage_timings.get("benchmark_base_eval_seconds", 0.0) + (
                time.perf_counter() - t0_bench
            )
        if "actual" in final_eval_roles:
            t0_bench = time.perf_counter()
            _eval_benchmark_model(
                actual_model,
                tokenizer,
                novelty_scorer,
                "Actual",
            )
            stage_timings["benchmark_actual_eval_seconds"] = time.perf_counter() - t0_bench
    if base_model_artifact.get("status") == "not_requested" and base_model is not None:
        base_model_artifact = _persist_base_model_bundle(
            cfg,
            model=base_model,
            tokenizer=tokenizer,
            eval_metrics=base_metrics,
            benchmark_metrics=(benchmark_metrics or {}).get("base"),
            use_wandb=use_wandb,
            wandb_run=wandb_run,
            is_xla=is_xla,
        )
    if novelty_scorer is not None:
        novelty_feature_stats = novelty_scorer.snapshot_feature_stats()

    if loaded_anchor_reference and anchor_reference_model is not None:
        del anchor_reference_model
    if base_model is not None and base_model is not actual_model and base_model is not baseline_model:
        del base_model
    if baseline_model is not None and baseline_model is not actual_model:
        del baseline_model
    if explorer_model is not None and explorer_model is not actual_model:
        del explorer_model
    _cleanup_runtime_memory(is_xla=is_xla)

    return {
        "base_metrics": base_metrics,
        "explorer_metrics": explorer_metrics,
        "actual_metrics": actual_metrics,
        "distill_losses": distill_losses,
        "explorer_log_hist": explorer_log_hist,
        "baseline_log_hist": baseline_log_hist,
        "actual_log_hist": actual_log_hist,
        "trajectories": trajectories,
        "traj_summary": traj_summary,
        "accepted_info": accepted_info,
        "stage_timings": stage_timings,
        "base_model_artifact": base_model_artifact,
        "actual_model": actual_model,
        "tokenizer": tokenizer,
        "prompt_diagnostics": prompt_diagnostics,
        "benchmark_metrics": benchmark_metrics,
        "benchmark_prompt_diagnostics": benchmark_prompt_diagnostics,
        "distill_dataset_meta": distill_dataset_meta,
        "actual_dataset_meta": actual_dataset_meta,
        "actual_training_info": actual_training_info,
        "novelty_feature_stats": novelty_feature_stats,
    }


def _write_run_outputs(
    cfg: RunConfig,
    dataset_manifest: dict,
    benchmark_manifest: dict | None,
    result: dict,
    use_wandb: bool,
    wandb_run=None,
    embedding_payload=None,
    extra_analysis: dict | None = None,
):
    holdout_topics = set(cfg.id_ood_holdout_topics or [])
    plot_results(
        base_metrics=result["base_metrics"],
        explorer_metrics=result["explorer_metrics"],
        actual_metrics=result["actual_metrics"],
        distill_losses=result["distill_losses"],
        explorer_log_hist=result["explorer_log_hist"],
        actual_log_hist=result["actual_log_hist"],
        trajectories=result["trajectories"],
        cfg=cfg,
        use_wandb=use_wandb,
        wandb_run=wandb_run,
        out_dir=cfg.plots_dir,
    )

    analysis_dir = _analysis_dir(cfg)
    topic_id_ood_payload = {
        "holdout_topics": sorted(list(holdout_topics)),
        "base": _topic_eval_slice(result["base_metrics"]),
        "explorer": _topic_eval_slice(result["explorer_metrics"]),
        "actual": _topic_eval_slice(result["actual_metrics"]),
    }
    topic_path = os.path.join(analysis_dir, "topic_id_ood_summary.json")
    _write_analysis_json(topic_path, topic_id_ood_payload)

    dataset_manifest_path = cfg.dataset_manifest_path or os.path.join(analysis_dir, "dataset_manifest.json")
    _write_analysis_json(dataset_manifest_path, dataset_manifest)
    benchmark_manifest_path = None
    if benchmark_manifest:
        benchmark_manifest_path = cfg.benchmark_manifest_path or os.path.join(analysis_dir, "benchmark_dataset_manifest.json")
        _write_analysis_json(benchmark_manifest_path, benchmark_manifest)

    benchmark_metrics_payload = result.get("benchmark_metrics") or {}
    benchmark_base_metric_name, benchmark_base_metric_value = _benchmark_primary_metric(benchmark_metrics_payload.get("base"))
    benchmark_explorer_metric_name, benchmark_explorer_metric_value = _benchmark_primary_metric(benchmark_metrics_payload.get("explorer"))
    benchmark_actual_metric_name, benchmark_actual_metric_value = _benchmark_primary_metric(benchmark_metrics_payload.get("actual"))
    runtime_payload = {
        "stage_timings": result.get("stage_timings", {}),
        "git": _git_metadata(),
        "base_acc": float(result["base_metrics"]["accuracy"]),
        "explorer_acc": float(result["explorer_metrics"]["accuracy"]),
        "actual_acc": float(result["actual_metrics"]["accuracy"]),
        "base_novelty_mean": float(result["base_metrics"].get("novelty_mean", 0.0)),
        "explorer_novelty_mean": float(result["explorer_metrics"].get("novelty_mean", 0.0)),
        "actual_novelty_mean": float(result["actual_metrics"].get("novelty_mean", 0.0)),
        "accepted_count": int((result.get("accepted_info") or {}).get("accepted", 0)),
        "benchmark_base_acc": benchmark_base_metric_value,
        "benchmark_explorer_acc": benchmark_explorer_metric_value,
        "benchmark_actual_acc": benchmark_actual_metric_value,
        "benchmark_base_metric_name": benchmark_base_metric_name,
        "benchmark_explorer_metric_name": benchmark_explorer_metric_name,
        "benchmark_actual_metric_name": benchmark_actual_metric_name,
    }
    if extra_analysis:
        runtime_payload.update(extra_analysis)
    runtime_path = os.path.join(analysis_dir, "runtime_summary.json")
    _write_analysis_json(runtime_path, runtime_payload)

    prompt_payload = result.get("prompt_diagnostics", {})
    prompt_path = os.path.join(analysis_dir, "prompt_diagnostics.json")
    _write_analysis_json(prompt_path, prompt_payload)
    benchmark_prompt_payload = result.get("benchmark_prompt_diagnostics", {})
    benchmark_prompt_path = None
    if benchmark_prompt_payload:
        benchmark_prompt_path = os.path.join(analysis_dir, "benchmark_prompt_diagnostics.json")
        _write_analysis_json(benchmark_prompt_path, benchmark_prompt_payload)

    completion_payload = {
        "accepted_trajectories": result.get("accepted_info", {}),
        "trajectory": _summarize_completion_records(
            result.get("trajectories", []),
            novelty_key="r_novel",
            reward_key="reward",
        ),
        "eval": {
            "base": _summarize_completion_records(
                result["base_metrics"].get("samples", []),
                novelty_key="novelty",
            ),
            "explorer": _summarize_completion_records(
                result["explorer_metrics"].get("samples", []),
                novelty_key="novelty",
            ),
            "actual": _summarize_completion_records(
                result["actual_metrics"].get("samples", []),
                novelty_key="novelty",
            ),
        },
    }
    completion_path = os.path.join(analysis_dir, "completion_diagnostics.json")
    _write_analysis_json(completion_path, completion_payload)

    benchmark_summary_payload = None
    benchmark_summary_path = None
    benchmark_eval_samples_path = None
    benchmark_rollout_paths = {}
    if result.get("benchmark_metrics"):
        benchmark_summary_payload = {
            "dataset_name": cfg.benchmark_dataset_name,
            "dataset_config_name": cfg.benchmark_dataset_config_name,
            "protocol": {
                "eval_mode": cfg.benchmark_eval_mode,
                "num_rollouts": int(getattr(cfg, "benchmark_num_rollouts", 1) or 1),
                "rollout_temperature": float(getattr(cfg, "benchmark_rollout_temperature", 1.0) or 1.0),
                "rollout_top_p": float(getattr(cfg, "benchmark_rollout_top_p", 1.0) or 1.0),
                "rollout_max_completion_len": int(
                    getattr(cfg, "benchmark_rollout_max_completion_len", None) or resolved_max_completion_len(cfg, phase="eval")
                ),
                "pass_k_values": [int(v) for v in (getattr(cfg, "benchmark_rollout_pass_k", []) or [])],
            },
            "base": _summarize_completion_records(
                result["benchmark_metrics"].get("base", {}).get("samples", []),
                novelty_key="novelty",
            ),
            "explorer": _summarize_completion_records(
                result["benchmark_metrics"].get("explorer", {}).get("samples", []),
                novelty_key="novelty",
            ),
            "actual": _summarize_completion_records(
                result["benchmark_metrics"].get("actual", {}).get("samples", []),
                novelty_key="novelty",
            ),
            "accuracies": {
                "base": benchmark_base_metric_value,
                "explorer": benchmark_explorer_metric_value,
                "actual": benchmark_actual_metric_value,
            },
            "primary_metrics": {
                "base": benchmark_base_metric_name,
                "explorer": benchmark_explorer_metric_name,
                "actual": benchmark_actual_metric_name,
            },
            "pass_at_k": {
                "base": result["benchmark_metrics"].get("base", {}).get("pass_at_k", {}),
                "explorer": result["benchmark_metrics"].get("explorer", {}).get("pass_at_k", {}),
                "actual": result["benchmark_metrics"].get("actual", {}).get("pass_at_k", {}),
            },
            "problem_summaries": {
                "base": result["benchmark_metrics"].get("base", {}).get("problem_summaries", []),
                "explorer": result["benchmark_metrics"].get("explorer", {}).get("problem_summaries", []),
                "actual": result["benchmark_metrics"].get("actual", {}).get("problem_summaries", []),
            },
        }
        benchmark_summary_path = os.path.join(analysis_dir, "benchmark_summary.json")
        _write_analysis_json(benchmark_summary_path, benchmark_summary_payload)

    trajectory_audit_payload = _build_trajectory_audit_payload(cfg, result)
    trajectory_audit_path = os.path.join(analysis_dir, "trajectory_audit.json")
    _write_analysis_json(trajectory_audit_path, trajectory_audit_payload)

    eval_samples_payload = {
        "base": result["base_metrics"].get("samples", []),
        "explorer": result["explorer_metrics"].get("samples", []),
        "actual": result["actual_metrics"].get("samples", []),
    }
    eval_samples_path = os.path.join(analysis_dir, "eval_samples.json")
    _write_analysis_json(eval_samples_path, eval_samples_payload)
    if result.get("benchmark_metrics"):
        benchmark_eval_samples_payload = {
            "dataset_name": cfg.benchmark_dataset_name,
            "base": result["benchmark_metrics"].get("base", {}).get("samples", []),
            "explorer": result["benchmark_metrics"].get("explorer", {}).get("samples", []),
            "actual": result["benchmark_metrics"].get("actual", {}).get("samples", []),
        }
        benchmark_eval_samples_path = os.path.join(analysis_dir, "benchmark_eval_samples.json")
        _write_analysis_json(benchmark_eval_samples_path, benchmark_eval_samples_payload)
        if any(result["benchmark_metrics"].get(role, {}).get("problem_rollouts") for role in ("base", "explorer", "actual")):
            configured_rollout_path = getattr(cfg, "benchmark_rollout_output_path", None)
            if configured_rollout_path:
                configured_rollout_path = os.path.abspath(configured_rollout_path)
                if configured_rollout_path.endswith(".json"):
                    rollout_root = os.path.dirname(configured_rollout_path) or analysis_dir
                    rollout_stem = os.path.splitext(os.path.basename(configured_rollout_path))[0]
                else:
                    rollout_root = configured_rollout_path
                    rollout_stem = "benchmark_rollouts"
            else:
                rollout_root = analysis_dir
                rollout_stem = "benchmark_rollouts"
            os.makedirs(rollout_root, exist_ok=True)
            for role in ("base", "explorer", "actual"):
                role_rollouts = result["benchmark_metrics"].get(role, {}).get("problem_rollouts")
                if not role_rollouts:
                    continue
                role_path = os.path.join(rollout_root, f"{rollout_stem}_{role}.json")
                _write_analysis_json(role_path, role_rollouts)
                benchmark_rollout_paths[role] = role_path
            if benchmark_summary_payload is not None and benchmark_summary_path is not None and benchmark_rollout_paths:
                benchmark_summary_payload["rollout_json_paths"] = benchmark_rollout_paths
                _write_analysis_json(benchmark_summary_path, benchmark_summary_payload)

    loss_payload = _build_loss_diagnostics(result)
    loss_path = os.path.join(analysis_dir, "loss_diagnostics.json")
    _write_analysis_json(loss_path, loss_payload)

    analysis_payload = {
        "config": {
            "model_name": cfg.model_name,
            "dataset_name": cfg.dataset_name,
            "dataset_config_name": cfg.dataset_config_name,
            "benchmark_dataset_name": cfg.benchmark_dataset_name,
            "benchmark_dataset_config_name": cfg.benchmark_dataset_config_name,
            "benchmark_eval_mode": cfg.benchmark_eval_mode,
            "benchmark_num_rollouts": cfg.benchmark_num_rollouts,
            "benchmark_rollout_temperature": cfg.benchmark_rollout_temperature,
            "benchmark_rollout_top_p": cfg.benchmark_rollout_top_p,
            "benchmark_rollout_pass_k": list(cfg.benchmark_rollout_pass_k),
            "benchmark_rollout_max_completion_len": cfg.benchmark_rollout_max_completion_len,
            "dataset_split_policy": cfg.dataset_split_policy,
            "dataset_split_seed": cfg.dataset_split_seed,
            "pipeline_mode": cfg.pipeline_mode,
            "num_rounds": cfg.num_rounds,
            "round_shard_policy": cfg.round_shard_policy,
            "reward_shape": cfg.reward_shape,
            "novelty_feature_source": cfg.novelty_feature_source,
            "novelty_layers": list(cfg.novelty_layers),
            "novelty_layer_pool": cfg.novelty_layer_pool,
            "novelty_layer_agg": cfg.novelty_layer_agg,
            "novelty_text_contract": cfg.novelty_text_contract,
            "novelty_feature_max_length": cfg.novelty_feature_max_length or cfg.max_total_len,
            "novelty_metric": cfg.novelty_metric,
            "lambda_novelty": cfg.lambda_novelty,
            "base_eval_mode": cfg.base_eval_mode,
            "actual_train_mode": cfg.actual_train_mode,
            "actual_target_contract": cfg.actual_target_contract,
            "actual_init_source": cfg.actual_init_source,
            "actual_reward_source": cfg.actual_reward_source,
            "actual_target_correct_fraction": cfg.actual_target_correct_fraction,
            "anchor_metric": cfg.anchor_metric,
            "anchor_weight": cfg.anchor_weight,
            "novelty_study_tag": cfg.novelty_study_tag,
            "novelty_study_mode": cfg.novelty_study_mode,
            "id_ood_holdout_topics": sorted(list(holdout_topics)),
            "embedding_novelty_eval": cfg.embedding_novelty_eval,
            "embedding_model_name": cfg.embedding_model_name,
            "wall_clock_budget_minutes": cfg.wall_clock_budget_minutes,
            "answer_constraint_mode": cfg.answer_constraint_mode,
            "device_preference": cfg.device_preference,
            "wandb_project": cfg.wandb_project,
            "wandb_entity": cfg.wandb_entity,
            "wandb_run_name": cfg.wandb_run_name,
            "format_penalty_weight": cfg.format_penalty_weight,
            "gate_novelty_by_quality": cfg.gate_novelty_by_quality,
            "save_base_model": cfg.save_base_model,
            "save_actual_model": cfg.save_actual_model,
            "saved_model_root": cfg.saved_model_root,
        },
        "dataset_manifest": dataset_manifest,
        "benchmark_dataset_manifest": benchmark_manifest or {},
        "prompt_diagnostics": prompt_payload,
        "benchmark_prompt_diagnostics": benchmark_prompt_payload,
        "trajectory_summary": result["traj_summary"],
        "accepted_trajectories": result.get("accepted_info", {}),
        "trajectory_audit": trajectory_audit_payload,
        "topic_id_ood_summary": topic_id_ood_payload,
        "distill_dataset_meta": result.get("distill_dataset_meta"),
        "actual_dataset_meta": result.get("actual_dataset_meta"),
        "embedding_summary": embedding_payload if isinstance(embedding_payload, dict) else {"status": "disabled"},
        "runtime_summary": runtime_payload,
        "benchmark_summary": benchmark_summary_payload or {"status": "disabled"},
        "base_model_artifact": (extra_analysis or {}).get("base_model_artifact", {}),
        "actual_model_artifact": (extra_analysis or {}).get("actual_model_artifact", {}),
        "results": {
            "base_accuracy": result["base_metrics"]["accuracy"],
            "explorer_accuracy": result["explorer_metrics"]["accuracy"],
            "actual_accuracy": result["actual_metrics"]["accuracy"],
            "base_valid_answer_fraction": _mean([1.0 if bool(s.get("valid_answer_presentation", False)) else 0.0 for s in result["base_metrics"].get("samples", [])]),
            "explorer_valid_answer_fraction": _mean([1.0 if bool(s.get("valid_answer_presentation", False)) else 0.0 for s in result["explorer_metrics"].get("samples", [])]),
            "actual_valid_answer_fraction": _mean([1.0 if bool(s.get("valid_answer_presentation", False)) else 0.0 for s in result["actual_metrics"].get("samples", [])]),
            "base_novelty_mean": result["base_metrics"].get("novelty_mean", 0.0),
            "explorer_novelty_mean": result["explorer_metrics"].get("novelty_mean", 0.0),
            "actual_novelty_mean": result["actual_metrics"].get("novelty_mean", 0.0),
            "base_id_accuracy": result["base_metrics"].get("id_accuracy", 0.0),
            "base_ood_accuracy": result["base_metrics"].get("ood_accuracy", 0.0),
            "explorer_id_accuracy": result["explorer_metrics"].get("id_accuracy", 0.0),
            "explorer_ood_accuracy": result["explorer_metrics"].get("ood_accuracy", 0.0),
            "actual_id_accuracy": result["actual_metrics"].get("id_accuracy", 0.0),
            "actual_ood_accuracy": result["actual_metrics"].get("ood_accuracy", 0.0),
            "base_id_novelty_mean": result["base_metrics"].get("id_novelty_mean", 0.0),
            "base_ood_novelty_mean": result["base_metrics"].get("ood_novelty_mean", 0.0),
            "explorer_id_novelty_mean": result["explorer_metrics"].get("id_novelty_mean", 0.0),
            "explorer_ood_novelty_mean": result["explorer_metrics"].get("ood_novelty_mean", 0.0),
            "actual_id_novelty_mean": result["actual_metrics"].get("id_novelty_mean", 0.0),
            "actual_ood_novelty_mean": result["actual_metrics"].get("ood_novelty_mean", 0.0),
            "benchmark_base_accuracy": benchmark_base_metric_value,
            "benchmark_explorer_accuracy": benchmark_explorer_metric_value,
            "benchmark_actual_accuracy": benchmark_actual_metric_value,
            "benchmark_base_metric_name": benchmark_base_metric_name,
            "benchmark_explorer_metric_name": benchmark_explorer_metric_name,
            "benchmark_actual_metric_name": benchmark_actual_metric_name,
            "benchmark_pass_at_k": {
                "base": benchmark_metrics_payload.get("base", {}).get("pass_at_k", {}),
                "explorer": benchmark_metrics_payload.get("explorer", {}).get("pass_at_k", {}),
                "actual": benchmark_metrics_payload.get("actual", {}).get("pass_at_k", {}),
            },
        },
    }
    if extra_analysis:
        analysis_payload.update(extra_analysis)

    novelty_path = os.path.join(analysis_dir, "novelty_summary.json")
    _write_analysis_json(novelty_path, analysis_payload)
    accuracy_path = os.path.join(analysis_dir, "accuracy_over_time.json")
    _write_analysis_json(
        accuracy_path,
        {
            "explorer_log_history": result["explorer_log_hist"],
            "baseline_log_history": result["baseline_log_hist"],
            "actual_log_history": result["actual_log_hist"],
            **(extra_analysis or {}),
        },
    )
    if extra_analysis and extra_analysis.get("round_summaries"):
        multi_round_path = os.path.join(analysis_dir, "multi_round_summary.json")
        _write_analysis_json(
            multi_round_path,
            {
                "round_summaries": extra_analysis.get("round_summaries", []),
                "final_summary": extra_analysis.get("final_summary", {}),
            },
        )

    if use_wandb and wandb_run is not None:
        trajectory_generation = trajectory_audit_payload.get("trajectory_generation", {})
        novelty_feature_stats = trajectory_audit_payload.get("novelty_feature_stats", {})
        distill_meta = (trajectory_audit_payload.get("actual_datasets", {}) or {}).get("distill") or {}
        actual_meta = (trajectory_audit_payload.get("actual_datasets", {}) or {}).get("offline_grpo") or {}
        wandb_run.log(
            {
                "audit/traj_clipped_fraction": float(trajectory_generation.get("clipped_fraction", 0.0) or 0.0),
                "audit/traj_prompt_truncated_fraction": float(trajectory_generation.get("prompt_truncated_fraction", 0.0) or 0.0),
                "audit/traj_normalization_changed_fraction": float(trajectory_generation.get("normalization_changed_fraction", 0.0) or 0.0),
                "audit/traj_valid_answer_fraction": float(trajectory_generation.get("valid_answer_fraction", 0.0) or 0.0),
                "audit/novelty_feature_truncated_fraction_trajectory": float(
                    ((novelty_feature_stats.get("trajectory") or {}).get("truncated_fraction", 0.0)) or 0.0
                ),
                "audit/novelty_feature_truncated_fraction_eval_actual": float(
                    ((novelty_feature_stats.get("eval_actual") or {}).get("truncated_fraction", 0.0)) or 0.0
                ),
                "audit/distill_survival_fraction": float(distill_meta.get("survival_fraction", 0.0) or 0.0),
                "audit/offline_grpo_survival_fraction": float(actual_meta.get("survival_fraction", 0.0) or 0.0),
            }
        )
        for path, name in (
            (dataset_manifest_path, "dataset_manifest"),
            (benchmark_manifest_path, "benchmark_dataset_manifest"),
            (runtime_path, "runtime_summary"),
            (prompt_path, "prompt_diagnostics"),
            (benchmark_prompt_path, "benchmark_prompt_diagnostics"),
            (topic_path, "topic_id_ood_summary"),
            (novelty_path, "novelty_summary"),
            (accuracy_path, "accuracy_over_time"),
            (completion_path, "completion_diagnostics"),
            (trajectory_audit_path, "trajectory_audit"),
            (eval_samples_path, "eval_samples"),
            (benchmark_summary_path, "benchmark_summary"),
            (benchmark_eval_samples_path, "benchmark_eval_samples"),
            (loss_path, "loss_diagnostics"),
        ):
            _log_wandb_artifact(wandb_run, path, name)
        for role, role_path in benchmark_rollout_paths.items():
            _log_wandb_artifact(wandb_run, role_path, f"benchmark_rollouts_{role}")


def run_pipeline(cfg: RunConfig):
    if cfg.novelty_study_mode == "full" and cfg.reward_shape != "linear":
        print("[novelty-study] Overriding reward_shape to linear for full study mode.")
        cfg.reward_shape = "linear"
    if (
        str(cfg.reward_shape or "linear").lower() == "linear"
        and float(cfg.lambda_novelty or 0.0) > 0.0
        and float(cfg.reward_incorrect or 0.0) >= 0.0
        and float(getattr(cfg, "incorrect_novelty_scale", 1.0) or 0.0) >= 1.0
    ):
        print(
            "[WARN] Linear explorer reward with non-negative incorrect reward can make "
            "wrong-but-novel completions net-positive. Consider lowering lambda_novelty, "
            "reducing incorrect_novelty_scale, making reward_incorrect negative, or switching reward shaping."
        )
    if (
        str(cfg.exploration_response_style or "reasoning_final").lower() == "answer_only"
        and int(resolved_max_completion_len(cfg, phase="explore")) > 64
    ):
        print(
            "[WARN] answer_only exploration is using a long completion budget "
            f"({resolved_max_completion_len(cfg, phase='explore')} tokens). "
            "Shorter budgets usually reduce clipped degenerate answers on CPU runs."
        )

    patch_torch_autocast_enabled_signature()
    device, is_xla, device_label = resolve_device(cfg.device_preference)
    print(f"Using device: {device_label} -> {device} (is_xla={is_xla})")
    if is_xla:
        patch_accelerate_gather_object_for_xla()
        patch_transformers_nested_xla_mesh_reduce_for_single_worker()
        patch_transformers_trainer_logging_for_single_worker()
        patch_torch_xla_device_module_for_rng()
        patch_torch_checkpoint_autocast_for_xla()
        if os.environ.get("TMX_XLA_FLASH_ATTENTION", "0") not in ("0", "false", "no", ""):
            _register_tpu_flash_attention()
        patch_torch_isin_for_xla()
        patch_transformers_isin_for_xla()
        patch_transformers_attention_mask_for_xla()
        patch_transformers_logits_processor_for_xla()
        patch_transformers_stopping_criteria_for_xla()

    set_seed(cfg.seed)
    wandb_run = init_wandb(cfg)
    use_wandb = cfg.use_wandb and wandb_run is not None

    train_pool_size = int(cfg.max_train_examples)
    if str(getattr(cfg, "train_selection_policy", "random") or "random").lower() == "base_solvable":
        configured_pool = getattr(cfg, "train_probe_pool_size", None)
        if configured_pool is None:
            configured_pool = min(max(int(cfg.max_train_examples) * 4, int(cfg.max_train_examples)), 64)
        train_pool_size = max(int(cfg.max_train_examples), int(configured_pool))

    print(f"Loading dataset subsets for '{cfg.dataset_name}'...")
    train_ds, eval_ds, dataset_manifest = build_math_dataset_bundle(
        dataset_name=cfg.dataset_name,
        max_train=train_pool_size,
        max_eval=cfg.max_eval_examples,
        dataset_config_name=cfg.dataset_config_name,
        eval_split=cfg.eval_split,
        dataset_split_train=cfg.dataset_split_train,
        dataset_split_eval=cfg.dataset_split_eval,
        topic_split_mode=cfg.topic_split_mode,
        dataset_split_policy=cfg.dataset_split_policy,
        dataset_split_seed=cfg.dataset_split_seed,
        topic_partition_policy=cfg.topic_partition_policy,
        requested_holdout_topics=cfg.id_ood_holdout_topics,
    )
    benchmark_eval_ds = None
    benchmark_manifest = None
    benchmark_eval_cfg = None
    if cfg.benchmark_dataset_name:
        benchmark_eval_cfg = _benchmark_eval_cfg(cfg)
        print(f"Loading benchmark eval subset for '{benchmark_eval_cfg.dataset_name}'...")
        benchmark_eval_ds, benchmark_manifest = build_math_eval_dataset_bundle(
            dataset_name=benchmark_eval_cfg.dataset_name,
            max_eval=benchmark_eval_cfg.max_eval_examples,
            dataset_config_name=benchmark_eval_cfg.dataset_config_name,
            dataset_split_eval=benchmark_eval_cfg.dataset_split_eval,
            topic_split_mode=benchmark_eval_cfg.topic_split_mode,
            dataset_split_policy=benchmark_eval_cfg.dataset_split_policy,
        )

    resolved_holdout_topics = dataset_manifest.get("resolved_holdout_topics", [])
    if resolved_holdout_topics:
        cfg.id_ood_holdout_topics = list(resolved_holdout_topics)
    holdout_topics = set(cfg.id_ood_holdout_topics or [])
    if holdout_topics and "topic" in train_ds.column_names:
        candidate_train_ds = train_ds.filter(lambda ex: ex["topic"] not in holdout_topics)
        if len(candidate_train_ds) <= 0:
            print(
                "[id/ood] Holdout topics would remove all training data; "
                "disabling holdout filtering for this run."
            )
            holdout_topics = set()
            cfg.id_ood_holdout_topics = []
            dataset_manifest["resolved_holdout_topics"] = []
        else:
            print(
                f"[id/ood] Holdout topics={sorted(holdout_topics)}. "
                f"Train filtered from {len(train_ds)} -> {len(candidate_train_ds)} examples."
            )
            train_ds = candidate_train_ds
            holdout_topics = set()
    train_ds = _filter_holdout_topics(train_ds, holdout_topics)
    selection_manifest = None
    selection_manifest_path = getattr(cfg, "train_selection_manifest_path", None)
    if str(getattr(cfg, "train_selection_policy", "random") or "random").lower() == "base_solvable":
        loaded_from_manifest = False
        if selection_manifest_path and os.path.exists(selection_manifest_path):
            try:
                cached_manifest = _load_train_selection_manifest(selection_manifest_path)
                train_ds, selection_manifest = _apply_train_selection_manifest(
                    train_ds,
                    cached_manifest,
                    manifest_path=selection_manifest_path,
                )
                loaded_from_manifest = True
                print(
                    "[train-selection] Loaded cached manifest: "
                    f"path={selection_manifest_path}, selected={len(train_ds)}."
                )
            except Exception as e:
                print(f"[WARN] Failed to load train-selection manifest from '{selection_manifest_path}': {e}")
        if not loaded_from_manifest:
            train_ds, selection_manifest = _select_train_subset_by_probe(
                train_ds,
                cfg=cfg,
                device=device,
                is_xla=is_xla,
                use_wandb=use_wandb,
                wandb_run=wandb_run,
            )
            if selection_manifest_path:
                _write_analysis_json(selection_manifest_path, selection_manifest)
                print(f"[train-selection] Wrote manifest to '{selection_manifest_path}'.")
        dataset_manifest["train_selection"] = selection_manifest
        dataset_manifest["train_count_selected"] = len(train_ds)
        dataset_manifest["train_row_ids"] = [int(v) for v in train_ds["__row_id__"]] if "__row_id__" in train_ds.column_names else []
        if selection_manifest.get("used_probe"):
            print(
                "[train-selection] Probe result: "
                f"solvable={int(selection_manifest.get('solvable_count', 0))}/"
                f"{int(selection_manifest.get('candidate_count', len(train_ds)))} "
                f"selected={int(selection_manifest.get('selected_count', len(train_ds)))}."
            )
    dataset_manifest["train_count_after_holdout"] = len(train_ds)
    dataset_manifest["train_row_ids_after_holdout"] = [int(v) for v in train_ds["__row_id__"]] if "__row_id__" in train_ds.column_names else []

    if cfg.prepare_train_selection_only:
        analysis_dir = _analysis_dir(cfg)
        dataset_manifest_path = cfg.dataset_manifest_path or os.path.join(analysis_dir, "dataset_manifest.json")
        _write_analysis_json(dataset_manifest_path, dataset_manifest)
        runtime_payload = {
            "status": "train_selection_prepared",
            "dataset_name": cfg.dataset_name,
            "model_name": cfg.model_name,
            "train_selection_policy": cfg.train_selection_policy,
            "train_selection_manifest_path": selection_manifest_path,
            "selected_count": int(len(train_ds)),
            "selection_manifest": selection_manifest or {},
        }
        runtime_path = os.path.join(analysis_dir, "runtime_summary.json")
        _write_analysis_json(runtime_path, runtime_payload)
        print("[train-selection] Preparation-only mode complete.")

        if wandb_run is not None:
            try:
                import wandb  # type: ignore
            except Exception:
                pass
            else:
                wandb.finish()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "base_acc": 0.0,
            "explorer_acc": 0.0,
            "actual_acc": 0.0,
            "benchmark_base_acc": 0.0,
            "benchmark_explorer_acc": 0.0,
            "benchmark_actual_acc": 0.0,
            "distill_losses": [],
        }

    overall_start = time.perf_counter()
    round_summaries = []
    result = None
    embedding_payload = {"status": "disabled"}

    if cfg.pipeline_mode == "multi_round":
        round_shards, shard_manifest = build_round_shards(
            train_ds,
            num_rounds=cfg.num_rounds,
            seed=cfg.dataset_split_seed,
            policy=cfg.round_shard_policy,
        )
        dataset_manifest["round_shards"] = shard_manifest
        current_actual_model = None
        for round_idx, shard_ds in enumerate(round_shards, start=1):
            round_cfg, budget_info = _build_round_cfg(cfg, round_idx, shard_ds)
            round_result = _run_training_round(
                round_cfg,
                shard_ds,
                eval_ds,
                benchmark_eval_ds,
                benchmark_eval_cfg,
                device=device,
                is_xla=is_xla,
                use_wandb=use_wandb,
                wandb_run=wandb_run,
                actual_init_model=current_actual_model,
            )
            round_summary = {
                "round_idx": round_idx,
                "num_examples": len(shard_ds),
                "base_accuracy": round_result["base_metrics"]["accuracy"],
                "explorer_accuracy": round_result["explorer_metrics"]["accuracy"],
                "actual_accuracy": round_result["actual_metrics"]["accuracy"],
                "base_novelty_mean": float(round_result["base_metrics"].get("novelty_mean", 0.0)),
                "explorer_novelty_mean": float(round_result["explorer_metrics"].get("novelty_mean", 0.0)),
                "actual_novelty_mean": float(round_result["actual_metrics"].get("novelty_mean", 0.0)),
                "trajectory_count": int(round_result["traj_summary"].get("traj/count", 0)),
                "accepted_count": int((round_result.get("accepted_info") or {}).get("accepted", 0)),
                "accepted_fraction": float(round_result["traj_summary"].get("traj/accepted_fraction", 0.0) or 0.0),
                "actual_target_contract": cfg.actual_target_contract,
                "round_budget": budget_info,
                "stage_timings": round_result.get("stage_timings", {}),
            }
            round_summaries.append(round_summary)
            if use_wandb and wandb_run is not None:
                wandb_run.log(
                    {
                        "multi_round/round_idx": round_idx,
                        f"multi_round/round_{round_idx}/base_accuracy": round_summary["base_accuracy"],
                        f"multi_round/round_{round_idx}/explorer_accuracy": round_summary["explorer_accuracy"],
                        f"multi_round/round_{round_idx}/actual_accuracy": round_summary["actual_accuracy"],
                        f"multi_round/round_{round_idx}/base_novelty_mean": round_summary["base_novelty_mean"],
                        f"multi_round/round_{round_idx}/explorer_novelty_mean": round_summary["explorer_novelty_mean"],
                        f"multi_round/round_{round_idx}/actual_novelty_mean": round_summary["actual_novelty_mean"],
                        f"multi_round/round_{round_idx}/accepted_count": round_summary["accepted_count"],
                        f"multi_round/round_{round_idx}/accepted_fraction": round_summary["accepted_fraction"],
                        f"multi_round/round_{round_idx}/grpo_max_steps": float(round_summary["round_budget"].get("grpo_max_steps", 0.0) or 0.0),
                        f"multi_round/round_{round_idx}/max_distill_examples": float(round_summary["round_budget"].get("max_distill_examples", 0.0) or 0.0),
                        f"multi_round/round_{round_idx}/max_actual_examples": float(round_summary["round_budget"].get("max_actual_examples", 0.0) or 0.0),
                    }
                )
            current_actual_model = round_result["actual_model"]
            result = round_result
        if result is None:
            raise RuntimeError("Multi-round pipeline produced no rounds.")
    else:
        result = _run_training_round(
            cfg,
            train_ds,
            eval_ds,
            benchmark_eval_ds,
            benchmark_eval_cfg,
            device=device,
            is_xla=is_xla,
            use_wandb=use_wandb,
            wandb_run=wandb_run,
            actual_init_model=None,
        )

    if cfg.embedding_novelty_eval and result is not None:
        try:
            embedding_payload = run_embedding_novelty_analysis(
                base_samples=result["base_metrics"].get("samples", []),
                explorer_samples=result["explorer_metrics"].get("samples", []),
                actual_samples=result["actual_metrics"].get("samples", []),
                cfg=cfg,
                plots_dir=cfg.plots_dir,
                analysis_dir=_analysis_dir(cfg),
                use_wandb=use_wandb,
                wandb_run=wandb_run,
            )
        except Exception as e:
            embedding_payload = {"status": "error", "error": str(e)}
            print(f"[WARN] Embedding novelty analysis failed: {e}")

    if result is None:
        raise RuntimeError("Pipeline did not produce results.")

    aggregated_stage_timings = (
        _aggregate_round_stage_timings(round_summaries) if round_summaries else dict(result.get("stage_timings", {}) or {})
    )
    aggregated_stage_timings["total_wall_clock_seconds"] = time.perf_counter() - overall_start
    result["stage_timings"] = aggregated_stage_timings
    extra_analysis = {}
    if round_summaries:
        extra_analysis["round_summaries"] = round_summaries
        extra_analysis["final_summary"] = {
            "num_rounds": int(len(round_summaries)),
            "round_budget_mode": str(getattr(cfg, "round_budget_mode", "constant_total") or "constant_total"),
            "final_base_accuracy": float(result["base_metrics"]["accuracy"]),
            "final_explorer_accuracy": float(result["explorer_metrics"]["accuracy"]),
            "final_actual_accuracy": float(result["actual_metrics"]["accuracy"]),
            "final_base_novelty_mean": float(result["base_metrics"].get("novelty_mean", 0.0)),
            "final_explorer_novelty_mean": float(result["explorer_metrics"].get("novelty_mean", 0.0)),
            "final_actual_novelty_mean": float(result["actual_metrics"].get("novelty_mean", 0.0)),
            "final_accepted_count": int((result.get("accepted_info") or {}).get("accepted", 0)),
            "actual_target_contract": cfg.actual_target_contract,
        }
    extra_analysis["base_model_artifact"] = result.get("base_model_artifact", {})
    extra_analysis["actual_model_artifact"] = _persist_actual_model_bundle(
        cfg,
        result,
        use_wandb=use_wandb,
        wandb_run=wandb_run,
        is_xla=is_xla,
    )
    _write_run_outputs(
        cfg=cfg,
        dataset_manifest=dataset_manifest,
        benchmark_manifest=benchmark_manifest,
        result=result,
        use_wandb=use_wandb,
        wandb_run=wandb_run,
        embedding_payload=embedding_payload,
        extra_analysis=extra_analysis,
    )

    print(f"\nDone. Check '{cfg.plots_dir}' and your W&B Charts for figures.")

    if wandb_run is not None:
        try:
            import wandb  # type: ignore
        except Exception:
            pass
        else:
            wandb.finish()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    benchmark_metrics_payload = result.get("benchmark_metrics") or {}
    _, benchmark_base_metric_value = _benchmark_primary_metric(benchmark_metrics_payload.get("base"))
    _, benchmark_explorer_metric_value = _benchmark_primary_metric(benchmark_metrics_payload.get("explorer"))
    _, benchmark_actual_metric_value = _benchmark_primary_metric(benchmark_metrics_payload.get("actual"))

    return {
        "base_acc": result["base_metrics"]["accuracy"],
        "explorer_acc": result["explorer_metrics"]["accuracy"],
        "actual_acc": result["actual_metrics"]["accuracy"],
        "benchmark_base_acc": benchmark_base_metric_value,
        "benchmark_explorer_acc": benchmark_explorer_metric_value,
        "benchmark_actual_acc": benchmark_actual_metric_value,
        "distill_losses": result["distill_losses"],
    }
