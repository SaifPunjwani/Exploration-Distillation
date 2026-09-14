"""Top-level pipeline driver: Explorer → trajectories → SFT → Actual GRPO → AIME eval.

Usage:
    TMX_VLLM_SERVER_URLS=http://ip:8000/v1,... \
    python -m tmx_jax.pipeline --wandb-run-name pipeline_lambda0_xxx ...
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
from contextlib import contextmanager
from dataclasses import asdict, replace
from typing import List

from .config import TrainConfig, parse_args, validate_contract


OFFICIAL_QWEN3_17B_AIME24 = {
    "avg_at_32": 0.491,
    "pass_at_4": 0.711,
    "pass_at_8": 0.763,
    "pass_at_16": 0.783,
    "pass_at_32": 0.800,
    "source": "user-provided official Qwen3-1.7B AIME24 table",
}


def _env_flag(name: str, default: str = "0") -> bool:
    value = (os.environ.get(name, default) or "").strip().lower()
    return value not in ("", "0", "false", "no", "off")


def _gcs_artifacts_enabled() -> bool:
    backend = (os.environ.get("TMX_ARTIFACT_BACKEND") or "hf").strip().lower()
    allow_gcs = _env_flag("TMX_ALLOW_GCS_ARTIFACTS", "0")
    return allow_gcs and backend in ("gcs", "dual", "both", "gcs+hf")


def _pipeline_wandb_run_name(cfg: TrainConfig) -> str:
    return cfg.wandb_run_name or "jax_pipeline"


def _pipeline_wandb_run_id(cfg: TrainConfig) -> str:
    if os.environ.get("TMX_WANDB_RUN_ID"):
        return os.environ["TMX_WANDB_RUN_ID"].strip()
    run_name = _pipeline_wandb_run_name(cfg)
    return "jaxe2e-" + hashlib.sha1(f"{run_name}:e2e".encode("utf-8")).hexdigest()[:24]


def _wandb_log_pipeline(cfg: TrainConfig, payload: dict, *, step: int | None = None) -> None:
    """Log pipeline-level metrics to a stable companion W&B run.

    Stage trainers own their own W&B runs. This helper mirrors the Torch/XLA
    pipeline-level metrics without keeping a W&B run open while sub-stages call
    ``wandb.init`` themselves.
    """
    if _env_flag("TMX_DISABLE_WANDB_NONZERO", "1"):
        process_id = (os.environ.get("JAX_PROCESS_ID") or "0").strip()
        try:
            import jax

            if jax.distributed.is_initialized():
                process_id = str(int(jax.process_index()))
        except Exception:
            pass
        if process_id and process_id != "0":
            return
        if (os.environ.get("WANDB_MODE") or "").strip().lower() == "disabled":
            os.environ.pop("WANDB_MODE", None)
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
            group=os.environ.get("TMX_WANDB_GROUP") or run_name,
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
    format_quality = _values("format_quality")
    format_penalty = _values("format_penalty")
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
        "traj/prompt_truncated_fraction": 0.0,
        "traj/normalization_changed_fraction": 0.0,
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


def collect_accepted(cfg: TrainConfig, traj_jsonl_path: str) -> str:
    """Filter Explorer rollouts to accepted (is_correct==True) completions.

    Input: jsonl where each line is {prompt_text, completion_text, is_correct, ...}.
    Output: sibling file with suffix `.accepted.jsonl` containing selected correct rows.
    """
    accepted: List[dict] = []
    with open(traj_jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("is_correct"):
                accepted.append(r)

    policy = (cfg.accepted_selection_policy or "quality_first").strip().lower()
    if policy == "quality_first":
        def quality_key(r: dict):
            # Prefer correct traces that actually terminated and expose a parseable answer.
            return (
                1 if r.get("terminated") else 0,
                1 if r.get("valid_answer") else 0,
                0 if r.get("clipped") else 1,
                float(r.get("blended_reward", r.get("reward", 0.0)) or 0.0),
                -float(r.get("completion_token_length", 0.0) or 0.0),
            )
        accepted.sort(key=quality_key, reverse=True)
    elif policy != "chronological":
        raise ValueError(f"Unsupported accepted_selection_policy={cfg.accepted_selection_policy!r}")

    min_required = int(cfg.min_accepted_trajectories)
    if len(accepted) < min_required:
        raise RuntimeError(
            f"Only {len(accepted)} accepted trajectories available, below "
            f"min_accepted_trajectories={min_required}. Lower the threshold for a smoke run "
            f"or continue Explorer collection."
        )

    max_selected = int(cfg.max_accepted_trajectories)
    if max_selected <= 0:
        max_selected = int(cfg.max_distill_examples)
    max_selected = max(1, min(max_selected, int(cfg.max_distill_examples)))
    selected = accepted[:max_selected]

    out_path = traj_jsonl_path.replace(".jsonl", ".accepted.jsonl")
    with open(out_path, "w") as f:
        for r in selected:
            f.write(json.dumps(r) + "\n")
    print(
        f"[pipeline] accepted {len(selected)}/{len(accepted)} correct trajectories "
        f"(min_required={min_required}, policy={policy}) -> {out_path}",
        flush=True,
    )
    return out_path


def _maybe_export_state_to_hf_dir(state, cfg: TrainConfig, output_dir: str) -> str | None:
    """Export a trained Flax/JAX state to an HF-style model directory.

    This runs only on process 0 so multihost launches do not duplicate export work.
    """
    try:
        import jax
        import numpy as np
        from .train import _make_qwen3_config
        from .weights import export_flax_params_to_hf_dir
    except Exception as e:
        print(f"[pipeline] WARNING: cannot import JAX/HF export helpers: {e}", flush=True)
        return None

    if jax.process_index() != 0:
        return None

    if state is None or not hasattr(state, "params"):
        print("[pipeline] WARNING: no train state available for HF export.", flush=True)
        return None

    try:
        m_cfg = _make_qwen3_config(cfg.model_name)
        params_host = jax.tree_util.tree_map(lambda x: np.asarray(x), state.params)
        out = os.path.abspath(output_dir)
        export_flax_params_to_hf_dir(
            params_host,
            cfg.model_name,
            int(m_cfg.num_hidden_layers),
            bool(m_cfg.tie_word_embeddings),
            out,
            use_qk_norm=bool(m_cfg.use_qk_norm),
            hf_weight_prefix=str(m_cfg.hf_weight_prefix),
        )
        print(f"[pipeline] exported HF model bundle -> {out}", flush=True)
        return out
    except Exception as e:
        print(f"[pipeline] WARNING: HF export failed: {e}", flush=True)
        return None


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


def _hf_artifact_repo() -> str:
    return (
        os.environ.get("TMX_HF_CHECKPOINT_REPO")
        or os.environ.get("TMX_HF_ARTIFACT_REPO")
        or ""
    ).strip()


def _hf_run_prefix(cfg: TrainConfig) -> str:
    return (os.environ.get("TMX_HF_ARTIFACT_PREFIX") or _pipeline_wandb_run_name(cfg)).strip().strip("/")


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


def _run_hf_upload_child(env: dict, code: str, cfg: TrainConfig, path_in_repo: str, source: str) -> bool:
    require = _env_flag("TMX_REQUIRE_HF_ARTIFACT_UPLOAD", "0")
    async_upload = _env_flag("TMX_HF_MIRROR_ASYNC", "1") and not require
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
            print(f"[pipeline] scheduled HF upload -> {env['TMX_HF_UPLOAD_REPO_ID']}/{path_in_repo} log={log_path}", flush=True)
            return True
        except Exception as exc:
            print(f"[pipeline] WARNING: failed to schedule HF upload for {path_in_repo}: {type(exc).__name__}: {exc}", flush=True)
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
        print(f"[pipeline] uploaded HF artifact -> {env['TMX_HF_UPLOAD_REPO_ID']}/{path_in_repo}", flush=True)
        return True
    print(f"[pipeline] WARNING: HF upload failed for {path_in_repo}: {(ret.stderr or ret.stdout)[-1000:]}", flush=True)
    if require:
        raise RuntimeError(f"HF upload failed for {path_in_repo}")
    return False


def _maybe_upload_path_to_hf(local_path: str, cfg: TrainConfig, path_in_repo: str, *, is_dir: bool) -> bool:
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
    return _run_hf_upload_child(env, _HF_LOCAL_UPLOAD_CODE, cfg, path_in_repo, os.path.abspath(local_path))


def _maybe_upload_gcs_uri_to_hf(gcs_uri: str, cfg: TrainConfig, path_in_repo: str, *, is_dir: bool) -> bool:
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
    return _run_hf_upload_child(env, _HF_GCS_UPLOAD_CODE, cfg, path_in_repo, gcs_uri)


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
    summary.setdefault("official_baseline", OFFICIAL_QWEN3_17B_AIME24)
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
    gcs_base = (os.environ.get("TMX_GCS_CHECKPOINT_BASE") or "").strip().rstrip("/")
    gcs_target = ""
    uploaded_to_gcs = False
    if gcs_base and _gcs_artifacts_enabled():
        gcs_target = f"{gcs_base}/run_summary.json"
        uploaded_to_gcs = _maybe_upload_file_to_gcs(path, gcs_target)
    if _env_flag("TMX_HF_MIRROR_RUN_SUMMARY", "1"):
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
    tpu_name = (os.environ.get("TMX_VLLM_RELOAD_TPU_NAME") or "").strip()
    zone = (os.environ.get("TMX_VLLM_RELOAD_ZONE") or "").strip()
    workers = (os.environ.get("TMX_VLLM_RELOAD_WORKERS") or "").strip()
    required = _env_flag("TMX_EXTERNAL_VLLM_RELOAD_REQUIRED", "1")
    retry_on_fail = _env_flag("TMX_EXTERNAL_VLLM_RELOAD_RETRY_ON_FAIL", "1")
    if not (tpu_name and zone and workers):
        msg = f"vLLM slice reload env not set for {stage}"
        if required:
            raise RuntimeError(msg)
        print(f"[pipeline] WARNING: {msg}; continuing with existing vLLM", flush=True)
        return None

    server_urls = (
        os.environ.get("TMX_VLLM_RELOAD_SERVER_URLS")
        or os.environ.get("TMX_VLLM_SERVER_URLS")
        or cfg.vllm_server_urls
    ).strip()
    mode = (
        os.environ.get("TMX_PIPELINE_VLLM_RELOAD_MODE")
        or os.environ.get("TMX_TRAIN_VLLM_RELOAD_MODE")
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
            "LOCAL_MODEL_DIR_BASENAME": f"tmx_{stage}_model",
            "MAX_MODEL_LEN": str(int(max_model_len)),
            "MAX_NUM_SEQS": str(int(max_num_seqs)),
            "MAX_NUM_BATCHED_TOKENS": os.environ.get("TMX_VLLM_RELOAD_MAX_BATCHED_TOKENS", "65536"),
            "VLLM_ENABLE_CHUNKED_PREFILL": os.environ.get("VLLM_ENABLE_CHUNKED_PREFILL", "1"),
            "PARALLELISM": os.environ.get("TMX_VLLM_RELOAD_PARALLELISM", "8"),
        }
    )
    timeout_s = float(os.environ.get("TMX_EXTERNAL_VLLM_RELOAD_TIMEOUT", "0") or "0")
    retry_s = float(os.environ.get("TMX_EXTERNAL_VLLM_RELOAD_RETRY_SECONDS", "30") or "30")
    attempt_timeout_s = float(os.environ.get("TMX_VLLM_RELOAD_ATTEMPT_TIMEOUT", "7200") or "7200")
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
                ["bash", "tmx_jax/scripts/reload_vllm_direct_slice.sh"],
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
    tpu_name = (os.environ.get("TMX_VLLM_RELOAD_TPU_NAME") or "").strip()
    zone = (os.environ.get("TMX_VLLM_RELOAD_ZONE") or "").strip()
    workers = (os.environ.get("TMX_VLLM_RELOAD_WORKERS") or "").strip()
    required = _env_flag("TMX_EXTERNAL_VLLM_RELOAD_REQUIRED", "1")
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
            os.environ.get("TMX_VLLM_RELOAD_TPU_PROJECT")
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

    if not _env_flag("TMX_EXTERNAL_VLLM_RELOAD_WAIT", "1"):
        return server_urls or None
    timeout_s = float(os.environ.get("TMX_EXTERNAL_VLLM_RELOAD_TIMEOUT", "0") or "0")
    poll_s = float(os.environ.get("TMX_EXTERNAL_VLLM_RELOAD_POLL_SECONDS", "5") or "5")
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
    reload_url = (os.environ.get("TMX_VLLM_RELOAD_WORKER_URL") or "").strip()
    reload_tpu_name = (os.environ.get("TMX_VLLM_RELOAD_TPU_NAME") or "").strip()
    reload_zone = (os.environ.get("TMX_VLLM_RELOAD_ZONE") or "").strip()
    if not (reload_url and reload_tpu_name and reload_zone):
        return

    reload_worker = (os.environ.get("TMX_VLLM_RELOAD_WORKER") or "0").strip() or "0"
    served_model_name = (os.environ.get("TMX_ACTUAL_VLLM_MODEL_NAME") or "actual_model").strip() or "actual_model"

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
            ["bash", "scripts/reload_vllm_with_model.sh"],
            capture_output=True,
            text=True,
            timeout=1800,
            env=env,
        )
        if ret.returncode == 0:
            os.environ["TMX_ACTUAL_VLLM_SERVER_URL"] = reload_url
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
    cluster_detection_method = os.environ.get("TMX_JAX_CLUSTER_DETECTION_METHOD", "").strip()
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
    validate_contract(cfg, require_eval_contract=True)
    _early_init_jax_distributed_from_env()
    root_output_dir = os.path.abspath(cfg.output_dir)
    pipeline_stage = (os.environ.get("TMX_PIPELINE_STAGE") or "explorer_novelty").strip()
    allowed_stages = {"explorer_correctness", "explorer_novelty", "actual_sft", "actual_grpo", "eval"}
    if pipeline_stage not in allowed_stages:
        raise ValueError(f"unsupported TMX_PIPELINE_STAGE={pipeline_stage!r}; expected one of {sorted(allowed_stages)}")
    if pipeline_stage == "actual_sft":
        os.environ.setdefault("TMX_PIPELINE_STOP_AFTER_SFT", "1")
    if pipeline_stage in {"actual_sft", "actual_grpo"}:
        if not _env_flag("TMX_PIPELINE_SKIP_EXPLORER", "0"):
            raise RuntimeError(
                f"{pipeline_stage} requires TMX_PIPELINE_SKIP_EXPLORER=1 so it cannot accidentally rerun Explorer"
            )
        if not (os.environ.get("TMX_EXISTING_TRAJECTORIES") or "").strip():
            raise RuntimeError(
                f"{pipeline_stage} requires TMX_EXISTING_TRAJECTORIES pointing at HF/local Explorer trajectories"
            )
    if (os.environ.get("TMX_WANDB_SINGLE_RUN", "1") or "").strip().lower() not in ("0", "false", "no", "off"):
        e2e_run_id = _pipeline_wandb_run_id(cfg)
        e2e_run_name = _pipeline_wandb_run_name(cfg)
        os.environ.setdefault("TMX_WANDB_RUN_ID", e2e_run_id)
        os.environ.setdefault("TMX_WANDB_EXPLORER_RUN_ID", e2e_run_id)
        os.environ.setdefault("TMX_WANDB_DISTILL_RUN_ID", e2e_run_id)
        os.environ.setdefault("TMX_WANDB_ACTUAL_RUN_ID", e2e_run_id)
        os.environ.setdefault("TMX_WANDB_ACTUAL_GRPO_RUN_ID", e2e_run_id)
        os.environ.setdefault("TMX_WANDB_GROUP", e2e_run_name)
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
        "dynamic_sampling": bool(cfg.dynamic_sampling),
        "dynamic_sampling_max_attempts": int(cfg.dynamic_sampling_max_attempts),
        "mask_truncated_completions": bool(cfg.mask_truncated_completions),
        "actual_online_grpo_max_steps": int(cfg.actual_online_grpo_max_steps),
        "min_accepted_trajectories": int(cfg.min_accepted_trajectories),
        "max_accepted_trajectories": int(cfg.max_accepted_trajectories),
        "distill_epochs": int(cfg.distill_epochs),
        "max_distill_examples": int(cfg.max_distill_examples),
    })
    explorer_cfg = replace(
        cfg,
        output_dir=os.path.join(root_output_dir, "explorer"),
        wandb_run_name=_pipeline_wandb_run_name(cfg),
    )

    from . import train as _train_mod
    skip_explorer = _env_flag("TMX_PIPELINE_SKIP_EXPLORER", "0")
    existing_traj = (os.environ.get("TMX_EXISTING_TRAJECTORIES") or "").strip()
    if skip_explorer:
        os.makedirs(explorer_cfg.output_dir, exist_ok=True)
        traj_dst = os.path.join(explorer_cfg.output_dir, "explorer_trajectories.jsonl")
        if existing_traj and existing_traj != traj_dst:
            if existing_traj.startswith("hf://"):
                from huggingface_hub import hf_hub_download

                raw = existing_traj[len("hf://") :].strip("/")
                parts = raw.split("/")
                if len(parts) < 3:
                    raise ValueError(
                        "HF trajectory URI must be hf://owner/repo/path/to/file; "
                        f"got {existing_traj!r}"
                    )
                repo_id = "/".join(parts[:2])
                filename = "/".join(parts[2:])
                downloaded = hf_hub_download(
                    repo_id=repo_id,
                    repo_type=os.environ.get("TMX_HF_REPO_TYPE", "dataset") or "dataset",
                    filename=filename,
                )
                shutil.copyfile(downloaded, traj_dst)
            elif existing_traj.startswith("gs://"):
                if not _gcs_artifacts_enabled() and not _env_flag("TMX_ALLOW_GCS_EXISTING_TRAJECTORIES", "0"):
                    raise RuntimeError(
                        "Refusing gs:// existing trajectories in HF-only mode. "
                        "Use hf:// or set TMX_ALLOW_GCS_EXISTING_TRAJECTORIES=1 explicitly."
                    )
                ret = subprocess.run(
                    ["gsutil", "cp", existing_traj, traj_dst],
                    capture_output=True,
                    text=True,
                    timeout=900,
                )
                if ret.returncode != 0:
                    raise RuntimeError(f"failed to fetch existing trajectories: {ret.stderr[:500]}")
            else:
                shutil.copyfile(os.path.abspath(existing_traj), traj_dst)
        if not os.path.exists(traj_dst):
            raise FileNotFoundError(
                "TMX_PIPELINE_SKIP_EXPLORER=1 but no explorer_trajectories.jsonl exists "
                f"at {traj_dst} and TMX_EXISTING_TRAJECTORIES was not usable."
            )
        print(f"[pipeline] skipping Explorer; resuming from trajectories: {traj_dst}", flush=True)
        _wandb_log_pipeline(cfg, {
            "pipeline/explorer_skipped": 1.0,
            "pipeline/resumed_from_existing_trajectories": 1.0,
        })
    else:
        # Stage 1: Explorer GRPO (already a __main__ in tmx_jax.train)
        explorer_state = _train_mod.run_training(explorer_cfg)
        explorer_hf_dir = _maybe_export_state_to_hf_dir(
            explorer_state,
            explorer_cfg,
            os.path.join(explorer_cfg.output_dir, "explorer_hf_final"),
        )
        if explorer_hf_dir is not None:
            gcs_base = (os.environ.get("TMX_GCS_CHECKPOINT_BASE") or "").strip()
            explorer_hf_gcs = None
            if gcs_base and _gcs_artifacts_enabled():
                explorer_hf_gcs = _maybe_upload_dir_to_gcs(
                    explorer_hf_dir,
                    f"{gcs_base.rstrip('/')}/explorer_hf_final",
                )
                if explorer_hf_gcs:
                    _wandb_log_pipeline(cfg, {"pipeline/explorer_hf_uploaded": 1.0})
            if _env_flag("TMX_HF_MIRROR_FINAL_EXPORTS", "1"):
                hf_path = f"{_hf_run_prefix(cfg)}/explorer_hf_final"
                if explorer_hf_gcs:
                    _maybe_upload_gcs_uri_to_hf(explorer_hf_gcs, cfg, hf_path, is_dir=True)
                else:
                    _maybe_upload_path_to_hf(explorer_hf_dir, cfg, hf_path, is_dir=True)

    if _env_flag("TMX_PIPELINE_STOP_AFTER_EXPLORER", "0"):
        print("[pipeline] TMX_PIPELINE_STOP_AFTER_EXPLORER=1; stopping after Explorer stage", flush=True)
        _wandb_log_pipeline(cfg, {
            "pipeline/stopped_after_explorer": 1.0,
            "pipeline/sft_skipped_by_stage": 1.0,
            "pipeline/actual_grpo_skipped_by_stage": 1.0,
        })
        _write_run_summary(root_output_dir, cfg, "stopped_after_explorer", {
            "status": "stopped_after_explorer",
            "stage": os.environ.get("TMX_PIPELINE_STAGE", "explorer"),
            "explorer_output_dir": explorer_cfg.output_dir,
            "reason": "TMX_PIPELINE_STOP_AFTER_EXPLORER",
        })
        return

    # Stage 2: filter accepted trajectories
    traj_path = os.path.join(explorer_cfg.output_dir, "explorer_trajectories.jsonl")
    if not os.path.exists(traj_path):
        print(f"[pipeline] WARNING: {traj_path} missing (Explorer wrote to WandB only). "
              f"Skipping SFT + Actual GRPO.", flush=True)
        _wandb_log_pipeline(cfg, {
            "pipeline/skipped_after_explorer": 1.0,
            "pipeline/skip_reason/missing_explorer_trajectories": 1.0,
        })
        return

    accepted = collect_accepted(cfg, traj_path)
    traj_payload = _summarize_trajectory_jsonl(traj_path, num_generations=cfg.grpo_num_generations)
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
            "audit/novelty_feature_truncated_fraction_trajectory": 0.0,
            "audit/novelty_feature_truncated_fraction_eval_actual": 0.0,
            "audit/distill_survival_fraction": 1.0,
            "audit/offline_grpo_survival_fraction": 1.0,
        },
    )
    _write_run_summary(root_output_dir, cfg, "explorer_trajectory_summary", {
        "trajectory_jsonl": traj_path,
        "accepted_jsonl": accepted,
        "trajectory_metrics": traj_payload,
        "accepted_metrics": accepted_payload,
        "accepted_count": accepted_count,
        "accepted_fraction": accepted_count / max(1.0, total_count),
        "selection_policy": cfg.accepted_selection_policy,
    })

    # Stage 3: SFT distillation
    from . import distill as _distill_mod
    actual_cfg = replace(
        cfg,
        # Novelty is an Explorer exploration bonus. Actual starts from Base via
        # SFT on accepted Explorer traces, then trains online on correctness-only
        # DR-GRPO for both lambda=0 and lambda>0 Explorer runs.
        lambda_novelty=0.0,
        incorrect_novelty_scale=0.0,
        gate_novelty_by_quality=False,
        novelty_zscore="off",
        output_dir=os.path.join(root_output_dir, "actual"),
        wandb_run_name=_pipeline_wandb_run_name(cfg),
    )
    actual_sft_state = _distill_mod.run_sft(actual_cfg, accepted)
    _wandb_log_pipeline(cfg, {"pipeline/sft_complete": 1.0})
    _write_run_summary(root_output_dir, cfg, "sft", {
        "status": "complete",
        "accepted_jsonl": accepted,
        "actual_sft_checkpoint": os.path.join(actual_cfg.output_dir, "actual_sft_final"),
        "distill_epochs": int(cfg.distill_epochs),
        "max_distill_examples": int(cfg.max_distill_examples),
    })

    actual_grpo_vllm_urls = cfg.vllm_server_urls
    sft_hf_dir = _maybe_export_state_to_hf_dir(
        actual_sft_state,
        actual_cfg,
        os.path.join(actual_cfg.output_dir, "actual_sft_hf"),
    )
    if sft_hf_dir is not None:
        gcs_base = (os.environ.get("TMX_GCS_CHECKPOINT_BASE") or "").strip()
        sft_hf_gcs = None
        sft_hf_repo_id = _hf_artifact_repo()
        sft_hf_path = f"{_hf_run_prefix(cfg)}/actual_sft_hf"
        sft_hf_uploaded = False
        if gcs_base and _gcs_artifacts_enabled():
            sft_hf_gcs = _maybe_upload_dir_to_gcs(
                sft_hf_dir,
                f"{gcs_base.rstrip('/')}/actual_sft_hf",
            )
        if _env_flag("TMX_HF_MIRROR_FINAL_EXPORTS", "0"):
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
                max_num_seqs=int(os.environ.get("TMX_ACTUAL_GRPO_VLLM_MAX_NUM_SEQS", "32")),
                stage="actual_sft",
                hf_repo_id=sft_hf_repo_id if sft_hf_uploaded else "",
                hf_path_in_repo=sft_hf_path if sft_hf_uploaded else "",
                hf_repo_type="dataset",
            )
            if reloaded_urls:
                actual_grpo_vllm_urls = reloaded_urls

    if _env_flag("TMX_PIPELINE_STOP_AFTER_SFT", "0"):
        print("[pipeline] TMX_PIPELINE_STOP_AFTER_SFT=1; stopping after SFT stage", flush=True)
        _wandb_log_pipeline(cfg, {
            "pipeline/stopped_after_sft": 1.0,
            "pipeline/actual_grpo_skipped_by_stage": 1.0,
        })
        _write_run_summary(root_output_dir, cfg, "stopped_after_sft", {
            "status": "stopped_after_sft",
            "stage": pipeline_stage,
            "actual_output_dir": actual_cfg.output_dir,
            "reason": "TMX_PIPELINE_STOP_AFTER_SFT",
        })
        return

    if _env_flag("TMX_PIPELINE_CLEAR_MEMORY_BEFORE_ACTUAL", "1"):
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
        grpo_kl_beta=cfg.actual_grpo_kl_coef,
        init_checkpoint=os.path.join(actual_cfg.output_dir, "actual_sft_final"),
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
        gcs_base = (os.environ.get("TMX_GCS_CHECKPOINT_BASE") or "").strip()
        actual_hf_repo_id = _hf_artifact_repo()
        actual_hf_path = f"{_hf_run_prefix(cfg)}/actual_hf_final"
        if gcs_base and _gcs_artifacts_enabled():
            actual_hf_gcs = _maybe_upload_dir_to_gcs(
                actual_hf_dir,
                f"{gcs_base.rstrip('/')}/actual_hf_final",
            )
        if _env_flag("TMX_HF_MIRROR_FINAL_EXPORTS", "0"):
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
            if _env_flag("TMX_SKIP_FINAL_EVAL", "0"):
                print(
                    f"[pipeline] TMX_SKIP_FINAL_EVAL=1; saved Actual HF bundle at {actual_source_checkpoint} and skipping AIME dispatch.",
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
                max_model_len=int(os.environ.get("TMX_FINAL_EVAL_MAX_MODEL_LEN", "40960")),
                max_num_seqs=int(os.environ.get("TMX_FINAL_EVAL_VLLM_MAX_NUM_SEQS", "4")),
                stage="actual_final",
                hf_repo_id=actual_hf_repo_id if actual_hf_uploaded else "",
                hf_path_in_repo=actual_hf_path if actual_hf_uploaded else "",
                hf_repo_type="dataset",
            )
            if final_urls:
                os.environ["TMX_ACTUAL_VLLM_SERVER_URLS"] = final_urls
            elif actual_hf_gcs:
                _maybe_reload_actual_vllm(actual_hf_gcs)

    if _env_flag("TMX_SKIP_FINAL_EVAL", "0"):
        source = actual_source_checkpoint or actual_hf_dir or actual_grpo_cfg.output_dir
        print(
            f"[pipeline] TMX_SKIP_FINAL_EVAL=1; Actual stage complete at {source} and skipping AIME dispatch.",
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

    # Stage 5: AIME24 benchmark eval
    from .generate import parse_server_urls
    from .eval import aime24_wandb_metrics, run_eval
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    actual_eval_server_urls = (
        os.environ.get("TMX_ACTUAL_VLLM_SERVER_URL") or
        os.environ.get("TMX_ACTUAL_VLLM_SERVER_URLS") or
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

    eval_temperature = float(os.environ.get("TMX_FINAL_EVAL_TEMPERATURE", "0.6") or "0.6")
    eval_top_p = float(os.environ.get("TMX_FINAL_EVAL_TOP_P", str(cfg.top_p)) or str(cfg.top_p))
    eval_top_k = int(os.environ.get("TMX_FINAL_EVAL_TOP_K", str(cfg.top_k)) or str(cfg.top_k))
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
        "official_baseline": OFFICIAL_QWEN3_17B_AIME24,
    }
    out = os.path.join(root_output_dir, "final_eval_aime24.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[pipeline] AIME24 avg@{cfg.benchmark_num_rollouts} = {res['avg_at_n']*100:.2f}%  (base=49.1%)", flush=True)
    gcs_base = (os.environ.get("TMX_GCS_CHECKPOINT_BASE") or "").strip().rstrip("/")
    gcs_target = ""
    uploaded_to_gcs = False
    if gcs_base and _gcs_artifacts_enabled():
        gcs_target = f"{gcs_base}/final_eval_aime24.json"
        uploaded_to_gcs = _maybe_upload_file_to_gcs(out, gcs_target)
    if _env_flag("TMX_HF_MIRROR_FINAL_EVAL", "1"):
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
        "benchmark/aime24/base_avg_at_32": OFFICIAL_QWEN3_17B_AIME24["avg_at_32"],
        "benchmark/aime24/actual_minus_base_avg_at_32": (
            float(res["avg_at_n"]) - OFFICIAL_QWEN3_17B_AIME24["avg_at_32"]
        ),
    }
    eval_payload.update(aime24_wandb_metrics(res))
    for k in (1, 4, 8, 16, 32):
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
        "official_baseline": OFFICIAL_QWEN3_17B_AIME24,
        "actual_minus_official_avg_at_32": float(res["avg_at_n"]) - OFFICIAL_QWEN3_17B_AIME24["avg_at_32"],
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
    if cfg is not None and _env_flag("TMX_HF_MIRROR_CHECKPOINTS", "0"):
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
    """Build a per-round config: scoped output_dir, shifted seed, split budgets, init from prev Actual."""
    round_dir = os.path.join(root_output_dir, f"round_{round_idx}")
    base_run_name = _pipeline_wandb_run_name(cfg)
    return replace(
        cfg,
        output_dir=round_dir,
        wandb_run_name=f"{base_run_name}_round_{round_idx:02d}",
        # Different seed per round controls stochastic sampling inside the
        # round. Disjoint dataset shards are enforced through TMX_DATASET_SHARD_*
        # env vars set by multi_round_main.
        seed=int(cfg.seed) + (round_idx - 1),
        # Split step/example budgets across rounds when budget_mode="split".
        grpo_max_steps=_round_budget(cfg.grpo_max_steps, round_idx, num_rounds, cfg.round_budget_mode),
        actual_online_grpo_max_steps=_round_budget(
            cfg.actual_online_grpo_max_steps, round_idx, num_rounds, cfg.round_budget_mode
        ),
        max_distill_examples=_round_budget(
            cfg.max_distill_examples, round_idx, num_rounds, cfg.round_budget_mode
        ),
        # Round 1: init_checkpoint="" → load Base via cfg.model_name. Round R+1:
        # init_checkpoint = prev round's final Actual GRPO checkpoint (Orbax).
        init_checkpoint=init_checkpoint or "",
    )


def multi_round_main():
    """Top-level entrypoint. Dispatches to single-round (two_model) or multi-round."""
    cfg = parse_args()
    if (cfg.pipeline_mode or "two_model").lower() != "multi_round" or int(cfg.num_rounds) <= 1:
        # Standard single-pass two-model pipeline (unchanged behavior).
        return main(cfg)

    # Multi-round: orchestrate N rounds with Explorer←Actual handoff.
    num_rounds = int(cfg.num_rounds)
    root_output_dir = os.path.abspath(cfg.output_dir)
    base_hf_prefix = (
        os.environ.get("TMX_HF_ARTIFACT_PREFIX")
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
            f"output_dir={round_cfg.output_dir} init_checkpoint={round_cfg.init_checkpoint!r} "
            f"grpo_max_steps={round_cfg.grpo_max_steps} actual_max_steps={round_cfg.actual_online_grpo_max_steps}",
            flush=True,
        )
        round_env = {
            # Fresh novelty each round; keep this explicit so restored policy
            # weights do not bring along stale RND predictor state.
            "TMX_MULTIROUND_RESET_RND": "1" if cfg.multi_round_reset_rnd else "0",
            # Cross-round Explorer should inherit Actual params only, not the
            # previous Actual optimizer moments.
            "TMX_RESTORE_OPT_STATE": "0",
            # Ensure a final checkpoint exists even when final_step is not on
            # the periodic save cadence.
            "TMX_FORCE_FINAL_PHASE_CHECKPOINT": "1",
            # Deterministic disjoint train-data shards.
            "TMX_DATASET_SHARD_INDEX": str(round_idx - 1),
            "TMX_DATASET_SHARD_COUNT": str(num_rounds),
            "TMX_DATASET_SHARD_POLICY": cfg.round_shard_policy,
            "TMX_DATASET_SHARD_SEED": str(int(cfg.seed)),
            # Avoid checkpoint-name collisions across rounds in HF artifact
            # storage and make hf:// handoff paths unambiguous.
            "TMX_HF_ARTIFACT_PREFIX": f"{base_hf_prefix}/round_{round_idx:02d}",
        }
        if _env_flag("TMX_HF_MIRROR_CHECKPOINTS", "0"):
            # The next round may restore immediately; async upload is unsafe
            # for handoff checkpoints.
            round_env["TMX_HF_MIRROR_ASYNC"] = "0"
        if round_idx < num_rounds and not _env_flag("TMX_MULTIROUND_EVAL_EACH_ROUND", "0"):
            round_env["TMX_SKIP_FINAL_EVAL"] = "1"
        blocked_env = (
            "TMX_PIPELINE_STOP_AFTER_EXPLORER",
            "TMX_PIPELINE_STOP_AFTER_SFT",
            "TMX_PIPELINE_SKIP_EXPLORER",
            "TMX_EXISTING_TRAJECTORIES",
            "TMX_INIT_CHECKPOINT_LOCAL_PATH",
            "TMX_SKIP_FINAL_EVAL",
            "TMX_WANDB_RUN_ID",
            "TMX_WANDB_EXPLORER_RUN_ID",
            "TMX_WANDB_DISTILL_RUN_ID",
            "TMX_WANDB_ACTUAL_RUN_ID",
            "TMX_WANDB_ACTUAL_GRPO_RUN_ID",
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
            if next_ckpt.startswith("hf://") is False and _shared_handoff_required():
                raise RuntimeError(
                    "multi-round handoff resolved to a local checkpoint in a multihost run: "
                    f"{next_ckpt}. Enable TMX_HF_MIRROR_CHECKPOINTS=1 and set "
                    "TMX_HF_CHECKPOINT_REPO/TMX_HF_ARTIFACT_REPO so every host can restore."
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
