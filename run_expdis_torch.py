#!/usr/bin/env python
"""
Entry point for Exploration-Distillation (Explorer + Actual) with TPU support.

Key flags (see README.md for more):
  --device {auto,cuda,cpu,tpu}
  --dataset-name {gsm8k,dapo_math_17k,deepscaler_preview,aime_2024,aime_2025}
  --grpo-loss-type {dr_grpo,grpo,dapo}
  --lambda-novelty or --lambda-sweep "0.0,0.1,0.3"
  --novelty-metric {mse,sqrt_mse,l2,l1,cosine}, --novelty-clip
  --novelty-feature-source {last,multilayer} --novelty-layers "6,12,18"
  --reward-shape {linear,piecewise}
  --actual-train-mode {grpo,distill,sft_grpo}
  --temperature-schedule "1.0,0.8,0.6"
"""

import os
import sys

from expdis_torch.config import build_configs_from_args, parse_args


def _load_env_file(path: str = ".env") -> None:
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except FileNotFoundError:
        return


def _ensure_hf_runtime_env(root_dir: str = ".") -> None:
    """
    Set robust Hugging Face runtime defaults for tokens and caches.
    Existing env vars always take precedence.
    """
    hf_access_token = os.environ.get("HF_ACCESS_TOKEN")
    if hf_access_token and not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = hf_access_token

    if not (
        os.environ.get("HF_HOME")
        or os.environ.get("HF_HUB_CACHE")
        or os.environ.get("HF_DATASETS_CACHE")
        or os.environ.get("TRANSFORMERS_CACHE")
    ):
        hf_home = os.path.join(os.path.abspath(root_dir), ".hf")
        os.environ["HF_HOME"] = hf_home
        os.environ["HF_HUB_CACHE"] = os.path.join(hf_home, "hub")
        os.environ["HF_DATASETS_CACHE"] = os.path.join(hf_home, "datasets")
        os.environ["TRANSFORMERS_CACHE"] = os.path.join(hf_home, "transformers")

    for key in ("HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE"):
        val = os.environ.get(key)
        if val:
            os.makedirs(val, exist_ok=True)


def _maybe_initialize_xla_persistent_cache() -> None:
    cache_dir = (os.environ.get("EXPDIS_XLA_PERSISTENT_CACHE_DIR") or "").strip()
    if cache_dir.lower() in {"0", "false", "no", "off", "disable", "__disable__"}:
        return
    if not cache_dir:
        return
    if os.environ.get("EXPDIS_XLA_PERSISTENT_CACHE_INITIALIZED") == "1":
        return
    if os.environ.get("PJRT_DEVICE", "").upper() != "TPU":
        return
    readonly = str(os.environ.get("EXPDIS_XLA_PERSISTENT_CACHE_READONLY", "0")).strip().lower() in (
        "1",
        "true",
        "yes",
    )
    try:
        os.makedirs(cache_dir, exist_ok=True)
        import torch_xla.runtime as xr  # type: ignore

        xr.initialize_cache(cache_dir, readonly=readonly)
        os.environ["EXPDIS_XLA_PERSISTENT_CACHE_INITIALIZED"] = "1"
        print(f"[xla-cache] initialized persistent cache at {cache_dir} readonly={readonly}")
    except Exception as exc:
        print(f"[WARN] Failed to initialize XLA persistent cache at {cache_dir}: {exc}")


def _parse_positive_int_env(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        print(f"[WARN] Invalid {name}={raw!r}; using {default}.")
        return default
    return max(1, value)


def _argv_requests_tpu(argv: list[str]) -> bool:
    for idx, arg in enumerate(argv):
        if arg == "--device" and idx + 1 < len(argv):
            return argv[idx + 1].lower() == "tpu"
        if arg.startswith("--device="):
            return arg.split("=", 1)[1].lower() == "tpu"
    return False


def _maybe_configure_tpu_host_threads() -> None:
    if os.environ.get("EXPDIS_TPU_HOST_THREADS_CONFIGURED") == "1":
        return
    wants_tpu = os.environ.get("PJRT_DEVICE", "").upper() == "TPU" or _argv_requests_tpu(sys.argv[1:])
    if not wants_tpu:
        return

    host_threads = _parse_positive_int_env("EXPDIS_TPU_HOST_TORCH_THREADS", 4)
    interop_threads = _parse_positive_int_env("EXPDIS_TPU_HOST_TORCH_INTEROP_THREADS", 1)
    default_threads = str(host_threads)
    default_interop = str(interop_threads)
    os.environ.setdefault("OMP_NUM_THREADS", default_threads)
    os.environ.setdefault("MKL_NUM_THREADS", default_threads)
    os.environ.setdefault("NUMEXPR_NUM_THREADS", default_threads)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", default_threads)
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", default_threads)
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", default_interop)
    print(
        "[host-threads] "
        f"target={host_threads} "
        f"interop_target={interop_threads} "
        f"OMP={os.environ.get('OMP_NUM_THREADS')} "
        f"MKL={os.environ.get('MKL_NUM_THREADS')}"
    )

    os.environ["EXPDIS_TPU_HOST_THREADS_CONFIGURED"] = "1"


def _maybe_enable_xla_spmd() -> None:
    """Enable SPMD mode for XLA FSDP v2 if EXPDIS_FSDP is set on TPU.

    Skipped when the decoupled Explorer path is active (EXPDIS_EXPLORER_DECOUPLED_GRPO),
    because that path uses plain (non-FSDP) models.  SPMD mode causes XLA C++ crashes
    when forward/backward runs on a model whose parameters lack SPMD sharding annotations.
    """
    fsdp_mode = (os.environ.get("EXPDIS_FSDP") or "").strip().lower()
    is_tpu = os.environ.get("PJRT_DEVICE", "").upper() == "TPU" or _argv_requests_tpu(sys.argv[1:])
    if not is_tpu or fsdp_mode not in ("1", "true", "full_shard"):
        return
    # The decoupled Explorer GRPO path manages its own models without FSDP.
    # Enabling SPMD globally causes the plain-model forward/backward to segfault.
    decoupled = (os.environ.get("EXPDIS_EXPLORER_DECOUPLED_GRPO") or "").strip().lower()
    if decoupled not in ("0", "false", "no"):
        print("[xla-spmd] Skipping SPMD — decoupled Explorer path uses plain models (no FSDP).")
        return
    try:
        import torch_xla.runtime as xr  # type: ignore
        xr.use_spmd()
        print("[xla-spmd] SPMD mode enabled for FSDP v2.")
    except Exception as exc:
        print(f"[WARN] Failed to enable XLA SPMD: {exc}")


def main():
    _load_env_file()
    _ensure_hf_runtime_env(root_dir=os.path.dirname(os.path.abspath(__file__)))
    _maybe_configure_tpu_host_threads()
    _maybe_enable_xla_spmd()
    _maybe_initialize_xla_persistent_cache()
    args = parse_args()

    from expdis_torch.pipeline import run_pipeline
    configs = build_configs_from_args(args)

    summary = []
    for cfg in configs:
        print("\n" + "=" * 80)
        print(
            f"Starting run: lambda_novelty={cfg.lambda_novelty}, "
            f"loss={cfg.grpo_loss_type}, "
            f"mode={cfg.decoupling_mode}, "
            f"device_pref={cfg.device_preference}"
        )
        res = run_pipeline(cfg)
        summary.append(
            {
                "lambda": cfg.lambda_novelty,
                "loss": cfg.grpo_loss_type,
                "mode": cfg.decoupling_mode,
                "metric": cfg.novelty_metric,
                "clip": cfg.novelty_clip,
                "benchmark_dataset": cfg.benchmark_dataset_name,
                **res,
            }
        )

    if len(summary) > 1:
        print("\nSweep summary (config -> base/explorer/actual accuracy):")
        for row in summary:
            suffix = ""
            if row.get("benchmark_dataset"):
                suffix = (
                    f", benchmark={row['benchmark_dataset']}: "
                    f"base_bench={row['benchmark_base_acc']:.3f}, "
                    f"explorer_bench={row['benchmark_explorer_acc']:.3f}, "
                    f"actual_bench={row['benchmark_actual_acc']:.3f}"
                )
            print(
                f"lambda={row['lambda']}, loss={row['loss']}, "
                f"mode={row['mode']}, metric={row['metric']}, clip={row['clip']}: "
                f"base={row['base_acc']:.3f}, "
                f"explorer={row['explorer_acc']:.3f}, "
                f"actual={row['actual_acc']:.3f}{suffix}"
            )


if __name__ == "__main__":
    main()
