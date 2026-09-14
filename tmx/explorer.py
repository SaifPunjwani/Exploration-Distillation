import csv
import json
import gc
import math
import os
import random
import re
import time
import inspect
import threading
import types
import warnings
from typing import Optional, Sequence

import torch
from tqdm.auto import tqdm

from .config import schedule_value
from .data import select_accepted_trajectories
from .prompting import (
    _resolved_thinking_mode,
    generation_answer_prefix,
    get_answer_token_constraints,
    render_generation_prompt,
    resolved_max_completion_len,
)
from .reward import (
    AnchorPenaltyScorer,
    GRPOReward,
    analyze_completion_text,
    build_actual_training_text,
    canonicalize_completion_for_training,
    completion_quality_metrics,
    extract_presented_answer,
    normalize_generated_completion,
)
from .rnd import RNDModule
from .reward import NoveltyScorer
from .embeddings import auto_layer_selection
from .devices import maybe_import_xla
from .artifacts import gcs_artifacts_enabled, hf_artifact_prefix, upload_path_to_hf
from .utils import (
    _single_worker_xla_collective_passthrough,
    causal_lm_from_pretrained,
    get_device,
    hf_from_pretrained_kwargs,
    patch_accelerate_gather_object_for_xla,
    patch_accelerate_gradient_state_for_xla,
    patch_model_generate_for_xla,
    patch_trl_grpo_compute_loss_for_xla,
    patch_trl_grpo_sampler_for_xla_spmd,
    patch_torch_xla_parallel_loader_for_single_worker,
    patch_transformers_accelerator_num_processes_for_xla_spmd,
    patch_transformers_nested_xla_mesh_reduce_for_single_worker,
    patch_transformers_trainer_logging_for_single_worker,
    patch_tokenizer_for_xla_fixed_padding,
    xla_safe_generate,
)

try:
    import wandb  # type: ignore
except Exception:
    wandb = None


# ── Model architecture cache ──
# Avoids repeated from_pretrained() calls which parse config and init
# architecture from disk on every GRPO step. Cache keyed by (model_name, is_xla).
_MODEL_SHELL_CACHE: dict = {}


def _get_or_create_model_shell(model_name: str, is_xla: bool):
    """Return a freshly-initialized model (no trained weights) from cache or disk."""
    key = (model_name, is_xla)
    if key not in _MODEL_SHELL_CACHE:
        shell = causal_lm_from_pretrained(model_name, is_xla=is_xla)
        # Store architecture config so we can re-create cheaply
        _MODEL_SHELL_CACHE[key] = shell.__class__, shell.config, is_xla
        print(f"[model-cache] cached architecture shell for {model_name} is_xla={is_xla}")
        return shell
    cls, config, _ = _MODEL_SHELL_CACHE[key]
    # Re-create from cached class + config (skips HF hub resolution + safetensors parsing)
    import copy
    cfg_copy = copy.deepcopy(config)
    if is_xla:
        use_flash = os.environ.get("TMX_XLA_FLASH_ATTENTION", "0") not in ("0", "false", "no", "")
        cfg_copy.attn_implementation = "tpu_flash" if use_flash else "eager"
    shell = cls(cfg_copy)
    _use_flash = is_xla and os.environ.get("TMX_XLA_FLASH_ATTENTION", "0") not in ("0", "false", "no", "")
    if _use_flash:
        from tmx.utils import _patch_model_bf16_for_flash_attention
        _patch_model_bf16_for_flash_attention(shell)
    return shell


def _round_up_to_bucket(length: int, bucket_size: int = 256) -> int:
    """Round up to the nearest multiple of bucket_size for XLA shape stability."""
    return ((length + bucket_size - 1) // bucket_size) * bucket_size


def _enable_model_grad_checkpointing(
    model,
    *,
    preserve_rng_state: bool = False,
    verbose: bool = True,
    label: str = "",
) -> bool:
    if not hasattr(model, "gradient_checkpointing_enable"):
        return False
    grad_ckpt_kwargs = {
        "gradient_checkpointing_kwargs": {
            "use_reentrant": False,
            "preserve_rng_state": bool(preserve_rng_state),
        }
    }
    try:
        model.gradient_checkpointing_enable(**grad_ckpt_kwargs)
    except TypeError:
        model.gradient_checkpointing_enable()
        if verbose:
            prefix = f"{label} " if label else ""
            print(
                f"[patch] {prefix}gradient checkpointing enabled without kwargs fallback; "
                "input-grad hook may still be required."
            )
    if hasattr(model, "enable_input_require_grads"):
        try:
            model.enable_input_require_grads()
        except Exception as e:
            if verbose:
                prefix = f"{label} " if label else ""
                print(f"[WARN] Could not enable input require grads for {prefix}model: {e}")
    return True


def _latest_hf_checkpoint(output_dir: str | None, *, max_step_exclusive: Optional[int] = None) -> Optional[str]:
    if not output_dir or not os.path.isdir(output_dir):
        return None
    candidates = []
    for name in os.listdir(output_dir):
        if not name.startswith("checkpoint-"):
            continue
        path = os.path.join(output_dir, name)
        if not os.path.isdir(path):
            continue
        try:
            step = int(name.split("-", 1)[1])
        except Exception:
            continue
        if max_step_exclusive is not None and step >= int(max_step_exclusive):
            continue
        trainer_state = os.path.join(path, "trainer_state.json")
        model_index = os.path.join(path, "model.safetensors.index.json")
        model_safetensors = os.path.join(path, "model.safetensors")
        model_bin = os.path.join(path, "pytorch_model.bin")
        optimizer_state = os.path.join(path, "optimizer.pt")
        scheduler_state = os.path.join(path, "scheduler.pt")
        if not os.path.exists(trainer_state):
            continue
        if not (os.path.exists(model_index) or os.path.exists(model_safetensors) or os.path.exists(model_bin)):
            continue
        if not os.path.exists(optimizer_state) or not os.path.exists(scheduler_state):
            continue
        try:
            checkpoint_files = os.listdir(path)
        except Exception:
            continue
        if any(entry.endswith(".gstmp") for entry in checkpoint_files):
            continue
        candidates.append((step, path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def _normalize_token_id(token_id):
    if token_id is None:
        return None
    if isinstance(token_id, (list, tuple)):
        return int(token_id[0]) if token_id else None
    if torch.is_tensor(token_id):
        if token_id.numel() == 0:
            return None
        return int(token_id.flatten()[0].item())
    return int(token_id)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() not in {"", "0", "false", "no", "off"}


def _cfg_run_root(cfg) -> str:
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


def _trajectory_csv_path(traj_path: str) -> str:
    root, _ = os.path.splitext(str(traj_path))
    return f"{root}.csv"


def _write_trajectory_csv(path: str, trajectories) -> None:
    fieldnames = [
        "explorer_step",
        "source_problem_id",
        "source_row_id",
        "temperature",
        "topic",
        "prompt",
        "ground_truth",
        "completion",
        "completion_canonical",
        "presented_answer",
        "is_correct",
        "r_correct",
        "r_novel",
        "reward",
        "reward_bucket",
        "prompt_token_length",
        "completion_token_length",
        "terminated",
        "clipped",
        "format_quality",
        "raw_format_quality",
    ]
    csv_dir = os.path.dirname(path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in trajectories:
            writer.writerow(row)


def _should_build_novelty_scorer(cfg) -> bool:
    if _env_flag("TMX_FORCE_NOVELTY_SCORER", default=False):
        return True
    try:
        lambda_novelty = float(getattr(cfg, "lambda_novelty", 0.0) or 0.0)
    except Exception:
        lambda_novelty = 0.0
    try:
        incorrect_scale = float(getattr(cfg, "incorrect_novelty_scale", 0.0) or 0.0)
    except Exception:
        incorrect_scale = 0.0
    return lambda_novelty > 0.0 or incorrect_scale > 0.0


def _use_cpu_novelty(is_xla: bool) -> bool:
    # TPU hidden-state extraction for multilayer novelty is the unstable path in
    # the decoupled rollout loop; keep CPU as the default unless explicitly
    # overridden for an isolated experiment.
    return not _env_flag("TMX_NOVELTY_ON_TPU", default=False) if is_xla else False


def _refresh_novelty_scorer_from_state(novelty_scorer, state_dict):
    if novelty_scorer is None:
        return None
    novelty_model = getattr(novelty_scorer, "embed_model", None)
    if novelty_model is None:
        return None
    novelty_model.load_state_dict(state_dict)
    novelty_model.eval()
    return novelty_scorer


def _xla_runtime_rank_world() -> tuple[int, int]:
    rank = 0
    world = 1
    xm_mod, xr_mod = maybe_import_xla()
    if xr_mod is not None:
        try:
            rank = int(xr_mod.global_ordinal())
        except Exception:
            rank = 0
        try:
            world = int(xr_mod.world_size())
        except Exception:
            world = 1
        if world <= 0:
            world = 1
        return rank, world
    if xm_mod is not None:
        try:
            rank = int(xm_mod.get_ordinal())
        except Exception:
            rank = 0
        try:
            world = int(xm_mod.xrt_world_size())
        except Exception:
            world = 1
    return rank, max(1, world)


def _xla_is_master_process(is_xla: bool) -> bool:
    if not is_xla:
        return True
    rank, _ = _xla_runtime_rank_world()
    return rank == 0


def _xla_rendezvous(name: str, is_xla: bool) -> None:
    if not is_xla:
        return
    _, world = _xla_runtime_rank_world()
    if world <= 1:
        return
    xm_mod, _ = maybe_import_xla()
    if xm_mod is None:
        return
    xm_mod.rendezvous(name)


def _pad_decoupled_batch_to_global_widths(
    prompt_ids_cpu: torch.Tensor,
    prompt_mask_cpu: torch.Tensor,
    completion_ids_cpu: torch.Tensor,
    completion_mask_cpu: torch.Tensor,
    old_per_token_logps_cpu: torch.Tensor,
    ref_per_token_logps_cpu: Optional[torch.Tensor],
    *,
    pad_token_id: int,
    tag: str,
    is_xla: bool,
    target_prompt_width: int = 0,
    target_completion_width: int = 0,
):
    if not is_xla:
        return (
            prompt_ids_cpu,
            prompt_mask_cpu,
            completion_ids_cpu,
            completion_mask_cpu,
            old_per_token_logps_cpu,
            ref_per_token_logps_cpu,
        )
    xm_mod, _ = maybe_import_xla()
    if xm_mod is None:
        return (
            prompt_ids_cpu,
            prompt_mask_cpu,
            completion_ids_cpu,
            completion_mask_cpu,
            old_per_token_logps_cpu,
            ref_per_token_logps_cpu,
        )

    bucket = int(os.environ.get("TMX_XLA_PAD_BUCKET", "256") or 256)
    local_prompt_width = int(prompt_ids_cpu.size(1))
    local_completion_width = int(completion_ids_cpu.size(1))
    global_prompt_width = int(xm_mod.mesh_reduce(f"{tag}_prompt_width", local_prompt_width, max))
    global_completion_width = int(xm_mod.mesh_reduce(f"{tag}_completion_width", local_completion_width, max))
    if target_prompt_width > 0:
        global_prompt_width = max(global_prompt_width, int(target_prompt_width))
    if target_completion_width > 0:
        global_completion_width = max(global_completion_width, int(target_completion_width))
    if bucket > 0:
        global_prompt_width = _round_up_to_bucket(global_prompt_width, bucket)
        global_completion_width = _round_up_to_bucket(global_completion_width, bucket)

    if local_prompt_width < global_prompt_width:
        pad_cols = global_prompt_width - local_prompt_width
        prompt_ids_cpu = torch.nn.functional.pad(prompt_ids_cpu, (pad_cols, 0), value=int(pad_token_id))
        prompt_mask_cpu = torch.nn.functional.pad(prompt_mask_cpu, (pad_cols, 0), value=0)
    if local_completion_width < global_completion_width:
        pad_cols = global_completion_width - local_completion_width
        completion_ids_cpu = torch.nn.functional.pad(completion_ids_cpu, (0, pad_cols), value=int(pad_token_id))
        completion_mask_cpu = torch.nn.functional.pad(completion_mask_cpu, (0, pad_cols), value=0)
    # Logps may already be wider than completion_ids (vLLM stores padded logprobs
    # while completion_ids/mask are stored unpadded per-row). Pad or truncate
    # independently to match global_completion_width.
    logp_w = int(old_per_token_logps_cpu.size(1))
    if logp_w < global_completion_width:
        old_per_token_logps_cpu = torch.nn.functional.pad(
            old_per_token_logps_cpu, (0, global_completion_width - logp_w), value=0.0
        )
    elif logp_w > global_completion_width:
        old_per_token_logps_cpu = old_per_token_logps_cpu[:, :global_completion_width]
    if ref_per_token_logps_cpu is not None:
        ref_w = int(ref_per_token_logps_cpu.size(1))
        if ref_w < global_completion_width:
            ref_per_token_logps_cpu = torch.nn.functional.pad(
                ref_per_token_logps_cpu, (0, global_completion_width - ref_w), value=0.0
            )
        elif ref_w > global_completion_width:
            ref_per_token_logps_cpu = ref_per_token_logps_cpu[:, :global_completion_width]

    return (
        prompt_ids_cpu,
        prompt_mask_cpu,
        completion_ids_cpu,
        completion_mask_cpu,
        old_per_token_logps_cpu,
        ref_per_token_logps_cpu,
    )


def _read_explorer_rollout_buffer(path: str):
    groups = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            groups.append(json.loads(line))
    return groups


def _split_rollout_groups_for_rank(groups, *, rank: int, world_size: int):
    if world_size <= 1:
        return groups
    if len(groups) % world_size != 0:
        raise ValueError(
            "Decoupled FSDP training requires rollout groups divisible by world size. "
            f"got groups={len(groups)} world_size={world_size}"
        )
    return groups[rank::world_size]


def _find_transformer_wrap_classes(model, wrap_class_names: Sequence[str]):
    found = set()
    wanted = {str(name).strip() for name in wrap_class_names if str(name).strip()}
    for module in model.modules():
        cls = module.__class__
        if cls.__name__ in wanted:
            found.add(cls)
    return found


def _wrap_model_with_xla_fsdp(explorer_model, *, use_grad_ckpt: bool):
    import functools

    from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as XFSDP
    from torch_xla.distributed.fsdp import checkpoint_module
    from torch_xla.distributed.fsdp.wrap import transformer_auto_wrap_policy

    wrap_cls_override = (os.environ.get("TMX_FSDP_WRAP_CLS") or "").strip()
    if wrap_cls_override:
        wrap_class_names = [item.strip() for item in wrap_cls_override.split(",") if item.strip()]
    else:
        wrap_class_names = ["Qwen3DecoderLayer"]
    wrap_classes = _find_transformer_wrap_classes(explorer_model, wrap_class_names)
    auto_wrap_policy = None
    if wrap_classes:
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=wrap_classes,
        )
    auto_wrapper_callable = None
    if use_grad_ckpt:
        if getattr(explorer_model, "config", None) is not None:
            setattr(explorer_model.config, "use_cache", False)

        def auto_wrapper_callable(module, *args, **kwargs):
            return XFSDP(checkpoint_module(module), *args, **kwargs)

    use_bf16 = os.environ.get("TMX_DISABLE_BF16", "0") != "1"
    compute_dtype = torch.bfloat16 if use_bf16 else None
    wrapped = XFSDP(
        explorer_model,
        auto_wrap_policy=auto_wrap_policy,
        auto_wrapper_callable=auto_wrapper_callable,
        compute_dtype=compute_dtype,
    )
    if _xla_is_master_process(True):
        wrap_msg = ",".join(sorted(cls.__name__ for cls in wrap_classes)) if wrap_classes else "<none>"
        print(
            "[decoupled-train] XLA FSDP enabled "
            f"wrap={wrap_msg} grad_ckpt={bool(use_grad_ckpt)} compute_dtype={compute_dtype}",
            flush=True,
        )
    return wrapped


def _unwrap_module_chain(model):
    current = model
    visited = set()
    while hasattr(current, "module") and id(current) not in visited:
        visited.add(id(current))
        next_model = getattr(current, "module", None)
        if next_model is None or next_model is current:
            break
        current = next_model
    return current


def _save_xla_fsdp_full_checkpoint(model, tokenizer, output_dir: str) -> None:
    from torch_xla.distributed.fsdp import consolidate_sharded_model_checkpoints

    xm_mod, _ = maybe_import_xla()
    if xm_mod is None:
        raise RuntimeError("XLA FSDP checkpoint save requested without torch_xla runtime.")
    rank, world = _xla_runtime_rank_world()
    debug_progress = _env_flag("TMX_DECOUPLED_TRAIN_DEBUG_PROGRESS", default=False)

    def _save_log(message: str) -> None:
        if debug_progress:
            print(f"[decoupled-save][rank {rank}] {message}", flush=True)

    os.makedirs(output_dir, exist_ok=True)
    _save_log("save_mark_step_start")
    xm_mod.mark_step()
    _save_log("save_mark_step_done")
    _save_log("save_rendezvous_start")
    xm_mod.rendezvous("tmx_decoupled_fsdp_save_start")
    _save_log("save_rendezvous_done")
    shard_suffix = f"rank{rank}-of-{world}-pytorch_model.bin"
    shard_path = os.path.join(output_dir, shard_suffix)
    ckpt = {
        "model": model.state_dict(),
        "shard_metadata": model.get_shard_metadata(),
    }
    _save_log(f"save_shard_start path={shard_suffix}")
    xm_mod.save(ckpt, shard_path, master_only=False)
    _save_log("save_shard_done")
    _save_log("save_shards_written_rendezvous_start")
    xm_mod.rendezvous("tmx_decoupled_fsdp_shards_written")
    _save_log("save_shards_written_rendezvous_done")
    if xm_mod.is_master_ordinal(local=False):
        _save_log("consolidate_start")
        full_state_dict, _ = consolidate_sharded_model_checkpoints(
            ckpt_prefix=os.path.join(output_dir, ""),
            ckpt_suffix="rank*-of-*-pytorch_model.bin",
            save_model=False,
        )
        _save_log("consolidate_done")
        model_to_save = _unwrap_module_chain(model)
        _save_log("save_pretrained_start")
        model_to_save.save_pretrained(
            output_dir,
            state_dict=full_state_dict,
            safe_serialization=True,
        )
        if tokenizer is not None:
            tokenizer.save_pretrained(output_dir)
        _save_log("save_pretrained_done")
        for name in os.listdir(output_dir):
            if name.startswith("rank") and name.endswith("-pytorch_model.bin"):
                try:
                    os.remove(os.path.join(output_dir, name))
                except OSError:
                    pass
        _save_log("save_shard_cleanup_done")
    _save_log("save_full_saved_rendezvous_start")
    xm_mod.rendezvous("tmx_decoupled_fsdp_full_saved")
    _save_log("save_full_saved_rendezvous_done")


def _save_plain_model_checkpoint(model, tokenizer, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    model_to_save = _unwrap_module_chain(model).to(torch.device("cpu"))
    state_dict = {
        key: value.detach().cpu()
        for key, value in model_to_save.state_dict().items()
    }
    model_to_save.save_pretrained(
        output_dir,
        state_dict=state_dict,
        safe_serialization=True,
    )
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)


def _sequence_generation_stats(sequence_ids, prompt_width: int, eos_id, pad_token_id, max_new_tokens: int):
    generated = sequence_ids[prompt_width:].detach().cpu().tolist()
    trimmed = list(generated)
    while trimmed and pad_token_id is not None and trimmed[-1] == pad_token_id:
        trimmed.pop()

    terminated = False
    effective_len = len(trimmed)
    if eos_id is not None and eos_id in trimmed:
        eos_idx = trimmed.index(eos_id)
        effective_len = eos_idx
        terminated = True

    completion_ids = trimmed[:effective_len]
    return {
        "completion_token_length": int(len(completion_ids)),
        "terminated": bool(terminated),
        "clipped": bool((not terminated) and len(completion_ids) >= int(max_new_tokens)),
    }


def _maybe_xla_mark_step(device_type: str) -> None:
    if device_type != "xla":
        return
    try:
        import torch_xla.core.xla_model as _xm
        _xm.mark_step()
    except Exception:
        pass


def _get_per_token_logps_chunked(
    trainer,
    model,
    prompt_completion_ids,
    attention_mask,
    logits_to_keep: int,
    batch_size: int,
    device_type: str,
):
    chunk_spec = (os.environ.get("TMX_GRPO_LOGPROB_CHUNK_SIZE") or "").strip()
    if not chunk_spec:
        return trainer._get_per_token_logps(
            model,
            prompt_completion_ids,
            attention_mask,
            logits_to_keep,
            batch_size,
        )

    try:
        chunk_size = max(1, int(chunk_spec))
    except Exception:
        chunk_size = 0
    if chunk_size <= 0 or int(prompt_completion_ids.size(0)) <= chunk_size:
        return trainer._get_per_token_logps(
            model,
            prompt_completion_ids,
            attention_mask,
            logits_to_keep,
            batch_size,
        )

    outputs = []
    total_rows = int(prompt_completion_ids.size(0))
    if device_type == "xla":
        print(
            "[patch] chunking per-token logprob scoring: "
            f"rows={total_rows} chunk_size={chunk_size} logits_to_keep={int(logits_to_keep)}"
        )
    for start in range(0, total_rows, chunk_size):
        stop = min(total_rows, start + chunk_size)
        outputs.append(
            trainer._get_per_token_logps(
                model,
                prompt_completion_ids[start:stop],
                attention_mask[start:stop],
                logits_to_keep,
                min(batch_size, stop - start),
            )
        )
        _maybe_xla_mark_step(device_type)
    return torch.cat(outputs, dim=0)


def _rebuild_explorer_model_for_generation(
    explorer_model,
    tokenizer,
    cfg,
    device,
    is_xla: bool,
    safe_eos_id,
    answer_constraints,
    reuse_novelty_scorer=None,
):
    if explorer_model is None:
        return None, None
    if not is_xla:
        return explorer_model.to(device), None

    print("[patch] Rebuilding Explorer model from CPU weights before trajectory generation.")
    cpu = torch.device("cpu")
    try:
        explorer_model = explorer_model.to(cpu)
        raw_state_dict = {
            key: value.detach().cpu().clone()
            for key, value in explorer_model.state_dict().items()
        }
        state_dict = {}
        renamed = 0
        for key, value in raw_state_dict.items():
            new_key = key
            while new_key.startswith("_orig_module."):
                new_key = new_key[len("_orig_module.") :]
            while "._orig_module." in new_key:
                new_key = new_key.replace("._orig_module.", ".")
            while new_key.startswith("_fsdp_wrapped_module."):
                new_key = new_key[len("_fsdp_wrapped_module.") :]
            while "._fsdp_wrapped_module." in new_key:
                new_key = new_key.replace("._fsdp_wrapped_module.", ".")
            while new_key.startswith("module."):
                new_key = new_key[len("module.") :]
            if new_key != key:
                renamed += 1
            state_dict[new_key] = value
        if renamed:
            print(f"[patch] Normalized {renamed} wrapped Explorer state_dict keys for rebuild.")
        del raw_state_dict
    except Exception as exc:
        print(f"[WARN] Could not snapshot Explorer model for XLA rebuild: {exc}")
        return explorer_model.to(device), None

    del explorer_model
    gc.collect()

    rebuilt_model = _get_or_create_model_shell(cfg.model_name, is_xla=is_xla)
    rebuilt_model.load_state_dict(state_dict)
    # Pallas flash attention requires the ENTIRE model in bf16.
    if is_xla and os.environ.get("TMX_XLA_FLASH_ATTENTION", "0") not in ("0", "false", "no", ""):
        rebuilt_model = rebuilt_model.to(torch.bfloat16)
    gc.collect()

    if tokenizer.pad_token_id is not None:
        rebuilt_model.config.pad_token_id = tokenizer.pad_token_id
        if getattr(rebuilt_model, "generation_config", None) is not None:
            rebuilt_model.generation_config.pad_token_id = tokenizer.pad_token_id
    if safe_eos_id is not None:
        rebuilt_model.config.eos_token_id = safe_eos_id
        if getattr(rebuilt_model, "generation_config", None) is not None:
            rebuilt_model.generation_config.eos_token_id = safe_eos_id

    if answer_constraints is not None:
        rebuilt_model._tmx_answer_constraints = answer_constraints
    for cfg_obj in (getattr(rebuilt_model, "config", None), getattr(rebuilt_model, "generation_config", None)):
        if cfg_obj is not None:
            setattr(cfg_obj, "suppress_tokens", None)
            setattr(cfg_obj, "begin_suppress_tokens", None)
            setattr(cfg_obj, "bad_words_ids", None)
            setattr(cfg_obj, "use_cache", False)
    if os.environ.get("TMX_XLA_SAFE_GENERATE", "1") != "0":
        patch_model_generate_for_xla(rebuilt_model)

    # When vLLM handles generation, keep rollout model on CPU to avoid TPU driver
    # contention during CPU-based novelty scoring.  The rollout model is only used
    # for weight snapshots / anchor scoring, not for generation.
    _vllm_active = bool((os.environ.get("TMX_VLLM_SERVER_URLS") or "").strip())
    if _vllm_active:
        rebuilt_model = rebuilt_model.to(torch.device("cpu"))
        print("[patch] Rebuilding Explorer model on CPU (vLLM active, no TPU needed for generation).", flush=True)
    else:
        rebuilt_model = rebuilt_model.to(device)
    rebuilt_novelty_scorer = None
    if _should_build_novelty_scorer(cfg):
        if _use_cpu_novelty(is_xla):
            cpu_device = torch.device("cpu")
            rebuilt_novelty_scorer = _refresh_novelty_scorer_from_state(reuse_novelty_scorer, state_dict)
            if rebuilt_novelty_scorer is None:
                novelty_model = _get_or_create_model_shell(cfg.model_name, is_xla=False)
                novelty_model.load_state_dict(state_dict)
                novelty_model.eval()
                rebuilt_novelty_scorer, _ = build_novelty_scorer(
                    novelty_model, tokenizer, cfg, cpu_device, is_xla=False
                )
                print("[novelty] Scorer built on CPU to avoid XLA hidden-state extraction hang.")
            else:
                print("[novelty] Reused persistent CPU scorer.")
        else:
            # When vLLM is active, rebuilt_model lives on CPU — need a
            # separate model on the XLA device for TPU novelty scoring.
            if _vllm_active and is_xla:
                # Reuse existing scorer if available (just update weights).
                _existing = getattr(reuse_novelty_scorer, "embed_model", None) if reuse_novelty_scorer else None
                if _existing is not None:
                    _existing.load_state_dict(
                        {k: v.to(device) for k, v in rebuilt_model.state_dict().items()}
                    )
                    reuse_novelty_scorer.embed_model = _existing.to(device).eval()
                    reuse_novelty_scorer.device = device
                    rebuilt_novelty_scorer = reuse_novelty_scorer
                    print(f"[novelty] Reloaded existing scorer model on {device}.")
                else:
                    novelty_model = _get_or_create_model_shell(cfg.model_name, is_xla=True)
                    novelty_model.load_state_dict(
                        {k: v.to(device) for k, v in rebuilt_model.state_dict().items()}
                    )
                    novelty_model = novelty_model.to(device).eval()
                    rebuilt_novelty_scorer, _ = build_novelty_scorer(
                        novelty_model, tokenizer, cfg, device, is_xla=is_xla
                    )
                    print(f"[novelty] Scorer built on {device} (explicit override).")
            else:
                rebuilt_novelty_scorer, _ = build_novelty_scorer(
                    rebuilt_model, tokenizer, cfg, device, is_xla=is_xla
                )
                print(f"[novelty] Scorer built on {device} (explicit override).")
    del state_dict
    return rebuilt_model, rebuilt_novelty_scorer


def _snapshot_rebuildable_model_state(explorer_model):
    if explorer_model is None:
        return None
    cpu = torch.device("cpu")
    explorer_model = explorer_model.to(cpu)
    raw_state_dict = {
        key: value.detach().cpu().clone()
        for key, value in explorer_model.state_dict().items()
    }
    state_dict = {}
    for key, value in raw_state_dict.items():
        new_key = key
        while new_key.startswith("_orig_module."):
            new_key = new_key[len("_orig_module.") :]
        while "._orig_module." in new_key:
            new_key = new_key.replace("._orig_module.", ".")
        while new_key.startswith("_fsdp_wrapped_module."):
            new_key = new_key[len("_fsdp_wrapped_module.") :]
        while "._fsdp_wrapped_module." in new_key:
            new_key = new_key.replace("._fsdp_wrapped_module.", ".")
        while new_key.startswith("module."):
            new_key = new_key[len("module.") :]
        state_dict[new_key] = value
    return state_dict


def _build_explorer_model_from_state(
    state_dict,
    tokenizer,
    cfg,
    device,
    is_xla: bool,
    safe_eos_id,
    *,
    answer_constraints=None,
    patch_generate: bool = False,
):
    rebuilt_model = _get_or_create_model_shell(cfg.model_name, is_xla=is_xla)
    rebuilt_model.load_state_dict(state_dict)
    # Pallas flash attention requires the ENTIRE model in bf16.
    if is_xla and os.environ.get("TMX_XLA_FLASH_ATTENTION", "0") not in ("0", "false", "no", ""):
        rebuilt_model = rebuilt_model.to(torch.bfloat16)
    if tokenizer.pad_token_id is not None:
        rebuilt_model.config.pad_token_id = tokenizer.pad_token_id
        if getattr(rebuilt_model, "generation_config", None) is not None:
            rebuilt_model.generation_config.pad_token_id = tokenizer.pad_token_id
    if safe_eos_id is not None:
        rebuilt_model.config.eos_token_id = safe_eos_id
        if getattr(rebuilt_model, "generation_config", None) is not None:
            rebuilt_model.generation_config.eos_token_id = safe_eos_id
    if answer_constraints is not None:
        rebuilt_model._tmx_answer_constraints = answer_constraints
    for cfg_obj in (getattr(rebuilt_model, "config", None), getattr(rebuilt_model, "generation_config", None)):
        if cfg_obj is not None:
            setattr(cfg_obj, "suppress_tokens", None)
            setattr(cfg_obj, "begin_suppress_tokens", None)
            setattr(cfg_obj, "bad_words_ids", None)
            setattr(cfg_obj, "use_cache", False)
            setattr(cfg_obj, "return_dict", True)
    if patch_generate and os.environ.get("TMX_XLA_SAFE_GENERATE", "1") != "0":
        patch_model_generate_for_xla(rebuilt_model)
    return rebuilt_model.to(device)


def _pad_tensor_rows(rows, *, pad_value, dtype, left_pad: bool = False, bucket_size: int = 0):
    max_len = max((len(row) for row in rows), default=0)
    if bucket_size > 0:
        max_len = _round_up_to_bucket(max_len, bucket_size)
    out = torch.full((len(rows), max_len), pad_value, dtype=dtype)
    if dtype.is_floating_point:
        mask = torch.zeros((len(rows), max_len), dtype=torch.float32)
    else:
        mask = torch.zeros((len(rows), max_len), dtype=torch.long)
    for idx, row in enumerate(rows):
        if not row:
            continue
        values = torch.tensor(row, dtype=dtype)
        if left_pad:
            start = max_len - len(row)
            out[idx, start:] = values
            mask[idx, start:] = 1
        else:
            out[idx, : len(row)] = values
            mask[idx, : len(row)] = 1
    return out, mask


def _selective_log_softmax(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=-1)
    target_safe = target_ids.clone()
    target_safe[target_safe < 0] = 0
    return log_probs.gather(-1, target_safe.unsqueeze(-1)).squeeze(-1)


def _compute_completion_logps(
    model,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    completion_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    *,
    temperature: float,
    batch_size: int,
) -> torch.Tensor:
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    logits_to_keep = completion_ids.size(1)
    all_logps = []
    batch_size = max(1, batch_size)
    total_rows = int(input_ids.size(0))
    total_chunks = max(1, math.ceil(total_rows / batch_size))
    progress_enabled = (os.environ.get("TMX_LOGPROB_PROGRESS") or "1").strip().lower() not in ("0", "false", "no")
    last_chunk_idx = 0
    for chunk_idx, start in enumerate(range(0, total_rows, batch_size), start=1):
        stop = min(total_rows, start + batch_size)
        if progress_enabled:
            print(
                f"[decoupled-explorer] logprob chunk {chunk_idx}/{total_chunks} rows={start}:{stop}",
                flush=True,
            )
        outputs = model(
            input_ids=input_ids[start:stop],
            attention_mask=attention_mask[start:stop],
            logits_to_keep=int(logits_to_keep) + 1,
            use_cache=False,
            return_dict=False,
        )
        logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits
        logits = logits[:, :-1, :]
        logits = logits[:, -logits_to_keep:]
        logits = logits / float(temperature)
        batch_targets = completion_ids[start:stop]
        batch_logps = _selective_log_softmax(logits, batch_targets)
        all_logps.append(batch_logps * completion_mask[start:stop].to(batch_logps.dtype))
        _maybe_xla_mark_step(get_device(model).type)
        last_chunk_idx = chunk_idx
    if progress_enabled:
        print(
            f"[decoupled-explorer] logprob chunks complete {last_chunk_idx}/{total_chunks}",
            flush=True,
    )
    return torch.cat(all_logps, dim=0)


def _compute_chunked_train_loss(
    model,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    completion_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    old_per_token_logps: torch.Tensor,
    ref_per_token_logps: Optional[torch.Tensor],
    advantages: torch.Tensor,
    *,
    train_temperature: float,
    beta_val: float,
    loss_type: str,
    total_rows: int,
    full_completion_tokens: float,
    seq_width: int,
    token_chunk_size: int,
    debug_log=None,
):
    chunk_size = max(1, int(token_chunk_size))
    row_count = int(prompt_ids.size(0))
    completion_width = int(completion_ids.size(1))
    metric_dtype = torch.float32

    def _dbg(message: str) -> None:
        if debug_log is not None:
            debug_log(message)

    cfg_objs = (
        getattr(model, "config", None),
        getattr(model, "generation_config", None),
    )
    prev_use_cache = [getattr(cfg_obj, "use_cache", None) if cfg_obj is not None else None for cfg_obj in cfg_objs]
    for cfg_obj in cfg_objs:
        if cfg_obj is not None:
            setattr(cfg_obj, "use_cache", True)

    row_loss_sum = None
    row_token_count = completion_mask.to(metric_dtype).sum(dim=1).clamp(min=1.0)
    kl_sum = torch.zeros((), device=prompt_ids.device, dtype=metric_dtype)
    clip_sum = torch.zeros((), device=prompt_ids.device, dtype=metric_dtype)
    token_sum = completion_mask.to(metric_dtype).sum().detach()

    try:
        _dbg(
            f"prefix_forward_start rows={row_count} prompt_width={int(prompt_ids.size(1))} "
            f"completion_width={completion_width} token_chunk_size={chunk_size}"
        )
        prefix_outputs = model(
            input_ids=prompt_ids,
            attention_mask=prompt_mask,
            logits_to_keep=1,
            use_cache=True,
            return_dict=True,
        )
        next_token_logits = prefix_outputs.logits[:, -1:, :]
        past_key_values = prefix_outputs.past_key_values
        running_attention_mask = prompt_mask
        _dbg("prefix_forward_done")

        total_chunks = max(1, math.ceil(completion_width / chunk_size))
        for chunk_index, token_start in enumerate(range(0, completion_width, chunk_size), start=1):
            token_stop = min(completion_width, token_start + chunk_size)
            chunk_ids = completion_ids[:, token_start:token_stop]
            chunk_mask = completion_mask[:, token_start:token_stop]
            chunk_old_logps = old_per_token_logps[:, token_start:token_stop]
            chunk_ref_logps = (
                ref_per_token_logps[:, token_start:token_stop]
                if ref_per_token_logps is not None
                else None
            )
            running_attention_mask = torch.cat([running_attention_mask, chunk_mask], dim=1)
            _dbg(
                f"token_chunk_forward_start chunk={chunk_index}/{total_chunks} "
                f"tokens={token_start}:{token_stop} attn_width={int(running_attention_mask.size(1))}"
            )
            chunk_outputs = model(
                input_ids=chunk_ids,
                attention_mask=running_attention_mask,
                past_key_values=past_key_values,
                logits_to_keep=int(chunk_ids.size(1)),
                use_cache=True,
                return_dict=True,
            )
            chunk_logits = chunk_outputs.logits
            target_logits = torch.cat([next_token_logits, chunk_logits[:, :-1, :]], dim=1)
            target_logits = target_logits / float(train_temperature)
            chunk_logps = _selective_log_softmax(target_logits, chunk_ids) * chunk_mask.to(target_logits.dtype)
            if chunk_ref_logps is None:
                chunk_ref_logps = chunk_old_logps
            chunk_kl = (
                torch.exp(chunk_ref_logps - chunk_logps) - (chunk_ref_logps - chunk_logps) - 1.0
            )
            coef_1 = torch.exp(chunk_logps - chunk_old_logps)
            coef_2 = torch.clamp(coef_1, 1 - 0.2, 1 + 0.2)
            per_token_loss1 = coef_1 * advantages.unsqueeze(1)
            per_token_loss2 = coef_2 * advantages.unsqueeze(1)
            chunk_loss_terms = -torch.min(per_token_loss1, per_token_loss2)
            if beta_val != 0.0:
                chunk_loss_terms = chunk_loss_terms + beta_val * chunk_kl

            masked_chunk_loss = chunk_loss_terms * chunk_mask
            chunk_row_loss_sum = masked_chunk_loss.sum(dim=1)
            row_loss_sum = chunk_row_loss_sum if row_loss_sum is None else (row_loss_sum + chunk_row_loss_sum)
            kl_sum = kl_sum + (chunk_kl * chunk_mask).sum().detach().to(metric_dtype)
            clip_sum = clip_sum + (((coef_1 != coef_2).float() * chunk_mask).sum()).detach().to(metric_dtype)

            next_token_logits = chunk_logits[:, -1:, :]
            past_key_values = chunk_outputs.past_key_values
            _dbg(
                f"token_chunk_forward_done chunk={chunk_index}/{total_chunks} "
                f"tokens={token_start}:{token_stop}"
            )

        if row_loss_sum is None:
            row_loss_sum = torch.zeros((row_count,), device=prompt_ids.device, dtype=metric_dtype)

        if loss_type == "grpo":
            row_loss_values = row_loss_sum / row_token_count.to(row_loss_sum.dtype)
            scaled_loss = row_loss_values.mean() * (float(row_count) / float(total_rows))
        elif loss_type in ("bnpo", "dr_grpo", "dapo"):
            scaled_loss = row_loss_sum.sum() / max(1.0, full_completion_tokens)
        else:
            raise ValueError(f"Unknown grpo_loss_type: {loss_type!r}")
        return (
            scaled_loss,
            kl_sum,
            clip_sum,
            token_sum,
        )
    finally:
        for cfg_obj, prev in zip(cfg_objs, prev_use_cache):
            if cfg_obj is not None and prev is not None:
                setattr(cfg_obj, "use_cache", prev)


def _collate_explorer_rollout_groups(batch, pad_token_id: int):
    # Use bucket_size=256 to keep tensor shapes stable across steps,
    # preventing XLA recompilation on every different context length.
    _bucket = int(os.environ.get("TMX_XLA_PAD_BUCKET", "256") or 256)
    flat_rows = [row for group in batch for row in group["rows"]]
    prompt_ids, prompt_mask = _pad_tensor_rows(
        [row["prompt_ids"] for row in flat_rows],
        pad_value=pad_token_id,
        dtype=torch.long,
        left_pad=True,
        bucket_size=_bucket,
    )
    completion_ids, _ = _pad_tensor_rows(
        [row["completion_ids"] for row in flat_rows],
        pad_value=pad_token_id,
        dtype=torch.long,
        left_pad=False,
        bucket_size=_bucket,
    )
    completion_mask, _ = _pad_tensor_rows(
        [row["completion_mask"] for row in flat_rows],
        pad_value=0,
        dtype=torch.long,
        left_pad=False,
        bucket_size=_bucket,
    )
    old_per_token_logps, _ = _pad_tensor_rows(
        [row["old_per_token_logps"] for row in flat_rows],
        pad_value=0.0,
        dtype=torch.float32,
        left_pad=False,
        bucket_size=_bucket,
    )
    if any(row.get("ref_per_token_logps") is not None for row in flat_rows):
        ref_rows = [
            row["ref_per_token_logps"] if row.get("ref_per_token_logps") is not None else []
            for row in flat_rows
        ]
        ref_per_token_logps, _ = _pad_tensor_rows(
            ref_rows,
            pad_value=0.0,
            dtype=torch.float32,
            left_pad=False,
            bucket_size=_bucket,
        )
    else:
        ref_per_token_logps = None
    return {
        "prompt_ids": prompt_ids,
        "prompt_mask": prompt_mask,
        "completion_ids": completion_ids,
        "completion_mask": completion_mask,
        "advantages": torch.tensor([float(row["advantage"]) for row in flat_rows], dtype=torch.float32),
        "old_per_token_logps": old_per_token_logps,
        "ref_per_token_logps": ref_per_token_logps,
        "reward": torch.tensor([float(row["reward"]) for row in flat_rows], dtype=torch.float32),
        "r_correct": torch.tensor([float(row["r_correct"]) for row in flat_rows], dtype=torch.float32),
        "r_novel": torch.tensor([float(row["r_novel"]) for row in flat_rows], dtype=torch.float32),
        "is_correct": torch.tensor([1.0 if row["is_correct"] else 0.0 for row in flat_rows], dtype=torch.float32),
        "group_count": len(batch),
        "row_count": len(flat_rows),
        "mean_completion_len": float(sum(int(row["completion_token_length"]) for row in flat_rows) / max(1, len(flat_rows))),
        "terminated_fraction": float(sum(1.0 if row["terminated"] else 0.0 for row in flat_rows) / max(1, len(flat_rows))),
        "clipped_fraction": float(sum(1.0 if row["clipped"] else 0.0 for row in flat_rows) / max(1, len(flat_rows))),
        "terminated_rows": [row for row in flat_rows if row.get("terminated", False)],
    }


def _write_explorer_rollout_buffer(path: str, groups) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for group in groups:
            handle.write(json.dumps(group) + "\n")


def _save_explorer_step_checkpoint(model, tokenizer, output_dir: str, step: int, novelty_scorer=None) -> None:
    if not output_dir:
        return
    os.makedirs(output_dir, exist_ok=True)
    ckpt_dir = os.path.join(output_dir, f"checkpoint-{int(step)}")
    os.makedirs(ckpt_dir, exist_ok=True)
    state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    try:
        model.save_pretrained(ckpt_dir, state_dict=state_dict, safe_serialization=True)
    except TypeError:
        model.save_pretrained(ckpt_dir, state_dict=state_dict)
    tokenizer.save_pretrained(ckpt_dir)
    # Save RND novelty scorer state so it can be restored on resume.
    if novelty_scorer is not None and hasattr(novelty_scorer, "rnd_map"):
        rnd_state = {}
        for key, rnd_mod in novelty_scorer.rnd_map.items():
            rnd_state[key] = {
                "target": {k: v.detach().cpu() for k, v in rnd_mod.target.state_dict().items()},
                "predictor": {k: v.detach().cpu() for k, v in rnd_mod.predictor.state_dict().items()},
                "optimizer": rnd_mod.optimizer.state_dict(),
            }
        rnd_path = os.path.join(ckpt_dir, "rnd_state.pt")
        torch.save(rnd_state, rnd_path)
        print(f"[checkpoint] saved RND state ({len(rnd_state)} modules) to {rnd_path}", flush=True)
    # Upload to Hugging Face by default, then enforce local limit.
    run_name = os.environ.get("RUN_NAME") or os.path.basename(os.path.dirname(os.path.abspath(output_dir)))
    hf_root = f"{hf_artifact_prefix(run_name)}/explorer"
    ckpt_uri = upload_path_to_hf(ckpt_dir, f"{hf_root}/checkpoint-{int(step)}", is_dir=True)
    if ckpt_uri:
        print(f"[checkpoint] uploaded to {ckpt_uri}", flush=True)

    import glob as _artifact_glob
    rollout_files = sorted(_artifact_glob.glob(os.path.join(output_dir, "rollout_buffer_step_*.jsonl")))
    for rollout_path in rollout_files:
        upload_path_to_hf(
            rollout_path,
            f"{hf_root}/rollouts/{os.path.basename(rollout_path)}",
            is_dir=False,
        )
    if rollout_files:
        print(f"[checkpoint] scheduled {len(rollout_files)} rollout-buffer HF uploads", flush=True)

    traj_candidates = [
        os.path.join(os.path.dirname(output_dir), "explorer_trajectories.jsonl"),
        os.path.join(output_dir, "explorer_trajectories.jsonl"),
    ]
    for traj_path in traj_candidates:
        if os.path.isfile(traj_path):
            upload_path_to_hf(
                traj_path,
                f"{hf_root}/trajectories/{os.path.basename(traj_path)}",
                is_dir=False,
            )
            print("[checkpoint] scheduled trajectories HF upload", flush=True)
            break

    # Legacy GCS writes are opt-in only.
    gcs_root = os.environ.get("TMX_GCS_CHECKPOINT_ROOT", "")
    if gcs_root and gcs_artifacts_enabled():
        import subprocess
        import glob as _gcs_glob
        gcs_dst = f"{gcs_root.rstrip('/')}/checkpoint-{int(step)}/"
        try:
            subprocess.run(["gsutil", "-m", "cp", "-r", ckpt_dir, gcs_dst],
                           check=True, timeout=300, capture_output=True)
            print(f"[checkpoint] uploaded legacy GCS copy to {gcs_dst}", flush=True)
        except Exception as e:
            print(f"[checkpoint] legacy GCS upload failed: {e}", flush=True)
        # Upload rollout buffers that haven't been uploaded yet.
        gcs_rollouts_dst = f"{gcs_root.rstrip('/')}/rollouts/"
        rollout_files = sorted(_gcs_glob.glob(os.path.join(output_dir, "rollout_buffer_step_*.jsonl")))
        if rollout_files:
            try:
                subprocess.run(
                    ["gsutil", "-m", "cp"] + rollout_files + [gcs_rollouts_dst],
                    check=True, timeout=600, capture_output=True,
                )
                print(f"[checkpoint] uploaded {len(rollout_files)} rollout buffers to {gcs_rollouts_dst}", flush=True)
            except Exception as e:
                print(f"[checkpoint] rollout buffer GCS upload failed: {e}", flush=True)
        # Upload trajectories file if it exists.
        traj_candidates = [
            os.path.join(os.path.dirname(output_dir), "explorer_trajectories.jsonl"),
            os.path.join(output_dir, "explorer_trajectories.jsonl"),
        ]
        for traj_path in traj_candidates:
            if os.path.isfile(traj_path):
                gcs_traj_dst = f"{gcs_root.rstrip('/')}/trajectories/"
                try:
                    subprocess.run(
                        ["gsutil", "cp", traj_path, gcs_traj_dst],
                        check=True, timeout=300, capture_output=True,
                    )
                    print(f"[checkpoint] uploaded trajectories to {gcs_traj_dst}", flush=True)
                except Exception as e:
                    print(f"[checkpoint] trajectories GCS upload failed: {e}", flush=True)
                break
    # Enforce local checkpoint limit to prevent disk-full.
    save_limit = int(os.environ.get("TMX_GRPO_SAVE_TOTAL_LIMIT", "3") or 3)
    if save_limit > 0:
        import glob as _glob
        existing = sorted(_glob.glob(os.path.join(output_dir, "checkpoint-*")),
                          key=lambda p: int(os.path.basename(p).split("-")[-1]))
        while len(existing) > save_limit:
            old = existing.pop(0)
            import shutil
            shutil.rmtree(old, ignore_errors=True)
            print(f"[checkpoint] removed old checkpoint {old}", flush=True)


def _generate_explorer_rollout_groups(
    explorer_model,
    tokenizer,
    train_ds,
    cfg,
    novelty_scorer,
    anchor_model,
    device,
    is_xla: bool,
    *,
    train_temperature: float,
    explorer_step: int,
):
    from tmx.vllm_generate import generate_completions_vllm

    reward_anchor = None
    if anchor_model is not None and cfg.anchor_metric != "none" and cfg.anchor_weight > 0:
        reward_anchor = AnchorPenaltyScorer(
            current_model=explorer_model,
            reference_model=anchor_model.to(device),
            tokenizer=tokenizer,
            metric=cfg.anchor_metric,
            device=device,
            max_length=cfg.max_total_len,
            trainable_only=cfg.anchor_trainable_only,
        )
    reward_fn = GRPOReward(
        novelty_scorer,
        cfg.lambda_novelty,
        reward_correct=cfg.reward_correct,
        reward_incorrect=cfg.reward_incorrect,
        reward_shape=cfg.reward_shape,
        incorrect_novelty_scale=cfg.incorrect_novelty_scale,
        piecewise_bounds=cfg.reward_piecewise_bounds,
        novelty_zscore=cfg.novelty_zscore,
        correctness_zscore=cfg.correctness_zscore,
        anchor_penalty_scorer=reward_anchor,
        anchor_weight=cfg.anchor_weight,
        format_penalty_weight=cfg.format_penalty_weight,
        gate_novelty_by_quality=cfg.gate_novelty_by_quality,
        answer_prefix=generation_answer_prefix(cfg, phase="explore"),
        novelty_text_contract=cfg.novelty_text_contract,
        novelty_phase="explorer_train",
    )
    vllm_urls = [u.strip() for u in ((os.environ.get("TMX_VLLM_SERVER_URLS") or "").split(",")) if u.strip()]
    use_ondevice_gen = not vllm_urls
    if use_ondevice_gen:
        print("[decoupled-explorer] No vLLM URLs set, using on-device generation.", flush=True)
        ondevice_gen_batch = max(1, int(os.environ.get("TMX_DECOUPLED_ONDEVICE_GEN_BATCH", "2") or 2))
    vllm_model = os.environ.get("TMX_VLLM_MODEL_NAME", "")
    vllm_api_key = os.environ.get("TMX_VLLM_API_KEY", "")
    vllm_concurrency = int(os.environ.get("TMX_VLLM_CONCURRENCY", "64"))
    vllm_top_p = float(os.environ.get("TMX_EXPLORER_VLLM_TOP_P", "0.95") or 0.95)
    vllm_top_k = int(os.environ.get("TMX_EXPLORER_VLLM_TOP_K", "20") or 20)
    num_generations = max(2, int(cfg.grpo_num_generations))
    rollout_prompt_batch = max(1, int(os.environ.get("TMX_EXPLORER_DECOUPLED_ROLLOUT_PROMPTS", getattr(cfg, "trajectory_batch_size", 4)) or 4))
    max_completion_len = int(resolved_max_completion_len(cfg, phase="explore"))
    safe_eos_id = _normalize_token_id(tokenizer.eos_token_id)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    logprob_chunk = max(1, int(os.environ.get("TMX_EXPLORER_DECOUPLED_LOGPROB_BATCH", "2") or 2))
    groups = []

    for start in range(0, len(train_ds), rollout_prompt_batch):
        stop = min(len(train_ds), start + rollout_prompt_batch)
        batch = train_ds.select(range(start, stop))
        prompts = list(batch["prompt"])
        model_prompts = [render_generation_prompt(tokenizer, prompt, cfg, phase="explore") for prompt in prompts]
        ground_truths = list(batch["ground_truth"])
        topics = list(batch["topic"]) if "topic" in batch.column_names else ["unknown"] * len(prompts)
        source_row_ids = (
            [int(v) for v in batch["__row_id__"]]
            if "__row_id__" in batch.column_names
            else list(range(start, stop))
        )
        prompt_token_lists = tokenizer(
            model_prompts,
            add_special_tokens=True,
            padding=False,
            truncation=True,
            max_length=cfg.max_prompt_len,
        )["input_ids"]
        thinking_mode = _resolved_thinking_mode(cfg, phase="explore")
        vllm_enable_thinking = True if thinking_mode == "think" else (False if thinking_mode == "no_think" else None)
        if use_ondevice_gen:
            # On-device generation: generate per-prompt to keep memory bounded.
            all_completions = []
            eos_id = _normalize_token_id(tokenizer.eos_token_id) or tokenizer.eos_token_id
            for pi in range(len(model_prompts)):
                prompt_comps = []
                enc = tokenizer(model_prompts[pi], return_tensors="pt", padding=False,
                                truncation=True, max_length=cfg.max_prompt_len)
                input_ids = enc["input_ids"].to(device)
                attn_mask = enc["attention_mask"].to(device)
                prompt_len = input_ids.shape[1]
                for gi in range(0, num_generations, ondevice_gen_batch):
                    n_this = min(ondevice_gen_batch, num_generations - gi)
                    batch_ids = input_ids.expand(n_this, -1)
                    batch_mask = attn_mask.expand(n_this, -1)
                    with torch.no_grad():
                        out = explorer_model.generate(
                            input_ids=batch_ids, attention_mask=batch_mask,
                            max_new_tokens=max_completion_len, do_sample=True,
                            temperature=float(train_temperature), top_p=0.95,
                            eos_token_id=eos_id, pad_token_id=pad_id,
                        )
                    if is_xla:
                        import torch_xla.core.xla_model as xm
                        xm.mark_step()
                    for j in range(n_this):
                        gen_ids = out[j][prompt_len:]
                        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                        finished = gen_ids[-1].item() == eos_id if len(gen_ids) > 0 else False
                        prompt_comps.append({
                            "text": text,
                            "finish_reason": "stop" if finished else "length",
                            "token_logprobs": None,
                        })
                all_completions.append(prompt_comps)
            print(
                f"[decoupled-explorer] rollout batch {start}:{stop} generated "
                f"{sum(len(g) for g in all_completions)} completions (on-device)",
                flush=True,
            )
        else:
            all_completions = generate_completions_vllm(
                prompts=model_prompts,
                server_urls=vllm_urls,
                model_name=vllm_model,
                n_per_prompt=num_generations,
                temperature=float(train_temperature),
                top_p=vllm_top_p,
                top_k=vllm_top_k,
                max_tokens=max_completion_len,
                concurrency=vllm_concurrency,
                api_key=vllm_api_key,
                enable_thinking=vllm_enable_thinking,
            )
            print(
                f"[decoupled-explorer] rollout batch {start}:{stop} generated "
                f"{sum(len(group) for group in all_completions)} completions",
                flush=True,
            )
        assert len(all_completions) == len(prompts), "prompt count mismatch."
        flat_completions = [d for group in all_completions for d in group]
        completion_texts_raw = [d["text"] for d in flat_completions]
        completion_finish_reasons = [d["finish_reason"] for d in flat_completions]
        vllm_token_logprobs_raw = [d.get("token_logprobs") for d in flat_completions]
        completion_token_lists = [
            tokenizer.encode(text, add_special_tokens=False)
            for text in completion_texts_raw
        ]
        completion_token_lists = [
            ids + [safe_eos_id] if finish_reason == "stop" and safe_eos_id is not None else ids
            for ids, finish_reason in zip(completion_token_lists, completion_finish_reasons)
        ]
        completion_token_lists = [ids[:max_completion_len] for ids in completion_token_lists]
        repeated_prompt_ids = [ids for ids in prompt_token_lists for _ in range(num_generations)]
        prompt_ids_padded, prompt_mask = _pad_tensor_rows(
            repeated_prompt_ids,
            pad_value=pad_id,
            dtype=torch.long,
            left_pad=True,
        )
        completion_ids_padded, _ = _pad_tensor_rows(
            completion_token_lists,
            pad_value=pad_id,
            dtype=torch.long,
            left_pad=False,
        )
        completion_masks = [
            [1] * len(ids)
            for ids in completion_token_lists
        ]
        completion_mask, _ = _pad_tensor_rows(
            completion_masks,
            pad_value=0,
            dtype=torch.long,
            left_pad=False,
        )
        prompt_ids_device = prompt_ids_padded.to(device)
        prompt_mask_device = prompt_mask.to(device)
        completion_ids_device = completion_ids_padded.to(device)
        completion_mask_device = completion_mask.to(device)
        # Use vLLM logprobs when available, skipping expensive XLA forward passes.
        # TMX_SKIP_VLLM_LOGPROBS=1 forces XLA re-scoring so old_logprobs match the current training
        # policy. vLLM only refreshes weights at end of Explorer GRPO, so stale vLLM logprobs
        # cause KL blow-up (ratio = exp(trained - initial)) and near-100% PPO clipping.
        skip_vllm_lp = _env_flag("TMX_SKIP_VLLM_LOGPROBS", default=False)
        have_vllm_logprobs = (not skip_vllm_lp) and all(lps is not None for lps in vllm_token_logprobs_raw)
        if have_vllm_logprobs:
            comp_width = completion_ids_padded.size(1)
            logp_rows = []
            for idx, (raw_lps, tok_ids) in enumerate(zip(vllm_token_logprobs_raw, completion_token_lists)):
                # Align vLLM logprobs with re-tokenized completion.
                # vLLM returns logprobs for generated tokens; we may have appended EOS.
                aligned = [float(lp) if lp is not None else 0.0 for lp in raw_lps]
                target_len = len(tok_ids)
                if len(aligned) < target_len:
                    aligned.extend([0.0] * (target_len - len(aligned)))
                aligned = aligned[:target_len]
                # Pad to padded width.
                aligned.extend([0.0] * (comp_width - len(aligned)))
                logp_rows.append(aligned)
            old_per_token_logps = torch.tensor(logp_rows, dtype=torch.float32)
            print(
                f"[decoupled-explorer] old logprobs from vLLM (skipped XLA scoring) "
                f"rows={old_per_token_logps.size(0)} width={comp_width}",
                flush=True,
            )
        else:
            print(
                f"[decoupled-explorer] scoring old logprobs rows={completion_ids_device.size(0)} "
                f"prompt_width={prompt_ids_device.size(1)} completion_width={completion_ids_device.size(1)} "
                f"chunk={logprob_chunk}",
                flush=True,
            )
            with torch.no_grad():
                old_per_token_logps = _compute_completion_logps(
                    explorer_model,
                    prompt_ids_device,
                    prompt_mask_device,
                    completion_ids_device,
                    completion_mask_device,
                    temperature=float(train_temperature),
                    batch_size=logprob_chunk,
                )
            print("[decoupled-explorer] old logprobs ready", flush=True)
        if (os.environ.get("TMX_GRPO_BETA") or "").strip():
            ref_per_token_logps = old_per_token_logps.detach().cpu().clone()
        else:
            ref_per_token_logps = None
        completions_text = tokenizer.batch_decode(completion_ids_device, skip_special_tokens=True)
        completion_lengths = completion_mask_device.sum(dim=1) - (completion_ids_device == safe_eos_id).any(dim=1).int()
        completion_stats = [
            {
                "completion_token_length": int(completion_lengths[idx].item()),
                "terminated": bool((completion_ids_device[idx] == safe_eos_id).any().item()) if safe_eos_id is not None else False,
                "clipped": bool(
                    safe_eos_id is None or not (completion_ids_device[idx] == safe_eos_id).any().item()
                ) and int(completion_lengths[idx].item()) >= max_completion_len,
            }
            for idx in range(completion_ids_device.size(0))
        ]
        repeated_ground_truths = [gt for gt in ground_truths for _ in range(num_generations)]
        repeated_model_prompts = [prompt for prompt in model_prompts for _ in range(num_generations)]
        rewards, details = reward_fn.compute_batch(
            completions_text,
            repeated_ground_truths,
            prompts=repeated_model_prompts,
            completion_stats=completion_stats,
            train=True,
            return_details=True,
        )
        print("[decoupled-explorer] reward batch ready", flush=True)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        mean_grouped_rewards = rewards_tensor.view(-1, num_generations).mean(dim=1)
        std_grouped_rewards = rewards_tensor.view(-1, num_generations).std(dim=1)
        use_batch_norm = bool((std_grouped_rewards < 1e-8).all().item()) and len(rewards_tensor) > 1
        if use_batch_norm:
            batch_mean = rewards_tensor.mean()
            batch_std = rewards_tensor.std()
            if batch_std > 1e-8:
                advantages = (rewards_tensor - batch_mean) / (batch_std + 1e-4)
            else:
                advantages = torch.zeros_like(rewards_tensor)
        else:
            advantages = rewards_tensor - mean_grouped_rewards.repeat_interleave(num_generations, dim=0)
            advantages = advantages / (std_grouped_rewards.repeat_interleave(num_generations, dim=0) + 1e-4)
            # Zero out advantages for groups with no reward variance to avoid phantom gradients
            zero_std_mask = (std_grouped_rewards < 1e-8).repeat_interleave(num_generations, dim=0)
            advantages[zero_std_mask] = 0.0

        for group_idx, (prompt, model_prompt, gt, topic, source_row_id, prompt_ids_unpadded) in enumerate(
            zip(prompts, model_prompts, ground_truths, topics, source_row_ids, prompt_token_lists)
        ):
            start_idx = group_idx * num_generations
            stop_idx = start_idx + num_generations
            rows = []
            for row_idx in range(start_idx, stop_idx):
                rows.append(
                    {
                        "prompt": prompt,
                        "model_prompt": model_prompt,
                        "ground_truth": gt,
                        "topic": topic,
                        "source_problem_id": int(source_row_id),
                        "source_row_id": int(source_row_id),
                        "sample_idx_within_prompt": int(row_idx - start_idx),
                        "explorer_step": int(explorer_step),
                        "prompt_ids": [int(x) for x in prompt_ids_unpadded],
                        "completion_ids": [int(x) for x in completion_token_lists[row_idx]],
                        "completion_mask": [int(x) for x in completion_masks[row_idx]],
                        "old_per_token_logps": [float(x) for x in old_per_token_logps[row_idx].detach().cpu().tolist()],
                        "ref_per_token_logps": None if ref_per_token_logps is None else [float(x) for x in ref_per_token_logps[row_idx].tolist()],
                        "advantage": float(advantages[row_idx].item()),
                        "reward": float(rewards[row_idx]),
                        "r_correct": float(details[row_idx].get("r_correct_used", details[row_idx].get("r_correct_raw", 0.0))),
                        "r_novel": float(details[row_idx].get("r_novel_used", details[row_idx].get("r_novel_raw", 0.0))),
                        "is_correct": bool(details[row_idx].get("is_correct", False)),
                        "completion_token_length": int(completion_stats[row_idx]["completion_token_length"]),
                        "terminated": bool(completion_stats[row_idx]["terminated"]),
                        "clipped": bool(completion_stats[row_idx]["clipped"]),
                        "reward_bucket": str(details[row_idx].get("reward_bucket", "unknown")),
                        "completion_text": completions_text[row_idx] if row_idx < len(completions_text) else "",
                    }
                )
            groups.append(
                {
                    "group_index": int(len(groups)),
                    "prompt": prompt,
                    "model_prompt": model_prompt,
                    "ground_truth": gt,
                    "topic": topic,
                    "source_problem_id": int(source_row_id),
                    "explorer_step": int(explorer_step),
                    "temperature": float(train_temperature),
                    "rows": rows,
                }
            )
        print(f"[decoupled-explorer] built {len(groups)} rollout groups so far", flush=True)
        _maybe_xla_mark_step(device.type)
    return groups, dict(reward_fn.last_batch_summary or {})


def _train_explorer_from_rollout_groups(
    explorer_model,
    tokenizer,
    groups,
    cfg,
    device,
    is_xla: bool,
    *,
    train_temperature: float,
    grpo_metric_prefix: str,
    train_metric_prefix: Optional[str],
    use_wandb: bool,
    wandb_run,
    output_dir: str,
    start_step: int,
    max_steps: int,
    novelty_scorer=None,
    reward_batch_summary: Optional[dict] = None,
):
    use_xla_fsdp = (
        is_xla
        and _env_flag("TMX_EXPLORER_DECOUPLED_TRAIN_FSDP", default=_env_flag("TMX_FSDP", default=False))
        and _xla_runtime_rank_world()[1] > 1
        and hasattr(explorer_model, "get_shard_metadata")
    )
    rank, world_size = _xla_runtime_rank_world() if use_xla_fsdp else (0, 1)
    groups_for_rank = (
        _split_rollout_groups_for_rank(groups, rank=rank, world_size=world_size)
        if use_xla_fsdp
        else groups
    )
    groups_for_rank = list(groups_for_rank)
    batch_groups = max(1, int(os.environ.get("TMX_EXPLORER_DECOUPLED_BATCH_GROUPS", str(getattr(cfg, "grpo_batch_size", 1) or 1)) or 1))
    grad_accum = max(
        1,
        int(
            os.environ.get(
                "TMX_EXPLORER_DECOUPLED_GRAD_ACCUM",
                str(getattr(cfg, "grpo_grad_accum", 1) or 1),
            )
            or 1
        ),
    )
    row_microbatch = max(
        1,
        int(
            os.environ.get(
                "TMX_EXPLORER_DECOUPLED_ROW_MICROBATCH",
                os.environ.get("TMX_EXPLORER_DECOUPLED_TRAIN_ROWS_PER_FWD", "0"),
            )
            or 0
        ),
    )
    beta_env = (os.environ.get("TMX_GRPO_BETA") or "").strip()
    beta_val = float(beta_env) if beta_env else 0.0
    debug_progress = _env_flag("TMX_DECOUPLED_TRAIN_DEBUG_PROGRESS", default=False)
    row_mark_step = is_xla and _env_flag("TMX_DECOUPLED_ROW_MARK_STEP", default=True)
    accum_mark_step = is_xla and _env_flag("TMX_DECOUPLED_ACCUM_MARK_STEP", default=use_xla_fsdp)
    accum_mark_reset_scope = _env_flag("TMX_DECOUPLED_ACCUM_MARK_RESET_SCOPE", default=False)
    pre_optimizer_mark_step = is_xla and _env_flag("TMX_DECOUPLED_PRE_OPTIMIZER_MARK_STEP", default=use_xla_fsdp)
    token_chunk_size = max(
        0,
        int(
            os.environ.get(
                "TMX_DECOUPLED_TRAIN_TOKEN_CHUNK",
                "256" if is_xla and not use_xla_fsdp else "0",
            )
            or 0
        ),
    )
    use_token_chunk_train = token_chunk_size > 0

    from .devices import optimizer_step, mark_step
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    _use_flash = is_xla and os.environ.get("TMX_XLA_FLASH_ATTENTION", "0") not in ("0", "false", "no", "")
    fixed_shapes = (use_xla_fsdp or _use_flash) and _env_flag("TMX_DECOUPLED_TRAIN_FIXED_SHAPES", default=True)
    fixed_prompt_width = int(
        os.environ.get(
            "TMX_DECOUPLED_TRAIN_FIXED_PROMPT_WIDTH",
            str(getattr(cfg, "max_prompt_len", 0) or 0) if fixed_shapes else "0",
        )
        or 0
    )
    fixed_completion_width = int(
        os.environ.get(
            "TMX_DECOUPLED_TRAIN_FIXED_COMPLETION_WIDTH",
            str(
                getattr(
                    cfg,
                    "exploration_max_completion_len",
                    getattr(cfg, "max_completion_len", 0),
                )
                or getattr(cfg, "max_completion_len", 0)
                or 0
            )
            if fixed_shapes
            else "0",
        )
        or 0
    )

    def _debug_log(message: str) -> None:
        if debug_progress:
            print(f"[decoupled-train][rank {rank}] {message}", flush=True)

    if len(groups_for_rank) > 1:
        random.Random(cfg.seed + int(start_step) + int(rank)).shuffle(groups_for_rank)
    group_batches = [
        groups_for_rank[start : start + batch_groups]
        for start in range(0, len(groups_for_rank), batch_groups)
    ]
    precollate_batches = _env_flag("TMX_DECOUPLED_PRECOLLATE_BATCHES", default=use_xla_fsdp)
    total_batch_count = len(group_batches)

    def _collate_group_batch(group_batch, batch_index: int):
        _debug_log(
            f"host_collate_start batch_index={batch_index}/{total_batch_count} "
            f"groups={len(group_batch)}"
        )
        collated = _collate_explorer_rollout_groups(group_batch, tokenizer.pad_token_id or 0)
        _debug_log(
            f"host_collate_done batch_index={batch_index}/{total_batch_count} "
            f"group_count={int(collated['group_count'])} row_count={int(collated['row_count'])} "
            f"mean_completion_len={float(collated['mean_completion_len']):.1f}"
        )
        return collated

    prebuilt_batches = None
    if precollate_batches:
        prebuilt_batches = [
            _collate_group_batch(group_batch, batch_index)
            for batch_index, group_batch in enumerate(group_batches, start=1)
        ]
    # Enable gradient checkpointing to fit 8k contexts on a single chip.
    # Controllable via TMX_DECOUPLED_GRAD_CKPT (default=1). Disable for short contexts
    # to avoid the ~15 min XLA compilation overhead of the checkpointed graph.
    _enable_grad_ckpt = (os.environ.get("TMX_DECOUPLED_GRAD_CKPT", "1") or "1").strip().lower() not in ("0", "false", "no")
    # Allow stacking: chunking + gradient checkpointing both reduce activation memory.
    # Override via TMX_DECOUPLED_ALLOW_CHUNK_AND_CKPT=1 to keep grad_ckpt on when chunking.
    if use_token_chunk_train and _enable_grad_ckpt and not _env_flag("TMX_DECOUPLED_ALLOW_CHUNK_AND_CKPT", default=True):
        _enable_grad_ckpt = False
        if _xla_is_master_process(is_xla):
            print(
                f"[decoupled-train] disabling gradient checkpointing for token-chunk train "
                f"(TMX_DECOUPLED_TRAIN_TOKEN_CHUNK={token_chunk_size})",
                flush=True,
            )
    if _enable_grad_ckpt and not use_xla_fsdp:
        try:
            _enable_model_grad_checkpointing(
                explorer_model,
                preserve_rng_state=False,
                verbose=_xla_is_master_process(is_xla),
                label="decoupled-train",
            )
            if _xla_is_master_process(is_xla):
                print("[decoupled-train] gradient checkpointing enabled for activation memory savings")
        except Exception as e:
            if _xla_is_master_process(is_xla):
                print(f"[decoupled-train] WARN: could not enable gradient checkpointing: {e}")
    elif not _enable_grad_ckpt and _xla_is_master_process(is_xla):
        print("[decoupled-train] gradient checkpointing DISABLED (TMX_DECOUPLED_GRAD_CKPT=0)")
    optim_override = (
        os.environ.get("TMX_OPTIM", "").strip()
        or os.environ.get("TMX_OPTIMIZER_OVERRIDE", "").strip()
        or "adamw_torch"
    )
    optim_name = optim_override.lower()
    if optim_name in {"adamw", "adamw_torch", "adamw_hf"}:
        optimizer = torch.optim.AdamW(explorer_model.parameters(), lr=cfg.grpo_lr)
    elif optim_name in {"adafactor", "transformers_adafactor"}:
        try:
            from transformers.optimization import Adafactor
        except Exception:
            from transformers import Adafactor  # type: ignore

        optimizer = Adafactor(
            explorer_model.parameters(),
            lr=cfg.grpo_lr,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
        )
    else:
        raise ValueError(
            f"Unsupported TMX_OPTIM={optim_override!r} for decoupled Explorer training. "
            "Supported values: adamw_torch, adafactor."
        )
    if _xla_is_master_process(is_xla):
        print(
            f"[decoupled-train] optimizer={optim_name} lr={cfg.grpo_lr} "
            f"token_chunk={token_chunk_size if use_token_chunk_train else 0}",
            flush=True,
        )
    explorer_model.train()
    global_step = int(start_step)
    save_steps = max(0, int(os.environ.get("TMX_GRPO_SAVE_STEPS", "0") or 0))
    log_history = []
    optimizer.zero_grad()
    micro_step = 0
    for batch_index, group_batch in enumerate(group_batches, start=1):
        batch = (
            prebuilt_batches[batch_index - 1]
            if prebuilt_batches is not None
            else _collate_group_batch(group_batch, batch_index)
        )
        _debug_log(
            f"batch_fetch_done batch_index={batch_index}/{total_batch_count} "
            f"group_count={int(batch['group_count'])} row_count={int(batch['row_count'])}"
        )
        prompt_ids_cpu = batch["prompt_ids"]
        prompt_mask_cpu = batch["prompt_mask"]
        completion_ids_cpu = batch["completion_ids"]
        completion_mask_cpu = batch["completion_mask"]
        old_per_token_logps_cpu = batch["old_per_token_logps"]
        ref_per_token_logps_cpu = batch["ref_per_token_logps"]
        advantages_cpu = batch["advantages"]
        rewards_cpu = batch["reward"]
        r_correct_cpu = batch["r_correct"]
        r_novel_cpu = batch["r_novel"]
        is_correct_cpu = batch["is_correct"]
        if use_xla_fsdp or (fixed_shapes and fixed_completion_width > 0):
            shape_tag = f"tmx_decoupled_train_shape_s{global_step:06d}_m{micro_step:06d}"
            local_prompt_width = int(prompt_ids_cpu.size(1))
            local_completion_width = int(completion_ids_cpu.size(1))
            (
                prompt_ids_cpu,
                prompt_mask_cpu,
                completion_ids_cpu,
                completion_mask_cpu,
                old_per_token_logps_cpu,
                ref_per_token_logps_cpu,
            ) = _pad_decoupled_batch_to_global_widths(
                prompt_ids_cpu,
                prompt_mask_cpu,
                completion_ids_cpu,
                completion_mask_cpu,
                old_per_token_logps_cpu,
                ref_per_token_logps_cpu,
                pad_token_id=pad_id,
                tag=shape_tag,
                is_xla=is_xla,
                target_prompt_width=fixed_prompt_width,
                target_completion_width=fixed_completion_width,
            )
            _debug_log(
                "batch_global_pad "
                f"local_prompt_width={local_prompt_width} local_completion_width={local_completion_width} "
                f"global_prompt_width={int(prompt_ids_cpu.size(1))} global_completion_width={int(completion_ids_cpu.size(1))}"
            )

        total_rows = int(prompt_ids_cpu.size(0))
        full_completion_mask = completion_mask_cpu.to(torch.float32)
        full_completion_tokens = float(full_completion_mask.sum().item())
        seq_width = int(completion_ids_cpu.size(1))
        row_microbatch_size = min(row_microbatch, total_rows)

        scalar_metric_tensors = bool(is_xla)
        if scalar_metric_tensors:
            batch_loss_value = torch.zeros((), device=device, dtype=torch.float32)
            batch_kl_sum = torch.zeros((), device=device, dtype=torch.float32)
            batch_clip_sum = torch.zeros((), device=device, dtype=torch.float32)
            batch_token_sum = torch.zeros((), device=device, dtype=torch.float32)
        else:
            batch_loss_value = 0.0
            batch_kl_sum = 0.0
            batch_clip_sum = 0.0
            batch_token_sum = 0.0
        batch_wall_start = time.perf_counter()
        accum_index = (micro_step % grad_accum) + 1
        row_chunks_total = max(1, (total_rows + row_microbatch_size - 1) // row_microbatch_size)
        _debug_log(
            f"batch_start global_step={global_step} local_groups={int(batch['group_count'])} "
            f"rows={total_rows} seq_width={seq_width} grad_accum={grad_accum} "
            f"accum_index={accum_index}/{grad_accum} row_chunks={row_chunks_total}"
        )

        # Pallas flash attention backward requires all graph operands in bf16.
        # Cast f32 training tensors to bf16 to prevent f32 gradient contamination.
        _flash_bf16 = is_xla and os.environ.get("TMX_XLA_FLASH_ATTENTION", "0") not in ("0", "false", "no", "")
        for row_start in range(0, total_rows, row_microbatch_size):
            row_stop = min(total_rows, row_start + row_microbatch_size)
            row_wall_start = time.perf_counter()
            prompt_ids = prompt_ids_cpu[row_start:row_stop].to(device)
            prompt_mask = prompt_mask_cpu[row_start:row_stop].to(device)
            full_completion_ids = completion_ids_cpu[row_start:row_stop].to(device)
            full_completion_mask = completion_mask_cpu[row_start:row_stop].to(device)
            full_old_per_token_logps = old_per_token_logps_cpu[row_start:row_stop].to(device)
            full_ref_per_token_logps = (
                ref_per_token_logps_cpu[row_start:row_stop].to(device)
                if ref_per_token_logps_cpu is not None
                else None
            )
            active_completion_width = max(
                1,
                int(completion_mask_cpu[row_start:row_stop].sum(dim=1).max().item()),
            )
            advantages = advantages_cpu[row_start:row_stop].to(device)
            if _flash_bf16:
                full_old_per_token_logps = full_old_per_token_logps.to(torch.bfloat16)
                if full_ref_per_token_logps is not None:
                    full_ref_per_token_logps = full_ref_per_token_logps.to(torch.bfloat16)
                advantages = advantages.to(torch.bfloat16)
                # Use full-width tensors for fixed XLA graph shapes (avoid 30s
                # recompilation per unique active_completion_width).  completion_mask
                # already zeros out padding positions in the loss computation.
                # Mask pre-computed logps at padding to prevent exp() overflow.
                # Ensure matching widths (vLLM stores padded logps, mask may differ).
                _mask_w = full_completion_mask.size(1)
                _logp_w = full_old_per_token_logps.size(1)
                if _mask_w != _logp_w:
                    _target_w = max(_mask_w, _logp_w)
                    if _mask_w < _target_w:
                        full_completion_mask = torch.nn.functional.pad(full_completion_mask, (0, _target_w - _mask_w), value=0)
                        full_completion_ids = torch.nn.functional.pad(full_completion_ids, (0, _target_w - _mask_w), value=0)
                    if _logp_w < _target_w:
                        full_old_per_token_logps = torch.nn.functional.pad(full_old_per_token_logps, (0, _target_w - _logp_w), value=0)
                        if full_ref_per_token_logps is not None:
                            full_ref_per_token_logps = torch.nn.functional.pad(full_ref_per_token_logps, (0, _target_w - _logp_w), value=0)
                _logp_mask = full_completion_mask.to(full_old_per_token_logps.dtype)
                full_old_per_token_logps = full_old_per_token_logps * _logp_mask
                if full_ref_per_token_logps is not None:
                    full_ref_per_token_logps = full_ref_per_token_logps * _logp_mask
                completion_ids = full_completion_ids
                completion_mask = full_completion_mask
                old_per_token_logps = full_old_per_token_logps
                ref_per_token_logps = full_ref_per_token_logps
            else:
                completion_ids = full_completion_ids[:, :active_completion_width]
                completion_mask = full_completion_mask[:, :active_completion_width]
                old_per_token_logps = full_old_per_token_logps[:, :active_completion_width]
                ref_per_token_logps = (
                    full_ref_per_token_logps[:, :active_completion_width]
                    if full_ref_per_token_logps is not None
                    else None
                )
            if use_token_chunk_train:
                _debug_log(
                    f"forward_start rows={row_start}:{row_stop} prompt_shape={tuple(prompt_ids.shape)} "
                    f"completion_shape={tuple(completion_ids.shape)} active_completion_width={active_completion_width} "
                    f"token_chunk={token_chunk_size}"
                )
                scaled_loss, chunk_kl_sum_tensor, chunk_clip_sum_tensor, chunk_token_sum_tensor = _compute_chunked_train_loss(
                    explorer_model,
                    prompt_ids,
                    prompt_mask,
                    completion_ids,
                    completion_mask,
                    old_per_token_logps,
                    ref_per_token_logps,
                    advantages,
                    train_temperature=float(train_temperature),
                    beta_val=beta_val,
                    loss_type=cfg.grpo_loss_type,
                    total_rows=total_rows,
                    full_completion_tokens=full_completion_tokens,
                    seq_width=seq_width,
                    token_chunk_size=token_chunk_size,
                    debug_log=_debug_log if debug_progress else None,
                )
                _debug_log(f"forward_done rows={row_start}:{row_stop}")
                if scalar_metric_tensors:
                    chunk_token_sum = chunk_token_sum_tensor
                else:
                    chunk_token_sum = float(chunk_token_sum_tensor.detach().item())
            else:
                input_ids = torch.cat([prompt_ids, full_completion_ids], dim=1)
                attention_mask = torch.cat([prompt_mask, full_completion_mask], dim=1)
                # Pad to 1024 multiples for Pallas flash attention (avoids XLA
                # mixed-precision pad ops that fail during graph compilation).
                _FLASH_PAD = 1024
                _input_seq_len = input_ids.size(1)
                _flash_pad = (-_input_seq_len) % _FLASH_PAD
                if _flash_pad > 0 and is_xla and os.environ.get("TMX_XLA_FLASH_ATTENTION", "0") not in ("0", "false", "no", ""):
                    _pad_ids = torch.full(
                        (input_ids.size(0), _flash_pad),
                        pad_id, dtype=input_ids.dtype, device=input_ids.device,
                    )
                    _pad_mask = torch.zeros(
                        input_ids.size(0), _flash_pad,
                        dtype=attention_mask.dtype, device=attention_mask.device,
                    )
                    input_ids = torch.cat([input_ids, _pad_ids], dim=1)
                    attention_mask = torch.cat([attention_mask, _pad_mask], dim=1)
                _debug_log(
                    f"forward_start rows={row_start}:{row_stop} input_shape={tuple(input_ids.shape)} "
                    f"completion_shape={tuple(full_completion_ids.shape)} active_completion_width={active_completion_width}"
                )
                outputs = explorer_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    logits_to_keep=int(full_completion_ids.size(1)) + 1,
                    use_cache=False,
                    return_dict=True,
                )
                _debug_log(f"forward_done rows={row_start}:{row_stop}")
                logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits
                logits = logits[:, :-1, :]
                logits = logits[:, -full_completion_ids.size(1):]
                if not _flash_bf16 and active_completion_width < int(full_completion_ids.size(1)):
                    _debug_log(
                        f"loss_trim rows={row_start}:{row_stop} "
                        f"full_completion_width={int(full_completion_ids.size(1))} "
                        f"active_completion_width={active_completion_width}"
                    )
                    logits = logits[:, :active_completion_width]
                logits = logits / float(train_temperature)
                per_token_logps = _selective_log_softmax(logits, completion_ids) * completion_mask.to(logits.dtype)
                # Flash attention may produce NaN logits at padding positions.
                # NaN * 0 (mask) = NaN in IEEE 754, poisoning .sum() metrics.
                # Replace NaN with 0 at masked positions — no effect on loss/gradients.
                per_token_logps = torch.nan_to_num(per_token_logps, nan=0.0)
                if ref_per_token_logps is None:
                    ref_per_token_logps = old_per_token_logps
                per_token_kl = (
                    torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1.0
                )
                per_token_kl = torch.nan_to_num(per_token_kl, nan=0.0)
                coef_1 = torch.exp(per_token_logps - old_per_token_logps)
                coef_2 = torch.clamp(coef_1, 1 - 0.2, 1 + 0.2)
                per_token_loss1 = coef_1 * advantages.unsqueeze(1)
                per_token_loss2 = coef_2 * advantages.unsqueeze(1)
                per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
                if beta_val != 0.0:
                    per_token_loss = per_token_loss + beta_val * per_token_kl

                if scalar_metric_tensors:
                    chunk_token_sum = completion_mask.to(torch.float32).sum().detach()
                else:
                    chunk_token_sum = float(completion_mask.sum().item())
                if cfg.grpo_loss_type == "grpo":
                    row_loss_values = (per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
                    scaled_loss = row_loss_values.mean() * (float(row_stop - row_start) / float(total_rows))
                elif cfg.grpo_loss_type in ("bnpo", "dr_grpo", "dapo"):
                    scaled_loss = (per_token_loss * completion_mask).sum() / max(1.0, full_completion_tokens)
                else:
                    raise ValueError(f"Unknown grpo_loss_type: {cfg.grpo_loss_type!r}")
            if scalar_metric_tensors:
                _debug_log(f"loss_ready rows={row_start}:{row_stop}")
                _debug_log(f"backward_start rows={row_start}:{row_stop} scaled_loss=deferred")
            else:
                _debug_log(
                    f"backward_start rows={row_start}:{row_stop} scaled_loss={float(scaled_loss.detach().item()):.6f}"
                )
            (scaled_loss / grad_accum).backward()
            _debug_log(f"backward_done rows={row_start}:{row_stop}")

            if scalar_metric_tensors:
                batch_loss_value = batch_loss_value + scaled_loss.detach().to(torch.float32)
                if use_token_chunk_train:
                    batch_kl_sum = batch_kl_sum + chunk_kl_sum_tensor
                    batch_clip_sum = batch_clip_sum + chunk_clip_sum_tensor
                else:
                    batch_kl_sum = batch_kl_sum + (per_token_kl * completion_mask).sum().detach().to(torch.float32)
                    batch_clip_sum = batch_clip_sum + (((coef_1 != coef_2).float() * completion_mask).sum()).detach().to(torch.float32)
                batch_token_sum = batch_token_sum + chunk_token_sum.to(torch.float32)
            else:
                batch_loss_value += float(scaled_loss.detach().item())
                if use_token_chunk_train:
                    batch_kl_sum += float(chunk_kl_sum_tensor.detach().item())
                    batch_clip_sum += float(chunk_clip_sum_tensor.detach().item())
                else:
                    batch_kl_sum += float((per_token_kl * completion_mask).sum().detach().item())
                    batch_clip_sum += float((((coef_1 != coef_2).float() * completion_mask).sum()).detach().item())
                batch_token_sum += chunk_token_sum

            del full_completion_ids
            del full_completion_mask
            del full_old_per_token_logps
            if full_ref_per_token_logps is not None:
                del full_ref_per_token_logps
            if not use_token_chunk_train:
                del outputs
                del logits
                del per_token_logps
                del per_token_kl
                del coef_1
                del coef_2
                del per_token_loss1
                del per_token_loss2
                del per_token_loss
                if cfg.grpo_loss_type == "grpo":
                    del row_loss_values
            if row_mark_step:
                _debug_log(f"row_mark_step_start rows={row_start}:{row_stop}")
                mark_step(is_xla=is_xla)
                _debug_log(f"row_mark_step_done rows={row_start}:{row_stop}")
            rows_done = row_stop
            row_chunks_done = max(1, (rows_done + row_microbatch_size - 1) // row_microbatch_size)
            batch_elapsed = time.perf_counter() - batch_wall_start
            row_elapsed = time.perf_counter() - row_wall_start
            eta_seconds = 0.0
            if rows_done < total_rows and rows_done > 0:
                eta_seconds = batch_elapsed * (float(total_rows - rows_done) / float(rows_done))
            _debug_log(
                f"row_progress rows_done={rows_done}/{total_rows} "
                f"row_chunks_done={row_chunks_done}/{row_chunks_total} "
                f"accum_index={accum_index}/{grad_accum} "
                f"row_s={row_elapsed:.1f} batch_elapsed_s={batch_elapsed:.1f} eta_s={eta_seconds:.1f}"
            )
        micro_step += 1
        if micro_step % grad_accum != 0:
            batch_elapsed = time.perf_counter() - batch_wall_start
            _debug_log(
                f"accum_progress accum_index={accum_index}/{grad_accum} "
                f"batch_elapsed_s={batch_elapsed:.1f} optimizer_pending=1"
            )
            if accum_mark_step:
                _debug_log(
                    f"accum_mark_step_start accum_index={accum_index}/{grad_accum} "
                    f"reset_scope={int(accum_mark_reset_scope)}"
                )
                mark_step(is_xla=is_xla, reset_scope=accum_mark_reset_scope)
                _debug_log(f"accum_mark_step_done accum_index={accum_index}/{grad_accum}")
            continue
        _debug_log("clip_start")
        if use_xla_fsdp and hasattr(explorer_model, "clip_grad_norm_"):
            total_grad_norm = explorer_model.clip_grad_norm_(max_norm=1.0)
        else:
            total_grad_norm = torch.nn.utils.clip_grad_norm_(explorer_model.parameters(), max_norm=1.0)
        _debug_log("clip_done")
        if pre_optimizer_mark_step:
            _debug_log("pre_optimizer_mark_step_start")
            mark_step(is_xla=is_xla)
            _debug_log("pre_optimizer_mark_step_done")
        if use_xla_fsdp:
            _debug_log("optimizer_step_start")
            optimizer.step()
            mark_step(is_xla=is_xla)
            _debug_log("optimizer_step_done")
        else:
            _debug_log("optimizer_step_start")
            optimizer_step(optimizer, is_xla=is_xla)
            mark_step(is_xla=is_xla)
            _debug_log("optimizer_step_done")
        batch_elapsed = time.perf_counter() - batch_wall_start
        _debug_log(
            f"batch_complete accum_index={accum_index}/{grad_accum} "
            f"batch_elapsed_s={batch_elapsed:.1f}"
        )
        optimizer.zero_grad()
        global_step += 1
        if scalar_metric_tensors:
            batch_loss_value = float(batch_loss_value.detach().item())
            batch_kl_sum = float(batch_kl_sum.detach().item())
            batch_clip_sum = float(batch_clip_sum.detach().item())
            batch_token_sum = float(batch_token_sum.detach().item())
        step_wall_time = time.perf_counter() - batch_wall_start
        grad_norm_value = float(total_grad_norm.item()) if hasattr(total_grad_norm, "item") else float(total_grad_norm)
        reward_mean = float(rewards_cpu.mean().item())
        reward_std = float(rewards_cpu.std(unbiased=False).item())
        correctness_mean = float(r_correct_cpu.mean().item())
        novelty_mean = float(r_novel_cpu.mean().item())
        correct_fraction = float(is_correct_cpu.mean().item())
        mean_kl = float(batch_kl_sum / max(1.0, batch_token_sum))
        clip_ratio = float(batch_clip_sum / max(1.0, batch_token_sum))
        t_rows = batch.get("terminated_rows", [])
        t_count = len(t_rows)
        if t_count > 0:
            t_final_mean = float(sum(float(r["reward"]) for r in t_rows) / t_count)
            t_corr_mean = float(sum(float(r["r_correct"]) for r in t_rows) / t_count)
            t_nov_mean = float(sum(float(r["r_novel"]) for r in t_rows) / t_count)
        else:
            t_final_mean = 0.0
            t_corr_mean = 0.0
            t_nov_mean = 0.0
        log_row = {
            "step": int(global_step),
            "loss": float(batch_loss_value),
            "reward": reward_mean,
            "reward_std": reward_std,
            "correctness_used_mean": correctness_mean,
            "novelty_used_mean": novelty_mean,
            "correct_fraction": correct_fraction,
            "kl": mean_kl,
            "clip_ratio": clip_ratio,
            "mean_completion_len": float(batch["mean_completion_len"]),
            "terminated_fraction": float(batch["terminated_fraction"]),
            "clipped_fraction": float(batch["clipped_fraction"]),
            "terminated_count": float(t_count),
            "terminated_final_mean": t_final_mean,
            "terminated_correctness_mean": t_corr_mean,
            "terminated_novelty_mean": t_nov_mean,
        }
        log_history.append(log_row)
        if use_wandb and wandb_run is not None and wandb is not None:
            payload = {
                f"{grpo_metric_prefix}/step": float(global_step),
                f"{grpo_metric_prefix}/reward": reward_mean,
                f"{grpo_metric_prefix}/reward_std": reward_std,
                f"{grpo_metric_prefix}/kl": mean_kl,
                f"{grpo_metric_prefix}/reward/final_mean": reward_mean,
                f"{grpo_metric_prefix}/reward/correctness_used_mean": correctness_mean,
                f"{grpo_metric_prefix}/reward/novelty_used_mean": novelty_mean,
                f"{grpo_metric_prefix}/clip_ratio": clip_ratio,
                f"{grpo_metric_prefix}/completion_token_length_mean": float(batch["mean_completion_len"]),
                f"{grpo_metric_prefix}/reward/completion_token_length_mean": float(batch["mean_completion_len"]),
                f"{grpo_metric_prefix}/terminated_fraction": float(batch["terminated_fraction"]),
                f"{grpo_metric_prefix}/reward/terminated_fraction": float(batch["terminated_fraction"]),
                f"{grpo_metric_prefix}/clipped_fraction": float(batch["clipped_fraction"]),
                f"{grpo_metric_prefix}/reward/clipped_fraction": float(batch["clipped_fraction"]),
                f"{grpo_metric_prefix}/reward/terminated_count": float(t_count),
                f"{grpo_metric_prefix}/reward/terminated_final_mean": t_final_mean,
                f"{grpo_metric_prefix}/reward/terminated_correctness_mean": t_corr_mean,
                f"{grpo_metric_prefix}/reward/terminated_novelty_mean": t_nov_mean,
                f"{grpo_metric_prefix}/correct_fraction": correct_fraction,
                f"{grpo_metric_prefix}/grad_norm": grad_norm_value,
                f"{grpo_metric_prefix}/wall_time_seconds": step_wall_time,
                f"{grpo_metric_prefix}/learning_rate": float(cfg.grpo_lr),
            }
            # Round-level reward_batch_summary intentionally NOT logged here;
            # per-step metrics above are more granular and avoid blocky charts.
            if train_metric_prefix:
                payload[f"{train_metric_prefix}/step"] = float(global_step)
                payload[f"{train_metric_prefix}/loss"] = float(batch_loss_value)
                payload[f"{train_metric_prefix}/reward"] = reward_mean
                payload[f"{train_metric_prefix}/grad_norm"] = grad_norm_value
            wandb_run.log(payload)
        if save_steps > 0 and global_step % save_steps == 0:
            _save_explorer_step_checkpoint(explorer_model, tokenizer, output_dir, global_step, novelty_scorer=novelty_scorer)
        if global_step >= int(max_steps):
            break
    return explorer_model, log_history


def train_explorer_with_decoupled_grpo_fsdp(
    train_ds,
    cfg,
    device,
    is_xla: bool,
    use_wandb: bool,
    wandb_run,
    tokenizer,
    explorer_model,
    anchor_model,
    *,
    grpo_metric_prefix: str,
    train_metric_prefix: Optional[str],
    run_label: str,
    output_dir: str,
    train_temperature: float,
    answer_constraints,
):
    if not is_xla:
        raise RuntimeError("Decoupled FSDP training path requires XLA/TPU.")
    rank, world_size = _xla_runtime_rank_world()
    if world_size <= 1:
        raise RuntimeError("Decoupled FSDP training path requires world_size > 1.")
    safe_eos_id = _normalize_token_id(tokenizer.eos_token_id)
    is_master = rank == 0
    data_indices = list(range(len(train_ds)))
    random.Random(cfg.seed).shuffle(data_indices)
    cursor = 0
    base_round_groups = max(1, int(os.environ.get("TMX_EXPLORER_DECOUPLED_BUFFER_GROUPS", "32") or 32))
    batch_groups = max(1, int(os.environ.get("TMX_EXPLORER_DECOUPLED_BATCH_GROUPS", str(getattr(cfg, "grpo_batch_size", 1) or 1)) or 1))
    grad_accum = max(
        1,
        int(
            os.environ.get(
                "TMX_EXPLORER_DECOUPLED_GRAD_ACCUM",
                str(getattr(cfg, "grpo_grad_accum", 1) or 1),
            )
            or 1
        ),
    )
    groups_per_step_per_rank = max(1, batch_groups * grad_accum)
    groups_per_step_global = max(1, world_size * groups_per_step_per_rank)
    total_steps = max(1, int(cfg.grpo_max_steps))
    global_step = 0
    log_history = []
    persistent_rollout_novelty_scorer = None
    os.makedirs(output_dir, exist_ok=True)
    live_model_dir = os.path.join(output_dir, "_decoupled_fsdp_live_model")
    if is_master:
        _save_plain_model_checkpoint(explorer_model, tokenizer, live_model_dir)
        print(
            f"[decoupled-explorer] starting {run_label} with total_steps={total_steps} "
            f"buffer_groups_per_round={base_round_groups} world_size={world_size} "
            f"groups_per_step_per_rank={groups_per_step_per_rank} mode=fsdp",
            flush=True,
        )
    _xla_rendezvous("tmx_decoupled_fsdp_init_model_saved", is_xla=True)
    current_model = explorer_model.to(device)
    fsdp_grad_ckpt = _env_flag("TMX_EXPLORER_DECOUPLED_TRAIN_FSDP_GRAD_CKPT", default=True)
    current_model = _wrap_model_with_xla_fsdp(current_model, use_grad_ckpt=fsdp_grad_ckpt)

    round_counter = 0
    while global_step < total_steps:
        remaining_steps = max(0, total_steps - global_step)
        groups_this_round = min(
            base_round_groups * world_size,
            remaining_steps * groups_per_step_global,
        )
        if cursor + groups_this_round > len(data_indices):
            random.Random(cfg.seed + global_step + 1).shuffle(data_indices)
            cursor = 0
        round_indices = data_indices[cursor : cursor + groups_this_round]
        cursor += groups_this_round
        round_ds = train_ds.select(round_indices)
        buffer_path = os.path.join(output_dir, f"rollout_buffer_step_{int(global_step):06d}.jsonl")

        if is_master:
            rollout_base_model = causal_lm_from_pretrained(live_model_dir, is_xla=False)
            rollout_model, rollout_novelty_scorer = _rebuild_explorer_model_for_generation(
                rollout_base_model,
                tokenizer,
                cfg,
                device,
                is_xla=is_xla,
                safe_eos_id=safe_eos_id,
                answer_constraints=answer_constraints,
                reuse_novelty_scorer=persistent_rollout_novelty_scorer,
            )
            persistent_rollout_novelty_scorer = rollout_novelty_scorer or persistent_rollout_novelty_scorer
            rollout_groups, _fsdp_reward_summary = _generate_explorer_rollout_groups(
                rollout_model,
                tokenizer,
                round_ds,
                cfg,
                rollout_novelty_scorer,
                anchor_model,
                device,
                is_xla=is_xla,
                train_temperature=train_temperature,
                explorer_step=global_step,
            )
            print(f"[decoupled-explorer] rollout groups ready count={len(rollout_groups)}", flush=True)
            _write_explorer_rollout_buffer(buffer_path, rollout_groups)
            print(f"[decoupled-explorer] wrote rollout buffer {buffer_path}", flush=True)
            del rollout_model
            gc.collect()
        _xla_rendezvous(f"tmx_decoupled_fsdp_rollout_ready_{int(global_step):06d}", is_xla=True)

        rollout_groups = _read_explorer_rollout_buffer(buffer_path)
        current_model, round_logs = _train_explorer_from_rollout_groups(
            current_model,
            tokenizer,
            rollout_groups,
            cfg,
            device,
            is_xla=is_xla,
            train_temperature=train_temperature,
            grpo_metric_prefix=grpo_metric_prefix,
            train_metric_prefix=train_metric_prefix,
            use_wandb=use_wandb,
            wandb_run=wandb_run,
            output_dir=output_dir,
            start_step=global_step,
            max_steps=global_step + max(1, len(rollout_groups) // groups_per_step_global),
        )
        if is_master:
            print(f"[decoupled-explorer] training round returned logs={len(round_logs)}", flush=True)
        _save_xla_fsdp_full_checkpoint(current_model, tokenizer, live_model_dir)
        _xla_rendezvous(f"tmx_decoupled_fsdp_live_model_ready_{int(global_step):06d}", is_xla=True)
        log_history.extend(round_logs)
        global_step += len(round_logs)
        if is_master:
            print(
                f"[decoupled-explorer] completed round with groups={len(rollout_groups)} "
                f"new_steps={len(round_logs)} total_steps={global_step}/{total_steps}",
                flush=True,
            )
        # Periodic vLLM reload: keeps vLLM weights fresh so its logprobs match the training policy.
        # All ranks must enter the helper — it does its own rank-0 gate internally via xm.rendezvous.
        # If we gate on is_master here, rank 0 deadlocks at the internal rendezvous waiting for ranks 1-3.
        # Counted by ROUND (not global_step) because each round advances global_step by >1 (often ~32),
        # so step-modulo would almost never hit exact multiples of N.
        round_counter += 1
        reload_every_rounds = int(os.environ.get("TMX_EXPLORER_VLLM_RELOAD_EVERY_N_ROUNDS", "1") or 1)
        should_reload = (
            reload_every_rounds > 0
            and global_step < total_steps
            and (round_counter % reload_every_rounds == 0)
            and bool(os.environ.get("TMX_EXPLORER_VLLM_RELOAD_TPU_NAME"))
        )
        if should_reload:
            try:
                from .pipeline import _save_model_to_gcs_and_reload_vllm
                run_name_env = os.environ.get("TMX_WANDB_RUN_NAME", "").strip() or os.environ.get("WANDB_RUN_NAME", "").strip() or "explorer_run"
                reload_model = None
                if is_master:
                    print(
                        f"[decoupled-explorer] reloading vLLM with fresh policy weights at step={global_step} (run={run_name_env})",
                        flush=True,
                    )
                    reload_model = causal_lm_from_pretrained(live_model_dir, is_xla=False)
                # All ranks call — helper gates on rank 0 via internal rendezvous.
                _save_model_to_gcs_and_reload_vllm(reload_model, tokenizer, run_name_env, role="explorer")
                if is_master and reload_model is not None:
                    del reload_model
                    gc.collect()
                    print(f"[decoupled-explorer] vLLM reload complete at step={global_step}", flush=True)
            except Exception as reload_exc:
                if is_master:
                    print(f"[decoupled-explorer] WARNING vLLM reload failed: {reload_exc}", flush=True)
        _xla_rendezvous(f"tmx_decoupled_fsdp_reload_done_{int(global_step):06d}", is_xla=True)

    final_model = causal_lm_from_pretrained(live_model_dir, is_xla=False) if is_master else current_model
    final_novelty_scorer = None
    if is_master and _should_build_novelty_scorer(cfg):
        novelty_device = torch.device("cpu") if _use_cpu_novelty(is_xla) else device
        final_novelty_scorer, _ = build_novelty_scorer(
            final_model,
            tokenizer,
            cfg,
            novelty_device,
            is_xla=False if novelty_device.type == "cpu" else is_xla,
        )
    return final_model, final_novelty_scorer, log_history


def train_explorer_with_decoupled_grpo(
    train_ds,
    cfg,
    device,
    is_xla: bool,
    use_wandb: bool,
    wandb_run,
    tokenizer,
    explorer_model,
    anchor_model,
    *,
    grpo_metric_prefix: str,
    train_metric_prefix: Optional[str],
    run_label: str,
    output_dir: str,
    train_temperature: float,
    answer_constraints,
):
    rank, world_size = _xla_runtime_rank_world() if is_xla else (0, 1)
    if is_xla and os.environ.get("TMX_FSDP", "1") == "0":
        world_size = 1  # single-chip training: ignore other visible devices
    if is_xla and world_size > 1:
        raise RuntimeError(
            "Decoupled non-FSDP Explorer training is not safe under multi-process XLA. "
            "Use TMX_EXPLORER_DECOUPLED_TRAIN_FSDP=1 for the multi-rank path or "
            "launch ddp_local with DDP_NUM_PROCESSES=1 / DDP_VISIBLE_CHIPS=0."
        )
    safe_eos_id = _normalize_token_id(tokenizer.eos_token_id)
    current_model = explorer_model.to(device)
    data_indices = list(range(len(train_ds)))
    random.Random(cfg.seed).shuffle(data_indices)
    cursor = 0
    buffer_groups_per_round = max(1, int(os.environ.get("TMX_EXPLORER_DECOUPLED_BUFFER_GROUPS", "32") or 32))
    batch_groups = max(1, int(os.environ.get("TMX_EXPLORER_DECOUPLED_BATCH_GROUPS", str(getattr(cfg, "grpo_batch_size", 1) or 1)) or 1))
    grad_accum = max(
        1,
        int(
            os.environ.get(
                "TMX_EXPLORER_DECOUPLED_GRAD_ACCUM",
                str(getattr(cfg, "grpo_grad_accum", 1) or 1),
            )
            or 1
        ),
    )
    groups_per_step = max(1, batch_groups * grad_accum)
    total_steps = max(1, int(cfg.grpo_max_steps))
    global_step = 0
    log_history = []
    persistent_rollout_novelty_scorer = None
    os.makedirs(output_dir, exist_ok=True)
    # Resume from checkpoint if TMX_RESUME_STEP is set.
    _resume_step = int(os.environ.get("TMX_RESUME_STEP", "0") or 0)
    _resume_ckpt = os.environ.get("TMX_RESUME_CHECKPOINT", "")
    if _resume_step > 0:
        global_step = _resume_step
        cursor = (_resume_step * groups_per_step) % len(data_indices)
        print(f"[decoupled-explorer] resuming from step={_resume_step} cursor={cursor}", flush=True)
    _resume_rnd_path = ""
    if _resume_ckpt and os.path.isdir(_resume_ckpt):
        from transformers import AutoModelForCausalLM
        print(f"[decoupled-explorer] loading checkpoint weights from {_resume_ckpt}", flush=True)
        _ckpt_model = AutoModelForCausalLM.from_pretrained(
            _resume_ckpt, torch_dtype=torch.bfloat16, attn_implementation="eager"
        )
        current_model.load_state_dict(_ckpt_model.state_dict())
        del _ckpt_model
        print("[decoupled-explorer] checkpoint weights loaded", flush=True)
        _rnd_candidate = os.path.join(_resume_ckpt, "rnd_state.pt")
        if os.path.isfile(_rnd_candidate):
            _resume_rnd_path = _rnd_candidate
            print(f"[decoupled-explorer] found RND state at {_rnd_candidate}", flush=True)
        else:
            print("[decoupled-explorer] WARNING: no rnd_state.pt in checkpoint — RND will start fresh", flush=True)
    print(
        f"[decoupled-explorer] starting {run_label} with total_steps={total_steps} "
        f"buffer_groups_per_round={buffer_groups_per_round} "
        f"groups_per_step={groups_per_step} rank={rank} world_size={world_size}"
    )
    while global_step < total_steps:
        train_temperature = schedule_value(cfg.temperature_schedule, global_step / total_steps)
        groups_this_round = min(buffer_groups_per_round, (total_steps - global_step) * groups_per_step)
        if cursor + groups_this_round > len(data_indices):
            random.Random(cfg.seed + global_step + 1).shuffle(data_indices)
            cursor = 0
        round_indices = data_indices[cursor : cursor + groups_this_round]
        cursor += groups_this_round
        round_ds = train_ds.select(round_indices)
        rollout_model, rollout_novelty_scorer = _rebuild_explorer_model_for_generation(
            current_model,
            tokenizer,
            cfg,
            device,
            is_xla=is_xla,
            safe_eos_id=safe_eos_id,
            answer_constraints=answer_constraints,
            reuse_novelty_scorer=persistent_rollout_novelty_scorer,
        )
        persistent_rollout_novelty_scorer = rollout_novelty_scorer or persistent_rollout_novelty_scorer
        # Restore RND state from checkpoint on first round after resume.
        if _resume_rnd_path and persistent_rollout_novelty_scorer is not None:
            try:
                rnd_state = torch.load(_resume_rnd_path, map_location="cpu")
                restored = 0
                for key, rnd_mod in persistent_rollout_novelty_scorer.rnd_map.items():
                    if key in rnd_state:
                        rnd_mod.target.load_state_dict(rnd_state[key]["target"])
                        rnd_mod.predictor.load_state_dict(rnd_state[key]["predictor"])
                        rnd_mod.optimizer.load_state_dict(rnd_state[key]["optimizer"])
                        restored += 1
                print(f"[decoupled-explorer] restored RND state for {restored}/{len(persistent_rollout_novelty_scorer.rnd_map)} modules from {_resume_rnd_path}", flush=True)
            except Exception as e:
                print(f"[decoupled-explorer] WARNING: failed to restore RND state: {e}", flush=True)
            _resume_rnd_path = ""  # Only restore once.
        rollout_groups, _reward_summary = _generate_explorer_rollout_groups(
            rollout_model,
            tokenizer,
            round_ds,
            cfg,
            rollout_novelty_scorer,
            anchor_model,
            device,
            is_xla=is_xla,
            train_temperature=train_temperature,
            explorer_step=global_step,
        )
        print(f"[decoupled-explorer] rollout groups ready count={len(rollout_groups)}", flush=True)
        # --- DARLING diversity metrics (logged once per generation round) ---
        if use_wandb and wandb_run is not None:
            try:
                from .diversity_metrics import compute_all_diversity_metrics
                _div_metrics = compute_all_diversity_metrics(
                    rollout_groups,
                    step=global_step,
                    log_interval=1,  # log every round (rounds are ~16 steps apart)
                )
                if _div_metrics:
                    _div_payload = {f"{grpo_metric_prefix}/{k}": v for k, v in _div_metrics.items()}
                    _div_payload[f"{grpo_metric_prefix}/step"] = float(global_step)
                    wandb_run.log(_div_payload)
                    print(
                        f"[diversity] Logged dist_3={_div_metrics.get('diversity/dist_3', 0):.4f} "
                        f"dist_4={_div_metrics.get('diversity/dist_4', 0):.4f} "
                        f"at step {global_step}",
                        flush=True,
                    )
            except Exception as _div_err:
                print(f"[diversity] WARNING: metrics computation failed: {_div_err}", flush=True)
        buffer_path = os.path.join(output_dir, f"rollout_buffer_step_{int(global_step):06d}.jsonl")
        # Write rollout buffer async — overlap disk I/O with training.
        import copy as _copy
        _buf_groups_copy = _copy.deepcopy(rollout_groups)
        _buf_thread = threading.Thread(
            target=_write_explorer_rollout_buffer,
            args=(buffer_path, _buf_groups_copy),
            daemon=True,
        )
        _buf_thread.start()
        print(f"[decoupled-explorer] rollout buffer write started async {buffer_path}", flush=True)
        state_dict = _snapshot_rebuildable_model_state(rollout_model)
        print("[decoupled-explorer] snapshotted rollout model state", flush=True)
        del rollout_model
        gc.collect()
        _maybe_xla_mark_step(device.type)
        # Offload novelty model from TPU to free HBM for training.
        if not _use_cpu_novelty(is_xla) and persistent_rollout_novelty_scorer is not None:
            _ns_model = getattr(persistent_rollout_novelty_scorer, "embed_model", None)
            if _ns_model is not None:
                persistent_rollout_novelty_scorer.embed_model = _ns_model.to(torch.device("cpu"))
                persistent_rollout_novelty_scorer.device = torch.device("cpu")
                gc.collect()
                _maybe_xla_mark_step(device.type)
                print("[novelty] Offloaded embed_model to CPU to free HBM for training.", flush=True)
        # Allow CPU training to skip XLA compilation overhead.
        _train_cpu = (os.environ.get("TMX_DECOUPLED_TRAIN_CPU", "") or "").strip().lower() in ("1", "true", "yes")
        _train_device = torch.device("cpu") if _train_cpu else device
        _train_is_xla = False if _train_cpu else is_xla
        current_model = _build_explorer_model_from_state(
            state_dict,
            tokenizer,
            cfg,
            _train_device,
            is_xla=_train_is_xla,
            safe_eos_id=safe_eos_id,
            answer_constraints=answer_constraints,
            patch_generate=False,
        )
        print(f"[decoupled-explorer] rebuilt training model on {'cpu' if _train_cpu else device}", flush=True)
        current_model, round_logs = _train_explorer_from_rollout_groups(
            current_model,
            tokenizer,
            rollout_groups,
            cfg,
            _train_device,
            is_xla=_train_is_xla,
            train_temperature=train_temperature,
            grpo_metric_prefix=grpo_metric_prefix,
            train_metric_prefix=train_metric_prefix,
            use_wandb=use_wandb,
            wandb_run=wandb_run,
            output_dir=output_dir,
            start_step=global_step,
            max_steps=global_step + max(1, len(rollout_groups) // groups_per_step),
            novelty_scorer=persistent_rollout_novelty_scorer,
            reward_batch_summary=_reward_summary,
        )
        print(f"[decoupled-explorer] training round returned logs={len(round_logs)}", flush=True)
        log_history.extend(round_logs)
        global_step += len(round_logs)
        print(
            f"[decoupled-explorer] completed round with groups={len(rollout_groups)} "
            f"new_steps={len(round_logs)} total_steps={global_step}/{total_steps}"
        )
    final_novelty_scorer = None
    if _should_build_novelty_scorer(cfg):
        if persistent_rollout_novelty_scorer is not None and _use_cpu_novelty(is_xla):
            final_state_dict = _snapshot_rebuildable_model_state(current_model)
            final_novelty_scorer = _refresh_novelty_scorer_from_state(
                persistent_rollout_novelty_scorer,
                final_state_dict,
            )
            print("[novelty] Refreshed persistent CPU scorer with final Explorer weights.")
        else:
            final_novelty_scorer, _ = build_novelty_scorer(current_model, tokenizer, cfg, device, is_xla)
    return current_model, final_novelty_scorer, log_history


def build_novelty_scorer(embed_model, tokenizer, cfg, device, is_xla: bool):
    hidden_dim = int(embed_model.config.hidden_size)
    feature_source = (cfg.novelty_feature_source or "last").lower()

    if feature_source == "multilayer":
        num_hidden_layers = int(getattr(embed_model.config, "num_hidden_layers", 0))
        selected_layers = list(cfg.novelty_layers) if cfg.novelty_layers else auto_layer_selection(num_hidden_layers)
        rnd = {
            f"layer_{int(layer_idx)}": RNDModule(
                input_dim=hidden_dim,
                hidden_dim=cfg.rnd_hidden,
                lr=cfg.rnd_lr,
                device=device,
                is_xla=is_xla,
            )
            for layer_idx in selected_layers
        }
        print(f"[novelty] Using multilayer RND with layers={selected_layers}, pool={cfg.novelty_layer_pool}, agg={cfg.novelty_layer_agg}")
    else:
        selected_layers = []
        rnd = RNDModule(
            input_dim=hidden_dim,
            hidden_dim=cfg.rnd_hidden,
            lr=cfg.rnd_lr,
            device=device,
            is_xla=is_xla,
        )
        print("[novelty] Using last-layer novelty features.")

    novelty_scorer = NoveltyScorer(
        embed_model=embed_model,
        tokenizer=tokenizer,
        rnd_module=rnd,
        metric=cfg.novelty_metric,
        clip_value=cfg.novelty_clip,
        device=device,
        use_input_embeddings=False,
        feature_source=feature_source,
        layers=selected_layers,
        layer_pool=cfg.novelty_layer_pool,
        layer_agg=cfg.novelty_layer_agg,
        max_length=cfg.novelty_feature_max_length or cfg.max_total_len,
    )
    return novelty_scorer, rnd


def patch_grpo_trainer_generation_for_xla(trainer, answer_constraints=None):
    """
    GRPOTrainer unwraps the model and calls generate() inside its own method.
    Patch that method directly so XLA always goes through xla_safe_generate.
    """
    try:
        from torch import nn
        from transformers import Trainer
        from trl.trainer.grpo_trainer import (
            apply_chat_template,
            gather,
            gather_object,
            is_conversational,
            maybe_apply_chat_template,
            nanstd,
            pad,
            profiling_context,
            unwrap_model_for_generation,
        )
    except Exception as e:
        print(f"[WARN] Could not patch GRPOTrainer generation for XLA: {e}")
        return

    original_generate_and_score = trainer._generate_and_score_completions
    shared_constraints = dict(answer_constraints or {})

    def _generate_and_score_completions_xla(self, inputs):
        device = self.accelerator.device
        use_vllm_for_grpo = _env_flag("TMX_GRPO_USE_VLLM", default=False)
        vllm_urls_raw = (
            (os.environ.get("TMX_VLLM_SERVER_URLS") or "").strip()
            if use_vllm_for_grpo
            else ""
        )
        if not vllm_urls_raw and (device.type != "xla" or os.environ.get("TMX_XLA_SAFE_GENERATE", "1") == "0"):
            return original_generate_and_score(inputs)

        # Free gradient tensors from the previous training step to reclaim HBM
        # before the generation buffer allocation.
        try:
            self.model.zero_grad(set_to_none=True)
        except Exception:
            pass
        import gc
        gc.collect()
        if not vllm_urls_raw:
            try:
                import torch_xla.core.xla_model as _xm
                _xm.mark_step()
            except Exception:
                pass
        use_incremental_xla = os.environ.get(
            "TMX_XLA_INCREMENTAL_GENERATE", "0"
        ) not in ("", "0", "false", "False", "no", "NO")

        mode = "eval" if self.control.should_evaluate else "train"
        prompts = [x["prompt"] for x in inputs]
        _orig_local_batch_size = len(inputs)
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        prompt_length = prompt_ids.size(1)
        generation_debug = os.environ.get("TMX_DEBUG_GENERATE", "0") == "1"
        if generation_debug:
            print(
                "[patch] GRPOTrainer XLA generation "
                f"prompt_batch={len(prompts)} prompt_length={int(prompt_length)} "
                f"max_completion_length={int(self.max_completion_length)}"
            )
            # Show first prompt decoded for verification
            decoded_prompt = self.processing_class.decode(prompt_ids[0], skip_special_tokens=False)
            print(f"[patch] first prompt decoded (last 300 chars): ...{decoded_prompt[-300:]}")
            print(f"[patch] first prompt token IDs (last 20): {prompt_ids[0, -20:].tolist()}")
            print(f"[patch] tokenizer eos_token_id={self.processing_class.eos_token_id} pad_token_id={self.processing_class.pad_token_id}")
            gen_cfg = self.generation_config
            print(f"[patch] generation_config: temperature={getattr(gen_cfg, 'temperature', None)} "
                  f"top_k={getattr(gen_cfg, 'top_k', None)} top_p={getattr(gen_cfg, 'top_p', None)} "
                  f"do_sample={getattr(gen_cfg, 'do_sample', None)} "
                  f"eos_token_id={getattr(gen_cfg, 'eos_token_id', None)}")

        if vllm_urls_raw:
            # ── vLLM-accelerated generation ──────────────────────────────
            from tmx.vllm_generate import generate_completions_vllm
            vllm_urls = [u.strip() for u in vllm_urls_raw.split(",") if u.strip()]
            vllm_model = os.environ.get("TMX_VLLM_MODEL_NAME", "")
            vllm_api_key = os.environ.get("TMX_VLLM_API_KEY", "")
            vllm_concurrency = int(os.environ.get("TMX_VLLM_CONCURRENCY", "64"))
            gen_cfg = self.generation_config

            # -- SPMD batch expansion --
            # In single-process SPMD, gather() is a no-op so the local
            # accumulated batch (batch_size * grad_accum) is smaller than
            # num_generations.  Deduplicate prompts (TRL repeats each one
            # num_generations times in the dataset) and generate the full
            # num_generations group via vLLM so reward grouping works.
            n_gens = self.num_generations
            seen = set()
            unique_indices = []
            for i, p in enumerate(prompts_text):
                if p not in seen:
                    seen.add(p)
                    unique_indices.append(i)
            n_unique = len(unique_indices)
            unique_prompts_text = [prompts_text[i] for i in unique_indices]

            all_completions = generate_completions_vllm(
                prompts=unique_prompts_text,
                server_urls=vllm_urls,
                model_name=vllm_model,
                n_per_prompt=n_gens,
                temperature=getattr(gen_cfg, 'temperature', 1.0),
                top_p=getattr(gen_cfg, 'top_p', 0.95),
                max_tokens=int(self.max_completion_length),
                concurrency=vllm_concurrency,
                api_key=vllm_api_key,
            )
            assert all(len(comps) == n_gens for comps in all_completions), (
                f"Expected {n_gens} completions per prompt, got {[len(c) for c in all_completions]}"
            )
            # Flatten: n_unique * n_gens completion dicts (block-ordered)
            completion_dicts_raw = [d for comps in all_completions for d in comps]

            # Expand prompt tensors and metadata blockwise to match completions
            unique_prompt_ids = prompt_ids[unique_indices]
            unique_prompt_mask = prompt_mask[unique_indices]
            unique_prompts = [prompts[i] for i in unique_indices]
            unique_inputs = [inputs[i] for i in unique_indices]
            prompt_ids = unique_prompt_ids.repeat_interleave(n_gens, dim=0)
            prompt_mask = unique_prompt_mask.repeat_interleave(n_gens, dim=0)
            prompts = [p for p in unique_prompts for _ in range(n_gens)]
            prompts_text = [p for p in unique_prompts_text for _ in range(n_gens)]
            inputs = [x for x in unique_inputs for _ in range(n_gens)]

            completion_texts_raw = [d["text"] for d in completion_dicts_raw]
            completion_finish_reasons = [d["finish_reason"] for d in completion_dicts_raw]
            eos_token_id = self.processing_class.eos_token_id
            # Tokenize completions and pad to uniform length
            comp_token_lists = [
                self.processing_class.encode(ct, add_special_tokens=False)
                for ct in completion_texts_raw
            ]
            # Append EOS token for completions that terminated naturally (finish_reason="stop")
            # so downstream EOS detection correctly identifies them as terminated
            comp_token_lists = [
                ids + [eos_token_id] if finish_reason == "stop" and eos_token_id is not None else ids
                for ids, finish_reason in zip(comp_token_lists, completion_finish_reasons)
            ]
            max_comp_len = int(self.max_completion_length)
            comp_token_lists = [ids[:max_comp_len] for ids in comp_token_lists]
            actual_max = max((len(ids) for ids in comp_token_lists), default=0)
            pad_id = self.processing_class.pad_token_id if self.processing_class.pad_token_id is not None else 0
            comp_padded = [
                ids + [pad_id] * (actual_max - len(ids))
                for ids in comp_token_lists
            ]
            comp_ids_tensor = torch.tensor(comp_padded, dtype=prompt_ids.dtype, device=device)
            prompt_completion_ids = torch.cat([prompt_ids, comp_ids_tensor], dim=1)
            if generation_debug:
                print(f"[patch] vLLM generation: {len(completion_texts_raw)} completions, "
                      f"max_comp_tokens={actual_max}, prompt_length={prompt_length}")
                if completion_texts_raw:
                    print(f"  [0] first 200 chars: {completion_texts_raw[0][:200]}")
        else:
            # ── XLA on-device generation ─────────────────────────────────
            with unwrap_model_for_generation(
                self.model_wrapped,
                self.accelerator,
                gather_deepspeed3_params=self.args.ds3_gather_for_generation,
            ) as unwrapped_model:
                live_constraints = getattr(unwrapped_model, "_tmx_answer_constraints", None) or shared_constraints
                cfg_objs = (
                    getattr(unwrapped_model, "config", None),
                    getattr(unwrapped_model, "generation_config", None),
                )
                prev_use_cache = [getattr(cfg_obj, "use_cache", None) if cfg_obj is not None else None for cfg_obj in cfg_objs]
                grad_ckpt_was_enabled = bool(getattr(unwrapped_model, "is_gradient_checkpointing", False))
                was_training = bool(getattr(unwrapped_model, "training", False))
                if use_incremental_xla:
                    for cfg_obj in cfg_objs:
                        if cfg_obj is not None:
                            setattr(cfg_obj, "use_cache", True)
                    if grad_ckpt_was_enabled and hasattr(unwrapped_model, "gradient_checkpointing_disable"):
                        unwrapped_model.gradient_checkpointing_disable()
                    if was_training:
                        unwrapped_model.eval()
                try:
                    prompt_completion_ids = xla_safe_generate(
                        unwrapped_model,
                        input_ids=prompt_ids,
                        attention_mask=prompt_mask,
                        generation_config=self.generation_config,
                        max_new_tokens=max(1, int(self.max_completion_length)),
                        use_cache=use_incremental_xla,
                        allowed_token_ids=(live_constraints or {}).get("allowed_token_ids"),
                        initial_token_ids=(live_constraints or {}).get("initial_token_ids"),
                    )
                finally:
                    if use_incremental_xla:
                        for cfg_obj, prev in zip(cfg_objs, prev_use_cache):
                            if cfg_obj is not None:
                                setattr(cfg_obj, "use_cache", prev)
                        if grad_ckpt_was_enabled and hasattr(unwrapped_model, "gradient_checkpointing_enable"):
                            try:
                                _enable_model_grad_checkpointing(
                                    unwrapped_model,
                                    preserve_rng_state=False,
                                    verbose=False,
                                )
                            except Exception:
                                unwrapped_model.gradient_checkpointing_enable()
                        if was_training:
                            unwrapped_model.train()

        prompt_ids = prompt_completion_ids[:, :prompt_length]
        completion_ids = prompt_completion_ids[:, prompt_length:]
        if generation_debug:
            print(
                "[patch] GRPOTrainer XLA generation result "
                f"prompt_completion_shape={tuple(prompt_completion_ids.shape)} "
                f"completion_shape={tuple(completion_ids.shape)}"
            )

        # Check for all EOS tokens (Qwen3 has [151645, 151643])
        _raw_eos = getattr(self.generation_config, "eos_token_id", None)
        if isinstance(_raw_eos, (list, tuple)):
            is_eos = torch.zeros_like(completion_ids, dtype=torch.bool)
            for _eid in _raw_eos:
                is_eos = is_eos | (completion_ids == int(_eid))
        else:
            _eid = int(_raw_eos) if _raw_eos is not None else self.processing_class.eos_token_id
            is_eos = completion_ids == _eid
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        if self.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = completion_mask * (~truncated_completions).unsqueeze(1).int()

        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size
        terminated_flags = is_eos.any(dim=1)
        completion_token_lengths = completion_mask.sum(dim=1) - terminated_flags.int()
        max_new_tokens = int(completion_ids.size(1))
        completion_stats = [
            {
                "completion_token_length": int(completion_token_lengths[idx].item()),
                "terminated": bool(terminated_flags[idx].item()),
                "clipped": bool((not terminated_flags[idx].item()) and int(completion_token_lengths[idx].item()) >= max_new_tokens),
            }
            for idx in range(completion_ids.size(0))
        ]

        with torch.no_grad():
            if self.num_iterations > 1:
                old_per_token_logps = _get_per_token_logps_chunked(
                    self,
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    device.type,
                )
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = _get_per_token_logps_chunked(
                    self,
                    self.ref_model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    device.type,
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = _get_per_token_logps_chunked(
                        self,
                        self.model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size,
                        device.type,
                    )

        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if generation_debug:
            preview = completions_text[: min(2, len(completions_text))]
            print(f"[patch] GRPOTrainer decoded completion preview (first 200 chars each):")
            for ci, ct in enumerate(preview):
                print(f"  [{ci}] {ct[:200]}")
            # Show raw token IDs for first completion
            first_comp_ids = completion_ids[0].detach().cpu().tolist()
            non_pad = [t for t in first_comp_ids if t != self.processing_class.pad_token_id]
            print(f"[patch] first completion non-pad token count={len(non_pad)} first_20_ids={non_pad[:20]}")
            eos_id = self.processing_class.eos_token_id
            has_eos = any(t == eos_id for t in first_comp_ids)
            print(f"[patch] first completion has_eos_token({eos_id})={has_eos}")
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        reward_kwargs = {}
        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes, self.reward_func_names)
        ):
            with profiling_context(self, reward_func_name):
                if isinstance(reward_func, nn.Module):
                    if is_conversational(inputs[0]):
                        messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                        texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions)]
                    reward_inputs = reward_processing_class(
                        text=texts,
                        return_tensors="pt",
                        padding=True,
                        padding_side="right",
                        add_special_tokens=False,
                    )
                    reward_inputs = Trainer._prepare_inputs(self, reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
                else:
                    keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                    reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                    reward_kwargs["completion_stats"] = completion_stats
                    output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                    output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
                    rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()}
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward."
            )

        rewards_per_func = gather(rewards_per_func)
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
        assert len(rewards) % self.num_generations == 0, (
            f"Reward count {len(rewards)} is not divisible by num_generations "
            f"{self.num_generations}. Local batch={_orig_local_batch_size}, "
            f"gathered={rewards_per_func.shape[0]}."
        )
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Fallback: when ALL groups have zero within-group variance (common with
        # binary math rewards), use batch-level normalization instead. This turns
        # GRPO into REINFORCE-with-batch-baseline, which still produces learning
        # signal by comparing correct vs incorrect prompts across the batch.
        use_batch_norm = bool((std_grouped_rewards < 1e-8).all().item()) and len(rewards) > 1
        if use_batch_norm:
            batch_mean = rewards.mean()
            batch_std = rewards.std()
            if batch_std > 1e-8:
                advantages = (rewards - batch_mean) / (batch_std + 1e-4)
                if generation_debug:
                    print(f"[patch] batch-level advantage fallback: batch_mean={batch_mean:.4f} "
                          f"batch_std={batch_std:.4f} (all groups had zero within-group variance)")
            else:
                advantages = torch.zeros_like(rewards)
            # Report batch std as reward_std for logging
            std_grouped_rewards = torch.full_like(std_grouped_rewards, float(batch_std.item()))
        else:
            advantages = rewards - mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            if self.scale_rewards:
                std_grouped_rewards_expanded = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
                advantages = advantages / (std_grouped_rewards_expanded + 1e-4)

        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)

        process_slice = slice(
            self.accelerator.process_index * _orig_local_batch_size,
            (self.accelerator.process_index + 1) * _orig_local_batch_size,
        )
        advantages = advantages[process_slice]

        if mode == "train":
            self.state.num_input_tokens_seen += self.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        agg_completion_mask = self.accelerator.gather_for_metrics(completion_mask.sum(1))
        self._metrics[mode]["completions/mean_length"].append(agg_completion_mask.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_mask.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_mask.float().max().item())

        agg_terminated_with_eos = self.accelerator.gather_for_metrics(is_eos.any(dim=1))
        term_completion_mask = agg_completion_mask[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_mask) / len(agg_completion_mask)
        self._metrics[mode]["completions/clipped_ratio"].append(clipped_completions_ratio)
        if len(term_completion_mask) == 0:
            term_completion_mask = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(term_completion_mask.float().mean().item())
        self._metrics[mode]["completions/min_terminated_length"].append(term_completion_mask.float().min().item())
        self._metrics[mode]["completions/max_terminated_length"].append(term_completion_mask.float().max().item())

        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        self._textual_logs["prompt"].extend(gather_object(prompts_text))
        self._textual_logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._textual_logs["rewards"][name].extend(rewards_per_func[:, i].tolist())

        # In SPMD the batch was expanded for correct reward grouping;
        # slice all returned tensors back to the local share so the
        # outer training loop sees the expected micro-batch size.
        if prompt_ids.size(0) > _orig_local_batch_size:
            prompt_ids = prompt_ids[process_slice]
            prompt_mask = prompt_mask[process_slice]
            completion_ids = completion_ids[process_slice]
            completion_mask = completion_mask[process_slice]
            if old_per_token_logps is not None:
                old_per_token_logps = old_per_token_logps[process_slice]
            if ref_per_token_logps is not None:
                ref_per_token_logps = ref_per_token_logps[process_slice]

        xla_cpu_handoff = (
            device.type == "xla"
            and _single_worker_xla_collective_passthrough()
            and _env_flag("TMX_XLA_CPU_INPUT_HANDOFF", default=True)
        )
        if xla_cpu_handoff:
            # TRL immediately reuses these tensors inside _compute_loss(). Keep
            # the buffered handoff tensors on CPU so TRL can rehydrate them to
            # fresh XLA tensors inside _compute_loss() right before cat/model
            # usage. Keeping these cached on XLA leaves stale handles around the
            # first accumulation boundary on the single-process SPMD path.
            prompt_ids = prompt_ids.detach().cpu().clone()
            prompt_mask = prompt_mask.detach().cpu().clone()
            completion_ids = completion_ids.detach().cpu().clone()
            completion_mask = completion_mask.detach().cpu().clone()
            advantages = advantages.detach().cpu().clone()
            if old_per_token_logps is not None:
                old_per_token_logps = old_per_token_logps.detach().cpu().clone()
            if ref_per_token_logps is not None:
                ref_per_token_logps = ref_per_token_logps.detach().cpu().clone()
            # The downstream handoff no longer depends on any live XLA tensors.
            # Flush the generation/scoring graph here so the subsequent training
            # forward/backward runs against a clean XLA step.
            try:
                import torch_xla.core.xla_model as _xm

                _xm.mark_step()
            except Exception:
                pass

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
        }

    trainer._generate_and_score_completions = types.MethodType(_generate_and_score_completions_xla, trainer)
    print("[patch] GRPOTrainer._generate_and_score_completions patched for XLA.")


def train_explorer_with_grpo(train_ds,
                             cfg,
                             device,
                             is_xla: bool,
                             use_wandb: bool,
                             wandb_run=None,
                             init_model=None,
                             anchor_model=None,
                             grpo_metric_prefix: str = "grpo",
                             train_metric_prefix: Optional[str] = "train",
                             run_label: str = "Explorer"):
    """
    Train EXPLORER policy with GRPO on GSM8K prompts using correctness+novelty.
    Returns explorer model, tokenizer, novelty_scorer, and log history.
    """
    if is_xla:
        os.environ.setdefault("USE_TORCH_XLA", "1")
        os.environ.setdefault("PJRT_DEVICE", "TPU")
    else:
        os.environ.setdefault("USE_TORCH_XLA", "0")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer
    if is_xla:
        patch_accelerate_gather_object_for_xla()
        patch_accelerate_gradient_state_for_xla()
        patch_trl_grpo_compute_loss_for_xla()
        patch_transformers_accelerator_num_processes_for_xla_spmd()
        patch_trl_grpo_sampler_for_xla_spmd()
        patch_torch_xla_parallel_loader_for_single_worker()
        patch_transformers_nested_xla_mesh_reduce_for_single_worker()
        patch_transformers_trainer_logging_for_single_worker()

    print(f"Loading tokenizer for model {cfg.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, **hf_from_pretrained_kwargs(cfg.model_name, repo_type="model"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if is_xla and os.environ.get("TMX_XLA_FIXED_PADDING", "1") != "0":
        patch_tokenizer_for_xla_fixed_padding(
            tokenizer,
            cfg.max_prompt_len,
            warning_max_length=max(int(cfg.max_prompt_len), int(cfg.max_total_len)),
        )
    answer_constraints = get_answer_token_constraints(tokenizer, cfg, phase="explore")
    if answer_constraints is not None:
        print(
            "[constraints] Using answer constraint mode="
            f"{answer_constraints['mode']} with "
            f"{len(answer_constraints['initial_token_ids'])} initial tokens / "
            f"{len(answer_constraints['allowed_token_ids'])} continuation tokens."
        )
    safe_eos_id = _normalize_token_id(tokenizer.eos_token_id)
    safe_bos_id = _normalize_token_id(tokenizer.bos_token_id)

    train_temperature = schedule_value(cfg.temperature_schedule, 0.0)
    use_decoupled_xla_explorer = is_xla and _env_flag("TMX_EXPLORER_DECOUPLED_GRPO", default=True)
    world_size = 1
    try:
        world_size = int(os.environ.get("WORLD_SIZE") or os.environ.get("LOCAL_WORLD_SIZE") or "1")
    except ValueError:
        world_size = 1
    if world_size <= 0:
        world_size = 1

    use_xla_world_size = is_xla and os.environ.get("TMX_USE_XLA_WORLD_SIZE", "0") == "1"
    if use_xla_world_size:
        xm_mod, xr_mod = maybe_import_xla()
        if xr_mod is not None:
            runtime_count = None
            for attr_name in ("global_runtime_device_count", "addressable_runtime_device_count"):
                fn = getattr(xr_mod, attr_name, None)
                if fn is None:
                    continue
                try:
                    runtime_count = int(fn())
                except Exception:
                    runtime_count = None
                if runtime_count and runtime_count > 0:
                    break
            if runtime_count and runtime_count > 0:
                world_size = runtime_count
            else:
                try:
                    world_size = int(xr_mod.world_size())
                except Exception:
                    world_size = 1
        elif xm_mod is not None:
            try:
                world_size = int(xm_mod.xrt_world_size())
            except Exception:
                world_size = 1

    if is_xla:
        if use_xla_world_size:
            print(f"[xla] Using device world_size={world_size} (TMX_USE_XLA_WORLD_SIZE=1).")
        else:
            print(
                f"[xla] Using process world_size={world_size} "
                "(set TMX_USE_XLA_WORLD_SIZE=1 to use device count)."
            )

    base_batch = cfg.grpo_batch_size * world_size
    grad_accum = max(1, cfg.grpo_grad_accum)
    num_generations = max(2, cfg.grpo_num_generations)

    def adjust_grad_accum(start_value: int) -> int:
        ga = max(1, start_value)
        while (base_batch * ga) % num_generations != 0:
            ga += 1
        return ga

    new_grad_accum = adjust_grad_accum(grad_accum)
    if new_grad_accum != grad_accum:
        if use_decoupled_xla_explorer:
            print(
                "[grpo] Decoupled XLA Explorer path ignores TRL geometry divisibility: "
                f"keeping grad_accum={grad_accum} with batch_size={cfg.grpo_batch_size}, "
                f"num_generations={num_generations}, world_size={world_size}."
            )
        else:
            strict_geometry = (os.environ.get("TMX_STRICT_GRPO_GEOMETRY", "0") or "0").strip().lower() in (
                "1",
                "true",
                "yes",
            )
            if strict_geometry:
                raise ValueError(
                    "Requested GRPO geometry is incompatible with world size without changing "
                    f"gradient accumulation: batch_size={cfg.grpo_batch_size}, "
                    f"grad_accum={cfg.grpo_grad_accum}, num_generations={num_generations}, "
                    f"world_size={world_size}. Disable TMX_STRICT_GRPO_GEOMETRY or change the "
                    "requested geometry explicitly."
                )
            print(
                "[WARN] Adjusting grpo_grad_accum to satisfy divisibility: "
                f"{grad_accum} -> {new_grad_accum} (num_generations={num_generations})."
            )
            grad_accum = new_grad_accum

    global_batch = base_batch * grad_accum

    gen_batch = max(cfg.grpo_generation_batch_size, global_batch)
    if gen_batch % global_batch != 0:
        new_gen_batch = ((gen_batch + global_batch - 1) // global_batch) * global_batch
        print(
            "[WARN] generation_batch_size must be divisible by global batch size "
            f"({global_batch}). Adjusting {gen_batch} -> {new_gen_batch}."
        )
        gen_batch = new_gen_batch
    grad_ckpt_env = os.environ.get("TMX_GRPO_GRAD_CKPT")
    if grad_ckpt_env is not None:
        grad_ckpt_env = str(grad_ckpt_env).strip()
    if not grad_ckpt_env:
        use_grad_ckpt = not is_xla
    else:
        use_grad_ckpt = grad_ckpt_env != "0"
    fsdp_mode_env = os.environ.get("TMX_FSDP", "").strip().lower()
    use_fsdp_full_shard = fsdp_mode_env in ("1", "true", "full_shard")
    fsdp_grad_ckpt_requested = use_fsdp_full_shard and bool(grad_ckpt_env and grad_ckpt_env != "0")
    if is_xla and use_fsdp_full_shard and use_grad_ckpt:
        print(
            "[patch] disabling Trainer/model gradient_checkpointing under XLA FSDP; "
            "using fsdp_config.xla_fsdp_grad_ckpt instead."
        )
        use_grad_ckpt = False
    if is_xla and use_fsdp_full_shard and fsdp_grad_ckpt_requested:
        allow_unstable_fsdp_grad_ckpt = (
            os.environ.get("TMX_XLA_ALLOW_UNSTABLE_FSDP_GRAD_CKPT", "0").strip().lower()
            in ("1", "true", "yes")
        )
        if not allow_unstable_fsdp_grad_ckpt:
            print(
                "[patch] disabling xla_fsdp_grad_ckpt on single-worker XLA SPMD; "
                "checkpointed decoder forward is hitting tensor_data runtime failures."
            )
            fsdp_grad_ckpt_requested = False

    output_dir = cfg.explorer_output_dir
    if not output_dir:
        output_dir = os.path.join(
            _cfg_run_root(cfg),
            f"explorer_grpo_{cfg.grpo_loss_type}_lambda{cfg.lambda_novelty}",
        )

    exploration_max_completion_len = resolved_max_completion_len(cfg, phase="explore")
    use_bf16 = (torch.cuda.is_available() or is_xla) and os.environ.get("TMX_DISABLE_BF16", "0") != "1"
    if is_xla and not use_bf16:
        print("[patch] bf16 disabled for XLA GRPO via TMX_DISABLE_BF16=1.")

    def _build_training_args(gen_batch_value: int) -> GRPOConfig:
        config_signature = inspect.signature(GRPOConfig.__init__)
        supported_args = set(config_signature.parameters)
        mask_truncated_env = os.environ.get("TMX_MASK_TRUNCATED_COMPLETIONS")
        if mask_truncated_env is None or mask_truncated_env == "":
            mask_truncated_completions = False
        else:
            mask_truncated_completions = str(mask_truncated_env).strip().lower() not in ("0", "false", "no")
        logging_strategy = str(os.environ.get("TMX_GRPO_LOGGING_STRATEGY", "") or "").strip().lower() or "steps"
        logging_steps = max(1, int(os.environ.get("TMX_GRPO_LOGGING_STEPS", "1") or 1))
        save_strategy = str(os.environ.get("TMX_GRPO_SAVE_STRATEGY", "") or "").strip().lower() or "no"
        save_steps = max(1, int(os.environ.get("TMX_GRPO_SAVE_STEPS", "8") or 8))
        save_total_limit = max(1, int(os.environ.get("TMX_GRPO_SAVE_TOTAL_LIMIT", "2") or 2))
        if is_xla and save_strategy != "no" and os.environ.get("TMX_ALLOW_XLA_GRPO_INTERMEDIATE_SAVE", "0") not in ("1", "true", "True", "yes", "YES"):
            print("[patch] disabling intermediate GRPO checkpoint saves on XLA to avoid HBM spikes.")
            save_strategy = "no"
        print(f"[grpo] logging_strategy={logging_strategy} save_strategy={save_strategy}")
        beta_env = os.environ.get("TMX_GRPO_BETA", "").strip()
        beta_val = float(beta_env) if beta_env else None
        config_kwargs = {
            "output_dir": output_dir,
            "max_prompt_length": cfg.max_prompt_len,
            "max_completion_length": exploration_max_completion_len,
            "per_device_train_batch_size": cfg.grpo_batch_size,
            "gradient_accumulation_steps": grad_accum,
            "learning_rate": cfg.grpo_lr,
            "max_steps": cfg.grpo_max_steps,
            "bf16": use_bf16,
            "gradient_checkpointing": use_grad_ckpt,
            "report_to": "none",
            "logging_strategy": logging_strategy,
            "save_strategy": save_strategy,
            "loss_type": cfg.grpo_loss_type,
            "num_generations": num_generations,
            "temperature": train_temperature,
            "dataloader_num_workers": 0,
            "dataloader_pin_memory": False,
            "dataloader_drop_last": True,
        }
        if beta_val is not None and "beta" in supported_args:
            config_kwargs["beta"] = beta_val
            print(f"[grpo] KL beta={beta_val} (TMX_GRPO_BETA)")
        optim_override = os.environ.get("TMX_OPTIM", "").strip() or os.environ.get("TMX_OPTIMIZER_OVERRIDE", "").strip()
        if optim_override and "optim" in supported_args:
            config_kwargs["optim"] = optim_override
            print(f"[grpo] Using optimizer override: {optim_override}")
        if logging_strategy != "no":
            config_kwargs["logging_steps"] = logging_steps
        if save_strategy == "steps":
            if "save_steps" in supported_args:
                config_kwargs["save_steps"] = save_steps
            if "save_total_limit" in supported_args:
                config_kwargs["save_total_limit"] = save_total_limit
        if use_grad_ckpt and is_xla and "gradient_checkpointing_kwargs" in supported_args:
            config_kwargs["gradient_checkpointing_kwargs"] = {
                "use_reentrant": False,
                "preserve_rng_state": False,
            }
        if "mask_truncated_completions" in supported_args:
            config_kwargs["mask_truncated_completions"] = mask_truncated_completions
        if "generation_batch_size" in supported_args:
            config_kwargs["generation_batch_size"] = gen_batch_value
        if "scale_rewards" in supported_args:
            scale_rewards_param = config_signature.parameters["scale_rewards"]
            if scale_rewards_param.annotation is bool or isinstance(scale_rewards_param.default, bool):
                config_kwargs["scale_rewards"] = True
            else:
                config_kwargs["scale_rewards"] = "batch"
        # FSDP v2 support for multi-chip TPU via SPMD
        if use_fsdp_full_shard:
            use_fsdp_v2 = is_xla  # FSDP v2 (SPMD) on XLA, v1 elsewhere
            fsdp_config = {
                "xla": is_xla,
                "xla_fsdp_v2": use_fsdp_v2,
                "xla_fsdp_grad_ckpt": fsdp_grad_ckpt_requested,
            }
            if not use_fsdp_v2:
                # FSDP v1 settings (non-XLA)
                fsdp_config.update({
                    "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                    "fsdp_sharding_strategy": "FULL_SHARD",
                    "fsdp_state_dict_type": "SHARDED_STATE_DICT",
                    "fsdp_backward_prefetch_policy": "BACKWARD_PRE",
                    "fsdp_sync_module_states": True,
                    "fsdp_use_orig_params": True,
                })
            # Explicit transformer layer class for wrapping
            wrap_cls_override = (os.environ.get("TMX_FSDP_WRAP_CLS") or "").strip()
            if wrap_cls_override:
                wrap_classes = [item.strip() for item in wrap_cls_override.split(",") if item.strip()]
            else:
                wrap_classes = ["Qwen3DecoderLayer"]
            fsdp_config["fsdp_transformer_layer_cls_to_wrap"] = wrap_classes
            config_kwargs["fsdp"] = "full_shard"
            config_kwargs["fsdp_config"] = fsdp_config
            v2_str = "v2/SPMD" if use_fsdp_v2 else "v1"
            print(f"[grpo] FSDP {v2_str} enabled: full_shard wrap={wrap_classes} grad_ckpt={fsdp_config['xla_fsdp_grad_ckpt']}")
        return GRPOConfig(
            **{key: value for key, value in config_kwargs.items() if key in supported_args}
        )

    try:
        training_args = _build_training_args(gen_batch)
    except ValueError as e:
        msg = str(e)
        match = re.search(
            r"generation_batch_size\s*\((\d+)\)\s*must be divisible by the global batch size\s*\((\d+)\)",
            msg,
        )
        if match:
            reported_global = int(match.group(2))
            new_gen_batch = max(gen_batch, reported_global)
            if new_gen_batch % reported_global != 0:
                new_gen_batch = ((new_gen_batch + reported_global - 1) // reported_global) * reported_global
            print(
                "[WARN] Adjusting generation_batch_size to satisfy TRL global batch: "
                f"{gen_batch} -> {new_gen_batch} (global={reported_global})."
            )
            gen_batch = new_gen_batch
            training_args = _build_training_args(gen_batch)
        else:
            raise
    if hasattr(training_args, "mask_truncated_completions"):
        print(
            "[grpo] mask_truncated_completions="
            f"{bool(getattr(training_args, 'mask_truncated_completions', False))}"
        )
    if is_xla:
        xla_train_top_p = float(
            os.environ.get(
                "TMX_EXPLORER_TRAIN_TOP_P",
                os.environ.get("TMX_EXPLORER_VLLM_TOP_P", "0.95"),
            )
            or 0.95
        )
        xla_train_top_k = int(
            os.environ.get(
                "TMX_EXPLORER_TRAIN_TOP_K",
                os.environ.get("TMX_EXPLORER_VLLM_TOP_K", "20"),
            )
            or 20
        )
        # Avoid XLA-heavy suppress/bad-words processors. Keep pad_token_id explicit.
        training_args.generation_kwargs = {
            "pad_token_id": tokenizer.pad_token_id,
            "suppress_tokens": None,
            "begin_suppress_tokens": None,
            "bad_words_ids": None,
            "stop_strings": None,
            "top_k": xla_train_top_k,
            "top_p": xla_train_top_p,
            "do_sample": True,
        }
        if safe_eos_id is not None:
            training_args.generation_kwargs["eos_token_id"] = safe_eos_id
        if safe_bos_id is not None:
            training_args.generation_kwargs["bos_token_id"] = safe_bos_id
        print(
            f"[grpo] XLA training sampling top_p={xla_train_top_p} top_k={xla_train_top_k}"
        )

    print(f"Instantiating GRPOTrainer ({run_label}) with loss={cfg.grpo_loss_type}...")

    explorer_model = causal_lm_from_pretrained(cfg.model_name, is_xla=is_xla)
    if use_grad_ckpt:
        try:
            enabled = _enable_model_grad_checkpointing(
                explorer_model,
                preserve_rng_state=False,
                verbose=True,
                label=run_label,
            )
            if enabled and is_xla:
                print(f"[patch] XLA-safe gradient checkpointing enabled for {run_label} run.")
        except Exception as e:
            print(f"[WARN] Could not enable gradient checkpointing explicitly: {e}")
    if not use_grad_ckpt and hasattr(explorer_model, "gradient_checkpointing_disable"):
        try:
            explorer_model.gradient_checkpointing_disable()
            if getattr(explorer_model, "config", None) is not None:
                setattr(explorer_model.config, "gradient_checkpointing", False)
            print(f"[patch] Gradient checkpointing disabled for {run_label} run.")
        except Exception as e:
            print(f"[WARN] Could not disable gradient checkpointing explicitly: {e}")
    if init_model is not None:
        print(f"[{run_label.lower().replace(' ', '-')}] Initializing from provided model state.")
        explorer_model.load_state_dict(init_model.state_dict())
        # Free init_model from TPU to reclaim HBM — weights are already copied.
        init_model.to(torch.device("cpu"))
        del init_model
        gc.collect()
        if is_xla:
            _maybe_xla_mark_step("xla")
        print("[cleanup] Freed init_model from device after weight copy.")
    if tokenizer.pad_token_id is not None:
        explorer_model.config.pad_token_id = tokenizer.pad_token_id
        if getattr(explorer_model, "generation_config", None) is not None:
            explorer_model.generation_config.pad_token_id = tokenizer.pad_token_id
    if safe_eos_id is not None:
        explorer_model.config.eos_token_id = safe_eos_id
        if getattr(explorer_model, "generation_config", None) is not None:
            explorer_model.generation_config.eos_token_id = safe_eos_id
    if answer_constraints is not None:
        explorer_model._tmx_answer_constraints = answer_constraints
    # Defensive: ensure model configs won't reintroduce suppress tokens on TPU.
    if is_xla:
        use_incremental_xla = os.environ.get(
            "TMX_XLA_INCREMENTAL_GENERATE", "0"
        ) not in ("", "0", "false", "False", "no", "NO")
        for cfg_obj in (getattr(explorer_model, "config", None), getattr(explorer_model, "generation_config", None)):
            if cfg_obj is not None:
                setattr(cfg_obj, "suppress_tokens", None)
                setattr(cfg_obj, "begin_suppress_tokens", None)
                setattr(cfg_obj, "bad_words_ids", None)
                setattr(cfg_obj, "use_cache", bool(use_incremental_xla))
        use_safe_generate = os.environ.get("TMX_XLA_SAFE_GENERATE", "1") != "0"
        if use_safe_generate:
            patch_model_generate_for_xla(explorer_model)
        else:
            if explorer_model.config is not None:
                explorer_model.config.use_cache = bool(use_incremental_xla)
            if getattr(explorer_model, "generation_config", None) is not None:
                explorer_model.generation_config.use_cache = bool(use_incremental_xla)
    novelty_scorer = None
    if _should_build_novelty_scorer(cfg):
        novelty_scorer, _ = build_novelty_scorer(explorer_model, tokenizer, cfg, device, is_xla)
    else:
        print(f"[novelty] Disabled for {run_label} GRPO (lambda_novelty=0 and incorrect_novelty_scale=0).")
    anchor_scorer = None
    if anchor_model is not None and cfg.anchor_metric != "none" and cfg.anchor_weight > 0:
        anchor_model = anchor_model.to(device)
        anchor_scorer = AnchorPenaltyScorer(
            current_model=explorer_model,
            reference_model=anchor_model,
            tokenizer=tokenizer,
            metric=cfg.anchor_metric,
            device=device,
            max_length=cfg.max_total_len,
            trainable_only=cfg.anchor_trainable_only,
        )
    _use_decoupled = (
        is_xla
        or _env_flag("TMX_DECOUPLED_TRAIN_CPU", default=False)
        or _env_flag("TMX_FORCE_DECOUPLED", default=False)
    )
    if _use_decoupled and _env_flag("TMX_EXPLORER_DECOUPLED_GRPO", default=True):
        if (
            _env_flag("TMX_EXPLORER_DECOUPLED_TRAIN_FSDP", default=_env_flag("TMX_FSDP", default=False))
            and _xla_runtime_rank_world()[1] > 1
        ):
            explorer_model, novelty_scorer, explorer_log_hist = train_explorer_with_decoupled_grpo_fsdp(
                train_ds,
                cfg=cfg,
                device=device,
                is_xla=is_xla,
                use_wandb=use_wandb,
                wandb_run=wandb_run,
                tokenizer=tokenizer,
                explorer_model=explorer_model,
                anchor_model=anchor_model,
                grpo_metric_prefix=grpo_metric_prefix,
                train_metric_prefix=train_metric_prefix,
                run_label=run_label,
                output_dir=output_dir,
                train_temperature=train_temperature,
                answer_constraints=answer_constraints,
            )
            return explorer_model, tokenizer, novelty_scorer, explorer_log_hist
        explorer_model, novelty_scorer, explorer_log_hist = train_explorer_with_decoupled_grpo(
            train_ds,
            cfg=cfg,
            device=device,
            is_xla=is_xla,
            use_wandb=use_wandb,
            wandb_run=wandb_run,
            tokenizer=tokenizer,
            explorer_model=explorer_model,
            anchor_model=anchor_model,
            grpo_metric_prefix=grpo_metric_prefix,
            train_metric_prefix=train_metric_prefix,
            run_label=run_label,
            output_dir=output_dir,
            train_temperature=train_temperature,
            answer_constraints=answer_constraints,
        )
        return explorer_model, tokenizer, novelty_scorer, explorer_log_hist
    reward_fn = GRPOReward(
        novelty_scorer,
        cfg.lambda_novelty,
        reward_correct=cfg.reward_correct,
        reward_incorrect=cfg.reward_incorrect,
        reward_shape=cfg.reward_shape,
        incorrect_novelty_scale=cfg.incorrect_novelty_scale,
        piecewise_bounds=cfg.reward_piecewise_bounds,
        novelty_zscore=cfg.novelty_zscore,
        correctness_zscore=cfg.correctness_zscore,
        anchor_penalty_scorer=anchor_scorer,
        anchor_weight=cfg.anchor_weight,
        format_penalty_weight=cfg.format_penalty_weight,
        gate_novelty_by_quality=cfg.gate_novelty_by_quality,
        answer_prefix=generation_answer_prefix(cfg, phase="explore"),
        novelty_text_contract=cfg.novelty_text_contract,
        novelty_phase="explorer_train",
    )
    reward_fn.__name__ = "GRPOReward"
    trainer_ds = train_ds.map(
        lambda example: {"prompt": render_generation_prompt(tokenizer, example["prompt"], cfg, phase="explore")},
        load_from_cache_file=False,
    )
    trainer = GRPOTrainer(
        model=explorer_model,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        args=training_args,
        train_dataset=trainer_ds,
    )
    vllm_active = bool((os.environ.get("TMX_VLLM_SERVER_URLS") or "").strip())
    if vllm_active or (is_xla and os.environ.get("TMX_XLA_SAFE_GENERATE", "1") != "0"):
        patch_grpo_trainer_generation_for_xla(trainer, answer_constraints=answer_constraints)

    try:
        from transformers import TrainerCallback
    except Exception:
        TrainerCallback = None

    if TrainerCallback is not None and anchor_model is not None and cfg.anchor_metric == "param_l2" and cfg.anchor_weight > 0:
        class ParamAnchorCallback(TrainerCallback):
            def on_step_end(self, args, state, control, model=None, optimizer=None, **kwargs):
                if model is None or anchor_model is None:
                    return
                lr = cfg.grpo_lr
                if optimizer is not None and getattr(optimizer, "param_groups", None):
                    try:
                        lr = float(optimizer.param_groups[0].get("lr", lr))
                    except Exception:
                        lr = cfg.grpo_lr
                coeff = max(0.0, min(1.0, 2.0 * float(cfg.anchor_weight) * float(lr)))
                sq_sum = 0.0
                count = 0
                with torch.no_grad():
                    for param, ref_param in zip(model.parameters(), anchor_model.parameters()):
                        if cfg.anchor_trainable_only and not param.requires_grad:
                            continue
                        diff = (param.data - ref_param.data).float()
                        sq_sum += float(diff.pow(2).sum().item())
                        count += int(diff.numel())
                        if coeff > 0:
                            param.data.add_(ref_param.data - param.data, alpha=coeff)
                if count > 0 and use_wandb and wandb_run is not None and wandb is not None:
                    wandb_run.log(
                        {
                            "anchor/param_l2": (sq_sum / count) ** 0.5,
                            "anchor/step": state.global_step,
                        }
                    )

        trainer.add_callback(ParamAnchorCallback())

    logged_live = False
    train_start = time.perf_counter()
    if use_wandb and wandb_run is not None and wandb is not None:
        if TrainerCallback is not None:
            class WandbLogCallback(TrainerCallback):
                def on_log(self, args, state, control, logs=None, **kwargs):
                    if not logs:
                        return
                    step = logs.get("step", state.global_step)
                    data = {f"{grpo_metric_prefix}/{k}": v for k, v in logs.items()}
                    reward_summary = getattr(reward_fn, "last_batch_summary", {}) or {}
                    for key, value in reward_summary.items():
                        data[f"{grpo_metric_prefix}/{key}"] = value
                    elapsed = time.perf_counter() - train_start
                    if "num_tokens" in logs and "step_time" in logs and logs["step_time"]:
                        data[f"{grpo_metric_prefix}/tokens_per_sec"] = float(logs["num_tokens"]) / float(logs["step_time"])
                    data[f"{grpo_metric_prefix}/step"] = step
                    data[f"{grpo_metric_prefix}/wall_time_seconds"] = elapsed
                    if train_metric_prefix:
                        data[f"{train_metric_prefix}/step"] = step
                        data[f"{train_metric_prefix}/wall_time_seconds"] = elapsed
                        for key in ("loss", "grad_norm", "entropy", "learning_rate", "num_tokens", "reward", "step_time"):
                            if key in logs:
                                data[f"{train_metric_prefix}/{key}"] = logs[key]
                        if "num_tokens" in logs and "step_time" in logs and logs["step_time"]:
                            data[f"{train_metric_prefix}/tokens_per_sec"] = float(logs["num_tokens"]) / float(logs["step_time"])
                    wandb_run.log(data)

            trainer.add_callback(WandbLogCallback())
            logged_live = True

    print(f"Starting GRPO training ({run_label})...")
    resume_checkpoint = None
    if os.environ.get("TMX_RESUME_TRAINING", "0") not in ("", "0", "false", "False", "no", "NO"):
        resume_checkpoint = _latest_hf_checkpoint(
            output_dir,
            max_step_exclusive=int(cfg.grpo_max_steps or 0) if int(cfg.grpo_max_steps or 0) > 0 else None,
        )
        if resume_checkpoint:
            print(f"[resume] Resuming Explorer GRPO from {resume_checkpoint}")
    ignored_post_step_xla_oom = False
    try:
        if resume_checkpoint:
            trainer.train(resume_from_checkpoint=resume_checkpoint)
        else:
            trainer.train()
    except RuntimeError as exc:
        exc_text = str(exc)
        completed_steps = int(getattr(trainer.state, "global_step", 0) or 0)
        target_steps = int(getattr(trainer.args, "max_steps", 0) or 0)
        max_logged_step = 0
        for row in getattr(trainer.state, "log_history", []) or []:
            try:
                max_logged_step = max(max_logged_step, int(row.get("step", 0) or 0))
            except Exception:
                continue
        effective_steps = max(completed_steps, max_logged_step)
        if "RESOURCE_EXHAUSTED" not in exc_text or effective_steps < max(1, target_steps):
            raise
        print(
            "[patch] Ignoring post-step XLA OOM after Explorer GRPO reached the scheduled "
            f"step budget ({effective_steps}/{target_steps})."
        )
        ignored_post_step_xla_oom = True

    log_hist = trainer.state.log_history
    elapsed = 0.0
    for row in log_hist:
        if "step_time" in row:
            elapsed += float(row["step_time"])
        else:
            elapsed = max(elapsed, time.perf_counter() - train_start)
        row.setdefault("wall_time_seconds", elapsed)
    if use_wandb and wandb_run is not None and wandb is not None and not logged_live:
        for i, h in enumerate(log_hist):
            step = h.get("step", i)
            data = {f"{grpo_metric_prefix}/{k}": v for k, v in h.items()}
            data[f"{grpo_metric_prefix}/step"] = step
            data[f"{grpo_metric_prefix}/wall_time_seconds"] = h.get("wall_time_seconds", 0.0)
            if train_metric_prefix:
                data[f"{train_metric_prefix}/wall_time_seconds"] = h.get("wall_time_seconds", 0.0)
            wandb_run.log(data)

    if _env_flag("TMX_ABORT_ON_DEGENERATE_EXPLORER", default=True):
        terminal_row = {}
        for row in reversed(log_hist):
            if any(
                key in row
                for key in (
                    "train_loss",
                    "loss",
                    "completions/mean_length",
                    "completions/clipped_ratio",
                    "reward",
                    "rewards/GRPOReward/mean",
                )
            ):
                terminal_row = row
                break
        train_loss = terminal_row.get("train_loss", terminal_row.get("loss"))
        mean_length = terminal_row.get("completions/mean_length")
        clipped_ratio = terminal_row.get("completions/clipped_ratio")
        reward_mean = terminal_row.get("rewards/GRPOReward/mean", terminal_row.get("reward"))
        train_loss_bad = train_loss is not None and (not isinstance(train_loss, (int, float)) or not math.isfinite(float(train_loss)))
        mean_length_bad = mean_length is not None and float(mean_length) <= 0.0
        clipped_bad = clipped_ratio is not None and float(clipped_ratio) >= 0.999
        reward_bad = reward_mean is not None and float(reward_mean) <= float(cfg.reward_incorrect)
        if train_loss_bad and mean_length_bad and clipped_bad and reward_bad:
            if use_wandb and wandb_run is not None and wandb is not None:
                wandb_run.log(
                    {
                        f"{grpo_metric_prefix}/health_gate_triggered": 1.0,
                        f"{grpo_metric_prefix}/health_gate_reason/degenerate_terminal": 1.0,
                        f"{grpo_metric_prefix}/health_gate_terminal_step": float(terminal_row.get("step", getattr(trainer.state, "global_step", 0) or 0)),
                        f"{grpo_metric_prefix}/health_gate_terminal_loss": float("nan") if train_loss is None else float(train_loss),
                        f"{grpo_metric_prefix}/health_gate_terminal_mean_length": 0.0 if mean_length is None else float(mean_length),
                        f"{grpo_metric_prefix}/health_gate_terminal_clipped_ratio": 0.0 if clipped_ratio is None else float(clipped_ratio),
                        f"{grpo_metric_prefix}/health_gate_terminal_reward": 0.0 if reward_mean is None else float(reward_mean),
                    }
                )
            raise RuntimeError(
                "Explorer GRPO health gate triggered: terminal metrics show non-finite loss, "
                "zero-length completions, fully clipped generations, and reward at the incorrect baseline."
            )

    explorer_model = trainer.model
    trainer.optimizer = None
    trainer.lr_scheduler = None
    trainer.train_dataset = None
    trainer.eval_dataset = None
    gc.collect()
    if is_xla and ignored_post_step_xla_oom:
        try:
            print("[patch] Rehydrating Explorer model through CPU after post-step XLA OOM.")
            explorer_model = explorer_model.to(torch.device("cpu"))
            gc.collect()
        except Exception as e:
            print(f"[WARN] Could not offload Explorer model to CPU after XLA OOM: {e}")
    rebuilt_model, rebuilt_novelty_scorer = _rebuild_explorer_model_for_generation(
        explorer_model,
        tokenizer,
        cfg,
        device=device,
        is_xla=is_xla,
        safe_eos_id=safe_eos_id,
        answer_constraints=answer_constraints,
    )
    explorer_model = rebuilt_model
    if rebuilt_novelty_scorer is not None:
        novelty_scorer = rebuilt_novelty_scorer
    explorer_model = explorer_model.to(device)
    return explorer_model, tokenizer, novelty_scorer, log_hist


def generate_offline_trajectories(explorer_model,
                                  tokenizer,
                                  train_ds,
                                  cfg,
                                  novelty_scorer: NoveltyScorer,
                                  use_wandb: bool,
                                  wandb_run=None,
                                  temperature_value: Optional[float] = None,
                                  novelty_train: bool = False,
                                  explorer_step: Optional[int] = None):
    """
    Use trained Explorer to generate offline trajectories:
      (prompt, completion, ground_truth, r_correct, is_correct, r_novel, reward)

    RND is treated as frozen here (train=False).
    """
    explorer_model.eval()
    device = get_device(explorer_model)
    safe_eos_id = _normalize_token_id(tokenizer.eos_token_id)

    trajectories = []
    if temperature_value is not None:
        temps = [temperature_value]
    else:
        temps = list(cfg.temperature_schedule) if cfg.temperature_schedule else [1.0]
    total_per_prompt = cfg.num_samples_per_prompt * len(temps)
    print(
        "Collecting trajectories with temps="
        f"{temps} and {total_per_prompt} samples/prompt."
    )

    reward_fn = GRPOReward(
        novelty_scorer,
        cfg.lambda_novelty,
        reward_correct=cfg.reward_correct,
        reward_incorrect=cfg.reward_incorrect,
        reward_shape=cfg.reward_shape,
        incorrect_novelty_scale=cfg.incorrect_novelty_scale,
        piecewise_bounds=cfg.reward_piecewise_bounds,
        novelty_zscore=cfg.novelty_zscore,
        correctness_zscore=cfg.correctness_zscore,
        format_penalty_weight=cfg.format_penalty_weight,
        gate_novelty_by_quality=cfg.gate_novelty_by_quality,
        answer_prefix=generation_answer_prefix(cfg, phase="explore"),
        novelty_text_contract=cfg.novelty_text_contract,
        novelty_phase="trajectory",
    )

    trajectory_batch_size = max(1, int(getattr(cfg, "trajectory_batch_size", getattr(cfg, "eval_batch_size", 1)) or 1))
    max_completion_len = resolved_max_completion_len(cfg, phase="explore")
    # Allow overriding trajectory collection token budget independently of Explorer
    # GRPO generation length. Useful when GRPO runs on CPU (short) but trajectories
    # are collected via vLLM (long thinking traces).
    traj_max_override = (os.environ.get("TMX_TRAJ_MAX_COMPLETION_LEN") or "").strip()
    if traj_max_override:
        max_completion_len = int(traj_max_override)
    health_window_prompts = max(0, int(os.environ.get("TMX_TRAJ_HEALTH_WINDOW_PROMPTS", "32") or 32))
    health_min_correct = max(0, int(os.environ.get("TMX_TRAJ_HEALTH_MIN_CORRECT", "1") or 1))
    accepted_target_raw = str(os.environ.get("TMX_TRAJ_TARGET_ACCEPTED", "") or "").strip()
    accepted_target = max(0, int(accepted_target_raw)) if accepted_target_raw else 0
    health_prompt_count = 0
    health_correct_count = 0
    early_stop_reached = False

    for temp in temps:
        for start in tqdm(range(0, len(train_ds), trajectory_batch_size), desc=f"Collecting Explorer trajectories (temp={temp})"):
            batch_t0 = time.perf_counter()
            stop = min(len(train_ds), start + trajectory_batch_size)
            batch = train_ds.select(range(start, stop))
            prompts = list(batch["prompt"])
            model_prompts = [render_generation_prompt(tokenizer, prompt, cfg, phase="explore") for prompt in prompts]
            ground_truths = list(batch["ground_truth"])
            topics = list(batch["topic"]) if "topic" in batch.column_names else ["unknown"] * len(prompts)
            source_row_ids = (
                [int(v) for v in batch["__row_id__"]]
                if "__row_id__" in batch.column_names
                else list(range(start, stop))
            )
            prompt_length_ids = tokenizer(
                model_prompts,
                add_special_tokens=True,
                padding=False,
                truncation=False,
            )["input_ids"]
            prompt_token_lengths = [len(ids) for ids in prompt_length_ids]

            enc = tokenizer(
                model_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=cfg.max_prompt_len,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            prompt_width = int(input_ids.shape[1])

            sampled_completions = [[] for _ in prompts]
            sampled_completion_stats = [[] for _ in prompts]
            answer_constraints = get_answer_token_constraints(tokenizer, cfg, phase="explore")

            # vLLM-accelerated generation path
            # NOTE: Only use vLLM for trajectory collection if the vLLM server
            # has been reloaded with the Explorer model's weights (TMX_TRAJ_USE_VLLM=1).
            # By default, vLLM serves BASE weights which are NOT the trained Explorer.
            traj_use_vllm = os.environ.get("TMX_TRAJ_USE_VLLM", "0").strip() == "1"
            vllm_urls_raw = (os.environ.get("TMX_VLLM_SERVER_URLS") or "").strip() if traj_use_vllm else ""
            if vllm_urls_raw:
                from tmx.vllm_generate import generate_completions_vllm
                vllm_urls = [u.strip() for u in vllm_urls_raw.split(",") if u.strip()]
                vllm_model = os.environ.get("TMX_VLLM_MODEL_NAME", "")
                vllm_api_key = os.environ.get("TMX_VLLM_API_KEY", "")
                vllm_concurrency = int(os.environ.get("TMX_VLLM_CONCURRENCY", "64"))
                vllm_top_p = float(os.environ.get("TMX_TRAJ_VLLM_TOP_P", os.environ.get("TMX_EXPLORER_VLLM_TOP_P", "0.95")) or 0.95)
                vllm_top_k = int(os.environ.get("TMX_TRAJ_VLLM_TOP_K", os.environ.get("TMX_EXPLORER_VLLM_TOP_K", "20")) or 20)
                thinking_mode = _resolved_thinking_mode(cfg, phase="explore")
                vllm_enable_thinking = True if thinking_mode == "think" else (False if thinking_mode == "no_think" else None)
                all_completions = generate_completions_vllm(
                    prompts=model_prompts,
                    server_urls=vllm_urls,
                    model_name=vllm_model,
                    n_per_prompt=cfg.num_samples_per_prompt,
                    temperature=temp,
                    top_p=vllm_top_p,
                    top_k=vllm_top_k,
                    max_tokens=max_completion_len,
                    concurrency=vllm_concurrency,
                    api_key=vllm_api_key,
                    enable_thinking=vllm_enable_thinking,
                )
                for row_idx, completion_list in enumerate(all_completions):
                    for completion_dict in completion_list:
                        completion_text = completion_dict["text"]
                        finish_reason = completion_dict["finish_reason"]
                        hit_eos = finish_reason == "stop"
                        hit_max = finish_reason == "length"
                        sampled_completions[row_idx].append(completion_text)
                        sampled_completion_stats[row_idx].append({
                            "gen_tokens": len(tokenizer.encode(completion_text, add_special_tokens=False)),
                            "hit_eos": hit_eos,
                            "hit_max": hit_max,
                            "pad_tokens": 0,
                        })

            with torch.no_grad():
                if vllm_urls_raw:
                    pass  # already generated via vLLM above
                elif device.type == "xla" or cfg.num_samples_per_prompt <= 1 or answer_constraints is not None:
                    for _ in range(cfg.num_samples_per_prompt):
                        if device.type == "xla" or answer_constraints is not None:
                            out_ids = xla_safe_generate(
                                explorer_model,
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                max_new_tokens=max_completion_len,
                                do_sample=True,
                                temperature=temp,
                                top_p=0.9,
                                eos_token_id=safe_eos_id if safe_eos_id is not None else tokenizer.eos_token_id,
                                pad_token_id=tokenizer.pad_token_id,
                                allowed_token_ids=(answer_constraints or {}).get("allowed_token_ids"),
                                initial_token_ids=(answer_constraints or {}).get("initial_token_ids"),
                            )
                        else:
                            out_ids = explorer_model.generate(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                max_new_tokens=max_completion_len,
                                do_sample=True,
                                temperature=temp,
                                top_p=0.9,
                                eos_token_id=safe_eos_id if safe_eos_id is not None else tokenizer.eos_token_id,
                                pad_token_id=tokenizer.pad_token_id,
                                suppress_tokens=None,
                                begin_suppress_tokens=None,
                                bad_words_ids=None,
                            )

                        decoded_completions = [
                            tokenizer.decode(out_ids[i][prompt_width:], skip_special_tokens=True)
                            for i in range(out_ids.shape[0])
                        ]
                        for row_idx, completion in enumerate(decoded_completions):
                            gen_stats = _sequence_generation_stats(
                                out_ids[row_idx],
                                prompt_width=prompt_width,
                                eos_id=safe_eos_id if safe_eos_id is not None else tokenizer.eos_token_id,
                                pad_token_id=tokenizer.pad_token_id,
                                max_new_tokens=max_completion_len,
                            )
                            sampled_completions[row_idx].append(completion)
                            sampled_completion_stats[row_idx].append(gen_stats)
                else:
                    out_ids = explorer_model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_completion_len,
                        do_sample=True,
                        temperature=temp,
                        top_p=0.9,
                        num_return_sequences=cfg.num_samples_per_prompt,
                        eos_token_id=safe_eos_id if safe_eos_id is not None else tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        suppress_tokens=None,
                        begin_suppress_tokens=None,
                        bad_words_ids=None,
                    )
                    decoded_completions = [
                        tokenizer.decode(out_ids[i][prompt_width:], skip_special_tokens=True)
                        for i in range(out_ids.shape[0])
                    ]
                    for row_idx in range(len(prompts)):
                        start_idx = row_idx * cfg.num_samples_per_prompt
                        stop_idx = start_idx + cfg.num_samples_per_prompt
                        sampled_completions[row_idx].extend(decoded_completions[start_idx:stop_idx])
                        sampled_completion_stats[row_idx].extend(
                            [
                                _sequence_generation_stats(
                                    out_ids[seq_idx],
                                    prompt_width=prompt_width,
                                    eos_id=safe_eos_id if safe_eos_id is not None else tokenizer.eos_token_id,
                                    pad_token_id=tokenizer.pad_token_id,
                                    max_new_tokens=max_completion_len,
                                )
                                for seq_idx in range(start_idx, stop_idx)
                            ]
                        )

            flat_completions = [completion for group in sampled_completions for completion in group]
            flat_completion_stats = [stats for group in sampled_completion_stats for stats in group]
            flat_ground_truths = [gt for gt, group in zip(ground_truths, sampled_completions) for _ in group]
            flat_prompts = [prompt for prompt, group in zip(prompts, sampled_completions) for _ in group]
            flat_model_prompts = [prompt for prompt, group in zip(model_prompts, sampled_completions) for _ in group]
            flat_topics = [topic for topic, group in zip(topics, sampled_completions) for _ in group]
            flat_source_row_ids = [row_id for row_id, group in zip(source_row_ids, sampled_completions) for _ in group]
            flat_prompt_token_lengths = [length for length, group in zip(prompt_token_lengths, sampled_completions) for _ in group]
            flat_sample_indices = [
                sample_idx
                for group in sampled_completions
                for sample_idx in range(len(group))
            ]

            rewards, details = reward_fn.compute_batch(
                flat_completions,
                flat_ground_truths,
                prompts=flat_model_prompts,
                completion_stats=flat_completion_stats,
                train=novelty_train,
                return_details=True,
            )
            batch_seconds = time.perf_counter() - batch_t0
            batch_correct = sum(1 for detail in details if bool(detail.get("is_correct", False)))
            print(
                "[traj] "
                f"temp={temp} prompts={len(prompts)} samples={len(flat_completions)} "
                f"correct={batch_correct} batch_seconds={batch_seconds:.1f}"
            )
            health_prompt_count += int(len(prompts))
            health_correct_count += int(batch_correct)
            if (
                _env_flag("TMX_ABORT_ON_BAD_TRAJ_WINDOW", default=True)
                and health_window_prompts > 0
                and health_prompt_count >= health_window_prompts
                and health_correct_count < health_min_correct
            ):
                raise RuntimeError(
                    "Trajectory collection health gate triggered: "
                    f"correct={health_correct_count} over the first {health_prompt_count} prompts."
                )

            for prompt, gt, topic, source_row_id, prompt_token_length, sample_idx, completion, completion_stats, total_r, detail in zip(
                flat_prompts,
                flat_ground_truths,
                flat_topics,
                flat_source_row_ids,
                flat_prompt_token_lengths,
                flat_sample_indices,
                flat_completions,
                flat_completion_stats,
                rewards,
                details,
            ):
                normalized = normalize_generated_completion(
                    completion,
                    answer_prefix=generation_answer_prefix(cfg, phase="explore"),
                )
                scored_completion = str(normalized.get("completion_scored", completion))
                raw_completion = str(normalized.get("completion_raw", completion))
                raw_full_completion = str(normalized.get("completion_full_raw", completion))
                completion_meta = analyze_completion_text(scored_completion)
                quality_metrics = completion_quality_metrics(completion_meta, extra_flags=completion_stats)
                raw_completion_meta = analyze_completion_text(raw_full_completion)
                raw_quality_metrics = completion_quality_metrics(raw_completion_meta, extra_flags=completion_stats)
                presented_answer = extract_presented_answer(scored_completion, meta=completion_meta)
                canonical_completion = canonicalize_completion_for_training(scored_completion, meta=completion_meta)
                actual_target_canonical = build_actual_training_text(
                    normalized,
                    fallback_completion=completion,
                    target_contract="canonical",
                )
                actual_target_full_trace = build_actual_training_text(
                    normalized,
                    fallback_completion=completion,
                    target_contract="full_trace",
                )
                normalization_changed = raw_full_completion.strip() != scored_completion.strip()
                trajectories.append(
                    {
                        "prompt": prompt,
                        "source_problem_id": int(source_row_id),
                        "source_row_id": int(source_row_id),
                        "explorer_step": int(explorer_step) if explorer_step is not None else None,
                        "sample_idx_within_prompt": int(sample_idx),
                        "completion": scored_completion,
                        "completion_raw": raw_completion,
                        "completion_full_raw": raw_full_completion,
                        "completion_first_line": str(normalized.get("completion_first_line", "")),
                        "completion_scored": scored_completion,
                        "presented_answer": presented_answer,
                        "completion_canonical": canonical_completion,
                        "actual_target_canonical": actual_target_canonical,
                        "actual_target_full_trace": actual_target_full_trace,
                        "ground_truth": gt,
                        "temperature": temp,
                        "topic": topic,
                        "r_correct": float(detail.get("r_correct_raw", 0.0)),
                        "r_correct_used": float(detail.get("r_correct_used", 0.0)),
                        "is_correct": bool(detail.get("is_correct", False)),
                        "r_novel": float(detail.get("r_novel_raw", 0.0)),
                        "r_novel_used": float(detail.get("r_novel_used", 0.0)),
                        "r_novel_reward": float(detail.get("r_novel_reward", detail.get("r_novel_used", 0.0))),
                        "novelty_gate": float(detail.get("novelty_gate", 0.0)),
                        "reward_bucket": str(detail.get("reward_bucket", "unknown")),
                        "reward_shape": str(detail.get("reward_shape", cfg.reward_shape)),
                        "r_novel_layers": dict(detail.get("r_novel_layers", {})),
                        "r_anchor": float(detail.get("r_anchor_raw", 0.0)),
                        "anchor_metric": str(detail.get("anchor_metric", "none")),
                        "reward_format_quality": float(detail.get("format_quality", 0.0)),
                        "reward_format_penalty": float(detail.get("format_penalty", 0.0)),
                        "reward_format_penalty_weight": float(detail.get("format_penalty_weight", cfg.format_penalty_weight)),
                        "reward_format_penalty_breakdown": dict(detail.get("format_penalty_breakdown", {})),
                        "reward": float(total_r),
                        "novelty_text": str(detail.get("novelty_text", scored_completion)),
                        "novelty_text_contract": str(detail.get("novelty_text_contract", cfg.novelty_text_contract)),
                        "format_quality": float(quality_metrics.get("quality", 0.0)),
                        "format_penalty": float(quality_metrics.get("penalty", 0.0)),
                        "format_penalty_breakdown": dict(quality_metrics.get("penalties", {})),
                        "raw_format_quality": float(raw_quality_metrics.get("quality", 0.0)),
                        "raw_format_penalty": float(raw_quality_metrics.get("penalty", 0.0)),
                        "raw_format_penalty_breakdown": dict(raw_quality_metrics.get("penalties", {})),
                        "normalization_changed": bool(normalization_changed),
                        "prompt_token_length": int(prompt_token_length),
                        "prompt_truncated": bool(int(prompt_token_length) > int(cfg.max_prompt_len)),
                        **completion_stats,
                        **completion_meta,
                    }
                )

            if accepted_target > 0:
                _, accepted_info = select_accepted_trajectories(trajectories)
                accepted_count = int(accepted_info.get("accepted", 0) or 0)
                if accepted_count >= accepted_target:
                    print(
                        "[traj] early-stop "
                        f"accepted={accepted_count} target={accepted_target} "
                        f"total_rows={len(trajectories)}"
                    )
                    early_stop_reached = True
                    break
        if early_stop_reached:
            break

    print(f"Collected {len(trajectories)} trajectories.")

    print("\nExample Explorer trajectories:")
    for t in trajectories[:3]:
        print("=" * 80)
        print("PROMPT:")
        print(t["prompt"])
        print("\nCOMPLETION:")
        print(t["completion"])
        print(f"\nreward={t['reward']:.4f}, r_correct={t['r_correct']:.4f}, r_novel={t['r_novel']:.4f}")
    print("=" * 80)

    traj_path = cfg.explorer_traj_path
    traj_dir = os.path.dirname(traj_path)
    if traj_dir:
        os.makedirs(traj_dir, exist_ok=True)
    with open(traj_path, "w") as f:
        for t in trajectories:
            f.write(json.dumps(t) + "\n")
    print(f"Saved trajectories to {traj_path}")
    traj_csv_path = _trajectory_csv_path(traj_path)
    _write_trajectory_csv(traj_csv_path, trajectories)
    print(f"Saved trajectory CSV to {traj_csv_path}")

    if use_wandb and wandb_run is not None and wandb is not None:
        num_traj = len(trajectories)
        num_correct = sum(1 for t in trajectories if t.get("is_correct", False))
        mean_reward = sum(t["reward"] for t in trajectories) / max(1, num_traj)
        mean_novelty = sum(float(t.get("r_novel", 0.0)) for t in trajectories) / max(1, num_traj)
        wandb_run.log(
            {
                "traj/num_trajectories": num_traj,
                "traj/num_correct": num_correct,
                "traj/correct_fraction": num_correct / max(1, num_traj),
                "traj/mean_reward": mean_reward,
                "traj/mean_novelty": mean_novelty,
                "traj/reward_shape": 0 if cfg.reward_shape == "linear" else 1,
                **reward_fn.last_batch_summary,
            }
        )

    return trajectories
