#!/usr/bin/env python3
"""Central SFT (distillation) on accepted Scout trajectories — GPU port of
tmx_jax/distill.py.

Contract:
- Cross-entropy on COMPLETION tokens only (prompt positions masked); labels
  are the shift-left of [prompt | completion], supervised on positions
  [p_len-1, seq_len-1).
- Tokenization: prompts truncate from the LEFT (keep the question at the end),
  completions from the RIGHT, into a max_total_len=18432 budget. The c8
  default appends and supervises exactly one model EOS/chat-stop token after
  every completion, reserving room for it before truncation. The historical
  c7 behavior is available only through --sft-termination-mode legacy.
- AdamW lr 5e-6 (betas 0.9/0.95, eps 1e-8, wd 0, grad clip 1.0), 2 epochs,
  per-epoch shuffle with np.default_rng(seed + epoch) like the TPU run.
- Chunked selective log-softmax CE (reuses grpo_gpu.chunked_selective_log_softmax)
  so full-vocab logits are never materialized for all positions; logits run in
  fp32 under fp32 master params (see grpo_gpu docstring).
- Metrics: sft/loss (paper-curve parity name) + sft/accepted_trajectories,
  and the distill/* diagnostics (epoch_loss, step_time, tokens_per_sec,
  wall_time_seconds), each with global_step / phase="central_sft" / round /
  lambda / num_scouts columns — to the stage's own W&B run (opened with
  reinit="create_new"; never hijacks the pipeline run) AND to the shared
  --metrics-jsonl. sft/acceptance_rate needs the funnel's raw count, so the
  pipeline logs it alongside the funnel counters.

CONFIG TIERS (config_tier logged to W&B):
- CONTRACT (defaults): AdamW lr 5e-6, max_total_len 18432 (2048 + 16384).
- GPU-MAX (opt-in): --optimizer adafactor (TPU-SFT parity ablations),
  --completion-budget N (max_total_len becomes 2048 + N).

DEVIATIONS (vs tmx_jax/distill.py):
- batch 1 with grad accumulation 1 by default, matching the canonical C9
  endpoint (500 examples x 2 epochs = 1,000 optimizer updates). Noncanonical
  ablations may raise --grad-accum; a trailing partial batch is flushed after
  each epoch with its gradients rescaled by accum/pending, so no example's
  gradient is dropped or under-weighted regardless of shuffle position.
- Optimizer defaults to AdamW: the TPU SFT stage used Adafactor purely as a
  memory compromise on v5e/v6e — with H100/A100 headroom the GPU default is
  AdamW (lr 5e-6, betas 0.9/0.95, eps 1e-8, wd 0); pass --optimizer adafactor
  to reproduce the TPU memory-compromise configuration.
- fp32 master params + bf16 autocast compute (see grpo_gpu docstring); the
  saved final model is exported as bf16 safetensors.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tmx_gpu import common  # noqa: E402
from tmx_gpu.grpo_gpu import (  # noqa: E402
    build_optimizer,
    chunked_selective_log_softmax,
    compute_config_tier,
    load_policy,
    save_model_bf16,
)

# GPU-MAX tier detection for the SFT stage (see grpo_gpu.compute_config_tier).
SFT_GPU_MAX_FLAG_DEFAULTS = {
    "optimizer": "adamw",
    "completion_budget": 0,
    "grad_accum": 1,
    "epochs": 2,
    "lr": 5e-6,
    "batch_size": 1,
    "max_total_len": 18432,
    "sft_termination_mode": "append_eos",
}


SFT_TERMINATION_MODES = ("append_eos", "legacy")
SFT_RUNTIME_CONTRACT_VERSION = "sft_runtime_contract_v2"


def init_sft_wandb_with_fallback(init_kwargs: Dict[str, Any]):
    """Open the SFT stage run without leaking fallback mode globally."""
    wandb_run = None
    last_exc: Optional[BaseException] = None
    for attempt in range(1, 4):
        try:
            wandb_run = common.init_stage_wandb(**init_kwargs)
            break
        except Exception as exc:  # noqa: BLE001 — W&B outage, not our bug
            last_exc = exc
            print(f"[sft] WARNING: wandb.init failed "
                  f"(attempt {attempt}/3): {exc}", flush=True)
            time.sleep(5.0 * attempt)
    if wandb_run is not None:
        return wandb_run
    print(f"[sft] WARNING: W&B unreachable after 3 attempts ({last_exc}); "
          "retrying in WANDB_MODE=offline — metrics.jsonl remains the "
          "durable record", flush=True)
    previous_wandb_mode = os.environ.get("WANDB_MODE")
    try:
        os.environ["WANDB_MODE"] = "offline"
        try:
            return common.init_stage_wandb(**init_kwargs)
        except Exception as exc:  # noqa: BLE001
            print(f"[sft] WARNING: offline wandb.init also failed ({exc}); "
                  "proceeding WITHOUT W&B", flush=True)
            return None
    finally:
        # SFT executes in the long-lived pipeline process. Leaving this
        # fallback behind would silently force Central's stage run and all
        # subsequent artifact uploads offline.
        if previous_wandb_mode is None:
            os.environ.pop("WANDB_MODE", None)
        else:
            os.environ["WANDB_MODE"] = previous_wandb_mode


def build_sft_runtime_contract(
    args: argparse.Namespace, *, resolved_max_total_len: int,
) -> Dict[str, Any]:
    """Exact training-affecting SFT contract stored in every model snapshot."""
    contract = {
        "version": SFT_RUNTIME_CONTRACT_VERSION,
        "config_tier": compute_config_tier(args, SFT_GPU_MAX_FLAG_DEFAULTS),
        "model_init": str(args.model),
        "base_model_revision": str(
            getattr(args, "model_revision", "") or ""),
        "optimizer": str(args.optimizer),
        "learning_rate": float(args.lr),
        "adam_beta1": float(args.adam_beta1),
        "adam_beta2": float(args.adam_beta2),
        "adam_eps": float(args.adam_eps),
        "weight_decay": float(args.weight_decay),
        "grad_clip": float(args.grad_clip),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "grad_accum": int(args.grad_accum),
        "completion_budget_flag": int(args.completion_budget),
        "resolved_max_total_len": int(resolved_max_total_len),
        "termination_mode": str(args.sft_termination_mode),
        "seed": int(args.seed),
        "shuffle_order": "numpy_default_rng_seed_plus_epoch",
        "parameter_dtype": str(args.param_dtype),
        "chunk_tokens": int(args.chunk_tokens),
        "loss": "completion_only_token_mean_cross_entropy",
        "prompt_truncation": "left",
        "completion_truncation": "right_reserve_exactly_one_eos",
    }
    if bool(getattr(
            args, "require_canonical_training_environment", False)):
        from tmx_gpu import grpo_gpu

        if str(getattr(args, "container_image_identity", "") or "") != \
                grpo_gpu.CANONICAL_TRAINING_CONTAINER_IMAGE:
            raise ValueError(
                "strict SFT requires the exact canonical container image")
        environment = getattr(args, "_training_execution_environment", None)
        if environment is None:
            from tmx_gpu import orchestrator_receipt

            code_binding = grpo_gpu.load_code_source_manifest(str(
                getattr(args, "code_source_manifest", "") or ""))
            if code_binding is None:
                raise ValueError(
                    "strict SFT requires --code-source-manifest")
            source = code_binding["source"]
            receipt_binding = orchestrator_receipt.load_binding(
                str(getattr(args, "orchestrator_receipt_binding", "") or ""),
                expected_run_name=str(getattr(
                    args, "orchestrator_run_name", "") or getattr(
                        args, "run_name", "") or ""),
                expected_container_image=str(getattr(
                    args, "container_image_identity", "") or ""),
                expected_code_artifact_ref=source["artifact_ref"],
                expected_code_artifact_digest=source["artifact_digest"],
                expected_command_sha256=str(getattr(
                    args, "orchestrator_command_sha256", "") or ""),
                expected_polling_artifact_ref=str(getattr(
                    args, "orchestrator_receipt_artifact", "") or ""),
            )
            environment = grpo_gpu.canonical_training_execution_environment(
                str(getattr(args, "container_image_identity", "") or ""),
                receipt_binding,
                orchestrator_receipt.observed_native_compatibility_facts(),
            )
        contract["execution_environment"] = environment
    return contract


def parse_sft_save_steps(value: str | None) -> tuple[int, ...]:
    """Parse an opt-in, comma-separated optimizer-step checkpoint schedule.

    Empty input preserves the historical behavior (only ``actual_sft_final``
    is saved).  Sorting and de-duplicating makes the on-disk result stable
    without changing which optimizer states are captured.
    """
    if value is None or not str(value).strip():
        return ()
    pieces = str(value).split(",")
    if any(not piece.strip() for piece in pieces):
        raise argparse.ArgumentTypeError(
            "SFT save steps must be a comma-separated list of positive integers")
    try:
        steps = [int(piece.strip()) for piece in pieces]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "SFT save steps must be a comma-separated list of positive integers") from exc
    if any(step <= 0 for step in steps):
        raise argparse.ArgumentTypeError("SFT save steps must all be positive")
    return tuple(sorted(set(steps)))


def _model_eos_token_id(tokenizer) -> int:
    """Return the tokenizer's canonical model EOS/chat-stop token id.

    Qwen3 exposes ``<|im_end|>`` as ``eos_token_id``. Requiring the tokenizer
    declaration (rather than hard-coding a Qwen id) keeps this correct for the
    external-family runs as well.
    """
    token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(token_id, (list, tuple)):
        token_id = token_id[0] if token_id else None
    if token_id is None:
        raise ValueError(
            "--sft-termination-mode append_eos requires tokenizer.eos_token_id; "
            "use a tokenizer with a declared conversational stop/EOS token "
            "or explicitly select --sft-termination-mode legacy")
    return int(token_id)


def pretokenize_sft_examples(
    tokenizer,
    examples: List[Dict[str, Any]],
    max_total_len: int,
    *,
    termination_mode: str = "append_eos",
):
    """Tokenize SFT rows with explicit, auditable termination supervision.

    ``append_eos`` is the c8 contract: reserve one token in the sequence
    budget, strip any *trailing* EOS copies already present in the completion,
    and append exactly one canonical model EOS. If the completion has content,
    the prompt budget also reserves one content token where the total budget
    permits. This makes right truncation unable to discard the stop target.

    ``legacy`` reproduces c7 byte-for-byte tokenization: no EOS is appended and
    a prompt may consume the entire sequence budget.

    Per-row audit fields are included beside ``p_ids``/``c_ids``; the loss
    ignores them and :func:`run_sft` aggregates them into metrics/provenance.
    """
    if termination_mode not in SFT_TERMINATION_MODES:
        raise ValueError(
            f"unknown SFT termination mode {termination_mode!r}; "
            f"choose one of {SFT_TERMINATION_MODES}")
    if int(max_total_len) <= 0:
        raise ValueError("max_total_len must be positive")
    if termination_mode == "append_eos" and int(max_total_len) < 2:
        raise ValueError("append_eos requires max_total_len >= 2 (prompt + EOS)")

    out = []
    prev_side = getattr(tokenizer, "truncation_side", "right")
    try:
        for ex in examples:
            if termination_mode == "legacy":
                tokenizer.truncation_side = "left"
                p = tokenizer(
                    ex["prompt_text"], truncation=True, max_length=max_total_len,
                    add_special_tokens=False,
                )["input_ids"]
                tokenizer.truncation_side = "right"
                c_max = max(0, max_total_len - len(p))
                c = tokenizer(
                    ex["completion_text"], truncation=True, max_length=c_max,
                    add_special_tokens=False,
                )["input_ids"] if c_max > 0 else []
                out.append({
                    "p_ids": [int(x) for x in p],
                    "c_ids": [int(x) for x in c],
                    "eos_supervised": False,
                    "eos_already_present": False,
                    "trailing_eos_removed": 0,
                    "prompt_truncated": False,
                    "completion_truncated": False,
                })
                continue

            eos_id = _model_eos_token_id(tokenizer)
            # Tokenize without truncation first so truncation and EOS handling
            # are deterministic and can be audited independently of tokenizer
            # implementation details.
            p_all = [int(x) for x in tokenizer(
                ex["prompt_text"], add_special_tokens=False,
            )["input_ids"]]
            c_all = [int(x) for x in tokenizer(
                ex["completion_text"], add_special_tokens=False,
            )["input_ids"]]

            trailing_eos = 0
            while c_all and c_all[-1] == eos_id:
                c_all.pop()
                trailing_eos += 1

            # Always reserve EOS. When there is completion content and enough
            # total capacity, reserve at least one content token as well.
            reserve = 1 + int(bool(c_all) and int(max_total_len) >= 3)
            prompt_budget = max(1, int(max_total_len) - reserve)
            p = p_all[-prompt_budget:]
            content_budget = max(0, int(max_total_len) - len(p) - 1)
            content = c_all[:content_budget]
            c = content + [eos_id]
            if len(p) + len(c) > int(max_total_len):  # defensive invariant
                raise AssertionError("EOS-aware SFT tokenization exceeded total budget")
            if c[-1] != eos_id or (len(c) >= 2 and c[-2] == eos_id):
                raise AssertionError("EOS-aware SFT row must end in exactly one EOS")

            out.append({
                "p_ids": p,
                "c_ids": c,
                "eos_supervised": True,
                "eos_already_present": trailing_eos > 0,
                "trailing_eos_removed": trailing_eos,
                "prompt_truncated": len(p) < len(p_all),
                "completion_truncated": len(content) < len(c_all),
            })
    finally:
        tokenizer.truncation_side = prev_side
    return out


def completion_ce_loss(model, ex: Dict[str, List[int]], device: str,
                       chunk_tokens: int, use_autocast: bool) -> torch.Tensor | None:
    """Token-mean CE over completion tokens for one [prompt|completion] row."""
    p_ids, c_ids = ex["p_ids"], ex["c_ids"]
    if not c_ids or len(p_ids) + len(c_ids) < 2:
        return None
    seq = p_ids + c_ids
    p_len, c_len = len(p_ids), len(c_ids)
    input_ids = torch.tensor([seq], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    # Supervise completion tokens only: targets seq[p_len : p_len+c_len]
    # predicted from hidden positions [p_len-1, p_len+c_len-1). p_len >= 1
    # always (chat-template prompts); guard degenerate rows anyway.
    if p_len == 0:
        p_len, c_len = 1, c_len - 1
        if c_len <= 0:
            return None
    targets = torch.tensor(seq[p_len:p_len + c_len], dtype=torch.long, device=device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast):
        hidden = model.model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).last_hidden_state[0]
        logps = chunked_selective_log_softmax(
            hidden[p_len - 1: p_len + c_len - 1], model.lm_head, targets,
            chunk_tokens=chunk_tokens,
        )
    return -logps.mean()


def run_sft(args: argparse.Namespace) -> str:
    """Distill accepted trajectories into the Central model; return model dir."""
    if bool(getattr(
            args, "require_canonical_training_environment", False)):
        from tmx_gpu import grpo_gpu, orchestrator_receipt

        if str(getattr(args, "container_image_identity", "") or "") != \
                grpo_gpu.CANONICAL_TRAINING_CONTAINER_IMAGE:
            raise ValueError(
                "strict SFT requires the exact canonical container image")
        code_binding = grpo_gpu.load_code_source_manifest(str(
            getattr(args, "code_source_manifest", "") or ""))
        if code_binding is None:
            raise ValueError("strict SFT requires --code-source-manifest")
        source = code_binding["source"]
        receipt_binding = orchestrator_receipt.load_binding(
            str(getattr(args, "orchestrator_receipt_binding", "") or ""),
            expected_run_name=str(getattr(
                args, "orchestrator_run_name", "") or getattr(
                    args, "run_name", "") or ""),
            expected_container_image=str(
                getattr(args, "container_image_identity", "") or ""),
            expected_code_artifact_ref=source["artifact_ref"],
            expected_code_artifact_digest=source["artifact_digest"],
            expected_command_sha256=str(getattr(
                args, "orchestrator_command_sha256", "") or ""),
            expected_polling_artifact_ref=str(getattr(
                args, "orchestrator_receipt_artifact", "") or ""),
        )
        args._code_artifact_binding = code_binding
        args._orchestrator_receipt_binding = receipt_binding
        args._training_execution_environment = \
            grpo_gpu.validate_canonical_training_execution_environment(
                str(getattr(args, "container_image_identity", "") or ""),
                receipt_binding,
            )
    from transformers import AutoTokenizer

    if int(getattr(args, "completion_budget", 0) or 0) > 0:
        args.max_total_len = 2048 + int(args.completion_budget)
        print(f"[sft] GPU-MAX completion budget {args.completion_budget}: "
              f"max_total_len={args.max_total_len}", flush=True)
    config_tier = compute_config_tier(args, SFT_GPU_MAX_FLAG_DEFAULTS)
    runtime_contract = build_sft_runtime_contract(
        args, resolved_max_total_len=int(args.max_total_len))
    print(f"[sft] config_tier={config_tier}", flush=True)
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True,
        **common.remote_revision_kwargs(args.model, args.model_revision))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    accepted_rows: List[Dict[str, Any]] = []
    with open(args.accepted) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            accepted_rows.append(r)
    examples = [
        {"prompt_text": r["prompt_text"], "completion_text": r["completion_text"]}
        for r in accepted_rows
    ]
    if not examples:
        raise RuntimeError(f"no accepted trajectories in {args.accepted}")
    accepted_library_digest = common.semantic_sft_dataset_digest(accepted_rows)
    accepted_file_sha256 = common.file_sha256(args.accepted)
    training_sequence_digest = common.sft_training_sequence_digest(
        accepted_rows, epochs=int(args.epochs), seed=int(args.seed))
    trajectory_source = None
    if args.trajectory_source_manifest:
        with open(args.trajectory_source_manifest, encoding="utf-8") as handle:
            source_manifest = json.load(handle)
        trajectory_source = {
            "manifest": source_manifest,
            "digest": common.trajectory_source_manifest_digest(source_manifest),
        }
    training_evidence = None
    fresh_c8_training_evidence = None
    c9_training_evidence = None
    if args.sft_training_evidence_manifest:
        with open(args.sft_training_evidence_manifest, encoding="utf-8") as handle:
            training_evidence = json.load(handle)
        if not isinstance(training_evidence, dict):
            raise ValueError("--sft-training-evidence-manifest must contain one object")
        # Validate at the consumer boundary so a hand-written JSON file cannot
        # be laundered into checkpoint provenance.  C8 and C9 are deliberately
        # disjoint evidence contracts.
        version = training_evidence.get("version")
        from tmx_gpu import c9_evidence
        if version == c9_evidence.SFT_TRAINING_EVIDENCE_VERSION:
            c9_evidence.validate_sft_training_evidence(training_evidence)
            c9_training_evidence = training_evidence
        else:
            from tmx_gpu import fresh_c8_sft

            fresh_c8_sft.validate_training_evidence(training_evidence)
            fresh_c8_training_evidence = training_evidence
    print(f"[sft] loaded {len(examples)} accepted trajectories from {args.accepted}", flush=True)

    t0 = time.perf_counter()
    pretokenized = pretokenize_sft_examples(
        tokenizer, examples, int(args.max_total_len),
        termination_mode=str(args.sft_termination_mode),
    )
    print(f"[sft] pretokenized {len(pretokenized)} examples in {time.perf_counter() - t0:.1f}s", flush=True)

    sft_contract_version = ("c8-eos-aware"
                            if args.sft_termination_mode == "append_eos"
                            else "c7-legacy")
    eos_id = (_model_eos_token_id(tokenizer)
              if args.sft_termination_mode == "append_eos" else None)
    eos_supervised = sum(bool(x.get("eos_supervised")) for x in pretokenized)
    eos_already_present = sum(bool(x.get("eos_already_present")) for x in pretokenized)
    trailing_eos_removed = sum(int(x.get("trailing_eos_removed", 0)) for x in pretokenized)
    prompt_truncated = sum(bool(x.get("prompt_truncated")) for x in pretokenized)
    completion_truncated = sum(bool(x.get("completion_truncated")) for x in pretokenized)
    trainable_examples = sum(bool(x["c_ids"]) and len(x["p_ids"]) + len(x["c_ids"]) >= 2
                             for x in pretokenized)
    accum = max(1, int(args.grad_accum))
    planned_optimizer_steps = int(args.epochs) * int(np.ceil(trainable_examples / accum))
    requested_save_steps = tuple(
        int(x) for x in getattr(args, "sft_save_steps", ()))
    unreachable_save_steps = [x for x in requested_save_steps
                              if x > planned_optimizer_steps]
    if unreachable_save_steps:
        raise ValueError(
            "requested SFT optimizer-step checkpoints exceed the planned "
            f"{planned_optimizer_steps} updates: {unreachable_save_steps}")
    print(
        f"[sft] termination_mode={args.sft_termination_mode} eos_token_id={eos_id} "
        f"eos_supervised={eos_supervised}/{len(pretokenized)} "
        f"completion_truncated={completion_truncated}",
        flush=True,
    )

    model = load_policy(
        args.model, device=args.device, param_dtype=args.param_dtype,
        model_revision=str(args.model_revision or ""))
    use_autocast = torch.device(args.device).type == "cuda" and str(args.param_dtype) == "float32"
    optimizer = build_optimizer(
        args.optimizer, model.parameters(), lr=float(args.lr),
        betas=(float(args.adam_beta1), float(args.adam_beta2)),
        eps=float(args.adam_eps), weight_decay=float(args.weight_decay),
    )

    wandb_run = None
    if args.wandb:
        # Own stage run (reinit="create_new"): never hijacks/finishes the
        # pipeline run active in this process. See common.init_stage_wandb.
        # SFT sits at the scout->central boundary of a multi-day run: a
        # transient W&B outage must NOT kill the pipeline — retry 3x, then
        # fall back to offline mode, then proceed without W&B (metrics.jsonl
        # stays the durable record either way).
        init_kwargs = dict(
            name=f"{args.run_name}-sft-r{int(args.round)}",
            group=os.environ.get("WANDB_RUN_GROUP", args.run_name),
            config={**vars(args), "config_tier": config_tier,
                    "resolved_max_total_len": int(args.max_total_len),
                    "num_examples": len(examples), "backend": "gpu-torch",
                    "sft_contract_version": sft_contract_version,
                    "resolved_eos_token_id": eos_id},
        )
        wandb_run = init_sft_wandb_with_fallback(init_kwargs)

    metrics_logger: Optional[common.MetricsLogger] = None
    global_step = int(args.global_step_offset)  # SFT sits at the scout/central boundary
    if args.metrics_jsonl:
        # SFT has no mid-stage resume: a rerun starts over, so drop any rows a
        # crashed attempt left for this (phase, round).
        common.trim_metrics_stage(args.metrics_jsonl, phase="central_sft",
                                  round_idx=int(args.round))
        metrics_logger = common.MetricsLogger(args.metrics_jsonl)

    def emit(record: Dict[str, Any], wandb_step: int) -> None:
        record = {"global_step": global_step, "phase": "central_sft",
                  "round": int(args.round), "lambda": float(args.round_lambda),
                  "lambda_novelty": float(args.round_lambda),
                  "num_scouts": int(args.num_scouts), **record}
        if wandb_run is not None:
            wandb_run.log(record, step=int(wandb_step))
        if metrics_logger is not None:
            metrics_logger.log(record)

    emit({
        "sft/accepted_trajectories": float(len(examples)),
        "sft/trainable_examples": float(trainable_examples),
        "sft/eos_supervised_examples": float(eos_supervised),
        "sft/eos_already_present_examples": float(eos_already_present),
        "sft/trailing_eos_removed": float(trailing_eos_removed),
        "sft/prompt_truncated_examples": float(prompt_truncated),
        "sft/completion_truncated_examples": float(completion_truncated),
        "sft/planned_optimizer_steps": float(planned_optimizer_steps),
        "sft/eos_token_id": float(eos_id) if eos_id is not None else -1.0,
        "sft/termination_eos_enabled": float(args.sft_termination_mode == "append_eos"),
    }, 0)

    total_steps = 0
    micro_steps = 0
    saved_optimizer_steps: List[int] = []
    train_start = time.perf_counter()

    def save_optimizer_step_checkpoint(step: int) -> str:
        """Save the post-update policy at an exact SFT optimizer step.

        These are full Hugging Face model/tokenizer directories intended for
        behavior-gate evaluation, not training-resume checkpoints (optimizer
        moments are deliberately not implied by the directory name).
        """
        checkpoint_dir = os.path.join(
            args.output_dir, f"optimizer_step_{int(step):06d}")
        save_model_bf16(model, tokenizer, checkpoint_dir)
        common.atomic_write_json(
            os.path.join(checkpoint_dir, "sft_checkpoint.json"),
            {
                "checkpoint_kind": "post_optimizer_step_hf_model",
                "optimizer_step": int(step),
                "micro_steps_seen": int(micro_steps),
                "epoch_index_zero_based": int(epoch),
                "model_init": str(args.model),
                "model_revision": str(args.model_revision or ""),
                "optimizer": str(args.optimizer),
                "learning_rate": float(args.lr),
                "epochs_planned": int(args.epochs),
                "grad_accum": accum,
                "accepted_examples": len(examples),
                "trainable_examples": trainable_examples,
                "planned_optimizer_steps": planned_optimizer_steps,
                "requested_save_optimizer_steps": list(requested_save_steps),
                "sft_contract_version": sft_contract_version,
                "termination_mode": str(args.sft_termination_mode),
                "accepted_library_digest": accepted_library_digest,
                "accepted_file_sha256": accepted_file_sha256,
                "training_sequence_digest": training_sequence_digest,
                "trajectory_source": trajectory_source,
                "fresh_c8_training_evidence": fresh_c8_training_evidence,
                "c9_training_evidence": c9_training_evidence,
                "sft_runtime_contract": runtime_contract,
            },
        )
        saved_optimizer_steps.append(int(step))
        print(
            f"[sft] saved post-update optimizer-step checkpoint {step} "
            f"-> {checkpoint_dir}", flush=True)
        return checkpoint_dir

    for epoch in range(int(args.epochs)):
        order = np.random.default_rng(int(args.seed) + epoch).permutation(len(pretokenized))
        epoch_loss_sum, epoch_micro = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        pending = 0

        def flush() -> float:
            """Optimizer step over the pending micro-batch; partial batches
            (epoch tail) rescale gradients by accum/pending so every example
            carries full weight."""
            nonlocal pending, total_steps
            if pending < accum:
                scale = accum / pending
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(scale)
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip)))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            pending = 0
            total_steps += 1
            if total_steps in requested_save_steps:
                save_optimizer_step_checkpoint(total_steps)
            return norm

        for idx in order.tolist():
            t_step = time.perf_counter()
            loss = completion_ce_loss(model, pretokenized[idx], args.device,
                                      int(args.chunk_tokens), use_autocast)
            if loss is None:
                continue
            (loss / accum).backward()
            pending += 1
            micro_steps += 1
            epoch_micro += 1
            loss_val = float(loss.detach())
            epoch_loss_sum += loss_val
            tokens = len(pretokenized[idx]["c_ids"])
            grad_norm = flush() if pending == accum else 0.0
            dt = time.perf_counter() - t_step
            emit({
                "sft/loss": loss_val,
                "sft/step": micro_steps,
                "distill/optimizer_step": total_steps,
                "distill/epoch": epoch,
                "distill/loss": loss_val,
                "distill/grad_norm": grad_norm,
                "distill/step_time": dt,
                "distill/label_tokens": tokens,
                "distill/tokens_per_sec": tokens / max(dt, 1e-6),
                "distill/wall_time_seconds": time.perf_counter() - train_start,
                "distill/learning_rate": float(args.lr),
            }, micro_steps)
            if micro_steps % 10 == 0:
                print(f"[sft] micro {micro_steps} (opt {total_steps}) epoch {epoch} "
                      f"loss={loss_val:.4f} dt={dt:.1f}s", flush=True)
        if pending > 0:  # epoch tail smaller than accum — never drop it
            flush()
        epoch_loss = epoch_loss_sum / max(1, epoch_micro)
        print(f"[sft] epoch {epoch + 1}/{args.epochs}: avg loss={epoch_loss:.4f}", flush=True)
        emit({
            "distill/epoch_loss": epoch_loss,
            "distill/epoch": epoch + 1,
            "distill/wall_time_seconds": time.perf_counter() - train_start,
        }, micro_steps)
    emit({"sft/optimizer_steps": float(total_steps),
          "sft/micro_steps": float(micro_steps)}, micro_steps)

    # Gate artifacts are uploaded only after SFT completes.  Amend every
    # preregistered snapshot with the completed-run facts so a standalone
    # checkpoint cannot masquerade as the exact c7-reuse ga8 diagnostic when
    # the run stopped early or omitted one of the four snapshots.
    training_evidence_sha = (
        training_evidence.get("evidence_sha256")
        if isinstance(training_evidence, dict) else None)
    completed_run_evidence = {
        "version": "sft_completed_run_evidence_v1",
        "training_completed": True,
        "optimizer_steps_executed": int(total_steps),
        "micro_steps_executed": int(micro_steps),
        "saved_optimizer_steps": list(saved_optimizer_steps),
        "training_evidence_sha256": training_evidence_sha,
    }
    completed_run_evidence["evidence_sha256"] = common.canonical_json_sha256(
        completed_run_evidence)
    common.atomic_write_json(
        os.path.join(args.output_dir, "sft_completed_run.json"),
        completed_run_evidence)
    for saved_step in saved_optimizer_steps:
        metadata_path = os.path.join(
            args.output_dir, f"optimizer_step_{int(saved_step):06d}",
            "sft_checkpoint.json",
        )
        with open(metadata_path, encoding="utf-8") as handle:
            metadata = json.load(handle)
        metadata.update({
            "training_completed": True,
            "optimizer_steps_executed": int(total_steps),
            "micro_steps_executed": int(micro_steps),
            "saved_optimizer_steps": list(saved_optimizer_steps),
            "completed_run_evidence": completed_run_evidence,
        })
        common.atomic_write_json(metadata_path, metadata)

    out_dir = os.path.join(args.output_dir, "actual_sft_final")
    save_model_bf16(model, tokenizer, out_dir)
    final_file_manifest = common.directory_file_manifest(
        out_dir, exclude_relative_paths=("sft_provenance.json",))
    common.atomic_write_json(os.path.join(out_dir, "sft_provenance.json"), {
        "sft_contract_version": sft_contract_version,
        "termination_mode": str(args.sft_termination_mode),
        "eos_token_id": eos_id,
        "accepted_examples": len(examples),
        "trainable_examples": trainable_examples,
        "eos_supervised_examples": eos_supervised,
        "eos_already_present_examples": eos_already_present,
        "trailing_eos_removed": trailing_eos_removed,
        "prompt_truncated_examples": prompt_truncated,
        "completion_truncated_examples": completion_truncated,
        "epochs": int(args.epochs),
        "model_init": str(args.model),
        "model_revision": str(args.model_revision or ""),
        "optimizer": str(args.optimizer),
        "learning_rate": float(args.lr),
        "grad_accum": accum,
        "micro_steps": micro_steps,
        "optimizer_steps": total_steps,
        "planned_optimizer_steps": planned_optimizer_steps,
        "requested_save_optimizer_steps": list(requested_save_steps),
        "saved_optimizer_steps": saved_optimizer_steps,
        "max_total_len": int(args.max_total_len),
        "accepted_library_digest": accepted_library_digest,
        "accepted_file_sha256": accepted_file_sha256,
        "training_sequence_digest": training_sequence_digest,
        "trajectory_source": trajectory_source,
        "fresh_c8_training_evidence": fresh_c8_training_evidence,
        "c9_training_evidence": c9_training_evidence,
        "completed_run_evidence": completed_run_evidence,
        "sft_runtime_contract": runtime_contract,
        "checkpoint_file_manifest": final_file_manifest,
    })
    print(f"[sft] saved central SFT model -> {out_dir}", flush=True)
    if wandb_run is not None:
        wandb_run.finish()
    return out_dir


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="Qwen/Qwen3-1.7B", help="SFT init (HF id or dir)")
    p.add_argument(
        "--model-revision", default=common.CANONICAL_TRAINING_MODEL_REVISION,
        help="immutable Hub revision for remote Base initialization only",
    )
    p.add_argument("--accepted", required=True, help="accepted.jsonl from filter_pool")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--run-name", default="sft-gpu")
    p.add_argument(
        "--orchestrator-run-name", default="",
        help="top-level remote RUN_NAME when a pipeline stage has another name",
    )
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--batch-size", type=int, default=1, help="fixed micro-batch of 1")
    p.add_argument(
        "--grad-accum", type=int, default=1,
        help="gradient accumulation; c8/TPU contract uses per-example "
             "optimizer stepping (1). Pass 8 explicitly to reproduce c7",
    )
    p.add_argument(
        "--sft-termination-mode", default="append_eos",
        choices=list(SFT_TERMINATION_MODES),
        help="c8 default append_eos reserves and supervises exactly one model "
             "EOS/chat-stop token; legacy reproduces c7 without a stop target",
    )
    p.add_argument(
        "--sft-save-steps", type=parse_sft_save_steps, default=(), metavar="STEPS",
        help="optional comma-separated SFT optimizer steps to save as full "
             "post-update Hugging Face checkpoints (the exposure-gate "
             "preregistration is 16,32,64,126); empty preserves final-only saving",
    )
    p.add_argument("--max-total-len", type=int, default=18432)
    p.add_argument("--completion-budget", type=int, default=0,
                   help="GPU-MAX: sets max_total_len = 2048 + N; 0 = contract")
    p.add_argument("--optimizer", default="adamw", choices=["adamw", "adafactor"],
                   help="contract: adamw (TPU used adafactor as a memory compromise)")
    p.add_argument("--adam-beta1", type=float, default=0.9)
    p.add_argument("--adam-beta2", type=float, default=0.95)
    p.add_argument("--adam-eps", type=float, default=1e-8)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--chunk-tokens", type=int, default=256)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--param-dtype", default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--round", type=int, default=1)
    p.add_argument("--round-lambda", type=float, default=0.0,
                   help="current round's lambda_novelty (metrics column only)")
    p.add_argument(
        "--container-image-identity", default="",
        help="immutable OCI image tag@sha256 identity for strict C9/MR-ME SFT",
    )
    p.add_argument(
        "--code-source-manifest", default="",
        help="exact digest-scoped executing-code artifact manifest for strict SFT",
    )
    p.add_argument(
        "--orchestrator-receipt-binding", default="",
        help="resolved immutable W&B/Lepton job-spec binding for strict SFT",
    )
    p.add_argument(
        "--orchestrator-command-sha256", default="",
        help="controller-preregistered SHA256 of the exact Lepton job command",
    )
    p.add_argument(
        "--orchestrator-receipt-artifact", default="",
        help="RUN_NAME-unique receipt polling alias captured by the job spec",
    )
    p.add_argument(
        "--require-canonical-training-environment", action="store_true",
        help="fail closed unless the immutable image and full installed stack "
             "match the canonical C9/MR-ME training environment",
    )
    p.add_argument("--metrics-jsonl", default="",
                   help="shared metrics JSONL (the pipeline passes "
                        "<output-dir>/metrics.jsonl)")
    p.add_argument("--global-step-offset", type=int, default=0,
                   help="pipeline global step at the scout/central boundary "
                        "(SFT metric rows log this constant global_step)")
    p.add_argument("--num-scouts", type=int, default=1,
                   help="K scouts feeding this pipeline run (logging column)")
    p.add_argument(
        "--trajectory-source-manifest", default="",
        help="optional W&B trajectory-source manifest; its normalized digest "
             "is bound into every SFT checkpoint/provenance sidecar",
    )
    p.add_argument(
        "--sft-training-evidence-manifest", default="",
        help="fresh-c8 only: signed pre-SFT Scout/filter/code evidence. The "
             "validated object is embedded in every checkpoint and final "
             "provenance sidecar",
    )
    p.add_argument("--wandb", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    out_dir = run_sft(args)
    print(f"[sft] DONE final={out_dir}", flush=True)


if __name__ == "__main__":
    main()
