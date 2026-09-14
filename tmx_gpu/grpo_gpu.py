#!/usr/bin/env python3
"""GPU GRPO trainer (Scout / Central phases) mirroring the measured TPU runs.

Single training GPU + a pool of vLLM servers on the remaining GPUs.
Semantically mirrors tmx_jax/train.py (_score_rollouts, _dynamic_sample_rollouts,
_build_rollout_batch) with the TPU-measured REINFORCE-with-group-baseline
semantics: one update per rollout batch, old_logp == current logp, ratio == 1,
DAPO clip inert => train/clip_fraction is 0.0 by construction.

Contract (from the GPU reproduction plan / contract.py):
- Reward: correct +1 / incorrect -1 (verifier tmx_jax.rewarding applied to the
  LAST 4096 chars of the completion); KL beta 0;
  blended = correctness + lambda * novelty_used + soft_overlong_penalty where
  novelty_used = novelty * (1.0 if correct else incorrect_novelty_scale=0.0 —
  measured-flagship value; wrong answers get NO novelty credit)
  and penalty = -min(1, (tok_len - 13107)/3277) for tok_len > 13107 using the
  TRUE tokenized completion length (clamped at max_completion_len=16384).
- Advantage = blended - group mean over the 16-generation group. NO std division.
- Dynamic sampling: group informative iff 0 < n_correct < 16. Legacy mode
  accepts homogeneous lambda>0 groups when full blended-reward std > 1e-6.
  The explicit c8 validity mode accepts them only when the actual novelty
  contribution varies, so overlong variance alone cannot pass the gate;
  max_attempts=8; leftover slots filled from discarded groups (fallback).
- Loss = -(sum over rows of adv_row * sum(logp * mask_row)) / (num_rows * 16384)
  with a FIXED denominator (Dr.GRPO); rows processed sequentially with grad
  accumulation across all 64 rows, then a single AdamW step
  (lr 5e-6 scout / 1e-6 central, betas 0.9/0.95, eps 1e-8, wd 0, clip 1.0).
- Sampling: T=1.0 / top_p 0.95 / top_k 20 / max_tokens 16384, /v1/completions
  with client-side chat-template-rendered prompts (never /chat/completions).
- vLLM weight sync: legacy mode reloads every --reload-every steps (default
  10). The explicit c8 validity mode requires a managed pool and reloads after
  every learner update, making the next rollout on-policy; served-model-name
  stays constant across restart-based bf16 exports.
- Novelty: multilayer RND (layers 7/14/21 of the LIVE policy), sqrt_mse,
  computed BEFORE the single Adam(1e-4) predictor step per scoring batch;
  the predictor trains on ALL scored candidates. Central forces lambda=0.

CONFIG TIERS (config_tier is logged to W&B so runs are self-describing):
- CONTRACT (all defaults; headline reproduction runs): completion budget
  16384 with soft-overlong 13107/3277, 4 prompts x 16 generations = 64
  rollouts/step, REINFORCE semantics (1 update/rollout, old_logp = detached
  current logp, clip inert), AdamW lr 5e-6 (scout).
- GPU-MAX (opt-in, H100/A100 headroom; any of these flips config_tier to
  "gpu-max:<changed-flags>"):
  * --completion-budget N: completion budget N, soft-overlong splits as
    expected_len=int(0.8*N), cache_len=N-expected_len; max_total_len=2048+N;
    a self-managed vLLM pool raises --max-model-len accordingly (the
    pipeline does the same for its shared pool).
  * --prompts-per-step 8|16 (128/256 rollouts per step).
  * --ppo-mode: real PPO clipping — old logprobs are a frozen snapshot
    rescored from the policy right after generation and BEFORE any update;
    asymmetric DAPO clip (--clip-epsilon-low 0.2 / --clip-epsilon-high 0.28)
    actually active; --updates-per-rollout K>1 supported (K epochs over the
    same rollout batch per step). train/clip_fraction becomes a measured
    quantity instead of 0.0-by-construction.
  * --optimizer {adamw,adafactor} (scout/central stage optimizer).

METRIC PARITY (paper wandb_style_curves CSVs): every step logs, under exactly
these names, train/reward_total_mean, train/reward_correctness_mean,
train/reward_novelty_mean, train/correctness_rate, train/clip_fraction,
train/grpo_loss, train/policy_kl (0.0 in REINFORCE mode),
quality/valid_answer_rate, quality/clipped_rate, quality/nontermination_rate,
quality/mean_completion_tokens, plus global_step / phase (scout_grpo |
central_grpo) / round / lambda / num_scouts — to W&B AND to the shared
--metrics-jsonl (rehydratable without W&B). global_step = --global-step-offset
+ step; the pipeline passes the cumulative offset of prior stages so
multi-round curves remain monotonic across round boundaries.

MID-TRAINING PROBE: --probe-every N (default 25; 0=off) evaluates the
non-headline eval/AIME24_avg_at_4 diagnostic between steps on a fixed subset —
the full 30 AIME24 problems
x 4 samples each, PROBE_MAX_TOKENS=32768 max tokens, eval sampling
(T=0.6/top_p 0.95/top_k 20) — through the serving pool (i.e. at the
last-synced weights). c8 probes are freshly synced; legacy probes log their
policy-step staleness. The pool must be sized max_prompt_len + 32768
(enforce_probe_serving_len auto-raises own pools, fails fast on attach-mode).
Logged at the current global_step so curves rehydrate at their training step.
An all-error probe logs only eval/probe_error_rows — never a fake 0.0 average.
--probe-dataset accepts a local {"problem","answer"} JSONL for smoke tests.

FAILURE HANDLING: generation groups containing a retry-exhausted request
(finish_reason=="error") are dropped BEFORE scoring — error rows never enter
group means, advantages, trajectory files, or metrics (counted only under
grpo/dynamic/generation_error_*). The pool raises after
--max-consecutive-gen-failures failing generate() calls, and dead managed
servers are health-checked/relaunched between steps (vllm_pool.health_check).

DEVIATIONS (vs tmx_jax):
- fp32 master params + bf16 autocast compute (the TPU contract was "fp32
  master params, bf16 compute"; the shorthand "policy in bf16" is interpreted
  as compute dtype — pure-bf16 AdamW at lr 5e-6 underflows bf16 weight
  precision). Logits are always computed in fp32 when master params are fp32
  (matching the TPU fp32-logit einsum). --param-dtype bfloat16 opts into pure
  bf16 for smoke runs.
- Rows are unpadded [prompt(-2048:) | completion(:16384)] instead of the TPU
  fixed padded layout; identical completion logprobs, less compute.
- Zero-advantage rows are skipped in the backward pass — an exact no-op under
  the fixed denominator (JAX did the same via compact_nonzero_adv).
- completion_token_length always comes from re-tokenizing the completion text
  (TPU used vLLM token_ids when present, else a chars/3.5 heuristic).
- Legacy checkpoints store bf16 weights + tokenizer + RND state +
  trainer_state.json and --resume-from restarts Adam moments (matching the
  previous GPU/TPU handoff behavior). c8 checkpoints additionally store and
  require optimizer moments on resume. On resume, trajectory/rollout JSONLs
  and this stage's metrics rows are trimmed to the resume step so a crashed
  attempt cannot leave duplicate rows (TPU persisted rows only after the
  optimizer step).
- vLLM sync dirs are named policy_{phase}_r{round}_step_N so phases AND
  rounds of one pipeline run cannot clobber each other; pruning keeps the
  current + previous export and never deletes the dir just reloaded.
- Per-step RNG (prompt choice + request seeds) is salted by phase so scout
  and central of the same round/seed do not replay identical prompt
  sequences (the TPU trainer used one continuous stream per phase process).
- novelty_feature_max_length defaults to 18432 (= max_total_len) per the GPU
  plan; the old TrainConfig default was 10240.
- Checkpoints (and each phase's final model) can be pushed to W&B on the
  --ckpt-artifact-every cadence — the Lepton workspace has no durable disk, so this is
  what --resume-from restores after a pod loss.
"""

from __future__ import annotations

import argparse
import glob
import importlib.metadata
import json
import math
import numbers
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tmx_gpu import common  # noqa: E402
from tmx_gpu.rnd_torch import MultilayerRND, extract_novelty_features  # noqa: E402
from tmx_gpu.vllm_pool import VLLMPool, completion_failed  # noqa: E402
from tmx_jax.rewarding import analyze_completion_text, is_correct_completion  # noqa: E402

TERMINATED_FINISH_REASONS = ("stop", "eos", "eos_token", "stop_sequence")
RL_VALIDITY_MODES = ("legacy", "c8")
TRUNCATION_BASELINE_MODES = ("all_rows", "unmasked_survivors")
OPTIMIZER_STATE_FILENAME = "optimizer_state.pt"
ORIGINAL_PAPER_DAPO_VARIANT = "original_paper_dapo_100_central_lr"
C8_RND_LIFECYCLE = "c8_candidate_attempt_update"
C9_RND_LIFECYCLE = "c9_selected_batch_update"
EXPDIS_STABILIZATION_MODES = (C8_RND_LIFECYCLE, C9_RND_LIFECYCLE)
NOVELTY_NORMALIZATION_MODES = ("raw", "selected_batch_zscore")
C9_SCIENTIFIC_VARIANT = "expdis_c9_selected_batch_rnd"
C9_MULTIEXPLORER_SCIENTIFIC_VARIANT = (
    "expdis_c9_selected_batch_multi_explorer")
C9_SCREEN_VARIANT = "expdis_c9_stability_screen"
C10_SCIENTIFIC_VARIANT = "expdis_c10_survivor_consistent_overlong_filter"
C10_DYNAMIC_FAILURE_JOURNAL_VERSION = \
    "c10_dynamic_sampling_failure_journal_v1"
C9_EXECUTION_PROFILES = ("", "canary", "full")
C10_EXECUTION_PROFILES = ("", "canary")
C9_STABILIZATION_CELLS = ("", "C1", "C2", "C3", "C4", "C5")
C9_NORMALIZATION_EPSILON = 1e-6


class C10DynamicSamplingExhaustion(RuntimeError):
    """Terminal model-behavior failure after all 64 candidate attempts."""

    def __init__(self, details: Dict[str, Any], *,
                 candidate_rows: Sequence[Dict[str, Any]]):
        self.details = dict(details)
        self.candidate_rows = [dict(row) for row in candidate_rows]
        super().__init__(
            "C10 dynamic sampling exhausted its preregistered attempts")


class C10TrainingHealthFailure(RuntimeError):
    """Terminal preregistered health failure before a policy update."""

    def __init__(self, failure_kind: str, message: str,
                 details: Dict[str, Any]):
        self.failure_kind = str(failure_kind)
        self.details = dict(details)
        super().__init__(message)

# Publication-eligible C9 and DAPO jobs execute in one immutable training
# image.  The image digest alone is insufficient: the launch bootstrap adds a
# small, pinned support stack, so both identities are part of the scientific
# runtime boundary.  ``torch.__version__`` intentionally retains the CUDA
# build suffix (``+cu130``), unlike distribution metadata on some images.
CANONICAL_TRAINING_CONTAINER_IMAGE = (
    "vllm/vllm-openai:v0.25.0@sha256:"
    "fc56161ee42a011aeee78b65d0a81b6683c7d04402fd40503d14d4d6c98f07cb"
)
TRAINING_EXECUTION_ENVIRONMENT_VERSION = "training_execution_environment_v2"
CANONICAL_TRAINING_RUNTIME_STACK: Dict[str, str] = {
    "datasets": "5.0.0",
    "hf-transfer": "0.1.9",
    "huggingface-hub": "1.21.0",
    "numpy": "2.2.6",
    "requests": "2.34.2",
    "safetensors": "0.8.0",
    "tokenizers": "0.22.2",
    "torch": "2.11.0+cu130",
    "transformers": "5.13.0",
    "vllm": "0.25.0",
    "wandb": "0.28.0",
}


def observed_training_runtime_stack() -> Dict[str, str]:
    """Return the exact installed stack used by a training process."""
    observed: Dict[str, str] = {}
    for distribution in CANONICAL_TRAINING_RUNTIME_STACK:
        if distribution == "torch":
            observed[distribution] = str(torch.__version__)
            continue
        try:
            observed[distribution] = str(
                importlib.metadata.version(distribution))
        except importlib.metadata.PackageNotFoundError:
            observed[distribution] = "<missing>"
    return observed


def canonical_training_execution_environment(
    container_image_identity: str,
    orchestrator_receipt_binding: Dict[str, Any],
    native_compatibility_facts: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the signed environment, separating invariants from compatibility."""
    from tmx_gpu import orchestrator_receipt

    if str(container_image_identity) != CANONICAL_TRAINING_CONTAINER_IMAGE:
        raise ValueError(
            "strict training requires --container-image-identity exactly "
            f"{CANONICAL_TRAINING_CONTAINER_IMAGE!r}; got "
            f"{container_image_identity!r}")
    binding = orchestrator_receipt.validate_binding(
        orchestrator_receipt_binding,
        expected_container_image=CANONICAL_TRAINING_CONTAINER_IMAGE,
    )
    receipt_family = binding["receipt"]["job"]["accelerator_family"]
    native = orchestrator_receipt.validate_native_compatibility_facts(
        native_compatibility_facts,
        expected_accelerator_family=receipt_family,
    )
    return {
        "version": TRAINING_EXECUTION_ENVIRONMENT_VERSION,
        "frozen_invariants": {
            "container_image_identity": CANONICAL_TRAINING_CONTAINER_IMAGE,
            "runtime_stack": dict(CANONICAL_TRAINING_RUNTIME_STACK),
            "orchestrator_receipt_binding": binding,
        },
        "compatibility_facts": {
            "policy": "exactly_8_observed_a100_or_h100_v1",
            "allowed_accelerator_families": ["A100", "H100"],
            "native_runtime": native,
        },
    }


def validate_canonical_training_execution_environment(
    container_image_identity: str,
    orchestrator_receipt_binding: Dict[str, Any],
) -> Dict[str, Any]:
    """Fail before learner/SFT updates when the live stack has drifted."""
    if str(container_image_identity) != CANONICAL_TRAINING_CONTAINER_IMAGE:
        raise ValueError(
            "strict training requires --container-image-identity exactly "
            f"{CANONICAL_TRAINING_CONTAINER_IMAGE!r}; got "
            f"{container_image_identity!r}")
    observed = observed_training_runtime_stack()
    if observed != CANONICAL_TRAINING_RUNTIME_STACK:
        keys = sorted(set(observed) | set(CANONICAL_TRAINING_RUNTIME_STACK))
        drift = {
            key: {
                "observed": observed.get(key),
                "expected": CANONICAL_TRAINING_RUNTIME_STACK.get(key),
            }
            for key in keys
            if observed.get(key) != CANONICAL_TRAINING_RUNTIME_STACK.get(key)
        }
        raise RuntimeError(
            "canonical training runtime stack drift: "
            + json.dumps(drift, sort_keys=True))
    from tmx_gpu import orchestrator_receipt

    return canonical_training_execution_environment(
        container_image_identity,
        orchestrator_receipt_binding,
        orchestrator_receipt.observed_native_compatibility_facts(),
    )


def require_canonical_training_execution_environment_contract(
    runtime_contract: Any,
) -> Dict[str, Any]:
    """Validate the signed environment embedded in a phase runtime."""
    if not isinstance(runtime_contract, dict):
        raise ValueError("training runtime contract must be one object")
    environment = runtime_contract.get("execution_environment")
    if not isinstance(environment, dict) or environment.get("version") != \
            TRAINING_EXECUTION_ENVIRONMENT_VERSION:
        raise ValueError(
            "training runtime contract lacks the versioned execution environment")
    if set(environment) != {
            "version", "frozen_invariants", "compatibility_facts"}:
        raise ValueError("training execution environment fields mismatch")
    frozen = environment.get("frozen_invariants")
    if not isinstance(frozen, dict) or frozen.get(
            "container_image_identity") != CANONICAL_TRAINING_CONTAINER_IMAGE \
            or frozen.get("runtime_stack") != CANONICAL_TRAINING_RUNTIME_STACK:
        raise ValueError(
            "training runtime contract lacks the exact immutable "
            "container/runtime stack")
    from tmx_gpu import orchestrator_receipt

    binding = orchestrator_receipt.validate_binding(
        frozen.get("orchestrator_receipt_binding"),
        expected_container_image=CANONICAL_TRAINING_CONTAINER_IMAGE,
    )
    compatibility = environment.get("compatibility_facts")
    if not isinstance(compatibility, dict) or compatibility.get("policy") != \
            "exactly_8_observed_a100_or_h100_v1" or compatibility.get(
                "allowed_accelerator_families") != ["A100", "H100"]:
        raise ValueError("training hardware compatibility policy mismatch")
    orchestrator_receipt.validate_native_compatibility_facts(
        compatibility.get("native_runtime"),
        expected_accelerator_family=binding[
            "receipt"]["job"]["accelerator_family"],
    )
    return environment


def requires_canonical_training_execution_environment(
    args: argparse.Namespace,
) -> bool:
    """Whether this request is a strict C9/MR-ME or DAPO experiment."""
    variant = str(getattr(args, "scientific_variant", "") or "")
    return (
        variant in {
            ORIGINAL_PAPER_DAPO_VARIANT,
            C9_SCIENTIFIC_VARIANT,
            C9_MULTIEXPLORER_SCIENTIFIC_VARIANT,
            C9_SCREEN_VARIANT,
            C10_SCIENTIFIC_VARIANT,
        }
        or str(getattr(args, "c9_execution_profile", "") or "") != ""
        or str(getattr(args, "expdis_stabilization_mode", "") or "")
        == C9_RND_LIFECYCLE
    )

# Distinct per-phase RNG salt: scout and central of one round/seed must not
# replay identical prompt sequences / vLLM request seeds.
PHASE_SEED_SALT = {"scout": 0, "central": 104_729}

# Fixed *diagnostic* protocol (see module docstring): full AIME24, 4
# samples/problem, evaluation sampling, and a 32,768-token completion budget.
# It shares P1 components but is never headline-eligible; only the external
# exact 30x64 evaluation may produce the paper's AIME24 avg@64. With the
# 2,048-token prompt allowance, the serving pool must expose at least a
# 36,864-token model window.
PROBE_NUM_SAMPLES = 4
# P1 eval budget. 16384 is useless here: measured base AIME24 median
# completion is 16,426 tokens (M1 aggregate), so a 16k probe clips >50% of
# even base generations and ~all early-RL ones (observed 0/120 at step 25).
# Requires the serving pool's max_model_len >= 2048 + 32768.
PROBE_MAX_TOKENS = 32768
PROBE_TEMPERATURE = 0.6
PROBE_TOP_P = 0.95
PROBE_TOP_K = 20
PROBE_MIN_P = 0.0
PROBE_SEED_BASE = 0  # exact first-four P1 seed rule for every fixed probe
PROBE_SAMPLE_SEED_RULE = "problem_idx*100003+sample_idx"
PROBE_BENCHMARK_REVISION = "8d88b2876a82a080e2f172cc9b25d0d9d2cb4792"
PROBE_BENCHMARK_ROWS_SHA256 = (
    "33f24dfe5e519b2f90c44fbb57fb8ce4db12f1e0af457fb92b47df027fa90630")
PROBE_PROTOCOL_ID = (
    "qwen3_p1_components_t0.6_p0.95_k20_minp0_32k_avg4_diagnostic_v2")
FINAL_MODEL_ARTIFACT_BINDING_VERSION = "grpo_final_model_artifact_v1"

# GPU-MAX tier detection: contract defaults per flag; any deviation flips the
# run's config_tier from "contract" to "gpu-max:<changed-flags>".
GRPO_GPU_MAX_FLAG_DEFAULTS: Dict[str, Any] = {
    "completion_budget": 0,
    "prompts_per_step": 4,
    "ppo_mode": False,
    "updates_per_rollout": 1,
    "optimizer": "adamw",
    # The measured TPU reference used soft-overlong shaping without masking
    # truncated rows out of the loss. Mask-on is a labeled ablation/c7 mode.
    "mask_truncated": False,
    "truncation_baseline_mode": "all_rows",
    # C9 is a labeled stabilization experiment, never an invisible change to
    # the frozen C8 path.
    "expdis_stabilization_mode": C8_RND_LIFECYCLE,
    "novelty_normalization": "raw",
}


def compute_config_tier(args: argparse.Namespace, flag_defaults: Dict[str, Any]) -> str:
    """"contract" if every GPU-MAX flag is at its contract default, else
    "gpu-max:<comma list of changed flags>"."""
    changed = [name for name, default in flag_defaults.items()
               if getattr(args, name, default) != default]
    return "contract" if not changed else "gpu-max:" + ",".join(sorted(changed))


def resolved_config_tier(args: argparse.Namespace) -> str:
    """Single source of truth for logs, step checkpoints, final state, resume."""
    tier = compute_config_tier(args, GRPO_GPU_MAX_FLAG_DEFAULTS)
    variant = str(getattr(args, "scientific_variant", "") or "")
    # A scientific label is an additional dimension, not a replacement for
    # the concrete optimizer/budget tier.  Replacing ``gpu-max:...`` with the
    # label previously hid runtime drift in external continuations.
    return f"{tier}+scientific-variant:{variant}" if variant else tier


GRPO_RUNTIME_CONTRACT_VERSION = "grpo_runtime_contract_v3"
C10_OVERLONG_FILTER_CONTRACT_VERSION = \
    "survivor_consistent_overlong_filter_v1"
REWARD_SCORING_TEXT_SCOPE = "full_completion_v1"
TRAJECTORY_RECORD_SCHEMA_VERSION = "gpu_grpo_trajectory_record_v2"
EXTERNAL_LINEAGE_CONTRACT_VERSION = "external_sft_central_lineage_v3"
FRESH_C8_LINEAGE_CONTRACT_VERSION = "fresh_c8_sft_central_lineage_v1"
FRESH_C8_EARLY_SELECTION_VARIANT = "fresh_c8_sft_exposure_gated"


def build_runtime_contract(
    args: argparse.Namespace, *, resolved_lr: float, resolved_steps: int,
) -> Dict[str, Any]:
    """Exact, JSON-stable GRPO execution contract after budget resolution.

    This is deliberately broader than ``config_tier``.  The tier is a useful
    human label, while this object is the machine equality boundary used for
    external-Central handoff, checkpoint resume, and completed-final reuse.
    Callers must invoke :func:`apply_completion_budget` and the probe serving
    length resolver first.
    """
    from tmx_jax.data import training_dataset_identity
    from tmx_jax.prompting import prompt_metadata

    contract = {
        "version": GRPO_RUNTIME_CONTRACT_VERSION,
        "phase": str(args.phase),
        "scientific_variant": str(getattr(args, "scientific_variant", "") or ""),
        "config_tier": resolved_config_tier(args),
        "model_init": str(args.model),
        "base_model_revision": str(
            getattr(args, "model_revision", "") or ""),
        "expected_steps": int(resolved_steps),
        "resolved_learning_rate": float(resolved_lr),
        "optimizer": str(args.optimizer),
        "adam_beta1": float(args.adam_beta1),
        "adam_beta2": float(args.adam_beta2),
        "adam_eps": float(args.adam_eps),
        "weight_decay": float(args.weight_decay),
        "grad_clip": float(args.grad_clip),
        "parameter_dtype": str(args.param_dtype),
        "rl_validity_mode": rl_validity_mode(args),
        "completion_budget_flag": int(getattr(args, "completion_budget", 0) or 0),
        "resolved_max_prompt_len": int(args.max_prompt_len),
        "resolved_max_completion_len": int(args.max_completion_len),
        "resolved_vllm_max_model_len": int(args.vllm_max_model_len),
        "reward_scoring_text_scope": REWARD_SCORING_TEXT_SCOPE,
        "trajectory_record_schema_version": TRAJECTORY_RECORD_SCHEMA_VERSION,
        "soft_overlong_expected_len": int(args.soft_overlong_expected_len),
        "soft_overlong_cache_len": int(args.soft_overlong_cache_len),
        "loss_denominator_len": int(args.loss_denominator_len),
        "loss_normalization": "dr_grpo_fixed_denominator",
        "advantage_standardization": False,
        "kl_beta": 0.0,
        "prompts_per_step": int(args.prompts_per_step),
        "num_generations": int(args.num_generations),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "top_k": int(args.top_k),
        "training_prompt": prompt_metadata(),
        "dynamic_sampling": True,
        "dynamic_max_attempts": int(args.dynamic_max_attempts),
        "mask_truncated": bool(args.mask_truncated),
        "clip_epsilon_low": float(args.clip_epsilon_low),
        "clip_epsilon_high": float(args.clip_epsilon_high),
        "ppo_mode": bool(args.ppo_mode),
        "updates_per_rollout": int(args.updates_per_rollout),
        "incorrect_novelty_scale": float(args.incorrect_novelty_scale),
        "lambda_novelty": (
            0.0 if str(args.phase) == "central" else float(args.lambda_novelty)
        ),
        "dataset": str(args.dataset),
        "training_dataset_identity": training_dataset_identity(args.dataset),
        "max_train_examples": int(args.max_train_examples),
        "seed": int(args.seed),
        "round": int(args.round),
        "max_consecutive_generation_failures": int(
            args.max_consecutive_gen_failures),
        "generation_timeout_seconds": float(args.gen_timeout),
        "generation_concurrency_per_server": int(args.gen_concurrency),
        "chunk_tokens": int(args.chunk_tokens),
    }
    if requires_canonical_training_execution_environment(args):
        image_identity = str(
            getattr(args, "container_image_identity", "") or "")
        if image_identity != CANONICAL_TRAINING_CONTAINER_IMAGE:
            raise ValueError(
                "strict training requires --container-image-identity exactly "
                f"{CANONICAL_TRAINING_CONTAINER_IMAGE!r}; got "
                f"{image_identity!r}")
        environment = getattr(args, "_training_execution_environment", None)
        if environment is None:
            code_binding = load_code_artifact_binding(args)
            receipt_binding = load_orchestrator_receipt_binding(
                args, code_artifact_binding=code_binding)
            from tmx_gpu import orchestrator_receipt

            environment = canonical_training_execution_environment(
                image_identity,
                receipt_binding,
                orchestrator_receipt.observed_native_compatibility_facts(),
            )
        contract["execution_environment"] = environment
        require_canonical_training_execution_environment_contract(contract)
    multi_contract_sha = str(
        getattr(args, "mrme_contract_sha256", "") or "")
    if multi_contract_sha:
        single_scout_multiround = bool(getattr(
            args, "single_scout_multiround", False))
        multi_binding = {
            "contract_sha256": multi_contract_sha,
            "method": str(getattr(args, "mrme_method", "") or ""),
            "stabilization_profile": str(
                getattr(args, "mrme_stabilization_profile", "") or ""),
            "scout_index": int(getattr(args, "mrme_scout_index", 0) or 0),
            "scouts_per_round": int(
                getattr(args, "mrme_scouts_per_round", 0) or 0),
            "shard_manifest_sha256": str(
                getattr(args, "mrme_shard_manifest_sha256", "") or ""),
        }
        if single_scout_multiround:
            multi_binding.update({
                "method_version": str(getattr(
                    args, "mrme_method_version", "") or ""),
                "single_scout_multiround": True,
                "mrme": False,
                "multi_explorer": False,
            })
        contract["multi_explorer_contract"] = multi_binding
    if str(getattr(args, "c9_execution_profile", "") or ""):
        contract.update({
            "c9_execution_profile": str(args.c9_execution_profile),
            "stabilization_cell": str(args.stabilization_cell),
            "probe_contract": {
                "cadence_steps": int(args.probe_every),
                "dataset": str(args.probe_dataset),
                "protocol_id": PROBE_PROTOCOL_ID,
                "headline_eligible": False,
                "metric_scale": "fraction",
                "num_samples_per_problem": PROBE_NUM_SAMPLES,
                "max_completion_tokens": PROBE_MAX_TOKENS,
                "temperature": PROBE_TEMPERATURE,
                "top_p": PROBE_TOP_P,
                "top_k": PROBE_TOP_K,
                "min_p": PROBE_MIN_P,
                "sample_seed_rule": PROBE_SAMPLE_SEED_RULE,
                "seed_base": PROBE_SEED_BASE,
                "benchmark_revision": PROBE_BENCHMARK_REVISION,
                "benchmark_rows_sha256": PROBE_BENCHMARK_ROWS_SHA256,
                "prompt_renderer": "tmx_gpu.eval_gpu.render_prompt:auto",
                "grader": "tmx_gpu.eval_gpu.grade",
            },
        })
    if survivor_consistent_overlong_filter(args):
        # C10 deliberately disables VLLMPool's legacy call-count breaker.  Its
        # preregistered transport decision is instead made from the exact
        # generated-row error rate (training and the fixed behavior panel both
        # use the same 5% boundary).  Do not leave the generic three-call
        # field in this signed contract: it would describe a guard that the
        # C10 callers explicitly do not execute.
        contract.pop("max_consecutive_generation_failures", None)
        contract["generation_failure_policy"] = {
            "owner": "c10_preregistered_generation_error_row_rate",
            "legacy_consecutive_failure_limit_enforced": False,
            "maximum_generation_error_row_rate": 0.05,
            "excess_classification": "transport_invalid",
            "dynamic_sampling_attempt_limit": 64,
        }
        contract["survivor_overlong_filter_contract"] = {
            "version": C10_OVERLONG_FILTER_CONTRACT_VERSION,
            "c10_execution_profile": str(args.c10_execution_profile),
            "truncation_baseline_mode": "unmasked_survivors",
            "dynamic_sampling_eligibility_population": "unmasked_survivors",
            "minimum_dynamic_sampling_survivors": 2,
            "masked_row_advantage": 0.0,
            "masked_rows_in_policy_forward": False,
            "masked_rows_in_policy_backward": False,
            "loss_denominator_population": "full_sampled_rows",
            "rnd_update_population": "all_selected_raw_rows",
            "novelty_normalization_population": "all_selected_raw_scores",
        }
        contract["c10_execution_profile"] = str(args.c10_execution_profile)
    if str(args.phase) == "scout":
        # These values affect the Scout's exploration signal but are irrelevant
        # to Central GRPO.  Keeping them Scout-only strengthens fresh-c8
        # lineage without changing the frozen external-Central v3 contract.
        contract["scout_novelty_contract"] = {
            "novelty_layers": str(args.novelty_layers),
            "rnd_learning_rate": float(args.rnd_lr),
            "rnd_hidden_dim": int(args.rnd_hidden),
            "novelty_feature_max_length": int(args.novelty_feature_max_length),
            "incorrect_novelty_scale": float(args.incorrect_novelty_scale),
        }
        if expdis_stabilization_mode(args) == C9_RND_LIFECYCLE:
            # C8's runtime object stays byte-for-byte schema-compatible.  These
            # fields exist only on the explicitly labeled C9 path.
            contract["scout_novelty_contract"].update({
                "rnd_lifecycle": C9_RND_LIFECYCLE,
                "rnd_scoring_predictor_state": "frozen_for_entire_learner_step",
                "rnd_update_scope": "final_selected_learner_rows",
                "rnd_updates_per_completed_step": 1,
                "novelty_normalization": novelty_normalization_mode(args),
                "novelty_normalization_population": "all_selected_raw_scores",
                "novelty_gate_order": "normalize_then_correct_only_gate",
                "novelty_normalization_epsilon": C9_NORMALIZATION_EPSILON,
                "dynamic_sampling_eligibility_novelty": "raw_correct_gated",
                "stabilization_cell": str(
                    getattr(args, "stabilization_cell", "") or ""),
            })
    return contract


def load_external_lineage(
    args: argparse.Namespace,
    *, expected_runtime_contract: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any] | None:
    path = str(getattr(args, "external_lineage_json", "") or "")
    if not path:
        return None
    if str(args.phase) != "central":
        raise ValueError("--external-lineage-json is valid only for Central GRPO")
    with open(path, encoding="utf-8") as handle:
        lineage = json.load(handle)
    if not isinstance(lineage, dict):
        raise ValueError("external lineage must be a JSON object")
    lineage_version = lineage.get("lineage_contract_version")
    # Import only when an external handoff is actually requested.  The C9
    # lineage schema is versioned in one place; hard-coding its former v1
    # marker here made the v2 evidence emitted by pipeline_gpu impossible to
    # consume at the SFT -> Central boundary.
    from tmx_gpu import c9_evidence

    c9_lineage = lineage_version == c9_evidence.CENTRAL_LINEAGE_VERSION
    recorded = lineage.get(
        "evidence_sha256" if c9_lineage else "lineage_sha256")
    unsigned = dict(lineage)
    unsigned.pop("evidence_sha256" if c9_lineage else "lineage_sha256", None)
    if recorded != common.canonical_json_sha256(unsigned):
        raise ValueError("external lineage SHA256 is missing or invalid")
    if lineage_version == EXTERNAL_LINEAGE_CONTRACT_VERSION:
        if lineage.get("scientific_variant") != "external_sft_exposure_gated":
            raise ValueError("external lineage scientific variant mismatch")
        if str(getattr(args, "scientific_variant", "")) != \
                "external_sft_exposure_gated":
            raise ValueError(
                "external lineage requires the noncanonical scientific variant")
    elif lineage_version == FRESH_C8_LINEAGE_CONTRACT_VERSION:
        selected_step = int(lineage.get("selected_optimizer_step", -1))
        canonical = lineage.get("canonical") is True
        expected_variant = "" if selected_step == 1000 \
            else FRESH_C8_EARLY_SELECTION_VARIANT
        if selected_step not in (125, 250, 500, 1000):
            raise ValueError("fresh-c8 lineage selected an unregistered SFT step")
        if canonical != (selected_step == 1000):
            raise ValueError("fresh-c8 lineage canonical flag disagrees with selected step")
        if lineage.get("scientific_variant") != expected_variant or \
                str(getattr(args, "scientific_variant", "")) != expected_variant:
            raise ValueError("fresh-c8 lineage scientific variant mismatch")
    elif c9_lineage:
        c9_evidence.validate_central_lineage(
            lineage, expected_runtime_contract=expected_runtime_contract)
        if str(getattr(args, "scientific_variant", "") or "") != \
                c9_evidence.C9_SCIENTIFIC_VARIANT:
            raise ValueError("C9 lineage scientific variant mismatch")
    elif lineage_version == "c9_multi_explorer_sft_central_lineage_v1":
        from tmx_gpu import mrme_contract

        contract = mrme_contract.validate_contract(
            _load_required_json_object(
                str(getattr(args, "mrme_contract_json", "") or ""),
                "C9 multi-explorer campaign contract"))
        if contract.get("contract_sha256") != str(
                getattr(args, "mrme_contract_sha256", "") or ""):
            raise ValueError("C9 multi-explorer lineage campaign SHA mismatch")
        provenance = _load_required_json_object(
            os.path.join(str(args.model), "sft_provenance.json"),
            "C9 multi-explorer SFT provenance")
        source = provenance.get("trajectory_source")
        mrme_contract.validate_c9_multi_explorer_central_lineage(
            lineage,
            contract=contract,
            round_index=int(args.round),
            expected_shard_manifest_sha256=str(
                getattr(args, "mrme_shard_manifest_sha256", "") or ""),
            expected_trajectory_source_digest=(
                source.get("digest") if isinstance(source, dict) else None),
            expected_accepted_digest=provenance.get(
                "accepted_library_digest"),
            expected_central_runtime_contract=expected_runtime_contract,
        )
        if str(getattr(args, "scientific_variant", "") or "") != \
                mrme_contract.C9_MULTIEXPLORER_SCIENTIFIC_VARIANT:
            raise ValueError("C9 multi-explorer lineage scientific variant mismatch")
        common.verify_directory_file_manifest(
            str(args.model), lineage["sft_model_file_manifest"],
            exclude_relative_paths=("sft_provenance.json",))
        common.verify_directory_file_manifest(
            str(args.model),
            lineage["sft_model_artifact"]["model_file_manifest"])
    else:
        raise ValueError("external lineage contract version mismatch")
    if expected_runtime_contract is None:
        raise ValueError("external lineage requires a resolved Central runtime contract")
    if lineage.get("central_runtime_contract") != expected_runtime_contract:
        raise ValueError(
            "external lineage Central runtime contract differs from resolved GRPO args")
    if int(expected_runtime_contract.get("expected_steps", -1)) != 100:
        raise ValueError("external SFT continuation requires exactly 100 Central steps")
    return lineage


def load_code_source_manifest(path: str) -> Dict[str, Any] | None:
    """Independently re-hash and normalize one executing-code artifact."""
    if not path:
        return None
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("code source manifest must be a JSON object")
    validated = common.validate_artifact_download_manifest(raw)
    common.require_immutable_wandb_artifact_ref(
        validated["source"]["artifact_ref"], field="fresh-c8 CODE_ARTIFACT")
    return common.stable_artifact_download_identity(validated)


def load_code_artifact_binding(args: argparse.Namespace) -> Dict[str, Any] | None:
    """Load code identity for any strict Scout/SFT/Central training phase."""
    path = str(getattr(args, "code_source_manifest", "") or "")
    if not path:
        return None
    phase = str(args.phase)
    variant = str(getattr(args, "scientific_variant", "") or "")
    external_lineage = str(
        getattr(args, "external_lineage_json", "") or "")
    if phase == "central" and variant == ORIGINAL_PAPER_DAPO_VARIANT and \
            external_lineage:
        raise ValueError(
            "the original-paper DAPO Central-route is lineage-free; "
            "--external-lineage-json would change its scientific identity")
    code_bound_dapo = (
        phase == "central" and variant == ORIGINAL_PAPER_DAPO_VARIANT
        and not external_lineage)
    if phase != "scout" and not code_bound_dapo and not \
            requires_canonical_training_execution_environment(args):
        raise ValueError(
            "--code-source-manifest is valid only for strict training, Scout "
            "GRPO, or the original-paper DAPO Central-route control")
    return load_code_source_manifest(path)


def load_orchestrator_receipt_binding(
    args: argparse.Namespace,
    *, code_artifact_binding: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Load and cross-bind the orchestrator receipt to this phase request."""
    from tmx_gpu import orchestrator_receipt

    path = str(getattr(args, "orchestrator_receipt_binding", "") or "")
    if code_artifact_binding is None:
        code_artifact_binding = load_code_artifact_binding(args)
    if code_artifact_binding is None:
        raise ValueError(
            "strict training requires --code-source-manifest before receipt "
            "validation")
    source = code_artifact_binding["source"]
    return orchestrator_receipt.load_binding(
        path,
        expected_run_name=str(getattr(
            args, "orchestrator_run_name", "") or getattr(
                args, "run_name", "") or ""),
        expected_container_image=str(
            getattr(args, "container_image_identity", "") or ""),
        expected_code_artifact_ref=str(source["artifact_ref"]),
        expected_code_artifact_digest=str(source["artifact_digest"]),
        expected_command_sha256=str(getattr(
            args, "orchestrator_command_sha256", "") or ""),
        expected_polling_artifact_ref=str(getattr(
            args, "orchestrator_receipt_artifact", "") or ""),
    )


def validate_code_bound_local_checkpoints(
    args: argparse.Namespace, current: Dict[str, Any] | None,
) -> None:
    """Fail before model loading if fresh code-bound checkpoints are unsafe."""
    if current is None:
        return
    checkpoints = sorted(glob.glob(os.path.join(
        str(args.output_dir), "step_[0-9]*")))
    for checkpoint in checkpoints:
        state_path = os.path.join(checkpoint, "trainer_state.json")
        if not os.path.isfile(state_path):
            raise RuntimeError(
                "code-bound Scout checkpoint exists but trainer_state is missing")
        with open(state_path, encoding="utf-8") as handle:
            state = json.load(handle)
        if state.get("code_artifact_binding") is None:
            raise RuntimeError(
                "code-bound Scout checkpoint exists but code binding is missing")
        if state["code_artifact_binding"] != current:
            raise RuntimeError(
                "code-bound Scout checkpoint differs from current code artifact")
    if checkpoints and not str(getattr(args, "resume_from", "") or ""):
        raise RuntimeError(
            "code-bound Scout checkpoints exist but --resume-from is missing")


def apply_completion_budget(args: argparse.Namespace) -> argparse.Namespace:
    """GPU-MAX --completion-budget N: rescale completion/overlong/model-len.

    soft-overlong splits as expected_len=int(0.8*N), cache_len=N-expected_len
    (contract 16384 -> 13107/3277 is exactly this split). Mutates args.
    """
    budget = int(getattr(args, "completion_budget", 0) or 0)
    if budget > 0:
        args.max_completion_len = budget
        args.soft_overlong_expected_len = int(0.8 * budget)
        args.soft_overlong_cache_len = budget - args.soft_overlong_expected_len
        # RND must see the full reasoning context (paper semantics): scale the
        # novelty feature window with the budget, never below it.
        args.novelty_feature_max_length = max(
            int(getattr(args, "novelty_feature_max_length", 0) or 0), budget)
        args.vllm_max_model_len = max(
            int(getattr(args, "vllm_max_model_len", 0) or 0),
            int(args.max_prompt_len) + budget,
        )
        # NOTE: the Dr.GRPO loss denominator (--loss-denominator-len) is
        # deliberately NOT scaled with the budget — it is an arbitrary
        # length-bias-removing constant, and pinning it at the contract 16384
        # preserves gradient scale parity across budget tiers and vs the TPU
        # runs (otherwise a 32k budget silently halves the effective LR).
        print(f"[grpo] GPU-MAX completion budget {budget}: soft-overlong "
              f"{args.soft_overlong_expected_len}/{args.soft_overlong_cache_len} "
              f"max_model_len {args.vllm_max_model_len} "
              f"novelty_feat_len {args.novelty_feature_max_length} "
              f"loss_denom_len {args.loss_denominator_len}", flush=True)
    return args


def enforce_probe_serving_len(args: argparse.Namespace,
                              pool: Optional["VLLMPool"] = None) -> None:
    """Frozen probe protocol sizing: with --probe-every > 0 the serving pool
    must fit max_prompt_len + PROBE_MAX_TOKENS, or every probe request 400s
    and the probe silently degrades to error rows.

    - Own managed pool (we will launch it): auto-raise --vllm-max-model-len.
    - Caller-provided managed pool: verify its actual max_model_len.
    - Attach-mode (endpoints we cannot resize): fail fast with instructions.
    """
    if int(args.probe_every) <= 0:
        return
    required = int(args.max_prompt_len) + PROBE_MAX_TOKENS
    hint = (f"probes need serving max_model_len >= {required} "
            f"(max_prompt_len {args.max_prompt_len} + PROBE_MAX_TOKENS "
            f"{PROBE_MAX_TOKENS}); raise the pool's max-model-len "
            "(expdis_job.sh MAX_MODEL_LEN) or set --probe-every 0")
    if pool is not None:
        pool_len = int(pool.max_model_len) if pool.managed else int(args.vllm_max_model_len)
        if pool_len < required:
            raise ValueError(f"serving pool max_model_len={pool_len} too small: {hint}")
    elif args.endpoints:
        if int(args.vllm_max_model_len) < required:
            raise ValueError(
                f"attach-mode pool (--vllm-max-model-len {args.vllm_max_model_len}) "
                f"too small: {hint}; relaunch the servers with a larger "
                "--max-model-len and pass the matching --vllm-max-model-len")
    elif int(args.vllm_max_model_len) < required:
        print(f"[grpo] raising --vllm-max-model-len {args.vllm_max_model_len} -> "
              f"{required} (frozen probe protocol)", flush=True)
        args.vllm_max_model_len = required


def rl_validity_mode(args: argparse.Namespace) -> str:
    """Resolved RL validity contract, tolerant of pre-c8 Namespace callers."""
    mode = str(getattr(args, "rl_validity_mode", "legacy") or "legacy")
    if mode not in RL_VALIDITY_MODES:
        raise ValueError(
            f"unsupported rl_validity_mode={mode!r}; expected one of "
            f"{', '.join(RL_VALIDITY_MODES)}")
    return mode


def truncation_baseline_mode(args: argparse.Namespace) -> str:
    """Resolve the group-mean population for truncated-loss masking."""
    mode = str(getattr(args, "truncation_baseline_mode", "all_rows")
               or "all_rows")
    if mode not in TRUNCATION_BASELINE_MODES:
        raise ValueError(
            f"unsupported truncation_baseline_mode={mode!r}; expected one of "
            f"{', '.join(TRUNCATION_BASELINE_MODES)}")
    return mode


def survivor_consistent_overlong_filter(args: argparse.Namespace) -> bool:
    """Whether the exact preregistered C10 filtering semantics are active."""
    return (
        bool(getattr(args, "mask_truncated", False))
        and truncation_baseline_mode(args) == "unmasked_survivors"
        and str(getattr(args, "scientific_variant", "") or "")
        == C10_SCIENTIFIC_VARIANT
        and str(getattr(args, "c10_execution_profile", "") or "")
        == "canary"
    )


def c10_exact_numeric(value: Any, expected: int | float) -> bool:
    """Exact finite numeric equality for C10 discrete contract fields."""
    # Runtime tensors/statistics can legitimately materialize their scalar
    # counts as NumPy integer/float scalars.  Accept the numeric protocol, but
    # keep booleans (including NumPy bools) and non-finite/fractional drifts
    # out of exact C10 contract fields.
    return isinstance(value, numbers.Real) and not isinstance(value, bool) \
        and not isinstance(value, np.bool_) \
        and math.isfinite(float(value)) \
        and float(value) == float(expected)


def c10_integral_numeric(value: Any, *, minimum: int = 0,
                         multiple_of: int = 1) -> bool:
    """Return whether a C10 counter is a finite integral numeric scalar."""
    if not isinstance(value, numbers.Real) or isinstance(value, bool) or \
            isinstance(value, np.bool_) or not math.isfinite(float(value)):
        return False
    numeric = float(value)
    return numeric.is_integer() and numeric >= minimum and \
        int(numeric) % multiple_of == 0


def validate_c10_selected_batch_before_update(
    args: argparse.Namespace, selected_rows: Sequence[Dict[str, Any]],
    dynamic_stats: Dict[str, Any],
) -> None:
    """Fail durably before policy update on impossible C10 geometry."""
    if str(getattr(args, "c10_execution_profile", "") or "") != "canary":
        return
    expected_rows = int(args.prompts_per_step) * int(args.num_generations)
    if not c10_exact_numeric(
            dynamic_stats.get("kept_group_count"),
            int(args.prompts_per_step)) or not c10_exact_numeric(
                dynamic_stats.get("fallback_group_count"), 0):
        raise C10TrainingHealthFailure(
            "selected_group_geometry_failure",
            "C10 selected-batch group geometry is invalid before update", {
                "kept_group_count": dynamic_stats.get("kept_group_count"),
                "fallback_group_count": dynamic_stats.get(
                    "fallback_group_count"),
            })
    if not c10_exact_numeric(
            dynamic_stats.get("selected_raw_row_count"), expected_rows) or \
            len(selected_rows) != expected_rows:
        raise C10TrainingHealthFailure(
            "survivor_filter_contract_failure",
            "C10 selected raw-row geometry is invalid before update", {
                "selected_raw_row_count": dynamic_stats.get(
                    "selected_raw_row_count"),
                "materialized_selected_row_count": len(selected_rows),
            })
    for group_start in range(0, expected_rows, int(args.num_generations)):
        group = selected_rows[group_start:group_start + int(
            args.num_generations)]
        survivors = sum(
            not bool(row.get("clipped", False)) for row in group)
        populations = sorted(set(str(row.get(
            "dynamic_sampling_eligibility_population")) for row in group))
        if survivors < 2 or populations != ["unmasked_survivors"]:
            raise C10TrainingHealthFailure(
                "survivor_filter_contract_failure",
                "C10 selected a group outside survivor-consistent "
                "eligibility before update", {
                    "group_start": group_start,
                    "survivors": survivors,
                    "populations": populations,
                })

        def integrity_failure(message: str, details: Dict[str, Any]) -> None:
            """Stop before mutation on an implementation-evidence defect."""
            raise C10TrainingHealthFailure(
                "selected_batch_integrity_failure", message, {
                    "group_start": group_start, **details})

        problem_ids = {str(row.get("problem_id", "")) for row in group}
        prompts = {str(row.get("prompt_text", "")) for row in group}
        if len(problem_ids) != 1 or "" in problem_ids or \
                len(prompts) != 1 or "" in prompts:
            integrity_failure(
                "C10 selected group does not contain exactly one prompt", {
                    "problem_id_count": len(problem_ids),
                    "prompt_count": len(prompts),
                    "has_empty_problem_id": "" in problem_ids,
                    "has_empty_prompt": "" in prompts,
                })

        for row_offset, row in enumerate(group):
            completion = row.get("completion_text")
            ground_truth = row.get("ground_truth")
            true_len_raw = row.get("completion_token_length_true")
            scored_len_raw = row.get("completion_token_length")
            if not isinstance(completion, str) or not isinstance(
                    ground_truth, str) or not c10_integral_numeric(
                        true_len_raw) or not c10_integral_numeric(
                            scored_len_raw):
                integrity_failure(
                    "C10 selected row lacks replayable raw evidence", {
                        "row_offset": row_offset,
                        "completion_is_text": isinstance(completion, str),
                        "ground_truth_is_text": isinstance(
                            ground_truth, str),
                    })
            true_len = int(float(true_len_raw))
            scored_len = int(float(scored_len_raw))
            expected_clipped = (
                str(row.get("finish_reason", "")).lower() == "length"
                or true_len >= 16384)
            expected_correct = is_correct_completion(completion, ground_truth)
            expected_valid = bool(analyze_completion_text(completion).get(
                "valid_answer_presentation", False))
            if scored_len != min(true_len, 16384) or \
                    type(row.get("clipped")) is not bool or bool(row.get(
                        "clipped")) != expected_clipped or type(row.get(
                            "is_correct")) is not bool or bool(row.get(
                                "is_correct")) != expected_correct or type(
                                    row.get("valid_answer")) is not bool or \
                    bool(row.get("valid_answer")) != expected_valid:
                integrity_failure(
                    "C10 selected row flags do not replay from raw evidence", {
                        "row_offset": row_offset,
                        "expected_clipped": expected_clipped,
                        "expected_correct": expected_correct,
                        "expected_valid": expected_valid,
                    })
            # This gate runs on the trainer's raw selected-row schema.  The
            # public ``c10_rnd_raw_novelty_score`` name is introduced only by
            # trajectory_record() after a successful policy update.
            raw_novelty = row.get("novelty_raw")
            try:
                raw_novelty_float = float(raw_novelty)
            except (TypeError, ValueError):
                raw_novelty_float = math.nan
            if not math.isfinite(raw_novelty_float):
                integrity_failure(
                    "C10 selected row raw novelty is not finite", {
                        "row_offset": row_offset})
            expected_correctness = 1.0 if expected_correct else -1.0
            expected_eligibility_novelty = (
                raw_novelty_float if expected_correct else 0.0)
            expected_eligibility_reward = (
                expected_correctness + 0.5 * expected_eligibility_novelty
                + common.soft_overlong_penalty(
                    scored_len, 13107, 3277))
            numeric_expectations = {
                "correctness_reward": expected_correctness,
                "dynamic_sampling_eligibility_novelty_used":
                    expected_eligibility_novelty,
                "dynamic_sampling_eligibility_blended_reward":
                    expected_eligibility_reward,
            }
            for field, expected in numeric_expectations.items():
                try:
                    observed = float(row.get(field, math.nan))
                except (TypeError, ValueError):
                    observed = math.nan
                if not math.isfinite(observed) or not math.isclose(
                        observed, expected, rel_tol=1e-7, abs_tol=1e-7):
                    integrity_failure(
                        "C10 selected row reward evidence does not replay", {
                            "row_offset": row_offset,
                            "field": field,
                            "expected": expected,
                            "observed": observed,
                        })

        replay_rows = [{
            "is_correct": bool(row["is_correct"]),
            "clipped": bool(row["clipped"]),
            "blended_reward": float(row[
                "dynamic_sampling_eligibility_blended_reward"]),
            "novelty_used": float(row[
                "dynamic_sampling_eligibility_novelty_used"]),
        } for row in group]
        reason, eligible, facts = group_eligibility_for_rows(
            replay_rows, lambda_novelty=0.5, validity_mode="c8",
            survivor_only=True)
        if not eligible:
            integrity_failure(
                "C10 dynamic sampling returned an ineligible group", {
                    "replayed_reason": reason,
                    "replayed_survivor_count": facts["survivor_count"],
                    "replayed_survivor_correct_count": facts[
                        "survivor_correct_count"],
                })
        exact_fields = {
            "dynamic_sampling_raw_group_size": facts["raw_group_size"],
            "dynamic_sampling_survivor_count": facts["survivor_count"],
            "dynamic_sampling_masked_count": facts["masked_count"],
            "dynamic_sampling_raw_correct_count": facts[
                "raw_correct_count"],
            "dynamic_sampling_survivor_correct_count": facts[
                "survivor_correct_count"],
            "dynamic_sampling_eligibility_population_size": facts[
                "eligibility_population_size"],
        }
        for row_offset, row in enumerate(group):
            for field, expected in exact_fields.items():
                if not c10_exact_numeric(row.get(field), expected):
                    integrity_failure(
                        "C10 selected group count metadata does not replay", {
                            "row_offset": row_offset,
                            "field": field,
                            "expected": expected,
                            "observed": row.get(field),
                        })
            if row.get("dynamic_sampling_reason") != reason or \
                    row.get("dynamic_sampling_group_selected") is not True or \
                    row.get("dynamic_sampling_policy_loss_eligible") is not \
                    (not bool(row["clipped"])):
                integrity_failure(
                    "C10 selected group eligibility metadata does not replay", {
                        "row_offset": row_offset,
                        "expected_reason": reason,
                    })
            for field, expected in (
                    ("dynamic_sampling_reward_std",
                     facts["eligibility_reward_std"]),
                    ("dynamic_sampling_novelty_contribution_std",
                     facts["eligibility_novelty_contribution_std"])):
                try:
                    observed = float(row.get(field, math.nan))
                except (TypeError, ValueError):
                    observed = math.nan
                if not math.isfinite(observed) or not math.isclose(
                        observed, float(expected),
                        rel_tol=1e-7, abs_tol=1e-7):
                    integrity_failure(
                        "C10 selected group dispersion metadata does not "
                        "replay", {
                            "row_offset": row_offset,
                            "field": field,
                            "expected": float(expected),
                            "observed": observed,
                        })


def validate_overlong_filter_contract(args: argparse.Namespace) -> None:
    """Fail closed on partial or mislabeled survivor filtering.

    Legacy ``mask + all_rows`` remains available only for exact c7 replay.
    Under c8, masking is permitted solely as the separately preregistered C10
    experiment, where eligibility, baselines, and gradients all use the same
    nontruncated survivor population.
    """
    mode = truncation_baseline_mode(args)
    mask = bool(getattr(args, "mask_truncated", False))
    validity = rl_validity_mode(args)
    variant = str(getattr(args, "scientific_variant", "") or "")
    c10_profile = str(getattr(args, "c10_execution_profile", "") or "")
    c10_label = variant == C10_SCIENTIFIC_VARIANT or bool(c10_profile)

    if c10_label:
        drift = []
        if variant != C10_SCIENTIFIC_VARIANT:
            drift.append(
                f"scientific_variant={variant!r} "
                f"(expected {C10_SCIENTIFIC_VARIANT!r})")
        if c10_profile != "canary":
            drift.append(
                f"c10_execution_profile={c10_profile!r} "
                "(expected 'canary')")
        if validity != "c8":
            drift.append(f"rl_validity_mode={validity!r} (expected 'c8')")
        if not mask:
            drift.append("mask_truncated=False (expected True)")
        if mode != "unmasked_survivors":
            drift.append(
                f"truncation_baseline_mode={mode!r} "
                "(expected 'unmasked_survivors')")
        if drift:
            raise ValueError("C10 Overlong Filtering contract drift: "
                             + "; ".join(drift))

    if validity == "c8" and mask and not c10_label:
        raise ValueError(
            "c8 --mask-truncated is permitted only for the separately "
            "preregistered C10 survivor-consistent Overlong Filtering "
            f"variant {C10_SCIENTIFIC_VARIANT!r}; all-row masking is the "
            "measured c7 length-ratchet defect")
    if mode == "unmasked_survivors" and not c10_label:
        raise ValueError(
            "--truncation-baseline-mode unmasked_survivors requires the exact "
            "C10 scientific variant and stabilization cell")
    if mode == "unmasked_survivors" and not mask:
        raise ValueError(
            "unmasked_survivors is meaningful only with --mask-truncated")


def expdis_stabilization_mode(args: argparse.Namespace) -> str:
    """Resolve the RND lifecycle while preserving pre-C9 callers as C8."""
    mode = str(getattr(args, "expdis_stabilization_mode", C8_RND_LIFECYCLE)
               or C8_RND_LIFECYCLE)
    if mode not in EXPDIS_STABILIZATION_MODES:
        raise ValueError(
            f"unsupported expdis_stabilization_mode={mode!r}; expected one of "
            f"{', '.join(EXPDIS_STABILIZATION_MODES)}")
    return mode


def novelty_normalization_mode(args: argparse.Namespace) -> str:
    """Resolve novelty scaling while preserving frozen C8 raw rewards."""
    mode = str(getattr(args, "novelty_normalization", "raw") or "raw")
    if mode not in NOVELTY_NORMALIZATION_MODES:
        raise ValueError(
            f"unsupported novelty_normalization={mode!r}; expected one of "
            f"{', '.join(NOVELTY_NORMALIZATION_MODES)}")
    return mode


def _load_required_json_object(path: str, label: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise ValueError(f"{label} is missing: {path!r}")
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return value


def _validate_c9_single_central_init(
    args: argparse.Namespace, model: str,
) -> None:
    """Require the signed, byte-bound SFT handoff before SingleScout Central."""
    from tmx_gpu import c9_evidence

    if not os.path.isdir(model):
        raise ValueError("C9 Central must initialize from a local SFT checkpoint")
    lineage_path = str(getattr(args, "external_lineage_json", "") or "")
    if not lineage_path:
        raise ValueError("C9 Central requires signed SFT Central lineage")
    lineage = _load_required_json_object(lineage_path, "C9 Central lineage")
    c9_evidence.validate_central_lineage(lineage)
    runtime = lineage.get("central_runtime_contract")
    if not isinstance(runtime, dict) or os.path.realpath(
            str(runtime.get("model_init", ""))) != os.path.realpath(model):
        raise ValueError("C9 Central lineage does not bind the local SFT init")
    manifest = lineage.get("sft_model_file_manifest")
    if not isinstance(manifest, dict):
        raise ValueError("C9 Central lineage lacks the SFT model-file manifest")
    common.verify_directory_file_manifest(
        model, manifest, exclude_relative_paths=("sft_provenance.json",))


def _validate_c9_multi_sft_init(
    args: argparse.Namespace, model: str,
) -> None:
    """Verify the exact pooled Base-SFT consumed by a C9 Multi Central."""
    if not os.path.isdir(model) or not os.path.isfile(
            os.path.join(model, "config.json")):
        raise ValueError(
            "C9 multi-explorer Central must initialize from a local SFT model")
    if not str(getattr(args, "external_lineage_json", "") or ""):
        raise ValueError(
            "C9 multi-explorer Central requires pooled-SFT gate lineage")
    if not str(getattr(args, "mrme_contract_json", "") or ""):
        raise ValueError(
            "C9 multi-explorer Central requires the immutable campaign contract")
    provenance = _load_required_json_object(
        os.path.join(model, "sft_provenance.json"),
        "C9 multi-explorer SFT provenance")
    expected_scalars = {
        "model_init": common.CANONICAL_TRAINING_MODEL_ID,
        "model_revision": common.CANONICAL_TRAINING_MODEL_REVISION,
        "termination_mode": "append_eos",
        "optimizer": "adamw",
        "accepted_examples": 500,
        "epochs": 2,
        "grad_accum": 1,
        "optimizer_steps": 1000,
        "planned_optimizer_steps": 1000,
    }
    drift = [
        f"{name}={provenance.get(name)!r} (expected {expected!r})"
        for name, expected in expected_scalars.items()
        if provenance.get(name) != expected
    ]
    if not math.isclose(
            float(provenance.get("learning_rate", -1.0)), 5e-6,
            rel_tol=0.0, abs_tol=1e-15):
        drift.append("learning_rate must equal 5e-6")
    runtime = provenance.get("sft_runtime_contract")
    if not isinstance(runtime, dict) or runtime.get(
            "model_init") != common.CANONICAL_TRAINING_MODEL_ID or runtime.get(
            "base_model_revision") != common.CANONICAL_TRAINING_MODEL_REVISION:
        drift.append("SFT runtime lacks the canonical Base identity")
    if drift:
        raise ValueError(
            "C9 multi-explorer SFT initialization drifted: " + "; ".join(drift))
    manifest = provenance.get("checkpoint_file_manifest")
    if not isinstance(manifest, dict):
        raise ValueError("C9 multi-explorer SFT lacks a model-file manifest")
    common.verify_directory_file_manifest(
        model, manifest, exclude_relative_paths=("sft_provenance.json",))

    source = provenance.get("trajectory_source")
    source_manifest = source.get("manifest") if isinstance(source, dict) else None
    if not isinstance(source_manifest, dict):
        raise ValueError("C9 multi-explorer SFT lacks its K-Scout source manifest")
    common.verify_trajectory_source_manifest(source_manifest)
    if source.get("digest") != common.trajectory_source_manifest_digest(
            source_manifest):
        raise ValueError("C9 multi-explorer SFT trajectory-source digest mismatch")
    contract_sha = str(getattr(args, "mrme_contract_sha256", "") or "")
    round_index = int(getattr(args, "round", 0) or 0)
    scouts = int(getattr(args, "mrme_scouts_per_round", 0) or 0)
    artifacts = source_manifest.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != scouts:
        raise ValueError("C9 multi-explorer SFT source count differs from K")
    pattern = re.compile(
        rf"^mrme-local/{re.escape(contract_sha)}/"
        rf"r{round_index:02d}-s([0-9]{{2}}):v1$")
    indexes = []
    for artifact in artifacts:
        match = pattern.fullmatch(str(artifact.get("artifact_ref", ""))) \
            if isinstance(artifact, dict) else None
        if match is None:
            raise ValueError(
                "C9 multi-explorer SFT source is not campaign/round bound")
        indexes.append(int(match.group(1)))
    if sorted(indexes) != list(range(1, scouts + 1)):
        raise ValueError("C9 multi-explorer SFT does not cover every Scout once")


def _validate_c9_multi_prior_central_init(
    args: argparse.Namespace, model: str,
) -> None:
    """Verify an R>1 Scout starts from the immediately prior MR-ME Central."""
    from tmx_gpu import mrme_contract

    round_index = int(getattr(args, "round", 0) or 0)
    if round_index <= 1:
        raise ValueError("round-1 C9 Multi Scouts must initialize from Base")
    contract = mrme_contract.validate_contract(
        _load_required_json_object(
            str(getattr(args, "mrme_contract_json", "") or ""),
            "C9 multi-explorer campaign contract"))
    identity = mrme_contract.model_handoff_identity(
        model, expected_phase="central")
    runtime = identity.get("runtime_contract")
    if not isinstance(runtime, dict):
        raise ValueError("prior MR-ME Central lacks a runtime contract")
    multi = runtime.get("multi_explorer_contract")
    contract_sha = str(getattr(args, "mrme_contract_sha256", "") or "")
    method = str(getattr(args, "mrme_method", "") or "")
    profile = str(getattr(args, "mrme_stabilization_profile", "") or "")
    scouts = int(getattr(args, "mrme_scouts_per_round", 0) or 0)
    if (
        contract.get("contract_sha256") != contract_sha
        or contract.get("method") != method
        or contract.get("stabilization_profile") != profile
        or int(contract.get("scouts_per_round", 0)) != scouts
        or round_index > int(contract.get("rounds", 0))
    ):
        raise ValueError("C9 multi-explorer Scout args differ from campaign contract")
    expected_multi = {
        "contract_sha256": contract_sha,
        "method": method,
        "stabilization_profile": profile,
        "scout_index": 0,
        "scouts_per_round": scouts,
    }
    if mrme_contract.is_single_scout_multiround(contract):
        expected_multi.update({
            "method_version": contract["method_version"],
            "single_scout_multiround": True,
            "mrme": False,
            "multi_explorer": False,
        })
    if not isinstance(multi, dict) or any(
            multi.get(name) != expected
            for name, expected in expected_multi.items()):
        raise ValueError("prior Central has the wrong MR-ME campaign binding")
    previous_shard = str(multi.get("shard_manifest_sha256", ""))
    if re.fullmatch(r"[0-9a-f]{64}", previous_shard) is None:
        raise ValueError("prior Central lacks an immutable shard binding")
    provenance = identity.get("mrme_provenance")
    expected_provenance = {
        "method": method,
        "stabilization_profile": profile,
        "scientific_variant": C9_MULTIEXPLORER_SCIENTIFIC_VARIANT,
        "mrme_contract_sha256": contract_sha,
        "mrme_shard_manifest_sha256": previous_shard,
        "mrme_scout_index": 0,
        "mrme_scouts_per_round": scouts,
        "stage_seed": runtime.get("seed"),
        "phase": "central",
    }
    if mrme_contract.is_single_scout_multiround(contract):
        expected_provenance.update({
            "method_version": contract["method_version"],
            "single_scout_multiround": True,
            "mrme": False,
            "multi_explorer": False,
        })
    if provenance != expected_provenance or identity.get("round") != round_index - 1 \
            or identity.get("step") != 100:
        raise ValueError(
            "C9 multi-explorer Scout init is not the immediately prior Central")
    if runtime.get("phase") != "central" or runtime.get(
            "base_model_revision") != common.CANONICAL_TRAINING_MODEL_REVISION or \
            runtime.get("scientific_variant") != \
            C9_MULTIEXPLORER_SCIENTIFIC_VARIANT or float(
                runtime.get("lambda_novelty", -1.0)) != 0.0 or \
            "scout_novelty_contract" in runtime or os.path.exists(
                os.path.join(model, "rnd_state.pt")):
        raise ValueError("prior MR-ME Central is not correctness-only C9")


def _validate_c9_model_init_boundary(
    args: argparse.Namespace, *, phase: str, stabilized_multi: bool,
) -> None:
    """Pin Base at R1 and prove every later local model handoff."""
    model = str(getattr(args, "model", "") or "")
    revision = str(getattr(args, "model_revision", "") or "")
    if revision != common.CANONICAL_TRAINING_MODEL_REVISION:
        raise ValueError(
            "C9 requires the immutable canonical training checkpoint revision: "
            f"{common.CANONICAL_TRAINING_MODEL_REVISION!r}; got {revision!r}")
    if phase == "scout":
        round_index = int(getattr(args, "round", 1) or 1)
        if round_index == 1:
            if model != common.CANONICAL_TRAINING_MODEL_ID:
                raise ValueError(
                    "C9 round-1 Scout requires the immutable canonical Base: "
                    f"{common.CANONICAL_TRAINING_MODEL_ID!r}; got {model!r}")
        elif stabilized_multi:
            _validate_c9_multi_prior_central_init(args, model)
        else:
            raise ValueError("only contract-bound MR-ME may use a local R>1 Scout init")
    elif phase == "central":
        if stabilized_multi:
            _validate_c9_multi_sft_init(args, model)
        else:
            _validate_c9_single_central_init(args, model)
    else:
        raise ValueError(f"unknown C9 phase {phase!r}")


def validate_expdis_stabilization(
    args: argparse.Namespace,
    *,
    phase: Optional[str] = None,
    lambda_novelty: Optional[float] = None,
) -> None:
    """Fail closed on unlabeled or internally inconsistent C9 requests.

    C8 remains the default raw/candidate-attempt behavior.  The repaired C9
    lifecycle is deliberately reachable only as a named scientific variant
    under the on-policy c8 validity contract and immutable code binding.
    """
    lifecycle = expdis_stabilization_mode(args)
    validate_overlong_filter_contract(args)
    normalization = novelty_normalization_mode(args)
    variant = str(getattr(args, "scientific_variant", "") or "")
    profile = str(getattr(args, "c9_execution_profile", "") or "")
    c10_profile = str(getattr(args, "c10_execution_profile", "") or "")
    cell = str(getattr(args, "stabilization_cell", "") or "")
    resolved_phase = str(phase or getattr(args, "phase", "scout"))
    lam = (float(lambda_novelty) if lambda_novelty is not None
           else float(getattr(args, "lambda_novelty", 0.0) or 0.0))
    multi_contract_sha = str(
        getattr(args, "mrme_contract_sha256", "") or "")
    multi_method = str(getattr(args, "mrme_method", "") or "")
    multi_method_version = str(
        getattr(args, "mrme_method_version", "") or "")
    single_scout_multiround = bool(getattr(
        args, "single_scout_multiround", False))
    multi_profile = str(
        getattr(args, "mrme_stabilization_profile", "") or "")
    stabilized_multi = variant == C9_MULTIEXPLORER_SCIENTIFIC_VARIANT
    if stabilized_multi:
        from tmx_gpu import mrme_contract

        exact_k1 = (
            multi_method ==
            mrme_contract.DISJOINT_SHARD_SINGLESCOUT_MULTIROUND_METHOD
            and multi_method_version ==
            mrme_contract.DISJOINT_SHARD_SINGLESCOUT_MULTIROUND_METHOD_VERSION
            and single_scout_multiround
            and int(getattr(args, "mrme_scouts_per_round", 0) or 0) == 1
        )
        exact_multi = (
            multi_method in {
                mrme_contract.INTEGRATED_MULTISCOUT_METHOD,
                mrme_contract.DISJOINT_SHARD_MRME_METHOD,
            }
            and not multi_method_version
            and not single_scout_multiround
            and int(getattr(args, "mrme_scouts_per_round", 0) or 0) > 1
        )
        if len(multi_contract_sha) != 64 or not (exact_k1 or exact_multi) or \
                multi_profile != mrme_contract.C9_MULTIEXPLORER_PROFILE:
            raise ValueError(
                "C9 round-campaign variant requires an immutable campaign "
                "contract, exact K/method/version identity, and "
                "stabilization profile")
        if profile != "full" or cell != "C5":
            raise ValueError(
                "C9 multi-explorer Scouts/Central require strict full/C5 "
                "probe behavior")

    if profile not in C9_EXECUTION_PROFILES:
        raise ValueError(f"unknown C9 execution profile {profile!r}")
    if c10_profile not in C10_EXECUTION_PROFILES:
        raise ValueError(f"unknown C10 execution profile {c10_profile!r}")
    if cell not in C9_STABILIZATION_CELLS:
        raise ValueError(f"unknown stabilization cell {cell!r}")
    if bool(profile) != bool(cell):
        raise ValueError(
            "C9 execution profile and stabilization cell must be supplied together")
    if c10_profile and (profile or cell or stabilized_multi):
        raise ValueError(
            "C10 is a separate execution identity and cannot be mixed with "
            "C9 cell/profile or multi-explorer identity")

    if lifecycle == C8_RND_LIFECYCLE:
        if normalization != "raw":
            raise ValueError(
                "selected-batch novelty normalization is a C9-only change; "
                f"use --expdis-stabilization-mode {C9_RND_LIFECYCLE} and the "
                f"scientific variant {C9_SCIENTIFIC_VARIANT!r}")
        if variant in {
                C9_SCIENTIFIC_VARIANT,
                C9_MULTIEXPLORER_SCIENTIFIC_VARIANT,
                C10_SCIENTIFIC_VARIANT}:
            raise ValueError(
                "C9 scientific-variant label cannot be attached to frozen C8 "
                "candidate-attempt RND updates")
        if profile:
            if profile != "canary" or cell != "C1":
                raise ValueError(
                    "the preregistered candidate-attempt branch is only C1 canary")
            if variant != C9_SCREEN_VARIANT:
                raise ValueError(
                    f"C1 canary requires scientific variant {C9_SCREEN_VARIANT!r}")
            if not str(getattr(args, "code_source_manifest", "") or ""):
                raise ValueError("C1 canary requires an immutable code manifest")
        if c10_profile:
            raise ValueError(
                "C10 requires the selected-batch RND lifecycle, not frozen C8")
        return

    _validate_c9_model_init_boundary(
        args, phase=resolved_phase, stabilized_multi=stabilized_multi)

    expected_variant = (
        C9_MULTIEXPLORER_SCIENTIFIC_VARIANT if stabilized_multi else
        C10_SCIENTIFIC_VARIANT if c10_profile else
        C9_SCIENTIFIC_VARIANT if cell == "C5" else C9_SCREEN_VARIANT)
    if not profile and not c10_profile:
        # Direct/unit callers may still exercise the implementation, but a
        # scientific GPU run must always carry the profile+cell pair.
        expected_variant = (
            C9_MULTIEXPLORER_SCIENTIFIC_VARIANT if stabilized_multi
            else C9_SCIENTIFIC_VARIANT)
    if variant != expected_variant:
        raise ValueError(
            f"{C9_RND_LIFECYCLE} requires --scientific-variant "
            f"{expected_variant!r}; refusing a mislabeled stabilization run")
    if profile == "full" and cell != "C5":
        raise ValueError("only preregistered C5 may use the full C9 profile")
    if cell and cell not in {"C2", "C3", "C4", "C5"}:
        raise ValueError(
            "selected-batch RND is valid only for C9 cells C2--C5")
    if rl_validity_mode(args) != "c8":
        raise ValueError(
            f"{C9_RND_LIFECYCLE} requires --rl-validity-mode c8")
    if str(getattr(args, "fresh_c8_evidence_profile", "") or ""):
        raise ValueError(
            "C9 is a separate labeled experiment and cannot claim a frozen "
            "fresh-c8 evidence profile")
    if resolved_phase == "scout":
        if lam <= 0.0:
            raise ValueError("C9 selected-batch RND requires Scout lambda > 0")
        if not math.isclose(
                float(getattr(args, "incorrect_novelty_scale", 0.0) or 0.0),
                0.0, rel_tol=0.0, abs_tol=0.0):
            raise ValueError(
                "C9 preserves correct-only novelty gating and requires "
                "--incorrect-novelty-scale 0")
        if not str(getattr(args, "code_source_manifest", "") or ""):
            raise ValueError(
                "C9 Scout requires an immutable --code-source-manifest")


def validate_rl_validity(args: argparse.Namespace,
                         pool: Optional["VLLMPool"]) -> None:
    """Fail closed when c8 cannot provide genuinely on-policy rollouts.

    A restart-based sync is only possible for a managed pool. c8 also pins the
    original one-update REINFORCE contract: with multiple optimizer epochs a
    single post-batch reload would not be a sync after every learner update,
    and PPO would change the frozen c8 campaign objective. A clipped objective
    can have KL=0; that is the latest paper's separate JAX contract.
    """
    validate_overlong_filter_contract(args)
    if rl_validity_mode(args) != "c8":
        return
    if pool is None or not bool(pool.managed):
        raise RuntimeError(
            "--rl-validity-mode c8 requires a managed vLLM pool so policy "
            "weights can be synced after every learner update; attach-mode "
            "endpoints cannot satisfy the on-policy contract")
    if int(getattr(args, "updates_per_rollout", 1)) != 1:
        raise ValueError(
            "--rl-validity-mode c8 requires --updates-per-rollout 1")
    if bool(getattr(args, "ppo_mode", False)):
        raise ValueError(
            "--rl-validity-mode c8 preserves the KL=0 REINFORCE contract and "
            "cannot be combined with --ppo-mode")


def sync_after_update_due(args: argparse.Namespace, step: int,
                          total_steps: int) -> bool:
    """Whether the just-updated learner must be exported to the rollout pool."""
    if rl_validity_mode(args) == "c8":
        return True
    cadence = int(getattr(args, "reload_every", 0))
    return cadence > 0 and int(step) % cadence == 0 and int(step) < int(total_steps)


def check_resume_compat(recorded: Dict[str, Any], *, phase: str, model: str,
                        config_tier: str,
                        rl_validity_mode_current: str = "legacy",
                        resolved_lr_current: Optional[float] = None,
                        external_lineage_current: Optional[Dict[str, Any]] = None,
                        runtime_contract_current: Optional[Dict[str, Any]] = None,
                        code_artifact_binding_current: Optional[
                            Dict[str, Any]] = None) -> None:
    """--resume-from guard: hard error when the checkpoint's recorded
    phase/model/config_tier/validity contract disagree with the current args.

    Checkpoints predating c8 have no validity field and are explicitly treated
    as legacy. This preserves old resumes while preventing a nominal c8 run
    from silently inheriting a stale-policy/optimizer-reset checkpoint.
    """
    recorded_mode = str(recorded.get("rl_validity_mode", "legacy"))
    mismatches = [
        f"{key}: checkpoint={recorded[key]!r} vs current={cur!r}"
        for key, cur in (("phase", phase), ("model", model),
                         ("config_tier", config_tier))
        if recorded.get(key) is not None and str(recorded[key]) != str(cur)
    ]
    if recorded_mode != str(rl_validity_mode_current):
        mismatches.append(
            "rl_validity_mode: "
            f"checkpoint={recorded_mode!r} vs current={rl_validity_mode_current!r}")
    if (resolved_lr_current is not None and recorded.get("resolved_lr") is not None
            and not math.isclose(float(recorded["resolved_lr"]),
                                 float(resolved_lr_current), rel_tol=0.0, abs_tol=1e-15)):
        mismatches.append(
            f"resolved_lr: checkpoint={recorded['resolved_lr']!r} "
            f"vs current={resolved_lr_current!r}")
    if (str(rl_validity_mode_current) == "c8"
            and recorded.get("optimizer_state_saved") is not True):
        mismatches.append(
            "optimizer_state_saved: checkpoint does not certify saved optimizer moments")
    recorded_lineage = recorded.get("external_lineage")
    if recorded_lineage != external_lineage_current:
        mismatches.append("external_lineage: checkpoint does not match current handoff")
    expected_lineage_sha = (
        external_lineage_current.get("lineage_sha256")
        or external_lineage_current.get("evidence_sha256")
        if external_lineage_current else None)
    if recorded.get("external_lineage_sha256") != expected_lineage_sha:
        mismatches.append("external_lineage_sha256: checkpoint lineage digest mismatch")
    require_runtime_contract = (
        runtime_contract_current is not None
        and (str(rl_validity_mode_current) == "c8"
             or external_lineage_current is not None
             or "runtime_contract" in recorded)
    )
    if require_runtime_contract and \
            recorded.get("runtime_contract") != runtime_contract_current:
        mismatches.append(
            "runtime_contract: checkpoint does not match resolved current args")
    recorded_code_binding = recorded.get("code_artifact_binding")
    if recorded_code_binding != code_artifact_binding_current and \
            (recorded_code_binding is not None
             or code_artifact_binding_current is not None):
        mismatches.append(
            "code_artifact_binding: checkpoint does not match current code")
    if mismatches:
        raise RuntimeError(
            "--resume-from checkpoint config mismatch: " + "; ".join(mismatches)
            + " — refusing to resume (fix the flags or point --resume-from at "
              "a checkpoint from THIS run configuration)")


def build_optimizer(name: str, params, *, lr: float, betas=(0.9, 0.95),
                    eps: float = 1e-8, weight_decay: float = 0.0):
    """AdamW (contract) or Adafactor (GPU-MAX / TPU-SFT parity ablations)."""
    name = str(name or "adamw").lower()
    if name == "adafactor":
        from transformers.optimization import Adafactor

        return Adafactor(
            params, lr=float(lr), scale_parameter=False, relative_step=False,
            warmup_init=False, weight_decay=float(weight_decay),
        )
    if name != "adamw":
        raise ValueError(f"unsupported optimizer={name!r}; expected adamw | adafactor")
    return torch.optim.AdamW(
        params, lr=float(lr), betas=tuple(betas), eps=float(eps),
        weight_decay=float(weight_decay),
    )


# ------------------------------------------------------------------- loss ----

def chunked_selective_log_softmax(
    hidden: torch.Tensor,
    lm_head: torch.nn.Module,
    targets: torch.Tensor,
    chunk_tokens: int = 256,
) -> torch.Tensor:
    """Per-token target logprobs from hidden states without materializing
    full-vocab logits for all positions.

    hidden: (T, H) aligned so lm_head(hidden[t]) predicts targets[t].
    With fp32 master params the chunk's logits matmul runs in fp32 OUTSIDE the
    callers' bf16 autocast region (the TPU reference upcasts both kernel and
    hidden before the einsum; casting after the matmul would keep bf16 noise).
    Pure-bf16 smoke mode (--param-dtype bfloat16) keeps the matmul in bf16 and
    upcasts the result. Chunks are wrapped in torch.utils.checkpoint so
    backward recomputes one chunk at a time (retaining all chunk logits would
    re-materialize the full logits tensor).
    Returns (T,) fp32 logprobs with gradients flowing to hidden/lm_head.
    """
    from torch.utils.checkpoint import checkpoint

    def _chunk_logp(h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=h.device.type, enabled=False):
            if lm_head.weight.dtype == torch.float32:
                logits = lm_head(h.float())
            else:  # pure-bf16 opt-in: weights are bf16, keep the matmul bf16
                logits = lm_head(h)
            logits = logits.float()
        return torch.log_softmax(logits, dim=-1).gather(-1, t.unsqueeze(-1)).squeeze(-1)

    out: List[torch.Tensor] = []
    total = int(hidden.shape[0])
    step = max(1, int(chunk_tokens))
    for start in range(0, total, step):
        h = hidden[start:start + step]
        t = targets[start:start + step]
        if hidden.requires_grad:
            out.append(checkpoint(_chunk_logp, h, t, use_reentrant=False))
        else:
            out.append(_chunk_logp(h, t))
    return torch.cat(out, dim=0)


# ---------------------------------------------------------------- scoring ----

def score_completions(
    tokenizer,
    example,
    completions,
    *,
    max_completion_len: int = 16384,
    answer_tail_chars: Optional[int] = None,
    reward_correct: float = 1.0,
    reward_incorrect: float = -1.0,
    soft_overlong_expected_len: int = 13107,
    soft_overlong_cache_len: int = 3277,
) -> List[Dict[str, Any]]:
    """Base (correctness + quality flags + overlong) scoring for one group.

    Mirrors tmx_jax.train._score_rollouts row construction. Novelty fields are
    zero-initialized; apply_novelty_and_blend() fills them in when lambda>0.
    Keeps the tokenized completion ids under "_comp_ids" and the novelty
    re-encode text under "_novelty_text" (underscore keys are stripped before
    any JSONL serialization) so the train step never re-tokenizes and rollout
    logs do not store every completion twice.

    Callers must filter retry-exhausted completions (vllm_pool
    completion_failed) BEFORE scoring; error rows never enter reward scoring.
    """
    if answer_tail_chars not in (None, 0):
        raise ValueError(
            "tail-only reward scoring is retired; correctness and validity must "
            "inspect the full completion")
    rows: List[Dict[str, Any]] = []
    for c in completions:
        text = getattr(c, "text", "") or ""
        # Reward and format validity must use the same complete response that is
        # persisted and later re-graded by coverage_pool_c8.  Tail-only grading
        # silently discarded otherwise-valid boxed answers followed by >4K
        # characters and made the Scout reward disagree with the filter/eval.
        scoring_text = text
        finish = str(getattr(c, "finish_reason", "") or "").lower()
        comp_ids = tokenizer(text, add_special_tokens=False)["input_ids"] if text else []
        retokenized_len = len(comp_ids)
        # vLLM's usage count is the authoritative length of the sampled token
        # stream. Decoding and re-tokenizing can merge/split tokens and, for a
        # cap-truncated request, can understate the tokens that actually drove
        # soft-overlong shaping. Fall back only for legacy/test completions
        # that do not carry usage metadata.
        server_len = int(getattr(c, "completion_tokens", 0) or 0)
        true_len = server_len if server_len > 0 else retokenized_len
        if finish == "length":
            true_len = max(true_len, int(max_completion_len))
        tok_len = min(true_len, int(max_completion_len))
        clipped = finish == "length" or true_len >= int(max_completion_len)
        # Missing finish metadata is not evidence of a natural stop. This is
        # deliberately fail-closed for filtering and SFT curation.
        terminated = finish in TERMINATED_FINISH_REASONS
        meta = analyze_completion_text(scoring_text)
        is_corr = is_correct_completion(scoring_text, example.ground_truth)
        correctness = float(reward_correct if is_corr else reward_incorrect)
        overlong = common.soft_overlong_penalty(
            tok_len, int(soft_overlong_expected_len), int(soft_overlong_cache_len))
        blended = correctness + overlong
        if not math.isfinite(blended):
            blended = float(reward_incorrect)
        rows.append({
            "problem_id": str(example.problem_id),
            "prompt_text": example.prompt_text,
            "completion_text": text,
            "ground_truth": example.ground_truth,
            "finish_reason": finish,
            "completion_token_length": int(tok_len),
            "completion_token_length_true": int(true_len),
            "completion_token_length_retokenized": int(retokenized_len),
            "clipped": bool(clipped),
            "terminated": bool(terminated),
            "valid_answer": bool(meta.get("valid_answer_presentation", False)),
            "is_correct": bool(is_corr),
            "correctness_raw": correctness,
            "correctness_reward": correctness,
            "_novelty_text": text.strip(),
            "novelty_raw": 0.0,
            "novelty_used": 0.0,
            "novelty_reward": 0.0,
            "r_novel_layers": {},
            "overlong_penalty": float(overlong),
            "blended_reward": float(blended),
            "_comp_ids": [int(x) for x in comp_ids[:int(max_completion_len)]],
        })
    return rows


def apply_novelty_and_blend(
    rows: List[Dict[str, Any]],
    novelty_final: np.ndarray,
    per_layer: Dict[int, np.ndarray],
    *,
    lambda_novelty: float,
    incorrect_novelty_scale: float = 0.0,
    reward_incorrect: float = -1.0,
) -> None:
    """Blend RND novelty into blended_reward (novelty computed pre-update)."""
    for i, row in enumerate(rows):
        nov_raw = float(novelty_final[i]) if i < len(novelty_final) else 0.0
        nov_used = nov_raw  # zscore contract: off
        if not bool(row["is_correct"]):
            nov_used *= float(max(0.0, incorrect_novelty_scale))
        blended = float(row["correctness_raw"]) + float(lambda_novelty) * nov_used + float(row["overlong_penalty"])
        if not math.isfinite(blended):
            blended = float(reward_incorrect)
        row.update({
            "novelty_raw": nov_raw,
            "novelty_used": nov_used,
            "novelty_reward": nov_used,
            "r_novel_layers": {f"layer_{layer}": float(vals[i]) for layer, vals in per_layer.items()},
            "blended_reward": float(blended),
        })


def recompute_selected_novelty_and_blend(
    rows: List[Dict[str, Any]],
    *,
    normalization: str,
    lambda_novelty: float,
    incorrect_novelty_scale: float = 0.0,
    reward_incorrect: float = -1.0,
) -> Dict[str, float]:
    """Recompute C9 learner rewards over the completed selected batch.

    Candidate eligibility always uses raw, correct-gated novelty while the RND
    predictor is frozen.  Once all learner rows are known, this function
    applies either raw scaling or a population z-score over *all selected raw
    scores*, then applies the correct-only gate and rebuilds every selected
    blended reward consistently.  Discarded candidates are never rewritten or
    used for the policy/RND update.
    """
    if normalization not in NOVELTY_NORMALIZATION_MODES:
        raise ValueError(f"unknown novelty normalization {normalization!r}")
    if not rows:
        raise ValueError("selected-batch novelty normalization needs learner rows")
    raw = np.asarray(
        [float(row.get("novelty_raw", 0.0)) for row in rows],
        dtype=np.float32,
    )
    if not np.all(np.isfinite(raw)):
        raise ValueError("selected-batch raw novelty contains non-finite values")
    if normalization == "raw":
        center = 0.0
        population_scale = 1.0
        effective_scale = 1.0
        normalized = raw.copy()
    else:
        center = float(np.mean(raw, dtype=np.float64))
        population_scale = float(np.std(raw, dtype=np.float64))
        if population_scale <= C9_NORMALIZATION_EPSILON:
            raise RuntimeError(
                "C9 selected-batch z-score has degenerate population scale: "
                f"{population_scale}")
        effective_scale = max(population_scale, C9_NORMALIZATION_EPSILON)
        normalized = ((raw.astype(np.float64) - center) / effective_scale).astype(
            np.float32)

    incorrect_scale = float(max(0.0, incorrect_novelty_scale))
    for row, nov_normalized in zip(rows, normalized.tolist(), strict=True):
        nov_used = float(nov_normalized)
        if not bool(row["is_correct"]):
            nov_used *= incorrect_scale
        blended = (float(row["correctness_raw"])
                   + float(lambda_novelty) * nov_used
                   + float(row["overlong_penalty"]))
        if not math.isfinite(blended):
            blended = float(reward_incorrect)
        row.update({
            "novelty_normalized_pre_gate": float(nov_normalized),
            "novelty_used": nov_used,
            "novelty_reward": nov_used,
            "blended_reward": float(blended),
            "novelty_normalization": normalization,
            "novelty_normalization_center": center,
            "novelty_normalization_scale": population_scale,
            "novelty_normalization_effective_scale": effective_scale,
            "rnd_lifecycle": C9_RND_LIFECYCLE,
            "rnd_update_scope": "final_selected_learner_rows",
        })
    return {
        "novelty_normalization_center": center,
        "novelty_normalization_scale": population_scale,
        "novelty_normalization_effective_scale": effective_scale,
        "novelty_normalized_selected_mean": float(
            np.mean(normalized, dtype=np.float64)),
        "novelty_normalized_selected_population_std": float(
            np.std(normalized, dtype=np.float64)),
    }


def trajectory_record(
    row: Dict[str, Any], step: int, *,
    fresh_c8_evidence_profile: str = "",
    component_seed: Optional[int] = None,
    seed: Optional[int] = None,
    mrme_trajectory_provenance: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Match tmx_jax.train._trajectory_record schema (filter_pool/SFT input)."""
    record = {
        "trajectory_schema_version": TRAJECTORY_RECORD_SCHEMA_VERSION,
        "problem_id": str(row["problem_id"]),
        "prompt_text": row["prompt_text"],
        "completion_text": row["completion_text"],
        "ground_truth": row["ground_truth"],
        "correctness_reward": float(row["correctness_reward"]),
        "novelty_reward": float(row["novelty_reward"]),
        "blended_reward": float(row["blended_reward"]),
        "reward": float(row["blended_reward"]),
        "r_correct": float(row["correctness_reward"]),
        "r_novel": float(row["novelty_reward"]),
        "is_correct": bool(row["is_correct"]),
        "explorer_step": int(step),
        "step": int(step),
        "finish_reason": str(row.get("finish_reason", "")),
        "completion_token_length": int(row.get("completion_token_length", 0)),
        "completion_token_length_true": int(
            row.get("completion_token_length_true",
                    row.get("completion_token_length", 0))),
        "completion_token_length_retokenized": int(
            row.get("completion_token_length_retokenized",
                    row.get("completion_token_length", 0))),
        "clipped": bool(row.get("clipped", False)),
        "terminated": bool(row.get("terminated", False)),
        "valid_answer": bool(row.get("valid_answer", False)),
    }
    if row.get("c9_execution_profile"):
        record.update({
            "scientific_variant": str(row.get("scientific_variant", "")),
            "c9_execution_profile": str(row["c9_execution_profile"]),
            "stabilization_cell": str(row.get("stabilization_cell", "")),
        })
    if row.get("dynamic_sampling_eligibility_population") == \
            "unmasked_survivors":
        record.update({
            "c10_execution_profile": str(
                row.get("c10_execution_profile", "")),
            "dynamic_sampling_reason": str(
                row.get("dynamic_sampling_reason", "")),
            "dynamic_sampling_raw_group_size": int(
                row["dynamic_sampling_raw_group_size"]),
            "dynamic_sampling_survivor_count": int(
                row["dynamic_sampling_survivor_count"]),
            "dynamic_sampling_masked_count": int(
                row["dynamic_sampling_masked_count"]),
            "dynamic_sampling_raw_correct_count": int(
                row["dynamic_sampling_raw_correct_count"]),
            "dynamic_sampling_survivor_correct_count": int(
                row["dynamic_sampling_survivor_correct_count"]),
            "dynamic_sampling_reward_std": float(
                row["dynamic_sampling_reward_std"]),
            "dynamic_sampling_novelty_contribution_std": float(
                row["dynamic_sampling_novelty_contribution_std"]),
            "dynamic_sampling_eligibility_population":
                "unmasked_survivors",
            "dynamic_sampling_eligibility_population_size": int(
                row["dynamic_sampling_eligibility_population_size"]),
            "dynamic_sampling_group_selected": bool(
                row["dynamic_sampling_group_selected"]),
            "dynamic_sampling_policy_loss_eligible": bool(
                row["dynamic_sampling_policy_loss_eligible"]),
            "dynamic_sampling_eligibility_novelty_used": float(
                row["dynamic_sampling_eligibility_novelty_used"]),
            "dynamic_sampling_eligibility_blended_reward": float(
                row["dynamic_sampling_eligibility_blended_reward"]),
            "c10_rnd_raw_novelty_score": float(row["novelty_raw"]),
            "c10_normalized_novelty_pre_gate": float(
                row["novelty_normalized_pre_gate"]),
        })
    if row.get("rnd_lifecycle") == C9_RND_LIFECYCLE:
        record.update({
            "expdis_stabilization_mode": C9_RND_LIFECYCLE,
            "scientific_variant": str(
                row.get("scientific_variant", C9_SCIENTIFIC_VARIANT)),
            "c9_execution_profile": str(
                row.get("c9_execution_profile", "")),
            "stabilization_cell": str(row.get("stabilization_cell", "")),
            "novelty_normalization": str(row["novelty_normalization"]),
            "novelty_normalization_center": float(
                row["novelty_normalization_center"]),
            "novelty_normalization_scale": float(
                row["novelty_normalization_scale"]),
            "novelty_normalization_effective_scale": float(
                row["novelty_normalization_effective_scale"]),
            "rnd_update_scope": "final_selected_learner_rows",
        })
    if fresh_c8_evidence_profile == "fresh_c8_multiscout_component":
        if type(component_seed) is not int or component_seed not in (1, 2):
            raise ValueError(
                "component trajectory provenance requires exact integer "
                "component_seed 1 or 2")
        # Provenance-only marker.  It is deliberately embedded in every
        # official component row so copying the JSONL away from its W&B
        # artifact or completion-evidence sibling cannot turn a preregistered
        # Scout component into an apparently generic pooling input.
        record["fresh_c8_evidence_profile"] = str(
            fresh_c8_evidence_profile)
        record["component_seed"] = int(component_seed)
    elif fresh_c8_evidence_profile == "fresh_c8_single_scout":
        if type(seed) is not int or seed != 0:
            raise ValueError(
                "canonical trajectory provenance requires exact integer seed 0")
        record["fresh_c8_evidence_profile"] = str(
            fresh_c8_evidence_profile)
        record["seed"] = seed
    elif fresh_c8_evidence_profile:
        raise ValueError(
            "unknown fresh-c8 trajectory evidence profile "
            f"{fresh_c8_evidence_profile!r}")
    if mrme_trajectory_provenance is not None:
        expected_fields = {
            "mrme_trajectory_provenance_version",
            "mrme_contract_sha256",
            "mrme_method",
            "mrme_stabilization_profile",
            "mrme_round_index",
            "mrme_scout_index",
            "mrme_scout_seed",
            "mrme_shard_manifest_sha256",
        }
        if set(mrme_trajectory_provenance) != expected_fields:
            raise ValueError(
                "MR-ME trajectory provenance must contain the exact campaign "
                f"slot fields; got {sorted(mrme_trajectory_provenance)}")
        if mrme_trajectory_provenance.get(
                "mrme_trajectory_provenance_version") != \
                "gpu_mrme_trajectory_slot_v1":
            raise ValueError("unknown MR-ME trajectory provenance version")
        for field in (
                "mrme_contract_sha256", "mrme_shard_manifest_sha256"):
            if not re.fullmatch(
                    r"[0-9a-f]{64}", str(mrme_trajectory_provenance[field])):
                raise ValueError(f"MR-ME trajectory provenance has invalid {field}")
        for field in (
                "mrme_round_index", "mrme_scout_index", "mrme_scout_seed"):
            if type(mrme_trajectory_provenance[field]) is not int:
                raise ValueError(f"MR-ME trajectory provenance has invalid {field}")
        if not str(mrme_trajectory_provenance["mrme_method"]) or not str(
                mrme_trajectory_provenance["mrme_stabilization_profile"]):
            raise ValueError("MR-ME trajectory provenance lacks method/profile")
        record.update(mrme_trajectory_provenance)
    return record


def group_reason(correct_count: int, num_generations: int, reward_std: float,
                 lambda_novelty: float, *, novelty_contribution_std: Optional[float] = None,
                 validity_mode: str = "legacy") -> Tuple[str, bool]:
    """Dynamic-sampling group classification (mirrors _dynamic_sample_rollouts).

    Returns (reason, eligible). Mixed-correctness groups are always eligible.
    For homogeneous lambda>0 groups, legacy uses the full blended-reward std.
    c8 uses only std(lambda * novelty_used), explicitly excluding correctness
    (constant within the group) and the soft-overlong penalty. Thus a lambda
    arm cannot pass merely because completion lengths vary.
    """
    if validity_mode not in RL_VALIDITY_MODES:
        raise ValueError(f"unknown validity_mode={validity_mode!r}")
    signal_std = (float(reward_std) if validity_mode == "legacy"
                  else float(novelty_contribution_std or 0.0))
    reward_informative = lambda_novelty > 0.0 and signal_std > 1e-6
    if correct_count == 0:
        reason = "reward_informative_all_wrong" if reward_informative else "all_wrong"
    elif correct_count == int(num_generations):
        reason = "reward_informative_all_correct" if reward_informative else "all_correct"
    else:
        reason = "informative"
    return reason, reason == "informative" or reason.startswith("reward_informative_")


def group_eligibility_for_rows(
    group_rows: Sequence[Dict[str, Any]], *, lambda_novelty: float,
    validity_mode: str, survivor_only: bool,
) -> Tuple[str, bool, Dict[str, Any]]:
    """Classify one candidate group on its declared eligibility population."""
    rows = list(group_rows)
    if not rows:
        raise ValueError("dynamic-sampling group cannot be empty")
    survivors = [row for row in rows if not bool(row.get("clipped", False))]
    population = survivors if survivor_only else rows
    raw_correct_count = sum(1 for row in rows if bool(row["is_correct"]))
    survivor_correct_count = sum(
        1 for row in survivors if bool(row["is_correct"]))
    facts: Dict[str, Any] = {
        "raw_group_size": len(rows),
        "survivor_count": len(survivors),
        "masked_count": len(rows) - len(survivors),
        "raw_correct_count": raw_correct_count,
        "survivor_correct_count": survivor_correct_count,
        "eligibility_population": (
            "unmasked_survivors" if survivor_only else "all_rows"),
    }
    if survivor_only and len(survivors) < 2:
        facts.update({
            "eligibility_correct_count": survivor_correct_count,
            "eligibility_population_size": len(survivors),
            "eligibility_reward_std": 0.0,
            "eligibility_novelty_contribution_std": 0.0,
        })
        return "insufficient_unmasked_survivors", False, facts

    rewards = np.asarray(
        [float(row["blended_reward"]) for row in population],
        dtype=np.float32,
    )
    novelty_contributions = np.asarray(
        [float(lambda_novelty) * float(row.get("novelty_used", 0.0))
         for row in population],
        dtype=np.float32,
    )
    correct_count = sum(1 for row in population if bool(row["is_correct"]))
    reward_std = float(rewards.std()) if rewards.size else 0.0
    novelty_std = (
        float(novelty_contributions.std())
        if novelty_contributions.size else 0.0)
    reason, eligible = group_reason(
        correct_count, len(population), reward_std, lambda_novelty,
        novelty_contribution_std=novelty_std,
        validity_mode=validity_mode,
    )
    facts.update({
        "eligibility_correct_count": correct_count,
        "eligibility_population_size": len(population),
        "eligibility_reward_std": reward_std,
        "eligibility_novelty_contribution_std": novelty_std,
    })
    return reason, eligible, facts


def group_mean_advantages(rewards: np.ndarray, num_generations: int) -> np.ndarray:
    """Advantage = blended reward minus its group mean. NO std division (contract)."""
    grouped = np.asarray(rewards, dtype=np.float32).reshape(-1, int(num_generations))
    return (grouped - grouped.mean(axis=1, keepdims=True)).reshape(-1)


def group_mean_advantages_unmasked(
    rewards: np.ndarray, survivor_mask: np.ndarray, num_generations: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Center rewards over nontruncated survivors; masked rows get zero."""
    group_size = int(num_generations)
    if group_size <= 0:
        raise ValueError("num_generations must be positive")
    flat_rewards = np.asarray(rewards, dtype=np.float32)
    flat_survivors = np.asarray(survivor_mask, dtype=bool)
    if flat_rewards.ndim != 1 or flat_survivors.ndim != 1:
        raise ValueError("rewards and survivor_mask must be one-dimensional")
    if flat_rewards.shape != flat_survivors.shape:
        raise ValueError(
            "rewards and survivor_mask must have identical shapes")
    if flat_rewards.size % group_size:
        raise ValueError(
            f"{flat_rewards.size} rows cannot be reshaped into groups of "
            f"{group_size}")

    grouped_rewards = flat_rewards.reshape(-1, group_size)
    grouped_survivors = flat_survivors.reshape(-1, group_size)
    grouped_advantages = np.zeros_like(grouped_rewards, dtype=np.float32)
    survivor_counts = grouped_survivors.sum(axis=1).astype(np.int64)
    if np.any(survivor_counts < 2):
        raise ValueError(
            "C10 selected learner groups must each have at least two survivors")
    advantage_sums = []
    for group_index, keep in enumerate(grouped_survivors):
        kept_rewards = grouped_rewards[group_index, keep]
        centered = kept_rewards - kept_rewards.mean(dtype=np.float32)
        grouped_advantages[group_index, keep] = centered
        advantage_sums.append(float(centered.sum(dtype=np.float64)))
    stats = {
        "raw_selected_row_count": float(flat_rewards.size),
        "survivor_row_count": float(survivor_counts.sum()),
        "masked_row_count": float(flat_rewards.size - survivor_counts.sum()),
        "group_count": float(grouped_rewards.shape[0]),
        "min_survivors_per_group": float(survivor_counts.min()),
        "max_survivors_per_group": float(survivor_counts.max()),
        "mean_survivors_per_group": float(survivor_counts.mean()),
        "max_abs_survivor_advantage_sum": float(
            max((abs(value) for value in advantage_sums), default=0.0)),
    }
    return grouped_advantages.reshape(-1), stats


def group_advantages_for_rows(
    rows: Sequence[Dict[str, Any]], num_generations: int,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Compute group advantages under the exact declared baseline contract."""
    rewards = np.asarray(
        [float(row["blended_reward"]) for row in rows], dtype=np.float32)
    if survivor_consistent_overlong_filter(args):
        survivors = np.asarray(
            [not bool(row.get("clipped", False)) for row in rows], dtype=bool)
        return group_mean_advantages_unmasked(
            rewards, survivors, num_generations)
    advantages = group_mean_advantages(rewards, num_generations)
    return advantages, {}


def probe_average(correct_flags: Sequence[bool]) -> float:
    """Probe accuracy as a fraction — avg@k semantics (mean over the
    fixed k samples/problem), matching eval_gpu's avg@k for k = all samples.

    Returns NaN (never a fake 0.0 datapoint) when there are no graded flags:
    an all-failed probe must not emit a fake zero datapoint."""
    flags = list(correct_flags)
    return (sum(1.0 for f in flags if f) / len(flags)) if flags else float("nan")


# ------------------------------------------------------------------ model ----

def load_policy(
    model_path: str, *, device: str, param_dtype: str = "float32",
    expected_attention_implementation: Optional[str] = None,
    model_revision: str = "",
):
    """Load without lowering checkpoint precision, then select attention.

    In particular, an FP32 c8 checkpoint must never be opened as BF16 and
    upcast: doing so irreversibly rounds the learner weights while pretending
    optimizer-complete resume continuity. A resume also pins the attention
    implementation recorded by the checkpoint, avoiding a silent FA2/SDPA
    numerical-contract change across pods.
    """
    from transformers import AutoModelForCausalLM

    want_fp32 = str(param_dtype) == "float32"
    dtype = torch.float32 if want_fp32 else torch.bfloat16
    model, attn = None, None
    expected = str(expected_attention_implementation or "")
    if expected and expected not in {"flash_attention_2", "sdpa"}:
        raise ValueError(
            f"unsupported recorded attention implementation: {expected!r}")
    strategies = [(expected, dtype)] if expected else [
        ("flash_attention_2", dtype), ("sdpa", dtype)]
    for attn_impl, load_dtype in strategies:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=load_dtype, attn_implementation=attn_impl,
                **common.remote_revision_kwargs(model_path, model_revision),
            )
            attn = attn_impl
            break
        except Exception as exc:  # noqa: BLE001
            print(f"[grpo] attn={attn_impl} dtype={load_dtype} load failed: {exc}", flush=True)
            model = None
    if model is None:
        suffix = f" with recorded attention={expected}" if expected else ""
        raise RuntimeError(f"could not load policy from {model_path}{suffix}")
    if want_fp32 and any(
            parameter.is_floating_point() and parameter.dtype != torch.float32
            for parameter in model.parameters()):
        raise RuntimeError("FP32 policy load returned lower-precision parameters")
    model.to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    model._tmx_attention_implementation = attn
    print(f"[grpo] policy loaded from {model_path} attn={attn} param_dtype={param_dtype}", flush=True)
    return model


def _c10_jsonl_rows(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.isfile(path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).touch()
        return rows
    with open(path, encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise RuntimeError(
                    f"C10 terminal JSONL row {line_number} is not an object")
            rows.append(value)
    return rows


def _write_c10_failed_selected_batch(
    args: argparse.Namespace, rows: Sequence[Dict[str, Any]], *, step: int,
    dynamic_stats: Dict[str, Any],
) -> Tuple[str, str]:
    """Atomically journal the exact pre-update batch behind a C10 failure."""
    path = os.path.join(args.output_dir, "c10_failed_selected_batch.jsonl")
    manifest_path = os.path.join(
        args.output_dir, "c10_failed_selected_batch_manifest.json")
    journal_rows = []
    for index, source in enumerate(rows):
        row = dict(source)
        row["c10_failed_step"] = int(step)
        row["c10_failed_selected_row_index"] = int(index)
        journal_rows.append(row)
    if os.path.isfile(path):
        if _c10_jsonl_rows(path) != journal_rows:
            raise RuntimeError(
                "C10 failed selected-batch journal already differs")
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            for row in journal_rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    manifest = {
        "version": "c10_failed_selected_batch_manifest_v1",
        "failed_step": int(step),
        "dynamic_finalization": {
            "kept_group_count": dynamic_stats.get("kept_group_count"),
            "fallback_group_count": dynamic_stats.get("fallback_group_count"),
            "selected_raw_row_count": dynamic_stats.get(
                "selected_raw_row_count"),
        },
        "selected_row_count": len(journal_rows),
        "selected_rows_sha256": common.canonical_json_sha256(journal_rows),
    }
    manifest["evidence_sha256"] = common.canonical_json_sha256(manifest)
    if os.path.isfile(manifest_path):
        with open(manifest_path, encoding="utf-8") as handle:
            if json.load(handle) != manifest:
                raise RuntimeError(
                    "C10 failed selected-batch manifest already differs")
    else:
        common.atomic_write_json(manifest_path, manifest)
    return path, manifest_path


def _record_c10_training_failure(
    args: argparse.Namespace, *, runtime_contract: Dict[str, Any],
    code_artifact: Dict[str, Any], failure_kind: str,
    completed_step: int, failed_step: int, failure_details: Dict[str, Any],
    trajectory_path: str, transport_invalid: bool = False,
    failed_step_candidates_path: str = "",
    failed_selected_batch_path: str = "",
    failed_selected_batch_manifest_path: str = "",
) -> Dict[str, Any]:
    """Persist a preregistered semantic failure before propagating it."""
    from tmx_gpu import c10_evidence

    if str(getattr(args, "c10_execution_profile", "") or "") != "canary":
        raise RuntimeError("C10 failure recorder used outside C10")
    metrics_path = str(args.metrics_jsonl)
    trajectory_rows = _c10_jsonl_rows(trajectory_path)
    all_metrics = _c10_jsonl_rows(metrics_path)
    learner_metrics = [
        row for row in all_metrics
        if row.get("phase") == common.PARITY_PHASES["scout"]
        and row.get("round") == int(args.round)
        and type(row.get("grpo/step")) is int
        and "train/grpo_loss" in row
        and int(row["grpo/step"]) <= int(completed_step)
    ]
    if len(trajectory_rows) != int(completed_step) * int(
            args.prompts_per_step) * int(args.num_generations) or len(
                learner_metrics) != int(completed_step):
        raise RuntimeError(
            "C10 terminal failure cannot bind incomplete prior learner rows")
    payload = {
        "run_name": str(
            getattr(args, "orchestrator_run_name", "") or args.run_name),
        "c10_execution_profile": "canary",
        "scientific_variant": C10_SCIENTIFIC_VARIANT,
        **({"transport_invalid": True} if transport_invalid else {
            "semantic_failure": True}),
        "failure_kind": str(failure_kind),
        "completed_step": int(completed_step),
        "failed_step": int(failed_step),
        "preregistration_file": c10_evidence.C10_PREREGISTRATION_FILE,
        "preregistration_sha256":
            c10_evidence.C10_PREREGISTRATION_SHA256,
        "runtime_contract": runtime_contract,
        "runtime_contract_sha256": common.canonical_json_sha256(
            runtime_contract),
        "code_artifact": code_artifact,
        "trajectory": {
            "file_sha256": common.file_sha256(trajectory_path),
            "canonical_rows_sha256": common.canonical_json_sha256(
                trajectory_rows),
            "row_count": len(trajectory_rows),
        },
        "metrics": {
            "file_sha256": common.file_sha256(metrics_path),
            "canonical_rows_sha256": common.canonical_json_sha256(
                learner_metrics),
            "row_count": len(learner_metrics),
        },
        "failure_details": dict(failure_details),
    }
    if failed_step_candidates_path:
        candidate_rows = _c10_jsonl_rows(failed_step_candidates_path)
        payload["failed_step_candidates"] = {
            "file_sha256": common.file_sha256(failed_step_candidates_path),
            "canonical_rows_sha256": common.canonical_json_sha256(
                candidate_rows),
            "row_count": len(candidate_rows),
        }
    if failed_selected_batch_path:
        if not failed_selected_batch_manifest_path:
            raise RuntimeError(
                "C10 failed selected batch lacks its finalization manifest")
        failed_rows = _c10_jsonl_rows(failed_selected_batch_path)
        with open(failed_selected_batch_manifest_path, encoding="utf-8") \
                as handle:
            failed_manifest = json.load(handle)
        payload["failed_selected_batch"] = {
            "file_sha256": common.file_sha256(failed_selected_batch_path),
            "canonical_rows_sha256": common.canonical_json_sha256(
                failed_rows),
            "row_count": len(failed_rows),
            "manifest_file_sha256": common.file_sha256(
                failed_selected_batch_manifest_path),
            "manifest_evidence_sha256": failed_manifest.get(
                "evidence_sha256"),
        }
    if transport_invalid:
        evidence = c10_evidence.signed_training_transport(payload)
        c10_evidence.validate_training_transport_invalid(
            evidence, expected_runtime_contract=runtime_contract,
            expected_code_artifact=code_artifact)
        filename = "c10_training_transport_invalid.json"
    else:
        evidence = c10_evidence.signed_training_failure(payload)
        c10_evidence.validate_training_failure(
            evidence, expected_runtime_contract=runtime_contract,
            expected_code_artifact=code_artifact)
        filename = "c10_training_terminal_outcome.json"
    path = os.path.join(args.output_dir, filename)
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != evidence:
            raise RuntimeError(
                "C10 training terminal outcome already exists and differs")
    else:
        common.atomic_write_json(path, evidence)
    return evidence


def _signed_c10_dynamic_failure_journal(
    args: argparse.Namespace, *, completed_step: int, failed_step: int,
    runtime_contract: Dict[str, Any], code_artifact: Dict[str, Any],
    details: Dict[str, Any], candidate_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    rows = [dict(row) for row in candidate_rows]
    payload = {
        "version": C10_DYNAMIC_FAILURE_JOURNAL_VERSION,
        "run_name": str(
            getattr(args, "orchestrator_run_name", "") or args.run_name),
        "scientific_variant": C10_SCIENTIFIC_VARIANT,
        "completed_step": int(completed_step),
        "failed_step": int(failed_step),
        "runtime_contract_sha256":
            common.canonical_json_sha256(runtime_contract),
        "code_artifact": code_artifact,
        "failure_details": dict(details),
        "candidate_rows": rows,
        "candidate_rows_sha256": common.canonical_json_sha256(rows),
    }
    payload["evidence_sha256"] = common.canonical_json_sha256(payload)
    return payload


def _validate_c10_dynamic_failure_journal(
    value: Any, args: argparse.Namespace, *,
    runtime_contract: Dict[str, Any], code_artifact: Dict[str, Any],
) -> Dict[str, Any]:
    if not isinstance(value, dict) or value.get("version") != \
            C10_DYNAMIC_FAILURE_JOURNAL_VERSION:
        raise RuntimeError("C10 dynamic failure journal version mismatch")
    unsigned = dict(value)
    signature = unsigned.pop("evidence_sha256", None)
    if signature != common.canonical_json_sha256(unsigned):
        raise RuntimeError("C10 dynamic failure journal signature mismatch")
    rows = value.get("candidate_rows")
    if value.get("run_name") != str(
            getattr(args, "orchestrator_run_name", "") or args.run_name) or \
            value.get("scientific_variant") != C10_SCIENTIFIC_VARIANT or \
            value.get("runtime_contract_sha256") != \
            common.canonical_json_sha256(runtime_contract) or value.get(
                "code_artifact") != code_artifact or not isinstance(
                    value.get("failure_details"), dict) or not isinstance(
                        rows, list) or len(rows) % int(
                            args.num_generations) or value.get(
                                "candidate_rows_sha256") != \
            common.canonical_json_sha256(rows):
        raise RuntimeError("C10 dynamic failure journal identity mismatch")
    completed = value.get("completed_step")
    failed = value.get("failed_step")
    if type(completed) is not int or type(failed) is not int or \
            failed != completed + 1 or not 0 <= completed < int(args.steps):
        raise RuntimeError("C10 dynamic failure journal step mismatch")
    # Recompute the exact 64-attempt geometry before accepting the atomic
    # journal.  This also permits the one legitimate empty-candidate case: all
    # 4,096 requested rows failed transport and every attempt summary proves
    # it.  Empty semantic-exhaustion journals cannot satisfy this replay.
    from tmx_gpu import c10_evidence

    try:
        c10_evidence._replay_failed_dynamic_candidates(
            rows, value["failure_details"])
    except ValueError as exc:
        raise RuntimeError(
            "C10 dynamic failure journal retry evidence is invalid") from exc
    return value


def _write_c10_candidate_rows(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    existing = _c10_jsonl_rows(path) if os.path.isfile(path) else None
    normalized = [dict(row) for row in rows]
    if existing is not None:
        if existing != normalized:
            raise RuntimeError("C10 failed-step candidate journal differs")
        return
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            for row in normalized:
                handle.write(json.dumps(
                    row, sort_keys=True, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _finalize_c10_dynamic_failure_journal(
    args: argparse.Namespace, *, runtime_contract: Dict[str, Any],
    code_artifact: Dict[str, Any], trajectory_path: str,
) -> tuple[bool, float]:
    """Finish an atomically journaled exhaustion without resampling it."""
    from tmx_gpu import c10_evidence

    journal_path = os.path.join(
        args.output_dir, "c10_dynamic_sampling_failure_journal.json")
    try:
        with open(journal_path, encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("cannot load C10 dynamic failure journal") from exc
    journal = _validate_c10_dynamic_failure_journal(
        raw, args, runtime_contract=runtime_contract,
        code_artifact=code_artifact)
    rows = journal["candidate_rows"]
    details = journal["failure_details"]
    c10_evidence._replay_failed_dynamic_candidates(rows, details)
    candidates_path = os.path.join(
        args.output_dir, "c10_failed_step_candidates.jsonl")
    _write_c10_candidate_rows(candidates_path, rows)
    requested = int(details.get("generation_requested_row_count", 0))
    error_rows = int(details.get("generation_error_row_count", 0))
    error_rate = (
        float(error_rows) / float(requested)
        if requested > 0 else math.inf)
    transport_invalid = not math.isfinite(error_rate) or error_rate > 0.05
    recorded_details = dict(details)
    if transport_invalid:
        recorded_details["observed_generation_error_rate"] = error_rate
    _record_c10_training_failure(
        args, runtime_contract=runtime_contract,
        code_artifact=code_artifact,
        failure_kind=(
            "dynamic_sampling_transport_invalid" if transport_invalid else
            "dynamic_sampling_exhaustion"),
        completed_step=int(journal["completed_step"]),
        failed_step=int(journal["failed_step"]),
        failure_details=recorded_details,
        trajectory_path=trajectory_path,
        failed_step_candidates_path=candidates_path,
        transport_invalid=transport_invalid,
    )
    return transport_invalid, error_rate


def save_model_bf16(model, tokenizer, out_dir: str) -> str:
    """save_pretrained with bf16 safetensors + tokenizer (vLLM-loadable)."""
    os.makedirs(out_dir, exist_ok=True)
    state_dict = {
        k: (v.to(torch.bfloat16) if v.is_floating_point() else v)
        for k, v in model.state_dict().items()
    }
    model.save_pretrained(out_dir, state_dict=state_dict, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    return out_dir


def save_model_training_dtype(model, tokenizer, out_dir: str) -> str:
    """Save learner parameters without the BF16 deployment conversion.

    C8 resumes restore AdamW moments, so rounding FP32 master weights to BF16
    would make a nominal resume diverge from uninterrupted training. Step
    checkpoints therefore retain every parameter in its live learner dtype;
    vLLM sync and final deployable exports continue to use ``save_model_bf16``.
    """
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(
        out_dir, state_dict=model.state_dict(), safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    return out_dir


def load_training_examples(dataset: str, tokenizer, max_examples: int):
    """Training prompts, exactly as the TPU trainer built them.

    - Named dataset (default dapo_math_17k): tmx_jax.data.load_examples with
      the shared Qwen user instruction + chat template (lazy import: needs
      `datasets` only in training environments).
    - Local *.jsonl path ({"problem","answer"} rows): smoke-test/offline hook.
      Prompts are chat-template rendered when the tokenizer has a template,
      else plain text. Both paths honor the TMX_DATASET_SHARD_* env vars
      (deterministic shuffle by shard seed, then contiguous slice) that the
      pipeline sets for multi-round disjoint shards.
    """
    if dataset.endswith(".jsonl") and os.path.exists(dataset):
        from tmx_jax.data import _apply_round_shard
        from tmx_jax.prompting import math_messages

        class _LocalExample:
            __slots__ = ("problem_id", "prompt_text", "ground_truth")

            def __init__(self, problem_id: str, prompt_text: str, ground_truth: str):
                self.problem_id = problem_id
                self.prompt_text = prompt_text
                self.ground_truth = ground_truth

        rows: List[Dict[str, Any]] = []
        with open(dataset) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        rows = _apply_round_shard(rows[:max_examples] if max_examples else rows)
        examples = []
        for i, row in enumerate(rows):
            problem = str(row["problem"])
            try:
                prompt = tokenizer.apply_chat_template(
                    math_messages(problem),
                    tokenize=False, add_generation_prompt=True,
                )
            except Exception:  # noqa: BLE001 — tokenizer without a chat template
                prompt = math_messages(problem)[0]["content"]
            examples.append(_LocalExample(str(row.get("problem_id", i)), prompt,
                                          str(row["answer"])))
    else:
        from tmx_jax.data import load_examples

        examples = load_examples(dataset, tokenizer, max_examples=max_examples,
                                 enable_thinking="auto")
    if not examples:
        raise RuntimeError(f"no training examples loaded for dataset={dataset!r}")
    print(f"[grpo] loaded {len(examples)} training examples ({dataset})", flush=True)
    return examples


# ------------------------------------------------------- dynamic sampling ----

def _phase_step_seed(args: argparse.Namespace) -> int:
    """Per-(seed, phase) RNG base; the caller mixes in step/attempt terms."""
    return (int(args.seed) + 1) * 1_000_003 + PHASE_SEED_SALT[str(args.phase)]


def _dynamic_sample_step(
    *,
    rng: np.random.Generator,
    pool: VLLMPool,
    tokenizer,
    model,
    rnd: Optional[MultilayerRND],
    examples,
    args: argparse.Namespace,
    lam: float,
    step: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, float], float]:
    """Mirror tmx_jax.train._dynamic_sample_rollouts (groups_per_attempt=1).

    Groups containing a retry-exhausted completion are dropped BEFORE scoring:
    error rows never reach reward scoring, group means, or trajectory files
    (they are only counted in the generation_error_* stats).
    """
    G = int(args.num_generations)
    prompts_per_step = int(args.prompts_per_step)
    validity_mode = rl_validity_mode(args)
    # C9 changes only the novelty-bearing Scout.  The downstream Central
    # intentionally retains the same pipeline label for lineage, but has
    # lambda=0 and no RND object; it must remain ordinary correctness-only
    # GRPO rather than entering the selected-batch RND branch.
    c9_selected_update = (
        str(args.phase) == "scout"
        and lam > 0.0
        and rnd is not None
        and expdis_stabilization_mode(args) == C9_RND_LIFECYCLE
    )
    c9_normalization = novelty_normalization_mode(args)
    max_attempts = max(1, int(args.dynamic_max_attempts))
    max_candidate_groups = max_attempts * prompts_per_step
    selected_groups: List[List[Dict[str, Any]]] = []
    selected_feature_groups: List[Dict[int, torch.Tensor]] = []
    fallback_groups: List[Tuple[List[Dict[str, Any]], str]] = []
    all_rows: List[Dict[str, Any]] = []
    candidate_groups = kept_groups = discarded_groups = 0
    error_groups = error_rows = 0
    generated_rows_requested = 0
    candidate_survivor_rows = 0
    candidate_masked_rows = 0
    insufficient_survivor_groups = 0
    rollout_dt = 0.0
    feature_stats_acc: Dict[str, float] = {}
    rnd_candidate_rows_scored = 0
    generation_attempt_summaries: List[Dict[str, Any]] = []

    attempt = 0
    while attempt < max_attempts and candidate_groups < max_candidate_groups:
        attempt += 1
        need = prompts_per_step - len(selected_groups)
        if need <= 0:
            break
        batch_need = min(need, max_candidate_groups - candidate_groups, len(examples))
        idxs = rng.choice(len(examples), size=batch_need, replace=False).tolist()
        attempt_examples = [examples[i] for i in idxs]
        print(f"[dynamic-sampling] step={step} attempt={attempt} need={need} "
              f"candidate_groups={candidate_groups}/{max_candidate_groups} "
              f"batch_groups={len(attempt_examples)}", flush=True)
        t0 = time.perf_counter()
        completions = pool.generate(
            [e.prompt_text for e in attempt_examples],
            n=G,
            max_tokens=int(args.max_completion_len),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            top_k=int(args.top_k),
            seed_base=(_phase_step_seed(args) + step * 8191 + attempt * 131) % (2 ** 31),
            timeout=float(args.gen_timeout),
            concurrency_per_server=int(args.gen_concurrency),
            enforce_consecutive_failure_limit=(
                str(getattr(args, "c10_execution_profile", "") or "")
                != "canary"),
        )
        generated_rows_requested += len(attempt_examples) * G
        rollout_dt += time.perf_counter() - t0

        # Drop error-bearing groups pre-scoring (see docstring).
        scored_examples = []
        rows_attempt: List[Dict[str, Any]] = []
        attempt_error_groups = 0
        attempt_error_rows = 0
        for ex, comps in zip(attempt_examples, completions):
            n_err = sum(1 for c in comps if completion_failed(c))
            if n_err:
                error_groups += 1
                error_rows += n_err
                attempt_error_groups += 1
                attempt_error_rows += n_err
                print(f"[dynamic-sampling] step={step} attempt={attempt} dropped "
                      f"group ({n_err}/{len(comps)} failed generations) for "
                      f"problem {ex.problem_id}", flush=True)
                continue
            scored_examples.append(ex)
            rows_attempt.extend(score_completions(
                tokenizer, ex, comps,
                max_completion_len=int(args.max_completion_len),
                soft_overlong_expected_len=int(args.soft_overlong_expected_len),
                soft_overlong_cache_len=int(args.soft_overlong_cache_len),
            ))
        features_attempt: Optional[Dict[int, torch.Tensor]] = None
        if lam > 0.0 and rnd is not None and rows_attempt:
            layers = tuple(int(x) for x in args.novelty_layers.split(","))
            features_attempt, fstats = extract_novelty_features(
                model, tokenizer, [r["_novelty_text"] for r in rows_attempt],
                layers=layers,
                max_length=int(args.novelty_feature_max_length),
                batch_size=int(args.novelty_batch_size),
                device=args.device,
            )
            for k, v in fstats.items():
                feature_stats_acc[k] = feature_stats_acc.get(k, 0.0) + float(v)
            rnd_candidate_rows_scored += len(rows_attempt)
            if c9_selected_update:
                # C9 holds the predictor fixed across every retry/candidate in
                # this learner step. Only the final 64 selected rows update it.
                novelty_final, per_layer = rnd.score(features_attempt)
            else:
                # Frozen C8 behavior: one predictor update per candidate
                # attempt, including attempts whose groups are later rejected.
                novelty_final, per_layer = rnd.score_and_update(
                    features_attempt, train=True)
            apply_novelty_and_blend(
                rows_attempt, novelty_final, per_layer,
                lambda_novelty=lam,
                incorrect_novelty_scale=float(args.incorrect_novelty_scale),
            )
            if c9_selected_update:
                for row in rows_attempt:
                    row.update({
                        "novelty_normalization": "raw_candidate_eligibility_only",
                        "rnd_lifecycle": C9_RND_LIFECYCLE,
                        "rnd_update_scope": "not_selected",
                    })

        for group_index in range(len(scored_examples)):
            group_rows = rows_attempt[group_index * G:(group_index + 1) * G]
            candidate_groups += 1
            c10_filter = survivor_consistent_overlong_filter(args)
            reason, eligible, eligibility = group_eligibility_for_rows(
                group_rows, lambda_novelty=lam,
                validity_mode=validity_mode, survivor_only=c10_filter,
            )
            candidate_survivor_rows += int(eligibility["survivor_count"])
            candidate_masked_rows += int(eligibility["masked_count"])
            if reason == "insufficient_unmasked_survivors":
                insufficient_survivor_groups += 1
            use_group = eligible and len(selected_groups) < prompts_per_step
            for row in group_rows:
                row_is_policy_eligible = bool(
                    use_group and not bool(row.get("clipped", False)))
                row.update({
                    "used_for_training": bool(use_group),
                    "dynamic_sampling_group_selected": bool(use_group),
                    "dynamic_sampling_reason": reason,
                    "dynamic_sampling_attempt": int(attempt),
                    "dynamic_sampling_group_index": int(group_index),
                    "dynamic_sampling_correct_count": int(
                        eligibility["eligibility_correct_count"]),
                    "dynamic_sampling_reward_std": float(
                        eligibility["eligibility_reward_std"]),
                    "dynamic_sampling_novelty_contribution_std": float(
                        eligibility[
                            "eligibility_novelty_contribution_std"]),
                    "dynamic_sampling_raw_group_size": int(
                        eligibility["raw_group_size"]),
                    "dynamic_sampling_survivor_count": int(
                        eligibility["survivor_count"]),
                    "dynamic_sampling_masked_count": int(
                        eligibility["masked_count"]),
                    "dynamic_sampling_raw_correct_count": int(
                        eligibility["raw_correct_count"]),
                    "dynamic_sampling_survivor_correct_count": int(
                        eligibility["survivor_correct_count"]),
                    "dynamic_sampling_eligibility_population": str(
                        eligibility["eligibility_population"]),
                    "dynamic_sampling_eligibility_population_size": int(
                        eligibility["eligibility_population_size"]),
                    "dynamic_sampling_policy_loss_eligible":
                        row_is_policy_eligible,
                    "dynamic_sampling_eligibility_novelty_used": float(
                        row.get("novelty_used", 0.0)),
                    "dynamic_sampling_eligibility_blended_reward": float(
                        row["blended_reward"]),
                    "dynamic_sampling_eligibility_signal": (
                        "raw_correct_gated_novelty_contribution"
                        if c9_selected_update else
                        "novelty_contribution" if validity_mode == "c8" else
                        "blended_reward"),
                    "dynamic_sampling_num_generations": G,
                })
            all_rows.extend(group_rows)
            if use_group:
                selected_groups.append(group_rows)
                if c9_selected_update:
                    if features_attempt is None:
                        raise RuntimeError(
                            "C9 selected group lacks frozen-predictor features")
                    start = group_index * G
                    end = (group_index + 1) * G
                    selected_feature_groups.append({
                        layer: tensor[start:end].detach().clone()
                        for layer, tensor in features_attempt.items()
                    })
                kept_groups += 1
            else:
                # c8 is fail-closed: a group rejected by the declared dynamic-
                # sampling rule can never re-enter the learner batch.  Legacy
                # keeps the historical fixed-geometry fallback so old runs
                # remain reproducible.
                if validity_mode == "legacy":
                    fallback_groups.append((group_rows, reason))
                discarded_groups += 1
        print(f"[dynamic-sampling] attempt={attempt} complete "
              f"selected={len(selected_groups)}/{prompts_per_step} kept={kept_groups} "
              f"discarded={discarded_groups}", flush=True)
        generation_attempt_summaries.append({
            "attempt": int(attempt),
            "requested_group_count": int(len(attempt_examples)),
            "candidate_group_count": int(len(scored_examples)),
            "generation_error_group_count": int(attempt_error_groups),
            "generation_error_row_count": int(attempt_error_rows),
            "selected_group_count_after_attempt": int(len(selected_groups)),
        })

    fallback_used = 0
    # Only legacy mode can populate fallback_groups.  Keep the explicit guard
    # here as a second line of defence against future refactors silently
    # weakening c8 eligibility.
    if validity_mode == "c8" and fallback_groups:
        raise RuntimeError("c8 dynamic sampling cannot use fallback groups")
    while len(selected_groups) < prompts_per_step and fallback_groups:
        group_rows, reason = fallback_groups.pop(0)
        for row in group_rows:
            row["used_for_training"] = True
            row["dynamic_sampling_reason"] = f"fallback_{reason}"
            row["dynamic_sampling_fallback"] = True
        selected_groups.append(group_rows)
        fallback_used += 1
    if len(selected_groups) != prompts_per_step:
        if str(getattr(args, "c10_execution_profile", "") or "") == \
                "canary":
            group_summaries = []
            for start in range(0, len(all_rows), G):
                group = all_rows[start:start + G]
                if len(group) != G:
                    raise RuntimeError(
                        "C10 exhaustion evidence has partial candidate group")
                group_summaries.append({
                    "problem_id": str(group[0].get("problem_id", "")),
                    "attempt": int(group[0].get(
                        "dynamic_sampling_attempt", -1)),
                    "group_index": int(group[0].get(
                        "dynamic_sampling_group_index", -1)),
                    "reason": str(group[0].get(
                        "dynamic_sampling_reason", "")),
                    "survivor_count": int(group[0].get(
                        "dynamic_sampling_survivor_count", -1)),
                    "masked_count": int(group[0].get(
                        "dynamic_sampling_masked_count", -1)),
                    "survivor_correct_count": int(group[0].get(
                        "dynamic_sampling_survivor_correct_count", -1)),
                    "eligibility_reward_std": float(group[0].get(
                        "dynamic_sampling_reward_std", math.nan)),
                    "eligibility_novelty_contribution_std": float(
                        group[0].get(
                            "dynamic_sampling_novelty_contribution_std",
                            math.nan)),
                    "selected": bool(group[0].get(
                        "dynamic_sampling_group_selected", False)),
                })
            compact_candidate_rows = []
            for group_ordinal, start in enumerate(
                    range(0, len(all_rows), G)):
                group = all_rows[start:start + G]
                for row_index, row in enumerate(group):
                    compact_candidate_rows.append({
                        "candidate_group_ordinal": group_ordinal,
                        "row_index": row_index,
                        "problem_id": str(row.get("problem_id", "")),
                        "dynamic_sampling_attempt": int(row.get(
                            "dynamic_sampling_attempt", -1)),
                        "dynamic_sampling_group_index": int(row.get(
                            "dynamic_sampling_group_index", -1)),
                        "clipped": bool(row.get("clipped", False)),
                        "is_correct": bool(row.get("is_correct", False)),
                        "blended_reward": float(row.get(
                            "blended_reward", math.nan)),
                        "novelty_used": float(row.get(
                            "novelty_used", math.nan)),
                        "dynamic_sampling_reason": str(row.get(
                            "dynamic_sampling_reason", "")),
                        "dynamic_sampling_group_selected": bool(row.get(
                            "dynamic_sampling_group_selected", False)),
                        "dynamic_sampling_survivor_count": int(row.get(
                            "dynamic_sampling_survivor_count", -1)),
                        "dynamic_sampling_masked_count": int(row.get(
                            "dynamic_sampling_masked_count", -1)),
                        "dynamic_sampling_raw_correct_count": int(row.get(
                            "dynamic_sampling_raw_correct_count", -1)),
                        "dynamic_sampling_survivor_correct_count": int(row.get(
                            "dynamic_sampling_survivor_correct_count", -1)),
                        "dynamic_sampling_eligibility_population": str(row.get(
                            "dynamic_sampling_eligibility_population", "")),
                        "dynamic_sampling_eligibility_population_size": int(
                            row.get(
                                "dynamic_sampling_eligibility_population_size",
                                -1)),
                        "dynamic_sampling_reward_std": float(row.get(
                            "dynamic_sampling_reward_std", math.nan)),
                        "dynamic_sampling_novelty_contribution_std": float(
                            row.get(
                                "dynamic_sampling_novelty_contribution_std",
                                math.nan)),
                    })
            raise C10DynamicSamplingExhaustion({
                "dynamic_max_attempts": int(max_attempts),
                "selected_group_count": int(len(selected_groups)),
                "needed_group_count": int(prompts_per_step),
                "candidate_group_count": int(candidate_groups),
                "discarded_group_count": int(discarded_groups),
                "generation_error_group_count": int(error_groups),
                "generation_error_row_count": int(error_rows),
                "generation_requested_row_count": int(
                    generated_rows_requested),
                "generation_attempt_summaries_sha256":
                    common.canonical_json_sha256(
                        generation_attempt_summaries),
                "generation_attempt_summaries":
                    generation_attempt_summaries,
                "candidate_group_summaries_sha256":
                    common.canonical_json_sha256(group_summaries),
                "candidate_group_summaries": group_summaries,
            }, candidate_rows=compact_candidate_rows)
        raise RuntimeError(
            f"dynamic sampling failed to assemble batch: selected={len(selected_groups)} "
            f"needed={prompts_per_step} attempts={max_attempts}"
        )

    selected_rows = [row for group in selected_groups for row in group]
    c9_profile = str(getattr(args, "c9_execution_profile", "") or "")
    if c9_profile:
        for row in selected_rows:
            row.update({
                "scientific_variant": str(
                    getattr(args, "scientific_variant", "") or ""),
                "c9_execution_profile": c9_profile,
                "stabilization_cell": str(
                    getattr(args, "stabilization_cell", "") or ""),
            })
    c10_profile = str(getattr(args, "c10_execution_profile", "") or "")
    if c10_profile:
        for row in selected_rows:
            row.update({
                "scientific_variant": str(
                    getattr(args, "scientific_variant", "") or ""),
                "c10_execution_profile": c10_profile,
            })
    c9_stats: Dict[str, float] = {}
    if c9_selected_update:
        if rnd is None or lam <= 0.0:
            message = "C9 selected-batch lifecycle requires active RND"
            if c10_profile == "canary":
                raise C10TrainingHealthFailure(
                    "selected_batch_rnd_update_failure", message,
                    {"rnd_present": rnd is not None,
                     "lambda_novelty": float(lam)})
            raise RuntimeError(message)
        if len(selected_feature_groups) != len(selected_groups):
            message = "C9 selected feature groups do not match learner groups"
            if c10_profile == "canary":
                raise C10TrainingHealthFailure(
                    "selected_batch_rnd_update_failure", message, {
                        "selected_feature_group_count":
                            len(selected_feature_groups),
                        "selected_group_count": len(selected_groups),
                    })
            raise RuntimeError(message)
        selected_features = {
            layer: torch.cat(
                [group[layer] for group in selected_feature_groups], dim=0)
            for layer in rnd.layers
        }
        expected_rows = prompts_per_step * G
        feature_rows = {
            layer: int(tensor.shape[0])
            for layer, tensor in selected_features.items()
        }
        if set(feature_rows.values()) != {expected_rows}:
            message = (
                "C9 RND update must contain exactly the final selected rows: "
                f"expected={expected_rows}, per_layer={feature_rows}")
            if c10_profile == "canary":
                raise C10TrainingHealthFailure(
                    "selected_batch_rnd_update_failure", message, {
                        "expected_rows": expected_rows,
                        "per_layer_feature_rows": feature_rows,
                    })
            raise RuntimeError(message)
        try:
            c9_stats.update(recompute_selected_novelty_and_blend(
                selected_rows,
                normalization=c9_normalization,
                lambda_novelty=lam,
                incorrect_novelty_scale=float(args.incorrect_novelty_scale),
            ))
        except (ValueError, RuntimeError) as exc:
            if c10_profile == "canary":
                raise C10TrainingHealthFailure(
                    "selected_batch_novelty_failure", str(exc), {
                        "normalization": c9_normalization,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }) from exc
            raise
        before_updates = int(rnd.selected_batch_update_count)
        c9_stats["rnd_selected_batch_loss"] = float(
            rnd.update_selected_batch(selected_features))
        if int(rnd.selected_batch_update_count) != before_updates + 1:
            message = "C9 RND predictor did not advance exactly once"
            if c10_profile == "canary":
                raise C10TrainingHealthFailure(
                    "selected_batch_rnd_update_failure", message, {
                        "before_update_count": before_updates,
                        "after_update_count": int(
                            rnd.selected_batch_update_count),
                    })
            raise RuntimeError(message)
        c9_stats.update({
            "rnd_candidate_rows_scored": float(rnd_candidate_rows_scored),
            "rnd_update_rows": float(expected_rows),
            "rnd_updates_this_step": 1.0,
            "rnd_selected_batch_update_count": float(
                rnd.selected_batch_update_count),
        })
    stats = {
        "candidate_group_count": float(candidate_groups),
        "kept_group_count": float(kept_groups),
        "discarded_group_count": float(max(0, discarded_groups - fallback_used)),
        "fallback_group_count": float(fallback_used),
        "generation_error_group_count": float(error_groups),
        "generation_error_row_count": float(error_rows),
        "generation_requested_row_count": float(generated_rows_requested),
        "generation_error_rate": (
            float(error_rows) / max(1.0, float(generated_rows_requested))
        ),
        "candidate_survivor_row_count": float(candidate_survivor_rows),
        "candidate_masked_row_count": float(candidate_masked_rows),
        "insufficient_survivor_group_count": float(
            insufficient_survivor_groups),
        "selected_raw_row_count": float(len(selected_rows)),
        "selected_survivor_row_count": float(sum(
            1 for row in selected_rows if not bool(row.get("clipped", False)))),
        "selected_masked_row_count": float(sum(
            1 for row in selected_rows if bool(row.get("clipped", False)))),
        **c9_stats,
    }
    feat_count = max(1.0, feature_stats_acc.get("count", 0.0))
    stats["novelty_feature_truncated_fraction"] = feature_stats_acc.get("truncated_count", 0.0) / feat_count
    return selected_rows, all_rows, stats, rollout_dt


# ------------------------------------------------------------ train update ----

def _row_logps(
    model,
    prompt_ids: List[int],
    comp_ids: List[int],
    args: argparse.Namespace,
    *,
    requires_grad: bool,
) -> torch.Tensor:
    """Per-token completion logprobs for one [prompt|completion] row."""
    device = args.device
    use_autocast = torch.device(device).type == "cuda" and str(args.param_dtype) == "float32"
    p_len, c_len = len(prompt_ids), len(comp_ids)
    input_ids = torch.tensor([prompt_ids + comp_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    targets = torch.tensor(comp_ids, dtype=torch.long, device=device)
    ctx = torch.enable_grad() if requires_grad else torch.no_grad()
    with ctx, torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast):
        hidden = model.model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).last_hidden_state[0]
        # logits at position t predict token t+1: completion targets at
        # positions [p_len, p_len+c_len) come from hidden [p_len-1, ...).
        return chunked_selective_log_softmax(
            hidden[p_len - 1: p_len + c_len - 1], model.lm_head, targets,
            chunk_tokens=int(args.chunk_tokens),
        )


def _prompt_ids(tokenizer, cache: Dict[str, List[int]], prompt: str, max_prompt_len: int) -> List[int]:
    if prompt not in cache:
        pids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        cache[prompt] = [int(x) for x in pids[-int(max_prompt_len):]]
    return cache[prompt]


def _train_update(
    model,
    optimizer,
    tokenizer,
    rows: List[Dict[str, Any]],
    advantages: np.ndarray,
    args: argparse.Namespace,
) -> Dict[str, float]:
    """Optimizer update(s) for one rollout batch, fixed Dr.GRPO denominator.

    CONTRACT tier: REINFORCE — one update, loss = -(adv * sum logp) / denom
    per row, grad accumulation across all rows, single step; old_logp is the
    detached current logp so ratio == 1 and clip is inert (clip_fraction 0).
    GPU-MAX --ppo-mode: old logprobs are snapshotted from the policy right
    after generation and BEFORE any update; K = --updates-per-rollout epochs
    of asymmetric-clip PPO (min(r*A, clip(r, 1-eps_low, 1+eps_high)*A)) run
    over the same batch, one optimizer step per epoch.
    """
    ppo = bool(args.ppo_mode)
    if not rows or len(rows) != len(advantages):
        raise ValueError("an update requires one advantage per selected rollout row")
    num_updates = max(1, int(args.updates_per_rollout)) if ppo else 1
    eps_low = float(args.clip_epsilon_low)
    eps_high = float(args.clip_epsilon_high)
    denominator = float(len(rows) * int(getattr(args, "loss_denominator_len", 0)
                                        or args.max_completion_len))
    prompt_cache: Dict[str, List[int]] = {}

    # Trainable rows: zero-advantage / empty rows contribute exactly 0 gradient
    # under both objectives (min(r*0, clip(r)*0) == 0), so skip their forwards.
    # Optional DAPO Overlong Filtering (--mask-truncated): clipped completions
    # are excluded from the policy loss. The canonical c8 replication leaves
    # this OFF, matching the measured TPU reference, and uses soft-overlong
    # reward shaping alone. The historical c7 GPU campaign enabled it.
    n_masked_truncated = 0
    n_survivor_rows = 0
    n_zero_advantage_survivors = 0
    work: List[Tuple[Dict[str, Any], float, List[int], List[int]]] = []
    for row, adv in zip(rows, advantages):
        comp_ids = row.get("_comp_ids") or []
        if bool(getattr(args, "mask_truncated", False)) and row.get("clipped"):
            n_masked_truncated += 1
            continue
        n_survivor_rows += 1
        if not comp_ids or abs(float(adv)) < 1e-12:
            n_zero_advantage_survivors += 1
            continue
        pids = _prompt_ids(tokenizer, prompt_cache, row["prompt_text"], args.max_prompt_len)
        work.append((row, float(adv), pids, comp_ids))

    old_logps: List[Optional[torch.Tensor]] = [None] * len(work)
    if ppo and work:
        # Frozen old-policy snapshot: rescore before ANY update this step.
        for i, (_row, _adv, pids, comp_ids) in enumerate(work):
            old_logps[i] = _row_logps(model, pids, comp_ids, args,
                                      requires_grad=False).detach().float().cpu()

    loss_total = 0.0
    grad_norm = 0.0
    logp_sum = 0.0
    token_count = 0
    clip_fraction = 0.0
    policy_kl = 0.0
    for update_idx in range(num_updates):
        optimizer.zero_grad(set_to_none=True)
        update_loss = 0.0
        # Per-epoch token stats; the last epoch's values are reported,
        # matching the TPU per-update metric semantics.
        out_of_range_tokens = 0
        kl_sum = 0.0
        epoch_tokens = 0
        for i, (_row, adv, pids, comp_ids) in enumerate(work):
            logps = _row_logps(model, pids, comp_ids, args, requires_grad=True)
            if ppo:
                old = old_logps[i].to(logps.device)
                log_ratio = logps - old
                ratio = torch.exp(log_ratio)
                unclipped = ratio * adv
                clipped = torch.clamp(ratio, 1.0 - eps_low, 1.0 + eps_high) * adv
                row_loss = -torch.minimum(unclipped, clipped).sum() / denominator
                # TPU clip_fraction counts ratio-out-of-range tokens over the
                # completion mask regardless of advantage sign
                # (tmx_jax/grpo.py), NOT min()-active tokens.
                out_of_range_tokens += int(
                    ((ratio < 1.0 - eps_low) | (ratio > 1.0 + eps_high)).sum())
                lr_det = log_ratio.detach()
                kl_sum += float((torch.exp(lr_det) - 1.0 - lr_det).sum())
                epoch_tokens += int(logps.shape[0])
            else:
                row_loss = -(adv * logps.sum()) / denominator
                epoch_tokens += int(logps.shape[0])
            row_loss.backward()
            update_loss += float(row_loss.detach())
            if update_idx == num_updates - 1:
                logp_sum += float(logps.detach().sum())
                token_count += len(comp_ids)
        if not work:
            # A zero-gradient batch still advances Adam's moments and step
            # counter. Skipping step() silently changes the fixed update budget
            # and its learning-rate schedule after dynamic-sampling fallback.
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    if parameter.requires_grad:
                        parameter.grad = torch.zeros_like(parameter)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip)))
        optimizer.step()
        loss_total = update_loss  # report the last epoch's loss
        if ppo and epoch_tokens:
            clip_fraction = out_of_range_tokens / epoch_tokens
            policy_kl = kl_sum / epoch_tokens
    optimizer.zero_grad(set_to_none=True)
    return {
        "loss": loss_total,
        "grad_norm": grad_norm,
        "rows_backward": float(len(work)),
        "masked_truncated_rows": float(n_masked_truncated),
        "raw_selected_rows": float(len(rows)),
        "survivor_rows": float(n_survivor_rows),
        "zero_advantage_survivor_rows": float(n_zero_advantage_survivors),
        "fixed_loss_denominator": float(denominator),
        "updates": float(num_updates),
        "selected_token_logprob_mean": logp_sum / max(1, token_count),
        # REINFORCE: ratio == 1 by construction => clip inert, KL exactly 0.
        "clip_fraction": float(clip_fraction),
        "policy_kl": float(policy_kl),
    }


# ------------------------------------------------------------- checkpoints ----

def _save_optimizer_state(optimizer, directory: str) -> str:
    """Atomically persist optimizer moments for a c8 crash-resume checkpoint."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, OPTIMIZER_STATE_FILENAME)
    tmp = path + ".tmp"
    try:
        torch.save(optimizer.state_dict(), tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return path


def _restore_optimizer_state(optimizer, directory: str, *, device: str,
                             required: bool) -> bool:
    """Restore optimizer moments; c8 fails closed if they are unavailable."""
    path = os.path.join(directory, OPTIMIZER_STATE_FILENAME)
    if not os.path.exists(path):
        if required:
            raise RuntimeError(
                f"c8 resume requires optimizer state, but {path} is missing; "
                "resume from a c8 checkpoint or restart the canary fresh")
        return False
    try:
        state = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch versions predating the weights_only keyword
        state = torch.load(path, map_location=device)
    optimizer.load_state_dict(state)
    return True


def _fresh_c8_component_artifact_metadata(
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Propagate exact producer identity to every artifact that can escape."""
    profile = str(getattr(args, "fresh_c8_evidence_profile", "") or "")
    if not profile:
        return {}
    seed = getattr(args, "seed", None)
    if profile == "fresh_c8_multiscout_component":
        if type(seed) is not int or seed not in (1, 2):
            raise ValueError(
                "component artifact provenance requires exact integer seed 1 or 2")
        return {
            "fresh_c8_evidence_profile": profile,
            "component_seed": seed,
        }
    if profile == "fresh_c8_single_scout":
        if type(seed) is not int or seed != 0:
            raise ValueError(
                "canonical artifact provenance requires exact integer seed 0")
        return {
            "fresh_c8_evidence_profile": profile,
            "seed": seed,
        }
    raise ValueError(f"unknown fresh-c8 artifact evidence profile {profile!r}")


def _c9_artifact_metadata(args: argparse.Namespace) -> Dict[str, Any]:
    """Immutable artifact labels for the opt-in C9 stabilization path."""
    lifecycle = expdis_stabilization_mode(args)
    profile = str(getattr(args, "c9_execution_profile", "") or "")
    if lifecycle != C9_RND_LIFECYCLE and not profile:
        return {}
    phase = str(getattr(args, "phase", "") or "")
    metadata: Dict[str, Any] = {
        "expdis_stabilization_mode": lifecycle,
        "scientific_variant": str(
            getattr(args, "scientific_variant", "") or ""),
        "c9_execution_profile": str(
            getattr(args, "c9_execution_profile", "") or ""),
        "c10_execution_profile": str(
            getattr(args, "c10_execution_profile", "") or ""),
        "stabilization_cell": str(
            getattr(args, "stabilization_cell", "") or ""),
        "novelty_normalization": novelty_normalization_mode(args),
    }
    if phase:
        metadata["novelty_active"] = phase == "scout"
    else:
        metadata["novelty_active_phase"] = "scout_only"
    if lifecycle == C9_RND_LIFECYCLE and phase != "central":
        # A pipeline/results artifact has no single phase and describes the
        # complete C9 method, so it carries the Scout lifecycle fields.  A
        # Central model artifact explicitly records novelty_active=false and
        # must not claim that its nonexistent RND updated once per step.
        metadata.update({
            "rnd_scoring_predictor_state": "frozen_for_entire_learner_step",
            "rnd_update_scope": "final_selected_learner_rows",
            "rnd_updates_per_completed_step": 1,
        })
    if str(getattr(args, "c10_execution_profile", "") or ""):
        metadata["survivor_overlong_filter_version"] = \
            C10_OVERLONG_FILTER_CONTRACT_VERSION
    return metadata


def _mrme_artifact_metadata(args: argparse.Namespace) -> Dict[str, Any]:
    """Bind every MR-ME Scout checkpoint/artifact to its campaign slot."""
    from tmx_gpu import mrme_contract

    contract_sha = str(getattr(args, "mrme_contract_sha256", "") or "")
    if not contract_sha:
        return {}
    shard_sha = str(
        getattr(args, "mrme_shard_manifest_sha256", "") or "")
    scout_index = int(getattr(args, "mrme_scout_index", 0) or 0)
    scouts_per_round = int(getattr(args, "mrme_scouts_per_round", 0) or 0)
    method = str(getattr(args, "mrme_method", "") or "")
    method_version = str(getattr(args, "mrme_method_version", "") or "")
    single_scout_multiround = bool(getattr(
        args, "single_scout_multiround", False))
    exact_k1 = (
        scouts_per_round == 1
        and method ==
        mrme_contract.DISJOINT_SHARD_SINGLESCOUT_MULTIROUND_METHOD
        and method_version ==
        mrme_contract.DISJOINT_SHARD_SINGLESCOUT_MULTIROUND_METHOD_VERSION
        and single_scout_multiround
    )
    exact_multi = (
        scouts_per_round > 1
        and method in {
            mrme_contract.INTEGRATED_MULTISCOUT_METHOD,
            mrme_contract.DISJOINT_SHARD_MRME_METHOD,
        }
        and not method_version
        and not single_scout_multiround
    )
    phase = str(getattr(args, "phase", ""))
    if (
        phase not in {"scout", "central"}
        or not re.fullmatch(r"[0-9a-f]{64}", contract_sha)
        or not re.fullmatch(r"[0-9a-f]{64}", shard_sha)
        or not (exact_k1 or exact_multi)
        or (phase == "scout" and not (1 <= scout_index <= scouts_per_round))
        or (phase == "central" and scout_index != 0)
    ):
        raise ValueError("invalid multi-explorer stage artifact provenance")
    metadata = {
        "method": method,
        "stabilization_profile": str(
            getattr(args, "mrme_stabilization_profile", "") or ""),
        "scientific_variant": str(
            getattr(args, "scientific_variant", "") or ""),
        "mrme_contract_sha256": contract_sha,
        "mrme_shard_manifest_sha256": shard_sha,
        "mrme_scout_index": scout_index,
        "mrme_scouts_per_round": scouts_per_round,
        "stage_seed": int(args.seed),
        "phase": phase,
    }
    if exact_k1:
        metadata.update({
            "method_version": method_version,
            "single_scout_multiround": True,
            "mrme": False,
            "multi_explorer": False,
        })
    if phase == "scout":
        # Backward-compatible key consumed by the Scout source validator.
        metadata["mrme_scout_seed"] = int(args.seed)
        metadata.pop("stage_seed")
        metadata.pop("phase")
    return metadata


def _mrme_trajectory_provenance(
    args: argparse.Namespace,
) -> Optional[Dict[str, Any]]:
    """Return the immutable MR-ME Scout slot embedded in every raw row.

    The surrounding source record already binds the JSONL bytes.  Embedding
    the campaign slot in each row closes a different substitution channel: a
    valid Scout-A JSONL cannot be copied to a new path and advertised as
    Scout-B by rebuilding only the outer source metadata.  Central rollouts
    and ordinary SingleScout trajectories deliberately carry no such fields.
    """
    if not str(getattr(args, "mrme_contract_sha256", "") or ""):
        return None
    if str(getattr(args, "phase", "")) != "scout":
        return None
    metadata = _mrme_artifact_metadata(args)
    return {
        "mrme_trajectory_provenance_version":
            "gpu_mrme_trajectory_slot_v1",
        "mrme_contract_sha256": metadata["mrme_contract_sha256"],
        "mrme_method": metadata["method"],
        "mrme_stabilization_profile": metadata["stabilization_profile"],
        "mrme_round_index": int(args.round),
        "mrme_scout_index": metadata["mrme_scout_index"],
        "mrme_scout_seed": metadata["mrme_scout_seed"],
        "mrme_shard_manifest_sha256":
            metadata["mrme_shard_manifest_sha256"],
    }


def _log_checkpoint_progress_artifact(
    wandb_run, *, args: argparse.Namespace, step: int, checkpoint_dir: str,
):
    """Durably pair append-only run evidence with a step checkpoint.

    Model/optimizer bytes and their trainer state live in the checkpoint
    artifact.  This smaller companion artifact carries the exact metrics and
    rollout records through the same learner step, allowing a fresh pod to
    resume without silently losing the pre-crash portion of the run.
    """
    import wandb

    name = re.sub(
        r"[^a-zA-Z0-9_.-]", "-",
        f"{args.run_name}-{args.phase}-r{int(args.round)}-progress",
    )
    code_binding = getattr(args, "_code_artifact_binding", None)
    metadata = {
        "step": int(step),
        "round": int(args.round),
        "phase": str(args.phase),
        "model": str(args.model),
        "scientific_variant": str(
            getattr(args, "scientific_variant", "") or ""),
        "code_artifact_ref": (
            code_binding.get("source", {}).get("artifact_ref")
            if isinstance(code_binding, dict) else ""),
        "code_artifact_digest": (
            code_binding.get("source", {}).get("artifact_digest")
            if isinstance(code_binding, dict) else ""),
        **_c9_artifact_metadata(args),
        **_fresh_c8_component_artifact_metadata(args),
        **_mrme_artifact_metadata(args),
    }
    artifact = wandb.Artifact(
        name,
        type="grpo-progress",
        metadata=metadata,
    )
    candidates = [
        (str(getattr(args, "metrics_jsonl", "") or ""), "metrics.jsonl"),
        (os.path.join(args.output_dir, "explorer_trajectories.jsonl"),
         "explorer_trajectories.jsonl"),
        (os.path.join(args.output_dir, "rollouts_all.jsonl"),
         "rollouts_all.jsonl"),
        (os.path.join(args.output_dir, "provenance", "dapo_code_binding.json"),
         "dapo_code_binding.json"),
        (str(getattr(args, "code_source_manifest", "") or ""),
         "dapo_code_manifest.json"),
        (os.path.join(checkpoint_dir, "trainer_state.json"),
         f"step_{int(step):06d}/trainer_state.json"),
    ]
    added = []
    for path, artifact_name in candidates:
        if path and os.path.isfile(path):
            artifact.add_file(path, name=artifact_name)
            added.append(artifact_name)
    required = {
        "metrics.jsonl",
        "explorer_trajectories.jsonl",
        "rollouts_all.jsonl",
        f"step_{int(step):06d}/trainer_state.json",
    }
    if not required.issubset(set(added)):
        raise RuntimeError(
            "checkpoint progress artifact lacks required run evidence: "
            f"{sorted(required - set(added))}")
    logged = wandb_run.log_artifact(
        artifact, aliases=["latest", f"step-{int(step)}"])
    logged.wait()
    return logged


def _save_checkpoint(model, tokenizer, rnd, optimizer, args, step: int, lam: float,
                     wandb_run=None) -> str:
    """Local step checkpoint; on the --ckpt-artifact-every cadence also push it
    to W&B (the only durable storage on the Lepton workspace — what a retried job
    downloads and passes back via --resume-from)."""
    code_artifact_binding = getattr(args, "_code_artifact_binding", None)
    if str(getattr(args, "code_source_manifest", "") or "") and \
            code_artifact_binding is None:
        raise RuntimeError(
            "code-bound Scout checkpoint lacks validated code artifact binding")
    validity_mode = rl_validity_mode(args)
    optimizer_state_saved = validity_mode == "c8"
    policy_attention = str(
        getattr(args, "_policy_attention_implementation", "") or "")
    if validity_mode == "c8" and policy_attention not in {
            "flash_attention_2", "sdpa"}:
        raise RuntimeError(
            "c8 checkpoint lacks the resolved policy attention implementation")
    ckpt_dir = os.path.join(args.output_dir, f"step_{step:06d}")
    save_model_training_dtype(model, tokenizer, ckpt_dir)
    if rnd is not None:
        rnd.save(os.path.join(ckpt_dir, "rnd_state.pt"))
    if optimizer_state_saved:
        _save_optimizer_state(optimizer, ckpt_dir)
    external_lineage = getattr(args, "_external_lineage", None)
    runtime_contract = getattr(args, "_runtime_contract", None)
    c9_rnd_update_count = (
        int(rnd.selected_batch_update_count)
        if rnd is not None
        and expdis_stabilization_mode(args) == C9_RND_LIFECYCLE
        and str(args.phase) == "scout"
        else None
    )
    c9_canary_health_state = getattr(args, "_c9_canary_health_state", None)
    c10_canary_health_state = getattr(args, "_c10_canary_health_state", None)
    if external_lineage is not None and not isinstance(runtime_contract, dict):
        raise RuntimeError("external Central checkpoint lacks resolved runtime contract")
    file_manifest = common.directory_file_manifest(
        ckpt_dir, exclude_relative_paths=("trainer_state.json",))
    # Atomic: a torn trainer_state.json (crash mid-write) would poison every
    # subsequent --resume-from into a crash loop.
    common.atomic_write_json(
        os.path.join(ckpt_dir, "trainer_state.json"),
        {"step": int(step), "phase": args.phase, "lambda": float(lam),
         "round": int(args.round), "model": args.model,
         "model_revision": str(args.model_revision or ""),
         "config_tier": resolved_config_tier(args),
         "rl_validity_mode": validity_mode,
         "optimizer_state_saved": optimizer_state_saved,
         "checkpoint_parameter_dtype": str(args.param_dtype),
         "policy_attention_implementation": policy_attention,
         "resolved_lr": float(optimizer.param_groups[0]["lr"]),
         "external_lineage": external_lineage,
         "external_lineage_sha256": (
             (external_lineage.get("lineage_sha256")
              or external_lineage.get("evidence_sha256"))
             if external_lineage else None),
         "runtime_contract": runtime_contract,
         "code_artifact_binding": code_artifact_binding,
         **({"mrme_provenance": _mrme_artifact_metadata(args)}
            if getattr(args, "mrme_contract_sha256", "") else {}),
         "rnd_selected_batch_update_count": c9_rnd_update_count,
         "c9_canary_health_state": c9_canary_health_state,
         "c10_canary_health_state": c10_canary_health_state,
         "checkpoint_file_manifest": file_manifest})
    if (wandb_run is not None and int(args.ckpt_artifact_every) > 0
            and step % int(args.ckpt_artifact_every) == 0):
        logged_checkpoint = common.log_dir_artifact(
            wandb_run,
            name=f"{args.run_name}-{args.phase}-r{int(args.round)}-ckpt",
            path=ckpt_dir, type_="checkpoint",
            metadata={
                "step": int(step), "round": int(args.round),
                "phase": str(args.phase),
                "model_revision": str(args.model_revision or ""),
                "runtime_contract_sha256": (
                    common.canonical_json_sha256(runtime_contract)
                    if isinstance(runtime_contract, dict) else None),
                "training_dataset_identity_sha256": (
                    runtime_contract.get("training_dataset_identity", {}).get(
                        "identity_sha256")
                    if isinstance(runtime_contract, dict) else None),
                **_c9_artifact_metadata(args),
                **_fresh_c8_component_artifact_metadata(args),
                **_mrme_artifact_metadata(args),
            },
        )
        logged_checkpoint.wait()
        _log_checkpoint_progress_artifact(
            wandb_run, args=args, step=step, checkpoint_dir=ckpt_dir)
        print(f"[grpo] checkpoint + progress step {step} durably pushed to W&B",
              flush=True)
    ckpts = sorted(glob.glob(os.path.join(args.output_dir, "step_[0-9]*")))
    for old in ckpts[:-max(1, int(args.keep_checkpoints))]:
        shutil.rmtree(old, ignore_errors=True)
    print(f"[grpo] checkpoint saved -> {ckpt_dir} (keeping last {args.keep_checkpoints})", flush=True)
    return ckpt_dir


def _sync_vllm(model, tokenizer, pool: Optional[VLLMPool], args,
               step: int) -> Tuple[Optional[str], float]:
    """Export bf16 weights and restart the vLLM pool on them (weight sync).

    Sync dirs are namespaced by phase AND round (step numbering restarts each
    round) so pruning stays chronological across a multi-round run; the dir
    just reloaded is never deleted (pool.model_dir must stay valid for
    health_check relaunches). Returns (sync_dir, seconds) — the cadence is
    logged to W&B per the plan (a TPU reload cadence once caused a KL collapse;
    reload behavior is evidence, not just ops).
    """
    if pool is None or not pool.managed:
        print("[grpo] WARNING: no managed vLLM pool; skipping weight sync "
              f"at step {step} (rollouts go stale)", flush=True)
        return None, 0.0
    sync_root = args.sync_dir or args.output_dir
    sync_dir = os.path.join(
        sync_root, f"policy_{args.phase}_r{int(args.round):02d}_step_{step:06d}")
    t0 = time.perf_counter()
    save_model_bf16(model, tokenizer, sync_dir)
    pool.reload(sync_dir)
    dt = time.perf_counter() - t0
    print(f"[grpo] vLLM weight sync at step {step} -> {sync_dir} ({dt:.1f}s)", flush=True)
    stale = sorted(glob.glob(os.path.join(sync_root, f"policy_{args.phase}_r*_step_*")))
    for old in stale[:-2]:  # keep current + previous export
        if os.path.abspath(old) != os.path.abspath(sync_dir):
            shutil.rmtree(old, ignore_errors=True)
    return sync_dir, dt


# ------------------------------------------------------------------- probe ----

def _load_probe_examples(args: argparse.Namespace, tokenizer) -> List[Tuple[str, str]]:
    """Fixed (prompt, ground_truth) probe set.

    Default: the full 30 AIME24 problems rendered with the eval-contract
    system prompt/chat template (eval_gpu). --probe-dataset may point at a
    local {"problem","answer"} JSONL for smoke tests; tokenizers without a
    chat template fall back to a plain-text prompt.
    """
    from tmx_gpu import eval_gpu

    if eval_gpu.SAMPLE_SEED_RULE != PROBE_SAMPLE_SEED_RULE or \
            eval_gpu.BENCHMARKS["AIME24"]["revision"] != \
            PROBE_BENCHMARK_REVISION or \
            eval_gpu.BENCHMARKS["AIME24"]["canonical_rows_sha256"] != \
            PROBE_BENCHMARK_ROWS_SHA256:
        raise RuntimeError("diagnostic probe drifted from canonical P1 identity")

    dataset = str(getattr(args, "probe_dataset", "") or "aime24")
    if dataset.lower() == "aime24":
        raw_rows = eval_gpu.load_benchmark("AIME24")
        observed_rows_sha256 = eval_gpu.validate_loaded_benchmark(
            "AIME24", raw_rows, publication=True)
        if observed_rows_sha256 != PROBE_BENCHMARK_ROWS_SHA256:
            raise RuntimeError(
                "diagnostic probe AIME24 row identity does not match the "
                "frozen P1 contract")
        rows = [{"problem": r["problem"], "ground_truth": r["ground_truth"]}
                for r in raw_rows]
    else:
        rows = []
        with open(dataset) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    rows.append({"problem": str(r["problem"]),
                                 "ground_truth": str(r["answer"])})
    probe: List[Tuple[str, str]] = []
    for r in rows:
        if dataset.lower() == "aime24":
            # Canonical evidence must never silently switch prompt formats.
            prompt = eval_gpu.render_prompt(tokenizer, r["problem"], "auto")
        else:
            try:
                prompt = eval_gpu.render_prompt(tokenizer, r["problem"], "auto")
            except Exception:  # smoke-only custom datasets may lack a template
                prompt = eval_gpu.math_messages(r["problem"])[0]["content"]
        probe.append((prompt, r["ground_truth"]))
    return probe


def _run_probe(pool: VLLMPool, probe_examples: List[Tuple[str, str]],
               args: argparse.Namespace) -> Dict[str, Any]:
    """One mid-training AIME24 probe pass through the serving pool.

    Fixed protocol (constants above) + fixed request seeds, so successive
    probes measure the same subset under the same sampling. Any failed request
    makes the protocol incomplete, so no accuracy key is emitted.
    """
    from tmx_gpu import eval_gpu

    completions = pool.generate(
        [p for p, _ in probe_examples],
        n=PROBE_NUM_SAMPLES,
        max_tokens=PROBE_MAX_TOKENS,
        temperature=PROBE_TEMPERATURE,
        top_p=PROBE_TOP_P,
        top_k=PROBE_TOP_K,
        min_p=PROBE_MIN_P,
        seed_base=PROBE_SEED_BASE,
        timeout=float(args.gen_timeout),
        concurrency_per_server=int(args.gen_concurrency),
    )
    flags: List[bool] = []
    per_problem_successes: List[int] = []
    n_err = 0
    for (_prompt, gt), comps in zip(
        probe_examples, completions, strict=False
    ):
        problem_successes = 0
        for c in comps:
            if completion_failed(c) or \
                    c.finish_reason not in eval_gpu.SUCCESS_FINISH_REASONS:
                n_err += 1
                continue
            _pred, ok = eval_gpu.grade(c.text, gt)
            flags.append(bool(ok))
            problem_successes += 1
        per_problem_successes.append(problem_successes)
    supplied_problems = len(probe_examples)
    is_aime24 = str(getattr(args, "probe_dataset", "aime24") or "aime24").lower() == "aime24"
    expected_problems = 30 if is_aime24 else supplied_problems
    expected_rows = expected_problems * PROBE_NUM_SAMPLES
    complete_problem_count = sum(
        int(count == PROBE_NUM_SAMPLES) for count in per_problem_successes)
    protocol_complete = (
        supplied_problems == expected_problems
        and len(completions) == expected_problems
        and complete_problem_count == expected_problems
        and len(flags) == expected_rows
        and n_err == 0
    )
    metrics: Dict[str, Any] = {
        "eval/probe_error_rows": float(n_err),
        "eval/probe_graded_rows": float(len(flags)),
        "eval/probe_expected_rows": float(expected_rows),
        "eval/probe_complete_problem_count": float(complete_problem_count),
        "eval/probe_expected_problem_count": float(expected_problems),
        "eval/probe_samples_per_problem": float(PROBE_NUM_SAMPLES),
        "eval/probe_protocol_complete": float(protocol_complete),
        "eval/probe_protocol": PROBE_PROTOCOL_ID,
        "eval/probe_headline_eligible": 0.0,
        "eval/probe_metric_scale": "fraction",
        "eval/probe_sample_seed_rule": PROBE_SAMPLE_SEED_RULE,
        "eval/probe_benchmark_revision": PROBE_BENCHMARK_REVISION,
        "eval/probe_benchmark_rows_sha256": PROBE_BENCHMARK_ROWS_SHA256,
        "eval/probe_prompt_matrix_sha256": common.canonical_json_sha256(
            [prompt for prompt, _ground_truth in probe_examples]),
        "eval/probe_max_completion_tokens": float(PROBE_MAX_TOKENS),
        "eval/probe_temperature": float(PROBE_TEMPERATURE),
        "eval/probe_top_p": float(PROBE_TOP_P),
        "eval/probe_top_k": float(PROBE_TOP_K),
        "eval/probe_min_p": float(PROBE_MIN_P),
        "eval/probe_seed_base": float(PROBE_SEED_BASE),
    }
    if protocol_complete:
        avg_at_4 = probe_average(flags)
        metrics["eval/AIME24_avg_at_4"] = avg_at_4
    else:
        print(
            "[grpo] WARNING: incomplete probe: "
            f"graded={len(flags)}/{expected_rows}, errors={n_err}, "
            f"complete_problems={complete_problem_count}/{expected_problems}; "
            "NOT logging eval/AIME24_avg_at_4",
            flush=True,
        )
    return metrics


# -------------------------------------------------------------------- run ----

def run_grpo(args: argparse.Namespace, pool: Optional[VLLMPool] = None) -> str:
    """Train one GRPO phase; returns the final HF model dir."""
    phase = str(args.phase)
    parity_phase = common.PARITY_PHASES[phase]
    global_step_offset = int(args.global_step_offset)
    lam = 0.0 if phase == "central" else float(args.lambda_novelty)
    lr = float(args.lr) if args.lr is not None else (1e-6 if phase == "central" else 5e-6)
    steps = int(args.steps) if args.steps is not None else (100 if phase == "central" else 200)
    args = apply_completion_budget(args)
    validate_expdis_stabilization(
        args, phase=phase, lambda_novelty=lam)
    code_artifact_binding = load_code_artifact_binding(args)
    if requires_canonical_training_execution_environment(args):
        receipt_binding = load_orchestrator_receipt_binding(
            args, code_artifact_binding=code_artifact_binding)
        args._orchestrator_receipt_binding = receipt_binding
        args._training_execution_environment = \
            validate_canonical_training_execution_environment(
                str(getattr(args, "container_image_identity", "") or ""),
                receipt_binding,
            )
    enforce_probe_serving_len(args, pool)
    validity_mode = rl_validity_mode(args)
    config_tier = resolved_config_tier(args)
    runtime_contract = build_runtime_contract(
        args, resolved_lr=lr, resolved_steps=steps)
    external_lineage = load_external_lineage(
        args, expected_runtime_contract=runtime_contract)
    args._external_lineage = external_lineage
    args._runtime_contract = runtime_contract
    args._code_artifact_binding = code_artifact_binding
    print(f"[grpo] config_tier={config_tier} rl_validity_mode={validity_mode}", flush=True)
    if int(args.updates_per_rollout) > 1 and not args.ppo_mode:
        print("[grpo] WARNING: --updates-per-rollout>1 needs --ppo-mode; "
              "contract REINFORCE runs exactly 1 update per rollout", flush=True)
    os.makedirs(args.output_dir, exist_ok=True)
    validate_code_bound_local_checkpoints(args, code_artifact_binding)
    rollouts_all_path = os.path.join(args.output_dir, "rollouts_all.jsonl")
    traj_path = os.path.join(args.output_dir, "explorer_trajectories.jsonl")

    start_step = 0
    init_path = args.model
    resume_rnd_path = None
    recorded_state: Optional[Dict[str, Any]] = None
    if args.resume_from:
        init_path = args.resume_from
        state_path = os.path.join(args.resume_from, "trainer_state.json")
        if os.path.exists(state_path):
            with open(state_path) as f:
                recorded_state = json.load(f)
            if validity_mode == "c8" and "step" not in recorded_state:
                raise RuntimeError(
                    f"c8 resume requires a recorded learner step in {state_path}")
            # Hard error if the checkpoint was trained under different
            # phase/model/config_tier — never silently mix configurations.
            check_resume_compat(recorded_state, phase=phase,
                                model=str(args.model), config_tier=config_tier,
                                rl_validity_mode_current=validity_mode,
                                resolved_lr_current=lr,
                                external_lineage_current=external_lineage,
                                runtime_contract_current=runtime_contract,
                                code_artifact_binding_current=code_artifact_binding)
            recorded_manifest = recorded_state.get("checkpoint_file_manifest")
            if isinstance(recorded_manifest, dict):
                common.verify_directory_file_manifest(
                    args.resume_from, recorded_manifest,
                    exclude_relative_paths=("trainer_state.json",))
            elif external_lineage is not None:
                raise RuntimeError(
                    "resume checkpoint lacks checkpoint_file_manifest")
            start_step = int(recorded_state.get("step", 0))
        elif validity_mode == "c8":
            raise RuntimeError(
                f"c8 resume requires trainer_state.json, but {state_path} is missing")
        rnd_candidate = os.path.join(args.resume_from, "rnd_state.pt")
        if os.path.exists(rnd_candidate):
            resume_rnd_path = rnd_candidate
        print(f"[grpo] resuming from {args.resume_from} at step {start_step}", flush=True)

    c10_dynamic_journal = os.path.join(
        args.output_dir, "c10_dynamic_sampling_failure_journal.json")
    if str(getattr(args, "c10_execution_profile", "") or "") == \
            "canary" and os.path.isfile(c10_dynamic_journal):
        # This journal is written before its derived terminal/transport files.
        # Complete that exact observed draw before model loading, trimming, or
        # any possible replacement sampling.
        transport_invalid, error_rate = \
            _finalize_c10_dynamic_failure_journal(
                args, runtime_contract=runtime_contract,
                code_artifact=code_artifact_binding,
                trajectory_path=traj_path)
        if transport_invalid:
            raise RuntimeError(
                "C10_CANARY_TRANSPORT_INVALID: recovered dynamic sampling "
                f"exhaustion with generation_error_rate={error_rate}")
        raise RuntimeError(
            "C10_TRAINING_SEMANTIC_FAILURE: recovered dynamic sampling "
            "exhaustion at the fixed 64-attempt boundary")

    # No tokenizer/model/checkpoint bytes are loaded until the resume binding
    # above has been compared with the independently re-hashed code artifact.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True,
        **common.remote_revision_kwargs(args.model, args.model_revision))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    recorded_attention = (
        str(recorded_state.get("policy_attention_implementation", ""))
        if recorded_state is not None else "")
    if recorded_state is not None and validity_mode == "c8" and not recorded_attention:
        raise RuntimeError(
            "c8 resume checkpoint lacks policy_attention_implementation; "
            "refusing a potentially different FA2/SDPA numerical path")
    # Crash recovery: drop rows a previous attempt appended past the resume
    # step (or rotate the old files aside when starting this phase fresh).
    common.trim_jsonl_to_step(rollouts_all_path, start_step)
    common.trim_jsonl_to_step(traj_path, start_step)
    metrics_logger: Optional[common.MetricsLogger] = None
    if args.metrics_jsonl:
        common.trim_metrics_stage(args.metrics_jsonl, phase=parity_phase,
                                  round_idx=int(args.round),
                                  keep_upto_global_step=global_step_offset + start_step)
        metrics_logger = common.MetricsLogger(args.metrics_jsonl)

    model = load_policy(
        init_path, device=args.device, param_dtype=args.param_dtype,
        expected_attention_implementation=recorded_attention or None,
        model_revision=str(args.model_revision or ""))
    args._policy_attention_implementation = str(
        getattr(model, "_tmx_attention_implementation", "") or "")
    optimizer = build_optimizer(
        args.optimizer, model.parameters(), lr=lr,
        betas=(float(args.adam_beta1), float(args.adam_beta2)),
        eps=float(args.adam_eps), weight_decay=float(args.weight_decay),
    )
    if args.resume_from and validity_mode == "c8":
        _restore_optimizer_state(
            optimizer, args.resume_from, device=args.device, required=True)
        print(f"[grpo] optimizer state restored from {args.resume_from}", flush=True)

    rnd: Optional[MultilayerRND] = None
    if lam > 0.0:
        rnd = MultilayerRND(
            input_dim=int(model.config.hidden_size),
            hidden_dim=int(args.rnd_hidden),
            layers=tuple(int(x) for x in args.novelty_layers.split(",")),
            lr=float(args.rnd_lr),
            seed=int(args.seed),
            device=args.device,
        )
        if resume_rnd_path:
            rnd.load(resume_rnd_path)
            print(f"[grpo] RND state restored from {resume_rnd_path}", flush=True)
        if expdis_stabilization_mode(args) == C9_RND_LIFECYCLE and \
                int(rnd.selected_batch_update_count) != int(start_step):
            raise RuntimeError(
                "C9 RND resume invariant failed: selected-batch predictor "
                f"updates={rnd.selected_batch_update_count}, learner_step={start_step}")

    own_pool = False
    if pool is None:
        served = args.served_model_name or args.model
        if args.endpoints:
            pool = VLLMPool.attach(args.endpoints.split(","), served)
        elif args.vllm_gpus:
            pool = VLLMPool(
                model_dir=init_path,
                model_revision=(
                    str(args.model_revision or "")
                    if not common.is_local_model_reference(init_path) else ""),
                gpu_ids=[int(x) for x in args.vllm_gpus.split(",")],
                served_model_name=served,
                port_base=int(args.vllm_port_base),
                max_model_len=int(args.vllm_max_model_len),
                gpu_memory_utilization=float(args.vllm_gpu_mem_util),
                max_num_seqs=int(args.vllm_max_num_seqs),
                log_dir=os.path.join(args.output_dir, "vllm_logs"),
                max_consecutive_failures=int(args.max_consecutive_gen_failures),
            )
            # Register for teardown BEFORE launch/wait_healthy: an early
            # failure must kill the servers, not orphan them holding GPU
            # memory behind the stale-port guard.
            own_pool = True
        else:
            raise ValueError("need a VLLMPool, --endpoints, or --vllm-gpus")
    try:
        if own_pool:
            pool.launch()
            pool.wait_healthy()
        pool.max_consecutive_failures = int(args.max_consecutive_gen_failures)
        # Rollouts must come from the phase's init policy (the resume
        # checkpoint when resuming) — a shared pipeline pool may still serve
        # the previous stage's weights.
        if pool.managed and pool.model_dir != init_path:
            print(f"[grpo] reloading pool to the {phase} init policy: {init_path}", flush=True)
            pool.reload(init_path)
        validate_rl_validity(args, pool)
    except Exception:
        if own_pool:
            pool.shutdown()
        raise

    # A managed pool was just verified/reloaded to the learner's init or
    # resume checkpoint. Attach-mode freshness is unknowable and is reported
    # explicitly as such on legacy probes.
    served_policy_step: Optional[int] = start_step if pool.managed else None

    probe_examples: Optional[List[Tuple[str, str]]] = None  # lazy (first probe)

    wandb_run = None
    try:
        examples = load_training_examples(args.dataset, tokenizer,
                                          int(args.max_train_examples))
        if args.wandb:
            # Own stage run (reinit="create_new"): never hijacks/finishes the
            # pipeline run active in this process. See common.init_stage_wandb.
            wandb_run = common.init_stage_wandb(
                name=f"{args.run_name}-{phase}-r{int(args.round)}",
                group=os.environ.get("WANDB_RUN_GROUP", args.run_name),
                config={**{k: v for k, v in vars(args).items()},
                        "config_tier": config_tier,
                        "resolved_lr": lr, "resolved_lambda": lam, "resolved_steps": steps,
                        "resolved_max_completion_len": int(args.max_completion_len),
                        "resolved_soft_overlong_expected_len": int(args.soft_overlong_expected_len),
                        "resolved_soft_overlong_cache_len": int(args.soft_overlong_cache_len),
                        "resolved_max_total_len": int(args.max_prompt_len) + int(args.max_completion_len),
                        "backend": "gpu-torch"},
            )
    except Exception:
        # Setup failure after the pool is up must never orphan our servers.
        if own_pool:
            pool.shutdown()
        raise

    G = int(args.num_generations)
    t_run = time.perf_counter()
    final_dir = os.path.join(args.output_dir, "final")
    c10_canary = str(getattr(args, "c10_execution_profile", "") or "") == \
        "canary"
    c9_canary_profile = str(
        getattr(args, "c9_execution_profile", "") or "") == "canary"
    canary_health_key = (
        "c10_canary_health_state" if c10_canary
        else "c9_canary_health_state")
    recorded_canary_health = (
        recorded_state.get(canary_health_key)
        if recorded_state is not None else None)
    if (c9_canary_profile or c10_canary) and start_step > 0:
        if not isinstance(recorded_canary_health, dict):
            raise RuntimeError(
                "stabilization canary resume checkpoint lacks rolling-health state")
        recent_health = recorded_canary_health.get("recent_clip_valid")
        if recorded_canary_health.get("completed_step") != start_step or \
                not isinstance(recent_health, list) or len(recent_health) != \
                min(8, start_step) or any(
                    not isinstance(item, list) or len(item) != 2
                    or not all(math.isfinite(float(value)) for value in item)
                    for item in recent_health) or int(recorded_canary_health.get(
                        "consecutive_bad_windows", -1)) < 0:
            raise RuntimeError(
                "stabilization canary resume checkpoint has invalid/stale "
                "rolling-health state")
    c9_canary_health: List[Tuple[float, float]] = [
        (float(item[0]), float(item[1]))
        for item in ((recorded_canary_health or {}).get(
            "recent_clip_valid", []))
    ]
    c9_consecutive_bad_windows = int(
        (recorded_canary_health or {}).get("consecutive_bad_windows", 0))

    def emit(record: Dict[str, Any], global_step: int) -> None:
        """One metric record to W&B and the shared metrics JSONL."""
        c9_record = ({
            "expdis_stabilization_mode": expdis_stabilization_mode(args),
            "scientific_variant": str(args.scientific_variant),
            "c9_execution_profile": str(args.c9_execution_profile),
            "stabilization_cell": str(args.stabilization_cell),
            "novelty_normalization": novelty_normalization_mode(args),
        } if (
            expdis_stabilization_mode(args) == C9_RND_LIFECYCLE
            or str(getattr(args, "c9_execution_profile", "") or "")
        ) else {})
        c10_record = ({
            "c10_execution_profile": str(args.c10_execution_profile),
            "survivor_overlong_filter_version":
                C10_OVERLONG_FILTER_CONTRACT_VERSION,
        } if str(getattr(args, "c10_execution_profile", "") or "") else {})
        mrme_record = ({
            "mrme_contract_sha256": str(args.mrme_contract_sha256),
            "mrme_method": str(args.mrme_method),
            "mrme_stabilization_profile": str(
                args.mrme_stabilization_profile),
            "mrme_round_index": int(args.round),
            "mrme_scout_index": int(args.mrme_scout_index),
            "mrme_scouts_per_round": int(args.mrme_scouts_per_round),
            "mrme_shard_manifest_sha256": str(
                args.mrme_shard_manifest_sha256),
            "mrme_stage_seed": int(args.seed),
        } if str(getattr(args, "mrme_contract_sha256", "") or "") else {})
        record = {"global_step": int(global_step), "phase": parity_phase,
                  "round": int(args.round), "lambda": lam, "lambda_novelty": lam,
                  "num_scouts": int(args.num_scouts),
                  "rl_validity_mode": validity_mode, **c9_record, **c10_record,
                  **mrme_record, **record}
        if wandb_run is not None:
            wandb_run.log(record, step=int(global_step))
        if metrics_logger is not None:
            metrics_logger.log(record)

    try:
        for step in range(start_step + 1, steps + 1):
            global_step = global_step_offset + step
            # Resurrect any managed server that died since the last step
            # (a dead server would otherwise only surface as request errors).
            if pool.health_check():
                print(f"[grpo] step {step}: restarted dead vLLM server(s)", flush=True)
            learner_policy_step = step - 1
            rollout_staleness = (
                learner_policy_step - served_policy_step
                if served_policy_step is not None else None)
            if validity_mode == "c8" and rollout_staleness != 0:
                raise RuntimeError(
                    "c8 on-policy invariant failed before rollout: "
                    f"learner_step={learner_policy_step}, "
                    f"served_policy_step={served_policy_step}")
            rng = np.random.default_rng(_phase_step_seed(args) + step)
            try:
                selected_rows, all_rows, dyn_stats, rollout_dt = \
                    _dynamic_sample_step(
                        rng=rng, pool=pool, tokenizer=tokenizer, model=model,
                        rnd=rnd, examples=examples, args=args, lam=lam,
                        step=step)
                validate_c10_selected_batch_before_update(
                    args, selected_rows, dyn_stats)
            except C10TrainingHealthFailure as exc:
                if exc.failure_kind not in {
                        "selected_group_geometry_failure",
                        "survivor_filter_contract_failure"}:
                    # These internal selected-batch/RND exceptions occur
                    # before a portable raw batch/feature proof exists.  They
                    # must consume the one-shot attempt fail-closed, never be
                    # mislabeled as a terminal scientific outcome.
                    raise RuntimeError(
                        "C10_INTERNAL_UNVERIFIABLE_PREUPDATE_FAILURE: "
                        + str(exc)) from exc
                failed_selected_batch_path = ""
                failed_selected_batch_manifest_path = ""
                failed_selected_batch_path, \
                    failed_selected_batch_manifest_path = \
                    _write_c10_failed_selected_batch(
                        args, selected_rows, step=step,
                        dynamic_stats=dyn_stats)
                _record_c10_training_failure(
                    args, runtime_contract=runtime_contract,
                    code_artifact=code_artifact_binding,
                    failure_kind=exc.failure_kind,
                    completed_step=step - 1, failed_step=step,
                    failure_details={"health_gate": {
                        "failure_kind": exc.failure_kind,
                        "failure_stage":
                            "selected_batch_finalization_pre_policy_update",
                        "observed": exc.details,
                    }},
                    trajectory_path=traj_path,
                    failed_selected_batch_path=
                        failed_selected_batch_path,
                    failed_selected_batch_manifest_path=
                        failed_selected_batch_manifest_path)
                raise RuntimeError(
                    "C10_TRAINING_SEMANTIC_FAILURE: " + str(exc)) from exc
            except C10DynamicSamplingExhaustion as exc:
                journal = _signed_c10_dynamic_failure_journal(
                    args, completed_step=step - 1, failed_step=step,
                    runtime_contract=runtime_contract,
                    code_artifact=code_artifact_binding,
                    details=exc.details, candidate_rows=exc.candidate_rows)
                journal_path = os.path.join(
                    args.output_dir,
                    "c10_dynamic_sampling_failure_journal.json")
                if os.path.isfile(journal_path):
                    with open(journal_path, encoding="utf-8") as handle:
                        existing_journal = json.load(handle)
                    if existing_journal != journal:
                        raise RuntimeError(
                            "C10 dynamic failure journal already differs") \
                            from exc
                else:
                    common.atomic_write_json(journal_path, journal)
                transport_invalid, error_rate = \
                    _finalize_c10_dynamic_failure_journal(
                        args, runtime_contract=runtime_contract,
                        code_artifact=code_artifact_binding,
                        trajectory_path=traj_path)
                if transport_invalid:
                    raise RuntimeError(
                        "C10_CANARY_TRANSPORT_INVALID: dynamic sampling "
                        f"exhausted with generation_error_rate={error_rate}") \
                        from exc
                raise RuntimeError(
                    "C10_TRAINING_SEMANTIC_FAILURE: dynamic sampling "
                    "exhausted the fixed 64 attempts") from exc
            for row in all_rows:
                row["step"] = int(step)
            common.append_jsonl(rollouts_all_path, all_rows)
            trajectory_provenance = _fresh_c8_component_artifact_metadata(args)
            mrme_trajectory_provenance = _mrme_trajectory_provenance(args)
            common.append_jsonl(
                traj_path,
                [trajectory_record(
                    r, step, **trajectory_provenance,
                    mrme_trajectory_provenance=mrme_trajectory_provenance)
                 for r in selected_rows],
            )

            rewards = np.asarray([float(r["blended_reward"]) for r in selected_rows],
                                 dtype=np.float32)
            advantages, overlong_filter_stats = group_advantages_for_rows(
                selected_rows, G, args)

            t0 = time.perf_counter()
            update = _train_update(model, optimizer, tokenizer, selected_rows, advantages, args)
            update_dt = time.perf_counter() - t0

            correctness = np.asarray([float(r["correctness_reward"]) for r in selected_rows])
            novelty_used = np.asarray([float(r["novelty_reward"]) for r in selected_rows])
            clipped = np.asarray([1.0 if r["clipped"] else 0.0 for r in selected_rows])
            terminated = np.asarray([1.0 if r["terminated"] else 0.0 for r in selected_rows])
            valid = np.asarray([1.0 if r["valid_answer"] else 0.0 for r in selected_rows])
            lengths = np.asarray([float(r["completion_token_length"]) for r in selected_rows])
            wall = time.perf_counter() - t_run
            metrics = {
                # -- paper-curve parity keys (wandb_style_curves CSVs) --------
                # Contract tier: ratio==1 REINFORCE => clip inert (0.0 by
                # construction) and policy_kl exactly 0. --ppo-mode: measured.
                "train/reward_total_mean": float(rewards.mean()),
                "train/reward_correctness_mean": float(correctness.mean()),
                "train/reward_novelty_mean": float(novelty_used.mean()),
                "train/correctness_rate": float(np.mean(
                    [1.0 if r["is_correct"] else 0.0 for r in selected_rows])),
                "train/grpo_loss": float(update["loss"]),
                "train/clip_fraction": float(update["clip_fraction"]),
                "train/policy_kl": float(update["policy_kl"]),
                "quality/valid_answer_rate": float(valid.mean()),
                "quality/clipped_rate": float(clipped.mean()),
                "quality/nontermination_rate": float(1.0 - terminated.mean()),
                "quality/mean_completion_tokens": float(lengths.mean()),
                # -- diagnostics ----------------------------------------------
                "grpo/step": step,
                "grpo/reward_std": float(rewards.std()),
                "grpo/reward/overlong_penalty_mean": float(np.mean(
                    [float(r["overlong_penalty"]) for r in selected_rows])),
                "grpo/reward/novelty_raw_mean": float(np.mean(
                    [float(r["novelty_raw"]) for r in selected_rows])),
                "train/grad_norm": float(update["grad_norm"]),
                "train/wall_time_seconds": wall,
                "train/rollout_time_s": rollout_dt,
                "train/update_time_s": update_dt,
                "train/rows_backward": update["rows_backward"],
                "train/masked_truncated_rows": update["masked_truncated_rows"],
                "train/raw_selected_rows": update["raw_selected_rows"],
                "train/survivor_rows": update["survivor_rows"],
                "train/zero_advantage_survivor_rows": update[
                    "zero_advantage_survivor_rows"],
                "train/fixed_loss_denominator": update[
                    "fixed_loss_denominator"],
                "train/updates_per_rollout": update["updates"],
                "train/selected_token_logprob_mean": update["selected_token_logprob_mean"],
                "train/learning_rate": lr,
                "train/rollout_policy_step": (
                    int(served_policy_step) if served_policy_step is not None else -1),
                "train/rollout_staleness_updates": (
                    int(rollout_staleness) if rollout_staleness is not None else -1),
                "train/rollout_staleness_known": float(rollout_staleness is not None),
                "train/on_policy_rollout": float(rollout_staleness == 0),
                **{f"grpo/dynamic/{k}": v for k, v in dyn_stats.items()},
                **{
                    f"train/overlong_filter/{key}": value
                    for key, value in overlong_filter_stats.items()
                },
            }
            c9_canary = (
                (str(getattr(args, "c9_execution_profile", "") or "")
                 == "canary" or
                 str(getattr(args, "c10_execution_profile", "") or "")
                 == "canary")
                and phase == "scout"
            )
            c9_transport_invalid = False
            if c9_canary:
                generation_error_rate = float(
                    dyn_stats.get("generation_error_rate", 1.0))
                c9_transport_invalid = generation_error_rate > 0.05
                metrics["grpo/c9_health/transport_invalid"] = float(
                    c9_transport_invalid)
                c9_canary_health.append((
                    float(metrics["quality/clipped_rate"]),
                    float(metrics["quality/valid_answer_rate"]),
                ))
                bad_window = False
                if step >= 10 and len(c9_canary_health) >= 5:
                    recent = c9_canary_health[-5:]
                    rolling_clip = sum(x[0] for x in recent) / 5.0
                    rolling_valid = sum(x[1] for x in recent) / 5.0
                    bad_window = rolling_clip >= 0.50 and rolling_valid <= 0.65
                    c9_consecutive_bad_windows = (
                        c9_consecutive_bad_windows + 1 if bad_window else 0)
                    metrics.update({
                        "grpo/c9_health/rolling_clipped_rate_5": rolling_clip,
                        "grpo/c9_health/rolling_valid_answer_rate_5": rolling_valid,
                        "grpo/c9_health/bad_window": float(bad_window),
                        "grpo/c9_health/consecutive_bad_windows": float(
                            c9_consecutive_bad_windows),
                    })
                metrics["grpo/c9_health/window_available"] = float(step >= 10)
                args._c9_canary_health_state = {
                    # Seven raw points are enough to reconstruct the last three
                    # overlapping five-step windows; keep eight for clarity.
                    "recent_clip_valid": [list(item) for item in c9_canary_health[-8:]],
                    "consecutive_bad_windows": int(c9_consecutive_bad_windows),
                    "completed_step": int(step),
                }
                if c10_canary:
                    args._c10_canary_health_state = dict(
                        args._c9_canary_health_state)
            # C9 lifecycle diagnostics exist only for the novelty-bearing
            # Scout.  The downstream Central deliberately keeps the C9
            # lineage label while running ordinary correctness-only GRPO, so
            # it has no RND statistics to read from dyn_stats.
            if (
                phase == "scout"
                and lam > 0.0
                and rnd is not None
                and expdis_stabilization_mode(args) == C9_RND_LIFECYCLE
            ):
                metrics.update({
                    "grpo/reward/novelty_normalization_center": float(
                        dyn_stats["novelty_normalization_center"]),
                    "grpo/reward/novelty_normalization_scale": float(
                        dyn_stats["novelty_normalization_scale"]),
                    "grpo/reward/novelty_normalization_effective_scale": float(
                        dyn_stats["novelty_normalization_effective_scale"]),
                    "grpo/reward/novelty_normalized_selected_mean": float(
                        dyn_stats["novelty_normalized_selected_mean"]),
                    "grpo/reward/novelty_normalized_selected_population_std": float(
                        dyn_stats[
                            "novelty_normalized_selected_population_std"]),
                    "grpo/rnd/updates_this_step": float(
                        dyn_stats["rnd_updates_this_step"]),
                    "grpo/rnd/update_rows": float(dyn_stats["rnd_update_rows"]),
                    "grpo/rnd/candidate_rows_scored": float(
                        dyn_stats["rnd_candidate_rows_scored"]),
                    "grpo/rnd/selected_batch_update_count": float(
                        dyn_stats["rnd_selected_batch_update_count"]),
                })
            c9_profile_active = bool(
                str(getattr(args, "c9_execution_profile", "") or "") or
                str(getattr(args, "c10_execution_profile", "") or ""))
            c10_health_failure = None
            c10_transport_failure = None
            if c9_profile_active and phase == "scout":
                # The preregistered health contract is evaluated on every
                # completed step, not merely at the endpoint.  Fail before
                # checkpointing when geometry, transport, or any learner/RND
                # scalar is invalid.
                geometry_invalid = (
                    (not c10_exact_numeric(
                        dyn_stats.get("kept_group_count"),
                        int(args.prompts_per_step))
                     or not c10_exact_numeric(
                        dyn_stats.get("fallback_group_count"), 0))
                    if c10_canary else
                    (int(dyn_stats.get("kept_group_count", -1)) !=
                     int(args.prompts_per_step)
                     or int(dyn_stats.get("fallback_group_count", -1)) != 0)
                )
                if geometry_invalid:
                    message = (
                        "C9 step lacks exactly four selected groups / zero "
                        "fallback")
                    if c10_canary:
                        c10_health_failure = (
                            "selected_group_geometry_failure", message, {
                                "kept_group_count": dyn_stats.get(
                                    "kept_group_count"),
                                "fallback_group_count": dyn_stats.get(
                                    "fallback_group_count"),
                            })
                    else:
                        raise RuntimeError(message)
                if c10_health_failure is None and \
                        survivor_consistent_overlong_filter(args):
                    if int(dyn_stats.get("selected_raw_row_count", -1)) != \
                            int(args.prompts_per_step) * int(
                                args.num_generations):
                        c10_health_failure = (
                            "survivor_filter_contract_failure",
                            "C10 selected raw-row geometry mismatch", {
                                "selected_raw_row_count": dyn_stats.get(
                                    "selected_raw_row_count"),
                            })
                    else:
                        for group_start in range(
                                0, len(selected_rows),
                                int(args.num_generations)):
                            group = selected_rows[
                                group_start:group_start
                                + int(args.num_generations)]
                            survivors = sum(
                                not bool(row.get("clipped", False))
                                for row in group)
                            populations = sorted(set(str(row.get(
                                "dynamic_sampling_eligibility_population"))
                                for row in group))
                            if survivors < 2 or populations != [
                                    "unmasked_survivors"]:
                                c10_health_failure = (
                                    "survivor_filter_contract_failure",
                                    "C10 selected a group outside the "
                                    "survivor-consistent eligibility contract",
                                    {"group_start": group_start,
                                     "survivors": survivors,
                                     "populations": populations})
                                break
                generation_error_rate = float(
                    dyn_stats.get("generation_error_rate", math.inf))
                if not math.isfinite(generation_error_rate) or \
                        generation_error_rate > 0.05:
                    message = (
                        "C9_TRANSPORT_INVALID: generation error rate "
                        f"{generation_error_rate!r} exceeds 0.05")
                    if c10_canary:
                        c10_transport_failure = (
                            "completed_step_transport_invalid", message, {
                                "generation_requested_row_count": dyn_stats.get(
                                    "generation_requested_row_count"),
                                "generation_error_row_count": dyn_stats.get(
                                    "generation_error_row_count"),
                                "observed_generation_error_rate":
                                    generation_error_rate,
                            })
                    else:
                        raise RuntimeError(message)
                finite_keys = (
                    "train/grpo_loss", "train/reward_total_mean",
                    "train/reward_correctness_mean", "train/reward_novelty_mean",
                    "train/grad_norm", "train/selected_token_logprob_mean",
                    "grpo/dynamic/rnd_selected_batch_loss",
                    "grpo/reward/novelty_normalization_center",
                    "grpo/reward/novelty_normalization_scale",
                    "grpo/reward/novelty_normalization_effective_scale",
                    "grpo/reward/novelty_normalized_selected_mean",
                    "grpo/reward/novelty_normalized_selected_population_std",
                )
                bad_finite = [
                    key for key in finite_keys
                    if not math.isfinite(float(metrics.get(key, math.nan)))
                ]
                if c10_transport_failure is None and \
                        c10_health_failure is None and bad_finite:
                    message = (
                        f"C9 step has non-finite learner values: {bad_finite}")
                    if c10_canary:
                        c10_health_failure = (
                            "learner_health_nonfinite", message,
                            {"nonfinite_fields": bad_finite})
                    else:
                        raise RuntimeError(message)
                if c10_transport_failure is None and \
                        c10_health_failure is None and \
                        expdis_stabilization_mode(args) == C9_RND_LIFECYCLE:
                    rnd_contract_invalid = (
                        (not c10_exact_numeric(
                            dyn_stats.get("rnd_updates_this_step"), 1)
                         or not c10_exact_numeric(
                            dyn_stats.get("rnd_update_rows"),
                            int(args.prompts_per_step) * int(
                                args.num_generations))
                         or not c10_exact_numeric(dyn_stats.get(
                            "rnd_selected_batch_update_count"), step))
                        if c10_canary else
                        (int(dyn_stats.get("rnd_updates_this_step", -1)) != 1
                         or int(dyn_stats.get("rnd_update_rows", -1)) !=
                         int(args.prompts_per_step) * int(args.num_generations)
                         or int(dyn_stats.get(
                            "rnd_selected_batch_update_count", -1)) != step)
                    )
                    if rnd_contract_invalid:
                        message = (
                            "C9 selected-batch RND update count/geometry "
                            "mismatch")
                        if c10_canary:
                            c10_health_failure = (
                                "rnd_update_contract_failure", message, {
                                    "rnd_updates_this_step": dyn_stats.get(
                                        "rnd_updates_this_step"),
                                    "rnd_update_rows": dyn_stats.get(
                                        "rnd_update_rows"),
                                    "rnd_selected_batch_update_count":
                                        dyn_stats.get(
                                            "rnd_selected_batch_update_count"),
                                })
                        else:
                            raise RuntimeError(message)
                    incorrect_credit_violations = [
                        {"row_index": row_index,
                         "novelty_reward": row.get("novelty_reward")}
                        for row_index, row in enumerate(selected_rows)
                        if not bool(row.get("is_correct", False))
                        and not math.isclose(float(row.get(
                            "novelty_reward", math.nan)), 0.0, abs_tol=1e-7)
                    ]
                    if c10_health_failure is None and \
                            incorrect_credit_violations:
                        message = (
                            "C9 incorrect learner row has nonzero novelty "
                            "credit")
                        if c10_canary:
                            c10_health_failure = (
                                "incorrect_novelty_credit_failure", message,
                                {"violations": incorrect_credit_violations})
                        else:
                            raise RuntimeError(message)
                    if c10_health_failure is None and \
                            novelty_normalization_mode(args) == \
                            "selected_batch_zscore":
                        mean = float(dyn_stats.get(
                            "novelty_normalized_selected_mean", math.nan))
                        std = float(dyn_stats.get(
                            "novelty_normalized_selected_population_std", math.nan))
                        if not math.isclose(mean, 0.0, abs_tol=1e-5) or \
                                not math.isclose(std, 1.0, abs_tol=1e-5):
                            message = (
                                "C9 selected-batch novelty is not population "
                                f"z-scored (mean={mean}, std={std})")
                            if c10_canary:
                                c10_health_failure = (
                                    "novelty_normalization_failure", message,
                                    {
                                        "mean": mean,
                                        "population_std": std,
                                        "normalized_pre_gate_sha256":
                                            common.canonical_json_sha256([
                                                float(row.get(
                                                    "novelty_normalized_pre_gate",
                                                    math.nan))
                                                for row in selected_rows]),
                                    })
                            else:
                                raise RuntimeError(message)
            print(f"[grpo] {phase} step {step}/{steps} (global {global_step}) "
                  f"loss={metrics['train/grpo_loss']:.5f} "
                  f"reward={metrics['train/reward_total_mean']:.3f} "
                  f"correct={metrics['train/correctness_rate']:.3f} "
                  f"len={metrics['quality/mean_completion_tokens']:.0f} "
                  f"grad={metrics['train/grad_norm']:.3f} "
                  f"rollout={rollout_dt:.0f}s update={update_dt:.0f}s", flush=True)
            emit(metrics, global_step)

            if c10_transport_failure is not None:
                failure_kind, message, observed = c10_transport_failure
                _record_c10_training_failure(
                    args, runtime_contract=runtime_contract,
                    code_artifact=code_artifact_binding,
                    failure_kind=failure_kind,
                    completed_step=step, failed_step=step,
                    failure_details=observed,
                    trajectory_path=traj_path, transport_invalid=True)
                raise RuntimeError(message)
            if c10_health_failure is not None:
                failure_kind, message, observed = c10_health_failure
                _record_c10_training_failure(
                    args, runtime_contract=runtime_contract,
                    code_artifact=code_artifact_binding,
                    failure_kind=failure_kind,
                    completed_step=step, failed_step=step,
                    failure_details={"health_gate": {
                        "failure_kind": failure_kind,
                        "observed": observed,
                    }},
                    trajectory_path=traj_path)
                raise RuntimeError(
                    "C10_TRAINING_SEMANTIC_FAILURE: " + message)

            if c9_canary and c9_transport_invalid:
                raise RuntimeError(
                    "C9_CANARY_TRANSPORT_INVALID: generation error rate "
                    f"{generation_error_rate:.6f} exceeds 0.05; rerun the "
                    "unchanged cell")
            if c9_canary and c9_consecutive_bad_windows >= 3:
                if c10_canary:
                    _record_c10_training_failure(
                        args, runtime_contract=runtime_contract,
                        code_artifact=code_artifact_binding,
                        failure_kind="rolling_behavior_collapse",
                        completed_step=step, failed_step=step,
                        failure_details={
                            "rolling_health": dict(
                                args._c10_canary_health_state),
                        },
                        trajectory_path=traj_path)
                raise RuntimeError(
                    "C9_CANARY_BEHAVIOR_FAILURE: three consecutive five-step "
                    "windows have clipped_rate>=0.50 and valid_answer_rate<=0.65")

            if sync_after_update_due(args, step, steps):
                _synced, reload_dt = _sync_vllm(model, tokenizer, pool, args, step)
                if _synced is not None:
                    served_policy_step = step
                    emit({"train/reload_time_s": float(reload_dt),
                          "train/reload_step": int(step),
                          "train/reload_for_on_policy": float(validity_mode == "c8")},
                         global_step)
                elif validity_mode == "c8":
                    raise RuntimeError(
                        "c8 requires a successful vLLM sync after every learner update")
            checkpoint_due = (
                int(args.save_every) > 0
                and step % int(args.save_every) == 0)
            probe_due = (
                int(args.probe_every) > 0
                and step % int(args.probe_every) == 0)
            strict_c9_probe = bool(
                str(getattr(args, "c9_execution_profile", "") or "") or
                str(getattr(args, "c10_execution_profile", "") or ""))
            # A C9 checkpoint at a probe boundary is promotable only after
            # that boundary's exact diagnostic matrix has completed.  Saving
            # first would let a retry resume at step+1 and silently skip the
            # failed probe forever.
            defer_checkpoint_until_probe = strict_c9_probe and probe_due
            if checkpoint_due and not defer_checkpoint_until_probe:
                _save_checkpoint(
                    model, tokenizer, rnd, optimizer, args, step, lam, wandb_run)
            if probe_due:
                probe_staleness = (
                    step - served_policy_step
                    if served_policy_step is not None else None)
                if validity_mode == "c8" and probe_staleness != 0:
                    raise RuntimeError(
                        "c8 probe freshness invariant failed: "
                        f"learner_step={step}, served_policy_step={served_policy_step}")
                try:
                    if probe_examples is None:
                        probe_examples = _load_probe_examples(args, tokenizer)
                    t0 = time.perf_counter()
                    probe_metrics = _run_probe(pool, probe_examples, args)
                    probe_metrics["eval/probe_time_s"] = time.perf_counter() - t0
                    probe_metrics.update({
                        "eval/probe_policy_step": (
                            int(served_policy_step) if served_policy_step is not None else -1),
                        "eval/probe_staleness_updates": (
                            int(probe_staleness) if probe_staleness is not None else -1),
                        "eval/probe_staleness_known": float(probe_staleness is not None),
                        "eval/probe_fresh_weights": float(probe_staleness == 0),
                    })
                    if "eval/AIME24_avg_at_4" in probe_metrics:
                        print(f"[grpo] probe step {step}: "
                              "AIME24_avg_at_4="
                              f"{probe_metrics['eval/AIME24_avg_at_4']:.2f}",
                              flush=True)
                    else:
                        print(
                            f"[grpo] probe step {step}: incomplete protocol "
                            f"({probe_metrics['eval/probe_graded_rows']:.0f}/"
                            f"{probe_metrics['eval/probe_expected_rows']:.0f} graded); "
                            "AIME24 avg@4 not logged",
                            flush=True,
                        )
                    emit(probe_metrics, global_step)
                    if strict_c9_probe and float(probe_metrics.get(
                            "eval/probe_protocol_complete", 0.0)) != 1.0:
                        raise RuntimeError(
                            "C9 probe did not complete the exact 30x4 AIME24 "
                            "matrix; checkpoint promotion is forbidden")
                except Exception as exc:  # noqa: BLE001 — C9 re-raises; legacy logs
                    if strict_c9_probe:
                        raise RuntimeError(
                            f"C9 required probe failed at step {step}") from exc
                    print(f"[grpo] WARNING: probe failed at step {step}: {exc}", flush=True)
            if checkpoint_due and defer_checkpoint_until_probe:
                _save_checkpoint(
                    model, tokenizer, rnd, optimizer, args, step, lam, wandb_run)

        save_model_bf16(model, tokenizer, final_dir)
        if rnd is not None:
            rnd.save(os.path.join(final_dir, "rnd_state.pt"))
        # The phase-final directory is a deployable/model-handoff artifact,
        # not a crash-resume checkpoint. Keep optimizer moments only in
        # step_* checkpoints: including AdamW state here roughly quintuples
        # every final-model upload and makes evaluation download state it
        # cannot use. A crash after the last update resumes from step_*.
        optimizer_state_saved = False
        final_file_manifest = common.directory_file_manifest(
            final_dir, exclude_relative_paths=("trainer_state.json",))
        c9_rnd_update_count = (
            int(rnd.selected_batch_update_count)
            if rnd is not None
            and expdis_stabilization_mode(args) == C9_RND_LIFECYCLE
            and phase == "scout"
            else None
        )
        c9_canary_health_state = getattr(args, "_c9_canary_health_state", None)
        c10_canary_health_state = getattr(
            args, "_c10_canary_health_state", None)
        common.atomic_write_json(
            os.path.join(final_dir, "trainer_state.json"),
            {"step": steps, "phase": phase, "lambda": lam,
             "round": int(args.round), "model": args.model,
             "model_revision": str(args.model_revision or ""),
             "config_tier": config_tier,
             "rl_validity_mode": validity_mode,
             "optimizer_state_saved": optimizer_state_saved,
             "policy_attention_implementation": str(
                 getattr(args, "_policy_attention_implementation", "") or ""),
             "checkpoint_kind": "deployable_model",
             "resolved_lr": float(optimizer.param_groups[0]["lr"]),
             "external_lineage": external_lineage,
             "external_lineage_sha256": (
                 (external_lineage.get("lineage_sha256")
                  or external_lineage.get("evidence_sha256"))
                 if external_lineage else None),
             "runtime_contract": runtime_contract,
             "code_artifact_binding": code_artifact_binding,
             **({"mrme_provenance": _mrme_artifact_metadata(args)}
                if getattr(args, "mrme_contract_sha256", "") else {}),
             "rnd_selected_batch_update_count": c9_rnd_update_count,
             "c9_canary_health_state": c9_canary_health_state,
             "c10_canary_health_state": c10_canary_health_state,
             "checkpoint_file_manifest": final_file_manifest})
        print(f"[grpo] {phase} final model -> {final_dir}", flush=True)
        if wandb_run is not None and int(args.ckpt_artifact_every) > 0:
            final_artifact_metadata = {
                "step": steps, "round": int(args.round), "phase": phase,
                "model_revision": str(args.model_revision or ""),
                "runtime_contract_sha256": (
                    common.canonical_json_sha256(runtime_contract)
                    if isinstance(runtime_contract, dict) else None),
                "training_dataset_identity_sha256": (
                    runtime_contract.get(
                        "training_dataset_identity", {}).get(
                            "identity_sha256")
                    if isinstance(runtime_contract, dict) else None),
                **_c9_artifact_metadata(args),
                **_fresh_c8_component_artifact_metadata(args),
                **_mrme_artifact_metadata(args),
            }
            logged_final = common.log_dir_artifact(
                wandb_run,
                name=f"{args.run_name}-{phase}-r{int(args.round)}-final",
                path=final_dir, type_="model",
                metadata=final_artifact_metadata,
            )
            logged_final.wait()
            artifact_ref = str(
                getattr(logged_final, "qualified_name", "")
                or getattr(logged_final, "name", "") or "")
            common.require_immutable_wandb_artifact_ref(
                artifact_ref, field="GRPO final model artifact")
            artifact_digest = str(
                getattr(logged_final, "digest", "") or "")
            if not artifact_digest:
                raise RuntimeError(
                    "GRPO final model artifact lacks a remote digest")
            final_artifact_binding = {
                "version": FINAL_MODEL_ARTIFACT_BINDING_VERSION,
                "source": {
                    "artifact_ref": artifact_ref,
                    "artifact_digest": artifact_digest,
                    "artifact_type": "model",
                },
                "artifact_subdir": ".",
                "model_file_manifest": common.directory_file_manifest(
                    final_dir),
                "policy_file_manifest": final_file_manifest,
                "artifact_metadata": final_artifact_metadata,
                "phase": phase,
                "round": int(args.round),
                "step": int(steps),
                "runtime_contract_sha256": (
                    common.canonical_json_sha256(runtime_contract)
                    if isinstance(runtime_contract, dict) else None),
                "mrme_provenance": (
                    _mrme_artifact_metadata(args)
                    if getattr(args, "mrme_contract_sha256", "") else None),
            }
            final_artifact_binding["binding_sha256"] = \
                common.canonical_json_sha256(final_artifact_binding)
            common.atomic_write_json(
                os.path.join(args.output_dir, "final_artifact_binding.json"),
                final_artifact_binding)
            print(f"[grpo] {phase} final model durably pushed to W&B", flush=True)
        # Leave a caller-provided pool serving the FINAL policy (pipeline
        # harvest passes rely on it); pointless for a pool we own and tear down.
        if pool is not None and pool.managed and not own_pool:
            pool.reload(final_dir)
    finally:
        if own_pool and pool is not None:
            pool.shutdown()
        if wandb_run is not None:
            wandb_run.finish()
    return final_dir


# -------------------------------------------------------------------- CLI ----

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="Qwen/Qwen3-1.7B", help="init policy (HF id or local dir)")
    p.add_argument(
        "--model-revision", default=common.CANONICAL_TRAINING_MODEL_REVISION,
        help="immutable Hub revision for the base model; applied only when "
             "the model reference is remote, never to local stage handoffs",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--run-name", default="grpo-gpu")
    p.add_argument(
        "--orchestrator-run-name", default="",
        help="top-level remote RUN_NAME when a pipeline gives this stage a "
             "different W&B run name",
    )
    p.add_argument("--phase", default="scout", choices=["scout", "central"])
    p.add_argument(
        "--scientific-variant", default="",
        help="noncanonical scientific-variant label propagated by a parent pipeline",
    )
    p.add_argument(
        "--external-lineage-json", default="",
        help="fail-closed external SFT handoff lineage (Central only); copied "
             "into every step/final trainer_state and required on resume",
    )
    p.add_argument(
        "--code-source-manifest", default="",
        help="exact downloaded code-artifact manifest; fresh-c8 Scout or the "
             "lineage-free original-paper DAPO Central-route control only. "
             "The immutable identity is embedded in every checkpoint and "
             "required before resume loads any checkpoint bytes",
    )
    p.add_argument(
        "--container-image-identity", default="",
        help="immutable OCI image tag@sha256 identity. Required exactly for "
             "strict C9/MR-ME and original-paper DAPO training; the live "
             "runtime stack is independently checked before any update",
    )
    p.add_argument(
        "--orchestrator-receipt-binding", default="",
        help="resolved immutable W&B/Lepton job-spec binding. Mandatory for "
             "strict C9/MR-ME and original-paper DAPO training",
    )
    p.add_argument(
        "--orchestrator-command-sha256", default="",
        help="controller-preregistered SHA256 of the exact Lepton job command",
    )
    p.add_argument(
        "--orchestrator-receipt-artifact", default="",
        help="RUN_NAME-unique entity/project/name:latest polling alias captured "
             "by the authoritative Lepton job spec",
    )
    p.add_argument("--steps", type=int, default=None, help="default: 200 scout / 100 central")
    p.add_argument("--lr", type=float, default=None, help="default: 5e-6 scout / 1e-6 central")
    p.add_argument("--lambda-novelty", dest="lambda_novelty", type=float, default=0.0)
    p.add_argument(
        "--expdis-stabilization-mode", default=C8_RND_LIFECYCLE,
        choices=list(EXPDIS_STABILIZATION_MODES),
        help="RND lifecycle. Default preserves frozen C8 candidate-attempt "
             "updates; C9 freezes scoring for the learner step and updates "
             "once on the final selected rows (requires exact scientific label)",
    )
    p.add_argument(
        "--novelty-normalization", default="raw",
        choices=list(NOVELTY_NORMALIZATION_MODES),
        help="Scout novelty scaling. selected_batch_zscore is C9-only and is "
             "computed over final selected raw scores before correct-only gating",
    )
    p.add_argument(
        "--c9-execution-profile", default="",
        choices=list(C9_EXECUTION_PROFILES),
        help="preregistered stabilization execution profile; scientific jobs "
             "must pair canary/full with an exact --stabilization-cell",
    )
    p.add_argument(
        "--c10-execution-profile", default="",
        choices=list(C10_EXECUTION_PROFILES),
        help="separately preregistered survivor-consistent Overlong Filtering "
             "profile; mutually exclusive with every C9 cell/profile",
    )
    p.add_argument(
        "--stabilization-cell", default="",
        choices=list(C9_STABILIZATION_CELLS),
        help="preregistered C9 mechanism cell C1--C5",
    )
    p.add_argument("--prompts-per-step", type=int, default=4,
                   help="contract 4; GPU-MAX 8/16 (128/256 rollouts per step)")
    p.add_argument("--num-generations", type=int, default=16)
    p.add_argument("--max-prompt-len", type=int, default=2048)
    p.add_argument("--max-completion-len", type=int, default=16384)
    p.add_argument("--loss-denominator-len", type=int, default=16384,
                   help="Dr.GRPO fixed loss-normalizer length (per row). "
                        "Pinned at the contract 16384 regardless of "
                        "--completion-budget so gradient scale is identical "
                        "across budget tiers and vs the TPU runs.")
    p.add_argument("--completion-budget", type=int, default=0,
                   help="GPU-MAX: completion budget N (e.g. 24576/32768); "
                        "overrides max-completion-len and splits soft-overlong "
                        "as int(0.8*N)/N-int(0.8*N); 0 = contract tier")
    p.add_argument("--mask-truncated", dest="mask_truncated", action="store_true",
                   default=False,
                   help="optional DAPO Overlong Filtering: exclude clipped "
                        "completions from the policy loss. Off by default for "
                        "the canonical c8/TPU-reference contract; enable only "
                        "as a labeled ablation or to reproduce c7")
    p.add_argument("--no-mask-truncated", dest="mask_truncated",
                   action="store_false")
    p.add_argument(
        "--truncation-baseline-mode", default="all_rows",
        choices=list(TRUNCATION_BASELINE_MODES),
        help="group-mean and eligibility population for truncated-loss "
             "masking. unmasked_survivors is valid only under exact C10",
    )
    p.add_argument("--ppo-mode", action="store_true",
                   help="GPU-MAX: real PPO clipping with frozen rollout-time "
                        "old logprobs (default: contract REINFORCE semantics)")
    p.add_argument("--updates-per-rollout", type=int, default=1,
                   help="GPU-MAX (with --ppo-mode): K PPO epochs per rollout batch")
    p.add_argument("--clip-epsilon-low", type=float, default=0.2)
    p.add_argument("--clip-epsilon-high", type=float, default=0.28)
    p.add_argument("--optimizer", default="adamw", choices=["adamw", "adafactor"])
    p.add_argument("--dynamic-max-attempts", type=int, default=8)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--adam-beta1", type=float, default=0.9)
    p.add_argument("--adam-beta2", type=float, default=0.95)
    p.add_argument("--adam-eps", type=float, default=1e-8)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--incorrect-novelty-scale", type=float, default=0.0,
               help="novelty credit for WRONG answers. The measured TPU flagship "
                    "(W&B uc1a-optim-20260427_104408) ran 0.0 (2026-04-05 fix: "
                    "wrong-but-novel must not be net-positive); contract.py's >0 "
                    "invariant contradicts the measured runs. 1.0 reproduces the "
                    "observed reward-hacking spiral (98%% clip by step 26).")
    p.add_argument("--soft-overlong-expected-len", type=int, default=13107)
    p.add_argument("--soft-overlong-cache-len", type=int, default=3277)
    p.add_argument("--rnd-lr", type=float, default=1e-4)
    p.add_argument("--rnd-hidden", type=int, default=512)
    p.add_argument("--novelty-layers", default="7,14,21")
    p.add_argument("--novelty-feature-max-length", type=int, default=16384)  # measured flagship value
    p.add_argument("--novelty-batch-size", type=int, default=2)
    p.add_argument("--dataset", default="dapo_math_17k")
    p.add_argument("--max-train-examples", type=int, default=20000)
    p.add_argument(
        "--rl-validity-mode", default="legacy", choices=list(RL_VALIDITY_MODES),
        help="legacy preserves the measured c7 sync/gating/resume behavior; "
             "c8 requires a managed pool, syncs after every learner update, "
             "uses novelty-only homogeneous-group eligibility, and restores "
             "optimizer state",
    )
    p.add_argument(
        "--reload-every", type=int, default=10,
        help="legacy vLLM weight-sync cadence (steps); c8 overrides this and "
             "syncs after every learner update",
    )
    p.add_argument("--save-every", type=int, default=25, help="checkpoint cadence (steps)")
    p.add_argument(
        "--keep-checkpoints", type=int, default=1,
        help="local step checkpoints retained; c8 keeps one optimizer-complete "
             "checkpoint while durable W&B versions remain available",
    )
    p.add_argument("--ckpt-artifact-every", type=int, default=100,
                   help="push the step checkpoint (and the phase-final model) to "
                        "W&B every N steps; 0 = never (the Lepton workspace has no durable "
                        "disk — this is what --resume-from restores after pod loss)")
    p.add_argument("--probe-every", type=int, default=25,
                   help="mid-training AIME24 probe cadence (steps); 0 = off. "
                        "Fixed protocol: 30 problems x 4 samples, 32768 tokens, "
                        "T=0.6, logged as the non-headline "
                        "eval/AIME24_avg_at_4 fraction at the global step "
                        "(serving pool must fit 2048+32768)")
    p.add_argument("--probe-dataset", default="aime24",
                   help='"aime24" (default) or a local {"problem","answer"} '
                        "JSONL for smoke tests")
    p.add_argument("--sync-dir", default="/workspace", help="root for vLLM sync exports")
    p.add_argument("--endpoints", default="", help="attach to running vLLM servers (no reload)")
    p.add_argument("--vllm-gpus", default="", help="comma GPU ids for a managed vLLM pool")
    p.add_argument("--served-model-name", default=None, help="constant across reloads")
    p.add_argument("--vllm-port-base", type=int, default=8000)
    p.add_argument("--vllm-max-model-len", type=int, default=18432)
    p.add_argument("--vllm-gpu-mem-util", type=float, default=0.90)
    p.add_argument("--vllm-max-num-seqs", type=int, default=32)
    p.add_argument("--gen-concurrency", type=int, default=8, help="per-server in-flight requests")
    p.add_argument("--gen-timeout", type=float, default=3600.0)
    p.add_argument("--max-consecutive-gen-failures", type=int, default=3,
                   help="abort after N consecutive generate() calls with failed "
                        "requests (error groups are dropped, never trained on)")
    p.add_argument("--resume-from", default="",
                   help="step_* checkpoint dir; restores model/step/RND, plus "
                        "optimizer moments in c8, and trims trajectory JSONLs "
                        "+ metrics rows past the resume step")
    p.add_argument("--metrics-jsonl", default="",
                   help="shared metrics JSONL (parity keys per step); the "
                        "pipeline passes <output-dir>/metrics.jsonl")
    p.add_argument("--global-step-offset", type=int, default=0,
                   help="added to the step for W&B/metrics global_step so "
                        "multi-round curves are monotonic")
    p.add_argument("--num-scouts", type=int, default=1,
                   help="K scouts feeding this pipeline run (logging column)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--round", type=int, default=1, help="pipeline round (logging + sync-dir/seed namespacing)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--param-dtype", default="float32", choices=["float32", "bfloat16"])
    p.add_argument("--chunk-tokens", type=int, default=256)
    p.add_argument("--wandb", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    final_dir = run_grpo(args)
    print(f"[grpo] DONE phase={args.phase} final={final_dir}", flush=True)


if __name__ == "__main__":
    main()
