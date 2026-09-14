"""Fail-closed contracts for multi-round ExpDis and multi-explorer ExpDis (MR-ME).

This module is intentionally orchestration-only and has no torch/vLLM imports.
It defines the scientific identity that the GPU pipeline must bind before it
launches expensive work:

* ``K >= 1`` independent Scouts in every round, all initialized from the same
  round input and trained at the same per-round novelty weight;
* distinct, deterministic Scout seeds (and therefore distinct RND targets and
  rollout/data-order RNG streams);
* one deterministic, disjoint ``1 / R`` dataset shard per round, shared by all
  ``K`` Scouts in that round;
* one concrete QualityPool operation over the union of all ``K`` *raw* Scout
  trajectory files, followed by one SFT and one Central per round; and
* every round-(r+1) Scout initialized from the one round-r Central.

Terminology note
----------------
The reviewed paper calls the merged operation ``QualityPool``.  The current
corrected GPU/ExpDis contract implements that role with ``coverage_pool_c8``:
verifier-correct, naturally terminated, matching boxed answer, unclipped,
one row per problem, deterministic reward-independent selection.  It is *not*
the older ``quality_pool`` token-window/loop heuristic.  MR-ME therefore binds
the concrete policy name ``coverage_pool_c8`` and must not silently switch
between the two implementations.  ``K=1`` uses the same evidence machinery but
has its own versioned method identity; it must never be relabelled as MR-ME.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tmx_gpu import common

MRME_CONTRACT_VERSION = "gpu_multi_explorer_contract_v3"
MRME_STATE_BINDING_VERSION = "gpu_multi_explorer_state_binding_v3"
MRME_POOLING_POLICY = "coverage_pool_c8"
MRME_POOLING_ROLE = "quality_pool"
MRME_SHARD_POLICY = "uniform"
INTEGRATED_MULTISCOUT_METHOD = "integrated_multiscout"
DISJOINT_SHARD_MRME_METHOD = "disjoint_shard_annealed_mrme"
DISJOINT_SHARD_SINGLESCOUT_MULTIROUND_METHOD = (
    "disjoint_shard_annealed_single_scout_multiround")
DISJOINT_SHARD_SINGLESCOUT_MULTIROUND_METHOD_VERSION = (
    "gpu_disjoint_shard_annealed_single_scout_multiround_v1")
C8_MULTIEXPLORER_PROFILE = "c8_candidate_attempt_multi_explorer"
C9_MULTIEXPLORER_PROFILE = "c9_selected_batch_multi_explorer"
SINGLESCOUT_MULTIROUND_PROFILE = (
    "c9_selected_batch_single_scout_multiround")
C9_MULTIEXPLORER_SCIENTIFIC_VARIANT = (
    "expdis_c9_selected_batch_multi_explorer")
C9_MULTIEXPLORER_SCOUT_EVIDENCE_VERSION = (
    "c9_multi_explorer_scout_completion_v2")
C9_MULTIEXPLORER_GATE_EVIDENCE_VERSION = (
    "c9_multi_explorer_behavior_gate_v2")
C9_MULTIEXPLORER_SFT_GATE_EVIDENCE_VERSION = (
    "c9_multi_explorer_pooled_sft_behavior_gate_v1")
C9_MULTIEXPLORER_CENTRAL_LINEAGE_VERSION = (
    "c9_multi_explorer_sft_central_lineage_v1")
GRPO_FINAL_MODEL_ARTIFACT_BINDING_VERSION = \
    "grpo_final_model_artifact_v1"
MRME_PORTABLE_CAMPAIGN_VERSION = \
    "gpu_multi_explorer_portable_campaign_v1"
MRME_TRAJECTORY_PROVENANCE_VERSION = "gpu_mrme_trajectory_slot_v1"
STRICT_EXTERNAL_P1_REQUIREMENT = {
    "required": True,
    "protocol": "strict_external_p1_six_benchmark_matrix",
    "may_use_training_probes": False,
    "training_completion_is_headline_eligible": False,
}

# The preregistered K=3 round-1 campaign uses seed 0 as canonical
# SingleScout and seeds 1/2 as the additional components. Later rounds take
# the next non-overlapping K-seed block, so every (round, Scout) gets a fresh
# RND/data-order seed while round 1 remains exactly 0/1/2.


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def is_c9_multi_explorer(contract: Mapping[str, Any]) -> bool:
    """Return true only for the explicitly stabilized multi-explorer profile."""
    return str(contract.get("stabilization_profile", "")) == \
        C9_MULTIEXPLORER_PROFILE


def is_single_scout_multiround(contract: Mapping[str, Any]) -> bool:
    """Return true only for the explicitly versioned K=1 round method."""
    return (
        str(contract.get("method", "")) ==
        DISJOINT_SHARD_SINGLESCOUT_MULTIROUND_METHOD
        and str(contract.get("method_version", "")) ==
        DISJOINT_SHARD_SINGLESCOUT_MULTIROUND_METHOD_VERSION
        and int(contract.get("scouts_per_round", 0)) == 1
        and 1 <= int(contract.get("rounds", 0)) <= 4
    )


def _signed_payload(version: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    if "version" in payload or "evidence_sha256" in payload:
        raise ValueError("multi-explorer signed payload contains reserved fields")
    value = {"version": str(version), **dict(payload)}
    value["evidence_sha256"] = canonical_json_sha256(value)
    return value


def _validate_signed_payload(
    value: Mapping[str, Any], expected_version: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("multi-explorer evidence must be an object")
    result = dict(value)
    if result.get("version") != expected_version:
        raise ValueError("multi-explorer evidence version mismatch")
    unsigned = dict(result)
    recorded = unsigned.pop("evidence_sha256", None)
    if recorded != canonical_json_sha256(unsigned):
        raise ValueError("multi-explorer evidence SHA256 mismatch")
    return result


def sign_c9_multi_explorer_gate_evidence(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    return _signed_payload(C9_MULTIEXPLORER_GATE_EVIDENCE_VERSION, payload)


def sign_c9_multi_explorer_sft_gate_evidence(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Sign the pooled-SFT health gate under its own campaign identity."""
    return _signed_payload(
        C9_MULTIEXPLORER_SFT_GATE_EVIDENCE_VERSION, payload)


def validate_grpo_final_artifact_binding(
    value: Mapping[str, Any], *, expected_phase: str,
    expected_round: int, expected_step: int,
    expected_policy_manifest: Mapping[str, Any] | None = None,
    expected_mrme_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate an immutable, synchronously committed GRPO final artifact."""
    if not isinstance(value, Mapping):
        raise ValueError("GRPO final model artifact binding must be an object")
    binding = dict(value)
    unsigned = dict(binding)
    recorded_sha = unsigned.pop("binding_sha256", None)
    if recorded_sha != canonical_json_sha256(unsigned):
        raise ValueError("GRPO final model artifact binding SHA256 mismatch")
    if binding.get("version") != GRPO_FINAL_MODEL_ARTIFACT_BINDING_VERSION or \
            binding.get("artifact_subdir") != "." or \
            binding.get("phase") != str(expected_phase) or \
            binding.get("round") != int(expected_round) or \
            binding.get("step") != int(expected_step):
        raise ValueError("GRPO final model artifact identity mismatch")
    source = binding.get("source")
    if not isinstance(source, Mapping) or \
            source.get("artifact_type") != "model" or \
            not source.get("artifact_digest"):
        raise ValueError("GRPO final model artifact source is incomplete")
    common.require_immutable_wandb_artifact_ref(
        str(source.get("artifact_ref", "")),
        field="GRPO final model artifact")
    for name in ("model_file_manifest", "policy_file_manifest"):
        manifest = binding.get(name)
        if not isinstance(manifest, Mapping) or manifest.get("version") != \
                common.DIRECTORY_FILE_MANIFEST_VERSION or \
                not isinstance(manifest.get("sha256"), str) or \
                not manifest.get("files"):
            raise ValueError(f"GRPO final artifact lacks {name}")
    if expected_policy_manifest is not None and \
            binding.get("policy_file_manifest") != dict(
                expected_policy_manifest):
        raise ValueError("GRPO final artifact policy manifest mismatch")
    if expected_mrme_provenance is not None and \
            binding.get("mrme_provenance") != dict(
                expected_mrme_provenance):
        raise ValueError("GRPO final artifact MR-ME provenance mismatch")
    metadata = binding.get("artifact_metadata")
    if not isinstance(metadata, Mapping) or \
            metadata.get("phase") != str(expected_phase) or \
            metadata.get("round") != int(expected_round) or \
            metadata.get("step") != int(expected_step):
        raise ValueError("GRPO final artifact metadata mismatch")
    return binding


@dataclass(frozen=True)
class ScoutIdentity:
    round_index: int
    scout_index: int
    seed: int
    stage_key: str
    directory_name: str


@dataclass(frozen=True)
class RoundIdentity:
    round_index: int
    lambda_novelty: float
    shard_index: int
    shard_count: int
    shard_fraction_numerator: int
    shard_fraction_denominator: int
    scouts: tuple[ScoutIdentity, ...]


def scout_seed(
    base_seed: int,
    round_index: int,
    scout_index: int,
    scouts_per_round: int,
) -> int:
    """Stable, collision-free seed for one Scout.

    Indices are one-based. Default base seed 0 and K=3 resolve to 0/1/2 in
    round 1, 3/4/5 in round 2, and so on. Python, numpy, torch, vLLM request
    seeds, and RND construction all consume this resolved seed downstream.
    """
    if int(round_index) < 1:
        raise ValueError("round_index must be >= 1")
    if int(scout_index) < 1:
        raise ValueError("scout_index must be >= 1")
    if int(scouts_per_round) < 1 or int(scout_index) > int(scouts_per_round):
        raise ValueError("scout_index must be within scouts_per_round")
    seed = int(base_seed) + (int(round_index) - 1) * int(scouts_per_round) + (int(scout_index) - 1)
    if seed < 0 or seed >= 2**31:
        raise ValueError(f"derived Scout seed {seed} falls outside vLLM's [0, 2^31) range")
    return seed


def scout_identity(
    base_seed: int,
    round_index: int,
    scout_index: int,
    scouts_per_round: int,
) -> ScoutIdentity:
    return ScoutIdentity(
        round_index=int(round_index),
        scout_index=int(scout_index),
        seed=scout_seed(base_seed, round_index, scout_index, scouts_per_round),
        stage_key=f"r{int(round_index):02d}/scout_{int(scout_index):02d}",
        directory_name=f"scout_{int(scout_index):02d}",
    )


def validate_lambda_schedule(
    schedule: Sequence[float],
    *,
    rounds: int,
) -> tuple[float, ...]:
    if int(rounds) < 1:
        raise ValueError("rounds must be >= 1")
    values = tuple(float(value) for value in schedule)
    if len(values) != int(rounds):
        raise ValueError(f"lambda schedule has {len(values)} entries for {int(rounds)} rounds")
    if any(not (value >= 0.0) for value in values):
        raise ValueError("lambda schedule values must be finite and >= 0")
    if any(value != value or value in (float("inf"), float("-inf")) for value in values):
        raise ValueError("lambda schedule values must be finite")
    return values


def build_contract(
    *,
    model: str,
    dataset: str,
    max_train_examples: int,
    rounds: int,
    scouts_per_round: int,
    lambda_schedule: Sequence[float],
    base_seed: int,
    scout_steps: int,
    central_steps: int,
    prompts_per_step: int,
    num_generations: int,
    pooling_policy: str,
    max_accepted: int,
    harvest_max_passes: int,
    stabilization_profile: str = C8_MULTIEXPLORER_PROFILE,
    training_contract: Mapping[str, Any] | None = None,
    code_identity: Mapping[str, Any] | None = None,
    model_revision: str = "",
    dataset_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the immutable multi-Scout or K=1 multi-round identity.

    ``K > 1, R == 1`` is the integrated MultiScout control.  ``K > 1, R > 1``
    is disjoint-shard annealed MR-ME.  ``K == 1, 1 <= R <= 4`` is a separate,
    explicitly versioned single-Scout multi-round method.  Both multi-round
    methods bind a predeclared deterministic 1/R partition and are never a
    failure-targeted curriculum.
    """
    rounds = int(rounds)
    scouts_per_round = int(scouts_per_round)
    if rounds < 1:
        raise ValueError("multi-explorer campaigns require --rounds >= 1")
    if scouts_per_round < 1:
        raise ValueError("round campaigns require --scouts-per-round >= 1")
    single_scout_multiround = scouts_per_round == 1
    if single_scout_multiround and not 1 <= rounds <= 4:
        raise ValueError(
            "single-Scout multi-round campaigns require 1 <= --rounds <= 4")
    stabilization_profile = str(stabilization_profile)
    if stabilization_profile not in {
            C8_MULTIEXPLORER_PROFILE, C9_MULTIEXPLORER_PROFILE}:
        raise ValueError(
            f"unknown multi-explorer stabilization profile "
            f"{stabilization_profile!r}")
    if single_scout_multiround and \
            stabilization_profile != C9_MULTIEXPLORER_PROFILE:
        raise ValueError(
            "single-Scout multi-round publication requires the explicit "
            "stabilized C9 profile")
    if not single_scout_multiround and rounds == 1 and \
            stabilization_profile != C9_MULTIEXPLORER_PROFILE:
        raise ValueError(
            "integrated R=1 MultiScout is enabled only under the explicit "
            "stabilized C9 multi-explorer profile")
    if int(max_train_examples) <= 0:
        raise ValueError("MR-ME requires max_train_examples > 0")
    if int(scout_steps) <= 0 or int(central_steps) <= 0:
        raise ValueError("MR-ME requires positive Scout and Central step budgets")
    if int(prompts_per_step) <= 0 or int(num_generations) <= 0:
        raise ValueError("MR-ME rollout geometry must be positive")
    if str(pooling_policy) != MRME_POOLING_POLICY:
        raise ValueError(
            "MR-ME QualityPool is concretely bound to "
            f"{MRME_POOLING_POLICY!r}, got {pooling_policy!r}"
        )
    if int(max_accepted) != 500:
        raise ValueError("MR-ME requires the frozen global QualityPool cap of 500")
    if int(harvest_max_passes) != 0:
        raise ValueError(
            "MR-ME pools exactly K raw Scout sources once per round; "
            "post-Scout harvest/top-up is forbidden"
        )
    if not isinstance(training_contract, Mapping) or not training_contract:
        raise ValueError("MR-ME requires an explicit resolved training contract")
    if not isinstance(code_identity, Mapping) or not code_identity:
        raise ValueError("MR-ME requires an immutable code-artifact identity")
    model_revision = str(model_revision or "")
    if not model_revision:
        raise ValueError("MR-ME requires an immutable base-model revision")
    if not isinstance(dataset_identity, Mapping) or not dataset_identity:
        raise ValueError("MR-ME requires an immutable training-dataset identity")
    dataset_identity = dict(dataset_identity)
    identity_core = dict(dataset_identity)
    identity_digest = identity_core.pop("identity_sha256", None)
    if identity_digest != canonical_json_sha256(identity_core):
        raise ValueError("MR-ME training-dataset identity SHA256 mismatch")
    if stabilization_profile == C9_MULTIEXPLORER_PROFILE and (
            str(model) != common.CANONICAL_TRAINING_MODEL_ID
            or model_revision != common.CANONICAL_TRAINING_MODEL_REVISION):
        raise ValueError("C9 MR-ME requires the immutable canonical Qwen3-1.7B revision")
    # Production templates expose the same identities independently at every
    # child stage.  Reject a self-consistent outer contract that tries to bind
    # different bytes inside Scout/Central/SFT runtime templates.
    for stage in ("scout_grpo", "central_grpo"):
        runtime = training_contract.get(stage)
        if isinstance(runtime, Mapping):
            if runtime.get("base_model_revision") != model_revision:
                raise ValueError(f"MR-ME {stage} base-model revision mismatch")
            if runtime.get("training_dataset_identity") != dataset_identity:
                raise ValueError(f"MR-ME {stage} training-dataset identity mismatch")
    sft_runtime = training_contract.get("central_sft")
    if isinstance(sft_runtime, Mapping) and \
            sft_runtime.get("base_model_revision") != model_revision:
        raise ValueError("MR-ME central_sft base-model revision mismatch")
    lambdas = validate_lambda_schedule(lambda_schedule, rounds=rounds)
    if stabilization_profile == C9_MULTIEXPLORER_PROFILE:
        if any(value <= 0.0 for value in lambdas):
            raise ValueError("C9 multi-explorer Scout lambdas must be > 0")
        if rounds == 1 and lambdas != (0.5,):
            raise ValueError("integrated C9 MultiScout requires lambda=0.5")
    if rounds > 1 and any(
            later >= earlier for earlier, later in zip(lambdas, lambdas[1:])):
        raise ValueError(
            "disjoint_shard_annealed_mrme requires a predeclared "
            "strictly decreasing lambda schedule")
    round_records = []
    for round_index, lam in enumerate(lambdas, start=1):
        scouts = tuple(
            scout_identity(
                base_seed,
                round_index,
                scout_index,
                scouts_per_round,
            )
            for scout_index in range(1, scouts_per_round + 1)
        )
        round_seeds = [item.seed for item in scouts]
        if len(round_seeds) != len(set(round_seeds)):
            raise ValueError(f"MR-ME Scout seed collision: {round_seeds}")
        round_record = asdict(
            RoundIdentity(
                round_index=round_index,
                lambda_novelty=float(lam),
                shard_index=round_index - 1,
                shard_count=rounds,
                shard_fraction_numerator=1,
                shard_fraction_denominator=rounds,
                scouts=scouts,
            )
        )
        # JSON is the durable representation; keep the in-memory object in
        # that same list-based shape so a save/load round trip is exact.
        round_record["scouts"] = list(round_record["scouts"])
        round_records.append(round_record)
    method = (
        DISJOINT_SHARD_SINGLESCOUT_MULTIROUND_METHOD
        if single_scout_multiround else
        INTEGRATED_MULTISCOUT_METHOD if rounds == 1 else
        DISJOINT_SHARD_MRME_METHOD)
    core: dict[str, Any] = {
        "version": MRME_CONTRACT_VERSION,
        "method": method,
        "method_is_failure_targeted_curriculum": False,
        "round_assignment_signal": "predeclared_round_index_only",
        "stabilization_profile": stabilization_profile,
        "scientific_variant": (
            C9_MULTIEXPLORER_SCIENTIFIC_VARIANT
            if stabilization_profile == C9_MULTIEXPLORER_PROFILE else ""),
        "model": str(model),
        "model_revision": model_revision,
        "dataset": str(dataset),
        "dataset_identity": dataset_identity,
        "max_train_examples": int(max_train_examples),
        "rounds": rounds,
        "scouts_per_round": scouts_per_round,
        "lambda_schedule": list(lambdas),
        "base_seed": int(base_seed),
        "scout_seed_rule": ("base_seed_plus_zero_based_round_times_k_plus_zero_based_scout_index"),
        "scout_steps_per_scout": int(scout_steps),
        "central_steps_per_round": int(central_steps),
        "prompts_per_step": int(prompts_per_step),
        "num_generations": int(num_generations),
        "expected_raw_rows_per_scout": (
            int(scout_steps) * int(prompts_per_step) * int(num_generations)
        ),
        "rollout_budget_mode": "full_per_scout_per_round",
        "shard_changes_rollout_budget": False,
        "pooling_role": MRME_POOLING_ROLE,
        "pooling_policy": MRME_POOLING_POLICY,
        "pooling_scope": "one_union_of_all_k_raw_scout_sources_per_round",
        "per_scout_preselection": False,
        "harvest_or_topup": False,
        "max_accepted": int(max_accepted),
        "central_count_per_round": 1,
        "shard_policy": MRME_SHARD_POLICY,
        "shard_semantics": (
            "full_unsharded_dataset_shared_by_all_k_scouts"
            if rounds == 1 and not single_scout_multiround else
            "deterministic_disjoint_one_over_r_shared_within_round"),
        "dataset_membership_mode": (
            "full_unsharded"
            if rounds == 1 and not single_scout_multiround else
            "deterministic_disjoint_one_over_r"),
        "external_publication_evaluation": dict(
            STRICT_EXTERNAL_P1_REQUIREMENT),
        "rounds_spec": round_records,
        "training_contract": dict(training_contract),
        "code_identity": dict(code_identity),
    }
    if single_scout_multiround:
        core["method_version"] = \
            DISJOINT_SHARD_SINGLESCOUT_MULTIROUND_METHOD_VERSION
        core["single_scout_multiround"] = True
    return {**core, "contract_sha256": canonical_json_sha256(core)}


def validate_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(contract, Mapping):
        raise ValueError("MR-ME contract must be an object")
    raw = dict(contract)
    digest = raw.pop("contract_sha256", None)
    if raw.get("version") != MRME_CONTRACT_VERSION:
        raise ValueError("MR-ME contract version mismatch")
    if digest != canonical_json_sha256(raw):
        raise ValueError("MR-ME contract SHA256 mismatch")
    # Rebuild from the authoritative scalar fields.  This catches a
    # self-consistently re-hashed but semantically malformed rounds_spec.
    rebuilt = build_contract(
        model=str(raw.get("model", "")),
        dataset=str(raw.get("dataset", "")),
        max_train_examples=int(raw.get("max_train_examples", 0)),
        rounds=int(raw.get("rounds", 0)),
        scouts_per_round=int(raw.get("scouts_per_round", 0)),
        lambda_schedule=raw.get("lambda_schedule", []),
        base_seed=int(raw.get("base_seed", 0)),
        scout_steps=int(raw.get("scout_steps_per_scout", 0)),
        central_steps=int(raw.get("central_steps_per_round", 0)),
        prompts_per_step=int(raw.get("prompts_per_step", 0)),
        num_generations=int(raw.get("num_generations", 0)),
        pooling_policy=str(raw.get("pooling_policy", "")),
        max_accepted=int(raw.get("max_accepted", 0)),
        harvest_max_passes=0 if raw.get("harvest_or_topup") is False else 1,
        stabilization_profile=str(raw.get("stabilization_profile", "")),
        training_contract=raw.get("training_contract", {}),
        code_identity=raw.get("code_identity", {}),
        model_revision=str(raw.get("model_revision", "")),
        dataset_identity=raw.get("dataset_identity", {}),
    )
    if rebuilt != dict(contract):
        raise ValueError("MR-ME contract fields are not canonical")
    return rebuilt


def bind_pipeline_state(
    state: dict[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind resume state to one exact MR-ME contract, or fail closed.

    A state file created by the legacy single-Scout pipeline has no binding and
    cannot be upgraded after stages already exist; doing so could relabel mixed
    rounds or Scouts as MR-ME.
    """
    validated = validate_contract(contract)
    existing = state.get("mrme_binding")
    expected = {
        "version": MRME_STATE_BINDING_VERSION,
        "contract": validated,
        "contract_sha256": validated["contract_sha256"],
    }
    if existing is None:
        if state.get("stages"):
            raise ValueError("cannot attach an MR-ME contract to non-empty unbound pipeline state")
        state["mrme_binding"] = expected
    elif existing != expected:
        raise ValueError("pipeline_state MR-ME contract mismatch")
    return state


def expected_scout_stage_keys(contract: Mapping[str, Any]) -> tuple[str, ...]:
    validated = validate_contract(contract)
    return tuple(
        scout["stage_key"]
        for round_record in validated["rounds_spec"]
        for scout in round_record["scouts"]
    )


def round_spec(contract: Mapping[str, Any], round_index: int) -> dict[str, Any]:
    validated = validate_contract(contract)
    idx = int(round_index)
    if idx < 1 or idx > int(validated["rounds"]):
        raise ValueError(f"round_index must be in [1, {validated['rounds']}], got {idx}")
    return dict(validated["rounds_spec"][idx - 1])


def scout_execution_plan(
    contract: Mapping[str, Any],
    round_index: int,
    *,
    output_dir: str,
    sync_dir: str,
    run_name: str,
    scout_init: str,
) -> list[dict[str, Any]]:
    """Resolve non-overlapping directories/names for every Scout in a round."""
    validated = validate_contract(contract)
    spec = round_spec(validated, round_index)
    round_dir = os.path.join(os.path.realpath(str(output_dir)), f"round_{int(round_index):02d}")
    sync_root = os.path.realpath(str(sync_dir))
    if not str(run_name).strip():
        raise ValueError("MR-ME run_name is empty")
    init_identity = model_handoff_identity(
        scout_init,
        expected_phase="central" if int(round_index) > 1 else None,
        model_revision=(
            str(validated["model_revision"]) if int(round_index) == 1 else ""),
    )
    plans = []
    for scout in spec["scouts"]:
        scout_index = int(scout["scout_index"])
        suffix = f"mrme-r{int(round_index):02d}-s{scout_index:02d}"
        plans.append(
            {
                "round_index": int(round_index),
                "scout_index": scout_index,
                "seed": int(scout["seed"]),
                "lambda_novelty": float(spec["lambda_novelty"]),
                "stage_key": str(scout["stage_key"]),
                "output_dir": os.path.join(round_dir, "scouts", f"scout_{scout_index:02d}"),
                "sync_dir": os.path.join(sync_root, suffix),
                "run_name": f"{run_name}-{suffix}",
                "scout_init": str(scout_init),
                "scout_init_identity": init_identity,
                "expected_raw_rows": int(contract["expected_raw_rows_per_scout"]),
            }
        )
    output_paths = [item["output_dir"] for item in plans]
    sync_paths = [item["sync_dir"] for item in plans]
    names = [item["run_name"] for item in plans]
    if (
        len(set(output_paths)) != len(plans)
        or len(set(sync_paths)) != len(plans)
        or len(set(names)) != len(plans)
    ):
        raise RuntimeError("MR-ME Scout execution namespaces collide")
    return plans


def shard_environment(
    contract: Mapping[str, Any],
    round_index: int,
    manifest_path: str,
) -> dict[str, str]:
    """Environment consumed by ``tmx_jax.data._apply_round_shard``.

    For ``R>1``, ``TMX_DATASET_SHARD_REQUIRED=1`` converts
    absence/malformed fields from a silent full-dataset fallback into an
    immediate error.  ``R=1`` (including the versioned K=1 method) instead uses
    a 1/1 membership manifest with ``REQUIRED=0`` because the shared loader
    reserves ``REQUIRED=1`` for shard counts greater than one.  The manifest
    path still forces exact full-membership evidence.  All K Scouts in a round
    receive the same path; the loader writes it once and byte-compares every
    later load.
    """
    spec = round_spec(contract, round_index)
    path = os.path.realpath(str(manifest_path))
    if not path:
        raise ValueError("MR-ME shard manifest path is empty")
    required = int(contract["rounds"]) > 1
    return {
        "TMX_DATASET_SHARD_REQUIRED": "1" if required else "0",
        "TMX_DATASET_SHARD_INDEX": str(int(spec["shard_index"])),
        "TMX_DATASET_SHARD_COUNT": str(int(spec["shard_count"])),
        "TMX_DATASET_SHARD_POLICY": MRME_SHARD_POLICY,
        "TMX_DATASET_SHARD_SEED": str(int(contract["base_seed"])),
        "TMX_DATASET_SHARD_MANIFEST_PATH": path,
    }


def validate_round_scout_sources(
    contract: Mapping[str, Any],
    round_index: int,
    sources: Sequence[Mapping[str, Any]],
    *,
    expected_shard_manifest_sha256: str,
    expected_scout_init_identity: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Validate the K raw Scout sources before the one pooling operation."""
    spec = round_spec(contract, round_index)
    expected = {int(item["scout_index"]): item for item in spec["scouts"]}
    if len(sources) != len(expected):
        raise ValueError(
            f"round {round_index} needs exactly {len(expected)} raw Scout "
            f"sources, got {len(sources)}"
        )
    normalized: list[dict[str, Any]] = []
    seen_indexes: set[int] = set()
    seen_paths: set[str] = set()
    for source in sources:
        item = dict(source)
        record_digest = item.pop("record_sha256", None)
        if record_digest != canonical_json_sha256(item):
            raise ValueError("MR-ME source record SHA256 mismatch")
        item["record_sha256"] = record_digest
        scout_index = int(item.get("scout_index", 0))
        if scout_index not in expected or scout_index in seen_indexes:
            raise ValueError(
                f"round {round_index} has duplicate/unexpected Scout index {scout_index}"
            )
        if int(item.get("round_index", 0)) != int(round_index):
            raise ValueError("MR-ME source round mismatch")
        if int(item.get("seed", -1)) != int(expected[scout_index]["seed"]):
            raise ValueError("MR-ME source Scout seed mismatch")
        if float(item.get("lambda_novelty", -1.0)) != float(spec["lambda_novelty"]):
            raise ValueError("MR-ME source lambda mismatch")
        if item.get("contract_sha256") != contract.get("contract_sha256"):
            raise ValueError("MR-ME source campaign contract mismatch")
        if item.get("scout_init_identity") != dict(expected_scout_init_identity):
            raise ValueError(
                "MR-ME sources were not all initialized from the exact same round model"
            )
        if item.get("shard_manifest_sha256") != str(expected_shard_manifest_sha256):
            raise ValueError("MR-ME source shard manifest mismatch")
        trajectory_path = os.path.realpath(str(item.get("trajectory_path", "")))
        if not trajectory_path or not os.path.isfile(trajectory_path):
            raise ValueError("MR-ME source trajectory file is missing")
        if trajectory_path in seen_paths:
            raise ValueError("MR-ME source trajectory path is duplicated")
        expected_sha = str(item.get("trajectory_sha256", ""))
        if _file_sha256(trajectory_path) != expected_sha:
            raise ValueError("MR-ME source trajectory SHA256 mismatch")
        final_identity = validate_scout_final(
            str(item.get("scout_final", "")),
            expected_round=int(round_index),
            expected_seed=int(expected[scout_index]["seed"]),
            expected_lambda=float(spec["lambda_novelty"]),
            expected_steps=int(contract["scout_steps_per_scout"]),
        )
        validate_scout_runtime_contract(
            contract,
            final_identity,
            expected_scout_init_identity=expected_scout_init_identity,
            expected_shard_manifest_sha256=str(expected_shard_manifest_sha256),
            expected_scout_index=scout_index,
        )
        if item.get("scout_final_identity") != final_identity:
            raise ValueError("MR-ME source Scout-final identity changed")
        final_artifact = item.get("scout_final_artifact")
        if final_artifact is not None:
            validate_grpo_final_artifact_binding(
                final_artifact, expected_phase="scout",
                expected_round=int(round_index),
                expected_step=int(contract["scout_steps_per_scout"]),
                expected_policy_manifest=final_identity.get(
                    "checkpoint_file_manifest"),
                expected_mrme_provenance=final_identity.get(
                    "mrme_provenance"))
            common.verify_directory_file_manifest(
                str(item.get("scout_final", "")),
                final_artifact["model_file_manifest"])
        elif is_c9_multi_explorer(contract):
            raise ValueError(
                "C9 multi-explorer Scout lacks a durable final-model artifact")
        if is_c9_multi_explorer(contract):
            completion = validate_c9_multi_explorer_scout_evidence(
                item.get("c9_multi_explorer_scout_evidence", {}),
                contract=contract,
                round_index=round_index,
                scout_index=scout_index,
            )
            gate_path = os.path.realpath(str(
                item.get("c9_behavior_gate_path", "")))
            if not os.path.isfile(gate_path) or \
                    load_json_object(gate_path) != completion["behavior_gate"]:
                raise ValueError(
                    "MR-ME source behavior-gate evidence is missing or drifted")
            validate_c9_behavior_gate_support_files(
                gate_path, completion["behavior_gate"])
        elif item.get("c9_multi_explorer_scout_evidence") is not None or \
                item.get("c9_behavior_gate_path") is not None:
            raise ValueError(
                "non-C9 MR-ME source carries C9 multi-explorer evidence")
        seen_indexes.add(scout_index)
        seen_paths.add(trajectory_path)
        normalized.append({**item, "trajectory_path": trajectory_path})
    return sorted(normalized, key=lambda item: int(item["scout_index"]))


def execute_round_scouts(
    contract: Mapping[str, Any],
    round_index: int,
    *,
    output_dir: str,
    sync_dir: str,
    run_name: str,
    scout_init: str,
    expected_shard_manifest_sha256: str,
    execute_one: Callable[[Mapping[str, Any]], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Execute/resume every Scout via a pipeline callback, then validate K.

    The callback owns GPU work and persistence; this function owns the
    scientific fan-out/fan-in invariant.  It never calls filtering.  Only
    after all K raw source records validate does it return the ordered union
    inputs for the pipeline's single QualityPool operation.
    """
    plans = scout_execution_plan(
        contract,
        round_index,
        output_dir=output_dir,
        sync_dir=sync_dir,
        run_name=run_name,
        scout_init=scout_init,
    )
    sources: list[dict[str, Any]] = []
    for plan in plans:
        result = execute_one(plan)
        if not isinstance(result, Mapping):
            raise ValueError("MR-ME Scout callback must return a source record")
        sources.append(dict(result))
    return validate_round_scout_sources(
        contract,
        round_index,
        sources,
        expected_shard_manifest_sha256=expected_shard_manifest_sha256,
        expected_scout_init_identity=plans[0]["scout_init_identity"],
    )


def build_trajectory_source_manifest(
    contract: Mapping[str, Any],
    round_index: int,
    sources: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Create the exact local-source manifest consumed and logged by SFT.

    The logical artifact identity is the signed MR-ME source-record digest;
    the durable pipeline results artifact later carries these same files.
    ``common.verify_trajectory_source_manifest`` re-hashes every local byte at
    the filter/SFT boundary.
    """
    validated = validate_contract(contract)
    artifacts = []
    for source in sorted(sources, key=lambda item: int(item["scout_index"])):
        path = os.path.realpath(str(source["trajectory_path"]))
        root = os.path.dirname(path)
        relative = os.path.basename(path)
        artifacts.append(
            {
                "artifact_ref": (
                    f"mrme-local/{validated['contract_sha256']}/"
                    f"r{int(round_index):02d}-s{int(source['scout_index']):02d}:v1"
                ),
                "artifact_digest": str(source["record_sha256"]),
                "download_root": root,
                "files": [
                    {
                        "relative_path": relative,
                        "local_path": path,
                        "sha256": str(source["trajectory_sha256"]),
                    }
                ],
            }
        )
    manifest = {
        "version": common.TRAJECTORY_SOURCE_DIGEST_VERSION,
        "artifacts": artifacts,
    }
    common.verify_trajectory_source_manifest(manifest)
    return manifest


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_c9_multi_explorer_gate_evidence(
    value: Mapping[str, Any], *, contract: Mapping[str, Any],
    round_index: int, scout_index: int,
    expected_runtime_contract: Mapping[str, Any] | None = None,
    expected_code_identity: Mapping[str, Any] | None = None,
    expected_model_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the fixed, correctness-blind post-Scout behavior panel.

    This intentionally does not reuse the SingleScout C9 signature: a K-Scout
    campaign has a different scientific identity and every Scout/round slot is
    contract-bound.  The sampling/threshold constants remain identical.
    """
    from tmx_gpu import c9_evidence, grpo_gpu

    validated = validate_contract(contract)
    if not is_c9_multi_explorer(validated):
        raise ValueError("C9 multi-explorer gate used with a non-C9 contract")
    evidence = _validate_signed_payload(
        value, C9_MULTIEXPLORER_GATE_EVIDENCE_VERSION)
    spec = round_spec(validated, round_index)
    scout = spec["scouts"][int(scout_index) - 1]
    expected_identity = {
        "contract_sha256": validated["contract_sha256"],
        "method": validated["method"],
        "stabilization_profile": C9_MULTIEXPLORER_PROFILE,
        "scientific_variant": C9_MULTIEXPLORER_SCIENTIFIC_VARIANT,
        "round_index": int(round_index),
        "scout_index": int(scout_index),
        "seed": int(scout["seed"]),
        "gate_stage": "scout_final",
        "selection_uses_correctness": False,
        "panel_role": c9_evidence.GATE_PANEL_ROLE,
        "panel_held_out_from_scout_training":
            c9_evidence.GATE_PANEL_HELD_OUT_FROM_SCOUT_TRAINING,
        "passed": True,
        "model_step": 200,
    }
    drift = [
        f"{key}={evidence.get(key)!r} (expected {expected!r})"
        for key, expected in expected_identity.items()
        if evidence.get(key) != expected
    ]
    sampling = {
        "num_prompts": c9_evidence.GATE_NUM_PROMPTS,
        "samples_per_prompt": c9_evidence.GATE_SAMPLES_PER_PROMPT,
        "selection_seed": c9_evidence.GATE_SELECTION_SEED,
        "generation_seed": c9_evidence.GATE_GENERATION_SEED,
        "temperature": c9_evidence.GATE_TEMPERATURE,
        "top_p": c9_evidence.GATE_TOP_P,
        "top_k": c9_evidence.GATE_TOP_K,
        "min_p": c9_evidence.GATE_MIN_P,
        "max_completion_tokens": c9_evidence.GATE_MAX_COMPLETION_TOKENS,
        "paired_seeds_across_prompt_modes": True,
    }
    if evidence.get("sampling") != sampling:
        drift.append("sampling contract mismatch")
    if evidence.get("thresholds") != c9_evidence.GATE_THRESHOLDS:
        drift.append("behavior thresholds mismatch")
    if evidence.get("raw_rollout_count") != (
            2 * c9_evidence.GATE_NUM_PROMPTS
            * c9_evidence.GATE_SAMPLES_PER_PROMPT):
        drift.append("raw rollout geometry mismatch")
    runtime = evidence.get("model_runtime_contract")
    novelty = runtime.get("scout_novelty_contract", {}) \
        if isinstance(runtime, Mapping) else {}
    if not isinstance(runtime, Mapping) or \
            runtime.get("phase") != "scout" or \
            runtime.get("expected_steps") != 200 or \
            runtime.get("scientific_variant") != \
            C9_MULTIEXPLORER_SCIENTIFIC_VARIANT or \
            runtime.get("c9_execution_profile") != "full" or \
            runtime.get("stabilization_cell") != "C5" or \
            novelty.get("rnd_lifecycle") != "c9_selected_batch_update" or \
            novelty.get("novelty_normalization") != "selected_batch_zscore" or \
            not math.isclose(float(runtime.get("lambda_novelty", -1.0)),
                             float(spec["lambda_novelty"])):
        drift.append("gated Scout runtime is not the stabilized C9 slot")
    if isinstance(runtime, Mapping):
        try:
            grpo_gpu.require_canonical_training_execution_environment_contract(
                dict(runtime))
        except ValueError as exc:
            drift.append(f"Scout execution environment: {exc}")
    if expected_runtime_contract is not None and \
            dict(runtime or {}) != dict(expected_runtime_contract):
        drift.append("gated Scout runtime differs from resolved arguments")
    if expected_code_identity is not None and \
            evidence.get("code_artifact") != dict(expected_code_identity):
        drift.append("gate code identity mismatch")
    if expected_model_manifest is not None and \
            evidence.get("model_file_manifest") != dict(expected_model_manifest):
        drift.append("gate model manifest mismatch")
    for field in (
            "heldout_selection_sha256", "heldout_selection_file_sha256",
            "raw_rollouts_file_sha256", "aggregate_file_sha256"):
        if not isinstance(evidence.get(field), str) or len(evidence[field]) != 64:
            drift.append(f"{field} is not a SHA256")
    if drift:
        raise ValueError("C9 multi-explorer behavior gate drift: " + "; ".join(drift))
    return evidence


def validate_c9_multi_explorer_sft_gate_evidence(
    value: Mapping[str, Any], *, contract: Mapping[str, Any],
    round_index: int,
    expected_runtime_contract: Mapping[str, Any] | None = None,
    expected_code_identity: Mapping[str, Any] | None = None,
    expected_model_manifest: Mapping[str, Any] | None = None,
    expected_accepted_digest: Mapping[str, Any] | None = None,
    expected_selection_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate the correctness-blind gate on the one pooled SFT endpoint.

    This is deliberately a separate signed object from every Scout gate.  The
    SFT endpoint belongs to the round/campaign (``scout_index == 0``), is
    initialized from Base, and is the only model that may authorize Central.
    """
    from tmx_gpu import c9_evidence, grpo_gpu

    validated = validate_contract(contract)
    if not is_c9_multi_explorer(validated):
        raise ValueError("C9 pooled-SFT gate used with a non-C9 contract")
    evidence = _validate_signed_payload(
        value, C9_MULTIEXPLORER_SFT_GATE_EVIDENCE_VERSION)
    stage_seed = int(validated["base_seed"]) + int(round_index) - 1
    expected_identity = {
        "contract_sha256": validated["contract_sha256"],
        "method": validated["method"],
        "stabilization_profile": C9_MULTIEXPLORER_PROFILE,
        "scientific_variant": C9_MULTIEXPLORER_SCIENTIFIC_VARIANT,
        "round_index": int(round_index),
        "scout_index": 0,
        "seed": stage_seed,
        "gate_stage": "sft_fixed_endpoint",
        "selection_uses_correctness": False,
        "panel_role": c9_evidence.GATE_PANEL_ROLE,
        "panel_held_out_from_scout_training":
            c9_evidence.GATE_PANEL_HELD_OUT_FROM_SCOUT_TRAINING,
        "passed": True,
        "model_step": 1000,
    }
    drift = [
        f"{key}={evidence.get(key)!r} (expected {expected!r})"
        for key, expected in expected_identity.items()
        if evidence.get(key) != expected
    ]
    sampling = {
        "num_prompts": c9_evidence.GATE_NUM_PROMPTS,
        "samples_per_prompt": c9_evidence.GATE_SAMPLES_PER_PROMPT,
        "selection_seed": c9_evidence.GATE_SELECTION_SEED,
        "generation_seed": c9_evidence.GATE_GENERATION_SEED,
        "temperature": c9_evidence.GATE_TEMPERATURE,
        "top_p": c9_evidence.GATE_TOP_P,
        "top_k": c9_evidence.GATE_TOP_K,
        "min_p": c9_evidence.GATE_MIN_P,
        "max_completion_tokens": c9_evidence.GATE_MAX_COMPLETION_TOKENS,
        "paired_seeds_across_prompt_modes": True,
    }
    if evidence.get("sampling") != sampling:
        drift.append("sampling contract mismatch")
    if evidence.get("thresholds") != c9_evidence.GATE_THRESHOLDS:
        drift.append("behavior thresholds mismatch")
    if evidence.get("raw_rollout_count") != (
            2 * c9_evidence.GATE_NUM_PROMPTS
            * c9_evidence.GATE_SAMPLES_PER_PROMPT):
        drift.append("raw rollout geometry mismatch")
    runtime = evidence.get("model_runtime_contract")
    if not isinstance(runtime, Mapping) or \
            runtime.get("model_init") != validated["model"] or \
            runtime.get("base_model_revision") != validated["model_revision"] or \
            runtime.get("optimizer") != "adamw" or \
            not math.isclose(float(runtime.get("learning_rate", -1.0)), 5e-6) or \
            runtime.get("epochs") != 2 or runtime.get("batch_size") != 1 or \
            runtime.get("grad_accum") != 1 or runtime.get("seed") != stage_seed or \
            runtime.get("termination_mode") != "append_eos" or \
            runtime.get("parameter_dtype") != "float32":
        drift.append("gated SFT runtime is not the fixed Base-init 500x2 endpoint")
    if isinstance(runtime, Mapping):
        try:
            grpo_gpu.require_canonical_training_execution_environment_contract(
                dict(runtime))
        except ValueError as exc:
            drift.append(f"SFT execution environment: {exc}")
    if expected_runtime_contract is not None and \
            dict(runtime or {}) != dict(expected_runtime_contract):
        drift.append("gated SFT runtime differs from resolved arguments")
    accepted = evidence.get("accepted_library_digest")
    if not isinstance(accepted, Mapping) or accepted.get("row_count") != 500 or \
            not isinstance(accepted.get("sha256"), str) or \
            len(str(accepted.get("sha256", ""))) != 64:
        drift.append("accepted library is not the exact pooled 500-row set")
    if expected_accepted_digest is not None and \
            dict(accepted or {}) != dict(expected_accepted_digest):
        drift.append("accepted-library digest mismatch")
    if not isinstance(evidence.get("accepted_file_sha256"), str) or \
            len(evidence["accepted_file_sha256"]) != 64:
        drift.append("accepted_file_sha256 is not a SHA256")
    if expected_code_identity is not None and \
            evidence.get("code_artifact") != dict(expected_code_identity):
        drift.append("gate code identity mismatch")
    if expected_model_manifest is not None and \
            evidence.get("model_file_manifest") != dict(expected_model_manifest):
        drift.append("gate model manifest mismatch")
    if expected_selection_sha256 is not None and \
            evidence.get("heldout_selection_sha256") != \
            str(expected_selection_sha256):
        drift.append("pooled SFT did not use the same fixed diagnostic panel")
    for field in (
            "heldout_selection_sha256", "heldout_selection_file_sha256",
            "raw_rollouts_file_sha256", "aggregate_file_sha256"):
        if not isinstance(evidence.get(field), str) or len(evidence[field]) != 64:
            drift.append(f"{field} is not a SHA256")
    if drift:
        raise ValueError(
            "C9 multi-explorer pooled-SFT behavior gate drift: "
            + "; ".join(drift))
    return evidence


def build_c9_multi_explorer_central_lineage(
    *, contract: Mapping[str, Any], round_index: int,
    shard_manifest_sha256: str,
    trajectory_source_digest: Mapping[str, Any],
    accepted_library_digest: Mapping[str, Any],
    sft_behavior_gate: Mapping[str, Any],
    sft_model_file_manifest: Mapping[str, Any],
    sft_model_artifact: Mapping[str, Any],
    central_runtime_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the exact pooled-SFT-to-Central authorization record."""
    validated = validate_contract(contract)
    payload = {
        "lineage_contract_version":
            C9_MULTIEXPLORER_CENTRAL_LINEAGE_VERSION,
        "contract_sha256": validated["contract_sha256"],
        "method": validated["method"],
        "stabilization_profile": validated["stabilization_profile"],
        "scientific_variant": validated["scientific_variant"],
        "round_index": int(round_index),
        "scouts_per_round": int(validated["scouts_per_round"]),
        "shard_manifest_sha256": str(shard_manifest_sha256),
        "trajectory_source_digest": dict(trajectory_source_digest),
        "accepted_library_digest": dict(accepted_library_digest),
        "sft_behavior_gate": dict(sft_behavior_gate),
        "sft_model_file_manifest": dict(sft_model_file_manifest),
        "sft_model_artifact": dict(sft_model_artifact),
        "central_runtime_contract": dict(central_runtime_contract),
    }
    payload["lineage_sha256"] = canonical_json_sha256(payload)
    validate_c9_multi_explorer_central_lineage(
        payload, contract=validated, round_index=round_index,
        expected_shard_manifest_sha256=shard_manifest_sha256,
        expected_trajectory_source_digest=trajectory_source_digest,
        expected_accepted_digest=accepted_library_digest,
        expected_central_runtime_contract=central_runtime_contract)
    return payload


def validate_c9_multi_explorer_central_lineage(
    value: Mapping[str, Any], *, contract: Mapping[str, Any],
    round_index: int, expected_shard_manifest_sha256: str,
    expected_trajectory_source_digest: Mapping[str, Any] | None = None,
    expected_accepted_digest: Mapping[str, Any] | None = None,
    expected_central_runtime_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the pooled SFT gate and every byte handed to Central."""
    from tmx_gpu import grpo_gpu

    validated = validate_contract(contract)
    if not is_c9_multi_explorer(validated):
        raise ValueError("C9 Central lineage used with a non-C9 contract")
    if not isinstance(value, Mapping):
        raise ValueError("C9 multi-explorer Central lineage must be an object")
    lineage = dict(value)
    unsigned = dict(lineage)
    recorded_sha = unsigned.pop("lineage_sha256", None)
    if recorded_sha != canonical_json_sha256(unsigned):
        raise ValueError("C9 multi-explorer Central lineage SHA256 mismatch")
    expected = {
        "lineage_contract_version":
            C9_MULTIEXPLORER_CENTRAL_LINEAGE_VERSION,
        "contract_sha256": validated["contract_sha256"],
        "method": validated["method"],
        "stabilization_profile": C9_MULTIEXPLORER_PROFILE,
        "scientific_variant": C9_MULTIEXPLORER_SCIENTIFIC_VARIANT,
        "round_index": int(round_index),
        "scouts_per_round": int(validated["scouts_per_round"]),
        "shard_manifest_sha256": str(expected_shard_manifest_sha256),
    }
    drift = [key for key, expected_value in expected.items()
             if lineage.get(key) != expected_value]
    trajectory = lineage.get("trajectory_source_digest")
    if not isinstance(trajectory, Mapping) or \
            trajectory.get("artifact_count") != int(
                validated["scouts_per_round"]) or \
            not isinstance(trajectory.get("sha256"), str):
        drift.append("trajectory_source_digest")
    if expected_trajectory_source_digest is not None and \
            dict(trajectory or {}) != dict(expected_trajectory_source_digest):
        drift.append("expected trajectory source digest")
    accepted = lineage.get("accepted_library_digest")
    if not isinstance(accepted, Mapping) or accepted.get("row_count") != 500:
        drift.append("accepted_library_digest")
    if expected_accepted_digest is not None and \
            dict(accepted or {}) != dict(expected_accepted_digest):
        drift.append("expected accepted-library digest")
    manifest = lineage.get("sft_model_file_manifest")
    artifact = lineage.get("sft_model_artifact")
    if not isinstance(manifest, Mapping) or manifest.get("version") != \
            common.DIRECTORY_FILE_MANIFEST_VERSION:
        drift.append("SFT model manifest")
    if not isinstance(artifact, Mapping):
        drift.append("SFT model artifact")
    else:
        unsigned_artifact = dict(artifact)
        artifact_sha = unsigned_artifact.pop("binding_sha256", None)
        full_manifest = artifact.get("model_file_manifest")
        source = artifact.get("source")
        if artifact_sha != canonical_json_sha256(unsigned_artifact) or \
                artifact.get("version") != "pipeline_sft_model_artifact_v1" or \
                artifact.get("durable_wandb_upload") is not True or \
                artifact.get("artifact_subdir") != "model" or \
                artifact.get("artifact_type") != "sft-model" or \
                not isinstance(artifact.get("artifact_metadata"), Mapping) or \
                artifact.get("policy_file_manifest") != manifest or \
                not isinstance(full_manifest, Mapping) or \
                full_manifest.get("version") != \
                common.DIRECTORY_FILE_MANIFEST_VERSION or \
                not isinstance(full_manifest.get("sha256"), str) or \
                not full_manifest.get("files") or \
                not isinstance(source, Mapping) or \
                not source.get("artifact_digest") or \
                artifact.get("accepted_library_digest") != accepted:
            drift.append("SFT model artifact binding")
        else:
            try:
                common.require_immutable_wandb_artifact_ref(
                    str(source.get("artifact_ref", "")),
                    field="C9 multi-explorer SFT model artifact")
            except (TypeError, ValueError) as exc:
                drift.append(f"SFT model artifact source: {exc}")
    gate = lineage.get("sft_behavior_gate")
    try:
        validate_c9_multi_explorer_sft_gate_evidence(
            gate, contract=validated, round_index=round_index,
            expected_code_identity=validated["code_identity"],
            expected_model_manifest=manifest,
            expected_accepted_digest=accepted)
    except (TypeError, ValueError) as exc:
        drift.append(f"SFT behavior gate: {exc}")
    runtime = lineage.get("central_runtime_contract")
    if not isinstance(runtime, Mapping) or runtime.get("phase") != "central" or \
            runtime.get("round") != int(round_index) or \
            float(runtime.get("lambda_novelty", -1.0)) != 0.0:
        drift.append("Central runtime contract")
    if isinstance(runtime, Mapping):
        try:
            grpo_gpu.require_canonical_training_execution_environment_contract(
                dict(runtime))
        except ValueError as exc:
            drift.append(f"Central execution environment: {exc}")
    if expected_central_runtime_contract is not None and \
            dict(runtime or {}) != dict(expected_central_runtime_contract):
        drift.append("expected Central runtime contract")
    if drift:
        raise ValueError(
            "C9 multi-explorer Central lineage drift: " + "; ".join(drift))
    return lineage


def _expected_trajectory_row_provenance(
    contract: Mapping[str, Any], *, round_index: int, scout_index: int,
    shard_manifest_sha256: str,
) -> dict[str, Any]:
    """Exact per-row identity for one MR-ME Scout campaign slot."""
    validated = validate_contract(contract)
    spec = round_spec(validated, round_index)
    if not 1 <= int(scout_index) <= len(spec["scouts"]):
        raise ValueError("MR-ME trajectory Scout index is outside the round")
    scout = spec["scouts"][int(scout_index) - 1]
    return {
        "mrme_trajectory_provenance_version":
            MRME_TRAJECTORY_PROVENANCE_VERSION,
        "mrme_contract_sha256": validated["contract_sha256"],
        "mrme_method": validated["method"],
        "mrme_stabilization_profile": validated["stabilization_profile"],
        "mrme_round_index": int(round_index),
        "mrme_scout_index": int(scout_index),
        "mrme_scout_seed": int(scout["seed"]),
        "mrme_shard_manifest_sha256": str(shard_manifest_sha256),
    }


def _validate_trajectory_row_provenance(
    row: Mapping[str, Any], *, expected: Mapping[str, Any],
) -> None:
    drift = {
        key: {"actual": row.get(key), "expected": value}
        for key, value in expected.items()
        if row.get(key) != value
    }
    if drift:
        raise ValueError(
            "C9 multi-explorer trajectory campaign-slot provenance drift: "
            f"{drift}")


def validate_c9_behavior_gate_support_files(
    evidence_path: str, evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-hash every support file named by a signed C9 gate.

    The evidence JSON signs the three byte digests.  Publication validation
    must still read those sibling files again: retaining only a copied signed
    JSON object is not evidence that the fixed panel, raw rollouts, and
    aggregate survived the campaign artifact upload unchanged.
    """
    path = os.path.realpath(str(evidence_path))
    if not os.path.isfile(path):
        raise ValueError("C9 behavior-gate evidence file is missing")
    persisted = load_json_object(path)
    if persisted != dict(evidence):
        raise ValueError("C9 behavior-gate signed evidence bytes drifted")
    gate_dir = os.path.dirname(path)
    bindings = {
        "heldout_selection.jsonl": "heldout_selection_file_sha256",
        "rollouts.jsonl": "raw_rollouts_file_sha256",
        "aggregate_metrics.json": "aggregate_file_sha256",
    }
    support: dict[str, Any] = {}
    for name, field in bindings.items():
        support_path = os.path.join(gate_dir, name)
        if not os.path.isfile(support_path):
            raise ValueError(
                f"C9 behavior-gate supporting file is missing: {name}")
        digest = _file_sha256(support_path)
        if digest != evidence.get(field):
            raise ValueError(
                f"C9 behavior-gate supporting file drifted: {name}")
        support[name] = {"path": support_path, "sha256": digest}

    selection_rows: list[dict[str, Any]] = []
    with open(os.path.join(gate_dir, "heldout_selection.jsonl"),
              encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "C9 behavior-gate selection JSON is invalid at line "
                    f"{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError("C9 behavior-gate selection row is not an object")
            selection_rows.append(row)
    if canonical_json_sha256(selection_rows) != evidence.get(
            "heldout_selection_sha256"):
        raise ValueError("C9 behavior-gate selection semantics drifted")

    rollout_rows: list[dict[str, Any]] = []
    with open(os.path.join(gate_dir, "rollouts.jsonl"),
              encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "C9 behavior-gate rollout JSON is invalid at line "
                    f"{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError("C9 behavior-gate rollout row is not an object")
            rollout_rows.append(row)
    rollout_count = len(rollout_rows)
    recorded_rollout_count = evidence.get("raw_rollout_count")
    if type(recorded_rollout_count) is not int or \
            rollout_count != recorded_rollout_count:
        raise ValueError("C9 behavior-gate rollout count drifted")

    aggregate = load_json_object(os.path.join(gate_dir, "aggregate_metrics.json"))
    selection = aggregate.get("selection")
    health_gate = aggregate.get("health_gate")
    if aggregate.get("sampling") != evidence.get("sampling") or \
            not isinstance(selection, Mapping) or selection.get(
                "heldout_selection_sha256") != evidence.get(
                    "heldout_selection_sha256") or \
            not isinstance(health_gate, Mapping) or \
            health_gate.get("passed") is not True:
        raise ValueError("C9 behavior-gate aggregate semantics drifted")
    # A signed aggregate is not sufficient evidence: deterministically rebuild
    # every metric and the pass/fail verdict from the raw rollout rows.  This
    # prevents a copied or re-signed ``passed: true`` aggregate from
    # authorizing SFT/Central when the generations do not satisfy the gate.
    from tmx_gpu import c9_evidence, sft_gate_gpu

    try:
        metrics, verdict = sft_gate_gpu.recompute_gate_from_rollouts(
            rollout_rows,
            num_prompts=c9_evidence.GATE_NUM_PROMPTS,
            samples_per_prompt=c9_evidence.GATE_SAMPLES_PER_PROMPT,
            generation_seed=c9_evidence.GATE_GENERATION_SEED,
            max_completion_tokens=c9_evidence.GATE_MAX_COMPLETION_TOKENS,
            ordered_problem_ids=[str(row["problem_id"])
                                 for row in selection_rows],
            thresholds=dict(c9_evidence.GATE_THRESHOLDS),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "C9 behavior-gate raw-rollout replay failed") from exc
    if metrics != aggregate.get("metrics_by_prompt_mode") or \
            verdict != aggregate.get("health_gate") or \
            verdict.get("passed") is not True:
        raise ValueError(
            "C9 behavior-gate aggregate differs from raw-rollout replay")
    return {
        "evidence_path": path,
        "support_files": support,
        "selection_row_count": len(selection_rows),
        "rollout_row_count": rollout_count,
    }


def _c9_multi_explorer_trajectory_facts(
    path: str, *, contract: Mapping[str, Any], round_index: int,
    scout_index: int, shard_manifest_sha256: str,
) -> dict[str, Any]:
    from tmx_gpu import grpo_gpu

    validated = validate_contract(contract)
    expected_steps = int(validated["scout_steps_per_scout"])
    expected_rows_per_step = (
        int(validated["prompts_per_step"]) * int(validated["num_generations"]))
    expected_provenance = _expected_trajectory_row_provenance(
        validated, round_index=round_index, scout_index=scout_index,
        shard_manifest_sha256=shard_manifest_sha256)
    per_step: dict[int, int] = {}
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid C9 multi-explorer trajectory line {line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError("C9 multi-explorer trajectory row is not an object")
            step = row.get("step")
            if type(step) is not int or not 1 <= step <= expected_steps:
                raise ValueError("C9 multi-explorer trajectory step mismatch")
            per_step[step] = per_step.get(step, 0) + 1
            expected = {
                "trajectory_schema_version": (
                    grpo_gpu.TRAJECTORY_RECORD_SCHEMA_VERSION),
                "expdis_stabilization_mode": "c9_selected_batch_update",
                "scientific_variant": C9_MULTIEXPLORER_SCIENTIFIC_VARIANT,
                "c9_execution_profile": "full",
                "stabilization_cell": "C5",
                "novelty_normalization": "selected_batch_zscore",
                "rnd_update_scope": "final_selected_learner_rows",
            }
            if any(row.get(key) != value for key, value in expected.items()):
                raise ValueError("C9 multi-explorer trajectory identity drift")
            lengths = {
                key: row.get(key) for key in (
                    "completion_token_length",
                    "completion_token_length_true",
                    "completion_token_length_retokenized",
                )
            }
            if any(type(value) is not int or value < 0
                   for value in lengths.values()):
                raise ValueError(
                    "C9 multi-explorer trajectory lacks exact completion "
                    "length accounting")
            capped = lengths["completion_token_length"]
            true = lengths["completion_token_length_true"]
            if capped != min(true, 16384):
                raise ValueError(
                    "C9 multi-explorer trajectory true/capped lengths differ")
            _validate_trajectory_row_provenance(
                row, expected=expected_provenance)
            for field in (
                    "novelty_reward", "blended_reward", "correctness_reward",
                    "novelty_normalization_center",
                    "novelty_normalization_scale",
                    "novelty_normalization_effective_scale"):
                if not math.isfinite(float(row.get(field, math.nan))):
                    raise ValueError(
                        f"C9 multi-explorer trajectory has non-finite {field}")
            if not bool(row.get("is_correct", False)) and not math.isclose(
                    float(row["novelty_reward"]), 0.0, abs_tol=1e-7):
                raise ValueError(
                    "C9 multi-explorer incorrect row has nonzero novelty credit")
            rows.append(row)
    expected_rows = expected_steps * expected_rows_per_step
    if len(rows) != expected_rows or set(per_step) != set(
            range(1, expected_steps + 1)) or any(
                count != expected_rows_per_step for count in per_step.values()):
        raise ValueError(
            "C9 multi-explorer trajectories are not exactly 200 selected "
            "learner batches of 64 rows")
    return {
        "row_count": len(rows),
        "step_count": len(per_step),
        "rows_per_step": expected_rows_per_step,
        "file_sha256": _file_sha256(path),
        "canonical_rows_sha256": canonical_json_sha256(rows),
    }


def _c9_multi_explorer_probe_facts(
    path: str, *, global_step_offset: int, contract: Mapping[str, Any],
    round_index: int, scout_index: int, shard_manifest_sha256: str,
) -> dict[str, Any]:
    from tmx_gpu import c9_evidence

    validated = validate_contract(contract)
    spec = round_spec(validated, round_index)
    scout = spec["scouts"][int(scout_index) - 1]
    expected_steps = [global_step_offset + step for step in range(25, 201, 25)]
    by_step: dict[int, list[dict[str, Any]]] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("phase") == "scout_grpo" and \
                    "eval/probe_protocol" in row:
                step = int(row.get("global_step", -1))
                if step in expected_steps:
                    by_step.setdefault(step, []).append(row)
    if set(by_step) != set(expected_steps) or any(
            len(by_step[step]) != 1 for step in expected_steps):
        raise ValueError("C9 multi-explorer Scout lacks every exact probe boundary")
    exact_rows: list[dict[str, Any]] = []
    for global_step in expected_steps:
        row = by_step[global_step][0]
        local_step = global_step - global_step_offset
        expected = {
            "phase": "scout_grpo",
            "round": int(round_index),
            "lambda": float(spec["lambda_novelty"]),
            "lambda_novelty": float(spec["lambda_novelty"]),
            "num_scouts": int(validated["scouts_per_round"]),
            "rl_validity_mode": "c8",
            "expdis_stabilization_mode": "c9_selected_batch_update",
            "scientific_variant": C9_MULTIEXPLORER_SCIENTIFIC_VARIANT,
            "c9_execution_profile": "full",
            "stabilization_cell": "C5",
            "novelty_normalization": "selected_batch_zscore",
            "mrme_contract_sha256": validated["contract_sha256"],
            "mrme_method": validated["method"],
            "mrme_stabilization_profile": C9_MULTIEXPLORER_PROFILE,
            "mrme_round_index": int(round_index),
            "mrme_scout_index": int(scout_index),
            "mrme_scouts_per_round": int(validated["scouts_per_round"]),
            "mrme_shard_manifest_sha256": str(shard_manifest_sha256),
            "mrme_stage_seed": int(scout["seed"]),
            "eval/probe_error_rows": 0.0,
            "eval/probe_protocol": c9_evidence.C9_PROBE_PROTOCOL_ID,
            "eval/probe_headline_eligible": 0.0,
            "eval/probe_metric_scale": "fraction",
            "eval/probe_sample_seed_rule": c9_evidence.C9_PROBE_CONTRACT[
                "sample_seed_rule"],
            "eval/probe_benchmark_revision": c9_evidence.C9_PROBE_CONTRACT[
                "benchmark_revision"],
            "eval/probe_benchmark_rows_sha256": c9_evidence.C9_PROBE_CONTRACT[
                "benchmark_rows_sha256"],
            "eval/probe_protocol_complete": 1.0,
            "eval/probe_graded_rows": 120.0,
            "eval/probe_expected_rows": 120.0,
            "eval/probe_complete_problem_count": 30.0,
            "eval/probe_expected_problem_count": 30.0,
            "eval/probe_samples_per_problem": 4.0,
            "eval/probe_max_completion_tokens": 32768.0,
            "eval/probe_temperature": 0.6,
            "eval/probe_top_p": 0.95,
            "eval/probe_top_k": 20.0,
            "eval/probe_min_p": 0.0,
            "eval/probe_seed_base": float(
                c9_evidence.C9_PROBE_CONTRACT["seed_base"]),
            "eval/probe_policy_step": local_step,
            "eval/probe_staleness_updates": 0,
            "eval/probe_fresh_weights": 1.0,
        }
        drift = [key for key, expected_value in expected.items()
                 if row.get(key) != expected_value]
        avg_at_4 = float(row.get("eval/AIME24_avg_at_4", math.nan))
        prompt_sha = row.get("eval/probe_prompt_matrix_sha256")
        if drift or not isinstance(prompt_sha, str) or len(prompt_sha) != 64 or \
                not math.isfinite(avg_at_4) or not 0.0 <= avg_at_4 <= 1.0:
            raise ValueError(
                f"C9 multi-explorer probe {global_step} drift: {drift}")
        exact_rows.append({
            "global_step": global_step,
            **{key: row[key] for key in expected},
            "eval/probe_prompt_matrix_sha256": prompt_sha,
            "eval/AIME24_avg_at_4": row["eval/AIME24_avg_at_4"],
        })
    if len({row["eval/probe_prompt_matrix_sha256"] for row in exact_rows}) != 1:
        raise ValueError("C9 multi-explorer Scout probe prompt matrix changed")
    return {
        "probe_protocol_id": c9_evidence.C9_PROBE_PROTOCOL_ID,
        "phase": "scout_grpo",
        "expected_global_steps": expected_steps,
        "row_count": len(exact_rows),
        "prompt_matrix_sha256": exact_rows[0][
            "eval/probe_prompt_matrix_sha256"],
        "canonical_rows_sha256": canonical_json_sha256(exact_rows),
    }


def _c9_multi_explorer_central_probe_facts(
    path: str, *, global_step_offset: int, contract: Mapping[str, Any],
    round_index: int, shard_manifest_sha256: str,
) -> dict[str, Any]:
    """Recompute the four exact correctness-only Central probe boundaries."""
    from tmx_gpu import c9_evidence

    validated = validate_contract(contract)
    expected_steps = [global_step_offset + step for step in range(25, 101, 25)]
    by_step: dict[int, list[dict[str, Any]]] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("phase") == "central_grpo" and \
                    "eval/probe_protocol" in row:
                step = int(row.get("global_step", -1))
                if step in expected_steps:
                    by_step.setdefault(step, []).append(row)
    if set(by_step) != set(expected_steps) or any(
            len(by_step[step]) != 1 for step in expected_steps):
        raise ValueError(
            "C9 multi-explorer Central lacks every exact probe boundary")
    exact_rows: list[dict[str, Any]] = []
    for global_step in expected_steps:
        row = by_step[global_step][0]
        local_step = global_step - global_step_offset
        exact = {
            "global_step": global_step,
            "phase": "central_grpo",
            "round": row.get("round"),
            "lambda": row.get("lambda"),
            "lambda_novelty": row.get("lambda_novelty"),
            "num_scouts": row.get("num_scouts"),
            "rl_validity_mode": row.get("rl_validity_mode"),
            "expdis_stabilization_mode": row.get("expdis_stabilization_mode"),
            "scientific_variant": row.get("scientific_variant"),
            "c9_execution_profile": row.get("c9_execution_profile"),
            "stabilization_cell": row.get("stabilization_cell"),
            "novelty_normalization": row.get("novelty_normalization"),
            "mrme_contract_sha256": row.get("mrme_contract_sha256"),
            "mrme_method": row.get("mrme_method"),
            "mrme_stabilization_profile": row.get(
                "mrme_stabilization_profile"),
            "mrme_round_index": row.get("mrme_round_index"),
            "mrme_scout_index": row.get("mrme_scout_index"),
            "mrme_scouts_per_round": row.get("mrme_scouts_per_round"),
            "mrme_shard_manifest_sha256": row.get(
                "mrme_shard_manifest_sha256"),
            "mrme_stage_seed": row.get("mrme_stage_seed"),
            "error_rows": row.get("eval/probe_error_rows"),
            "protocol": row.get("eval/probe_protocol"),
            "headline_eligible": row.get("eval/probe_headline_eligible"),
            "metric_scale": row.get("eval/probe_metric_scale"),
            "sample_seed_rule": row.get("eval/probe_sample_seed_rule"),
            "benchmark_revision": row.get("eval/probe_benchmark_revision"),
            "benchmark_rows_sha256": row.get(
                "eval/probe_benchmark_rows_sha256"),
            "prompt_matrix_sha256": row.get(
                "eval/probe_prompt_matrix_sha256"),
            "protocol_complete": row.get("eval/probe_protocol_complete"),
            "graded_rows": row.get("eval/probe_graded_rows"),
            "expected_rows": row.get("eval/probe_expected_rows"),
            "complete_problem_count": row.get(
                "eval/probe_complete_problem_count"),
            "expected_problem_count": row.get(
                "eval/probe_expected_problem_count"),
            "samples_per_problem": row.get("eval/probe_samples_per_problem"),
            "max_completion_tokens": row.get(
                "eval/probe_max_completion_tokens"),
            "temperature": row.get("eval/probe_temperature"),
            "top_p": row.get("eval/probe_top_p"),
            "top_k": row.get("eval/probe_top_k"),
            "min_p": row.get("eval/probe_min_p"),
            "seed_base": row.get("eval/probe_seed_base"),
            "policy_step": row.get("eval/probe_policy_step"),
            "staleness_updates": row.get("eval/probe_staleness_updates"),
            "fresh_weights": row.get("eval/probe_fresh_weights"),
            "avg_at_4": row.get("eval/AIME24_avg_at_4"),
        }
        expected = {
            "round": int(round_index),
            "lambda": 0.0,
            "lambda_novelty": 0.0,
            "num_scouts": int(validated["scouts_per_round"]),
            "rl_validity_mode": "c8",
            "expdis_stabilization_mode": "c9_selected_batch_update",
            "scientific_variant": C9_MULTIEXPLORER_SCIENTIFIC_VARIANT,
            "c9_execution_profile": "full",
            "stabilization_cell": "C5",
            "novelty_normalization": "selected_batch_zscore",
            "mrme_contract_sha256": validated["contract_sha256"],
            "mrme_method": validated["method"],
            "mrme_stabilization_profile": C9_MULTIEXPLORER_PROFILE,
            "mrme_round_index": int(round_index),
            "mrme_scout_index": 0,
            "mrme_scouts_per_round": int(validated["scouts_per_round"]),
            "mrme_shard_manifest_sha256": str(shard_manifest_sha256),
            "mrme_stage_seed": int(validated["base_seed"]) + int(
                round_index) - 1,
            "error_rows": 0.0,
            "protocol": c9_evidence.C9_PROBE_PROTOCOL_ID,
            "headline_eligible": 0.0,
            "metric_scale": "fraction",
            "sample_seed_rule": c9_evidence.C9_PROBE_CONTRACT[
                "sample_seed_rule"],
            "benchmark_revision": c9_evidence.C9_PROBE_CONTRACT[
                "benchmark_revision"],
            "benchmark_rows_sha256": c9_evidence.C9_PROBE_CONTRACT[
                "benchmark_rows_sha256"],
            "protocol_complete": 1.0,
            "graded_rows": 120.0,
            "expected_rows": 120.0,
            "complete_problem_count": 30.0,
            "expected_problem_count": 30.0,
            "samples_per_problem": 4.0,
            "max_completion_tokens": 32768.0,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20.0,
            "min_p": 0.0,
            "seed_base": float(c9_evidence.C9_PROBE_CONTRACT["seed_base"]),
            "policy_step": local_step,
            "staleness_updates": 0,
            "fresh_weights": 1.0,
        }
        drift = [key for key, expected_value in expected.items()
                 if exact.get(key) != expected_value]
        avg_at_4 = float(exact.get("avg_at_4", math.nan))
        prompt_sha = exact.get("prompt_matrix_sha256")
        if drift or not isinstance(prompt_sha, str) or len(prompt_sha) != 64 or \
                not math.isfinite(avg_at_4) or not 0.0 <= avg_at_4 <= 1.0:
            raise ValueError(
                f"C9 multi-explorer Central probe {global_step} drift: "
                f"{drift or ['avg_at_4']}")
        exact_rows.append(exact)
    if len({row["prompt_matrix_sha256"] for row in exact_rows}) != 1:
        raise ValueError("C9 multi-explorer Central probe prompt matrix changed")
    return {
        "probe_protocol_id": c9_evidence.C9_PROBE_PROTOCOL_ID,
        "phase": "central_grpo",
        "expected_global_steps": expected_steps,
        "row_count": len(exact_rows),
        "prompt_matrix_sha256": exact_rows[0]["prompt_matrix_sha256"],
        "canonical_rows_sha256": canonical_json_sha256(exact_rows),
    }


def _validate_c9_multi_explorer_scout_probe_replays(
    metrics_path: str, *, contract: Mapping[str, Any], round_index: int,
    sources: Sequence[Mapping[str, Any]], shard_manifest_sha256: str,
) -> None:
    """Replay every Scout probe slot from durable campaign metrics.

    A Scout source record stores a signed summary created at handoff time.  The
    completed-campaign validator must not trust that summary alone: portable
    replay carries the authoritative ``metrics.jsonl`` too, so derive every
    Scout's global-step slot from the immutable MR-ME contract and reproduce
    the summary from those raw rows.
    """
    validated = validate_contract(contract)
    if not is_c9_multi_explorer(validated):
        return
    scouts_per_round = int(validated["scouts_per_round"])
    scout_steps = int(validated["scout_steps_per_scout"])
    central_steps = int(validated["central_steps_per_round"])
    round_offset = (int(round_index) - 1) * (
        scouts_per_round * scout_steps + central_steps)
    for source in sources:
        scout_index = int(source.get("scout_index", 0))
        global_step_offset = round_offset + (scout_index - 1) * scout_steps
        replayed = _c9_multi_explorer_probe_facts(
            metrics_path, global_step_offset=global_step_offset,
            contract=validated, round_index=round_index,
            scout_index=scout_index,
            shard_manifest_sha256=shard_manifest_sha256)
        completion = source.get("c9_multi_explorer_scout_evidence")
        recorded = completion.get("probe_history") \
            if isinstance(completion, Mapping) else None
        if replayed != recorded:
            raise ValueError(
                f"completed C9 Scout {scout_index} probe history differs "
                "from durable metrics replay")


def build_c9_multi_explorer_scout_evidence(
    *, contract: Mapping[str, Any], round_index: int, scout_index: int,
    trajectory_path: str, final_identity: Mapping[str, Any],
    metrics_path: str, global_step_offset: int,
    behavior_gate: Mapping[str, Any], shard_manifest_sha256: str,
) -> dict[str, Any]:
    validated = validate_contract(contract)
    if not is_c9_multi_explorer(validated):
        raise ValueError("C9 Scout evidence requested for a non-C9 campaign")
    spec = round_spec(validated, round_index)
    scout = spec["scouts"][int(scout_index) - 1]
    runtime = final_identity.get("runtime_contract")
    model_manifest = final_identity.get("checkpoint_file_manifest")
    evidence = _signed_payload(
        C9_MULTIEXPLORER_SCOUT_EVIDENCE_VERSION,
        {
            "contract_sha256": validated["contract_sha256"],
            "method": validated["method"],
            "stabilization_profile": C9_MULTIEXPLORER_PROFILE,
            "scientific_variant": C9_MULTIEXPLORER_SCIENTIFIC_VARIANT,
            "round_index": int(round_index),
            "scout_index": int(scout_index),
            "seed": int(scout["seed"]),
            "scout_steps": int(validated["scout_steps_per_scout"]),
            "rnd_selected_batch_update_count": final_identity.get(
                "rnd_selected_batch_update_count"),
            "scout_runtime_contract": runtime,
            "code_artifact": final_identity.get("code_artifact_binding"),
            "model_file_manifest": model_manifest,
            "trajectory": _c9_multi_explorer_trajectory_facts(
                trajectory_path, contract=validated, round_index=round_index,
                scout_index=scout_index,
                shard_manifest_sha256=shard_manifest_sha256),
            "probe_history": _c9_multi_explorer_probe_facts(
                metrics_path, global_step_offset=global_step_offset,
                contract=validated, round_index=round_index,
                scout_index=scout_index,
                shard_manifest_sha256=shard_manifest_sha256),
            "behavior_gate": dict(behavior_gate),
        },
    )
    validate_c9_multi_explorer_scout_evidence(
        evidence, contract=validated, round_index=round_index,
        scout_index=scout_index)
    return evidence


def validate_c9_multi_explorer_scout_evidence(
    value: Mapping[str, Any], *, contract: Mapping[str, Any],
    round_index: int, scout_index: int,
) -> dict[str, Any]:
    validated = validate_contract(contract)
    evidence = _validate_signed_payload(
        value, C9_MULTIEXPLORER_SCOUT_EVIDENCE_VERSION)
    spec = round_spec(validated, round_index)
    scout = spec["scouts"][int(scout_index) - 1]
    exact = {
        "contract_sha256": validated["contract_sha256"],
        "method": validated["method"],
        "stabilization_profile": C9_MULTIEXPLORER_PROFILE,
        "scientific_variant": C9_MULTIEXPLORER_SCIENTIFIC_VARIANT,
        "round_index": int(round_index),
        "scout_index": int(scout_index),
        "seed": int(scout["seed"]),
        "scout_steps": 200,
        "rnd_selected_batch_update_count": 200,
        "code_artifact": validated["code_identity"],
    }
    drift = [key for key, expected in exact.items()
             if evidence.get(key) != expected]
    trajectory = evidence.get("trajectory")
    if not isinstance(trajectory, Mapping) or \
            trajectory.get("row_count") != 12800 or \
            trajectory.get("step_count") != 200 or \
            trajectory.get("rows_per_step") != 64:
        drift.append("trajectory geometry")
    probes = evidence.get("probe_history")
    if not isinstance(probes, Mapping) or \
            probes.get("row_count") != 8 or \
            probes.get("phase") != "scout_grpo" or \
            not isinstance(probes.get("prompt_matrix_sha256"), str) or \
            len(probes["prompt_matrix_sha256"]) != 64:
        drift.append("probe history")
    runtime = evidence.get("scout_runtime_contract")
    novelty = runtime.get("scout_novelty_contract", {}) \
        if isinstance(runtime, Mapping) else {}
    if not isinstance(runtime, Mapping) or \
            runtime.get("phase") != "scout" or \
            runtime.get("expected_steps") != 200 or \
            runtime.get("seed") != int(scout["seed"]) or \
            runtime.get("round") != int(round_index) or \
            runtime.get("scientific_variant") != \
            C9_MULTIEXPLORER_SCIENTIFIC_VARIANT or \
            runtime.get("c9_execution_profile") != "full" or \
            runtime.get("stabilization_cell") != "C5" or \
            novelty.get("rnd_lifecycle") != "c9_selected_batch_update" or \
            novelty.get("novelty_normalization") != "selected_batch_zscore":
        drift.append("runtime C9 lifecycle")
    gate = evidence.get("behavior_gate")
    try:
        validate_c9_multi_explorer_gate_evidence(
            gate, contract=validated, round_index=round_index,
            scout_index=scout_index, expected_runtime_contract=runtime,
            expected_code_identity=validated["code_identity"],
            expected_model_manifest=evidence.get("model_file_manifest"))
    except (TypeError, ValueError) as exc:
        drift.append(f"behavior gate: {exc}")
    if drift:
        raise ValueError(
            "C9 multi-explorer Scout completion drift: " + "; ".join(drift))
    return evidence


def build_scout_source_record(
    *,
    contract: Mapping[str, Any],
    round_index: int,
    scout_index: int,
    trajectory_path: str,
    shard_manifest: Mapping[str, Any],
    scout_init_identity: Mapping[str, Any],
    scout_final: str,
    metrics_path: str | None = None,
    global_step_offset: int | None = None,
    behavior_gate: Mapping[str, Any] | None = None,
    behavior_gate_path: str | None = None,
    scout_final_artifact: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Byte-bound handoff record for one finished raw Scout pool."""
    validated = validate_contract(contract)
    spec = round_spec(validated, round_index)
    expected = spec["scouts"][int(scout_index) - 1]
    if int(expected["scout_index"]) != int(scout_index):
        raise ValueError("MR-ME Scout index lookup mismatch")
    path = os.path.realpath(str(trajectory_path))
    if not os.path.isfile(path):
        raise ValueError(f"Scout trajectory file is missing: {path!r}")
    expected_rows = int(validated["expected_raw_rows_per_scout"])
    expected_per_step = int(validated["prompts_per_step"]) * int(validated["num_generations"])
    expected_steps = int(validated["scout_steps_per_scout"])
    shard = dict(shard_manifest)
    shard_sha = shard.get("manifest_sha256")
    if not shard_sha:
        raise ValueError("Scout source lacks a shard manifest SHA256")
    expected_row_provenance = (
        _expected_trajectory_row_provenance(
            validated, round_index=round_index, scout_index=scout_index,
            shard_manifest_sha256=str(shard_sha))
        if is_c9_multi_explorer(validated) else None
    )
    rows_by_step: dict[int, int] = {}
    raw_rows = 0
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid Scout trajectory JSON at line {line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError("Scout trajectory rows must be JSON objects")
            step = int(row.get("step", row.get("explorer_step", 0)) or 0)
            if step < 1 or step > expected_steps:
                raise ValueError(f"Scout trajectory has out-of-contract step {step}")
            if is_c9_multi_explorer(validated):
                _validate_trajectory_row_provenance(
                    row, expected=expected_row_provenance or {})
            rows_by_step[step] = rows_by_step.get(step, 0) + 1
            raw_rows += 1
    if raw_rows != expected_rows:
        raise ValueError(
            f"Scout trajectory source has {raw_rows} rows; expected exactly "
            f"{expected_rows} ({expected_steps} steps x {expected_per_step})"
        )
    if set(rows_by_step) != set(range(1, expected_steps + 1)) or any(
        count != expected_per_step for count in rows_by_step.values()
    ):
        raise ValueError(
            "Scout trajectory source does not contain exactly one full selected "
            "learner batch for every step"
        )
    final_identity = validate_scout_final(
        scout_final,
        expected_round=int(round_index),
        expected_seed=int(expected["seed"]),
        expected_lambda=float(spec["lambda_novelty"]),
        expected_steps=expected_steps,
    )
    validate_scout_runtime_contract(
        validated,
        final_identity,
        expected_scout_init_identity=scout_init_identity,
        expected_shard_manifest_sha256=str(shard_sha),
        expected_scout_index=int(scout_index),
    )
    if final_identity.get("code_artifact_binding") != validated["code_identity"]:
        raise ValueError(
            "completed MR-ME Scout was not trained by the campaign's exact "
            "immutable code artifact"
        )
    c9_completion: dict[str, Any] | None = None
    gate_path: str | None = None
    if is_c9_multi_explorer(validated):
        if not metrics_path or not os.path.isfile(str(metrics_path)):
            raise ValueError("C9 multi-explorer Scout lacks durable metrics")
        if global_step_offset is None:
            raise ValueError("C9 multi-explorer Scout lacks its global-step slot")
        if not isinstance(behavior_gate, Mapping):
            raise ValueError("C9 multi-explorer Scout lacks a behavior gate")
        gate_path = os.path.realpath(str(behavior_gate_path or ""))
        if not os.path.isfile(gate_path):
            raise ValueError("C9 multi-explorer behavior-gate evidence is missing")
        if load_json_object(gate_path) != dict(behavior_gate):
            raise ValueError("C9 multi-explorer behavior-gate bytes drifted")
        validate_c9_behavior_gate_support_files(gate_path, behavior_gate)
        c9_completion = build_c9_multi_explorer_scout_evidence(
            contract=validated,
            round_index=round_index,
            scout_index=scout_index,
            trajectory_path=path,
            final_identity=final_identity,
            metrics_path=os.path.realpath(str(metrics_path)),
            global_step_offset=int(global_step_offset),
            behavior_gate=behavior_gate,
            shard_manifest_sha256=str(shard_sha),
        )
    record = {
        "version": "gpu_mrme_scout_source_v1",
        "contract_sha256": validated["contract_sha256"],
        "round_index": int(round_index),
        "scout_index": int(scout_index),
        "seed": int(expected["seed"]),
        "lambda_novelty": float(spec["lambda_novelty"]),
        "trajectory_path": path,
        "trajectory_sha256": _file_sha256(path),
        "trajectory_rows": raw_rows,
        "rows_per_step": expected_per_step,
        "shard_manifest_sha256": str(shard_sha),
        "scout_init_identity": dict(scout_init_identity),
        "scout_final": os.path.realpath(str(scout_final)),
        "scout_final_identity": final_identity,
        "scout_final_artifact": (
            dict(scout_final_artifact)
            if scout_final_artifact is not None else None),
        "source_kind": "raw_scout_trajectory_pool",
        "per_scout_preselection": False,
        "c9_multi_explorer_scout_evidence": c9_completion,
        "c9_behavior_gate_path": gate_path,
    }
    return {**record, "record_sha256": canonical_json_sha256(record)}


def model_handoff_identity(
    model: str,
    *,
    expected_phase: str | None = None,
    model_revision: str = "",
) -> dict[str, Any]:
    """Bind a registry model or a completed local model handoff.

    Round 1 commonly uses a registry ref (for example Qwen/Qwen3-1.7B).  Every
    later MR-ME round must use the previous round's local Central, whose model
    bytes and trainer state are verified here.
    """
    value = str(model)
    if not os.path.isdir(value):
        if expected_phase is not None:
            raise ValueError(f"expected a completed local {expected_phase} handoff, got {value!r}")
        if not value.strip():
            raise ValueError("model registry reference is empty")
        revision = str(model_revision or "")
        if not revision:
            raise ValueError("registry model handoff lacks immutable revision")
        core = {
            "kind": "registry_model", "model_ref": value,
            "model_revision": revision,
        }
        return {**core, "identity_sha256": canonical_json_sha256(core)}

    root = os.path.realpath(value)
    if not os.path.isfile(os.path.join(root, "config.json")):
        raise ValueError(f"local model handoff lacks config.json: {root!r}")
    state_path = os.path.join(root, "trainer_state.json")
    if not os.path.isfile(state_path):
        if expected_phase is not None:
            raise ValueError(f"completed local {expected_phase} handoff lacks trainer_state.json")
        # Local Base fixtures/checkpoints without trainer state are still
        # byte-bound as a directory, but may only initialize round 1.
        manifest = common.directory_file_manifest(root)
        core = {
            "kind": "local_base_model",
            "path": root,
            "directory_file_manifest": manifest,
        }
        return {**core, "identity_sha256": canonical_json_sha256(core)}
    state = load_json_object(state_path)
    if expected_phase is not None and state.get("phase") != expected_phase:
        raise ValueError(f"model handoff phase={state.get('phase')!r}, expected {expected_phase!r}")
    manifest = state.get("checkpoint_file_manifest")
    if not isinstance(manifest, dict):
        raise ValueError("model handoff trainer_state lacks checkpoint_file_manifest")
    common.verify_directory_file_manifest(
        root, manifest, exclude_relative_paths=("trainer_state.json",)
    )
    core = {
        "kind": "completed_model",
        "path": root,
        "phase": state.get("phase"),
        "round": state.get("round"),
        "step": state.get("step"),
        "runtime_contract": state.get("runtime_contract"),
        "code_artifact_binding": state.get("code_artifact_binding"),
        "mrme_provenance": state.get("mrme_provenance"),
        "rnd_selected_batch_update_count": state.get(
            "rnd_selected_batch_update_count"),
        "external_lineage_sha256": state.get("external_lineage_sha256"),
        "checkpoint_file_manifest": manifest,
        "trainer_state_sha256": common.file_sha256(state_path),
    }
    return {**core, "identity_sha256": canonical_json_sha256(core)}


def validate_scout_final(
    model_dir: str,
    *,
    expected_round: int,
    expected_seed: int,
    expected_lambda: float,
    expected_steps: int,
) -> dict[str, Any]:
    """Verify one completed Scout model before its trajectories can pool."""
    root = os.path.realpath(str(model_dir))
    identity = model_handoff_identity(root, expected_phase="scout")
    state = load_json_object(os.path.join(root, "trainer_state.json"))
    expected = {
        "round": int(expected_round),
        "step": int(expected_steps),
    }
    drift = [
        f"{name}={state.get(name)!r} (expected {value!r})"
        for name, value in expected.items()
        if state.get(name) != value
    ]
    if float(state.get("lambda", -1.0)) != float(expected_lambda):
        drift.append(f"lambda={state.get('lambda')!r} (expected {float(expected_lambda)!r})")
    runtime = state.get("runtime_contract")
    if not isinstance(runtime, dict):
        drift.append("runtime_contract is missing")
    else:
        if int(runtime.get("seed", -1)) != int(expected_seed):
            drift.append(f"runtime seed={runtime.get('seed')!r} (expected {int(expected_seed)!r})")
        if int(runtime.get("round", -1)) != int(expected_round):
            drift.append(
                f"runtime round={runtime.get('round')!r} (expected {int(expected_round)!r})"
            )
        if float(runtime.get("lambda_novelty", -1.0)) != float(expected_lambda):
            drift.append(
                f"runtime lambda={runtime.get('lambda_novelty')!r} "
                f"(expected {float(expected_lambda)!r})"
            )
    if drift:
        raise ValueError("completed MR-ME Scout mismatch: " + "; ".join(drift))
    return identity


def validate_scout_runtime_contract(
    contract: Mapping[str, Any],
    final_identity: Mapping[str, Any],
    *,
    expected_scout_init_identity: Mapping[str, Any],
    expected_shard_manifest_sha256: str,
    expected_scout_index: int,
) -> None:
    """Require the completed Scout to match the campaign training template."""
    training = contract.get("training_contract")
    expected_template = (
        training.get("scout_grpo") if isinstance(training, Mapping) else None
    )
    # Lightweight contract fixtures exercise source arithmetic independently.
    # Production contracts always contain this template, and the publication
    # validator below requires it rather than accepting this compatibility path.
    if expected_template is None:
        return
    runtime_value = final_identity.get("runtime_contract")
    if not isinstance(runtime_value, Mapping):
        raise ValueError("completed MR-ME Scout lacks a runtime contract")
    runtime = dict(runtime_value)
    runtime_multi = runtime.pop("multi_explorer_contract", None)
    model_init = str(runtime.get("model_init", ""))
    init_identity = dict(expected_scout_init_identity)
    if init_identity.get("kind") == "registry_model":
        correct_init = model_init == str(init_identity.get("model_ref", ""))
    else:
        correct_init = os.path.realpath(model_init) == os.path.realpath(
            str(init_identity.get("path", ""))
        )
    if not correct_init:
        raise ValueError("completed MR-ME Scout model-init handoff mismatch")
    for field in ("model_init", "seed", "round", "lambda_novelty"):
        runtime.pop(field, None)
    if runtime != dict(expected_template):
        raise ValueError("completed MR-ME Scout training contract drifted")
    expected_multi = {
        "contract_sha256": contract["contract_sha256"],
        "method": contract["method"],
        "stabilization_profile": contract["stabilization_profile"],
        "scout_index": int(expected_scout_index),
        "scouts_per_round": int(contract["scouts_per_round"]),
        "shard_manifest_sha256": str(expected_shard_manifest_sha256),
    }
    if is_single_scout_multiround(contract):
        expected_multi.update({
            "method_version": contract["method_version"],
            "single_scout_multiround": True,
            "mrme": False,
            "multi_explorer": False,
        })
    if runtime_multi != expected_multi:
        raise ValueError("completed Scout runtime multi-explorer binding drifted")
    expected_provenance = {
        "method": contract["method"],
        "stabilization_profile": contract["stabilization_profile"],
        "scientific_variant": contract["scientific_variant"],
        "mrme_contract_sha256": contract["contract_sha256"],
        "mrme_shard_manifest_sha256": str(expected_shard_manifest_sha256),
        "mrme_scout_index": int(expected_scout_index),
        "mrme_scouts_per_round": int(contract["scouts_per_round"]),
        "mrme_scout_seed": int(runtime_value["seed"]),
    }
    if is_single_scout_multiround(contract):
        expected_provenance.update({
            "method_version": contract["method_version"],
            "single_scout_multiround": True,
            "mrme": False,
            "multi_explorer": False,
        })
    provenance = final_identity.get("mrme_provenance")
    if not isinstance(provenance, Mapping) or dict(provenance) != expected_provenance:
        raise ValueError("completed MR-ME Scout artifact provenance mismatch")


def load_json_object(path: str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path!r}")
    return value


def assert_round_manifests_disjoint(
    manifests: Iterable[Mapping[str, Any]],
    *,
    expected_rounds: int,
    expected_method: str | None = None,
) -> dict[str, Any]:
    """Verify exact 1/R partition evidence after all round manifests exist."""
    method = str(expected_method or (
        INTEGRATED_MULTISCOUT_METHOD if int(expected_rounds) == 1
        else DISJOINT_SHARD_MRME_METHOD))
    allowed_methods = {
        INTEGRATED_MULTISCOUT_METHOD,
        DISJOINT_SHARD_MRME_METHOD,
        DISJOINT_SHARD_SINGLESCOUT_MULTIROUND_METHOD,
    }
    if method not in allowed_methods:
        raise ValueError("round shard partition has an unrecognized method")
    if method == INTEGRATED_MULTISCOUT_METHOD and int(expected_rounds) != 1:
        raise ValueError("integrated MultiScout partition requires R=1")
    if method == DISJOINT_SHARD_MRME_METHOD and int(expected_rounds) <= 1:
        raise ValueError("disjoint-shard MR-ME partition requires R>1")
    if method == DISJOINT_SHARD_SINGLESCOUT_MULTIROUND_METHOD and not (
            1 <= int(expected_rounds) <= 4):
        raise ValueError("single-Scout multi-round partition requires 1 <= R <= 4")
    items = sorted((dict(item) for item in manifests), key=lambda x: int(x["shard_index"]))
    if len(items) != int(expected_rounds):
        raise ValueError(f"need {int(expected_rounds)} shard manifests, got {len(items)}")
    expected_indices = list(range(int(expected_rounds)))
    if [int(item.get("shard_index", -1)) for item in items] != expected_indices:
        raise ValueError("shard manifests do not cover every round exactly once")
    full_sha = {item.get("full_dataset_sha256") for item in items}
    total_rows = {int(item.get("total_rows", -1)) for item in items}
    shard_seeds = {int(item.get("shard_seed", -1)) for item in items}
    if len(full_sha) != 1 or None in full_sha or len(total_rows) != 1 or len(shard_seeds) != 1:
        raise ValueError("round shard manifests do not bind one identical full dataset")
    seen: set[str] = set()
    counts: list[int] = []
    for item in items:
        if int(item.get("shard_count", -1)) != int(expected_rounds):
            raise ValueError("round shard_count mismatch")
        expected_required = int(expected_rounds) > 1
        if item.get("shard_policy") != MRME_SHARD_POLICY or item.get(
                "required") is not expected_required:
            raise ValueError(
                "multi-explorer round manifest has the wrong sharding mode")
        if int(item.get("fraction_numerator", -1)) != 1 or int(
            item.get("fraction_denominator", -1)
        ) != int(expected_rounds):
            raise ValueError("MR-ME round manifest does not declare 1/R membership")
        total_for_item = int(item.get("total_rows", -1))
        shard_index = int(item["shard_index"])
        if int(item.get("start", -1)) != (total_for_item * shard_index) // int(
            expected_rounds
        ) or int(item.get("end", -1)) != (total_for_item * (shard_index + 1)) // int(
            expected_rounds
        ):
            raise ValueError("MR-ME shard does not use exact floor boundaries")
        members = item.get("members")
        if not isinstance(members, list):
            raise ValueError("round shard manifest lacks members")
        identities = {f"{int(member['source_index'])}:{member['row_sha256']}" for member in members}
        if len(identities) != len(members):
            raise ValueError("duplicate member within a round shard")
        overlap = seen.intersection(identities)
        if overlap:
            raise ValueError("round data shards overlap")
        seen.update(identities)
        counts.append(len(members))
    total = total_rows.pop()
    if len(seen) != total or sum(counts) != total:
        raise ValueError("round shard union does not reconstruct the full capped dataset")
    if max(counts) - min(counts) > 1:
        raise ValueError("round shards are not an exact floor/ceiling 1/R partition")
    summary = {
        "version": "gpu_multi_explorer_dataset_membership_v2",
        "method": method,
        "dataset_membership_mode": (
            "full_unsharded"
            if method == INTEGRATED_MULTISCOUT_METHOD else
            "deterministic_disjoint_one_over_r"),
        "rounds": int(expected_rounds),
        "total_rows": total,
        "counts": counts,
        "full_dataset_sha256": next(iter(full_sha)),
        "union_member_sha256": canonical_json_sha256(sorted(seen)),
    }
    return {**summary, "partition_sha256": canonical_json_sha256(summary)}


def replay_coverage_pool_c8(
    source_paths: Sequence[str], *, accepted_path: str,
    recorded_funnel: Mapping[str, Any], max_examples: int,
    max_completion_tokens: int,
) -> dict[str, Any]:
    """Re-run the exact MR-ME QualityPool selector and funnel arithmetic.

    Merely checking that every accepted row is valid permits an arbitrary
    500-row subset of the clean pool.  The scientific contract is stronger:
    it fixes the reward-independent per-problem winner, global deterministic
    order, and serialized SFT library.  This helper reproduces
    :func:`filter_pool.pool_trajectories` from all K raw inputs and compares
    both the rows (including order) and every funnel counter.
    """
    from tmx_gpu import filter_pool

    paths = [os.path.realpath(str(path)) for path in source_paths]
    if len(paths) != len(set(paths)) or any(
            not os.path.isfile(path) for path in paths):
        raise ValueError("MR-ME QualityPool sources are missing or duplicated")
    output = os.path.realpath(str(accepted_path))
    if not os.path.isfile(output):
        raise ValueError("MR-ME QualityPool accepted library is missing")
    rows = filter_pool.load_trajectories(paths)
    correct = [row for row in rows if bool(row.get("is_correct", False))]
    clean = [row for row in correct
             if filter_pool.passes_c8_contract_gates(row)]
    selected = filter_pool.select_coverage_pool_c8(
        clean, max_examples=int(max_examples), source_balance=False)
    expected_rows = [
        {
            "prompt_text": row.get("prompt_text", ""),
            "completion_text": row.get("completion_text", ""),
            **{key: row[key] for key in filter_pool.METADATA_KEYS
               if key in row},
            "source_file": row.get("source_file", ""),
            "selection_policy": MRME_POOLING_POLICY,
            "selection_source_balance": False,
        }
        for row in selected
    ]
    accepted: list[dict[str, Any]] = []
    with open(output, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid accepted-library JSON at line {line_number}") \
                    from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"accepted-library row {line_number} is not an object")
            accepted.append(row)
    if accepted != expected_rows:
        raise ValueError(
            "completed campaign accepted library differs from deterministic "
            "coverage_pool_c8 replay")

    expected_funnel = {
        "policy": MRME_POOLING_POLICY,
        "num_sources": len(set(paths)),
        "raw": len(rows),
        "correct": len(correct),
        "clean_correct": len(clean),
        "accepted": len(selected),
        "max_examples": int(max_examples),
        "max_completion_tokens": int(max_completion_tokens),
        # The recorded path is a logical campaign identity.  Compare it after
        # normalization but preserve its spelling in the exact dict check so
        # portable replay at the original root remains byte-compatible.
        "output": recorded_funnel.get("output"),
        "unique_clean_problems": len({
            filter_pool.c8_problem_key(row) for row in clean}),
        "accepted_unique_problems": len({
            filter_pool.c8_problem_key(row) for row in selected}),
        "accepted_per_source": dict(sorted(Counter(
            str(row.get("source_file", "") or "")
            for row in selected).items())),
        "source_balance": False,
    }
    if os.path.realpath(str(recorded_funnel.get("output", ""))) != output:
        raise ValueError("completed campaign QualityPool output path drifted")
    if dict(recorded_funnel) != expected_funnel:
        raise ValueError(
            "completed campaign QualityPool funnel differs from exact replay")
    return {
        "accepted_rows": accepted,
        "funnel": expected_funnel,
    }


def validate_completed_campaign(output_dir: str) -> dict[str, Any]:
    """Validate a complete MR-ME campaign before publishing its artifacts.

    Runtime validation prevents a bad handoff from starting.  This second,
    read-only pass protects the durable scientific claim: it re-hashes every
    raw Scout source and model handoff, verifies the one pooled SFT provenance
    per round, and proves that the round shards form the declared partition.
    """
    from tmx_gpu import c9_evidence, filter_pool
    from tmx_jax.data import validate_round_shard_manifest

    root = os.path.realpath(str(output_dir))
    contract = validate_contract(load_json_object(os.path.join(root, "mrme_contract.json")))
    training_contract = contract.get("training_contract")
    required_training_templates = {"scout_grpo", "central_grpo", "central_sft"}
    if not isinstance(training_contract, dict) or not required_training_templates.issubset(
        training_contract
    ):
        raise ValueError("completed campaign lacks resolved training templates")
    state = load_json_object(os.path.join(root, "pipeline_state.json"))
    summary = load_json_object(os.path.join(root, "run_summary.json"))
    expected_binding = {
        "version": MRME_STATE_BINDING_VERSION,
        "contract": contract,
        "contract_sha256": contract["contract_sha256"],
    }
    if state.get("mrme_binding") != expected_binding:
        raise ValueError("completed campaign pipeline-state binding mismatch")
    if summary.get("mrme_contract_sha256") != contract["contract_sha256"]:
        raise ValueError("completed campaign run-summary contract mismatch")
    single_scout_multiround = is_single_scout_multiround(contract)
    expected_summary_identity = {
        "method": contract["method"],
        "mrme": not single_scout_multiround,
        "multi_explorer": not single_scout_multiround,
        "single_scout_multiround": single_scout_multiround,
    }
    if single_scout_multiround:
        expected_summary_identity["method_version"] = contract[
            "method_version"]
    if any(summary.get(key) != value
           for key, value in expected_summary_identity.items()):
        raise ValueError("completed campaign run-summary method identity drifted")

    round_summaries = summary.get("rounds")
    if not isinstance(round_summaries, list) or len(round_summaries) != int(
        contract["rounds"]
    ):
        raise ValueError("completed campaign run summary does not cover every round")

    shard_manifests: list[dict[str, Any]] = []
    previous_central: str | None = None
    source_index_digests: list[str] = []
    pooled_sft_gate_digests: list[str] = []
    central_lineage_digests: list[str] = []
    central_probe_digests: list[str] = []
    campaign_probe_histories: list[Mapping[str, Any]] = []
    stages = state.get("stages")
    if not isinstance(stages, dict):
        raise ValueError("completed campaign pipeline state lacks stages")
    expected_stage_keys = set(expected_scout_stage_keys(contract))
    downstream_stages = ["harvest", "filter", "sft", "central"]
    if is_c9_multi_explorer(contract):
        downstream_stages.append("sft_behavior_gate")
    expected_stage_keys.update(
        f"r{round_index:02d}/{stage}"
        for round_index in range(1, int(contract["rounds"]) + 1)
        for stage in downstream_stages
    )
    if set(stages) != expected_stage_keys:
        raise ValueError(
            "completed campaign stage set mismatch: missing="
            f"{sorted(expected_stage_keys - set(stages))}, unexpected="
            f"{sorted(set(stages) - expected_stage_keys)}"
        )

    for round_index in range(1, int(contract["rounds"]) + 1):
        round_dir = os.path.join(root, f"round_{round_index:02d}")
        spec = round_spec(contract, round_index)
        required_stages = {
            *(str(item["stage_key"]) for item in spec["scouts"]),
            f"r{round_index:02d}/harvest",
            f"r{round_index:02d}/filter",
            f"r{round_index:02d}/sft",
            f"r{round_index:02d}/central",
        }
        if is_c9_multi_explorer(contract):
            required_stages.add(f"r{round_index:02d}/sft_behavior_gate")
        missing = sorted(required_stages - set(stages))
        if missing:
            raise ValueError(
                f"completed campaign round {round_index} lacks stages: {missing}"
            )

        shard = validate_round_shard_manifest(
            load_json_object(os.path.join(round_dir, "dataset_shard_manifest.json"))
        )
        if int(shard.get("shard_index", -1)) != round_index - 1 or int(
            shard.get("shard_count", -1)
        ) != int(contract["rounds"]):
            raise ValueError("completed campaign round shard identity mismatch")
        shard_manifests.append(shard)

        source_index = load_json_object(
            os.path.join(round_dir, "mrme_scout_sources.json")
        )
        unsigned_source_index = dict(source_index)
        source_index_sha = unsigned_source_index.pop("source_index_sha256", None)
        if source_index_sha != common.canonical_json_sha256(unsigned_source_index):
            raise ValueError("completed campaign Scout-source index SHA mismatch")
        if source_index.get("contract_sha256") != contract["contract_sha256"] or int(
            source_index.get("round_index", 0)
        ) != round_index:
            raise ValueError("completed campaign Scout-source index identity mismatch")
        if source_index.get("shard_manifest_sha256") != shard["manifest_sha256"]:
            raise ValueError("completed campaign Scout sources use the wrong shard")

        scout_init = contract["model"] if round_index == 1 else previous_central
        if not scout_init:
            raise ValueError("completed campaign lacks the previous Central handoff")
        init_identity = model_handoff_identity(
            str(scout_init),
            expected_phase="central" if round_index > 1 else None,
            model_revision=(
                str(contract["model_revision"]) if round_index == 1 else ""),
        )
        sources = validate_round_scout_sources(
            contract,
            round_index,
            source_index.get("sources", []),
            expected_shard_manifest_sha256=str(shard["manifest_sha256"]),
            expected_scout_init_identity=init_identity,
        )
        _validate_c9_multi_explorer_scout_probe_replays(
            os.path.join(root, "metrics.jsonl"), contract=contract,
            round_index=round_index, sources=sources,
            shard_manifest_sha256=str(shard["manifest_sha256"]))
        if is_c9_multi_explorer(contract):
            campaign_probe_histories.extend(
                source["c9_multi_explorer_scout_evidence"]["probe_history"]
                for source in sources)
        source_index_digests.append(str(source_index_sha))
        for source in sources:
            stage = stages[str(spec["scouts"][int(source["scout_index"]) - 1]["stage_key"])]
            source_record_path = os.path.realpath(
                str(stage.get("source_record_path", ""))
            )
            if (
                os.path.realpath(str(stage.get("final", "")))
                != os.path.realpath(str(source["scout_final"]))
                or stage.get("source_record_sha256") != source["record_sha256"]
                or load_json_object(source_record_path) != source
            ):
                raise ValueError("completed campaign Scout stage/source mismatch")

        source_paths = [source["trajectory_path"] for source in sources]
        harvest_stage = stages[f"r{round_index:02d}/harvest"]
        if list(harvest_stage.get("traj_paths", [])) != source_paths:
            raise ValueError("completed campaign harvest/source paths mismatch")

        trajectory_manifest = load_json_object(
            os.path.join(round_dir, "mrme_trajectory_source_manifest.json")
        )
        common.verify_trajectory_source_manifest(trajectory_manifest)
        if trajectory_manifest != build_trajectory_source_manifest(
            contract, round_index, sources
        ):
            raise ValueError("completed campaign pooled trajectory manifest drifted")
        trajectory_digest = common.trajectory_source_manifest_digest(
            trajectory_manifest
        )

        filter_stage = stages[f"r{round_index:02d}/filter"]
        funnel = filter_stage.get("funnel")
        expected_raw = int(contract["scouts_per_round"]) * int(
            contract["expected_raw_rows_per_scout"]
        )
        if not isinstance(funnel, dict) or int(funnel.get("raw", -1)) != expected_raw:
            raise ValueError("completed campaign filter did not pool all K raw Scouts")
        if (
            funnel.get("policy") != MRME_POOLING_POLICY
            or int(funnel.get("num_sources", -1))
            != int(contract["scouts_per_round"])
            or int(funnel.get("max_examples", -1)) != int(contract["max_accepted"])
            or funnel.get("source_balance") is not False
            or int(funnel.get("accepted_unique_problems", -1))
            != int(contract["max_accepted"])
            or int(funnel.get("accepted", -1)) != int(contract["max_accepted"])
        ):
            raise ValueError("completed campaign QualityPool cap was not met")
        accepted_path = os.path.realpath(str(filter_stage.get("accepted", "")))
        if accepted_path != os.path.realpath(os.path.join(round_dir, "accepted.jsonl")):
            raise ValueError("completed campaign accepted-library path mismatch")
        scout_budget = int(training_contract["scout_grpo"].get(
            "completion_budget_flag", 0) or 0)
        expected_filter_max_tokens = (
            scout_budget - 384 if scout_budget > 0
            else filter_pool.MAX_COMPLETION_TOKENS)
        replay_coverage_pool_c8(
            source_paths, accepted_path=accepted_path,
            recorded_funnel=funnel,
            max_examples=int(contract["max_accepted"]),
            max_completion_tokens=expected_filter_max_tokens)
        accepted_digest = common.semantic_sft_jsonl_digest(accepted_path)
        if filter_stage.get("accepted_library_digest") != accepted_digest:
            raise ValueError("completed campaign filter accepted digest mismatch")

        sft_stage = stages[f"r{round_index:02d}/sft"]
        sft_final = os.path.realpath(str(sft_stage.get("final", "")))
        sft_provenance = load_json_object(
            os.path.join(sft_final, "sft_provenance.json")
        )
        if sft_provenance.get("trajectory_source", {}).get("digest") != trajectory_digest:
            raise ValueError("completed campaign SFT is not bound to all K Scouts")
        if sft_provenance.get("trajectory_source", {}).get("manifest") != \
                trajectory_manifest:
            raise ValueError("completed campaign SFT trajectory manifest mismatch")
        if sft_provenance.get("accepted_library_digest") != accepted_digest:
            raise ValueError("completed campaign SFT accepted-library digest mismatch")
        if str(sft_provenance.get("model_init", "")) != str(contract["model"]):
            raise ValueError("completed campaign SFT did not initialize from Base")
        if is_c9_multi_explorer(contract) and (
                int(sft_provenance.get("accepted_examples", -1)) != 500
                or int(sft_provenance.get("epochs", -1)) != 2
                or int(sft_provenance.get("optimizer_steps", -1)) != 1000):
            raise ValueError(
                "C9 multi-explorer requires one Base-init SFT with exactly "
                "500 examples x 2 epochs = 1,000 optimizer updates")
        sft_runtime_value = sft_provenance.get("sft_runtime_contract")
        if not isinstance(sft_runtime_value, dict):
            raise ValueError("completed campaign SFT lacks a runtime contract")
        sft_runtime = dict(sft_runtime_value)
        if int(sft_runtime.pop("seed", -1)) != int(contract["base_seed"]) + round_index - 1:
            raise ValueError("completed campaign SFT seed mismatch")
        if sft_runtime != dict(training_contract["central_sft"]):
            raise ValueError("completed campaign SFT training contract drifted")
        sft_manifest = sft_provenance.get("checkpoint_file_manifest")
        if not isinstance(sft_manifest, dict):
            raise ValueError("completed campaign SFT lacks a model-file manifest")
        common.verify_directory_file_manifest(
            sft_final,
            sft_manifest,
            exclude_relative_paths=("sft_provenance.json",),
        )
        sft_artifact = sft_stage.get("sft_model_artifact")
        if is_c9_multi_explorer(contract):
            if not isinstance(sft_artifact, dict):
                raise ValueError(
                    "completed C9 pooled SFT lacks its durable artifact binding")
            unsigned_artifact = dict(sft_artifact)
            artifact_sha = unsigned_artifact.pop("binding_sha256", None)
            if artifact_sha != common.canonical_json_sha256(unsigned_artifact):
                raise ValueError("completed C9 pooled SFT artifact SHA mismatch")
            if sft_artifact.get("version") != \
                    "pipeline_sft_model_artifact_v1" or \
                    sft_artifact.get("durable_wandb_upload") is not True or \
                    sft_artifact.get("artifact_subdir") != "model" or \
                    sft_artifact.get("artifact_type") != "sft-model" or \
                    not isinstance(sft_artifact.get(
                        "artifact_metadata"), Mapping) or \
                    sft_artifact.get("policy_file_manifest") != sft_manifest or \
                    sft_artifact.get("accepted_library_digest") != accepted_digest or \
                    sft_artifact.get("round") != round_index or \
                    sft_artifact.get("stage_mode") != \
                    "c9_multi_explorer_fixed_endpoint_trained":
                raise ValueError(
                    "completed C9 pooled SFT artifact binding drifted")
            expected_sft_campaign_identity = {
                "method": contract["method"],
                "mrme": not single_scout_multiround,
                "multi_explorer": not single_scout_multiround,
                "single_scout_multiround": single_scout_multiround,
            }
            if single_scout_multiround:
                expected_sft_campaign_identity["method_version"] = contract[
                    "method_version"]
            artifact_metadata = sft_artifact["artifact_metadata"]
            if any(artifact_metadata.get(key) != value for key, value in
                   expected_sft_campaign_identity.items()):
                raise ValueError(
                    "completed pooled SFT artifact method identity drifted")
            source = sft_artifact.get("source")
            if not isinstance(source, dict) or not source.get("artifact_digest"):
                raise ValueError(
                    "completed C9 pooled SFT artifact lacks a durable source")
            common.require_immutable_wandb_artifact_ref(
                str(source.get("artifact_ref", "")),
                field="completed C9 multi-explorer SFT model artifact")
            common.verify_directory_file_manifest(
                sft_final, sft_artifact.get("model_file_manifest"))

        pooled_sft_gate: dict[str, Any] | None = None
        pooled_central_lineage: dict[str, Any] | None = None
        if is_c9_multi_explorer(contract):
            gate_stage = stages[f"r{round_index:02d}/sft_behavior_gate"]
            expected_gate_path = os.path.realpath(os.path.join(
                round_dir, "c9_behavior_gates", "sft_fixed_endpoint",
                "evidence.json"))
            gate_path = os.path.realpath(str(
                gate_stage.get("evidence_path", "")))
            if gate_path != expected_gate_path:
                raise ValueError(
                    "completed pooled-SFT gate evidence path mismatch")
            pooled_sft_gate = load_json_object(gate_path)
            validate_c9_behavior_gate_support_files(
                gate_path, pooled_sft_gate)
            source_panels = {
                str(source.get("c9_multi_explorer_scout_evidence", {}).get(
                    "behavior_gate", {}).get(
                        "heldout_selection_sha256", ""))
                for source in sources
            }
            if len(source_panels) != 1 or "" in source_panels:
                raise ValueError(
                    "completed C9 Scouts do not share one diagnostic panel")
            validate_c9_multi_explorer_sft_gate_evidence(
                pooled_sft_gate, contract=contract,
                round_index=round_index,
                expected_runtime_contract=sft_runtime_value,
                expected_code_identity=contract["code_identity"],
                expected_model_manifest=sft_manifest,
                expected_accepted_digest=accepted_digest,
                expected_selection_sha256=next(iter(source_panels)))
            if gate_stage.get("evidence_sha256") != pooled_sft_gate[
                    "evidence_sha256"] or gate_stage.get(
                        "contract_sha256") != contract["contract_sha256"] or \
                    int(gate_stage.get("round_index", 0)) != round_index or \
                    gate_stage.get("model_file_manifest") != sft_manifest or \
                    gate_stage.get("accepted_library_digest") != accepted_digest or \
                    gate_stage.get("trajectory_source_digest") != \
                    trajectory_digest:
                raise ValueError(
                    "completed pooled-SFT gate pipeline-state binding mismatch")
            pooled_sft_gate_digests.append(
                str(pooled_sft_gate["evidence_sha256"]))
            expected_lineage_path = os.path.realpath(os.path.join(
                round_dir, "c9_multi_explorer_central_lineage.json"))
            lineage_path = os.path.realpath(str(
                gate_stage.get("lineage_path", "")))
            if lineage_path != expected_lineage_path:
                raise ValueError(
                    "completed C9 Central-lineage path mismatch")
            pooled_central_lineage = load_json_object(lineage_path)

        central_stage = stages[f"r{round_index:02d}/central"]
        central_final = os.path.realpath(str(central_stage.get("final", "")))
        central_identity = model_handoff_identity(central_final, expected_phase="central")
        central_state = load_json_object(
            os.path.join(central_final, "trainer_state.json")
        )
        runtime = central_state.get("runtime_contract")
        if (
            int(central_state.get("round", -1)) != round_index
            or int(central_state.get("step", -1))
            != int(contract["central_steps_per_round"])
            or float(central_state.get("lambda", -1.0)) != 0.0
            or not isinstance(runtime, dict)
            or int(runtime.get("seed", -1))
            != int(contract["base_seed"]) + round_index - 1
            or int(runtime.get("round", -1)) != round_index
            or float(runtime.get("lambda_novelty", -1.0)) != 0.0
            or os.path.realpath(str(runtime.get("model_init", ""))) != sft_final
        ):
            raise ValueError("completed campaign Central runtime contract mismatch")
        if is_c9_multi_explorer(contract) and (
                "scout_novelty_contract" in runtime
                or os.path.exists(os.path.join(central_final, "rnd_state.pt"))):
            raise ValueError(
                "C9 multi-explorer Central must be correctness-only and contain "
                "no RND state")
        central_runtime = dict(runtime)
        central_multi = central_runtime.pop("multi_explorer_contract", None)
        for field in ("model_init", "seed", "round", "lambda_novelty"):
            central_runtime.pop(field, None)
        if central_runtime != dict(training_contract["central_grpo"]):
            raise ValueError("completed campaign Central training contract drifted")
        expected_central_multi = {
            "contract_sha256": contract["contract_sha256"],
            "method": contract["method"],
            "stabilization_profile": contract["stabilization_profile"],
            "scout_index": 0,
            "scouts_per_round": int(contract["scouts_per_round"]),
            "shard_manifest_sha256": str(shard["manifest_sha256"]),
        }
        if is_single_scout_multiround(contract):
            expected_central_multi.update({
                "method_version": contract["method_version"],
                "single_scout_multiround": True,
                "mrme": False,
                "multi_explorer": False,
            })
        if central_multi != expected_central_multi:
            raise ValueError(
                "completed Central multi-explorer runtime binding drifted")
        expected_central_provenance = {
            "method": contract["method"],
            "stabilization_profile": contract["stabilization_profile"],
            "scientific_variant": contract["scientific_variant"],
            "mrme_contract_sha256": contract["contract_sha256"],
            "mrme_shard_manifest_sha256": str(shard["manifest_sha256"]),
            "mrme_scout_index": 0,
            "mrme_scouts_per_round": int(contract["scouts_per_round"]),
            "stage_seed": int(contract["base_seed"]) + round_index - 1,
            "phase": "central",
        }
        if is_single_scout_multiround(contract):
            expected_central_provenance.update({
                "method_version": contract["method_version"],
                "single_scout_multiround": True,
                "mrme": False,
                "multi_explorer": False,
            })
        if central_state.get("mrme_provenance") != expected_central_provenance:
            raise ValueError("completed Central artifact provenance drifted")
        central_artifact = central_stage.get("model_artifact")
        if central_artifact is not None:
            validate_grpo_final_artifact_binding(
                central_artifact, expected_phase="central",
                expected_round=round_index,
                expected_step=int(contract["central_steps_per_round"]),
                expected_policy_manifest=central_state.get(
                    "checkpoint_file_manifest"),
                expected_mrme_provenance=expected_central_provenance)
            common.verify_directory_file_manifest(
                central_final, central_artifact["model_file_manifest"])
        elif is_c9_multi_explorer(contract):
            raise ValueError(
                "completed C9 Central lacks a durable final-model artifact")
        if is_c9_multi_explorer(contract):
            if pooled_sft_gate is None or pooled_central_lineage is None:
                raise ValueError(
                    "completed C9 round lacks pooled-SFT authorization")
            validate_c9_multi_explorer_central_lineage(
                pooled_central_lineage, contract=contract,
                round_index=round_index,
                expected_shard_manifest_sha256=str(shard["manifest_sha256"]),
                expected_trajectory_source_digest=trajectory_digest,
                expected_accepted_digest=accepted_digest,
                expected_central_runtime_contract=runtime)
            gate_stage = stages[f"r{round_index:02d}/sft_behavior_gate"]
            if gate_stage.get("lineage_sha256") != \
                    pooled_central_lineage["lineage_sha256"] or \
                    pooled_central_lineage.get("sft_model_artifact") != \
                    sft_artifact or \
                    central_state.get("external_lineage") != \
                    pooled_central_lineage or \
                    central_state.get("external_lineage_sha256") != \
                    pooled_central_lineage["lineage_sha256"]:
                raise ValueError(
                    "completed Central did not consume the pooled-SFT lineage")
            central_lineage_digests.append(
                str(pooled_central_lineage["lineage_sha256"]))

            central_offset = (
                (round_index - 1) * (
                    int(contract["scouts_per_round"])
                    * int(contract["scout_steps_per_scout"])
                    + int(contract["central_steps_per_round"]))
                + int(contract["scouts_per_round"])
                * int(contract["scout_steps_per_scout"]))
            central_probes = _c9_multi_explorer_central_probe_facts(
                os.path.join(root, "metrics.jsonl"),
                global_step_offset=central_offset, contract=contract,
                round_index=round_index,
                shard_manifest_sha256=str(shard["manifest_sha256"]))
            if central_stage.get("c9_probe_history") != central_probes:
                raise ValueError(
                    "completed C9 Central probe history is missing or drifted")
            central_probe_digests.append(
                str(central_probes["canonical_rows_sha256"]))
            campaign_probe_histories.append(central_probes)

        round_summary = round_summaries[round_index - 1]
        single_scout_round = is_single_scout_multiround(contract)
        expected_round_identity = {
            "method": contract["method"],
            "mrme": not single_scout_round,
            "multi_explorer": not single_scout_round,
            "single_scout_multiround": single_scout_round,
        }
        if single_scout_round:
            expected_round_identity["method_version"] = contract[
                "method_version"]
        if (
            not isinstance(round_summary, dict)
            or int(round_summary.get("round", 0)) != round_index
            or any(round_summary.get(key) != value for key, value in
                   expected_round_identity.items())
            or round_summary.get("stabilization_profile") !=
            contract["stabilization_profile"]
            or round_summary.get("scientific_variant") !=
            contract["scientific_variant"]
            or (is_c9_multi_explorer(contract) and
                round_summary.get("c9_sft_gate_passed") is not True)
            or (is_c9_multi_explorer(contract) and
                round_summary.get("c9_sft_gate_evidence_sha256") !=
                pooled_sft_gate["evidence_sha256"])
            or (is_c9_multi_explorer(contract) and
                round_summary.get("c9_central_lineage_sha256") !=
                pooled_central_lineage["lineage_sha256"])
            or (is_c9_multi_explorer(contract) and
                round_summary.get("c9_central_probe_history") !=
                central_probes)
            or round_summary.get("external_p1_required") is not True
            or round_summary.get("headline_eligible") is not False
            or int(round_summary.get("scouts_per_round", 0))
            != int(contract["scouts_per_round"])
            or os.path.realpath(str(round_summary.get("central_final", "")))
            != central_final
            or round_summary.get("trajectory_source_digest") != trajectory_digest
            or round_summary.get("dataset_shard_manifest_sha256")
            != shard["manifest_sha256"]
            or [os.path.realpath(str(path)) for path in round_summary.get(
                "scout_finals", [])]
            != [os.path.realpath(str(source["scout_final"])) for source in sources]
            or os.path.realpath(str(round_summary.get("mrme_source_index", "")))
            != os.path.realpath(os.path.join(round_dir, "mrme_scout_sources.json"))
        ):
            raise ValueError("completed campaign per-round summary mismatch")
        previous_central = central_final

    partition = assert_round_manifests_disjoint(
        shard_manifests, expected_rounds=int(contract["rounds"]),
        expected_method=str(contract["method"]),
    )
    persisted_partition = load_json_object(
        os.path.join(root, "mrme_shard_partition.json")
    )
    if persisted_partition != partition or summary.get("mrme_shard_partition") != partition:
        raise ValueError("completed campaign shard-partition evidence mismatch")
    if not previous_central or os.path.realpath(str(summary.get("final_model", ""))) != \
            os.path.realpath(previous_central):
        raise ValueError("completed campaign final model is not the last Central")

    campaign_prompt_matrix_sha256 = None
    if is_c9_multi_explorer(contract):
        campaign_prompt_matrix_sha256 = \
            c9_evidence.require_shared_probe_prompt_matrix(
                campaign_probe_histories,
                context="completed C9 MR-ME campaign")

    single_scout_multiround = is_single_scout_multiround(contract)
    evidence = {
        "version": "gpu_multi_explorer_completed_campaign_v2",
        "method": contract["method"],
        "mrme": not single_scout_multiround,
        "multi_explorer": not single_scout_multiround,
        "single_scout_multiround": single_scout_multiround,
        "stabilization_profile": contract["stabilization_profile"],
        "scientific_variant": contract["scientific_variant"],
        "contract_sha256": contract["contract_sha256"],
        "partition_sha256": partition["partition_sha256"],
        "source_index_sha256": source_index_digests,
        "pooled_sft_gate_evidence_sha256": pooled_sft_gate_digests,
        "central_lineage_sha256": central_lineage_digests,
        "central_probe_history_sha256": central_probe_digests,
        "final_model": os.path.realpath(previous_central),
        "final_model_identity_sha256": central_identity["identity_sha256"],
        "rounds": int(contract["rounds"]),
        "scouts_per_round": int(contract["scouts_per_round"]),
        "external_p1_required": True,
        "headline_eligible": False,
        "external_publication_evaluation": dict(
            STRICT_EXTERNAL_P1_REQUIREMENT),
    }
    if single_scout_multiround:
        evidence["method_version"] = contract["method_version"]
    if campaign_prompt_matrix_sha256 is not None:
        evidence["probe_prompt_matrix_sha256"] = \
            campaign_prompt_matrix_sha256
    return {**evidence, "evidence_sha256": canonical_json_sha256(evidence)}


def _portable_relative_path(root: str, path: str, *, field: str) -> str:
    """Return a canonical path below one recorded campaign root."""
    root_real = os.path.realpath(str(root))
    raw = os.path.abspath(str(path))
    real = os.path.realpath(raw)
    if raw != real:
        raise ValueError(f"portable MR-ME {field} may not traverse symlinks")
    try:
        if os.path.commonpath([root_real, real]) != root_real:
            raise ValueError
    except ValueError as exc:
        raise ValueError(
            f"portable MR-ME {field} escapes the campaign root") from exc
    relative = os.path.relpath(real, root_real).replace(os.sep, "/")
    pure = Path(relative)
    if relative in {"", "."} or pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"portable MR-ME {field} is not a safe relative path")
    return relative


def _validate_sft_artifact_for_portability(
    binding: Mapping[str, Any], *, expected_round: int,
) -> dict[str, Any]:
    if not isinstance(binding, Mapping):
        raise ValueError("portable MR-ME SFT artifact binding is absent")
    value = dict(binding)
    unsigned = dict(value)
    recorded = unsigned.pop("binding_sha256", None)
    if recorded != canonical_json_sha256(unsigned) or value.get(
            "version") != "pipeline_sft_model_artifact_v1" or value.get(
                "durable_wandb_upload") is not True or value.get(
                    "artifact_subdir") != "model" or value.get(
                        "artifact_type") != "sft-model" or value.get(
                            "round") != int(expected_round):
        raise ValueError("portable MR-ME SFT artifact identity drifted")
    source = value.get("source")
    if not isinstance(source, Mapping) or not source.get("artifact_digest"):
        raise ValueError("portable MR-ME SFT artifact source is incomplete")
    common.require_immutable_wandb_artifact_ref(
        str(source.get("artifact_ref", "")),
        field="portable MR-ME SFT artifact")
    for name in ("model_file_manifest", "policy_file_manifest"):
        manifest = value.get(name)
        if not isinstance(manifest, Mapping) or manifest.get("version") != \
                common.DIRECTORY_FILE_MANIFEST_VERSION or not manifest.get(
                    "files") or not isinstance(manifest.get("sha256"), str):
            raise ValueError(f"portable MR-ME SFT artifact lacks {name}")
    metadata = value.get("artifact_metadata")
    if not isinstance(metadata, Mapping) or metadata.get("round") != int(
            expected_round) or metadata.get("stage_mode") != value.get(
                "stage_mode") or metadata.get(
                    "model_file_manifest_sha256") != value[
                        "model_file_manifest"]["sha256"] or metadata.get(
                            "accepted_file_sha256") != value.get(
                                "accepted_file_sha256") or metadata.get(
                                    "training_sequence_sha256") != value.get(
                                        "training_sequence_digest", {}).get(
                                            "sha256"):
        raise ValueError("portable MR-ME SFT artifact metadata drifted")
    return value


def build_portable_completed_campaign(output_dir: str) -> dict[str, Any]:
    """Build a signed, thin, independently rehydratable MR-ME package.

    W&B result artifacts intentionally omit multi-gigabyte model directories.
    This package binds every remaining evidence byte plus immutable model
    artifact refs/digests/manifests.  A strict P1 worker can therefore restore
    the original absolute campaign tree and run :func:`validate_completed_campaign`
    unchanged, rather than trusting relocated JSON or mutable ``latest`` aliases.
    """
    root = os.path.realpath(str(output_dir))
    completion_path = os.path.join(root, "mrme_completed_campaign.json")
    completion = load_json_object(completion_path)
    independently_completed = validate_completed_campaign(root)
    if completion != independently_completed:
        raise ValueError(
            "portable MR-ME stored completion differs from local replay")
    contract = validate_contract(load_json_object(
        os.path.join(root, "mrme_contract.json")))
    state = load_json_object(os.path.join(root, "pipeline_state.json"))
    stages = state.get("stages")
    if not isinstance(stages, Mapping):
        raise ValueError("portable MR-ME pipeline state lacks stages")

    thin_paths: set[str] = set()
    model_records: list[dict[str, Any]] = []

    def add_file(path: str, *, field: str) -> None:
        real = os.path.realpath(str(path))
        _portable_relative_path(root, real, field=field)
        if not os.path.isfile(real) or os.path.islink(real):
            raise ValueError(f"portable MR-ME {field} is not a regular file")
        thin_paths.add(real)

    def add_gate(evidence_path: str, *, field: str) -> None:
        gate_path = os.path.realpath(str(evidence_path))
        add_file(gate_path, field=f"{field} evidence")
        for name in ("heldout_selection.jsonl", "rollouts.jsonl",
                     "aggregate_metrics.json"):
            add_file(os.path.join(os.path.dirname(gate_path), name),
                     field=f"{field} {name}")

    def add_model(
        *, role: str, round_index: int, logical_path: str,
        binding: Mapping[str, Any], scout_index: int | None = None,
    ) -> None:
        path = os.path.realpath(str(logical_path))
        relative = _portable_relative_path(
            root, path, field=f"{role} logical model path")
        if not os.path.isdir(path) or os.path.islink(path):
            raise ValueError(f"portable MR-ME {role} model is missing")
        value = dict(binding)
        manifest = value.get("model_file_manifest")
        if not isinstance(manifest, Mapping):
            raise ValueError(f"portable MR-ME {role} model lacks a manifest")
        common.verify_directory_file_manifest(path, dict(manifest))
        source = value.get("source")
        if not isinstance(source, Mapping):
            raise ValueError(f"portable MR-ME {role} source is incomplete")
        artifact_type = str(value.get("artifact_type") or source.get(
            "artifact_type") or "")
        metadata = value.get("artifact_metadata")
        if artifact_type not in {"model", "sft-model"} or not isinstance(
                metadata, Mapping):
            raise ValueError(f"portable MR-ME {role} artifact identity is incomplete")
        model_records.append({
            "role": role,
            "round_index": int(round_index),
            "scout_index": (int(scout_index)
                             if scout_index is not None else None),
            "logical_path": path,
            "relative_logical_path": relative,
            "artifact_subdir": str(value.get("artifact_subdir", "")),
            "artifact_type": artifact_type,
            "artifact_ref": str(source.get("artifact_ref", "")),
            "artifact_digest": str(source.get("artifact_digest", "")),
            "artifact_metadata": dict(metadata),
            "model_file_manifest": dict(manifest),
            "binding": value,
        })

    for name in (
        "metrics.jsonl", "pipeline_state.json", "run_summary.json",
        "mrme_contract.json", "mrme_shard_partition.json",
        "mrme_completed_campaign.json",
    ):
        add_file(os.path.join(root, name), field=name)

    for round_index in range(1, int(contract["rounds"]) + 1):
        round_dir = os.path.join(root, f"round_{round_index:02d}")
        for name in (
            "dataset_shard_manifest.json", "mrme_scout_sources.json",
            "mrme_trajectory_source_manifest.json", "accepted.jsonl",
        ):
            add_file(os.path.join(round_dir, name),
                     field=f"round {round_index} {name}")
        source_index = load_json_object(os.path.join(
            round_dir, "mrme_scout_sources.json"))
        spec = round_spec(contract, round_index)
        for source in source_index.get("sources", []):
            scout_index = int(source["scout_index"])
            stage_key = str(spec["scouts"][scout_index - 1]["stage_key"])
            stage = stages[stage_key]
            source_record_path = str(stage["source_record_path"])
            add_file(source_record_path,
                     field=f"round {round_index} Scout {scout_index} source")
            if load_json_object(source_record_path) != source:
                raise ValueError("portable MR-ME Scout source record drifted")
            add_file(str(source["trajectory_path"]),
                     field=f"round {round_index} Scout {scout_index} trajectories")
            if is_c9_multi_explorer(contract):
                add_gate(str(source["c9_behavior_gate_path"]),
                         field=f"round {round_index} Scout {scout_index} gate")
            binding = source.get("scout_final_artifact")
            validate_grpo_final_artifact_binding(
                binding, expected_phase="scout", expected_round=round_index,
                expected_step=int(contract["scout_steps_per_scout"]),
                expected_policy_manifest=source.get(
                    "scout_final_identity", {}).get(
                        "checkpoint_file_manifest"),
                expected_mrme_provenance=source.get(
                    "scout_final_identity", {}).get("mrme_provenance"))
            sidecar = os.path.join(os.path.dirname(str(source["scout_final"])),
                                   "final_artifact_binding.json")
            add_file(sidecar, field="Scout artifact sidecar")
            if load_json_object(sidecar) != binding:
                raise ValueError("portable MR-ME Scout artifact sidecar drifted")
            add_model(
                role="scout", round_index=round_index,
                scout_index=scout_index,
                logical_path=str(source["scout_final"]), binding=binding)

        sft_stage = stages[f"r{round_index:02d}/sft"]
        sft_binding = _validate_sft_artifact_for_portability(
            sft_stage.get("sft_model_artifact"),
            expected_round=round_index)
        sft_metadata = sft_binding["artifact_metadata"]
        shard = load_json_object(os.path.join(
            round_dir, "dataset_shard_manifest.json"))
        expected_sft_metadata = {
            "method": contract["method"],
            "mrme": not is_single_scout_multiround(contract),
            "multi_explorer": not is_single_scout_multiround(contract),
            "single_scout_multiround": is_single_scout_multiround(contract),
            "stabilization_profile": contract["stabilization_profile"],
            "scientific_variant": contract["scientific_variant"],
            "mrme_contract_sha256": contract["contract_sha256"],
            "mrme_shard_manifest_sha256": shard["manifest_sha256"],
            "mrme_scout_index": 0,
            "mrme_scouts_per_round": int(contract["scouts_per_round"]),
            "phase": "sft",
        }
        if is_single_scout_multiround(contract):
            expected_sft_metadata["method_version"] = contract[
                "method_version"]
        if any(sft_metadata.get(key) != value
               for key, value in expected_sft_metadata.items()):
            raise ValueError("portable MR-ME SFT artifact campaign metadata drifted")
        sft_sidecar = os.path.join(round_dir, "sft_model_artifact.json")
        add_file(sft_sidecar, field="SFT artifact sidecar")
        if load_json_object(sft_sidecar) != sft_binding:
            raise ValueError("portable MR-ME SFT artifact sidecar drifted")
        add_model(
            role="sft", round_index=round_index,
            logical_path=str(sft_stage["final"]), binding=sft_binding)

        if is_c9_multi_explorer(contract):
            gate_stage = stages[f"r{round_index:02d}/sft_behavior_gate"]
            add_gate(str(gate_stage["evidence_path"]),
                     field=f"round {round_index} pooled SFT gate")
            add_file(str(gate_stage["lineage_path"]),
                     field=f"round {round_index} Central lineage")

        central_stage = stages[f"r{round_index:02d}/central"]
        central_state = load_json_object(os.path.join(
            str(central_stage["final"]), "trainer_state.json"))
        central_binding = central_stage.get("model_artifact")
        validate_grpo_final_artifact_binding(
            central_binding, expected_phase="central",
            expected_round=round_index,
            expected_step=int(contract["central_steps_per_round"]),
            expected_policy_manifest=central_state.get(
                "checkpoint_file_manifest"),
            expected_mrme_provenance=central_state.get("mrme_provenance"))
        central_sidecar = os.path.join(
            os.path.dirname(str(central_stage["final"])),
            "final_artifact_binding.json")
        add_file(central_sidecar, field="Central artifact sidecar")
        if load_json_object(central_sidecar) != central_binding:
            raise ValueError("portable MR-ME Central artifact sidecar drifted")
        add_model(
            role="central", round_index=round_index,
            logical_path=str(central_stage["final"]), binding=central_binding)

    logical_paths = [item["logical_path"] for item in model_records]
    remote_ids = [(item["artifact_ref"], item["artifact_digest"])
                  for item in model_records]
    if len(logical_paths) != len(set(logical_paths)) or len(remote_ids) != len(
            set(remote_ids)):
        raise ValueError("portable MR-ME model artifact identities collide")
    files = [
        {
            "relative_path": _portable_relative_path(
                root, path, field="thin evidence file"),
            "sha256": _file_sha256(path),
            "size_bytes": os.path.getsize(path),
        }
        for path in sorted(thin_paths)
    ]
    payload = {
        "recorded_output_root": root,
        "contract_sha256": contract["contract_sha256"],
        "completion": completion,
        "completion_evidence_sha256": completion["evidence_sha256"],
        "partition_sha256": completion["partition_sha256"],
        "files": files,
        "model_artifacts": sorted(
            model_records,
            key=lambda item: (
                item["round_index"], item["role"],
                item["scout_index"] if item["scout_index"] is not None else 0)),
    }
    return _signed_payload(MRME_PORTABLE_CAMPAIGN_VERSION, payload)


def _default_portable_artifact_resolver(
    record: Mapping[str, Any], destination: str,
) -> dict[str, Any]:
    """Download one exact W&B version for portable campaign replay."""
    import wandb

    artifact = wandb.Api().artifact(str(record["artifact_ref"]))
    path = artifact.download(root=destination)
    return {
        "download_root": os.path.realpath(str(path)),
        "artifact_ref": str(getattr(artifact, "qualified_name", "")
                            or getattr(artifact, "name", "") or ""),
        "artifact_digest": str(getattr(artifact, "digest", "") or ""),
        "artifact_type": str(getattr(artifact, "type", "") or ""),
        "artifact_metadata": dict(getattr(artifact, "metadata", {}) or {}),
    }


def validate_portable_completed_campaign(
    campaign_root: str, *,
    artifact_resolver: Callable[[Mapping[str, Any], str], Mapping[str, Any]] | None = None,
    allowed_output_parent: str = "/workspace/runs",
) -> dict[str, Any]:
    """Rehydrate a downloaded thin campaign and run the unchanged validator.

    The original absolute paths are part of model/SFT provenance.  We preserve
    that identity by restoring into the recorded path, but only below an
    explicit trusted parent and only when the destination does not yet exist.
    Every downloaded model is checked against its immutable ref, W&B digest,
    type, metadata, and full directory manifest before it enters that tree.
    """
    package_root = os.path.realpath(str(campaign_root))
    package_path = os.path.join(package_root, "mrme_portable_campaign.json")
    package = _validate_signed_payload(
        load_json_object(package_path), MRME_PORTABLE_CAMPAIGN_VERSION)
    packaged_completion = package.get("completion")
    if not isinstance(packaged_completion, Mapping) or package.get(
            "contract_sha256") != packaged_completion.get(
                "contract_sha256") or package.get(
                    "partition_sha256") != packaged_completion.get(
                        "partition_sha256") or package.get(
                            "completion_evidence_sha256") != \
            packaged_completion.get("evidence_sha256"):
        raise ValueError("portable MR-ME package/completion binding drifted")
    recorded_root = os.path.realpath(str(package.get("recorded_output_root", "")))
    allowed_parent = os.path.realpath(str(allowed_output_parent))
    try:
        contained = os.path.commonpath([allowed_parent, recorded_root]) == \
            allowed_parent
    except ValueError:
        contained = False
    if not recorded_root or recorded_root == allowed_parent or not contained:
        raise ValueError("portable MR-ME restore root is outside the allowed parent")
    if os.path.lexists(recorded_root):
        raise ValueError("portable MR-ME restore root already exists")

    raw_files = package.get("files")
    raw_models = package.get("model_artifacts")
    if not isinstance(raw_files, list) or not raw_files or not isinstance(
            raw_models, list) or not raw_models:
        raise ValueError("portable MR-ME package is incomplete")
    relative_files: set[str] = set()
    for entry in raw_files:
        if not isinstance(entry, Mapping):
            raise ValueError("portable MR-ME file entry is not an object")
        relative = str(entry.get("relative_path", ""))
        pure = Path(relative)
        if not relative or pure.is_absolute() or ".." in pure.parts or \
                relative in relative_files:
            raise ValueError("portable MR-ME file path is unsafe or duplicated")
        source = os.path.join(package_root, relative)
        if os.path.islink(source) or not os.path.isfile(source) or \
                _file_sha256(source) != entry.get("sha256") or \
                os.path.getsize(source) != entry.get("size_bytes"):
            raise ValueError("portable MR-ME thin evidence file drifted")
        relative_files.add(relative)

    resolver = artifact_resolver or _default_portable_artifact_resolver
    created = False
    scratch = tempfile.mkdtemp(prefix="mrme-portable-artifacts-")
    try:
        os.makedirs(recorded_root, exist_ok=False)
        created = True
        for entry in raw_files:
            relative = str(entry["relative_path"])
            destination = os.path.join(recorded_root, relative)
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            shutil.copy2(os.path.join(package_root, relative), destination)

        seen_models: set[str] = set()
        for index, record_value in enumerate(raw_models):
            if not isinstance(record_value, Mapping):
                raise ValueError("portable MR-ME model record is not an object")
            record = dict(record_value)
            logical_path = os.path.realpath(str(record.get("logical_path", "")))
            relative = _portable_relative_path(
                recorded_root, logical_path, field="restored model")
            if relative != record.get("relative_logical_path") or \
                    logical_path in seen_models or os.path.lexists(logical_path):
                raise ValueError("portable MR-ME model destination drifted")
            seen_models.add(logical_path)
            binding = record.get("binding")
            role = str(record.get("role", ""))
            if role == "sft":
                _validate_sft_artifact_for_portability(
                    binding, expected_round=int(record["round_index"]))
            elif role in {"scout", "central"}:
                validate_grpo_final_artifact_binding(
                    binding, expected_phase=role,
                    expected_round=int(record["round_index"]),
                    expected_step=(
                        int(load_json_object(os.path.join(
                            recorded_root, "mrme_contract.json"))[
                                "scout_steps_per_scout"])
                        if role == "scout" else
                        int(load_json_object(os.path.join(
                            recorded_root, "mrme_contract.json"))[
                                "central_steps_per_round"])))
            else:
                raise ValueError("portable MR-ME model role is invalid")
            source = binding.get("source") if isinstance(binding, Mapping) else None
            expected = {
                "artifact_ref": str(source.get("artifact_ref", "")),
                "artifact_digest": str(source.get("artifact_digest", "")),
                "artifact_type": str(record.get("artifact_type", "")),
            }
            if not isinstance(source, Mapping) or record.get(
                    "artifact_ref") != expected["artifact_ref"] or record.get(
                        "artifact_digest") != expected["artifact_digest"] or \
                    record.get("artifact_metadata") != binding.get(
                        "artifact_metadata") or record.get(
                            "model_file_manifest") != binding.get(
                                "model_file_manifest"):
                raise ValueError("portable MR-ME model record/binding drifted")
            common.require_immutable_wandb_artifact_ref(
                expected["artifact_ref"], field="portable MR-ME model artifact")
            download_dir = os.path.join(scratch, f"artifact-{index:03d}")
            resolved = dict(resolver(record, download_dir))
            if any(resolved.get(key) != value for key, value in expected.items()):
                raise ValueError("portable MR-ME resolved artifact identity drifted")
            remote_metadata = resolved.get("artifact_metadata")
            if not isinstance(remote_metadata, Mapping) or any(
                    remote_metadata.get(key) != value
                    for key, value in record["artifact_metadata"].items()):
                raise ValueError("portable MR-ME resolved artifact metadata drifted")
            download_root = os.path.realpath(str(resolved.get("download_root", "")))
            artifact_subdir = str(record.get("artifact_subdir", ""))
            model_source = (download_root if artifact_subdir == "." else
                            os.path.join(download_root, artifact_subdir))
            common.verify_directory_file_manifest(
                model_source, dict(record["model_file_manifest"]))
            os.makedirs(os.path.dirname(logical_path), exist_ok=True)
            shutil.copytree(model_source, logical_path, symlinks=False)
            common.verify_directory_file_manifest(
                logical_path, dict(record["model_file_manifest"]))

        stored_completion = load_json_object(os.path.join(
            recorded_root, "mrme_completed_campaign.json"))
        if stored_completion != package.get("completion") or stored_completion.get(
                "evidence_sha256") != package.get(
                    "completion_evidence_sha256"):
            raise ValueError("portable MR-ME completion binding drifted")
        independently_completed = validate_completed_campaign(recorded_root)
        if independently_completed != stored_completion:
            raise ValueError(
                "portable MR-ME replay differs from stored completion")
        return independently_completed
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        if created:
            shutil.rmtree(recorded_root, ignore_errors=True)
