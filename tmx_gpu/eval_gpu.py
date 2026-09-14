#!/usr/bin/env python3
"""GPU benchmark eval harness reproducing the measured_base_aggregates contract.

Produces one aggregate JSON per (model, benchmark) matching the schema of
docs/paper/handoffs/2026-07-07_agent_reimplementation/measured_base_aggregates/,
plus rollout/per-problem JSONL dumps, and logs everything to W&B.

Sampling contract (paper protocol): temperature=0.6, top_p=0.95, top_k=20,
thinking auto, max_completion_tokens=32768. avg@k = mean accuracy over the
FIRST k samples per problem; pass@k = unbiased 1 - C(n-c,k)/C(n,k);
maj@32 = majority vote over the first 32 samples.

Verifier: tmx_jax.rewarding (pure Python, same module the TPU pipeline used).

Integrity: official launchers require an exact problem-by-sample matrix.  A
single missing/error generation therefore aborts with a nonzero exit before
an aggregate is published.  Manual diagnostic callers may omit that strict
flag; in that mode error completions (finish_reason == "error",
retry-exhausted transport sentinels) are excluded from every metric and the
job still aborts when their rate exceeds 5%.  pass@k / maj@k / avg@k all
guard per-problem n >= k, so under-sampled problems never inflate an
explicitly diagnostic aggregate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tmx_jax.rewarding import (  # noqa: E402
    REWARDING_CONTRACT_VERSION,
    extract_presented_answer,
    _normalize_text_answer,
    _numeric_answers_equal,
    _parse_simple_numeric,
)

from tmx_jax.prompting import (
    MATH_INSTRUCTION as SYSTEM_PROMPT, math_messages, prompt_metadata,
)

_IMMUTABLE_WANDB_ARTIFACT_REF = re.compile(
    r"^[^/\s:]+/[^/\s:]+/[^/\s:]+:v[0-9]+$")
_TRAINED_EXPDIS_METHODS = {
    "expdis_single_scout",
    "integrated_multiscout",
    "disjoint_shard_annealed_mrme",
    "disjoint_shard_annealed_single_scout_multiround",
}
DAPO_PUBLICATION_METHOD = "DAPO-style correctness"
EXECUTING_CODE_FILES = (
    "tmx_gpu/eval_gpu.py",
    "tmx_gpu/eval_matrix.py",
    "tmx_gpu/common.py",
    "tmx_gpu/orchestrator_receipt.py",
    "tmx_gpu/dapo_evidence.py",
    "tmx_gpu/c9_evidence.py",
    "tmx_gpu/mrme_contract.py",
    "tmx_gpu/filter_pool.py",
    "tmx_gpu/sft_gate_gpu.py",
    "tmx_gpu/vllm_pool.py",
    "tmx_jax/data.py",
    "tmx_jax/prompting.py",
    "tmx_jax/rewarding.py",
    "tmx_gpu/launch/base_eval_job.sh",
)


def _is_trained_expdis_method(value: Any) -> bool:
    method = str(value or "")
    return method in _TRAINED_EXPDIS_METHODS or "expdis" in method.lower()


def _requires_training_campaign(value: Any) -> bool:
    return _is_trained_expdis_method(value) or str(value or "") == \
        DAPO_PUBLICATION_METHOD

BENCHMARKS: dict[str, dict[str, Any]] = {
    "AIME24": {
        "hf": ("Maxwell-Jia/AIME_2024", None, "train"),
        "revision": "8d88b2876a82a080e2f172cc9b25d0d9d2cb4792",
        "canonical_count": 30,
        "canonical_rows_sha256":
            "33f24dfe5e519b2f90c44fbb57fb8ce4db12f1e0af457fb92b47df027fa90630",
        "n": 64,
    },
    "AIME25": {
        "hf": ("opencompass/AIME2025", ["AIME2025-I", "AIME2025-II"], "test"),
        "revision": "a6ad95f611d72cf628a80b58bd0432ef6638f958",
        "canonical_count": 30,
        "canonical_rows_sha256":
            "027682101e10b38d718c7724064c13b3e7accd2cb656340672a1346d328187e7",
        "n": 64,
    },
    "MATH500": {
        "hf": ("HuggingFaceH4/MATH-500", None, "test"),
        "revision": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
        "canonical_count": 500,
        "canonical_rows_sha256":
            "553f055c842476cc47f45cd84e518ef1b423473f455624eda37b0649be232767",
        "n": 32,
    },
    "AMC23": {
        "hf": ("math-ai/amc23", None, "test"),
        "revision": "80815d37005feb82cd7f8fbc6901d5d3eff43057",
        "canonical_count": 40,
        "canonical_rows_sha256":
            "56076d9cff03e6883ff590c23681fa4c4a884b5992771008cffbd6eb4ed7239f",
        "n": 32,
    },
    "Minerva-Math": {
        "hf": ("math-ai/minervamath", None, "test"),
        "revision": "ee46ddc498933b1977577953250ca5c66be64f96",
        "canonical_count": 272,
        "canonical_rows_sha256":
            "b25fb30e62b0f58aeef96cb30549c8c6cb6e5ea2bf8d15a45ed78ade48ba8836",
        "n": 64,
    },
    "GSM8K": {
        "hf": ("openai/gsm8k", "main", "test"),
        "revision": "740312add88f781978c0658806c59bc2815b9866",
        "canonical_count": 1319,
        "canonical_rows_sha256":
            "cf4df0e96348bd158f00d6665691cf3b2f344bfe4b4796be2231bc1e85802612",
        "n": 8,
    },
}

SUCCESS_FINISH_REASONS = frozenset({"stop", "length"})
MODEL_FAMILIES = frozenset({"qwen3-1.7b", "qwen3-4b", "ministral-3-3b"})
MODEL_REVISIONS = {
    "qwen3-1.7b": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    "qwen3-4b": "1cfa9a7208912126459214e8b04321603b3df60c",
    "ministral-3-3b": "7046a0e237b436c8fb4927061ab3773772e53741",
}
CANONICAL_MODEL_SOURCE_NAMES = {
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "ministral-3-3b": "mistralai/Ministral-3-3B-Instruct-2512",
}
PUBLICATION_EVAL_CONTAINER_IMAGE_IDENTITY = (
    "vllm/vllm-openai:v0.24.0@"
    "sha256:251eba5cc7c12fed0b75da22a9240e582b1c9e39f6fbc064f86781b963bd814f"
)
# Measured inside the immutable image above after installing the launcher's
# exact lightweight dependency pins.  Publication mode compares the complete
# mapping (including membership), so adding an unrecorded package to this
# identity requires an explicit contract revision rather than silently
# changing the evaluator environment.
PUBLICATION_RUNTIME_STACK_VERSIONS = {
    "datasets": "5.0.0",
    "hf-transfer": "0.1.9",
    "huggingface-hub": "1.21.0",
    "numpy": "2.2.6",
    "requests": "2.34.2",
    "safetensors": "0.8.0",
    "tokenizers": "0.22.2",
    "torch": "2.11.0+cu130",
    "transformers": "5.12.1",
    "vllm": "0.24.0",
    "wandb": "0.28.0",
}
PUBLICATION_VLLM_VERSION = PUBLICATION_RUNTIME_STACK_VERSIONS["vllm"]
EVALUATION_EXECUTION_ENVIRONMENT_VERSION = (
    "publication_evaluation_execution_environment_v1")
MODEL_CONFIG_SIGNATURES = {
    "qwen3-1.7b": {
        "model_type": "qwen3", "hidden_size": 2048,
        "num_hidden_layers": 28, "intermediate_size": 6144,
        "num_attention_heads": 16, "num_key_value_heads": 8,
    },
    "qwen3-4b": {
        "model_type": "qwen3", "hidden_size": 2560,
        "num_hidden_layers": 36, "intermediate_size": 9728,
        "num_attention_heads": 32, "num_key_value_heads": 8,
    },
    "ministral-3-3b": {
        "model_type": "ministral3", "hidden_size": 3072,
        "num_hidden_layers": 26, "intermediate_size": 9216,
        "num_attention_heads": 32, "num_key_value_heads": 8,
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--endpoints", required=True, help="comma-separated vLLM /v1 endpoints")
    p.add_argument("--model", default="Qwen/Qwen3-1.7B", help="HF id or local path (tokenizer + hash)")
    p.add_argument("--served-model-name", default=None)
    p.add_argument("--model-source-name", default=None,
                   help="stable upstream/base-model identity when --model is a local artifact")
    p.add_argument("--model-family", choices=sorted(MODEL_FAMILIES), default=None)
    p.add_argument("--model-revision", default=None,
                   help="immutable Hub revision for a non-artifact base model")
    p.add_argument("--model-artifact-ref", default=None)
    p.add_argument("--model-artifact-digest", default=None)
    p.add_argument("--code-artifact-ref", default=None)
    p.add_argument("--code-artifact-digest", default=None)
    p.add_argument("--training-campaign-binding-sha256", default=None)
    p.add_argument("--expected-vllm-version", default=None,
                   help="publication runs must match the frozen base-anchor runtime")
    p.add_argument(
        "--container-image-identity", default=None,
        help="redundant launcher assertion cross-checked against the immutable "
             "orchestrator receipt; publication identity comes from the raw "
             "Lepton job-spec receipt, not this string",
    )
    p.add_argument(
        "--evaluation-execution-environment", default=None,
        help="validated receipt/native-runtime JSON captured before vLLM startup; "
             "required for publication evaluations",
    )
    p.add_argument("--method-name", default="Base")
    p.add_argument("--benchmark", required=True, choices=sorted(BENCHMARKS))
    p.add_argument("--num-samples", type=int, default=None, help="override samples/problem")
    p.add_argument("--max-problems", type=int, default=None, help="debug cap on problem count")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--min-p", type=float, default=0.0)
    p.add_argument("--protocol", default="P1", choices=["P1", "P2"],
                   help="P1=project 32k budget; P2=Qwen report 38912 budget "
                        "(P2 needs servers launched with MAX_MODEL_LEN >= 40960)")
    p.add_argument("--max-completion-tokens", type=int, default=None)
    p.add_argument("--server-max-model-len", type=int, default=None,
                   help="max-model-len the vLLM servers were launched with "
                        "(base_eval_job.sh passes its MAX_MODEL_LEN); the job "
                        "fails fast if 2048 + the protocol budget cannot fit")
    p.add_argument("--thinking-mode", default="auto", choices=["auto", "on", "off"])
    p.add_argument("--concurrency", type=int, default=96)
    p.add_argument("--timeout", type=int, default=3600)
    p.add_argument("--out-dir", default="baseline_runs_gpu/results")
    p.add_argument("--run-name", required=True)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--resume", action="store_true", help="skip (problem,sample) pairs already in rollouts jsonl")
    p.add_argument(
        "--require-exact-sample-matrix", action="store_true",
        help="fail before publishing unless every requested problem has "
             "exactly --num-samples successful generations (required by "
             "the official P1/P2 launcher)")
    p.add_argument(
        "--require-publication-protocol", action="store_true",
        help="also require the full benchmark, benchmark-default sample "
             "count, and frozen P1/P2 decoding settings; implies the exact "
             "sample-matrix publication standard")
    return p.parse_args()


# ---------------------------------------------------------------- dataset ----

def load_benchmark(name: str) -> list[dict[str, Any]]:
    from datasets import load_dataset, concatenate_datasets

    repo, config, split = BENCHMARKS[name]["hf"]
    revision = BENCHMARKS[name]["revision"]
    if isinstance(config, list):
        parts = [load_dataset(repo, c, split=split, revision=revision) for c in config]
        ds = concatenate_datasets(parts)
    elif config:
        ds = load_dataset(repo, config, split=split, revision=revision)
    else:
        ds = load_dataset(repo, split=split, revision=revision)
    rows = []
    for i, row in enumerate(ds):
        problem = str(
            row.get("problem") or row.get("Problem") or row.get("question")
            or row.get("Question") or ""
        ).strip()
        gt = extract_ground_truth(name, row)
        if problem and gt is not None:
            rows.append({"problem_idx": i, "problem": problem, "ground_truth": gt})
    return rows


def extract_ground_truth(name: str, row: dict[str, Any]) -> str | None:
    if name == "GSM8K":
        raw = str(row.get("answer") or "")
        if "####" in raw:
            return raw.split("####")[-1].strip().replace(",", "")
        return None
    raw = row.get("answer")
    if raw is None:
        raw = row.get("Answer")
    if raw is None:
        raw = row.get("final_answer")
    if raw is None and row.get("solution") is not None:
        raw = row["solution"]
    if raw is None:
        return None
    return str(raw).strip()


# ---------------------------------------------------------------- scoring ----

def grade(pred_text: str, ground_truth: str) -> tuple[str | None, bool]:
    """Return (extracted_answer, is_correct) using the repo verifier semantics."""
    pred = extract_presented_answer(pred_text)
    gt = _normalize_text_answer(ground_truth)
    if not pred or not gt:
        return pred, False
    pred_num = _parse_simple_numeric(pred)
    gt_num = _parse_simple_numeric(gt)
    if pred_num is not None and gt_num is not None:
        return pred, _numeric_answers_equal(pred_num, gt_num)
    return pred, pred == gt


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k for one problem. Callers must guard n >= k (see
    pass_at): with n < k this returns 1.0 even for c == 0."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


# -------------------------------------------------------------- aggregation ----

# Abort (nonzero exit, no aggregate published) above this fraction of error
# completions: a partially-dead serving pool must not poison published numbers.
GENERATION_ERROR_ABORT_RATE = 0.05
EVAL_PROTOCOL_VERSION = "tmx_gpu.eval_gpu_exact_matrix_v4"
SAMPLE_SEED_RULE = "problem_idx*100003+sample_idx"


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
    ).encode("utf-8")


def payload_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    """Cryptographic identity for exact prompt/problem UTF-8 bytes."""
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def canonical_rollout_row_identity(
    row: dict[str, Any], rendered_prompt: str,
) -> dict[str, Any]:
    """Compact identity binding one rollout to its canonical benchmark row.

    Repeating complete prompts in every sample would make the 64-sample
    artifacts unnecessarily large.  These SHA256 fields bind the exact problem
    and rendered prompt bytes while ``gold`` remains explicit for verifier
    replay.  The publication matrix independently reconstructs the same values
    from the pinned dataset revision and base tokenizer.
    """
    problem_idx = int(row["problem_idx"])
    problem = str(row["problem"])
    gold = str(row["ground_truth"])
    prompt = str(rendered_prompt)
    canonical_payload = {
        "problem_idx": problem_idx,
        "problem": problem,
        "ground_truth": gold,
        "rendered_prompt": prompt,
    }
    return {
        "canonical_row_sha256": payload_sha256(canonical_payload),
        "problem_sha256": text_sha256(problem),
        "prompt_sha256": text_sha256(prompt),
    }


def atomic_write_json(path: Path, value: Any) -> None:
    """Replace *path* atomically so a killed writer cannot publish a partial file."""
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def infer_model_family(
    explicit: str | None, *identity_strings: str | None,
) -> str | None:
    if explicit:
        return explicit
    for identity in identity_strings:
        value = str(identity or "").lower()
        if "qwen3-4b" in value or "qwen3_4b" in value:
            return "qwen3-4b"
        if "qwen3-1.7b" in value or "qwen3_1.7b" in value or \
                "qwen3-1p7b" in value:
            return "qwen3-1.7b"
        if "ministral-3-3b" in value or "ministral_3_3b" in value:
            return "ministral-3-3b"
    return None


def benchmark_sample_count(benchmark: str, model_family: str | None) -> int:
    return int(BENCHMARKS[benchmark]["n"])


def validate_model_family_config(
    model: str, model_family: str, revision: str | None = None,
) -> str:
    """Cross-check the declared family against immutable architecture facts."""
    model_path = Path(model)
    path = model_path / "config.json"
    if not path.is_file() and not model_path.exists():
        try:
            from huggingface_hub import hf_hub_download

            path = Path(hf_hub_download(
                model, "config.json", revision=revision))
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(
                f"[eval] FATAL: cannot resolve checkpoint config.json: {exc}") from exc
    if not path.is_file():
        raise SystemExit(
            f"[eval] FATAL: checkpoint config.json is unavailable at {path}")
    try:
        outer = json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"[eval] FATAL: invalid checkpoint config.json: {exc}") from exc
    config = outer.get("text_config") if isinstance(
        outer.get("text_config"), dict) else outer
    expected = MODEL_CONFIG_SIGNATURES[model_family]
    observed = {key: config.get(key) for key in expected}
    drift = [
        f"{key}={observed[key]!r} (expected {value!r})"
        for key, value in expected.items() if observed[key] != value
    ]
    if drift:
        raise SystemExit(
            f"[eval] FATAL: checkpoint config contradicts model_family "
            f"{model_family!r}: " + "; ".join(drift))
    return hashlib.sha256(canonical_json_bytes(observed)).hexdigest()


def validate_loaded_benchmark(
    benchmark: str, rows: list[dict[str, Any]], *, publication: bool,
) -> str:
    """Bind publication to the preregistered dataset revision and row bytes."""
    digest = benchmark_rows_hash(rows)
    if not publication:
        return digest
    spec = BENCHMARKS[benchmark]
    problems = [str(r["problem"]) for r in rows]
    if len(rows) != int(spec["canonical_count"]):
        raise SystemExit(
            f"[eval] FATAL: canonical {benchmark} count mismatch: "
            f"observed {len(rows)}, expected {spec['canonical_count']}")
    if len(set(problems)) != len(problems):
        raise SystemExit(
            f"[eval] FATAL: canonical {benchmark} contains duplicate problem text")
    if digest != str(spec["canonical_rows_sha256"]):
        raise SystemExit(
            f"[eval] FATAL: canonical {benchmark} row hash mismatch: "
            f"observed {digest}, expected {spec['canonical_rows_sha256']}")
    return digest


def benchmark_rows_hash(rows: list[dict[str, Any]]) -> str:
    """Stable identity for the exact eligible benchmark problem set.

    Upstream dataset revisions are not consistently exposed by every HF
    builder used here.  Hashing the normalized rows makes the evaluated bytes
    auditable even when a repository's moving default revision changes.
    """
    h = hashlib.sha256()
    for row in rows:
        payload = {
            "problem_idx": int(row["problem_idx"]),
            "problem": str(row["problem"]),
            "ground_truth": str(row["ground_truth"]),
        }
        h.update(json.dumps(
            payload, sort_keys=True, ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def exact_sample_matrix_status(
    per_problem: list[dict[str, Any]], *, expected_problem_ids: set[int],
    expected_samples: int, n_error: int,
) -> dict[str, Any]:
    """Return fail-closed matrix accounting used before publication."""
    by_problem = {int(p["problem_idx"]): int(p["n"]) for p in per_problem}
    observed_ids = set(by_problem)
    missing_ids = sorted(expected_problem_ids - observed_ids)
    unexpected_ids = sorted(observed_ids - expected_problem_ids)
    wrong_counts = {
        str(pi): by_problem.get(pi, 0)
        for pi in sorted(expected_problem_ids)
        if by_problem.get(pi, 0) != int(expected_samples)
    }
    counts = [by_problem.get(pi, 0) for pi in sorted(expected_problem_ids)]
    complete = (
        bool(expected_problem_ids)
        and not missing_ids and not unexpected_ids and not wrong_counts
        and int(n_error) == 0
    )
    return {
        "exact_sample_matrix": bool(complete),
        "requested_problem_count": len(expected_problem_ids),
        "observed_problem_count": len(observed_ids),
        "requested_samples_per_problem": int(expected_samples),
        "successful_sample_count_min": min(counts, default=0),
        "successful_sample_count_max": max(counts, default=0),
        "complete_problem_count": sum(
            int(n == int(expected_samples)) for n in counts),
        "generation_error_count": int(n_error),
        "missing_problem_ids": missing_ids,
        "unexpected_problem_ids": unexpected_ids,
        "wrong_sample_counts": wrong_counts,
    }


def runtime_stack_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in PUBLICATION_RUNTIME_STACK_VERSIONS:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    return versions


def build_publication_evaluation_execution_environment(
    orchestrator_binding: dict[str, Any],
    native_runtime: dict[str, Any],
    *,
    observed_runtime_stack: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build the exact evaluator environment before any vLLM server starts.

    The immutable image is derived from the authoritative raw Lepton job-spec
    receipt. ``--container-image-identity`` remains only a redundant launcher
    assertion and is never accepted as the source of this object.
    """
    from tmx_gpu import orchestrator_receipt

    binding = orchestrator_receipt.validate_binding(
        orchestrator_binding,
        expected_container_image=PUBLICATION_EVAL_CONTAINER_IMAGE_IDENTITY,
        profile=orchestrator_receipt.EVALUATION_PROFILE,
    )
    stack = dict(
        runtime_stack_versions()
        if observed_runtime_stack is None else observed_runtime_stack)
    if stack != PUBLICATION_RUNTIME_STACK_VERSIONS:
        raise ValueError("publication evaluation runtime stack mismatch")
    receipt_family = binding["receipt"]["job"]["accelerator_family"]
    native = orchestrator_receipt.validate_native_compatibility_facts(
        native_runtime, expected_accelerator_family=receipt_family)
    return {
        "version": EVALUATION_EXECUTION_ENVIRONMENT_VERSION,
        "frozen_invariants": {
            "container_image_identity": binding["receipt"]["job"][
                "container_image_identity"],
            "runtime_stack_versions": stack,
            "orchestrator_receipt_binding": binding,
        },
        "compatibility_facts": {
            "policy": "exactly_8_observed_a100_or_h100_v1",
            "allowed_accelerator_families": ["A100", "H100"],
            "native_runtime": native,
        },
    }


def validate_publication_evaluation_execution_environment(
    value: Any, *,
    expected_run_name: str | None = None,
    expected_code_artifact_ref: str | None = None,
    expected_code_artifact_digest: str | None = None,
) -> dict[str, Any]:
    """Validate a persisted evaluator environment without trusting its claims."""
    if not isinstance(value, dict) or value.get("version") != \
            EVALUATION_EXECUTION_ENVIRONMENT_VERSION:
        raise ValueError(
            "publication evaluation lacks the versioned execution environment")
    if set(value) != {"version", "frozen_invariants", "compatibility_facts"}:
        raise ValueError("publication evaluation environment fields mismatch")
    frozen = value.get("frozen_invariants")
    if not isinstance(frozen, dict) or set(frozen) != {
            "container_image_identity", "runtime_stack_versions",
            "orchestrator_receipt_binding"}:
        raise ValueError("publication evaluation frozen invariants mismatch")
    if frozen.get("container_image_identity") != \
            PUBLICATION_EVAL_CONTAINER_IMAGE_IDENTITY or frozen.get(
                "runtime_stack_versions") != PUBLICATION_RUNTIME_STACK_VERSIONS:
        raise ValueError(
            "publication evaluation environment lacks the frozen image/runtime")

    from tmx_gpu import orchestrator_receipt

    binding = orchestrator_receipt.validate_binding(
        frozen.get("orchestrator_receipt_binding"),
        expected_run_name=expected_run_name,
        expected_container_image=PUBLICATION_EVAL_CONTAINER_IMAGE_IDENTITY,
        expected_code_artifact_ref=expected_code_artifact_ref,
        expected_code_artifact_digest=expected_code_artifact_digest,
        profile=orchestrator_receipt.EVALUATION_PROFILE,
    )
    compatibility = value.get("compatibility_facts")
    if not isinstance(compatibility, dict) or set(compatibility) != {
            "policy", "allowed_accelerator_families", "native_runtime"} or \
            compatibility.get("policy") != \
            "exactly_8_observed_a100_or_h100_v1" or compatibility.get(
                "allowed_accelerator_families") != ["A100", "H100"]:
        raise ValueError("publication evaluation hardware policy mismatch")
    orchestrator_receipt.validate_native_compatibility_facts(
        compatibility.get("native_runtime"),
        expected_accelerator_family=binding[
            "receipt"]["job"]["accelerator_family"],
    )
    return value


def load_publication_evaluation_execution_environment(
    path: str, *,
    expected_run_name: str | None = None,
    expected_code_artifact_ref: str | None = None,
    expected_code_artifact_digest: str | None = None,
) -> dict[str, Any]:
    source = Path(str(path or ""))
    if source.is_symlink() or not source.is_file():
        raise ValueError(
            f"evaluation execution environment is not a regular file: {path!r}")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("cannot load evaluation execution environment") from exc
    return validate_publication_evaluation_execution_environment(
        value,
        expected_run_name=expected_run_name,
        expected_code_artifact_ref=expected_code_artifact_ref,
        expected_code_artifact_digest=expected_code_artifact_digest,
    )


def _execution_environment_from_args(args: argparse.Namespace) -> dict[str, Any]:
    expected_run_name = getattr(args, "run_name", None) or None
    expected_code_ref = getattr(args, "code_artifact_ref", None) or None
    expected_code_digest = getattr(args, "code_artifact_digest", None) or None
    cached = getattr(args, "_evaluation_execution_environment", None)
    if cached is not None:
        environment = validate_publication_evaluation_execution_environment(
            cached,
            expected_run_name=expected_run_name,
            expected_code_artifact_ref=expected_code_ref,
            expected_code_artifact_digest=expected_code_digest,
        )
    else:
        environment = load_publication_evaluation_execution_environment(
            str(getattr(args, "evaluation_execution_environment", "") or ""),
            expected_run_name=expected_run_name,
            expected_code_artifact_ref=expected_code_ref,
            expected_code_artifact_digest=expected_code_digest,
        )
    args._evaluation_execution_environment = environment
    return environment


def executing_code_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in EXECUTING_CODE_FILES:
        path = REPO_ROOT / relative
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def validate_publication_protocol(
    args: argparse.Namespace, *, benchmark_default_samples: int,
    resolved_samples: int,
) -> None:
    """Reject debug/subsampled decoding under a publication-eligible label."""
    if not bool(getattr(args, "require_publication_protocol", False)):
        return
    expected_budget = 38912 if args.protocol == "P2" else 32768
    drift = []
    if args.max_problems is not None:
        drift.append(f"max_problems={args.max_problems!r} (expected None)")
    if not getattr(args, "model_family", None):
        drift.append("model_family is unresolved")
    inferred_family = infer_model_family(
        None, getattr(args, "model_artifact_ref", None),
        getattr(args, "model_source_name", None), getattr(args, "model", None),
    )
    if inferred_family is not None and inferred_family != getattr(
            args, "model_family", None):
        drift.append(
            f"model_family={getattr(args, 'model_family', None)!r} contradicts "
            f"source identity {inferred_family!r}")
    if args.protocol == "P2" and args.benchmark not in {"AIME24", "AIME25"}:
        drift.append(
            f"protocol=P2 is restricted to AIME24/AIME25, got {args.benchmark}")
    if args.server_max_model_len is None:
        drift.append("server_max_model_len is required")
    if getattr(args, "expected_vllm_version", None) != PUBLICATION_VLLM_VERSION:
        drift.append(
            f"expected_vllm_version={getattr(args, 'expected_vllm_version', None)!r} "
            f"(frozen publication version {PUBLICATION_VLLM_VERSION!r})")
    actual_runtime = runtime_stack_versions()
    if getattr(args, "expected_vllm_version", None) == PUBLICATION_VLLM_VERSION:
        actual_vllm = actual_runtime.get("vllm", "unknown")
        if actual_vllm == "unknown" or actual_vllm != args.expected_vllm_version:
            drift.append(
                f"vllm={actual_vllm!r} "
                f"(expected base-anchor {args.expected_vllm_version!r})")
    if actual_runtime != PUBLICATION_RUNTIME_STACK_VERSIONS:
        mismatches = {
            package: {
                "actual": actual_runtime.get(package),
                "expected": PUBLICATION_RUNTIME_STACK_VERSIONS.get(package),
            }
            for package in sorted(
                set(actual_runtime) | set(PUBLICATION_RUNTIME_STACK_VERSIONS))
            if actual_runtime.get(package) !=
            PUBLICATION_RUNTIME_STACK_VERSIONS.get(package)
        }
        drift.append(f"runtime_stack_versions={mismatches!r}")
    evaluation_environment = None
    try:
        evaluation_environment = _execution_environment_from_args(args)
    except (OSError, ValueError) as exc:
        drift.append(f"evaluation_execution_environment={exc}")
    receipt_image = (
        evaluation_environment.get("frozen_invariants", {}).get(
            "container_image_identity")
        if isinstance(evaluation_environment, dict) else None)
    if receipt_image != PUBLICATION_EVAL_CONTAINER_IMAGE_IDENTITY:
        drift.append(
            f"receipt_container_image={receipt_image!r} "
            f"(expected {PUBLICATION_EVAL_CONTAINER_IMAGE_IDENTITY!r})")
    if getattr(args, "container_image_identity", None) != receipt_image:
        drift.append(
            "container_image_identity redundant assertion differs from the "
            f"raw-spec receipt: cli={getattr(args, 'container_image_identity', None)!r}, "
            f"receipt={receipt_image!r}")
    if bool(getattr(args, "model_artifact_ref", None)) != bool(
            getattr(args, "model_artifact_digest", None)):
        drift.append("model artifact ref and digest must be supplied together")
    if not getattr(args, "code_artifact_ref", None) or not getattr(
            args, "code_artifact_digest", None):
        drift.append("code artifact ref and digest are required")
    elif _IMMUTABLE_WANDB_ARTIFACT_REF.fullmatch(
            str(args.code_artifact_ref)) is None:
        drift.append("code artifact ref must use an immutable :vN version")
    if getattr(args, "model_artifact_ref", None) and \
            _IMMUTABLE_WANDB_ARTIFACT_REF.fullmatch(
                str(args.model_artifact_ref)) is None:
        drift.append("model artifact ref must use an immutable :vN version")
    campaign_binding = getattr(
        args, "training_campaign_binding_sha256", None)
    publication_method = str(getattr(args, "method_name", None) or "")
    if publication_method == "Base":
        canonical_source = CANONICAL_MODEL_SOURCE_NAMES.get(str(args.model_family))
        if canonical_source is None:
            drift.append("Base model family has no canonical registry source")
        else:
            if str(args.model) != canonical_source:
                drift.append(
                    f"Base model={str(args.model)!r} "
                    f"(expected canonical registry {canonical_source!r})")
            if str(args.model_source_name or args.model) != canonical_source:
                drift.append(
                    "Base model_source_name must equal the canonical registry model")
            if str(args.served_model_name or args.model) != canonical_source:
                drift.append(
                    "Base served_model_name must equal the canonical registry model")
        if getattr(args, "model_artifact_ref", None):
            drift.append("Base cannot be supplied by a model artifact")
        if campaign_binding:
            drift.append("Base cannot carry a training campaign binding")
    elif not _requires_training_campaign(publication_method):
        drift.append(
            f"unsupported publication method {publication_method!r}; only Base or "
            "a signed canonical trained method is eligible")
    if _requires_training_campaign(getattr(args, "method_name", None)):
        if not isinstance(campaign_binding, str) or \
                re.fullmatch(r"[0-9a-f]{64}", campaign_binding) is None:
            drift.append(
                "trained publication method requires a signed training campaign binding")
    elif campaign_binding:
        drift.append(
            "training campaign binding cannot be attached to an untrained method")
    expected_revision = MODEL_REVISIONS.get(str(args.model_family))
    if not getattr(args, "model_artifact_ref", None) and \
            getattr(args, "model_revision", None) != expected_revision:
        drift.append(
            f"model_revision={getattr(args, 'model_revision', None)!r} "
            f"(expected immutable {expected_revision!r})")
    if int(resolved_samples) != int(benchmark_default_samples):
        drift.append(
            f"num_samples={resolved_samples!r} "
            f"(expected benchmark default {benchmark_default_samples!r})")
    for name, actual, expected in (
        ("temperature", args.temperature, 0.6),
        ("top_p", args.top_p, 0.95),
        ("top_k", args.top_k, 20),
        ("min_p", args.min_p, 0.0),
        ("thinking_mode", args.thinking_mode, "auto"),
        ("max_completion_tokens", args.max_completion_tokens, expected_budget),
    ):
        if actual != expected:
            drift.append(f"{name}={actual!r} (expected {expected!r})")
    if drift:
        raise SystemExit(
            "[eval] FATAL: publication protocol drift: " + "; ".join(drift))


def build_per_problem(results: dict[int, dict[int, dict[str, Any]]]) -> tuple[list[dict[str, Any]], int, int]:
    """Per-problem ordered rows containing only terminal model responses.

    vLLM's publication-eligible terminal states are exactly ``stop`` and
    ``length``.  Transport sentinels and null/unknown terminal states are
    incomplete generations: they are excluded from every metric, counted in
    ``n_error``, and retried by a provenance-matched resume.
    """
    per_problem: list[dict[str, Any]] = []
    n_error = 0
    n_total = 0
    for pi, by_s in results.items():
        ordered = []
        for si in sorted(by_s):
            rec = by_s[si]
            n_total += 1
            if rec.get("finish_reason") not in SUCCESS_FINISH_REASONS:
                n_error += 1
                continue
            ordered.append(rec)
        per_problem.append({
            "problem_idx": pi,
            "n": len(ordered),
            "correct_flags": [r["correct"] for r in ordered],
            "preds": [r["pred"] for r in ordered],
            "clipped_flags": [r["finish_reason"] == "length" for r in ordered],
            "valid_flags": [r["pred"] is not None for r in ordered],
            "tokens": [r.get("completion_tokens") or 0 for r in ordered],
        })
    return per_problem, n_error, n_total


def avg_at(per_problem: list[dict[str, Any]], k: int) -> float | None:
    vals = [sum(p["correct_flags"][:k]) / k for p in per_problem if p["n"] >= k]
    return sum(vals) / len(vals) if vals else None


def pass_at(per_problem: list[dict[str, Any]], k: int) -> float | None:
    # Same per-problem n >= k guard as avg_at: pass_at_k(n < k) returns 1.0
    # regardless of correctness, silently inflating pass@k on under-sampled
    # problems (e.g. a resume with reduced --num-samples).
    vals = [pass_at_k(p["n"], sum(p["correct_flags"]), k)
            for p in per_problem if p["n"] >= k]
    return sum(vals) / len(vals) if vals else None


def maj_at(per_problem: list[dict[str, Any]], gt_by_problem: dict[int, str],
           k: int) -> float | None:
    # Same per-problem n >= k guard; the denominator is the ELIGIBLE problem
    # count (an under-sampled problem must not count as an automatic miss).
    eligible = [p for p in per_problem if p["n"] >= k]
    if not eligible:
        return None
    hits = 0
    for p in eligible:
        preds = [x for x in p["preds"][:k] if x is not None]
        if not preds:
            continue
        top = Counter(preds).most_common(1)[0][0]
        gt = _normalize_text_answer(gt_by_problem[p["problem_idx"]])
        gt_num, top_num = _parse_simple_numeric(gt), _parse_simple_numeric(top)
        ok = (_numeric_answers_equal(top_num, gt_num)
              if (gt_num is not None and top_num is not None)
              else (top == gt))
        hits += int(ok)
    return hits / len(eligible)


def enforce_server_len(args: argparse.Namespace) -> None:
    """Fail fast when the protocol budget cannot fit the servers' context
    window (otherwise EVERY request 400s and the run degrades to 100% error
    completions). P2 (38912 completion) needs MAX_MODEL_LEN >= 40960."""
    if args.server_max_model_len is None:
        if args.protocol == "P2":
            print("[eval] WARNING: --protocol P2 without --server-max-model-len; "
                  "cannot verify the servers fit 2048 + 38912 tokens "
                  "(base_eval_job.sh passes it and auto-raises MAX_MODEL_LEN "
                  "to 40960 for P2)", flush=True)
        return
    required = 2048 + int(args.max_completion_tokens)
    if int(args.server_max_model_len) < required:
        raise SystemExit(
            f"[eval] FATAL: protocol {args.protocol} needs server "
            f"max_model_len >= {required} (2048 prompt + "
            f"{args.max_completion_tokens} completion) but the servers were "
            f"launched with {args.server_max_model_len}. Relaunch with "
            f"MAX_MODEL_LEN >= {required} (base_eval_job.sh auto-raises to "
            "40960 for PROTOCOL=P2) or lower the budget.")


# ---------------------------------------------------------------- serving ----

def render_prompt(tokenizer: Any, question: str, thinking_mode: str) -> str:
    messages = math_messages(question)
    kwargs: dict[str, Any] = {}
    if thinking_mode == "on":
        kwargs["enable_thinking"] = True
    elif thinking_mode == "off":
        kwargs["enable_thinking"] = False
    # "auto": leave template default (Qwen3 default = thinking on)
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **kwargs
    )


def post_one(args: argparse.Namespace, endpoints: list[str], prompt: str, rid: int) -> dict[str, Any]:
    payload = {
        "model": args.served_model_name or args.model,
        "prompt": prompt,
        "max_tokens": args.max_completion_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "n": 1,
        "stream": False,
        "seed": rid,
    }
    order = [endpoints[rid % len(endpoints)]] + [
        e for i, e in enumerate(endpoints) if i != rid % len(endpoints)
    ]
    t0 = time.time()
    last_error = None
    for attempt in range(2):
        for ep in order:
            try:
                r = requests.post(f"{ep}/completions", json=payload, timeout=args.timeout)
                r.raise_for_status()
                ch = r.json()["choices"][0]
                usage = r.json().get("usage", {})
                return {
                    "text": ch.get("text", ""),
                    "finish_reason": ch.get("finish_reason"),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "time_s": round(time.time() - t0, 2),
                }
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(5)
    return {"text": "", "finish_reason": "error", "error": last_error,
            "completion_tokens": 0, "time_s": round(time.time() - t0, 2)}


def model_hash(model: str, revision: str | None = None) -> str:
    """Cryptographic identity for the exact checkpoint consumed by vLLM.

    Local artifacts are hashed by file *contents*, not names/sizes.  For a Hub
    model, the immutable repository revision and LFS blob identities are bound.
    """
    p = Path(model)
    if p.exists():
        h = hashlib.sha256()
        files = sorted(
            f for f in p.rglob("*")
            if f.is_file() and not any(part.startswith(".") for part in f.relative_to(p).parts)
        )
        if not files:
            return "unknown"
        for f in files:
            relative = f.relative_to(p).as_posix()
            h.update(relative.encode("utf-8"))
            h.update(b"\0")
            h.update(str(f.stat().st_size).encode("ascii"))
            h.update(b"\0")
            with f.open("rb") as source:
                for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    h.update(chunk)
            h.update(b"\n")
        return h.hexdigest()
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(
            model, revision=revision, files_metadata=True)
        h = hashlib.sha256()
        h.update(str(info.sha).encode("utf-8"))
        h.update(b"\n")
        for s in sorted(info.siblings, key=lambda s: s.rfilename):
            if s.rfilename.endswith((".safetensors", ".bin")):
                lfs_sha = getattr(getattr(s, "lfs", None), "sha256", None)
                h.update(
                    f"{s.rfilename}:{s.size}:{lfs_sha or s.blob_id}".encode())
                h.update(b"\n")
        return h.hexdigest()
    except Exception:
        return "unknown"


def build_evaluation_manifest(
    args: argparse.Namespace, *, rows: list[dict[str, Any]],
    rows_sha256: str, prompt_template_sha256: str, checkpoint_sha256: str,
    model_config_signature_sha256: str | None, n_samples: int,
) -> dict[str, Any]:
    repo, config, split = BENCHMARKS[args.benchmark]["hf"]
    evaluation_environment = getattr(
        args, "_evaluation_execution_environment", None)
    headline_eligible = bool(
        args.require_publication_protocol and args.protocol == "P1")
    return {
        "manifest_version": "tmx_gpu.evaluation_manifest_v1",
        "eval_protocol_version": EVAL_PROTOCOL_VERSION,
        **prompt_metadata(),
        "protocol": args.protocol,
        "headline_eligible_requested": headline_eligible,
        "benchmark": args.benchmark,
        "benchmark_dataset": {
            "repository": repo,
            "config": config,
            "split": split,
            "revision": BENCHMARKS[args.benchmark]["revision"],
            "rows_sha256": rows_sha256,
            "problem_count": len(rows),
        },
        "num_samples_per_problem": int(n_samples),
        "sample_seed_rule": SAMPLE_SEED_RULE,
        "decoding": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
            "thinking_mode": args.thinking_mode,
            "max_completion_tokens": args.max_completion_tokens,
            "server_max_model_len": args.server_max_model_len,
        },
        "model": {
            "local_or_hf_path": args.model,
            "source_name": args.model_source_name or args.model,
            "source_revision": args.model_revision,
            "served_model_name": args.served_model_name or args.model,
            "family": args.model_family,
            "checkpoint_sha256": checkpoint_sha256,
            "config_signature_sha256": model_config_signature_sha256,
            "artifact_ref": args.model_artifact_ref,
            "artifact_digest": args.model_artifact_digest,
        },
        "code": {
            "artifact_ref": args.code_artifact_ref,
            "artifact_digest": args.code_artifact_digest,
            "executing_file_sha256": executing_code_hashes(),
        },
        "runtime_stack_versions": runtime_stack_versions(),
        "expected_vllm_version": args.expected_vllm_version,
        "container_image_identity": getattr(
            args, "container_image_identity", None),
        "evaluation_execution_environment": evaluation_environment,
        "evaluation_execution_environment_sha256": (
            payload_sha256(evaluation_environment)
            if isinstance(evaluation_environment, dict) else None),
        "prompt_template_sha256": prompt_template_sha256,
        "answer_extractor_version": REWARDING_CONTRACT_VERSION,
        "verifier_version": REWARDING_CONTRACT_VERSION,
        "training_campaign_binding_sha256": (
            getattr(args, "training_campaign_binding_sha256", None)),
    }


def prepare_evaluation_manifest(
    manifest_path: Path, aggregate_path: Path, rollouts_path: Path,
    manifest: dict[str, Any], *, resume: bool,
) -> str:
    """Remove stale publication and establish an exact resume boundary."""
    aggregate_path.unlink(missing_ok=True)
    digest = payload_sha256(manifest)
    if resume and rollouts_path.exists():
        if not manifest_path.exists():
            raise SystemExit(
                "[eval] FATAL: refusing provenance-free resume: rollouts exist "
                "but evaluation_manifest.json is missing")
        try:
            recorded = json.loads(manifest_path.read_text())
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(
                f"[eval] FATAL: unreadable resume manifest: {exc}") from exc
        if canonical_json_bytes(recorded) != canonical_json_bytes(manifest):
            raise SystemExit(
                "[eval] FATAL: resume manifest mismatch; checkpoint, code, "
                "dataset, prompt, verifier, runtime, or sampling contract changed")
    else:
        atomic_write_json(manifest_path, manifest)
    return digest


def load_resume_rows(
    rollouts_path: Path, *, manifest_sha256: str,
    expected_problem_ids: set[int], expected_samples: int,
    ground_truth_by_problem: dict[int, str],
    canonical_identity_by_problem: dict[int, dict[str, Any]] | None = None,
) -> tuple[set[tuple[int, int]], dict[int, dict[int, dict[str, Any]]]]:
    """Load only signed, terminal rows and recompute their verifier result."""
    raw = rollouts_path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        last_start = raw.rfind(b"\n") + 1
        tail = raw[last_start:]
        try:
            json.loads(tail.decode("utf-8"))
        except Exception:  # a killed append: discard only the unterminated tail
            repaired = raw[:last_start]
        else:
            repaired = raw + b"\n"
        tmp = rollouts_path.with_name(
            f".{rollouts_path.name}.repair-{os.getpid()}")
        tmp.write_bytes(repaired)
        os.replace(tmp, rollouts_path)
    results: dict[int, dict[int, dict[str, Any]]] = {
        pi: {} for pi in expected_problem_ids
    }
    done: set[tuple[int, int]] = set()
    with rollouts_path.open() as source:
        for line_number, line in enumerate(source, start=1):
            try:
                rec = json.loads(line)
            except Exception as exc:  # noqa: BLE001
                raise SystemExit(
                    f"[eval] FATAL: malformed resume row {line_number}: {exc}") from exc
            if rec.get("evaluation_manifest_sha256") != manifest_sha256:
                raise SystemExit(
                    f"[eval] FATAL: resume row {line_number} has wrong manifest digest")
            try:
                pi, si = int(rec["problem_idx"]), int(rec["sample_idx"])
            except Exception as exc:  # noqa: BLE001
                raise SystemExit(
                    f"[eval] FATAL: resume row {line_number} lacks valid indices") from exc
            if pi not in expected_problem_ids or not 0 <= si < int(expected_samples):
                raise SystemExit(
                    f"[eval] FATAL: resume row {line_number} index outside matrix")
            if str(rec.get("gold")) != str(ground_truth_by_problem[pi]):
                raise SystemExit(
                    f"[eval] FATAL: resume row {line_number} ground truth mismatch")
            if canonical_identity_by_problem is not None:
                expected_identity = canonical_identity_by_problem.get(pi)
                if not isinstance(expected_identity, dict):
                    raise SystemExit(
                        f"[eval] FATAL: resume row {line_number} lacks canonical "
                        "problem identity")
                for field in (
                    "canonical_row_sha256", "problem_sha256", "prompt_sha256",
                ):
                    if rec.get(field) != expected_identity.get(field):
                        raise SystemExit(
                            f"[eval] FATAL: resume row {line_number} {field} mismatch")
                expected_seed = pi * 100003 + si
                if rec.get("sample_seed") != expected_seed:
                    raise SystemExit(
                        f"[eval] FATAL: resume row {line_number} sample seed mismatch")
            if rec.get("finish_reason") not in SUCCESS_FINISH_REASONS:
                continue
            pred, correct = grade(str(rec.get("text", "")), ground_truth_by_problem[pi])
            rebuilt = dict(rec, pred=pred, correct=bool(correct))
            key = (pi, si)
            if key in done:
                previous = results[pi][si]
                if canonical_json_bytes(previous) != canonical_json_bytes(rebuilt):
                    raise SystemExit(
                        f"[eval] FATAL: conflicting completed resume rows for {key}")
                continue
            done.add(key)
            results[pi][si] = rebuilt
    return done, results


def finish_failed_evaluation(
    wandb_run: Any, *, reason: str, aggregate_path: Path,
    rollouts_path: Path, manifest_path: Path,
    summary: dict[str, Any] | None = None,
) -> None:
    """Durably mark a failed matrix without leaving a publishable aggregate."""
    aggregate_path.unlink(missing_ok=True)
    if wandb_run is None:
        return
    wandb_run.summary["headline_eligible"] = False
    wandb_run.summary["evaluation_status"] = "incomplete"
    wandb_run.summary["failure_reason"] = reason
    for key, value in (summary or {}).items():
        if isinstance(value, (int, float, bool, str)):
            wandb_run.summary[key] = value
    import wandb

    artifact = wandb.Artifact(
        f"{wandb_run.name}-incomplete", type="eval-incomplete",
        metadata={
            "headline_eligible": False,
            "evaluation_status": "incomplete",
            "failure_reason": reason,
        },
    )
    recovery_prefix = (
        f"{manifest_path.parent.parent.name}/{manifest_path.parent.name}")
    for path in (manifest_path, rollouts_path):
        if path.exists():
            artifact.add_file(
                str(path), name=f"{recovery_prefix}/{path.name}")
    wandb_run.log_artifact(artifact)
    wandb_run.finish(exit_code=1)


# -------------------------------------------------------------------- main ----

def main() -> None:
    args = parse_args()
    from transformers import AutoTokenizer

    out_dir = Path(args.out_dir) / args.run_name / args.benchmark
    out_dir.mkdir(parents=True, exist_ok=True)
    rollouts_path = out_dir / "rollouts.jsonl"
    agg_path = out_dir / "aggregate_metrics.json"
    manifest_path = out_dir / "evaluation_manifest.json"
    artifact_receipt_path = out_dir / "evaluation_artifact_receipt.json"
    # A prior successful file must never survive a new failed attempt under
    # the same run name.
    agg_path.unlink(missing_ok=True)
    artifact_receipt_path.unlink(missing_ok=True)

    endpoints = [e.rstrip("/") for e in args.endpoints.split(",") if e.strip()]
    endpoints = [e if e.endswith("/v1") else e + "/v1" for e in endpoints]
    args.model_family = infer_model_family(
        args.model_family, args.model_artifact_ref, args.model_source_name,
        args.model, args.served_model_name,
    )
    benchmark_default_samples = benchmark_sample_count(
        args.benchmark, args.model_family)
    n_samples = args.num_samples or benchmark_default_samples
    if args.max_completion_tokens is None:
        args.max_completion_tokens = 38912 if args.protocol == "P2" else 32768
    validate_publication_protocol(
        args,
        benchmark_default_samples=benchmark_default_samples,
        resolved_samples=int(n_samples),
    )
    if args.require_publication_protocol:
        args.require_exact_sample_matrix = True
    enforce_server_len(args)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=(args.model_revision if not Path(args.model).exists() else None),
    )
    rows = load_benchmark(args.benchmark)
    if args.max_problems:
        rows = rows[: args.max_problems]
    rows_sha256 = validate_loaded_benchmark(
        args.benchmark, rows,
        publication=bool(args.require_publication_protocol),
    )
    print(f"[eval] {args.benchmark}: {len(rows)} problems x {n_samples} samples "
          f"-> {len(rows) * n_samples} generations", flush=True)

    prompt_by_problem = {
        r["problem_idx"]: render_prompt(tokenizer, r["problem"], args.thinking_mode)
        for r in rows
    }
    row_by_problem = {int(r["problem_idx"]): r for r in rows}
    canonical_identity_by_problem = {
        pi: canonical_rollout_row_identity(row_by_problem[pi], prompt)
        for pi, prompt in prompt_by_problem.items()
    }
    template_hash = hashlib.sha256(
        render_prompt(tokenizer, "{PROBLEM}", args.thinking_mode).encode()
    ).hexdigest()
    checkpoint_hash = model_hash(args.model, revision=args.model_revision)
    if args.require_publication_protocol and checkpoint_hash == "unknown":
        raise SystemExit(
            "[eval] FATAL: publication checkpoint has no cryptographic identity")
    if (args.require_publication_protocol and Path(args.model).exists()
            and not args.model_artifact_ref):
        raise SystemExit(
            "[eval] FATAL: a local publication checkpoint requires its source "
            "W&B model artifact ref and digest")
    model_config_signature = None
    if args.model_family:
        model_config_signature = validate_model_family_config(
            args.model, args.model_family, revision=args.model_revision)

    manifest = build_evaluation_manifest(
        args, rows=rows, rows_sha256=rows_sha256,
        prompt_template_sha256=template_hash,
        checkpoint_sha256=checkpoint_hash,
        model_config_signature_sha256=model_config_signature,
        n_samples=n_samples,
    )
    manifest_sha256 = prepare_evaluation_manifest(
        manifest_path, agg_path, rollouts_path, manifest, resume=args.resume)

    done: set[tuple[int, int]] = set()
    results: dict[int, dict[int, dict[str, Any]]] = {r["problem_idx"]: {} for r in rows}
    gt_by_problem = {r["problem_idx"]: r["ground_truth"] for r in rows}
    if args.resume and rollouts_path.exists():
        done, results = load_resume_rows(
            rollouts_path, manifest_sha256=manifest_sha256,
            expected_problem_ids=set(prompt_by_problem),
            expected_samples=n_samples,
            ground_truth_by_problem=gt_by_problem,
            canonical_identity_by_problem=canonical_identity_by_problem,
        )
        print(f"[eval] resume: {len(done)} generations already complete", flush=True)

    tasks = [
        (pi, si)
        for pi in prompt_by_problem
        for si in range(n_samples)
        if (pi, si) not in done
    ]

    headline_eligible = bool(
        args.require_publication_protocol and args.protocol == "P1")
    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "two-model-explore-gpu"),
            entity=os.environ.get("WANDB_ENTITY") or None,
            name=f"{args.run_name}-{args.benchmark}",
            config={
                **vars(args), "num_problems": len(rows), "backend": "gpu-vllm",
                "evaluation_manifest_sha256": manifest_sha256,
                "headline_eligible": headline_eligible,
            },
            resume="allow",
            id=re.sub(
                r"[^a-zA-Z0-9_-]", "-",
                f"{args.run_name}-{args.benchmark}-{manifest_sha256[:12]}",
            ),
        )

    t0 = time.time()
    n_done = 0
    mode = "a" if (args.resume and rollouts_path.exists()) else "w"
    with rollouts_path.open(mode) as out_f, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = {
            pool.submit(post_one, args, endpoints, prompt_by_problem[pi], pi * 100003 + si): (pi, si)
            for pi, si in tasks
        }
        for fut in as_completed(futs):
            pi, si = futs[fut]
            item = fut.result()
            pred, correct = grade(item.get("text", ""), gt_by_problem[pi])
            rec = {
                "problem_idx": pi, "sample_idx": si,
                "sample_seed": pi * 100003 + si,
                "evaluation_manifest_sha256": manifest_sha256,
                "gold": gt_by_problem[pi], "pred": pred, "correct": bool(correct),
                **canonical_identity_by_problem[pi],
                "finish_reason": item.get("finish_reason"),
                "completion_tokens": item.get("completion_tokens", 0),
                "time_s": item.get("time_s"),
                "text": item.get("text", ""),
            }
            results[pi][si] = rec
            out_f.write(json.dumps(rec) + "\n")
            out_f.flush()
            n_done += 1
            if n_done % 50 == 0 or n_done == len(tasks):
                flat = [r for by_s in results.values() for r in by_s.values()
                        if r.get("finish_reason") in SUCCESS_FINISH_REASONS]
                acc = sum(r["correct"] for r in flat) / max(1, len(flat))
                msg = (f"[eval] {n_done}/{len(tasks)} new done "
                       f"(total {len(flat)}) running-acc={acc:.4f} "
                       f"elapsed={(time.time()-t0)/60:.1f}m")
                print(msg, flush=True)
                if wandb_run:
                    wandb_run.log({"progress/done": len(flat), "progress/running_acc": acc})

    # ---------------------------------------------------------- aggregate ----
    # Error completions are EXCLUDED from every metric (counted only in
    # generation_error_rate); above the abort threshold the job exits nonzero
    # WITHOUT publishing an aggregate — rollouts.jsonl is kept, and --resume
    # retries the error rows (resume treats them as not-done).
    per_problem, n_error, n_total = build_per_problem(results)
    generation_error_rate = n_error / max(1, n_total)
    matrix_status = exact_sample_matrix_status(
        per_problem,
        expected_problem_ids=set(prompt_by_problem),
        expected_samples=n_samples,
        n_error=n_error,
    )
    if n_error:
        print(f"[eval] WARNING: {n_error}/{n_total} generations lack a valid "
              f"terminal state (rate {generation_error_rate:.4f}); they are "
              "EXCLUDED from all metrics", flush=True)
    if generation_error_rate > GENERATION_ERROR_ABORT_RATE:
        print(f"[eval] FATAL: generation error rate {generation_error_rate:.1%} "
              f"> {GENERATION_ERROR_ABORT_RATE:.0%} — the serving pool was "
              "unhealthy; aborting WITHOUT publishing an aggregate. Rerun with "
              "--resume to retry the failed rows.", flush=True)
        finish_failed_evaluation(
            wandb_run, reason="generation_error_rate",
            aggregate_path=agg_path, rollouts_path=rollouts_path,
            manifest_path=manifest_path,
            summary={"generation_error_rate": generation_error_rate},
        )
        sys.exit(1)
    if args.require_exact_sample_matrix and not matrix_status["exact_sample_matrix"]:
        print(
            "[eval] FATAL: exact sample matrix incomplete; aborting WITHOUT "
            "publishing an aggregate. Rerun with --resume. Matrix status: "
            + json.dumps(matrix_status, sort_keys=True),
            flush=True,
        )
        finish_failed_evaluation(
            wandb_run, reason="incomplete_exact_sample_matrix",
            aggregate_path=agg_path, rollouts_path=rollouts_path,
            manifest_path=manifest_path,
            summary={f"matrix/{key}": value for key, value in matrix_status.items()},
        )
        sys.exit(1)
    n_max = max((p["n"] for p in per_problem), default=0)

    all_tokens = [t for p in per_problem for t in p["tokens"] if t]
    flat_clipped = [f for p in per_problem for f in p["clipped_flags"]]
    flat_valid = [f for p in per_problem for f in p["valid_flags"]]
    agg = {
        "eval_protocol_version": EVAL_PROTOCOL_VERSION,
        **prompt_metadata(),
        "protocol": args.protocol,
        "protocol_id": f"{EVAL_PROTOCOL_VERSION}:{args.protocol}",
        "evaluation_manifest_sha256": manifest_sha256,
        "headline_eligible": headline_eligible,
        "evaluation_status": "complete",
        "strict_exact_sample_matrix_required": bool(
            args.require_exact_sample_matrix),
        "publication_protocol_required": bool(
            args.require_publication_protocol),
        **matrix_status,
        "answer_extractor_version": REWARDING_CONTRACT_VERSION,
        "verifier_version": REWARDING_CONTRACT_VERSION,
        "benchmark": args.benchmark,
        "method_name": args.method_name,
        "model_name": args.model_source_name or args.model,
        "model_family": args.model_family,
        "model_revision": args.model_revision,
        "served_model_name": args.served_model_name or args.model,
        "checkpoint_path": args.model,
        "checkpoint_hash": checkpoint_hash,
        "model_config_signature_sha256": model_config_signature,
        "model_artifact_ref": args.model_artifact_ref,
        "model_artifact_digest": args.model_artifact_digest,
        "code_artifact_ref": args.code_artifact_ref,
        "code_artifact_digest": args.code_artifact_digest,
        "training_campaign_binding_sha256": (
            getattr(args, "training_campaign_binding_sha256", None)),
        "prompt_template_hash": template_hash,
        "num_problems": len(per_problem),
        "num_samples_per_problem": int(n_samples),
        "observed_num_samples_per_problem_max": n_max,
        "generation_error_rate": generation_error_rate,
        "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k,
        "min_p": args.min_p,
        "sample_seed_rule": SAMPLE_SEED_RULE,
        "server_max_model_len": args.server_max_model_len,
        "runtime_stack_versions": runtime_stack_versions(),
        "container_image_identity": getattr(
            args, "container_image_identity", None),
        "evaluation_execution_environment": getattr(
            args, "_evaluation_execution_environment", None),
        "evaluation_execution_environment_sha256": (
            payload_sha256(args._evaluation_execution_environment)
            if isinstance(getattr(
                args, "_evaluation_execution_environment", None), dict)
            else None),
        "benchmark_dataset_spec": {
            "repository": BENCHMARKS[args.benchmark]["hf"][0],
            "config": BENCHMARKS[args.benchmark]["hf"][1],
            "split": BENCHMARKS[args.benchmark]["hf"][2],
            "revision": BENCHMARKS[args.benchmark]["revision"],
        },
        "benchmark_rows_sha256": rows_sha256,
        "eligible_problem_ids_sha256": hashlib.sha256(
            json.dumps(sorted(prompt_by_problem), separators=(",", ":")).encode()
        ).hexdigest(),
        "eligible_problem_count_by_k": {
            str(k): sum(int(p["n"] >= k) for p in per_problem)
            for k in (1, 2, 4, 8, 16, 32, 64)
        },
        "thinking_mode": args.thinking_mode,
        "max_completion_tokens": args.max_completion_tokens,
        **{f"avg@{k}": avg_at(per_problem, k) for k in (1, 4, 8, 16, 32, 64)
           if avg_at(per_problem, k) is not None},
        **{f"pass@{k}": pass_at(per_problem, k) for k in (1, 2, 4, 8, 16, 32, 64)
           if pass_at(per_problem, k) is not None},
        "maj@16": maj_at(per_problem, gt_by_problem, 16),
        "maj@32": maj_at(per_problem, gt_by_problem, 32),
        "valid_answer_rate": sum(flat_valid) / max(1, len(flat_valid)),
        "parse_failure_rate": 1.0 - sum(flat_valid) / max(1, len(flat_valid)),
        "clipped_rate": sum(flat_clipped) / max(1, len(flat_clipped)),
        "nonterminated_rate": sum(flat_clipped) / max(1, len(flat_clipped)),
        "distinct_answer_mean": (
            sum(len(set(x for x in p["preds"] if x is not None)) for p in per_problem)
            / max(1, len(per_problem))
        ),
        "mean_completion_tokens": (sum(all_tokens) / len(all_tokens)) if all_tokens else 0,
        "median_completion_tokens": statistics.median(all_tokens) if all_tokens else 0,
        "p90_completion_tokens": (
            sorted(all_tokens)[int(0.9 * (len(all_tokens) - 1))] if all_tokens else 0
        ),
        "completed_at_unix": time.time(),
        "elapsed_s": time.time() - t0,
    }
    atomic_write_json(agg_path, agg)
    print(json.dumps({k: v for k, v in agg.items() if not k.startswith("check")},
                     indent=2, sort_keys=True), flush=True)

    if wandb_run:
        wandb_run.log({f"eval/{args.benchmark}/{k}": v for k, v in agg.items()
                       if isinstance(v, (int, float)) and v is not None})
        wandb_run.summary["headline_eligible"] = False
        wandb_run.summary["evaluation_status"] = "uploading"
        try:
            art = __import__("wandb").Artifact(
                f"{args.run_name}-{args.benchmark}-{manifest_sha256[:12]}",
                type="eval-results",
                metadata={
                    "headline_eligible": headline_eligible,
                    "evaluation_status": "complete",
                    "evaluation_manifest_sha256": manifest_sha256,
                })
            source_files = {
                "aggregate_metrics.json": agg_path,
                "rollouts.jsonl": rollouts_path,
                "evaluation_manifest.json": manifest_path,
            }
            for name, path in source_files.items():
                art.add_file(str(path), name=name)
            logged = wandb_run.log_artifact(art)
            logged.wait()
            if getattr(logged, "state", None) != "COMMITTED":
                raise RuntimeError(
                    "eval artifact did not commit: "
                    f"state={getattr(logged, 'state', None)!r}")
            verify_root = out_dir / ".artifact_upload_verify"
            shutil.rmtree(verify_root, ignore_errors=True)
            logged.download(root=str(verify_root))
            for name, source_path in source_files.items():
                committed_path = verify_root / name
                if not committed_path.is_file() or \
                        file_sha256(committed_path) != file_sha256(source_path):
                    raise RuntimeError(
                        f"committed eval artifact member mismatch: {name}")
            shutil.rmtree(verify_root, ignore_errors=True)
            receipt = {
                "schema_version": "tmx_gpu.evaluation_artifact_receipt_v1",
                "artifact_ref": logged.qualified_name,
                "artifact_digest": logged.digest,
                "evaluation_manifest_sha256": manifest_sha256,
                "source_file_sha256": {
                    name: file_sha256(path) for name, path in source_files.items()
                },
                "upload_verified": True,
            }
            atomic_write_json(artifact_receipt_path, receipt)
            wandb_run.summary["eval_artifact_ref"] = logged.qualified_name
            wandb_run.summary["eval_artifact_digest"] = logged.digest
            wandb_run.summary["evaluation_status"] = "complete"
            wandb_run.summary["headline_eligible"] = headline_eligible
            wandb_run.finish()
        except Exception:
            agg_path.unlink(missing_ok=True)
            artifact_receipt_path.unlink(missing_ok=True)
            wandb_run.summary["headline_eligible"] = False
            wandb_run.summary["evaluation_status"] = "artifact_upload_failed"
            wandb_run.finish(exit_code=1)
            raise
    print(f"[eval] DONE {args.benchmark}: aggregate={agg_path}", flush=True)


if __name__ == "__main__":
    main()
