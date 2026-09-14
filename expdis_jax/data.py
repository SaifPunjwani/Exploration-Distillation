"""Dataset loaders (DAPO-Math-17k, DeepScaleR, JSONL) and Qwen3 prompt rendering.

DAPO rows from ``open-r1/DAPO-Math-17k-Processed`` carry ``prompt`` (the problem
text) and ``solution`` (the short gold answer); the loaders also accept
``problem``/``question`` and ``answer``/``final_answer`` keys. Prompts are
rendered with the tokenizer's chat template here so vLLM receives a fully
formatted string via ``/completions`` (never ``/chat/completions``, which would
template twice).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from datasets import load_dataset


# Compatibility name; the instruction is now in the user turn.
from .prompting import MATH_INSTRUCTION as SYSTEM_PROMPT, math_messages


@dataclass
class DapoExample:
    problem_id: str
    prompt_text: str       # Chat-template-rendered prompt to feed vLLM /completions
    ground_truth: str      # Normalized expected answer
    raw_solution: str
    topic: str             # "dapo" (all items share a topic in this dataset)


DAPO_DATASET_ID = "open-r1/DAPO-Math-17k-Processed"
DAPO_DATASET_CONFIG = "all"
DAPO_DATASET_REVISION = "31dd309567e3da778038cc87d868b6097a3ccf68"
DEEPSCALER_DATASET_ID = "agentica-org/DeepScaleR-Preview-Dataset"
DEEPSCALER_DATASET_REVISION = "b6ae8c60f5c1f2b594e2140b91c49c9ad0949e29"
TRAINING_DATASET_IDENTITY_VERSION = "training_dataset_source_v1"


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_dapo_training_dataset_identity() -> Dict[str, Any]:
    """Return the immutable Hub identity, ignoring ambient local overrides."""
    core = {
        "version": TRAINING_DATASET_IDENTITY_VERSION,
        "kind": "huggingface_dataset",
        "dataset_label": "dapo_math_17k",
        "repo_id": DAPO_DATASET_ID,
        "config": DAPO_DATASET_CONFIG,
        "split": "train",
        "revision": DAPO_DATASET_REVISION,
    }
    return {**core, "identity_sha256": _canonical_json_sha256(core)}


def training_dataset_identity(dataset_name: str) -> Dict[str, Any]:
    """Return the immutable source identity used by :func:`load_examples`.

    The local-JSONL precedence intentionally mirrors ``load_examples``.  A
    path alone is never considered an identity: local inputs bind both their
    resolved path and full byte SHA256/size, while Hub inputs bind the exact
    repository/config/split/commit tuple.
    """
    name = (dataset_name or "dapo_math_17k").strip().lower()
    jsonl_path = os.environ.get("EXPDIS_TRAIN_DATASET_JSONL", "").strip()
    if name.startswith("jsonl:"):
        jsonl_path = jsonl_path or str(dataset_name).split(":", 1)[1]
    elif os.path.isfile(os.path.expanduser(str(dataset_name))):
        # grpo_gpu.load_training_examples accepts a direct JSONL path for
        # smoke/local runs; identity resolution must mirror that route too.
        jsonl_path = str(dataset_name)
    if jsonl_path:
        path = os.path.realpath(os.path.expanduser(jsonl_path))
        if not os.path.isfile(path):
            raise ValueError(f"local training JSONL is not a file: {path!r}")
        core = {
            "version": TRAINING_DATASET_IDENTITY_VERSION,
            "kind": "local_jsonl",
            "dataset_label": name,
            "resolved_path": path,
            "content_sha256": _file_sha256(path),
            "size_bytes": int(os.path.getsize(path)),
        }
    elif name in ("dapo_math_17k", "dapo", "dapo_math", "dapo-math-17k"):
        return canonical_dapo_training_dataset_identity()
    elif name in (
            "deepscaler", "deepscaler_preview",
            "agentica-org/deepscaler-preview-dataset"):
        core = {
            "version": TRAINING_DATASET_IDENTITY_VERSION,
            "kind": "huggingface_dataset",
            "dataset_label": "deepscaler",
            "repo_id": DEEPSCALER_DATASET_ID,
            "config": None,
            "split": "train",
            "revision": DEEPSCALER_DATASET_REVISION,
        }
    else:
        raise ValueError(
            f"Unknown dataset_name={dataset_name!r}; supported: "
            "dapo_math_17k, deepscaler, jsonl:<label>")
    return {**core, "identity_sha256": _canonical_json_sha256(core)}


def _load_raw():
    """Load the immutable DAPO training snapshot used by all ExpDis arms."""
    return load_dataset(
        DAPO_DATASET_ID,
        DAPO_DATASET_CONFIG,
        split="train",
        revision=DAPO_DATASET_REVISION,
    )


ROUND_SHARD_MANIFEST_VERSION = "deterministic_round_shard_v1"


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _round_shard_required() -> bool:
    return (os.environ.get("EXPDIS_DATASET_SHARD_REQUIRED", "") or "").strip().lower() \
        in ("1", "true", "yes")


def _round_shard_settings() -> Tuple[int, int, str, int, bool]:
    """Resolve shard settings, failing closed for MR-ME-required loads."""
    required = _round_shard_required()
    names = (
        "EXPDIS_DATASET_SHARD_INDEX", "EXPDIS_DATASET_SHARD_COUNT",
        "EXPDIS_DATASET_SHARD_POLICY", "EXPDIS_DATASET_SHARD_SEED",
    )
    if required:
        missing = [name for name in names if name not in os.environ]
        if missing:
            raise ValueError(
                "required deterministic dataset shard environment is incomplete: "
                + ", ".join(missing)
            )
    shard_count = int(os.environ.get("EXPDIS_DATASET_SHARD_COUNT", "1") or "1")
    if shard_count < 1:
        raise ValueError(
            f"EXPDIS_DATASET_SHARD_COUNT must be >= 1, got {shard_count}"
        )
    if required and shard_count <= 1:
        raise ValueError("required MR-ME dataset sharding needs SHARD_COUNT > 1")
    shard_index = int(os.environ.get("EXPDIS_DATASET_SHARD_INDEX", "0") or "0")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(
            f"EXPDIS_DATASET_SHARD_INDEX must be in [0, {shard_count}), got {shard_index}"
        )
    policy = (
        os.environ.get("EXPDIS_DATASET_SHARD_POLICY", "uniform") or "uniform"
    ).strip().lower()
    if policy in ("shuffle", "shuffled"):
        policy = "uniform"
    elif policy == "contiguous":
        policy = "sequential"
    if policy not in ("uniform", "sequential"):
        raise ValueError(
            "unsupported EXPDIS_DATASET_SHARD_POLICY="
            f"{policy!r}; expected uniform or sequential"
        )
    if required and policy != "uniform":
        raise ValueError(
            "MR-ME requires deterministic uniform round sharding, got "
            f"{policy!r}"
        )
    seed = int(os.environ.get("EXPDIS_DATASET_SHARD_SEED", "0") or "0")
    return shard_index, shard_count, policy, seed, required


def _write_or_verify_round_shard_manifest(
    path: str, manifest: Dict[str, Any],
) -> None:
    """Atomically create one manifest, or byte-compare an existing one.

    All K Scouts in an MR-ME round share this path.  ``link`` gives us an
    atomic create-if-absent operation: concurrent writers can only succeed
    when they computed identical content; the loser verifies the winner.
    """
    target = os.path.realpath(path)
    if not target:
        raise ValueError("EXPDIS_DATASET_SHARD_MANIFEST_PATH is empty")
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    payload = json.dumps(
        manifest, sort_keys=True, indent=2, ensure_ascii=False,
    ) + "\n"
    fd, tmp = tempfile.mkstemp(
        prefix=f".{os.path.basename(target)}.", suffix=".tmp",
        dir=os.path.dirname(target) or ".",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(tmp, target)
        except FileExistsError:
            with open(target, encoding="utf-8") as handle:
                existing = json.load(handle)
            if existing != manifest:
                raise ValueError(
                    "dataset shard membership/provenance differs from the "
                    f"existing round manifest: {target!r}"
                ) from None
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)


def build_round_shard(
    rows: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Return the configured deterministic shard and exact membership proof."""
    shard_index, shard_count, policy, seed, required = _round_shard_settings()
    indexed = list(enumerate(rows))
    full_members = [
        {
            "source_index": int(index),
            "row_sha256": _canonical_json_sha256(row),
        }
        for index, row in indexed
    ]
    full_dataset_sha256 = _canonical_json_sha256(full_members)
    # Preserve the pre-existing single-round loader order exactly.  A shard
    # seed must not reorder ordinary (SHARD_COUNT=1) training data.
    if policy == "uniform" and shard_count > 1:
        random.Random(seed).shuffle(indexed)
    n = len(indexed)
    start = (n * shard_index) // shard_count
    end = (n * (shard_index + 1)) // shard_count
    selected = indexed[start:end]
    members = [
        {
            "source_index": int(index),
            "row_sha256": _canonical_json_sha256(row),
        }
        for index, row in selected
    ]
    core = {
        "version": ROUND_SHARD_MANIFEST_VERSION,
        "algorithm": (
            "seeded_shuffle_then_floor_boundary_contiguous_slice"
            if shard_count > 1 else "identity_no_shard"
        ),
        "shard_index": int(shard_index),
        "shard_count": int(shard_count),
        "shard_policy": policy,
        "shard_seed": int(seed),
        "required": bool(required),
        "fraction_numerator": 1,
        "fraction_denominator": int(shard_count),
        "total_rows": int(n),
        "start": int(start),
        "end": int(end),
        "num_rows": int(end - start),
        "full_dataset_sha256": full_dataset_sha256,
        "members": members,
        "membership_sha256": _canonical_json_sha256(members),
    }
    manifest = {**core, "manifest_sha256": _canonical_json_sha256(core)}
    manifest_path = (
        os.environ.get("EXPDIS_DATASET_SHARD_MANIFEST_PATH", "") or ""
    ).strip()
    if required and not manifest_path:
        raise ValueError(
            "MR-ME requires EXPDIS_DATASET_SHARD_MANIFEST_PATH so every Scout "
            "is bound to exact round membership"
        )
    if manifest_path:
        _write_or_verify_round_shard_manifest(manifest_path, manifest)
    return [row for _, row in selected], manifest


def validate_round_shard_manifest(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a loaded exact-membership manifest without dataset access."""
    if not isinstance(manifest, dict):
        raise ValueError("round shard manifest must be an object")
    raw = dict(manifest)
    digest = raw.pop("manifest_sha256", None)
    if raw.get("version") != ROUND_SHARD_MANIFEST_VERSION:
        raise ValueError("round shard manifest version mismatch")
    if digest != _canonical_json_sha256(raw):
        raise ValueError("round shard manifest SHA256 mismatch")
    members = raw.get("members")
    if not isinstance(members, list) or len(members) != int(raw.get("num_rows", -1)):
        raise ValueError("round shard manifest member count mismatch")
    if raw.get("membership_sha256") != _canonical_json_sha256(members):
        raise ValueError("round shard membership SHA256 mismatch")
    shard_count = int(raw.get("shard_count", 0))
    shard_index = int(raw.get("shard_index", -1))
    total_rows = int(raw.get("total_rows", -1))
    if shard_count < 1 or shard_index < 0 or shard_index >= shard_count:
        raise ValueError("round shard manifest index/count is invalid")
    expected_start = (total_rows * shard_index) // shard_count
    expected_end = (total_rows * (shard_index + 1)) // shard_count
    if int(raw.get("start", -1)) != expected_start \
            or int(raw.get("end", -1)) != expected_end:
        raise ValueError("round shard manifest does not use exact floor boundaries")
    if expected_end - expected_start != len(members):
        raise ValueError("round shard floor-boundary count mismatch")
    source_indices = [int(member.get("source_index", -1)) for member in members]
    if len(source_indices) != len(set(source_indices)) or any(
        index < 0 or index >= total_rows for index in source_indices
    ):
        raise ValueError("round shard manifest source indices are invalid/duplicated")
    if any(len(str(member.get("row_sha256", ""))) != 64 for member in members):
        raise ValueError("round shard manifest row SHA256 is invalid")
    return manifest


def _apply_round_shard(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply a deterministic disjoint dataset shard for multi-round training.

    The pipeline sets EXPDIS_DATASET_SHARD_* per round. Sharding happens after the
    configured dataset cap is applied, so max_train_examples remains the total
    campaign dataset budget when round_budget_mode=split.
    """
    _index, shard_count, _policy, _seed, required = _round_shard_settings()
    manifest_path = (
        os.environ.get("EXPDIS_DATASET_SHARD_MANIFEST_PATH", "") or ""
    ).strip()
    if shard_count == 1 and not required and not manifest_path:
        return rows
    sharded, _manifest = build_round_shard(rows)
    return sharded


def normalize_answer(s: str) -> str:
    """Basic numeric normalization for comparison."""
    if s is None:
        return ""
    s = s.strip()
    # strip $ signs, commas, degree, percent
    s = s.replace("$", "").replace(",", "").replace("°", "").replace("%", "")
    # leading + or zero-pad
    if s.startswith("+"):
        s = s[1:]
    try:
        # Normalize trivial floats (3.0 -> 3)
        f = float(s)
        if f.is_integer():
            return str(int(f))
        return s
    except Exception:
        return s


# Compatibility alias for callers using the old constant name.
EVAL_SYSTEM_PROMPT_P1 = SYSTEM_PROMPT


def render_chat_prompt(problem: str, tokenizer, enable_thinking: bool = True, system_prompt: str | None = None) -> str:
    """Apply the model chat template so vLLM sees a fully formatted prompt.

    Qwen3 and Gemma 4 both expose ``enable_thinking`` through their HF chat
    template helpers. Older tokenizers may not; in that case we retry without
    the model-specific kwarg.
    """
    if system_prompt not in (None, SYSTEM_PROMPT):
        raise ValueError("math prompts use the shared Qwen user instruction")
    messages = math_messages(problem)
    kwargs = {
        "add_generation_prompt": True,
        "tokenize": False,
        "enable_thinking": bool(enable_thinking),
    }
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def render_qwen3_prompt(problem: str, tokenizer, enable_thinking: bool = True, system_prompt: str | None = None) -> str:
    """Backward-compatible alias for older call sites."""
    return render_chat_prompt(problem, tokenizer, enable_thinking=enable_thinking, system_prompt=system_prompt)


def load_dapo(tokenizer, max_examples: int | None = None, enable_thinking: str = "auto") -> List[DapoExample]:
    ds = _load_raw()
    rows = ds.to_list() if max_examples is None else ds.select(range(min(max_examples, len(ds)))).to_list()
    rows = _apply_round_shard(rows)
    out: List[DapoExample] = []
    et = True if enable_thinking != "never" else False
    for i, row in enumerate(rows):
        problem = row.get("prompt") or row.get("problem") or row.get("question") or ""
        answer = row.get("solution") or row.get("answer") or row.get("final_answer") or ""
        solution = row.get("reasoning") or ""
        if not problem or not answer:
            continue
        prompt_text = render_qwen3_prompt(problem, tokenizer, enable_thinking=et)
        out.append(DapoExample(
            problem_id=str(row.get("id", i)),
            prompt_text=prompt_text,
            ground_truth=str(answer),
            raw_solution=str(solution),
            topic="dapo",
        ))
    return out


def _load_raw_deepscaler():
    return load_dataset(
        DEEPSCALER_DATASET_ID,
        split="train",
        revision=DEEPSCALER_DATASET_REVISION,
    )


def load_deepscaler(tokenizer, max_examples: int | None = None, enable_thinking: str = "auto") -> List[DapoExample]:
    """Load agentica-org/DeepScaleR-Preview-Dataset as DapoExample list.

    Schema: {problem, answer, solution}. Answers may be LaTeX (e.g. '-\\frac{2}{3}')
    rather than plain numerics; reward extraction handles the comparison via
    rewarding.is_correct_completion.
    """
    ds = _load_raw_deepscaler()
    rows = ds.to_list() if max_examples is None else ds.select(range(min(max_examples, len(ds)))).to_list()
    rows = _apply_round_shard(rows)
    out: List[DapoExample] = []
    et = True if enable_thinking != "never" else False
    for i, row in enumerate(rows):
        problem = row.get("problem") or row.get("prompt") or row.get("question") or ""
        answer = row.get("answer") or row.get("final_answer") or ""
        solution = row.get("solution") or ""
        if not problem or not answer:
            continue
        prompt_text = render_qwen3_prompt(problem, tokenizer, enable_thinking=et)
        out.append(DapoExample(
            problem_id=str(row.get("id", i)),
            prompt_text=prompt_text,
            ground_truth=str(answer),
            raw_solution=str(solution),
            topic="deepscaler",
        ))
    return out


def load_jsonl_examples(path: str, tokenizer, max_examples: int | None = None,
                        enable_thinking: str = "auto", topic: str = "jsonl") -> List[DapoExample]:
    """Load a local JSONL math dataset as DapoExample rows.

    This is used by the baseline-only DeepScaleR decontamination audit. It keeps
    the trainer on the exact cleaned split instead of silently falling back to
    the raw HF dataset.
    """
    rows: list[dict[str, Any]] = []
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if max_examples is not None and len(rows) >= max_examples:
                    break
    rows = _apply_round_shard(rows)
    out: List[DapoExample] = []
    et = True if enable_thinking != "never" else False
    for i, row in enumerate(rows):
        problem = row.get("problem") or row.get("prompt") or row.get("question") or row.get("input") or ""
        answer = row.get("answer") or row.get("final_answer") or row.get("target") or ""
        solution = row.get("solution") or row.get("reasoning") or ""
        if not problem or not answer:
            continue
        prompt_text = render_qwen3_prompt(str(problem), tokenizer, enable_thinking=et)
        out.append(DapoExample(
            problem_id=str(row.get("id", row.get("_deepscaler_original_index", i))),
            prompt_text=prompt_text,
            ground_truth=str(answer),
            raw_solution=str(solution),
            topic=topic,
        ))
    return out


def load_examples(dataset_name: str, tokenizer, max_examples: int | None = None,
                  enable_thinking: str = "auto") -> List[DapoExample]:
    """Dispatch by dataset_name. Adds new datasets here."""
    name = (dataset_name or "dapo_math_17k").strip().lower()
    jsonl_path = os.environ.get("EXPDIS_TRAIN_DATASET_JSONL", "").strip()
    if name.startswith("jsonl:"):
        # `jsonl:<label>` is used in baseline configs while the launcher syncs
        # the actual file path through EXPDIS_TRAIN_DATASET_JSONL.
        jsonl_path = jsonl_path or dataset_name.split(":", 1)[1]
    if jsonl_path:
        return load_jsonl_examples(
            jsonl_path,
            tokenizer,
            max_examples=max_examples,
            enable_thinking=enable_thinking,
            topic=name.replace("jsonl:", "") or "jsonl",
        )
    if name in ("dapo_math_17k", "dapo", "dapo_math", "dapo-math-17k"):
        return load_dapo(tokenizer, max_examples=max_examples, enable_thinking=enable_thinking)
    if name in ("deepscaler", "deepscaler_preview", "agentica-org/deepscaler-preview-dataset"):
        return load_deepscaler(tokenizer, max_examples=max_examples, enable_thinking=enable_thinking)
    raise ValueError(f"Unknown dataset_name={dataset_name!r}; supported: dapo_math_17k, deepscaler")
