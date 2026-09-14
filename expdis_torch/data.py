import ast
import json
import os
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from datasets import DatasetDict, DownloadConfig, concatenate_datasets, load_dataset

from .prompting import (
    generation_answer_prefix,
    render_supervised_prompt,
    render_supervised_text,
)
from .reward import (
    build_actual_training_text,
    canonicalize_completion_for_training,
    extract_presented_answer,
    normalize_generated_completion,
)


def extract_gsm8k_answer(answer_str: str) -> str:
    """
    GSM8K answers typically end with '#### 41'.
    We extract the integer after '####'. Fallback: last integer in string.
    """
    m = re.search(r"####\s*(-?\d+)", answer_str or "")
    if m:
        return m.group(1).strip()
    nums = re.findall(r"-?\d+", answer_str or "")
    return nums[-1].strip() if nums else (answer_str or "").strip()


def extract_aime_answer(answer_str: str) -> str:
    """
    AIME answers are integers, often zero-padded in solutions like \\boxed{033}.
    Normalize to a plain integer string.
    """
    m = re.search(r"\\boxed\{0*([0-9]+)\}", answer_str or "")
    if m:
        return str(int(m.group(1)))
    nums = re.findall(r"-?\d+", answer_str or "")
    if nums:
        return str(int(nums[-1]))
    return (answer_str or "").strip()


def _normalize_text(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def _strip_wrapping_math_delimiters(text: str) -> str:
    t = _normalize_text(text)
    if len(t) >= 2 and t[0] == "$" and t[-1] == "$":
        t = t[1:-1].strip()
    return t


def _extract_answer_text(raw_value) -> str:
    if raw_value is None:
        return ""
    text = str(raw_value)

    m = re.search(r"####\s*(-?\d+)", text)
    if m:
        return m.group(1).strip()

    normalized = normalize_generated_completion(text, answer_prefix="")
    presented = extract_presented_answer(text)
    if presented:
        return _normalize_text(presented)
    canonical = normalized.get("completion_canonical")
    if canonical:
        return _normalize_text(str(canonical))

    nums = re.findall(r"-?\d+", text)
    if nums:
        return nums[-1].strip()

    return _normalize_text(text)


def _normalize_completion_for_training(text: str, cfg) -> str:
    raw = _normalize_text(text)
    if not raw:
        return generation_answer_prefix(cfg).strip()

    prefix = generation_answer_prefix(cfg)
    normalized = normalize_generated_completion(raw, answer_prefix=prefix)
    canonical = normalized.get("completion_canonical") or canonicalize_completion_for_training(raw)
    if canonical:
        answer = _normalize_text(str(canonical))
        return f"{prefix} {answer}".strip() if prefix else answer
    return f"{prefix} {raw}".strip() if prefix else raw


def _percentile(values, pct: float) -> float:
    if not values:
        return 0.0
    vals = sorted(float(v) for v in values)
    if len(vals) == 1:
        return float(vals[0])
    pos = (pct / 100.0) * (len(vals) - 1)
    low = int(pos)
    high = min(low + 1, len(vals) - 1)
    weight = pos - low
    return float(vals[low] * (1 - weight) + vals[high] * weight)


def _length_summary(values):
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
    vals = [float(v) for v in values]
    return {
        "mean": float(sum(vals) / len(vals)),
        "p50": _percentile(vals, 50.0),
        "p90": _percentile(vals, 90.0),
        "max": float(max(vals)),
    }


def _resolve_actual_training_text(trajectory, cfg, target_contract: str) -> Optional[str]:
    contract = (target_contract or getattr(cfg, "actual_target_contract", "canonical") or "canonical").lower()
    if contract == "compare":
        raise ValueError("actual_target_contract=compare must be expanded before dataset construction.")

    cached_key = "actual_target_full_trace" if contract == "full_trace" else "actual_target_canonical"
    cached_value = trajectory.get(cached_key)
    if cached_value:
        return str(cached_value).strip() or None

    normalized = {
        "completion_raw": trajectory.get("completion_raw", trajectory.get("completion", "")),
        "completion_full_raw": trajectory.get("completion_full_raw", trajectory.get("completion", "")),
        "completion_first_line": trajectory.get("completion_first_line", ""),
        "completion_scored": trajectory.get("completion_scored", trajectory.get("completion", "")),
        "completion_canonical": trajectory.get("completion_canonical"),
    }
    fallback_completion = str(
        trajectory.get("completion_full_raw")
        or trajectory.get("completion_scored")
        or trajectory.get("completion")
        or ""
    )
    text = build_actual_training_text(
        normalized,
        fallback_completion=fallback_completion,
        target_contract=contract,
    )
    if text:
        return text

    if contract == "canonical":
        return _normalize_completion_for_training(fallback_completion, cfg)
    return fallback_completion.strip() or None


def _safe_first(example, keys):
    for key in keys:
        if key in example and example[key] not in (None, ""):
            return example[key]
    return None


def _parse_reward_model_ground_truth(reward_model_field):
    if reward_model_field is None:
        return None
    if isinstance(reward_model_field, dict):
        return reward_model_field.get("ground_truth")

    text = str(reward_model_field)
    for parser in (json.loads, ast.literal_eval):
        try:
            obj = parser(text)
            if isinstance(obj, dict) and "ground_truth" in obj:
                return obj["ground_truth"]
        except Exception:
            continue
    return None


def _infer_topic(prompt: str) -> str:
    p = (prompt or "").lower()
    if any(k in p for k in ("integral", "derivative", "limit", "calculus")):
        return "calculus"
    if any(k in p for k in ("matrix", "eigen", "linear algebra", "vector")):
        return "linear_algebra"
    if any(k in p for k in ("triangle", "circle", "geometry", "angle", "polygon")):
        return "geometry"
    if any(k in p for k in ("probability", "combinatorics", "permutation", "combination")):
        return "probability_combinatorics"
    if any(k in p for k in ("equation", "polynomial", "algebra")):
        return "algebra"
    if any(k in p for k in ("number theory", "mod", "prime", "divisible")):
        return "number_theory"
    return "other"


def _format_gsm8k_prompt(question: str) -> str:
    return (
        "Solve the math word problem.\n"
        "Return only the final integer.\n"
        f"Problem: {question}"
    )


def _format_general_math_prompt(question: str) -> str:
    return (
        "Solve the math problem.\n"
        "Return only the final answer.\n"
        f"Problem: {question}"
    )


def _format_aime_prompt(question: str) -> str:
    return (
        "Solve the AIME competition problem.\n"
        "Return only the final integer answer from 0 to 999.\n"
        f"Problem: {question}"
    )


def _resolve_dataset_spec(dataset_name: str, dataset_config_name: Optional[str] = None) -> Tuple[str, str, Optional[str]]:
    ds_name = (dataset_name or "gsm8k").lower()
    if ds_name == "gsm8k":
        return ds_name, "openai/gsm8k", dataset_config_name or "main"
    if ds_name == "deepscaler_preview":
        return ds_name, "agentica-org/DeepScaleR-Preview-Dataset", dataset_config_name or "default"
    if ds_name == "dapo_math_17k":
        return ds_name, "open-r1/DAPO-Math-17k-Processed", dataset_config_name or "all"
    if ds_name == "aime_2024":
        return ds_name, "Maxwell-Jia/AIME_2024", None
    if ds_name == "aime_2025":
        return ds_name, "opencompass/AIME2025", dataset_config_name or "all"
    raise ValueError(f"Unsupported dataset_name: {dataset_name}")


def _standardize_example(example, dataset_name: str, topic_split_mode: str = "none"):
    dataset_name = (dataset_name or "gsm8k").lower()

    if dataset_name == "gsm8k":
        question = _safe_first(example, ["question", "prompt", "problem"]) or ""
        answer = _safe_first(example, ["answer", "solution", "ground_truth"]) or ""
        prompt = _format_gsm8k_prompt(str(question))
        ground_truth = extract_gsm8k_answer(str(answer))
    elif dataset_name == "deepscaler_preview":
        question = _safe_first(example, ["problem", "prompt", "question"]) or ""
        answer = _safe_first(example, ["answer", "ground_truth", "solution"]) or ""
        prompt = _format_general_math_prompt(str(question))
        ground_truth = _strip_wrapping_math_delimiters(str(answer))
    elif dataset_name in {"aime_2024", "aime_2025"}:
        question = _safe_first(example, ["problem", "question", "Problem"]) or ""
        answer = _safe_first(example, ["answer", "Answer", "solution", "Solution"]) or ""
        prompt = _format_aime_prompt(str(question))
        ground_truth = extract_aime_answer(str(answer))
    else:
        # open-r1/DAPO-Math-17k-Processed default schema:
        # prompt, solution, data_source, source_prompt, ability, reward_model, extra_info
        prompt_val = _safe_first(example, ["prompt", "question", "problem", "source_prompt"]) or ""
        gt_val = _safe_first(example, ["solution", "answer", "ground_truth", "final_answer"])
        if gt_val is None:
            gt_val = _parse_reward_model_ground_truth(example.get("reward_model"))

        prompt = _format_general_math_prompt(str(prompt_val))
        ground_truth = _extract_answer_text(gt_val)

    row = {"prompt": prompt, "ground_truth": ground_truth}
    if (topic_split_mode or "none") == "heuristic_math_topics":
        row["topic"] = _infer_topic(prompt)
    return row


def _map_standardized(split, dataset_name: str, topic_split_mode: str = "none"):
    def _preprocess(example, idx):
        row = _standardize_example(example, dataset_name, topic_split_mode)
        row["__row_id__"] = int(idx)
        return row

    return split.map(
        _preprocess,
        with_indices=True,
        remove_columns=split.column_names,
        load_from_cache_file=False,
    )


def _choose_split(raw, preferred: str, fallback: str):
    if preferred in raw:
        return preferred
    if fallback in raw:
        return fallback
    return next(iter(raw.keys()))


def _resolve_eval_split(raw, preferred: Optional[str], fallback: Optional[str] = None, policy: str = "auto"):
    split_policy = (policy or "auto").lower()
    if preferred and preferred in raw:
        return preferred
    if fallback and fallback in raw:
        return fallback
    if split_policy in {"auto", "first_available"}:
        return next(iter(raw.keys()))
    raise RuntimeError(
        f"Requested eval split {preferred!r} not found and split policy '{split_policy}' "
        "does not allow falling back."
    )


def _dataset_cache_root_candidates() -> List[Path]:
    roots: List[Path] = []
    hub_cache = (os.environ.get("HF_HUB_CACHE") or "").strip()
    if hub_cache:
        roots.append(Path(hub_cache))
    hf_home = (os.environ.get("HF_HOME") or "").strip()
    if hf_home:
        roots.append(Path(hf_home) / "hub")
    roots.append(Path(".hf") / "hub")
    seen = set()
    unique: List[Path] = []
    for root in roots:
        resolved = root.expanduser()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        unique.append(resolved)
    return unique


def _dataset_snapshot_cached(dataset_path: str) -> bool:
    if "/" not in dataset_path:
        return False
    repo_dirname = f"datasets--{dataset_path.replace('/', '--')}"
    for root in _dataset_cache_root_candidates():
        repo_dir = root / repo_dirname
        if not repo_dir.exists():
            continue
        refs_main = repo_dir / "refs" / "main"
        snapshots = repo_dir / "snapshots"
        if refs_main.exists() or any(snapshots.glob("*")):
            return True
    return False


def _dataset_download_kwargs(dataset_path: str) -> Dict[str, object]:
    if os.environ.get("EXPDIS_HF_DATASETS_LOCAL_ONLY", "").strip().lower() in ("1", "true", "yes"):
        return {
            "download_config": DownloadConfig(local_files_only=True, max_retries=1),
            "download_mode": "reuse_dataset_if_exists",
        }
    if os.environ.get("EXPDIS_HF_DATASETS_LOCAL_IF_CACHED", "1").strip().lower() in ("0", "false", "no"):
        return {}
    if not _dataset_snapshot_cached(dataset_path):
        return {}
    print(f"[data] Using local HF dataset cache for {dataset_path}.")
    return {
        "download_config": DownloadConfig(local_files_only=True, max_retries=1),
        "download_mode": "reuse_dataset_if_exists",
    }


def _load_dataset_or_raise(dataset_path: str, dataset_config_name: Optional[str] = None):
    load_kwargs = _dataset_download_kwargs(dataset_path)
    try:
        if dataset_path == "opencompass/AIME2025" and (dataset_config_name is None or str(dataset_config_name).lower() == "all"):
            cfg_names = ("AIME2025-I", "AIME2025-II")
            splits = []
            for cfg_name in cfg_names:
                raw = load_dataset(dataset_path, cfg_name, **load_kwargs)
                split_name = "test" if "test" in raw else next(iter(raw.keys()))
                splits.append(raw[split_name])
            return DatasetDict({"test": concatenate_datasets(splits)})
        if dataset_config_name is None:
            return load_dataset(dataset_path, **load_kwargs)
        return load_dataset(dataset_path, dataset_config_name, **load_kwargs)
    except Exception as e:
        raise RuntimeError(
            "Failed to load Hugging Face dataset "
            f"'{dataset_path}' (config={dataset_config_name!r}). "
            "Check network/token access and ensure HF cache paths are writable "
            "(HF_HOME, HF_HUB_CACHE, HF_DATASETS_CACHE)."
        ) from e


def _topic_histogram(ds) -> Dict[str, int]:
    if "topic" not in ds.column_names:
        return {}
    counts = Counter(str(topic) for topic in ds["topic"])
    return dict(sorted(counts.items()))


def _select_limited(ds, max_examples: int):
    if max_examples is None or max_examples <= 0 or len(ds) <= max_examples:
        return ds
    return ds.select(range(max_examples))


def _derive_80_10_10_splits(ds, seed: int):
    shuffled = ds.shuffle(seed=seed)
    n = len(shuffled)
    if n <= 1:
        return shuffled, shuffled.select([])

    train_end = max(1, int(n * 0.8))
    eval_start = max(train_end, int(n * 0.9))
    if eval_start >= n:
        eval_start = max(train_end, n - max(1, n // 10))
    train_full = shuffled.select(range(train_end))
    eval_full = shuffled.select(range(eval_start, n))
    return train_full, eval_full


def _choose_holdout_topics(eval_ds, requested: Sequence[str], policy: str) -> List[str]:
    explicit = [str(topic) for topic in (requested or []) if str(topic).strip()]
    if explicit:
        return explicit
    if "topic" not in eval_ds.column_names:
        return []
    if (policy or "explicit_or_heuristic") != "largest_holdout":
        return []

    topic_counts = Counter(str(topic) for topic in eval_ds["topic"] if str(topic).strip())
    ranked = [topic for topic, _ in topic_counts.most_common() if topic != "other"]
    if not ranked:
        ranked = [topic for topic, _ in topic_counts.most_common()]
    return ranked[:1]


def build_round_shards(train_ds, num_rounds: int, seed: int, policy: str = "topic_balanced"):
    rounds = max(1, int(num_rounds))
    if rounds <= 1 or len(train_ds) <= 1:
        shard_manifest = [{"round_idx": 1, "num_examples": len(train_ds), "row_ids": list(train_ds["__row_id__"]) if "__row_id__" in train_ds.column_names else []}]
        return [train_ds], shard_manifest

    indices = list(range(len(train_ds)))
    rnd = random.Random(seed)
    shards: List[List[int]] = [[] for _ in range(rounds)]
    if policy == "topic_balanced" and "topic" in train_ds.column_names:
        by_topic: Dict[str, List[int]] = {}
        for idx, topic in enumerate(train_ds["topic"]):
            by_topic.setdefault(str(topic), []).append(idx)
        for topic_indices in by_topic.values():
            rnd.shuffle(topic_indices)
            for i, idx in enumerate(topic_indices):
                shards[i % rounds].append(idx)
    else:
        rnd.shuffle(indices)
        for i, idx in enumerate(indices):
            shards[i % rounds].append(idx)

    datasets = []
    manifest = []
    for round_idx, shard_indices in enumerate(shards, start=1):
        shard_indices = sorted(shard_indices)
        shard_ds = train_ds.select(shard_indices)
        datasets.append(shard_ds)
        manifest.append(
            {
                "round_idx": round_idx,
                "num_examples": len(shard_ds),
                "row_ids": [int(v) for v in shard_ds["__row_id__"]] if "__row_id__" in shard_ds.column_names else [],
                "topic_counts": _topic_histogram(shard_ds),
            }
        )
    return datasets, manifest


def build_math_dataset_bundle(
    dataset_name: str,
    max_train: int,
    max_eval: int,
    dataset_config_name: Optional[str] = None,
    eval_split: str = "test",
    dataset_split_train: str = "train",
    dataset_split_eval: str = "test",
    topic_split_mode: str = "none",
    dataset_split_policy: str = "auto",
    dataset_split_seed: int = 42,
    topic_partition_policy: str = "explicit_or_heuristic",
    requested_holdout_topics: Optional[Sequence[str]] = None,
):
    ds_name, dataset_path, dataset_config = _resolve_dataset_spec(dataset_name, dataset_config_name)

    raw = _load_dataset_or_raise(dataset_path, dataset_config)
    train_key = _choose_split(raw, dataset_split_train, "train")
    preferred_eval_key = dataset_split_eval if dataset_split_eval in raw else None
    if preferred_eval_key is None and ds_name == "gsm8k":
        preferred_eval_key = "train" if (eval_split or "test").lower() == "train" else _choose_split(raw, "test", "train")

    split_policy = (dataset_split_policy or "auto").lower()
    has_native_eval = preferred_eval_key is not None
    if split_policy == "auto":
        split_policy = "native" if has_native_eval else "derived_80_10_10"

    train_full = _map_standardized(raw[train_key], ds_name, topic_split_mode)
    if split_policy == "native":
        if preferred_eval_key is None:
            raise RuntimeError(
                f"Dataset '{dataset_name}' has no native eval split; "
                "use --dataset-split-policy derived_80_10_10 or auto."
            )
        eval_full = _map_standardized(raw[preferred_eval_key], ds_name, topic_split_mode)
        source_eval_split = preferred_eval_key
    else:
        train_full, eval_full = _derive_80_10_10_splits(train_full, dataset_split_seed)
        source_eval_split = "derived_test"

    train_ds = _select_limited(train_full, max_train)
    eval_ds = _select_limited(eval_full, max_eval)
    holdout_topics = _choose_holdout_topics(eval_ds, requested_holdout_topics or [], topic_partition_policy)

    manifest = {
        "dataset_name": ds_name,
        "dataset_path": dataset_path,
        "dataset_config_name": dataset_config,
        "split_policy": split_policy,
        "seed": int(dataset_split_seed),
        "source_train_split": train_key,
        "source_eval_split": source_eval_split,
        "topic_split_mode": topic_split_mode,
        "topic_partition_policy": topic_partition_policy,
        "requested_holdout_topics": [str(topic) for topic in (requested_holdout_topics or [])],
        "resolved_holdout_topics": holdout_topics,
        "train_count_full": len(train_full),
        "eval_count_full": len(eval_full),
        "train_count_selected": len(train_ds),
        "eval_count_selected": len(eval_ds),
        "train_row_ids": [int(v) for v in train_ds["__row_id__"]] if "__row_id__" in train_ds.column_names else [],
        "eval_row_ids": [int(v) for v in eval_ds["__row_id__"]] if "__row_id__" in eval_ds.column_names else [],
        "train_topic_counts": _topic_histogram(train_ds),
        "eval_topic_counts": _topic_histogram(eval_ds),
    }
    return train_ds, eval_ds, manifest


def build_math_eval_dataset_bundle(
    dataset_name: str,
    max_eval: int,
    dataset_config_name: Optional[str] = None,
    dataset_split_eval: str = "test",
    topic_split_mode: str = "none",
    dataset_split_policy: str = "auto",
):
    ds_name, dataset_path, dataset_config = _resolve_dataset_spec(dataset_name, dataset_config_name)
    raw = _load_dataset_or_raise(dataset_path, dataset_config)

    fallback_eval = "test"
    if ds_name in {"aime_2024", "dapo_math_17k"}:
        fallback_eval = "train"

    eval_key = _resolve_eval_split(
        raw,
        preferred=dataset_split_eval,
        fallback=fallback_eval,
        policy=dataset_split_policy,
    )
    eval_full = _map_standardized(raw[eval_key], ds_name, topic_split_mode)
    eval_ds = _select_limited(eval_full, max_eval)

    manifest = {
        "dataset_name": ds_name,
        "dataset_path": dataset_path,
        "dataset_config_name": dataset_config,
        "split_policy": (dataset_split_policy or "auto").lower(),
        "source_eval_split": eval_key,
        "topic_split_mode": topic_split_mode,
        "eval_count_full": len(eval_full),
        "eval_count_selected": len(eval_ds),
        "eval_row_ids": [int(v) for v in eval_ds["__row_id__"]] if "__row_id__" in eval_ds.column_names else [],
        "eval_topic_counts": _topic_histogram(eval_ds),
    }
    return eval_ds, manifest


def build_math_datasets(
    dataset_name: str,
    max_train: int,
    max_eval: int,
    dataset_config_name: Optional[str] = None,
    eval_split: str = "test",
    dataset_split_train: str = "train",
    dataset_split_eval: str = "test",
    topic_split_mode: str = "none",
    dataset_split_policy: str = "auto",
    dataset_split_seed: int = 42,
    topic_partition_policy: str = "explicit_or_heuristic",
    requested_holdout_topics: Optional[Sequence[str]] = None,
):
    """
    Build standardized train/eval datasets with columns:
      - prompt
      - ground_truth
      - topic (optional)
    """
    train_ds, eval_ds, _ = build_math_dataset_bundle(
        dataset_name=dataset_name,
        max_train=max_train,
        max_eval=max_eval,
        dataset_config_name=dataset_config_name,
        eval_split=eval_split,
        dataset_split_train=dataset_split_train,
        dataset_split_eval=dataset_split_eval,
        topic_split_mode=topic_split_mode,
        dataset_split_policy=dataset_split_policy,
        dataset_split_seed=dataset_split_seed,
        topic_partition_policy=topic_partition_policy,
        requested_holdout_topics=requested_holdout_topics,
    )
    return train_ds, eval_ds



def _trajectory_is_correct(trajectory) -> bool:
    if "is_correct" in trajectory:
        return bool(trajectory.get("is_correct", False))
    return bool(float(trajectory.get("r_correct", 0.0) or 0.0) > 0.5)


def _trajectory_format_quality(trajectory) -> float:
    return float(
        trajectory.get("format_quality", trajectory.get("reward_format_quality", 0.0)) or 0.0
    )


def _trajectory_is_high_quality(trajectory, min_quality: float = 0.75) -> bool:
    return (
        bool(trajectory.get("valid_answer_presentation", False) or trajectory.get("completion_canonical"))
        and not bool(trajectory.get("clipped", False))
        and _trajectory_format_quality(trajectory) >= float(min_quality)
        and not bool(trajectory.get("has_multiple_choice", False))
        and not bool(trajectory.get("has_markdown", False))
        and not bool(trajectory.get("mentions_instruction", False))
        and not bool(trajectory.get("starts_with_instruction_echo", False))
    )


def select_accepted_trajectories(trajectories):
    rows = list(trajectories or [])
    correct = [
        t for t in rows
        if _trajectory_is_correct(t)
    ]
    strict = [
        t for t in correct
        if _trajectory_is_high_quality(t, min_quality=0.75)
    ]
    relaxed = [
        t for t in correct
        if bool(t.get("valid_answer_presentation", False) or t.get("completion_canonical"))
        and not bool(t.get("clipped", False))
        and _trajectory_format_quality(t) >= 0.45
        and not bool(t.get("has_multiple_choice", False))
    ]

    if strict:
        selected = strict
        policy = "strict_format_correct"
    elif relaxed:
        selected = relaxed
        policy = "relaxed_correct"
    else:
        selected = correct
        policy = "correct_only"

    info = {
        "total": len(rows),
        "correct": len(correct),
        "strict": len(strict),
        "relaxed": len(relaxed),
        "accepted": len(selected),
        "policy": policy,
    }
    return selected, info


def choose_actual_bootstrap_trajectories(
    accepted_trajectories,
    all_trajectories,
    min_examples: int,
    seed: int = 42,
):
    accepted = list(accepted_trajectories or [])
    all_rows = list(all_trajectories or [])
    min_required = max(1, int(min_examples))

    if len(accepted) >= min_required:
        selected = [dict(row) for row in accepted]
        policy = "accepted_only"
        reason = "accepted_meets_minimum"
        oversampled_count = 0
    elif accepted:
        rnd = random.Random(seed)
        selected = [dict(row) for row in accepted]
        while len(selected) < min_required:
            sampled = dict(rnd.choice(accepted))
            sampled["_oversampled_from_accepted"] = True
            selected.append(sampled)
        policy = "accepted_oversampled"
        reason = "accepted_below_minimum"
        oversampled_count = max(0, len(selected) - len(accepted))
    elif all_rows:
        selected = [dict(row) for row in all_rows]
        policy = "all_trajectories_fallback"
        reason = "no_accepted_trajectories"
        oversampled_count = 0
    else:
        selected = []
        policy = "accepted_only"
        reason = "no_trajectories"
        oversampled_count = 0

    info = {
        "policy": policy,
        "reason": reason,
        "accepted_count": int(len(accepted)),
        "selected_count": int(len(selected)),
        "total_count": int(len(all_rows)),
        "min_required": int(min_required),
        "oversampled_count": int(oversampled_count),
        "used_fallback": bool(policy == "all_trajectories_fallback"),
    }
    return selected, info


def choose_actual_policy_trajectories(
    accepted_trajectories,
    all_trajectories,
    policy: str = "accepted_balanced",
    max_incorrect_ratio: float = 1.0,
    seed: int = 42,
    quality_floor: float = 0.75,
):
    policy_key = str(policy or "accepted_balanced").lower()
    all_rows = [dict(row) for row in (all_trajectories or [])]
    accepted = [dict(row) for row in (accepted_trajectories or [])]
    correct = [dict(row) for row in all_rows if _trajectory_is_correct(row)]
    incorrect = [dict(row) for row in all_rows if not _trajectory_is_correct(row)]
    high_quality_incorrect = [
        dict(row) for row in incorrect if _trajectory_is_high_quality(row, min_quality=quality_floor)
    ]

    if policy_key == "all":
        return all_rows, {
            "policy": "all",
            "reason": "explicit_all",
            "positive_source": "all",
            "selected_count": int(len(all_rows)),
            "positive_count": int(len(correct)),
            "incorrect_selected_count": int(len(incorrect)),
            "high_quality_incorrect_available": int(len(high_quality_incorrect)),
            "incorrect_available": int(len(incorrect)),
            "max_incorrect_ratio": float(max_incorrect_ratio),
            "used_fallback": False,
        }

    if policy_key in {"accepted_balanced", "accepted_only"}:
        positive_pool = accepted if accepted else correct
        positive_source = "accepted" if accepted else ("correct" if correct else "none")
    elif policy_key in {"correct_balanced", "correct_only"}:
        positive_pool = correct
        positive_source = "correct" if correct else "none"
    else:
        raise ValueError(f"Unsupported actual dataset policy: {policy}")

    if not positive_pool:
        return all_rows, {
            "policy": "all_trajectories_fallback",
            "reason": "no_positive_trajectories",
            "positive_source": "none",
            "selected_count": int(len(all_rows)),
            "positive_count": 0,
            "incorrect_selected_count": int(len(incorrect)),
            "high_quality_incorrect_available": int(len(high_quality_incorrect)),
            "incorrect_available": int(len(incorrect)),
            "max_incorrect_ratio": float(max_incorrect_ratio),
            "used_fallback": bool(all_rows),
        }

    selected = [dict(row) for row in positive_pool]
    incorrect_selected = []
    if policy_key.endswith("_balanced"):
        incorrect_budget = max(0, int(round(float(max_incorrect_ratio) * len(positive_pool))))
        if incorrect_budget > 0 and high_quality_incorrect:
            rnd = random.Random(seed)
            ranked = sorted(
                high_quality_incorrect,
                key=lambda row: (
                    _trajectory_format_quality(row),
                    float(row.get("r_novel", 0.0) or 0.0),
                    float(row.get("reward", 0.0) or 0.0),
                    rnd.random(),
                ),
                reverse=True,
            )
            incorrect_selected = [dict(row) for row in ranked[:incorrect_budget]]
            selected.extend(incorrect_selected)

    info = {
        "policy": policy_key,
        "reason": "balanced_from_positive_pool" if policy_key.endswith("_balanced") else "positive_only",
        "positive_source": positive_source,
        "selected_count": int(len(selected)),
        "positive_count": int(len(positive_pool)),
        "incorrect_selected_count": int(len(incorrect_selected)),
        "high_quality_incorrect_available": int(len(high_quality_incorrect)),
        "incorrect_available": int(len(incorrect)),
        "max_incorrect_ratio": float(max_incorrect_ratio),
        "used_fallback": False,
    }
    return selected, info


def build_supervised_dataset_from_trajectories(trajectories,
                                               tokenizer,
                                               cfg,
                                               max_examples: int,
                                               max_total_len: int,
                                               target_contract: Optional[str] = None):
    """
    Keep only CORRECT trajectories and build a TensorDataset for distillation.
    If there are 0 correct trajectories, fall back to top-K highest reward.
    """
    good = [
        t for t in trajectories
        if (t.get("is_correct") if "is_correct" in t else (t["r_correct"] > 0.5))
    ]
    print(f"Correct trajectories: {len(good)}")

    fallback_used = False
    if len(good) == 0:
        print(
            "[WARN] No correct trajectories found with r_correct > 0.5.\n"
            "       Falling back to top-K highest-reward trajectories for distillation."
        )
        sorted_traj = sorted(trajectories, key=lambda t: t["reward"], reverse=True)
        good = sorted_traj[:min(max_examples, len(sorted_traj))]
        fallback_used = True
    else:
        if len(good) > max_examples:
            good = random.sample(good, max_examples)

    target_contract = (target_contract or getattr(cfg, "actual_target_contract", "canonical") or "canonical").lower()
    filtered = []
    texts = []
    prompt_lens = []
    full_lens = []
    target_lens = []
    raw_prompt_lens = []
    raw_full_lens = []
    dropped_missing_target = 0
    dropped_empty_completion = 0
    for trajectory in good:
        completion_text = _resolve_actual_training_text(trajectory, cfg, target_contract)
        if not completion_text:
            dropped_missing_target += 1
            continue
        rendered_prompt = render_supervised_prompt(
            tokenizer,
            trajectory["prompt"],
            cfg,
            target_contract=target_contract,
        )
        rendered_text = render_supervised_text(
            tokenizer,
            trajectory["prompt"],
            completion_text,
            cfg,
            target_contract=target_contract,
        )
        # TPU runs monkey-patch tokenizer() to fixed-width padding for stable XLA shapes.
        # Explicitly disable padding/truncation for boundary measurement so prompt/completion
        # masking still sees the true token delta after chat templating.
        prompt_ids = tokenizer(rendered_prompt, padding=False, truncation=False).input_ids
        full_ids = tokenizer(rendered_text, padding=False, truncation=False).input_ids
        target_ids = tokenizer(completion_text, padding=False, truncation=False).input_ids
        raw_prompt_len = len(prompt_ids)
        raw_full_len = len(full_ids)
        prompt_len = min(len(prompt_ids), max_total_len)
        full_len = min(len(full_ids), max_total_len)
        if full_len <= prompt_len:
            dropped_empty_completion += 1
            continue
        filtered.append(trajectory)
        texts.append(rendered_text)
        prompt_lens.append(prompt_len)
        full_lens.append(full_len)
        target_lens.append(len(target_ids))
        raw_prompt_lens.append(raw_prompt_len)
        raw_full_lens.append(raw_full_len)

    if not filtered:
        raise RuntimeError(
            "No distillation examples survived truncation; increase --max-total-len "
            "or reduce prompt/completion lengths."
        )

    # Use right-padding to max_length for SFT: flash attention (Pallas) ignores
    # attention_mask and relies on causal mask only.  Right-padding is safe because
    # causal mask prevents attending to future (padding) positions.
    # padding="max_length" ensures all sequences are exactly max_total_len tokens,
    # giving one XLA compilation and guaranteed 1024-block alignment for Pallas.
    _orig_pad_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "right"
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_total_len,
    )
    tokenizer.padding_side = _orig_pad_side

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    labels = input_ids.clone()
    labels = labels.masked_fill(attention_mask == 0, -100)

    # Mask prompt tokens: we only train on completion tokens.
    # Right-padding: real tokens at start, padding at end.  No left-pad offset.
    seq_width = labels.size(1)
    for i, (plen, full_len) in enumerate(zip(prompt_lens, full_lens)):
        full_len = min(int(full_len), seq_width)
        plen = min(int(plen), full_len)
        labels[i, :plen] = -100  # ignore_index for CE

    dataset = torch.utils.data.TensorDataset(input_ids, attention_mask, labels)
    metadata = {
        "target_contract": target_contract,
        "source_total": int(len(trajectories)),
        "candidate_count": int(len(good)),
        "survived_count": int(len(filtered)),
        "survival_fraction": float(len(filtered) / max(1, len(good))),
        "fallback_used": bool(fallback_used),
        "dropped_missing_target": int(dropped_missing_target),
        "dropped_empty_completion": int(dropped_empty_completion),
        "prompt_length": _length_summary(raw_prompt_lens),
        "rendered_total_length": _length_summary(raw_full_lens),
        "target_length": _length_summary(target_lens),
        "prompt_truncated_fraction": float(sum(1 for v in raw_prompt_lens if int(v) > int(max_total_len)) / max(1, len(raw_prompt_lens))),
        "rendered_total_truncated_fraction": float(sum(1 for v in raw_full_lens if int(v) > int(max_total_len)) / max(1, len(raw_full_lens))),
    }
    return dataset, metadata


def build_rewarded_dataset_from_trajectories(trajectories,
                                             tokenizer,
                                             cfg,
                                             max_examples: Optional[int],
                                             max_total_len: int,
                                             reward_key: str = "reward",
                                             target_correct_fraction: float = 0.0,
                                             seed: int = 42,
                                             target_contract: Optional[str] = None):
    """
    Build an offline RL dataset from Explorer trajectories without filtering by correctness.
    Returns TensorDataset(input_ids, attention_mask, labels, rewards, r_correct, r_novel, is_correct).
    """
    trajectories = list(trajectories)
    source_total = int(len(trajectories))
    correct = [t for t in trajectories if bool(t.get("is_correct", False))]
    incorrect = [t for t in trajectories if not bool(t.get("is_correct", False))]

    target_correct_fraction = float(max(0.0, min(0.999, target_correct_fraction)))
    if target_correct_fraction > 0.0 and correct:
        current_fraction = len(correct) / max(1, len(trajectories))
        if current_fraction < target_correct_fraction:
            rnd = random.Random(seed)
            extra_needed = int(
                max(
                    0,
                    round(
                        (target_correct_fraction * len(trajectories) - len(correct))
                        / max(1e-6, 1.0 - target_correct_fraction)
                    ),
                )
            )
            if extra_needed > 0:
                trajectories.extend(dict(rnd.choice(correct)) for _ in range(extra_needed))
                correct = [t for t in trajectories if bool(t.get("is_correct", False))]
                incorrect = [t for t in trajectories if not bool(t.get("is_correct", False))]
                print(
                    "[actual-dataset] Oversampled correct trajectories "
                    f"from {current_fraction:.3f} to {len(correct) / max(1, len(trajectories)):.3f} "
                    f"(target={target_correct_fraction:.3f})."
                )

    if max_examples is not None and max_examples > 0 and len(trajectories) > max_examples:
        rnd = random.Random(seed)
        if target_correct_fraction > 0.0 and correct:
            desired_correct = min(len(correct), int(round(target_correct_fraction * max_examples)))
            desired_correct = max(1, desired_correct)
            desired_incorrect = max(0, max_examples - desired_correct)
            selected = []
            selected.extend(rnd.sample(correct, min(len(correct), desired_correct)))
            if incorrect and desired_incorrect > 0:
                selected.extend(rnd.sample(incorrect, min(len(incorrect), desired_incorrect)))
            if len(selected) < max_examples:
                remaining = [t for t in trajectories if t not in selected]
                if remaining:
                    selected.extend(rnd.sample(remaining, min(len(remaining), max_examples - len(selected))))
            trajectories = selected[:max_examples]
        else:
            trajectories = rnd.sample(trajectories, max_examples)

    target_contract = (target_contract or getattr(cfg, "actual_target_contract", "canonical") or "canonical").lower()
    filtered = []
    texts = []
    prompt_lens = []
    full_lens = []
    target_lens = []
    raw_prompt_lens = []
    raw_full_lens = []
    dropped_missing_target = 0
    dropped_empty_completion = 0
    for trajectory in trajectories:
        completion_text = _resolve_actual_training_text(trajectory, cfg, target_contract)
        if not completion_text:
            dropped_missing_target += 1
            continue
        rendered_prompt = render_supervised_prompt(
            tokenizer,
            trajectory["prompt"],
            cfg,
            target_contract=target_contract,
        )
        rendered_text = render_supervised_text(
            tokenizer,
            trajectory["prompt"],
            completion_text,
            cfg,
            target_contract=target_contract,
        )
        prompt_ids = tokenizer(rendered_prompt, padding=False, truncation=False).input_ids
        full_ids = tokenizer(rendered_text, padding=False, truncation=False).input_ids
        target_ids = tokenizer(completion_text, padding=False, truncation=False).input_ids
        raw_prompt_len = len(prompt_ids)
        raw_full_len = len(full_ids)
        prompt_len = min(len(prompt_ids), max_total_len)
        full_len = min(len(full_ids), max_total_len)
        if full_len <= prompt_len:
            dropped_empty_completion += 1
            continue
        filtered.append(trajectory)
        texts.append(rendered_text)
        prompt_lens.append(prompt_len)
        full_lens.append(full_len)
        target_lens.append(len(target_ids))
        raw_prompt_lens.append(raw_prompt_len)
        raw_full_lens.append(raw_full_len)

    trajectories = filtered
    if not trajectories:
        raise RuntimeError(
            "No offline-GRPO trajectory examples survived truncation; increase --max-total-len "
            "or reduce prompt/completion lengths."
        )

    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_total_len,
    )

    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    labels = input_ids.clone()
    labels = labels.masked_fill(attention_mask == 0, -100)

    seq_width = labels.size(1)
    for i, (plen, full_len) in enumerate(zip(prompt_lens, full_lens)):
        full_len = min(int(full_len), seq_width)
        plen = min(int(plen), full_len)
        left_pad = max(0, seq_width - full_len)
        prompt_end = min(seq_width, left_pad + plen)
        labels[i, :prompt_end] = -100

    rewards = torch.tensor(
        [float(t.get(reward_key, 0.0)) for t in trajectories],
        dtype=torch.float,
    )
    r_correct = torch.tensor(
        [float(t.get("r_correct", 0.0)) for t in trajectories],
        dtype=torch.float,
    )
    r_novel = torch.tensor(
        [float(t.get("r_novel", 0.0)) for t in trajectories],
        dtype=torch.float,
    )
    is_correct = torch.tensor(
        [1.0 if t.get("is_correct", False) else 0.0 for t in trajectories],
        dtype=torch.float,
    )

    dataset = torch.utils.data.TensorDataset(
        input_ids, attention_mask, labels, rewards, r_correct, r_novel, is_correct
    )
    metadata = {
        "target_contract": target_contract,
        "source_total": int(source_total),
        "pool_total_after_oversample": int(len(correct) + len(incorrect)),
        "selected_count": int(len(trajectories)),
        "survived_count": int(len(filtered)),
        "survival_fraction": float(len(filtered) / max(1, len(trajectories))),
        "input_correct_fraction": float(len(correct) / max(1, len(correct) + len(incorrect))),
        "selected_correct_fraction": float(sum(1 for t in trajectories if bool(t.get("is_correct", False))) / max(1, len(trajectories))),
        "target_correct_fraction": float(target_correct_fraction),
        "dropped_missing_target": int(dropped_missing_target),
        "dropped_empty_completion": int(dropped_empty_completion),
        "reward_mean": float(sum(float(t.get(reward_key, 0.0)) for t in trajectories) / max(1, len(trajectories))),
        "prompt_length": _length_summary(raw_prompt_lens),
        "rendered_total_length": _length_summary(raw_full_lens),
        "target_length": _length_summary(target_lens),
        "prompt_truncated_fraction": float(sum(1 for v in raw_prompt_lens if int(v) > int(max_total_len)) / max(1, len(raw_prompt_lens))),
        "rendered_total_truncated_fraction": float(sum(1 for v in raw_full_lens if int(v) > int(max_total_len)) / max(1, len(raw_full_lens))),
    }
    return dataset, metadata
