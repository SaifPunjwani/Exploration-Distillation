#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import requests
from transformers import AutoConfig, AutoTokenizer

from expdis_torch.data import build_math_eval_dataset_bundle
from expdis_torch.eval_plot import _pass_at_k_estimate
from expdis_torch.prompting import (
    build_generation_messages,
    generation_answer_prefix,
    render_generation_prompt,
    uses_chat_template,
)
from expdis_torch.reward import (
    analyze_completion_text,
    extract_presented_answer,
    is_correct_completion,
    normalize_generated_completion,
)
from expdis_torch.wandb_metrics import filter_wandb_payload, wandb_minimal_mode

try:
    import wandb  # type: ignore
except Exception:
    wandb = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark AIME rollouts against a vLLM server.")
    parser.add_argument("--server-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--served-model-name", default="")
    parser.add_argument("--tokenizer-name", default="")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--role-name", default="base")
    parser.add_argument("--dataset-name", default="aime_2024")
    parser.add_argument("--dataset-config-name", default="")
    parser.add_argument("--dataset-split-policy", default="first_available")
    parser.add_argument("--max-benchmark-examples", type=int, default=30)
    parser.add_argument("--problem-shard-index", type=int, default=0)
    parser.add_argument("--problem-shard-count", type=int, default=1)
    parser.add_argument("--num-rollouts", type=int, default=32)
    parser.add_argument("--rollout-temperature", type=float, default=1.0)
    parser.add_argument("--rollout-top-p", type=float, default=1.0)
    parser.add_argument("--rollout-top-k", type=int, default=0)
    parser.add_argument("--rollout-max-completion-len", type=int, default=16384)
    parser.add_argument("--pass-k", default="1,4,8,16,32")
    parser.add_argument("--request-concurrency", type=int, default=32)
    parser.add_argument("--request-timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--max-prompt-len", type=int, default=2048)
    parser.add_argument("--eval-response-style", default="reasoning_final")
    parser.add_argument("--answer-constraint-mode", default="none")
    parser.add_argument("--thinking-mode", default="auto", choices=["auto", "think", "no_think"])
    parser.add_argument("--wandb-project", default="expdis")
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb-group", default="")
    parser.add_argument("--use-wandb", action="store_true")
    return parser.parse_args()


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _load_json_if_exists(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return default


def _write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _cfg(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        dataset_name=args.dataset_name,
        eval_response_style=args.eval_response_style,
        exploration_response_style=args.eval_response_style,
        answer_constraint_mode=args.answer_constraint_mode,
        eval_answer_constraint_mode=args.answer_constraint_mode,
        exploration_answer_constraint_mode=args.answer_constraint_mode,
        eval_thinking_mode=args.thinking_mode,
        exploration_thinking_mode=args.thinking_mode,
        thinking_mode=args.thinking_mode,
        max_completion_len=args.rollout_max_completion_len,
        eval_max_completion_len=args.rollout_max_completion_len,
        max_prompt_len=args.max_prompt_len,
        actual_target_contract="full_trace",
    )


def _pass_k_values(spec: str, num_rollouts: int) -> list[int]:
    values: list[int] = []
    for chunk in str(spec or "").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            value = int(chunk)
        except Exception:
            continue
        if value > 0:
            values.append(min(value, max(1, int(num_rollouts))))
    values = sorted(set(values))
    if 1 not in values:
        values.insert(0, 1)
    return values


def _post_generation(
    *,
    server_base_url: str,
    model_name: str,
    prompt: str | None,
    messages: list[dict[str, str]] | None,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    use_chat = bool(messages)
    endpoint = "/chat/completions" if use_chat else "/completions"
    payload: dict[str, Any] = {
        "model": model_name,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "n": 1,
        "stream": False,
    }
    if int(top_k) > 0:
        payload["top_k"] = int(top_k)
    if use_chat:
        payload["messages"] = messages
        payload["max_completion_tokens"] = int(max_tokens)
    else:
        payload["prompt"] = prompt
        payload["max_tokens"] = int(max_tokens)
    response = requests.post(
        f"{server_base_url.rstrip('/')}{endpoint}",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=timeout_seconds,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"vLLM request failed with status {response.status_code}: {response.text[:2000]}")
    payload = response.json()
    choices = payload.get("choices") or []
    if not choices:
        raise RuntimeError("vLLM response had no choices")
    return payload


@dataclass
class RolloutResult:
    rollout_idx: int
    raw_text: str
    latency_seconds: float


def _sample_rollouts(
    *,
    server_base_url: str,
    served_model_name: str,
    prompt: str,
    messages: list[dict[str, str]] | None,
    num_rollouts: int,
    concurrency: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    timeout_seconds: float,
) -> list[RolloutResult]:
    results: list[RolloutResult] = []

    def _one(rollout_idx: int) -> RolloutResult:
        start = time.time()
        payload = _post_generation(
            server_base_url=server_base_url,
            model_name=served_model_name,
            prompt=prompt,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            timeout_seconds=timeout_seconds,
        )
        latency = time.time() - start
        choice = (payload.get("choices") or [{}])[0]
        if messages:
            text = str(((choice.get("message") or {}).get("content")) or "")
        else:
            text = str(choice.get("text", ""))
        return RolloutResult(rollout_idx=rollout_idx, raw_text=text, latency_seconds=latency)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(concurrency))) as pool:
        futures = [pool.submit(_one, rollout_idx) for rollout_idx in range(int(num_rollouts))]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda item: item.rollout_idx)
    return results


def _apply_thinking_mode_switch(
    prompt: str,
    messages: list[dict[str, str]] | None,
    thinking_mode: str,
) -> tuple[str, list[dict[str, str]] | None]:
    mode = str(thinking_mode or "auto").strip().lower()
    if mode not in {"think", "no_think"}:
        return prompt, messages

    switch = "/think" if mode == "think" else "/no_think"

    if messages:
        adjusted_messages: list[dict[str, str]] = []
        applied = False
        for message in messages:
            cloned = dict(message)
            if not applied and str(cloned.get("role", "")).lower() == "user":
                content = str(cloned.get("content", "") or "")
                stripped = content.lstrip()
                if not stripped.startswith("/think") and not stripped.startswith("/no_think"):
                    cloned["content"] = f"{switch}\n{content}".strip()
                applied = True
            adjusted_messages.append(cloned)
        return prompt, adjusted_messages

    stripped_prompt = str(prompt or "").lstrip()
    if stripped_prompt.startswith("/think") or stripped_prompt.startswith("/no_think"):
        return prompt, messages
    return f"{switch}\n{prompt}".strip(), messages


def _summarize_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {"count": 0}
    correct = [1.0 if bool(row.get("is_correct", False)) else 0.0 for row in samples]
    prompt_lengths = [float(row.get("prompt_token_length", 0) or 0.0) for row in samples]
    completion_lengths = [float(row.get("completion_token_length", 0) or 0.0) for row in samples]
    latencies = [float(row.get("latency_seconds", 0.0) or 0.0) for row in samples]
    valid_answer = [1.0 if bool(row.get("valid_answer_presentation", False)) else 0.0 for row in samples]
    return {
        "count": int(len(samples)),
        "correct_fraction": _mean(correct),
        "mean_prompt_token_length": _mean(prompt_lengths),
        "mean_completion_token_length": _mean(completion_lengths),
        "valid_answer_fraction": _mean(valid_answer),
        "mean_latency_seconds": _mean(latencies),
        "examples": {
            "correct": [row for row in samples if bool(row.get("is_correct", False))][:3],
            "incorrect": [row for row in samples if not bool(row.get("is_correct", False))][:3],
        },
    }


def _build_summary_payload(
    *,
    args: argparse.Namespace,
    role: str,
    manifest: dict[str, Any],
    served_model_name: str,
    pass_k_values: list[int],
    samples: list[dict[str, Any]],
    problem_summaries: list[dict[str, Any]],
    benchmark_rollout_path: str,
    rollout_latencies: list[float],
) -> dict[str, Any]:
    total_correct = int(sum(1 for row in samples if bool(row.get("is_correct", False))))
    total_rollouts = int(len(samples))
    accuracy = float(total_correct / max(1, total_rollouts))
    num_rollouts = int(args.num_rollouts)
    pass_at_k = {
        str(k): float(
            _mean([_pass_at_k_estimate(num_rollouts, row["num_correct"], int(k)) for row in problem_summaries])
        )
        for k in pass_k_values
    }
    primary_metric_name = f"avg@{num_rollouts}"
    return {
        "dataset_name": args.dataset_name,
        "dataset_config_name": args.dataset_config_name or manifest.get("dataset_config_name"),
        "manifest": manifest,
        "protocol": {
            "eval_mode": "sampled",
            "num_rollouts": int(args.num_rollouts),
            "rollout_temperature": float(args.rollout_temperature),
            "rollout_top_p": float(args.rollout_top_p),
            "rollout_top_k": int(args.rollout_top_k),
            "rollout_max_completion_len": int(args.rollout_max_completion_len),
            "thinking_mode": args.thinking_mode,
            "pass_k_values": pass_k_values,
            "request_concurrency": int(args.request_concurrency),
            "server_base_url": args.server_base_url,
            "served_model_name": served_model_name,
        },
        "base": _summarize_samples(samples) if role == "base" else {},
        "explorer": _summarize_samples(samples) if role == "explorer" else {},
        "actual": _summarize_samples(samples) if role == "actual" else {},
        "accuracies": {
            "base": accuracy if role == "base" else None,
            "explorer": accuracy if role == "explorer" else None,
            "actual": accuracy if role == "actual" else None,
        },
        "rollout_accuracy": {
            "base": accuracy if role == "base" else None,
            "explorer": accuracy if role == "explorer" else None,
            "actual": accuracy if role == "actual" else None,
        },
        "primary_metrics": {
            "base": primary_metric_name if role == "base" else None,
            "explorer": primary_metric_name if role == "explorer" else None,
            "actual": primary_metric_name if role == "actual" else None,
        },
        "pass_at_k": {
            "base": pass_at_k if role == "base" else {},
            "explorer": pass_at_k if role == "explorer" else {},
            "actual": pass_at_k if role == "actual" else {},
        },
        "problem_summaries": {
            "base": problem_summaries if role == "base" else [],
            "explorer": problem_summaries if role == "explorer" else [],
            "actual": problem_summaries if role == "actual" else [],
        },
        "rollout_json_paths": {
            role: benchmark_rollout_path,
        },
        "mean_latency_seconds": _mean(rollout_latencies),
    }


def main() -> None:
    args = _parse_args()
    cfg = _cfg(args)
    pass_k_values = _pass_k_values(args.pass_k, args.num_rollouts)
    served_model_name = args.served_model_name or args.model_name
    tokenizer_name = args.tokenizer_name or args.model_name

    run_dir = os.path.join("runs", args.run_name)
    analysis_dir = os.path.join(run_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    model_config = AutoConfig.from_pretrained(tokenizer_name, trust_remote_code=True)
    model_context_len = getattr(model_config, "max_position_embeddings", None)
    use_chat_api = uses_chat_template(tokenizer)
    eval_ds, manifest = build_math_eval_dataset_bundle(
        dataset_name=args.dataset_name,
        max_eval=args.max_benchmark_examples,
        dataset_config_name=(args.dataset_config_name or None),
        dataset_split_policy=args.dataset_split_policy,
    )
    shard_count = max(1, int(args.problem_shard_count))
    shard_index = int(args.problem_shard_index)
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"problem shard index must be in [0, {shard_count}), got {shard_index}")
    selected_problem_indices = list(range(len(eval_ds)))
    if shard_count > 1:
        selected_problem_indices = [
            idx for idx in selected_problem_indices if (idx % shard_count) == shard_index
        ]
        eval_ds = eval_ds.select(selected_problem_indices)
    manifest = dict(manifest)
    manifest["problem_shard_index"] = shard_index
    manifest["problem_shard_count"] = shard_count
    manifest["selected_problem_indices"] = list(selected_problem_indices)
    manifest["eval_count_selected"] = int(len(selected_problem_indices))

    wandb_run = None
    if args.use_wandb:
        if wandb is None:
            raise RuntimeError("wandb is not installed in the selected Python environment")
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            group=args.wandb_group or None,
            name=args.run_name,
            config={
                "model_name": args.model_name,
                "served_model_name": served_model_name,
                "dataset_name": args.dataset_name,
                "dataset_config_name": args.dataset_config_name,
                "problem_shard_index": shard_index,
                "problem_shard_count": shard_count,
                "num_rollouts": int(args.num_rollouts),
                "rollout_temperature": float(args.rollout_temperature),
                "rollout_top_p": float(args.rollout_top_p),
                "rollout_top_k": int(args.rollout_top_k),
                "rollout_max_completion_len": int(args.rollout_max_completion_len),
                "thinking_mode": args.thinking_mode,
                "request_concurrency": int(args.request_concurrency),
                "server_base_url": args.server_base_url,
            },
        )

    role = args.role_name
    benchmark_rollout_path = os.path.abspath(os.path.join(analysis_dir, f"benchmark_rollouts_{role}.json"))
    benchmark_eval_samples_path = os.path.abspath(os.path.join(analysis_dir, "benchmark_eval_samples.json"))
    benchmark_summary_path = os.path.abspath(os.path.join(analysis_dir, "benchmark_summary.json"))

    existing_rollouts = _load_json_if_exists(benchmark_rollout_path, {})
    if not isinstance(existing_rollouts, dict):
        existing_rollouts = {}
    existing_eval_samples = _load_json_if_exists(benchmark_eval_samples_path, {})
    existing_samples = existing_eval_samples.get(role, []) if isinstance(existing_eval_samples, dict) else []
    if not isinstance(existing_samples, list):
        existing_samples = []
    existing_summary = _load_json_if_exists(benchmark_summary_path, {})
    existing_problem_summaries = {}
    if isinstance(existing_summary, dict):
        existing_problem_summaries = (existing_summary.get("problem_summaries") or {}).get(role, [])
    if not isinstance(existing_problem_summaries, list):
        existing_problem_summaries = []

    samples: list[dict[str, Any]] = [
        row for row in existing_samples if int(row.get("problem_idx", -1)) in selected_problem_indices
    ]
    problem_rollouts: dict[str, list[dict[str, Any]]] = {
        str(problem_idx): list(rows)
        for problem_idx, rows in dict(existing_rollouts).items()
        if int(problem_idx) in selected_problem_indices
    }
    problem_summaries: list[dict[str, Any]] = [
        row for row in existing_problem_summaries if int(row.get("problem_idx", -1)) in selected_problem_indices
    ]
    completed_problem_indices = {int(problem_idx) for problem_idx in problem_rollouts.keys()}
    rollout_latencies: list[float] = [float(row.get("latency_seconds", 0.0) or 0.0) for row in samples]

    for local_problem_idx, row in enumerate(eval_ds):
        problem_idx = int(selected_problem_indices[local_problem_idx])
        if problem_idx in completed_problem_indices:
            print(f"[resume] skipping completed problem_idx={problem_idx}")
            continue
        prompt = str(row["prompt"])
        ground_truth = str(row["ground_truth"])
        topic = str(row["topic"]) if "topic" in eval_ds.column_names else "unknown"
        model_prompt = render_generation_prompt(tokenizer, prompt, cfg, phase="eval")
        request_messages = build_generation_messages(prompt, cfg, phase="eval") if use_chat_api else None
        model_prompt, request_messages = _apply_thinking_mode_switch(
            model_prompt,
            request_messages,
            args.thinking_mode,
        )
        prompt_token_length = len(
            tokenizer(
                model_prompt,
                add_special_tokens=True,
                padding=False,
                truncation=False,
            )["input_ids"]
        )
        effective_rollout_max_completion_len = int(args.rollout_max_completion_len)
        if model_context_len is not None:
            try:
                effective_rollout_max_completion_len = min(
                    effective_rollout_max_completion_len,
                    max(1, int(model_context_len) - int(prompt_token_length)),
                )
            except Exception:
                effective_rollout_max_completion_len = int(args.rollout_max_completion_len)

        rollout_results = _sample_rollouts(
            server_base_url=args.server_base_url,
            served_model_name=served_model_name,
            prompt=model_prompt,
            messages=request_messages,
            num_rollouts=args.num_rollouts,
            concurrency=args.request_concurrency,
            max_tokens=effective_rollout_max_completion_len,
            temperature=args.rollout_temperature,
            top_p=args.rollout_top_p,
            top_k=args.rollout_top_k,
            timeout_seconds=args.request_timeout_seconds,
        )

        prompt_rollouts: list[dict[str, Any]] = []
        for rollout in rollout_results:
            normalized = normalize_generated_completion(
                rollout.raw_text,
                answer_prefix=generation_answer_prefix(cfg, phase="eval"),
            )
            scored_completion = str(normalized.get("completion_scored", rollout.raw_text))
            raw_completion_full = str(normalized.get("completion_full_raw", rollout.raw_text))
            completion_meta = analyze_completion_text(scored_completion)
            presented_answer = extract_presented_answer(scored_completion, meta=completion_meta)
            is_correct = bool(is_correct_completion(scored_completion, ground_truth))
            completion_token_length = len(
                tokenizer(
                    raw_completion_full,
                    add_special_tokens=False,
                    padding=False,
                    truncation=False,
                )["input_ids"]
            )
            sample = {
                "problem_idx": int(problem_idx),
                "rollout_idx": int(rollout.rollout_idx),
                "prompt": prompt,
                "prompt_rendered": model_prompt,
                "ground_truth": ground_truth,
                "topic": topic,
                "completion_raw": rollout.raw_text,
                "completion_full_raw": raw_completion_full,
                "completion_scored": scored_completion,
                "presented_answer": presented_answer,
                "is_correct": bool(is_correct),
                "prompt_token_length": int(prompt_token_length),
                "prompt_truncated": bool(int(prompt_token_length) > int(args.max_prompt_len)),
                "completion_token_length": int(completion_token_length),
                "latency_seconds": float(rollout.latency_seconds),
                **completion_meta,
            }
            samples.append(sample)
            prompt_rollouts.append(
                {
                    "generation": raw_completion_full,
                    "answer": presented_answer,
                    "is_correct": bool(is_correct),
                    "rollout_idx": int(rollout.rollout_idx),
                }
            )
            rollout_latencies.append(float(rollout.latency_seconds))

        correct_count = int(sum(1 for item in prompt_rollouts if bool(item.get("is_correct", False))))
        distinct_answers = len(
            {
                str(item.get("answer"))
                for item in prompt_rollouts
                if item.get("answer") not in (None, "")
            }
        )
        problem_rollouts[str(problem_idx)] = prompt_rollouts
        problem_summaries.append(
            {
                "problem_idx": int(problem_idx),
                "topic": topic,
                "num_rollouts": int(args.num_rollouts),
                "num_correct": int(correct_count),
                "avg_correct": float(correct_count / max(1, int(args.num_rollouts))),
                "distinct_answer_count": int(distinct_answers),
                "presented_answers": [item.get("answer") for item in prompt_rollouts],
            }
        )
        completed_problem_indices.add(problem_idx)

        summary_payload = _build_summary_payload(
            args=args,
            role=role,
            manifest=manifest,
            served_model_name=served_model_name,
            pass_k_values=pass_k_values,
            samples=samples,
            problem_summaries=problem_summaries,
            benchmark_rollout_path=benchmark_rollout_path,
            rollout_latencies=rollout_latencies,
        )
        _write_json(benchmark_rollout_path, problem_rollouts)
        _write_json(benchmark_eval_samples_path, {role: samples})
        _write_json(benchmark_summary_path, summary_payload)
        print(
            f"[checkpoint] completed {len(completed_problem_indices)}/{len(selected_problem_indices)} "
            f"problem_idx={problem_idx}"
        )

    summary_payload = _build_summary_payload(
        args=args,
        role=role,
        manifest=manifest,
        served_model_name=served_model_name,
        pass_k_values=pass_k_values,
        samples=samples,
        problem_summaries=problem_summaries,
        benchmark_rollout_path=benchmark_rollout_path,
        rollout_latencies=rollout_latencies,
    )
    _write_json(benchmark_rollout_path, problem_rollouts)
    _write_json(benchmark_eval_samples_path, {role: samples})
    _write_json(benchmark_summary_path, summary_payload)

    accuracy = float((summary_payload.get("accuracies") or {}).get(role) or 0.0)
    primary_metric_name = f"avg@{int(args.num_rollouts)}"
    pass_at_k = dict(((summary_payload.get("pass_at_k") or {}).get(role) or {}))
    total_correct = int(sum(1 for row in samples if bool(row.get("is_correct", False))))
    total_rollouts = int(len(samples))

    print(
        f"{role} {primary_metric_name} (= rollout accuracy): {accuracy*100:.1f}% "
        f"({total_correct}/{max(1, total_rollouts)} correct rollouts)"
    )
    print(f"summary: {benchmark_summary_path}")
    print(f"rollouts: {benchmark_rollout_path}")

    if wandb_run is not None:
        payload = {
            f"eval/{role}_{primary_metric_name}": accuracy,
            f"eval/{role}_accuracy": accuracy,
            f"eval/{role}_rollout_accuracy": accuracy,
            f"eval/{role}_mean_latency_seconds": _mean(rollout_latencies),
            f"eval/{role}_rollout_count": float(args.num_rollouts),
            f"eval/{role}_problem_count": float(len(problem_summaries)),
        }
        for k, value in pass_at_k.items():
            payload[f"eval/{role}_pass@{k}"] = value
        if not wandb_minimal_mode():
            try:
                pass_table = wandb.Table(
                    columns=["k", "pass_at_k"],
                    data=[
                        [int(k), float(v)]
                        for k, v in sorted(pass_at_k.items(), key=lambda item: int(item[0]))
                    ],
                )
                payload[f"eval/{role}_pass_table"] = pass_table
                payload[f"eval/{role}_pass_bar"] = wandb.plot.bar(
                    pass_table,
                    "k",
                    "pass_at_k",
                    title=f"{role} pass@k",
                )

                problem_table = wandb.Table(
                    columns=[
                        "problem_idx",
                        "topic",
                        "num_rollouts",
                        "num_correct",
                        "avg_correct",
                        "distinct_answer_count",
                    ],
                    data=[
                        [
                            int(row.get("problem_idx", -1)),
                            str(row.get("topic", "unknown")),
                            int(row.get("num_rollouts", 0) or 0),
                            int(row.get("num_correct", 0) or 0),
                            float(row.get("avg_correct", 0.0) or 0.0),
                            int(row.get("distinct_answer_count", 0) or 0),
                        ]
                        for row in problem_summaries
                    ],
                )
                payload[f"eval/{role}_problem_table"] = problem_table
                payload[f"eval/{role}_num_correct_hist"] = wandb.plot.histogram(
                    problem_table,
                    "num_correct",
                    title=f"{role} correct rollouts per problem",
                )
                payload[f"eval/{role}_diversity_vs_correct"] = wandb.plot.scatter(
                    problem_table,
                    "distinct_answer_count",
                    "num_correct",
                    title=f"{role} distinct answers vs correct rollouts",
                )
            except Exception as exc:
                print(f"[warn] Could not build W&B tables/plots: {exc}")
        filtered_payload = filter_wandb_payload(payload)
        if filtered_payload:
            wandb_run.log(filtered_payload)
        try:
            artifact = wandb.Artifact(f"{args.run_name}-{role}-benchmark", type="benchmark")
            artifact.add_file(benchmark_summary_path, name=os.path.basename(benchmark_summary_path))
            artifact.add_file(benchmark_rollout_path, name=os.path.basename(benchmark_rollout_path))
            artifact.add_file(benchmark_eval_samples_path, name=os.path.basename(benchmark_eval_samples_path))
            wandb_run.log_artifact(artifact)
        except Exception as exc:
            print(f"[warn] Could not log W&B benchmark artifact: {exc}")
        wandb_run.finish()


if __name__ == "__main__":
    main()
