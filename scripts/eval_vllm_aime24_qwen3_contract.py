#!/usr/bin/env python3
"""Evaluate a vLLM-served Qwen3 model on AIME24 with the Qwen3 sampling contract."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from datasets import load_dataset
from transformers import AutoTokenizer


BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--endpoints", required=True, help="Comma-separated vLLM /v1 endpoints")
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--tokenizer", default="Qwen/Qwen3-1.7B")
    p.add_argument("--num-problems", type=int, default=30)
    p.add_argument("--num-rollouts", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=32000)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--min-p", type=float, default=0.0)
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--timeout", type=int, default=3600)
    p.add_argument("--out-dir", default="rollouts_local/evals")
    p.add_argument("--run-name", required=True)
    return p.parse_args()


def extract_answer(text: str) -> int | None:
    matches = BOXED_RE.findall(text or "")
    if not matches:
        return None
    raw = matches[-1].strip().replace(",", "").replace(" ", "")
    raw = raw.strip("$")
    try:
        return int(raw)
    except Exception:
        try:
            return int(float(raw))
        except Exception:
            return None


def render_prompt(tokenizer: Any, question: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful mathematician. Solve the problem step by step. "
                "End with your final answer in \\boxed{}."
            ),
        },
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def normalize_endpoint(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    if endpoint.endswith("/v1"):
        return endpoint
    return f"{endpoint}/v1"


def post_one(args: argparse.Namespace, endpoints: list[str], prompt: str, request_id: int) -> dict[str, Any]:
    endpoint = endpoints[request_id % len(endpoints)]
    payload = {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "n": 1,
        "stream": False,
    }
    t0 = time.time()
    last_error = None
    order = [endpoint] + [e for e in endpoints if e != endpoint]
    for ep in order:
        try:
            resp = requests.post(f"{ep}/completions", json=payload, timeout=args.timeout)
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            return {
                "endpoint": ep,
                "text": choice.get("text", ""),
                "finish_reason": choice.get("finish_reason"),
                "stop_reason": choice.get("stop_reason"),
                "completion_tokens": data.get("usage", {}).get("completion_tokens"),
                "total_tokens": data.get("usage", {}).get("total_tokens"),
                "time_s": time.time() - t0,
            }
        except Exception as exc:
            last_error = str(exc)
    return {
        "endpoint": endpoint,
        "text": "",
        "finish_reason": "error",
        "error": last_error,
        "completion_tokens": 0,
        "total_tokens": 0,
        "time_s": time.time() - t0,
    }


def load_aime24() -> list[dict[str, Any]]:
    try:
        return list(load_dataset("HuggingFaceH4/aime_2024", split="train"))
    except Exception:
        return list(load_dataset("Maxwell-Jia/AIME_2024", split="train"))


def get_problem(row: dict[str, Any]) -> str:
    return str(row.get("problem") or row.get("Problem") or row.get("question") or row.get("Question") or "")


def get_answer(row: dict[str, Any]) -> int:
    raw = str(row.get("answer") or row.get("Answer") or row.get("final_answer") or "")
    nums = re.findall(r"-?\d+", raw)
    if not nums:
        raise ValueError(f"Could not parse AIME answer from row: {row}")
    return int(nums[-1])


def main() -> None:
    args = parse_args()
    endpoints = [normalize_endpoint(e) for e in args.endpoints.split(",") if e.strip()]
    if not endpoints:
        raise SystemExit("No endpoints provided")

    out_dir = Path(args.out_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    rollouts_path = out_dir / "aime24_rollouts.jsonl"
    summary_path = out_dir / "aime24_summary.json"

    print(f"[eval] endpoints={len(endpoints)} model={args.model}", flush=True)
    print(
        "[eval] contract: AIME24 Qwen3 thinking, "
        f"temperature={args.temperature} top_p={args.top_p} top_k={args.top_k} "
        f"min_p={args.min_p} max_tokens={args.max_tokens}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    rows = load_aime24()[: args.num_problems]
    tasks = []
    request_id = 0
    for problem_idx, row in enumerate(rows):
        prompt = render_prompt(tokenizer, get_problem(row))
        gold = get_answer(row)
        for rollout_idx in range(args.num_rollouts):
            tasks.append((request_id, problem_idx, rollout_idx, prompt, gold))
            request_id += 1

    results_by_problem: dict[int, list[dict[str, Any]]] = {i: [] for i in range(len(rows))}
    total_correct = 0
    t0 = time.time()
    with rollouts_path.open("w") as out_f, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(post_one, args, endpoints, prompt, rid): (rid, pidx, ridx, gold)
            for rid, pidx, ridx, prompt, gold in tasks
        }
        for done_idx, fut in enumerate(as_completed(futures), start=1):
            rid, problem_idx, rollout_idx, gold = futures[fut]
            item = fut.result()
            pred = extract_answer(item.get("text", ""))
            correct = pred == gold
            total_correct += int(correct)
            record = {
                "request_id": rid,
                "problem_idx": problem_idx,
                "rollout_idx": rollout_idx,
                "gold": gold,
                "pred": pred,
                "correct": correct,
                **item,
            }
            results_by_problem[problem_idx].append(record)
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
            if done_idx % 10 == 0 or done_idx == len(tasks):
                elapsed = time.time() - t0
                print(
                    f"[eval] {done_idx}/{len(tasks)} done "
                    f"correct={total_correct}/{done_idx} avg={total_correct/done_idx:.3f} "
                    f"elapsed={elapsed/60:.1f}m",
                    flush=True,
                )

    total = len(tasks)
    avg = total_correct / total if total else 0.0
    pass_at = {}
    for k in [1, 2, 4, 8, 16, 32]:
        if k <= args.num_rollouts:
            pass_at[f"pass@{k}"] = sum(
                1 for rs in results_by_problem.values() if sum(r["correct"] for r in rs) >= k
            ) / len(rows)

    maj_correct = 0
    per_problem = []
    for problem_idx, rs in results_by_problem.items():
        answers = [r["pred"] for r in rs if r.get("pred") is not None]
        majority = Counter(answers).most_common(1)[0][0] if answers else None
        gold = get_answer(rows[problem_idx])
        n_correct = sum(r["correct"] for r in rs)
        maj_is_correct = majority == gold
        maj_correct += int(maj_is_correct)
        per_problem.append(
            {
                "problem_idx": problem_idx,
                "gold": gold,
                "n_correct": n_correct,
                "majority": majority,
                "majority_correct": maj_is_correct,
            }
        )
    maj = maj_correct / len(rows) if rows else 0.0

    summary = {
        "run_name": args.run_name,
        "model": args.model,
        "dataset": "AIME24",
        "num_problems": len(rows),
        "num_rollouts": args.num_rollouts,
        "total_rollouts": total,
        "decoder": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
            "max_tokens": args.max_tokens,
            "thinking": True,
        },
        f"avg@{args.num_rollouts}": avg,
        **pass_at,
        f"maj@{args.num_rollouts}": maj,
        "correct": total_correct,
        "total": total,
        "elapsed_s": time.time() - t0,
        "per_problem": per_problem,
        "rollouts_path": str(rollouts_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"[eval] saved summary={summary_path} rollouts={rollouts_path}", flush=True)


if __name__ == "__main__":
    main()
