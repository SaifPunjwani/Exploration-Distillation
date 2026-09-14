"""AIME24 benchmark evaluation (final correctness metric).

Contract: 32 rollouts per problem × 32k completion length. Report
avg@32 (fraction correct across 32 samples, averaged over problems).
Base Qwen3-1.7B (thinking, AIME24 avg@32) = 49.1%. Goal: Actual > Base.

We delegate generation to vLLM (OpenAI-compat) so this file has no direct
dependency on the training state other than the served-model endpoint URL.
Expected to be called AFTER Actual online GRPO completes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, List

from datasets import load_dataset

from .data import normalize_answer, render_qwen3_prompt
from .generate import batched_generate, parse_server_urls
from .rewarding import extract_presented_answer


AIME24_DATASETS = ["Maxwell-Jia/AIME_2024", "opencompass/AIME2025"]
OFFICIAL_QWEN3_17B_AIME24 = {
    "avg_at_32": 0.491,
    "pass_at_4": 0.711,
    "pass_at_8": 0.763,
    "pass_at_16": 0.783,
    "pass_at_32": 0.800,
}


@dataclass
class AimeProblem:
    prompt_text: str
    answer: str


def _load_aime(
    tokenizer,
    which: str = "AIME_2024",
    enable_thinking: bool = True,
    max_problems: int = 0,
) -> List[AimeProblem]:
    if which == "AIME_2024":
        ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
        def get(row):
            p = row.get("Problem") or row.get("problem")
            a = row.get("Answer") or row.get("answer")
            return p, a
    else:  # AIME_2025
        ds1 = load_dataset("opencompass/AIME2025", "AIME2025-I", split="test")
        ds2 = load_dataset("opencompass/AIME2025", "AIME2025-II", split="test")
        from itertools import chain
        rows = list(chain(ds1, ds2))
        def get(row):
            p = row.get("question")
            a = row.get("answer")
            return p, a
        rows_iter = rows

        out: List[AimeProblem] = []
        for row in rows_iter:
            p, a = get(row)
            if not p or a is None:
                continue
            out.append(AimeProblem(render_qwen3_prompt(p, tokenizer, enable_thinking=enable_thinking), str(a)))
        return out[:max_problems] if max_problems > 0 else out

    out: List[AimeProblem] = []
    for row in ds:
        p, a = get(row)
        if not p or a is None:
            continue
        out.append(AimeProblem(render_qwen3_prompt(p, tokenizer, enable_thinking=enable_thinking), str(a)))
    return out[:max_problems] if max_problems > 0 else out


def run_eval(tokenizer, server_urls: List[str], num_rollouts: int = 32,
             max_tokens: int = 32768, model: str = "Qwen/Qwen3-1.7B",
             temperature: float = 0.6, which: str = "AIME_2024",
             enable_thinking: str = "auto", concurrency: int = 32,
             top_p: float = 0.95, top_k: int = 20,
             fanout_per_prompt: int = 8, timeout: int = 600,
             max_retries: int = 3, max_error_completions: int = -1,
             max_problems: int = 0,
             progress_callback: Callable[[int, int, float], None] | None = None) -> dict:
    problems = _load_aime(
        tokenizer,
        which=which,
        enable_thinking=enable_thinking != "never",
        max_problems=max_problems,
    )
    print(f"[eval] {which}: {len(problems)} problems x {num_rollouts} rollouts", flush=True)

    prompts = [p.prompt_text for p in problems]
    completions = batched_generate(
        prompts, server_urls, n_per_prompt=num_rollouts,
        max_tokens=max_tokens, temperature=temperature,
        top_p=top_p, top_k=top_k, enable_thinking=enable_thinking, model=model,
        concurrency=concurrency, fanout_per_prompt=fanout_per_prompt,
        timeout=timeout, max_retries=max_retries,
        allow_error_completions=True,
        progress_callback=progress_callback,
    )

    def pass_at_k(n: int, c: int, k: int) -> float:
        if k <= 0:
            return 0.0
        if c <= 0:
            return 0.0
        if n - c < k:
            return 1.0
        try:
            return 1.0 - (math.comb(n - c, k) / math.comb(n, k))
        except Exception:
            prod = 1.0
            for i in range(k):
                prod *= (n - c - i) / max(1, n - i)
            return 1.0 - prod

    per_problem = []
    problem_summaries = []
    rollout_rows = []
    correct_total = 0
    total = 0
    error_total = 0
    for problem_idx, (prob, comps) in enumerate(zip(problems, completions)):
        hits = 0
        predictions = []
        for rollout_idx, c in enumerate(comps):
            pred = extract_presented_answer(c.text)
            pred_norm = "" if pred is None else normalize_answer(pred)
            correct = bool(pred is not None and pred_norm == normalize_answer(prob.answer))
            finish_reason = getattr(c, "finish_reason", "")
            is_error = finish_reason == "error"
            if is_error:
                error_total += 1
            predictions.append(pred_norm)
            if correct:
                hits += 1
            rollout_rows.append({
                "problem_idx": problem_idx,
                "rollout_idx": rollout_idx,
                "prediction": pred_norm,
                "answer": normalize_answer(prob.answer),
                "correct": correct,
                "finish_reason": finish_reason,
                "error": is_error,
                "completion_text": c.text,
            })
        denom = max(1, len(comps))
        per_problem.append(hits / denom)
        problem_summaries.append({
            "problem_idx": problem_idx,
            "num_rollouts": len(comps),
            "num_correct": hits,
            "num_errors": sum(1 for c in comps if getattr(c, "finish_reason", "") == "error"),
            "avg_at_n": hits / denom,
            "any_correct": hits > 0,
            "distinct_answer_count": len({p for p in predictions if p}),
        })
        correct_total += hits
        total += len(comps)

    avg = sum(per_problem) / len(per_problem)
    pass1 = sum(1 for x in per_problem if x > 0) / len(per_problem)
    pass_metrics = {}
    for k in (1, 2, 4, 8, 16, 32):
        if k <= num_rollouts:
            pass_metrics[f"pass_at_{k}"] = sum(
                pass_at_k(int(s["num_rollouts"]), int(s["num_correct"]), k)
                for s in problem_summaries
            ) / max(1, len(problem_summaries))
    distinct_answer_mean = sum(float(s["distinct_answer_count"]) for s in problem_summaries) / max(1, len(problem_summaries))
    error_fraction = error_total / max(1, total)
    print(f"[eval] avg@{num_rollouts}={avg*100:.2f}% any>0={pass1*100:.2f}% (n={len(problems)})", flush=True)
    print(f"[eval] error_completions={error_total}/{total} ({error_fraction*100:.2f}%)", flush=True)
    if max_error_completions >= 0 and error_total > max_error_completions:
        raise RuntimeError(
            f"eval exceeded error budget: error_completions={error_total} "
            f"max_error_completions={max_error_completions}"
        )
    return {
        "avg_at_n": avg,
        "any_correct_fraction": pass1,
        "num_problems": len(problems),
        "num_rollouts": num_rollouts,
        "correct_total": correct_total,
        "total": total,
        "error_total": error_total,
        "error_fraction": error_fraction,
        "distinct_answer_mean": distinct_answer_mean,
        "problem_summaries": problem_summaries,
        "rollouts": rollout_rows,
        **pass_metrics,
    }


def aime24_wandb_metrics(res: dict, *, prefix: str = "eval/aime24_32k") -> dict[str, float]:
    """Compact W&B metrics for deciding whether Actual beats base."""
    num_rollouts = int(res.get("num_rollouts", 32) or 32)
    avg = float(res.get("avg_at_n", 0.0) or 0.0)
    is_probe = prefix.startswith("eval_probe/")
    if is_probe:
        step = float(res.get("protocol", {}).get("step", 0.0) or 0.0)
        pass_at_4 = float(res.get("pass_at_4", avg) or 0.0)
        return {
            f"{prefix}/step": step,
            f"{prefix}/accuracy": avg,
            f"{prefix}/avg_at_{num_rollouts}": avg,
            f"{prefix}/avg_at_n": avg,
            f"{prefix}/pass_at_4": pass_at_4,
            f"{prefix}/error_fraction": float(res.get("error_fraction", 0.0) or 0.0),
            f"{prefix}/delta_vs_initial": float(res.get("delta_vs_initial_avg_at_n", 0.0) or 0.0),
            f"{prefix}/best_avg_at_n": float(res.get("best_avg_at_n_so_far", avg) or avg),
            "aime24_probe_step": step,
            "aime24_probe_accuracy": avg,
            "aime24_probe_avg_at_n": avg,
            "aime24_probe_delta_vs_initial": float(res.get("delta_vs_initial_avg_at_n", 0.0) or 0.0),
            "aime24_probe_best_so_far": float(res.get("best_avg_at_n_so_far", avg) or avg),
        }
    metrics: dict[str, float] = {
        f"{prefix}/avg_at_{num_rollouts}": avg,
        f"{prefix}/avg_at_n": avg,
        f"{prefix}/base_avg_at_32": OFFICIAL_QWEN3_17B_AIME24["avg_at_32"],
        f"{prefix}/delta_vs_base_avg_at_32": avg - OFFICIAL_QWEN3_17B_AIME24["avg_at_32"],
        f"{prefix}/any_correct_fraction": float(res.get("any_correct_fraction", 0.0) or 0.0),
        f"{prefix}/num_problems": float(res.get("num_problems", 0.0) or 0.0),
        f"{prefix}/num_rollouts": float(num_rollouts),
        f"{prefix}/correct_total": float(res.get("correct_total", 0.0) or 0.0),
        f"{prefix}/total": float(res.get("total", 0.0) or 0.0),
        f"{prefix}/error_fraction": float(res.get("error_fraction", 0.0) or 0.0),
        f"{prefix}/distinct_answer_mean": float(res.get("distinct_answer_mean", 0.0) or 0.0),
    }
    metrics["aime24_avg_at_32"] = avg
    metrics["aime24_delta_vs_base_avg_at_32"] = avg - OFFICIAL_QWEN3_17B_AIME24["avg_at_32"]
    for k in (1, 2, 4, 8, 16, 32):
        key = f"pass_at_{k}"
        if key in res:
            value = float(res.get(key, 0.0) or 0.0)
            metrics[f"{prefix}/{key}"] = value
            if key in OFFICIAL_QWEN3_17B_AIME24:
                metrics[f"{prefix}/delta_vs_base_{key}"] = value - OFFICIAL_QWEN3_17B_AIME24[key]
    return metrics


def annotate_probe_trend(res: dict, history_dir: Path | str | None = None) -> dict:
    """Add simple trend fields for lightweight eval probes.

    The probe is intentionally small, so the useful signal is the direction
    across checkpoint boundaries, not a single point estimate.
    """
    avg = float(res.get("avg_at_n", 0.0) or 0.0)
    history: list[tuple[int, float]] = []
    if history_dir is not None:
        hdir = Path(history_dir)
        for path in hdir.glob("aime24*step_*.json"):
            try:
                prior = json.loads(path.read_text())
                step = int(prior.get("protocol", {}).get("step", 0) or 0)
                history.append((step, float(prior.get("avg_at_n", 0.0) or 0.0)))
            except Exception:
                continue
    step = int(res.get("protocol", {}).get("step", 0) or 0)
    history.append((step, avg))
    history.sort(key=lambda x: x[0])
    initial = history[0][1] if history else avg
    best = max((v for _s, v in history), default=avg)
    res["initial_avg_at_n"] = float(initial)
    res["delta_vs_initial_avg_at_n"] = float(avg - initial)
    res["best_avg_at_n_so_far"] = float(best)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    p.add_argument("--served-model-name", default="")
    p.add_argument("--tokenizer-name", default="")
    p.add_argument("--server-urls", required=True)
    p.add_argument("--num-rollouts", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=32768)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--which", default="AIME_2024")
    p.add_argument("--enable-thinking", default="auto")
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--fanout-per-prompt", type=int, default=8)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument(
        "--max-error-completions",
        type=int,
        default=-1,
        help="Fail if more than this many completions are infrastructure errors; -1 disables.",
    )
    p.add_argument("--max-problems", type=int, default=0)
    p.add_argument("--output-path", default="")
    p.add_argument("--use-wandb", action="store_true")
    p.add_argument("--wandb-run-name", default="")
    p.add_argument("--wandb-run-id", default="")
    p.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "two-model-explore"))
    p.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY") or None)
    p.add_argument("--wandb-group", default=os.environ.get("TMX_WANDB_GROUP", ""))
    p.add_argument("--wandb-prefix", default="eval/aime24_32k")
    args = p.parse_args()
    tokenizer_name = args.tokenizer_name or args.model_name
    served_model_name = args.served_model_name or args.model_name
    if "gemma-4" in tokenizer_name.lower() or "gemma4" in tokenizer_name.lower():
        from transformers import AutoProcessor

        tok = AutoProcessor.from_pretrained(tokenizer_name, trust_remote_code=True)
    else:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    wandb_run = None
    progress_callback = None
    if args.use_wandb:
        import wandb
        run_name = args.wandb_run_name or "aime24_eval"
        run_id = (
            args.wandb_run_id
            or os.environ.get("TMX_WANDB_RUN_ID", "").strip()
            or "jaxeval-" + hashlib.sha1(f"{run_name}:aime24".encode("utf-8")).hexdigest()[:24]
        )
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=run_name,
            id=run_id,
            resume="allow",
            group=args.wandb_group or run_name,
            job_type="eval",
            config=vars(args),
        )
        if args.wandb_prefix.startswith("eval_probe/"):
            try:
                wandb.define_metric(f"{args.wandb_prefix}/step")
                wandb.define_metric(f"{args.wandb_prefix}/*", step_metric=f"{args.wandb_prefix}/step")
                wandb.define_metric("aime24_probe_step")
                wandb.define_metric("aime24_probe_*", step_metric="aime24_probe_step")
            except Exception:
                pass

        def progress_callback(done: int, total: int, elapsed: float) -> None:
            eta = (elapsed / done) * (total - done) if done else 0.0
            if args.wandb_prefix.startswith("eval_probe/"):
                wandb.log({
                    f"{args.wandb_prefix}/progress_fraction": float(done / total if total else 0.0),
                })
            else:
                wandb.log({
                    f"{args.wandb_prefix}/subrequests_done": float(done),
                    f"{args.wandb_prefix}/subrequests_total": float(total),
                    f"{args.wandb_prefix}/progress_fraction": float(done / total if total else 0.0),
                    f"{args.wandb_prefix}/elapsed_seconds": float(elapsed),
                    f"{args.wandb_prefix}/eta_seconds": float(eta),
                })

    res = run_eval(tok, parse_server_urls(args.server_urls),
                   num_rollouts=args.num_rollouts, max_tokens=args.max_tokens,
                   model=served_model_name, temperature=args.temperature,
                   which=args.which, enable_thinking=args.enable_thinking,
                   concurrency=args.concurrency, top_p=args.top_p, top_k=args.top_k,
                   fanout_per_prompt=args.fanout_per_prompt,
                   timeout=args.timeout, max_retries=args.max_retries,
                   max_error_completions=args.max_error_completions,
                   max_problems=args.max_problems,
                   progress_callback=progress_callback)
    res["protocol"] = {
        "model_name": args.model_name,
        "served_model_name": served_model_name,
        "tokenizer_name": tokenizer_name,
        "server_urls": parse_server_urls(args.server_urls),
        "num_rollouts": int(args.num_rollouts),
        "max_tokens": int(args.max_tokens),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "top_k": int(args.top_k),
        "which": args.which,
        "enable_thinking": args.enable_thinking,
        "concurrency": int(args.concurrency),
        "fanout_per_prompt": int(args.fanout_per_prompt),
        "timeout": int(args.timeout),
        "max_retries": int(args.max_retries),
        "max_error_completions": int(args.max_error_completions),
        "max_problems": int(args.max_problems),
    }
    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if args.wandb_prefix.startswith("eval_probe/"):
            annotate_probe_trend(res, output_path.parent)
        output_path.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
        print(f"[eval] wrote {output_path}", flush=True)
    else:
        import sys
        if args.wandb_prefix.startswith("eval_probe/"):
            annotate_probe_trend(res, None)
        json.dump(res, sys.stdout, indent=2); print()
        output_path = None
    if args.use_wandb:
        import wandb
        if wandb_run is not None:
            wandb.log(aime24_wandb_metrics(res, prefix=args.wandb_prefix))
            if output_path is not None and output_path.exists():
                artifact = wandb.Artifact(f"{wandb_run.name}_aime24_eval", type="eval_result")
                artifact.add_file(str(output_path))
                wandb.log_artifact(artifact)
            wandb.finish()


if __name__ == "__main__":
    main()
