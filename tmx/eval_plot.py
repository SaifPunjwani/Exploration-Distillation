import math
import os
import torch
from tqdm.auto import tqdm

from .reward import (
    analyze_completion_text,
    build_actual_training_text,
    canonicalize_completion_for_training,
    completion_quality_metrics,
    extract_presented_answer,
    is_correct_completion,
    normalize_generated_completion,
    resolve_novelty_text,
)
from .prompting import (
    _resolved_thinking_mode,
    generation_answer_prefix,
    get_answer_token_constraints,
    render_generation_prompt,
    resolved_max_completion_len,
)
from .utils import xla_safe_generate
from .utils import get_device

try:
    import wandb  # type: ignore
except Exception:
    wandb = None


def _percentile(values, pct: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    if len(vals) == 1:
        return float(vals[0])
    pos = (pct / 100.0) * (len(vals) - 1)
    low = int(pos)
    high = min(low + 1, len(vals) - 1)
    weight = pos - low
    return float(vals[low] * (1 - weight) + vals[high] * weight)


def _mean(values) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


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


def evaluate_model_on_gsm8k(model, tokenizer, eval_ds, cfg, novelty_scorer, desc="Model", use_wandb: bool = False, wandb_run=None):
    """
    Greedy evaluation:
      - generate solution
      - parse final integer
      - compute accuracy
      - ALSO: compute mean novelty of completions if novelty_scorer is provided.
    """
    model.eval()
    device = get_device(model)

    correct = 0
    total = 0
    novelties = []
    samples = []

    holdout_topics = set(getattr(cfg, "id_ood_holdout_topics", []) or [])
    id_correct = 0
    id_total = 0
    ood_correct = 0
    ood_total = 0
    id_nov = []
    ood_nov = []

    subset = eval_ds.select(range(min(len(eval_ds), cfg.max_eval_examples)))
    eval_batch_size = max(1, int(getattr(cfg, "eval_batch_size", 1) or 1))
    max_completion_len = resolved_max_completion_len(cfg, phase="eval")
    eos_id = tokenizer.eos_token_id
    if isinstance(eos_id, (list, tuple)):
        eos_id = eos_id[0] if eos_id else None
    if torch.is_tensor(eos_id):
        eos_id = eos_id.flatten()[0].item() if eos_id.numel() > 0 else None

    vllm_urls_raw = (os.environ.get("TMX_VLLM_SERVER_URLS") or "").strip()
    # Use vLLM for Base eval (servers serve base weights).
    # For Actual eval, use TMX_ACTUAL_VLLM_SERVER_URL if set (a server reloaded with Actual weights).
    actual_vllm_url = (os.environ.get("TMX_ACTUAL_VLLM_SERVER_URL") or "").strip()
    if desc.lower() == "actual" and actual_vllm_url:
        vllm_urls_raw = actual_vllm_url
        use_vllm_eval = True
    elif desc.lower() in ("base", "model") and vllm_urls_raw:
        use_vllm_eval = True
    else:
        use_vllm_eval = False

    for start in tqdm(range(0, len(subset), eval_batch_size), desc=f"Eval {desc}"):
        stop = min(len(subset), start + eval_batch_size)
        batch = subset.select(range(start, stop))
        prompts = list(batch["prompt"])
        model_prompts = [render_generation_prompt(tokenizer, prompt, cfg, phase="eval") for prompt in prompts]
        ground_truths = list(batch["ground_truth"])
        topics = list(batch["topic"]) if "topic" in batch.column_names else ["unknown"] * len(prompts)
        is_ood_batch = [bool(holdout_topics and topic in holdout_topics) for topic in topics]
        prompt_length_ids = tokenizer(
            model_prompts,
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )["input_ids"]
        prompt_token_lengths = [len(ids) for ids in prompt_length_ids]

        if use_vllm_eval:
            from .vllm_generate import generate_completions_vllm
            vllm_urls = [u.strip() for u in vllm_urls_raw.split(",") if u.strip()]
            vllm_model = os.environ.get("TMX_VLLM_MODEL_NAME", "")
            vllm_concurrency = int(os.environ.get("TMX_VLLM_CONCURRENCY", "64"))
            eval_thinking_mode = _resolved_thinking_mode(cfg, phase="eval")
            eval_enable_thinking = True if eval_thinking_mode == "think" else (False if eval_thinking_mode == "no_think" else None)
            all_completions = generate_completions_vllm(
                prompts=model_prompts,
                model_name=vllm_model,
                server_urls=vllm_urls,
                n_per_prompt=1,
                max_tokens=max_completion_len,
                temperature=0.0,
                top_p=1.0,
                concurrency=vllm_concurrency,
                enable_thinking=eval_enable_thinking,
            )
            raw_completions = [comps[0]["text"] if comps else "" for comps in all_completions]
            # Build approximate generation stats from vLLM results
            vllm_finish_reasons = [comps[0].get("finish_reason", "length") if comps else "error" for comps in all_completions]
            vllm_generation_stats = []
            for rc, fr in zip(raw_completions, vllm_finish_reasons):
                approx_tokens = len(tokenizer.encode(rc, add_special_tokens=False)) if rc else 0
                vllm_generation_stats.append({
                    "completion_token_length": approx_tokens,
                    "terminated": fr == "stop",
                    "clipped": fr == "length",
                })
        else:
            enc = tokenizer(
                model_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=cfg.max_prompt_len,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            prompt_width = int(enc["input_ids"].shape[1])

            with torch.no_grad():
                answer_constraints = get_answer_token_constraints(tokenizer, cfg, phase="eval")
                out_ids = _run_eval_generate(
                    model,
                    tokenizer,
                    enc,
                    max_new_tokens=max_completion_len,
                    do_sample=False,
                    eos_id=eos_id,
                    answer_constraints=answer_constraints,
                )

            raw_completions = [
                tokenizer.decode(out_ids[i][prompt_width:], skip_special_tokens=True)
                for i in range(out_ids.shape[0])
            ]
        normalized_batch = [
            normalize_generated_completion(
                raw_completion,
                answer_prefix=generation_answer_prefix(cfg, phase="eval"),
            )
            for raw_completion in raw_completions
        ]
        completions = [str(item.get("completion_scored", raw_completion)) for item, raw_completion in zip(normalized_batch, raw_completions)]
        novelty_texts = [
            resolve_novelty_text(
                normalized,
                fallback_completion=raw_completion,
                novelty_text_contract=cfg.novelty_text_contract,
            )
            for normalized, raw_completion in zip(normalized_batch, raw_completions)
        ]
        novelty_batch = (
            [float(v) for v in novelty_scorer.score_batch(novelty_texts, train=False, phase=f"eval_{desc.lower()}")]
            if novelty_scorer is not None
            else [0.0 for _ in completions]
        )

        for row_idx, (prompt, completion, raw_completion, normalized_completion, novelty_text, prompt_token_length, gt, topic, is_ood, nov) in enumerate(
            zip(
                prompts,
                completions,
                raw_completions,
                normalized_batch,
                novelty_texts,
                prompt_token_lengths,
                ground_truths,
                topics,
                is_ood_batch,
                novelty_batch,
            )
        ):
            is_correct = bool(is_correct_completion(completion, gt))
            correct += int(is_correct)
            total += 1
            if use_vllm_eval:
                generation_stats = vllm_generation_stats[row_idx]
            else:
                generation_stats = _sequence_generation_stats(
                    out_ids[row_idx],
                    prompt_width=prompt_width,
                    eos_id=eos_id,
                    pad_token_id=tokenizer.pad_token_id,
                    max_new_tokens=max_completion_len,
                )
            completion_meta = analyze_completion_text(completion)
            quality_metrics = completion_quality_metrics(completion_meta, extra_flags=generation_stats)
            raw_completion_full = str(normalized_completion.get("completion_full_raw", raw_completion))
            raw_completion_meta = analyze_completion_text(raw_completion_full)
            raw_quality_metrics = completion_quality_metrics(raw_completion_meta, extra_flags=generation_stats)
            presented_answer = extract_presented_answer(completion, meta=completion_meta)
            canonical_completion = canonicalize_completion_for_training(completion, meta=completion_meta)
            actual_target_canonical = build_actual_training_text(
                normalized_completion,
                fallback_completion=raw_completion,
                target_contract="canonical",
            )
            actual_target_full_trace = build_actual_training_text(
                normalized_completion,
                fallback_completion=raw_completion,
                target_contract="full_trace",
            )
            normalization_changed = raw_completion_full.strip() != completion.strip()

            if novelty_scorer is not None:
                novelties.append(float(nov))
                if is_ood:
                    ood_nov.append(float(nov))
                else:
                    id_nov.append(float(nov))

            if is_ood:
                ood_total += 1
                ood_correct += int(is_correct)
            else:
                id_total += 1
                id_correct += int(is_correct)

            samples.append(
                {
                    "prompt": prompt,
                    "completion": completion,
                    "completion_raw": raw_completion,
                    "completion_full_raw": raw_completion_full,
                    "completion_first_line": str(normalized_completion.get("completion_first_line", "")),
                    "completion_scored": completion,
                    "ground_truth": gt,
                    "presented_answer": presented_answer,
                    "completion_canonical": canonical_completion,
                    "actual_target_canonical": actual_target_canonical,
                    "actual_target_full_trace": actual_target_full_trace,
                    "topic": topic,
                    "is_ood": is_ood,
                    "is_correct": is_correct,
                    "novelty": float(nov),
                    "novelty_text": novelty_text,
                    "novelty_text_contract": cfg.novelty_text_contract,
                    "format_quality": float(quality_metrics.get("quality", 0.0)),
                    "format_penalty": float(quality_metrics.get("penalty", 0.0)),
                    "format_penalty_breakdown": dict(quality_metrics.get("penalties", {})),
                    "raw_format_quality": float(raw_quality_metrics.get("quality", 0.0)),
                    "raw_format_penalty": float(raw_quality_metrics.get("penalty", 0.0)),
                    "raw_format_penalty_breakdown": dict(raw_quality_metrics.get("penalties", {})),
                    "normalization_changed": bool(normalization_changed),
                    "prompt_token_length": int(prompt_token_length),
                    "prompt_truncated": bool(int(prompt_token_length) > int(cfg.max_prompt_len)),
                    **generation_stats,
                    **completion_meta,
                }
            )

    acc = correct / max(1, total)

    mean_nov = float(sum(novelties) / len(novelties)) if novelties else 0.0
    if novelties:
        print(
            f"{desc} accuracy: {acc*100:.1f}% ({correct}/{total}), "
            f"mean novelty: {mean_nov:.4f}"
        )
    else:
        print(f"{desc} accuracy: {acc*100:.1f}% ({correct}/{total})")

    if use_wandb and wandb_run is not None and wandb is not None:
        payload = {
            f"eval/{desc}_accuracy": acc,
            f"eval/{desc}_mean_novelty": mean_nov,
            f"eval/{desc}_novelty_p50": _percentile(novelties, 50.0),
            f"eval/{desc}_novelty_p90": _percentile(novelties, 90.0),
            f"eval/{desc}_format_quality_mean": _mean([float(sample.get("format_quality", 0.0)) for sample in samples]),
            f"eval/{desc}_clipped_fraction": _mean([1.0 if bool(sample.get("clipped", False)) else 0.0 for sample in samples]),
            f"eval/{desc}_final_answer_tag_fraction": _mean([1.0 if bool(sample.get("has_final_answer_tag", False)) else 0.0 for sample in samples]),
            f"eval/{desc}_valid_answer_fraction": _mean([1.0 if bool(sample.get("valid_answer_presentation", False)) else 0.0 for sample in samples]),
            f"eval/{desc}_id_accuracy": (id_correct / max(1, id_total)),
            f"eval/{desc}_ood_accuracy": (ood_correct / max(1, ood_total)),
            f"eval/{desc}_id_mean_novelty": float(sum(id_nov) / len(id_nov)) if id_nov else 0.0,
            f"eval/{desc}_ood_mean_novelty": float(sum(ood_nov) / len(ood_nov)) if ood_nov else 0.0,
        }
        if novelties:
            payload[f"eval/{desc}_novelty_hist"] = wandb.Histogram(novelties)
        wandb_run.log(payload)

    return {
        "accuracy": acc,
        "primary_metric_name": "accuracy",
        "novelty_mean": mean_nov,
        "novelty_values": novelties,
        "correct": correct,
        "total": total,
        "id_accuracy": id_correct / max(1, id_total),
        "ood_accuracy": ood_correct / max(1, ood_total),
        "id_total": id_total,
        "ood_total": ood_total,
        "id_novelty_mean": float(sum(id_nov) / len(id_nov)) if id_nov else 0.0,
        "ood_novelty_mean": float(sum(ood_nov) / len(ood_nov)) if ood_nov else 0.0,
        "samples": samples,
    }


def _benchmark_eval_batch_size(cfg) -> int:
    batch_size = getattr(cfg, "benchmark_eval_batch_size", None)
    if batch_size is None:
        batch_size = getattr(cfg, "eval_batch_size", 1)
    return max(1, int(batch_size or 1))


def _benchmark_max_completion_len(cfg) -> int:
    benchmark_len = getattr(cfg, "benchmark_rollout_max_completion_len", None)
    if benchmark_len is not None:
        try:
            parsed = int(benchmark_len)
            if parsed > 0:
                return parsed
        except Exception:
            pass
    return resolved_max_completion_len(cfg, phase="eval")


def _prefer_native_xla_eval_generate(answer_constraints, *, do_sample: bool = False) -> bool:
    if do_sample:
        return False
    if answer_constraints is not None:
        return False
    flag = str(os.environ.get("TMX_XLA_EVAL_NATIVE_GENERATE", "1") or "1").strip().lower()
    return flag not in {"0", "false", "no", "off"}


def _run_eval_generate(
    model,
    tokenizer,
    enc,
    *,
    max_new_tokens: int,
    do_sample: bool,
    eos_id,
    answer_constraints,
    temperature: float | None = None,
    top_p: float | None = None,
):
    device = enc["input_ids"].device
    generate_kwargs = {
        **enc,
        "max_new_tokens": int(max_new_tokens),
        "do_sample": bool(do_sample),
        "eos_token_id": eos_id,
        "pad_token_id": tokenizer.pad_token_id,
        "use_cache": True,
    }
    if do_sample:
        if temperature is not None:
            generate_kwargs["temperature"] = float(temperature)
        if top_p is not None:
            generate_kwargs["top_p"] = float(top_p)

    if device.type == "xla" and _prefer_native_xla_eval_generate(answer_constraints, do_sample=bool(do_sample)):
        return model.generate(
            **generate_kwargs,
            suppress_tokens=None,
            begin_suppress_tokens=None,
            bad_words_ids=None,
        )

    if device.type == "xla" or answer_constraints is not None:
        return xla_safe_generate(
            model,
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_id,
            pad_token_id=tokenizer.pad_token_id,
            allowed_token_ids=(answer_constraints or {}).get("allowed_token_ids"),
            initial_token_ids=(answer_constraints or {}).get("initial_token_ids"),
            use_cache=True,
        )

    return model.generate(
        **generate_kwargs,
        suppress_tokens=None,
        begin_suppress_tokens=None,
        bad_words_ids=None,
    )


def _normalized_pass_k_values(cfg, num_rollouts: int) -> list[int]:
    values = []
    for value in getattr(cfg, "benchmark_rollout_pass_k", []) or []:
        try:
            parsed = int(value)
        except Exception:
            continue
        if parsed > 0:
            values.append(min(parsed, int(num_rollouts)))
    if not values:
        values = [1, min(32, int(num_rollouts))]
    values = sorted(set(values))
    if 1 not in values:
        values.insert(0, 1)
    return values


def _pass_at_k_estimate(num_samples: int, num_correct: int, k: int) -> float:
    n = max(1, int(num_samples))
    c = max(0, min(int(num_correct), n))
    kk = max(1, min(int(k), n))
    if c <= 0:
        return 0.0
    if kk > n - c:
        return 1.0
    return 1.0 - (math.comb(n - c, kk) / math.comb(n, kk))


def evaluate_model_with_rollouts(model, tokenizer, eval_ds, cfg, novelty_scorer, desc="Model", use_wandb: bool = False, wandb_run=None):
    model.eval()
    device = get_device(model)

    subset = eval_ds.select(range(min(len(eval_ds), cfg.max_eval_examples)))
    num_rollouts = max(1, int(getattr(cfg, "benchmark_num_rollouts", 1) or 1))
    batch_size = _benchmark_eval_batch_size(cfg)
    max_completion_len = _benchmark_max_completion_len(cfg)
    temperature = float(getattr(cfg, "benchmark_rollout_temperature", 1.0) or 1.0)
    top_p = float(getattr(cfg, "benchmark_rollout_top_p", 1.0) or 1.0)
    pass_k_values = _normalized_pass_k_values(cfg, num_rollouts)

    eos_id = tokenizer.eos_token_id
    if isinstance(eos_id, (list, tuple)):
        eos_id = eos_id[0] if eos_id else None
    if torch.is_tensor(eos_id):
        eos_id = eos_id.flatten()[0].item() if eos_id.numel() > 0 else None

    holdout_topics = set(getattr(cfg, "id_ood_holdout_topics", []) or [])
    total_correct = 0
    total = 0
    novelties = []
    samples = []
    problem_rollouts = {}
    problem_summaries = []
    distinct_answer_counts = []
    id_correct = 0
    id_total = 0
    ood_correct = 0
    ood_total = 0
    id_nov = []
    ood_nov = []

    for problem_idx, row in enumerate(tqdm(subset, desc=f"Eval {desc} rollouts")):
        prompt = str(row["prompt"])
        ground_truth = str(row["ground_truth"])
        topic = str(row["topic"]) if "topic" in subset.column_names else "unknown"
        is_ood = bool(holdout_topics and topic in holdout_topics)
        model_prompt = render_generation_prompt(tokenizer, prompt, cfg, phase="eval")
        prompt_token_length = len(
            tokenizer(
                model_prompt,
                add_special_tokens=True,
                padding=False,
                truncation=False,
            )["input_ids"]
        )

        rollout_rows = []
        rollout_index = 0
        while rollout_index < num_rollouts:
            current_batch = min(batch_size, num_rollouts - rollout_index)
            model_prompts = [model_prompt] * current_batch
            enc = tokenizer(
                model_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=cfg.max_prompt_len,
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            prompt_width = int(enc["input_ids"].shape[1])

            with torch.no_grad():
                answer_constraints = get_answer_token_constraints(tokenizer, cfg, phase="eval")
                out_ids = _run_eval_generate(
                    model,
                    tokenizer,
                    enc,
                    max_new_tokens=max_completion_len,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    eos_id=eos_id,
                    answer_constraints=answer_constraints,
                )

            raw_completions = [
                tokenizer.decode(out_ids[i][prompt_width:], skip_special_tokens=True)
                for i in range(out_ids.shape[0])
            ]
            normalized_batch = [
                normalize_generated_completion(
                    raw_completion,
                    answer_prefix=generation_answer_prefix(cfg, phase="eval"),
                )
                for raw_completion in raw_completions
            ]
            novelty_texts = [
                resolve_novelty_text(
                    normalized,
                    fallback_completion=raw_completion,
                    novelty_text_contract=cfg.novelty_text_contract,
                )
                for normalized, raw_completion in zip(normalized_batch, raw_completions)
            ]
            novelty_batch = (
                [float(v) for v in novelty_scorer.score_batch(novelty_texts, train=False, phase=f"benchmark_{desc.lower()}")]
                if novelty_scorer is not None
                else [0.0 for _ in raw_completions]
            )

            for batch_idx, (raw_completion, normalized_completion, novelty_text, nov) in enumerate(
                zip(raw_completions, normalized_batch, novelty_texts, novelty_batch)
            ):
                scored_completion = str(normalized_completion.get("completion_scored", raw_completion))
                raw_completion_full = str(normalized_completion.get("completion_full_raw", raw_completion))
                generation_stats = _sequence_generation_stats(
                    out_ids[batch_idx],
                    prompt_width=prompt_width,
                    eos_id=eos_id,
                    pad_token_id=tokenizer.pad_token_id,
                    max_new_tokens=max_completion_len,
                )
                completion_meta = analyze_completion_text(scored_completion)
                quality_metrics = completion_quality_metrics(completion_meta, extra_flags=generation_stats)
                raw_completion_meta = analyze_completion_text(raw_completion_full)
                raw_quality_metrics = completion_quality_metrics(raw_completion_meta, extra_flags=generation_stats)
                presented_answer = extract_presented_answer(scored_completion, meta=completion_meta)
                canonical_completion = canonicalize_completion_for_training(scored_completion, meta=completion_meta)
                actual_target_canonical = build_actual_training_text(
                    normalized_completion,
                    fallback_completion=raw_completion,
                    target_contract="canonical",
                )
                actual_target_full_trace = build_actual_training_text(
                    normalized_completion,
                    fallback_completion=raw_completion,
                    target_contract="full_trace",
                )
                normalization_changed = raw_completion_full.strip() != scored_completion.strip()
                is_correct = bool(is_correct_completion(scored_completion, ground_truth))
                total_correct += int(is_correct)
                total += 1
                novelties.append(float(nov))
                if is_ood:
                    ood_correct += int(is_correct)
                    ood_total += 1
                    ood_nov.append(float(nov))
                else:
                    id_correct += int(is_correct)
                    id_total += 1
                    id_nov.append(float(nov))

                sample = {
                    "problem_idx": int(problem_idx),
                    "rollout_idx": int(rollout_index + batch_idx),
                    "prompt": prompt,
                    "completion": scored_completion,
                    "completion_raw": raw_completion,
                    "completion_full_raw": raw_completion_full,
                    "completion_first_line": str(normalized_completion.get("completion_first_line", "")),
                    "completion_scored": scored_completion,
                    "ground_truth": ground_truth,
                    "presented_answer": presented_answer,
                    "completion_canonical": canonical_completion,
                    "actual_target_canonical": actual_target_canonical,
                    "actual_target_full_trace": actual_target_full_trace,
                    "topic": topic,
                    "is_ood": is_ood,
                    "is_correct": is_correct,
                    "novelty": float(nov),
                    "novelty_text": novelty_text,
                    "novelty_text_contract": cfg.novelty_text_contract,
                    "format_quality": float(quality_metrics.get("quality", 0.0)),
                    "format_penalty": float(quality_metrics.get("penalty", 0.0)),
                    "format_penalty_breakdown": dict(quality_metrics.get("penalties", {})),
                    "raw_format_quality": float(raw_quality_metrics.get("quality", 0.0)),
                    "raw_format_penalty": float(raw_quality_metrics.get("penalty", 0.0)),
                    "raw_format_penalty_breakdown": dict(raw_quality_metrics.get("penalties", {})),
                    "normalization_changed": bool(normalization_changed),
                    "prompt_token_length": int(prompt_token_length),
                    "prompt_truncated": bool(int(prompt_token_length) > int(cfg.max_prompt_len)),
                    **generation_stats,
                    **completion_meta,
                }
                samples.append(sample)
                rollout_rows.append(sample)

            rollout_index += current_batch

        problem_rollouts[str(problem_idx)] = [
            {
                "generation": str(sample.get("completion_full_raw", sample.get("completion_raw", ""))),
                "answer": sample.get("presented_answer"),
                "is_correct": bool(sample.get("is_correct", False)),
                "rollout_idx": int(sample.get("rollout_idx", 0)),
            }
            for sample in rollout_rows
        ]
        correct_count = int(sum(1 for sample in rollout_rows if bool(sample.get("is_correct", False))))
        distinct_answers = len(
            {
                str(sample.get("presented_answer"))
                for sample in rollout_rows
                if sample.get("presented_answer") not in (None, "")
            }
        )
        distinct_answer_counts.append(float(distinct_answers))
        problem_summaries.append(
            {
                "problem_idx": int(problem_idx),
                "topic": topic,
                "is_ood": is_ood,
                "num_rollouts": int(num_rollouts),
                "num_correct": int(correct_count),
                "avg_correct": float(correct_count / max(1, num_rollouts)),
                "distinct_answer_count": int(distinct_answers),
                "presented_answers": [sample.get("presented_answer") for sample in rollout_rows],
            }
        )

    avg_accuracy = float(total_correct / max(1, total))
    pass_at_k = {
        str(k): float(
            _mean([_pass_at_k_estimate(num_rollouts, summary["num_correct"], k) for summary in problem_summaries])
        )
        for k in pass_k_values
    }
    mean_nov = float(sum(novelties) / len(novelties)) if novelties else 0.0
    primary_metric_name = f"avg@{num_rollouts}"
    primary_metric_value = avg_accuracy

    print(
        f"{desc} {primary_metric_name}: {primary_metric_value*100:.1f}% "
        f"({total_correct}/{max(1, total)} correct rollouts)"
    )

    if use_wandb and wandb_run is not None and wandb is not None:
        payload = {
            f"eval/{desc}_{primary_metric_name}": primary_metric_value,
            f"eval/{desc}_accuracy": primary_metric_value,
            f"eval/{desc}_mean_novelty": mean_nov,
            f"eval/{desc}_rollout_count": float(num_rollouts),
            f"eval/{desc}_distinct_answer_mean": _mean(distinct_answer_counts),
            f"eval/{desc}_temperature": float(temperature),
            f"eval/{desc}_top_p": float(top_p),
            f"eval/{desc}_max_new_tokens": float(max_completion_len),
            f"eval/{desc}_id_accuracy": (id_correct / max(1, id_total)),
            f"eval/{desc}_ood_accuracy": (ood_correct / max(1, ood_total)),
            f"eval/{desc}_id_mean_novelty": float(sum(id_nov) / len(id_nov)) if id_nov else 0.0,
            f"eval/{desc}_ood_mean_novelty": float(sum(ood_nov) / len(ood_nov)) if ood_nov else 0.0,
        }
        for k, value in pass_at_k.items():
            payload[f"eval/{desc}_pass@{k}"] = value
        if novelties:
            payload[f"eval/{desc}_novelty_hist"] = wandb.Histogram(novelties)
        wandb_run.log(payload)

    return {
        "accuracy": primary_metric_value,
        "primary_metric_name": primary_metric_name,
        "num_rollouts": int(num_rollouts),
        "rollout_temperature": float(temperature),
        "rollout_top_p": float(top_p),
        "rollout_max_completion_len": int(max_completion_len),
        "avg_at": {str(num_rollouts): primary_metric_value},
        "pass_at_k": pass_at_k,
        "distinct_answer_mean": _mean(distinct_answer_counts),
        "problem_count": int(len(problem_summaries)),
        "novelty_mean": mean_nov,
        "novelty_values": novelties,
        "correct": total_correct,
        "total": total,
        "id_accuracy": id_correct / max(1, id_total),
        "ood_accuracy": ood_correct / max(1, ood_total),
        "id_total": id_total,
        "ood_total": ood_total,
        "id_novelty_mean": float(sum(id_nov) / len(id_nov)) if id_nov else 0.0,
        "ood_novelty_mean": float(sum(ood_nov) / len(ood_nov)) if ood_nov else 0.0,
        "samples": samples,
        "problem_rollouts": problem_rollouts,
        "problem_summaries": problem_summaries,
    }


def plot_results(base_metrics,
                 explorer_metrics,
                 actual_metrics,
                 distill_losses,
                 explorer_log_hist,
                 actual_log_hist,
                 trajectories,
                 cfg,
                 use_wandb: bool = False,
                 wandb_run=None,
                 out_dir="plots"):
    base_acc = base_metrics["accuracy"]
    explorer_acc = explorer_metrics["accuracy"]
    actual_acc = actual_metrics["accuracy"]
    names = ["Base", "Explorer", "Actual"]
    accs = [base_acc, explorer_acc, actual_acc]

    os.makedirs(out_dir, exist_ok=True)

    if use_wandb and wandb_run is not None and wandb is not None:
        wandb_run.log({
            "final/base_accuracy": base_acc,
            "final/explorer_accuracy": explorer_acc,
            "final/actual_accuracy": actual_acc,
            "final/base_valid_answer_fraction": _mean([1.0 if bool(s.get("valid_answer_presentation", False)) else 0.0 for s in base_metrics.get("samples", [])]),
            "final/explorer_valid_answer_fraction": _mean([1.0 if bool(s.get("valid_answer_presentation", False)) else 0.0 for s in explorer_metrics.get("samples", [])]),
            "final/actual_valid_answer_fraction": _mean([1.0 if bool(s.get("valid_answer_presentation", False)) else 0.0 for s in actual_metrics.get("samples", [])]),
        })

    r_novel_all = [t["r_novel"] for t in trajectories]
    reward_key = "reward_actual" if any("reward_actual" in t for t in trajectories) else "reward"
    r_total_all = [t.get(reward_key, t["reward"]) for t in trajectories]
    is_correct_flags = [
        bool(t.get("is_correct", t.get("r_correct", 0.0) > 0.5)) for t in trajectories
    ]
    r_novel_correct = [t["r_novel"] for t, ok in zip(trajectories, is_correct_flags) if ok]
    r_novel_incorrect = [t["r_novel"] for t, ok in zip(trajectories, is_correct_flags) if not ok]
    layer_keys = sorted(
        {
            key
            for t in trajectories
            for key in (t.get("r_novel_layers", {}) or {}).keys()
        }
    )

    if use_wandb and wandb_run is not None and wandb is not None:
        payload = {}
        if r_novel_all:
            payload["traj/novelty_all"] = wandb.Histogram(r_novel_all)
        if r_novel_correct:
            payload["traj/novelty_correct"] = wandb.Histogram(r_novel_correct)
        if r_novel_incorrect:
            payload["traj/novelty_incorrect"] = wandb.Histogram(r_novel_incorrect)
        if payload:
            wandb_run.log(payload)
        if r_total_all:
            wandb_run.log({"traj/reward_all": wandb.Histogram(r_total_all)})

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] matplotlib not available ({e}); skipping local plots.")
        return

    plot_paths = {}

    def save_fig(fig, name):
        path = os.path.join(out_dir, name)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plot_paths[name] = path

    # Accuracy comparison
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(names, [100 * a for a in accs], color=["#4C78A8", "#F58518", "#54A24B"])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Accuracy comparison")
    save_fig(fig, "accuracy_comparison.png")

    format_quality_means = [
        _mean([float(s.get("format_quality", 0.0)) for s in base_metrics.get("samples", [])]),
        _mean([float(s.get("format_quality", 0.0)) for s in explorer_metrics.get("samples", [])]),
        _mean([float(s.get("format_quality", 0.0)) for s in actual_metrics.get("samples", [])]),
    ]
    valid_answer_fracs = [
        _mean([1.0 if bool(s.get("valid_answer_presentation", False)) else 0.0 for s in base_metrics.get("samples", [])]),
        _mean([1.0 if bool(s.get("valid_answer_presentation", False)) else 0.0 for s in explorer_metrics.get("samples", [])]),
        _mean([1.0 if bool(s.get("valid_answer_presentation", False)) else 0.0 for s in actual_metrics.get("samples", [])]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(names, format_quality_means, color=["#4C78A8", "#F58518", "#54A24B"])
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Mean format quality")
    axes[1].bar(names, valid_answer_fracs, color=["#4C78A8", "#F58518", "#54A24B"])
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Valid answer fraction")
    save_fig(fig, "answer_quality_comparison.png")

    # ID vs OOD accuracy comparison when holdout topics are configured.
    if (
        base_metrics.get("id_total", 0) > 0
        or explorer_metrics.get("id_total", 0) > 0
        or actual_metrics.get("id_total", 0) > 0
        or base_metrics.get("ood_total", 0) > 0
        or explorer_metrics.get("ood_total", 0) > 0
        or actual_metrics.get("ood_total", 0) > 0
    ):
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        x = list(range(len(names)))
        width = 0.36
        id_vals = [
            100.0 * float(base_metrics.get("id_accuracy", 0.0)),
            100.0 * float(explorer_metrics.get("id_accuracy", 0.0)),
            100.0 * float(actual_metrics.get("id_accuracy", 0.0)),
        ]
        ood_vals = [
            100.0 * float(base_metrics.get("ood_accuracy", 0.0)),
            100.0 * float(explorer_metrics.get("ood_accuracy", 0.0)),
            100.0 * float(actual_metrics.get("ood_accuracy", 0.0)),
        ]
        ax.bar([i - width / 2 for i in x], id_vals, width=width, label="ID", color="#4C78A8", alpha=0.9)
        ax.bar([i + width / 2 for i in x], ood_vals, width=width, label="OOD", color="#E45756", alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)
        ax.set_title("ID vs OOD accuracy")
        ax.legend()
        save_fig(fig, "id_ood_accuracy_comparison.png")

    # Accuracy vs novelty (mean) by model
    novelty_means = [
        base_metrics.get("novelty_mean", 0.0),
        explorer_metrics.get("novelty_mean", 0.0),
        actual_metrics.get("novelty_mean", 0.0),
    ]
    fig, ax1 = plt.subplots(figsize=(6.2, 4.2))
    ax1.bar(names, [100 * a for a in accs], color=["#4C78A8", "#F58518", "#54A24B"], alpha=0.85)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(0, 100)
    ax2 = ax1.twinx()
    ax2.plot(names, novelty_means, color="#B279A2", marker="o", linewidth=2, label="Mean novelty")
    ax2.set_ylabel("Mean novelty")
    ax1.set_title("Accuracy vs Novelty by Model")
    save_fig(fig, "accuracy_vs_novelty.png")

    # Novelty histogram by model (eval)
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, metrics, color in (
        ("Base", base_metrics, "#4C78A8"),
        ("Explorer", explorer_metrics, "#F58518"),
        ("Actual", actual_metrics, "#54A24B"),
    ):
        vals = metrics.get("novelty_values", [])
        if vals:
            ax.hist(vals, bins=30, alpha=0.45, label=label, color=color)
    ax.set_title("Eval novelty distributions")
    ax.set_xlabel("Novelty score")
    ax.set_ylabel("Count")
    ax.legend()
    save_fig(fig, "eval_novelty_hist.png")

    # Trajectory novelty hist (correct vs incorrect)
    if r_novel_all:
        fig, ax = plt.subplots(figsize=(6, 4))
        if r_novel_correct:
            ax.hist(r_novel_correct, bins=30, alpha=0.6, label="Correct", color="#54A24B")
        if r_novel_incorrect:
            ax.hist(r_novel_incorrect, bins=30, alpha=0.6, label="Incorrect", color="#E45756")
        ax.set_title("Trajectory novelty (correct vs incorrect)")
        ax.set_xlabel("Novelty score")
        ax.set_ylabel("Count")
        ax.legend()
        save_fig(fig, "trajectory_novelty_hist.png")

    # Trajectory reward hist
    if r_total_all:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(r_total_all, bins=30, color="#4C78A8", alpha=0.8)
        ax.set_title("Trajectory reward distribution")
        ax.set_xlabel("Reward")
        ax.set_ylabel("Count")
        save_fig(fig, "trajectory_reward_hist.png")

    if trajectories:
        traj_quality = [float(t.get("format_quality", t.get("reward_format_quality", 0.0)) or 0.0) for t in trajectories]
        traj_valid = [1.0 if bool(t.get("valid_answer_presentation", False)) else 0.0 for t in trajectories]
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        axes[0].hist(traj_quality, bins=20, color="#72B7B2", alpha=0.8)
        axes[0].set_title("Trajectory format quality")
        axes[0].set_xlabel("Quality")
        axes[1].bar(["invalid", "valid"], [len(traj_valid) - int(sum(traj_valid)), int(sum(traj_valid))], color=["#E45756", "#54A24B"])
        axes[1].set_title("Trajectory valid answers")
        save_fig(fig, "trajectory_answer_quality.png")

    # Reward vs novelty scatter
    if r_total_all and r_novel_all:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(r_novel_all, r_total_all, s=12, alpha=0.5, color="#72B7B2")
        ax.set_title("Reward vs novelty")
        ax.set_xlabel("Novelty score")
        ax.set_ylabel("Reward")
        save_fig(fig, "reward_vs_novelty.png")

    # Per-layer novelty summary (for multilayer novelty experiments)
    if layer_keys:
        means = []
        stds = []
        for key in layer_keys:
            vals = [float((t.get("r_novel_layers", {}) or {}).get(key, 0.0)) for t in trajectories]
            if vals:
                mean_v = float(sum(vals) / len(vals))
                var_v = float(sum((v - mean_v) ** 2 for v in vals) / max(1, len(vals)))
                std_v = var_v ** 0.5
            else:
                mean_v, std_v = 0.0, 0.0
            means.append(mean_v)
            stds.append(std_v)
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        ax.bar(layer_keys, means, yerr=stds, alpha=0.85, color="#B279A2")
        ax.set_title("Trajectory novelty by layer")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Novelty (mean +/- std)")
        ax.tick_params(axis="x", rotation=45)
        save_fig(fig, "trajectory_layer_novelty.png")

    # Reward component decomposition if present
    if trajectories and any("r_correct_used" in t for t in trajectories):
        corr_used = [float(t.get("r_correct_used", 0.0)) for t in trajectories]
        nov_used = [float(t.get("r_novel_used", 0.0)) for t in trajectories]
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        axes[0].hist(corr_used, bins=30, color="#4C78A8", alpha=0.8)
        axes[0].set_title("Correctness component (used)")
        axes[1].hist(nov_used, bins=30, color="#F58518", alpha=0.8)
        axes[1].set_title("Novelty component (used)")
        save_fig(fig, "reward_component_hist.png")

    def plot_series(ax, history, key, label):
        xs, ys = [], []
        for i, h in enumerate(history):
            if key in h:
                xs.append(h.get("step", i + 1))
                ys.append(h[key])
        if ys:
            ax.plot(xs, ys, label=label)

    # Explorer GRPO curves
    if explorer_log_hist:
        fig, axes = plt.subplots(2, 2, figsize=(9, 6))
        ax = axes[0, 0]
        plot_series(ax, explorer_log_hist, "reward", "reward")
        plot_series(ax, explorer_log_hist, "rewards/GRPOReward/mean", "env_reward")
        ax.set_title("Explorer reward")
        ax.legend()

        ax = axes[0, 1]
        plot_series(ax, explorer_log_hist, "loss", "loss")
        ax.set_title("Explorer loss")

        ax = axes[1, 0]
        plot_series(ax, explorer_log_hist, "entropy", "entropy")
        ax.set_title("Explorer entropy")

        ax = axes[1, 1]
        plot_series(ax, explorer_log_hist, "grad_norm", "grad_norm")
        ax.set_title("Explorer grad norm")

        save_fig(fig, "explorer_grpo_curves.png")

    # Actual offline GRPO curves
    if actual_log_hist:
        fig, axes = plt.subplots(2, 2, figsize=(9, 6))
        ax = axes[0, 0]
        plot_series(ax, actual_log_hist, "loss", "loss")
        plot_series(ax, actual_log_hist, "loss_pg", "loss_pg")
        plot_series(ax, actual_log_hist, "loss_kl", "loss_kl")
        ax.set_title("Actual GRPO loss")
        ax.legend()

        ax = axes[0, 1]
        plot_series(ax, actual_log_hist, "reward_mean", "reward_mean")
        plot_series(ax, actual_log_hist, "adv_mean", "adv_mean")
        ax.set_title("Actual reward/adv")
        ax.legend()

        ax = axes[1, 0]
        plot_series(ax, actual_log_hist, "kl_mean", "kl_mean")
        ax.set_title("Actual KL")

        ax = axes[1, 1]
        plot_series(ax, actual_log_hist, "mean_logp", "mean_logp")
        plot_series(ax, actual_log_hist, "mean_len", "mean_len")
        ax.set_title("Actual logp/len")
        ax.legend()

        save_fig(fig, "actual_grpo_curves.png")

    # Distillation loss curve (if used)
    if distill_losses:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(range(1, len(distill_losses) + 1), distill_losses, marker="o")
        ax.set_title("Actual distillation loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        save_fig(fig, "distill_loss.png")

    if use_wandb and wandb_run is not None and wandb is not None:
        for name, path in plot_paths.items():
            wandb_run.log({f"plots/{name}": wandb.Image(path)})
