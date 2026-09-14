import gc
import math
import re
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .embeddings import get_completion_features
from .devices import mark_step
from .utils import get_device


FINAL_ANSWER_PATTERN = r"(?:final answer\s*:|the final answer is|answer\s*:)"
ANSWER_CUE_PATTERN = r"(?:the answer is|answer\s*:)"
SIMPLE_NUMBER_PATTERN = r"-?\d[\d,]*(?:\.\d+)?"
SHORT_NUMERIC_ANSWER_PATTERN = rf"{SIMPLE_NUMBER_PATTERN}(?:\s*/\s*{SIMPLE_NUMBER_PATTERN})?%?"


def _extract_final_answer(completion: str) -> Optional[str]:
    text = completion or ""
    candidates = []

    # Pattern handles two levels of nested braces: \boxed{\frac{\sqrt{2}}{3}}, etc.
    _BRACE_L1 = r"[^{}]*(?:\{[^{}]*\}[^{}]*)*"
    _BOXED_INNER = r"[^{}]*(?:\{" + _BRACE_L1 + r"\}[^{}]*)*"
    candidate_patterns = (
        (rf"{FINAL_ANSWER_PATTERN}\s*\$?\\boxed\{{({_BOXED_INNER})\}}\$?", 4),
        (rf"{ANSWER_CUE_PATTERN}\s*\$?\\boxed\{{({_BOXED_INNER})\}}\$?", 4),
        (rf"\$?\\boxed\{{({_BOXED_INNER})\}}\$?", 3),
        (rf"{FINAL_ANSWER_PATTERN}\s*({SHORT_NUMERIC_ANSWER_PATTERN})", 3),
        (rf"{ANSWER_CUE_PATTERN}\s*({SHORT_NUMERIC_ANSWER_PATTERN})", 3),
        (rf"{FINAL_ANSWER_PATTERN}\s*([^\n]+)", 1),
        (rf"{ANSWER_CUE_PATTERN}\s*([^\n]+)", 1),
    )

    for pattern, priority in candidate_patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            value = match.group(1).strip().rstrip(".")
            if value:
                candidates.append((match.start(), priority, value))

    for match in re.finditer(rf"####\s*({SIMPLE_NUMBER_PATTERN})", text):
        candidates.append((match.start(), 2, match.group(1).strip().rstrip(".")))

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    final_line = lines[-1] if lines else ""
    if _is_short_final_line_candidate(final_line):
        candidates.append((len(text), 2, final_line.strip().rstrip(".")))

    if not candidates:
        return None

    # Prefer highest priority first, then latest position as tiebreaker.
    # This ensures \boxed{} (priority=3-4) always wins over a trailing
    # "answer: ..." cue (priority=1) that appears later in the text.
    candidates.sort(key=lambda item: (item[1], item[0]))
    return candidates[-1][2]


def _is_short_answer_like(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False
    # Strip markdown bold/italic before checking
    stripped = stripped.replace("**", "").replace("*", "").strip()
    if not stripped:
        return False
    compact = " ".join(stripped.split())
    patterns = (
        r"^(?:the\s+final\s+answer\s+is\s+)?\$?\\boxed\{[^{}]+\}\$?\.?$",
        rf"^(?:the\s+final\s+answer\s+is\s+)?{SIMPLE_NUMBER_PATTERN}%?\.?$",
        rf"^(?:the\s+final\s+answer\s+is\s+)?{SIMPLE_NUMBER_PATTERN}\s*/\s*{SIMPLE_NUMBER_PATTERN}\.?$",
        r"^(?:the\s+final\s+answer\s+is\s+)?[a-z0-9\-\.\+\*/\^\(\)=<>,:%| ]{1,32}\.?$",
    )
    return any(re.fullmatch(pattern, compact, flags=re.IGNORECASE) for pattern in patterns)


def _is_prefix_only_answer(text: str) -> bool:
    compact = " ".join((text or "").strip().split()).lower().rstrip(" .:")
    if not compact:
        return False
    return compact in {
        "the final answer is",
        "final answer",
        "final answer is",
        "answer",
    }


def _is_short_final_line_candidate(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped or _is_prefix_only_answer(stripped):
        return False
    # Strip markdown bold before bullet-point check to avoid rejecting **25**
    clean = stripped.replace("**", "").replace("*", "").strip()
    if re.search(r"^\s*(?:[-]|\d+\.)\s+", clean):
        return False
    return _is_short_answer_like(stripped)


def _normalize_text_answer(text: str) -> str:
    t = (text or "").strip().lower()
    if not t:
        return ""
    t = re.sub(r"^\s*(?:the\s+)?final answer(?:\s+is)?\s*:?\s*", "", t, flags=re.IGNORECASE)
    # Strip markdown bold/italic markers
    t = t.replace("**", "").replace("*", "")
    # Strip common units that models add to numeric answers
    t = re.sub(r"\b(?:ml|mg|kg|km|cm|mm|m|g|lb|oz|ft|in|hr|hrs|hours?|minutes?|days?|years?|dollars?|cents?|percent)\b", "", t)
    t = t.replace("$", "")
    t = t.replace("\\left", "").replace("\\right", "")
    boxed_re = re.compile(r"\\boxed\s*\{([^{}]+)\}")
    while True:
        new_t = boxed_re.sub(r"\1", t)
        if new_t == t:
            break
        t = new_t
    t = t.replace("\\%", "%").replace("%", "%")
    t = t.replace("\\pi", "pi")
    t = t.replace("^\\circ", "deg").replace("\\circ", "deg")
    t = t.replace("−", "-")

    # LaTeX fractions -> a/b (repeat until no nested \frac remains).
    frac_re = re.compile(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
    while True:
        new_t = frac_re.sub(r"(\1)/(\2)", t)
        if new_t == t:
            break
        t = new_t

    # Mixed numbers: -13\frac{1}{2} -> -27/2 ; 13\frac{1}{2} -> 27/2
    mixed_re = re.compile(r"(-?\d+)\s*\(\s*(\d+)\s*\)\s*/\s*\(\s*(\d+)\s*\)")
    while True:
        m = mixed_re.search(t)
        if not m:
            break
        whole = int(m.group(1))
        num = int(m.group(2))
        den = int(m.group(3))
        if den == 0:
            repl = f"{whole}/{den}"
        else:
            sign = -1 if whole < 0 else 1
            abs_whole = abs(whole)
            new_num = sign * (abs_whole * den + num)
            repl = f"{new_num}/{den}"
        t = t[: m.start()] + repl + t[m.end() :]
    t = re.sub(r"-\((\d+)\)/\((\d+)\)", r"-\1/\2", t)
    t = re.sub(r"\(([-+]?\d+)\)/\(([-+]?\d+)\)", r"\1/\2", t)

    t = t.replace("{", "").replace("}", "")
    t = re.sub(r"\s+", "", t)
    t = re.sub(r"[^a-z0-9\-\.\+\*/\^\(\)=<>,:%|]", "", t)
    return t.strip()


def _parse_simple_numeric(text: str) -> Optional[float]:
    if not text:
        return None
    compact = text.replace(",", "")
    if re.fullmatch(r"-?\d+(?:\.\d+)?", compact):
        return float(compact)
    if re.fullmatch(r"-?\d+/-?\d+", compact):
        num_s, den_s = compact.split("/", 1)
        den = float(den_s)
        if den == 0:
            return None
        return float(num_s) / den
    if re.fullmatch(r"-?\d+(?:\.\d+)?%", compact):
        return float(compact[:-1]) / 100.0
    return None


def _canonicalize_numeric_text(text: str) -> str:
    compact = (text or "").strip()
    if compact and re.fullmatch(SHORT_NUMERIC_ANSWER_PATTERN, compact):
        return compact.replace(",", "")
    return compact


def analyze_completion_text(completion: str) -> Dict[str, object]:
    text = completion or ""
    stripped = text.strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    final_line = lines[-1] if lines else ""
    lower = stripped.lower()

    has_final_answer_tag = bool(re.search(rf"^\s*{FINAL_ANSWER_PATTERN}", text, flags=re.IGNORECASE | re.MULTILINE))
    parsed_final_answer = _extract_final_answer(text)
    prefix_only = bool(has_final_answer_tag and parsed_final_answer is None and _is_prefix_only_answer(stripped))
    has_boxed = "\\boxed" in text
    has_latex = bool(
        has_boxed
        or "$" in text
        or "\\frac" in text
        or "\\text{" in text
        or "\\(" in text
        or "\\[" in text
    )
    has_markdown = bool(
        "###" in text
        or "**" in text
        or re.search(r"^\s*[-*]\s+", text, flags=re.MULTILINE)
        or re.search(r"^\s*\d+\.\s+", text, flags=re.MULTILINE)
    )
    has_multiple_choice = bool(
        re.search(r"(?:^|\n)\s*(?:\(?[A-D]\)?[\.\):])", text)
        or re.search(r"\b[A-D]\)\s", text)
    )
    mentions_instruction = any(
        phrase in lower
        for phrase in (
            "write exactly",
            "stop immediately",
            "do not use markdown",
            "answer line",
            "use at most",
            "solve the problem briefly",
            "step-by-step solution",
        )
    )
    starts_with_instruction_echo = lower.startswith(
        (
            "answer line",
            "write exactly",
            "do not use",
            "use at most",
            "stop immediately",
            "provide a clear",
        )
    )
    final_answer_on_last_line = bool(final_line) and bool(
        re.match(rf"^\s*{FINAL_ANSWER_PATTERN}", final_line, flags=re.IGNORECASE)
    )
    compact_answer_candidate = bool(
        not has_markdown
        and not has_multiple_choice
        and not mentions_instruction
        and not starts_with_instruction_echo
        and not prefix_only
        and len(lines) <= 2
        and len(re.findall(r"\S+", stripped)) <= 16
        and _is_short_answer_like(stripped)
    )
    final_line_answer_candidate = bool(
        parsed_final_answer is not None
        and _is_short_final_line_candidate(final_line)
        and not has_multiple_choice
    )
    valid_answer_presentation = bool(
        ((has_final_answer_tag or has_boxed) and parsed_final_answer is not None)
        or final_line_answer_candidate
        or compact_answer_candidate
    )

    if has_multiple_choice:
        format_bucket = "multiple_choice"
    elif starts_with_instruction_echo or (mentions_instruction and not has_final_answer_tag):
        format_bucket = "instruction_echo"
    elif has_markdown:
        format_bucket = "markdown_or_steps"
    elif has_boxed or has_latex:
        format_bucket = "latex_or_boxed"
    elif final_answer_on_last_line:
        format_bucket = "final_answer_last_line"
    elif has_final_answer_tag:
        format_bucket = "final_answer_not_last_line"
    else:
        format_bucket = "free_form"

    return {
        "parsed_final_answer": parsed_final_answer,
        "has_final_answer_tag": has_final_answer_tag,
        "prefix_only": prefix_only,
        "final_answer_on_last_line": final_answer_on_last_line,
        "compact_answer_candidate": compact_answer_candidate,
        "final_line_answer_candidate": final_line_answer_candidate,
        "valid_answer_presentation": valid_answer_presentation,
        "has_boxed": has_boxed,
        "has_latex": has_latex,
        "has_markdown": has_markdown,
        "has_multiple_choice": has_multiple_choice,
        "mentions_instruction": mentions_instruction,
        "starts_with_instruction_echo": starts_with_instruction_echo,
        "line_count": len(lines),
        "char_length": len(stripped),
        "word_count": len(re.findall(r"\S+", stripped)),
        "format_bucket": format_bucket,
        "preview": stripped[:200],
    }


def completion_quality_metrics(
    completion_or_meta,
    extra_flags: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    if isinstance(completion_or_meta, dict):
        meta = dict(completion_or_meta)
    else:
        meta = analyze_completion_text(str(completion_or_meta or ""))
    if extra_flags:
        meta.update(extra_flags)

    penalties: Dict[str, float] = {}

    if not bool(meta.get("has_final_answer_tag", False)) and not bool(meta.get("valid_answer_presentation", False)):
        penalties["missing_final_answer_tag"] = 0.30
    elif not bool(meta.get("final_answer_on_last_line", False)):
        penalties["final_answer_not_last_line"] = 0.10
    if bool(meta.get("prefix_only", False)):
        penalties["missing_answer_payload"] = 0.45

    if bool(meta.get("has_multiple_choice", False)):
        penalties["multiple_choice"] = 0.25
    if bool(meta.get("has_markdown", False)):
        penalties["markdown_or_steps"] = 0.20
    if (bool(meta.get("has_latex", False)) or bool(meta.get("has_boxed", False))) and not bool(meta.get("valid_answer_presentation", False)):
        penalties["latex_or_boxed"] = 0.15
    if bool(meta.get("mentions_instruction", False)) or bool(meta.get("starts_with_instruction_echo", False)):
        penalties["instruction_echo"] = 0.25
    if int(meta.get("line_count", 0) or 0) > 4:
        penalties["too_many_lines"] = 0.10
    if int(meta.get("word_count", 0) or 0) > 60:
        penalties["too_many_words"] = 0.10
    if bool(meta.get("clipped", False)):
        penalties["clipped"] = 0.35

    total_penalty = min(1.0, float(sum(penalties.values())))
    quality = max(0.0, 1.0 - total_penalty)
    return {
        "quality": float(quality),
        "penalty": float(total_penalty),
        "penalties": penalties,
    }


def extract_presented_answer(completion: str, meta: Optional[Dict[str, object]] = None) -> Optional[str]:
    completion = completion or ""
    meta = dict(meta or analyze_completion_text(completion))

    explicit = _extract_final_answer(completion)
    if explicit is not None:
        normalized = _canonicalize_numeric_text(_normalize_text_answer(explicit))
        return normalized or None

    if not bool(meta.get("compact_answer_candidate", False)):
        return None

    stripped = completion.strip()
    if not stripped:
        return None

    normalized = _canonicalize_numeric_text(_normalize_text_answer(stripped))
    return normalized or None


def canonicalize_completion_for_training(completion: str, meta: Optional[Dict[str, object]] = None) -> Optional[str]:
    answer = extract_presented_answer(completion, meta=meta)
    if not answer:
        return None
    return answer


def normalize_generated_completion(
    completion: str,
    *,
    answer_prefix: Optional[str] = None,
) -> Dict[str, object]:
    raw_completion = completion or ""
    raw_stripped = raw_completion.lstrip()
    prefix = (answer_prefix or "").strip()
    if prefix and not re.match(rf"^\s*{FINAL_ANSWER_PATTERN}", raw_stripped, flags=re.IGNORECASE):
        full_raw = f"{prefix} {raw_stripped}".strip()
    else:
        full_raw = raw_completion.strip()

    lines = [line.strip() for line in full_raw.splitlines() if line.strip()]
    first_line = lines[0] if lines else full_raw

    first_line_canonical = canonicalize_completion_for_training(first_line) if first_line else None
    full_raw_canonical = canonicalize_completion_for_training(full_raw) if full_raw else None
    scored_completion = full_raw_canonical or first_line_canonical or full_raw or raw_completion

    return {
        "completion_raw": raw_completion,
        "completion_full_raw": full_raw,
        "completion_first_line": first_line,
        "completion_scored": scored_completion,
        "completion_canonical": full_raw_canonical or first_line_canonical,
        "normalized_with_prefix": bool(prefix),
        "used_first_line": bool(first_line_canonical and not full_raw_canonical),
    }


def resolve_novelty_text(
    normalized: Dict[str, object],
    *,
    fallback_completion: str = "",
    novelty_text_contract: str = "full_raw",
) -> str:
    contract = (novelty_text_contract or "full_raw").lower()
    if contract == "scored":
        text = normalized.get("completion_scored")
    elif contract == "first_line":
        text = normalized.get("completion_first_line") or normalized.get("completion_scored")
    else:
        text = normalized.get("completion_full_raw") or normalized.get("completion_scored")
    value = str(text or fallback_completion or "").strip()
    if value:
        return value
    return str(normalized.get("completion_scored", fallback_completion or "") or "").strip()


def build_actual_training_text(
    normalized: Dict[str, object],
    *,
    fallback_completion: str = "",
    target_contract: str = "canonical",
) -> Optional[str]:
    contract = (target_contract or "canonical").lower()
    if contract == "full_trace":
        text = normalized.get("completion_full_raw") or normalized.get("completion_scored") or fallback_completion
        value = str(text or "").strip()
        return value or None

    text = normalized.get("completion_canonical")
    if not text:
        text = canonicalize_completion_for_training(
            str(normalized.get("completion_scored", fallback_completion or "")),
        )
    value = str(text or "").strip()
    return value or None


def is_correct_completion(completion: str, ground_truth: str) -> bool:
    """
    Parse model output and compare with ground truth.
    Only accept answers that are actually presented as answers, not hidden
    integers buried inside long reasoning traces.
    """
    gt_norm = _normalize_text_answer(ground_truth)
    meta = analyze_completion_text(completion)
    pred_norm = extract_presented_answer(completion, meta=meta)
    if not pred_norm or not gt_norm:
        return False

    pred_num = _parse_simple_numeric(pred_norm)
    gt_num = _parse_simple_numeric(gt_norm)
    if pred_num is not None and gt_num is not None:
        return abs(pred_num - gt_num) <= 1e-6

    return pred_norm == gt_norm



class NoveltyScorer:
    def __init__(
        self,
        embed_model,
        tokenizer,
        rnd_module,
        metric: str,
        clip_value,
        device,
        use_input_embeddings: bool = False,
        feature_source: str = "last",
        layers: Optional[Sequence[int]] = None,
        layer_pool: str = "mean",
        layer_agg: str = "mean",
        max_length: Optional[int] = None,
    ):
        self.embed_model = embed_model
        self.tokenizer = tokenizer
        self.metric = metric
        self.clip_value = clip_value
        self.device = device
        self.use_input_embeddings = use_input_embeddings
        self.feature_source = (feature_source or "last").lower()
        self.layers = list(layers) if layers else []
        self.layer_pool = (layer_pool or "mean").lower()
        self.layer_agg = (layer_agg or "mean").lower()
        self.max_length = int(max(8, max_length or 128))
        self.last_feature_stats: Dict[str, object] = {}
        self.feature_stats_by_phase: Dict[str, Dict[str, float]] = {}

        if isinstance(rnd_module, dict):
            self.rnd_map = rnd_module
        elif rnd_module is None:
            self.rnd_map = {}
        else:
            self.rnd_map = {"last": rnd_module}

    def _get_rnd(self, key: str):
        if key in self.rnd_map:
            return self.rnd_map[key]
        if "last" in self.rnd_map:
            return self.rnd_map["last"]
        return None

    def _aggregate_layers(self, per_layer_values: Dict[str, List[float]]) -> Tuple[List[float], List[Dict[str, float]]]:
        if not per_layer_values:
            return [], []

        keys = sorted(per_layer_values.keys())
        n = len(next(iter(per_layer_values.values())))

        if self.layer_agg == "weighted":
            # Default weighted mode: emphasize deeper layers mildly.
            raw_weights = [i + 1 for i, _ in enumerate(keys)]
            total_w = float(sum(raw_weights))
            weights = [w / total_w for w in raw_weights]
        else:
            weights = [1.0 / len(keys) for _ in keys]

        merged = []
        details = []
        for idx in range(n):
            layer_detail = {k: float(per_layer_values[k][idx]) for k in keys}
            v = 0.0
            for w, k in zip(weights, keys):
                v += w * layer_detail[k]
            merged.append(float(v))
            details.append(layer_detail)
        return merged, details

    def _record_feature_stats(self, phase: str, stats: Dict[str, object]) -> None:
        if not stats:
            return
        self.last_feature_stats = dict(stats)
        phase_key = str(phase or "default")
        bucket = self.feature_stats_by_phase.setdefault(
            phase_key,
            {
                "count": 0.0,
                "truncated_count": 0.0,
                "token_length_sum": 0.0,
                "max_input_tokens": 0.0,
                "max_length": float(self.max_length),
            },
        )
        count = float(stats.get("count", 0.0) or 0.0)
        truncated_count = float(stats.get("truncated_count", 0.0) or 0.0)
        token_length_sum = float(stats.get("token_length_sum", 0.0) or 0.0)
        max_input_tokens = float(stats.get("max_input_tokens", 0.0) or 0.0)
        bucket["count"] += count
        bucket["truncated_count"] += truncated_count
        bucket["token_length_sum"] += token_length_sum
        bucket["max_input_tokens"] = max(bucket.get("max_input_tokens", 0.0), max_input_tokens)
        bucket["max_length"] = float(stats.get("max_length", self.max_length) or self.max_length)

    def snapshot_feature_stats(self) -> Dict[str, Dict[str, float]]:
        snapshot: Dict[str, Dict[str, float]] = {}
        for phase, stats in self.feature_stats_by_phase.items():
            count = float(stats.get("count", 0.0) or 0.0)
            truncated_count = float(stats.get("truncated_count", 0.0) or 0.0)
            snapshot[phase] = {
                "count": count,
                "truncated_count": truncated_count,
                "truncated_fraction": float(truncated_count / max(1.0, count)),
                "mean_input_tokens": float(stats.get("token_length_sum", 0.0) / max(1.0, count)),
                "max_input_tokens": float(stats.get("max_input_tokens", 0.0) or 0.0),
                "max_length": float(stats.get("max_length", self.max_length) or self.max_length),
            }
        return snapshot

    def score_batch_with_details(self, completions: Sequence[str], train: bool, phase: str = "default"):
        if self.embed_model is None or self.tokenizer is None or not completions:
            return [0.0 for _ in completions], [{} for _ in completions]

        device = self.device or get_device(self.embed_model)
        is_xla = getattr(device, "type", "") == "xla"
        use_input_embeddings = self.use_input_embeddings and self.feature_source != "multilayer"

        feat_map, feature_stats = get_completion_features(
            list(completions),
            model=self.embed_model,
            tokenizer=self.tokenizer,
            device=device,
            max_length=self.max_length,
            use_input_embeddings=use_input_embeddings,
            feature_source=self.feature_source,
            layers=self.layers,
            layer_pool=self.layer_pool,
            return_stats=True,
        )
        self._record_feature_stats(phase, feature_stats)

        per_layer_values: Dict[str, List[float]] = {}
        for key, emb in feat_map.items():
            rnd = self._get_rnd(key)
            if rnd is None:
                per_layer_values[key] = [0.0 for _ in completions]
                continue
            novelty_vec = rnd.compute_novelty(
                emb.to(device),
                train=train,
                metric=self.metric,
                clip_value=self.clip_value,
                return_vec=True,
            )
            per_layer_values[key] = [float(v) for v in novelty_vec.detach().cpu().tolist()]
            del novelty_vec
            del emb
            gc.collect()
            mark_step(is_xla=is_xla)

        del feat_map
        gc.collect()
        mark_step(is_xla=is_xla)

        merged, details = self._aggregate_layers(per_layer_values)
        if not merged:
            merged = [0.0 for _ in completions]
            details = [{} for _ in completions]
        return merged, details

    def score(self, completion: str, train: bool) -> float:
        vals, _ = self.score_batch_with_details([completion], train=train)
        return float(vals[0]) if vals else 0.0

    def score_batch(self, completions: Sequence[str], train: bool, phase: str = "default") -> Sequence[float]:
        vals, _ = self.score_batch_with_details(completions, train=train, phase=phase)
        return vals


class AnchorPenaltyScorer:
    def __init__(
        self,
        current_model,
        reference_model,
        tokenizer,
        metric: str,
        device,
        max_length: int,
        trainable_only: bool = True,
    ):
        self.current_model = current_model
        self.reference_model = reference_model
        self.tokenizer = tokenizer
        self.metric = (metric or "none").lower()
        self.device = device
        self.max_length = int(max(1, max_length))
        self.trainable_only = bool(trainable_only)

        if self.reference_model is not None:
            self.reference_model.eval()
            for param in self.reference_model.parameters():
                param.requires_grad = False

    def _zeros(self, n: int):
        return [0.0 for _ in range(n)], [{} for _ in range(n)]

    def _param_l2_value(self) -> float:
        if self.reference_model is None or self.current_model is None:
            return 0.0
        sq_sum = 0.0
        count = 0
        with torch.no_grad():
            for param, ref_param in zip(self.current_model.parameters(), self.reference_model.parameters()):
                if self.trainable_only and not param.requires_grad:
                    continue
                diff = (param.detach() - ref_param.detach()).float()
                sq_sum += float(diff.pow(2).sum().item())
                count += int(diff.numel())
        return math.sqrt(sq_sum / max(1, count))

    def _output_kl_scores(self, prompts: Sequence[str], completions: Sequence[str]):
        if self.reference_model is None or self.current_model is None:
            return self._zeros(len(completions))
        texts = [f"{prompt}{completion}" for prompt, completion in zip(prompts, completions)]
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        prompt_enc = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        prompt_mask = prompt_enc["attention_mask"]

        with torch.no_grad():
            cur_logits = self.current_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits[:, :-1, :]
            ref_logits = self.reference_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits[:, :-1, :]

        cur_log_probs = F.log_softmax(cur_logits.float(), dim=-1)
        ref_probs = F.softmax(ref_logits.float(), dim=-1)
        token_kl = F.kl_div(cur_log_probs, ref_probs, reduction="none").sum(dim=-1)

        seq_lens = attention_mask.sum(dim=1).cpu().tolist()
        prompt_lens = prompt_mask.sum(dim=1).cpu().tolist()
        total_width = input_ids.size(1)
        score_values = []
        details = []
        for row_idx, (seq_len_val, prompt_len_val) in enumerate(zip(seq_lens, prompt_lens)):
            seq_len = int(seq_len_val)
            prompt_len = int(prompt_len_val)
            left_pad = total_width - seq_len
            start = max(0, left_pad + max(prompt_len - 1, 0))
            end = max(start + 1, left_pad + seq_len - 1)
            value = float(token_kl[row_idx, start:end].mean().item()) if end > start else 0.0
            score_values.append(value)
            details.append({"output_kl": value})
        return score_values, details

    def score_batch(self, prompts: Sequence[str], completions: Sequence[str], include_param_metric: bool = False):
        n = min(len(prompts), len(completions))
        prompts = list(prompts[:n])
        completions = list(completions[:n])
        if n == 0 or self.metric == "none":
            return self._zeros(n)
        if self.metric == "output_kl":
            return self._output_kl_scores(prompts, completions)
        if self.metric == "param_l2":
            value = self._param_l2_value() if include_param_metric else 0.0
            return [value for _ in range(n)], [{"param_l2": value} for _ in range(n)]
        return self._zeros(n)


class GRPOReward:
    """Callable reward function for TRL GRPOTrainer."""

    def __init__(
        self,
        novelty_scorer: NoveltyScorer,
        lambda_novelty: float,
        reward_correct: float = 1.0,
        reward_incorrect: float = 0.0,
        reward_shape: str = "linear",
        incorrect_novelty_scale: float = 0.0,
        piecewise_bounds: Optional[Sequence[float]] = None,
        novelty_zscore: str = "off",
        correctness_zscore: str = "off",
        anchor_penalty_scorer: Optional[AnchorPenaltyScorer] = None,
        anchor_weight: float = 0.0,
        format_penalty_weight: float = 0.2,
        gate_novelty_by_quality: bool = True,
        answer_prefix: Optional[str] = None,
        novelty_text_contract: str = "full_raw",
        novelty_phase: str = "default",
    ):
        self.novelty_scorer = novelty_scorer
        self.lambda_novelty = lambda_novelty
        self.reward_correct = reward_correct
        self.reward_incorrect = reward_incorrect
        self.reward_shape = (reward_shape or "linear").lower()
        self.incorrect_novelty_scale = float(max(0.0, incorrect_novelty_scale or 0.0))
        self.piecewise_bounds = list(piecewise_bounds) if piecewise_bounds else [-1.0, -0.5, 0.0, 0.5, 1.0]
        self.novelty_zscore = (novelty_zscore or "off").lower()
        self.correctness_zscore = (correctness_zscore or "off").lower()
        self.anchor_penalty_scorer = anchor_penalty_scorer
        self.anchor_weight = float(anchor_weight or 0.0)
        self.format_penalty_weight = float(format_penalty_weight or 0.0)
        self.gate_novelty_by_quality = bool(gate_novelty_by_quality)
        self.answer_prefix = answer_prefix if answer_prefix is not None else ""
        self.novelty_text_contract = (novelty_text_contract or "full_raw").lower()
        self.novelty_phase = novelty_phase or "default"

        if len(self.piecewise_bounds) != 5:
            raise ValueError("piecewise_bounds must have exactly 5 values.")
        for i in range(4):
            if self.piecewise_bounds[i] >= self.piecewise_bounds[i + 1]:
                raise ValueError("piecewise_bounds must be strictly increasing.")

        self._nov_ema = {"mean": 0.0, "std": 1.0, "initialized": False}
        self._corr_ema = {"mean": 0.0, "std": 1.0, "initialized": False}
        self.last_batch_summary: Dict[str, float] = {}

    @staticmethod
    def _stats(values: Sequence[float]) -> Tuple[float, float]:
        if not values:
            return 0.0, 1.0
        mean = float(sum(values) / len(values))
        var = float(sum((v - mean) ** 2 for v in values) / max(1, len(values)))
        std = math.sqrt(max(var, 1e-12))
        return mean, std

    def _normalize(self, values: Sequence[float], mode: str, state: Dict[str, float]) -> Tuple[List[float], float, float]:
        vals = [float(v) for v in values]
        if not vals:
            return [], 0.0, 1.0

        mode = (mode or "off").lower()
        mean, std = self._stats(vals)

        if mode == "off":
            return vals, mean, std

        if mode == "batch":
            norm = [(v - mean) / max(std, 1e-8) for v in vals]
            return norm, mean, std

        # EMA mode
        alpha = 0.05
        if not state.get("initialized", False):
            state["mean"] = mean
            state["std"] = std
            state["initialized"] = True
        else:
            state["mean"] = (1.0 - alpha) * float(state["mean"]) + alpha * mean
            state["std"] = (1.0 - alpha) * float(state["std"]) + alpha * std

        ref_mean = float(state["mean"])
        ref_std = max(float(state["std"]), 1e-8)
        norm = [(v - ref_mean) / ref_std for v in vals]
        return norm, ref_mean, ref_std

    @staticmethod
    def _sanitize_finite(values: Sequence[float], default: float = 0.0) -> List[float]:
        sanitized: List[float] = []
        for value in values:
            try:
                parsed = float(value)
            except Exception:
                parsed = float(default)
            if not math.isfinite(parsed):
                parsed = float(default)
            sanitized.append(parsed)
        return sanitized

    def _piecewise_reward(self, is_correct: bool, novelty_signal: float) -> Tuple[float, str]:
        a, b, c, d, e = self.piecewise_bounds
        is_novel = novelty_signal > 0.0

        if is_correct and is_novel:
            low, high, bucket = d, e, "D_correct_novel"
        elif is_correct and not is_novel:
            low, high, bucket = c, d, "C_correct_not_novel"
        elif (not is_correct) and is_novel:
            low, high, bucket = b, c, "B_incorrect_novel"
        else:
            low, high, bucket = a, b, "A_incorrect_not_novel"

        return float((low + high) / 2.0), bucket

    def compute_batch(
        self,
        completions,
        ground_truth,
        prompts=None,
        completion_stats=None,
        train: bool = True,
        return_details: bool = False,
    ):
        completions = list(completions or [])
        ground_truth = list(ground_truth or [])
        prompts = list(prompts or [])
        completion_stats = list(completion_stats or [])
        if completions and ground_truth and len(ground_truth) != len(completions):
            factor = max(1, len(completions) // len(ground_truth))
            expanded = []
            for gt in ground_truth:
                expanded.extend([gt] * factor)
            while len(expanded) < len(completions):
                expanded.append(ground_truth[-1])
            ground_truth = expanded[:len(completions)]

        n = min(len(completions), len(ground_truth))
        completions = completions[:n]
        ground_truth = ground_truth[:n]
        if completion_stats:
            if len(completion_stats) != len(completions):
                factor = max(1, len(completions) // max(1, len(completion_stats)))
                expanded_stats = []
                for item in completion_stats:
                    expanded_stats.extend([item] * factor)
                while len(expanded_stats) < len(completions):
                    expanded_stats.append(completion_stats[-1])
                completion_stats = expanded_stats[:len(completions)]
            else:
                completion_stats = completion_stats[:len(completions)]
        else:
            completion_stats = [{} for _ in range(n)]
        if n == 0:
            return ([] if not return_details else ([], []))

        normalized = [
            normalize_generated_completion(completion, answer_prefix=self.answer_prefix)
            for completion in completions
        ]
        raw_completions = completions
        completions = [str(item.get("completion_scored", "")) for item in normalized]
        novelty_texts = [
            resolve_novelty_text(
                item,
                fallback_completion=raw_completion,
                novelty_text_contract=self.novelty_text_contract,
            )
            for item, raw_completion in zip(normalized, raw_completions)
        ]

        if self.lambda_novelty == 0 or self.novelty_scorer is None:
            r_nov_raw = [0.0 for _ in range(n)]
            r_nov_layers = [{} for _ in range(n)]
        else:
            import time as _time
            _nov_t0 = _time.monotonic()
            print(f"[reward] novelty scoring {n} completions...", flush=True)
            r_nov_raw, r_nov_layers = self.novelty_scorer.score_batch_with_details(
                novelty_texts,
                train=train,
                phase=self.novelty_phase,
            )
            print(f"[reward] novelty scoring done in {_time.monotonic() - _nov_t0:.1f}s", flush=True)
        r_nov_raw = self._sanitize_finite(r_nov_raw, default=0.0)

        # Analyze format quality on the RAW completions (not canonical "42").
        completion_metas = [analyze_completion_text(c) for c in raw_completions]
        quality_info = [
            completion_quality_metrics(
                meta,
                extra_flags=completion_stats[i] if i < len(completion_stats) else None,
            )
            for i, meta in enumerate(completion_metas)
        ]
        quality_scores = self._sanitize_finite([float(info["quality"]) for info in quality_info], default=0.0)
        format_penalties = self._sanitize_finite([float(info["penalty"]) for info in quality_info], default=1.0)

        if prompts and self.anchor_penalty_scorer is not None and self.anchor_weight > 0:
            r_anchor_raw, r_anchor_details = self.anchor_penalty_scorer.score_batch(
                prompts,
                completions,
                include_param_metric=return_details,
            )
        else:
            r_anchor_raw = [0.0 for _ in range(n)]
            r_anchor_details = [{} for _ in range(n)]
        r_anchor_raw = self._sanitize_finite(r_anchor_raw, default=0.0)

        # Check correctness on raw completions so answer extraction sees the
        # full reasoning trace with \boxed{}, not the pre-extracted canonical.
        is_correct_flags = [is_correct_completion(c, gt) for c, gt in zip(raw_completions, ground_truth)]
        r_corr_raw = [
            self.reward_correct if flag else self.reward_incorrect
            for flag in is_correct_flags
        ]
        r_corr_raw = self._sanitize_finite(r_corr_raw, default=float(self.reward_incorrect))

        r_corr_used, corr_center, corr_scale = self._normalize(r_corr_raw, self.correctness_zscore, self._corr_ema)
        r_nov_used, nov_center, nov_scale = self._normalize(r_nov_raw, self.novelty_zscore, self._nov_ema)
        r_corr_used = self._sanitize_finite(r_corr_used, default=float(self.reward_incorrect))
        r_nov_used = self._sanitize_finite(r_nov_used, default=0.0)

        if self.novelty_zscore == "off":
            mean_raw_nov = self._stats(r_nov_raw)[0]
            novelty_gate = self._sanitize_finite([v - mean_raw_nov for v in r_nov_raw], default=0.0)
        else:
            novelty_gate = self._sanitize_finite(list(r_nov_used), default=0.0)

        novelty_reward_values = self._sanitize_finite(list(r_nov_used), default=0.0)
        if self.gate_novelty_by_quality:
            novelty_reward_values = self._sanitize_finite([
                float(q * v)
                for q, v in zip(quality_scores, novelty_reward_values)
            ], default=0.0)
            novelty_gate = self._sanitize_finite([
                float(q * v)
                for q, v in zip(quality_scores, novelty_gate)
            ], default=0.0)
        novelty_reward_values = self._sanitize_finite([
            float(v if is_correct else self.incorrect_novelty_scale * v)
            for v, is_correct in zip(novelty_reward_values, is_correct_flags)
        ], default=0.0)

        rewards: List[float] = []
        details: List[Dict[str, object]] = []
        for i in range(n):
            if self.reward_shape == "piecewise":
                reward_i, bucket = self._piecewise_reward(is_correct_flags[i], novelty_gate[i])
            else:
                reward_i = float(r_corr_used[i] + self.lambda_novelty * novelty_reward_values[i])
                reward_i -= float(self.format_penalty_weight * format_penalties[i])
                if (
                    self.anchor_penalty_scorer is not None
                    and self.anchor_weight > 0
                    and self.anchor_penalty_scorer.metric == "output_kl"
                ):
                    reward_i -= float(self.anchor_weight * r_anchor_raw[i])
                bucket = "linear"

            if self.reward_shape == "piecewise":
                reward_i -= float(self.format_penalty_weight * format_penalties[i])

            if not math.isfinite(reward_i):
                reward_i = float(self.reward_incorrect) - float(self.format_penalty_weight * format_penalties[i])
                bucket = f"{bucket}_nonfinite_fallback"

            rewards.append(float(reward_i))
            details.append(
                {
                    "is_correct": bool(is_correct_flags[i]),
                    "r_correct_raw": float(r_corr_raw[i]),
                    "r_correct_used": float(r_corr_used[i]),
                    "r_novel_raw": float(r_nov_raw[i]),
                    "r_novel_used": float(r_nov_used[i]),
                    "r_novel_reward": float(novelty_reward_values[i]),
                    "incorrect_novelty_scale": float(self.incorrect_novelty_scale),
                    "novelty_gate": float(novelty_gate[i]),
                    "format_quality": float(quality_scores[i]),
                    "format_penalty": float(format_penalties[i]),
                    "format_penalty_weight": float(self.format_penalty_weight),
                    "format_penalty_breakdown": dict(quality_info[i].get("penalties", {})),
                    "completion_stats": dict(completion_stats[i]) if i < len(completion_stats) else {},
                    "reward_bucket": bucket,
                    "reward_shape": self.reward_shape,
                    "r_novel_layers": dict(r_nov_layers[i]) if i < len(r_nov_layers) else {},
                    "r_anchor_raw": float(r_anchor_raw[i]) if i < len(r_anchor_raw) else 0.0,
                    "anchor_metric": self.anchor_penalty_scorer.metric if self.anchor_penalty_scorer is not None else "none",
                    "anchor_weight": float(self.anchor_weight),
                    "anchor_details": dict(r_anchor_details[i]) if i < len(r_anchor_details) else {},
                    "completion_raw": str(normalized[i].get("completion_raw", raw_completions[i])),
                    "completion_full_raw": str(normalized[i].get("completion_full_raw", raw_completions[i])),
                    "completion_first_line": str(normalized[i].get("completion_first_line", "")),
                    "completion_scored": str(normalized[i].get("completion_scored", completions[i])),
                    "completion_canonical": normalized[i].get("completion_canonical"),
                    "novelty_text": novelty_texts[i],
                    "novelty_text_contract": self.novelty_text_contract,
                    "used_first_line": bool(normalized[i].get("used_first_line", False)),
                    "reward_total": float(reward_i),
                }
            )

        self.last_batch_summary = {
            "reward/correctness_raw_mean": self._stats(r_corr_raw)[0],
            "reward/novelty_raw_mean": self._stats(r_nov_raw)[0],
            "reward/correctness_used_mean": self._stats(r_corr_used)[0],
            "reward/novelty_used_mean": self._stats(r_nov_used)[0],
            "reward/novelty_reward_mean": self._stats(novelty_reward_values)[0],
            "reward/incorrect_novelty_scale": float(self.incorrect_novelty_scale),
            "reward/final_mean": self._stats(rewards)[0],
            "reward/correctness_center": float(corr_center),
            "reward/correctness_scale": float(corr_scale),
            "reward/novelty_center": float(nov_center),
            "reward/novelty_scale": float(nov_scale),
            "reward/anchor_raw_mean": self._stats(r_anchor_raw)[0],
            "reward/format_quality_mean": self._stats(quality_scores)[0],
            "reward/format_penalty_mean": self._stats(format_penalties)[0],
        }
        if completion_stats:
            clipped_count = sum(1.0 for stats in completion_stats if bool(dict(stats).get("clipped", False)))
            terminated_count = sum(1.0 for stats in completion_stats if bool(dict(stats).get("terminated", False)))
            token_lengths = [
                float(dict(stats).get("completion_token_length", 0.0) or 0.0)
                for stats in completion_stats
            ]
            self.last_batch_summary["reward/clipped_fraction"] = float(clipped_count / max(1.0, float(len(completion_stats))))
            self.last_batch_summary["reward/terminated_fraction"] = float(terminated_count / max(1.0, float(len(completion_stats))))
            self.last_batch_summary["reward/completion_token_length_mean"] = self._stats(token_lengths)[0]
            terminated_mask = [bool(dict(s).get("terminated", False)) for s in completion_stats]
            t_rewards = [v for v, m in zip(rewards, terminated_mask) if m]
            t_corr = [v for v, m in zip(r_corr_used, terminated_mask) if m]
            t_nov = [v for v, m in zip(r_nov_used, terminated_mask) if m]
            self.last_batch_summary["reward/terminated_final_mean"] = self._stats(t_rewards)[0]
            self.last_batch_summary["reward/terminated_correctness_mean"] = self._stats(t_corr)[0]
            self.last_batch_summary["reward/terminated_novelty_mean"] = self._stats(t_nov)[0]
            self.last_batch_summary["reward/terminated_count"] = float(len(t_rewards))
        if self.novelty_scorer is not None:
            feature_stats = self.novelty_scorer.last_feature_stats or {}
            if feature_stats:
                count = float(feature_stats.get("count", 0.0) or 0.0)
                truncated = float(feature_stats.get("truncated_count", 0.0) or 0.0)
                self.last_batch_summary["reward/novelty_feature_truncated_fraction"] = float(truncated / max(1.0, count))
                self.last_batch_summary["reward/novelty_feature_mean_input_tokens"] = float(
                    feature_stats.get("token_length_sum", 0.0) / max(1.0, count)
                )
                self.last_batch_summary["reward/novelty_feature_max_input_tokens"] = float(
                    feature_stats.get("max_input_tokens", 0.0) or 0.0
                )
                self.last_batch_summary["reward/novelty_feature_max_length"] = float(
                    feature_stats.get("max_length", 0.0) or 0.0
                )

        if return_details:
            return rewards, details
        return rewards

    def compute_single(self, completion: str, ground_truth: str, train: bool = False):
        rewards, details = self.compute_batch([completion], [ground_truth], prompts=[""], train=train, return_details=True)
        if not rewards:
            return 0.0, {}
        return float(rewards[0]), details[0]

    def __call__(self, prompts=None, completions=None, ground_truth=None, completion_stats=None, **kwargs):
        completions = completions or []
        ground_truth = ground_truth or []
        prompts = prompts or []
        return self.compute_batch(
            completions,
            ground_truth,
            prompts=prompts,
            completion_stats=completion_stats,
            train=True,
            return_details=False,
        )
