"""Host-side reward and trajectory helpers for JAX GRPO.

This is intentionally pure Python: it mirrors the research reward contract
without depending on the Torch package being installed in the JAX TPU env.
"""

from __future__ import annotations

import math
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Dict, Optional, Sequence, Tuple


FINAL_ANSWER_PATTERN = r"(?:final answer\s*:|the final answer is|answer\s*:)"
ANSWER_CUE_PATTERN = r"(?:the answer is|answer\s*:)"
REWARDING_CONTRACT_VERSION = "expdis_jax.rewarding@gpu_repro_v2_exact_rational"
# This grammar is deliberately narrow.  It covers the numeric forms promised
# by the evaluation protocol without turning the verifier into a general
# expression evaluator (or accepting a numeric prefix of a symbolic answer).
SIMPLE_NUMBER_PATTERN = (
    r"[+-]?(?:(?:\d[\d,]*(?:\.\d*)?)|(?:\.\d+))"
    r"(?:[eE][+-]?\d+)?"
)
SHORT_NUMERIC_ANSWER_PATTERN = (
    rf"{SIMPLE_NUMBER_PATTERN}(?:\s*/\s*{SIMPLE_NUMBER_PATTERN})?%?"
)
_LATEX_FRAC_COMMAND = re.compile(r"\\(?:dfrac|tfrac|frac)\s*")
_LATEX_SCIENTIFIC_PATTERN = re.compile(
    rf"(?P<coefficient>{SIMPLE_NUMBER_PATTERN})\s*"
    r"(?:\\times|\\cdot|×)\s*10\s*\^\s*"
    r"(?:\{\s*(?P<braced_exp>[+-]?\d+)\s*\}|(?P<plain_exp>[+-]?\d+))",
    flags=re.IGNORECASE,
)
NUMERIC_ABSOLUTE_TOLERANCE = Fraction(1, 1_000_000)
_MAX_NUMERIC_LITERAL_CHARS = 256
_MAX_NUMERIC_DIGITS = 256
_MAX_ABS_DECIMAL_EXPONENT = 1_000


def _balanced_group(text: str, start: int, opener: str, closer: str) -> Optional[Tuple[str, int]]:
    """Return a balanced group's contents and exclusive end, or ``None``."""
    if start >= len(text) or text[start] != opener:
        return None
    depth = 1
    i = start + 1
    while i < len(text):
        if text[i] == opener:
            depth += 1
        elif text[i] == closer:
            depth -= 1
            if depth == 0:
                return text[start + 1:i], i + 1
        i += 1
    return None


def _latex_fraction_at(text: str, start: int = 0) -> Optional[Tuple[str, str, int]]:
    """Parse one brace-balanced LaTeX fraction beginning at ``start``."""
    match = _LATEX_FRAC_COMMAND.match(text, start)
    if match is None:
        return None
    i = match.end()
    while i < len(text) and text[i].isspace():
        i += 1
    numerator = _balanced_group(text, i, "{", "}")
    if numerator is None:
        return None
    numerator_text, i = numerator
    while i < len(text) and text[i].isspace():
        i += 1
    denominator = _balanced_group(text, i, "{", "}")
    if denominator is None:
        return None
    denominator_text, i = denominator
    return numerator_text, denominator_text, i


def _extract_boxed_candidates(text: str) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    i = 0
    while True:
        match = re.search(r"\\boxed\s*\{", text[i:])
        if match is None:
            break
        start = i + match.start()
        open_brace = i + match.end() - 1
        group = _balanced_group(text, open_brace, "{", "}")
        if group is None:
            i = open_brace + 1
            continue
        payload, end = group
        out.append((start, payload.strip()))
        i = end
    return out


def _is_prefix_only_answer(text: str) -> bool:
    compact = " ".join((text or "").strip().split()).lower().rstrip(" .:")
    return compact in {"the final answer is", "final answer", "final answer is", "answer"}


def _is_short_answer_like(text: str) -> bool:
    stripped = (text or "").strip().replace("**", "").replace("*", "").strip()
    if not stripped:
        return False
    compact = " ".join(stripped.split())
    if not re.search(r"\d|\\boxed|\\frac", compact):
        return False
    patterns = (
        r"^(?:the\s+final\s+answer\s+is\s+)?\$?\\boxed\{.+\}\$?\.?$",
        rf"^(?:the\s+final\s+answer\s+is\s+)?{SIMPLE_NUMBER_PATTERN}%?\.?$",
        rf"^(?:the\s+final\s+answer\s+is\s+)?{SIMPLE_NUMBER_PATTERN}\s*/\s*{SIMPLE_NUMBER_PATTERN}\.?$",
        r"^(?:the\s+final\s+answer\s+is\s+)?[a-z0-9\-\.\+\*/\^\(\)=<>,:%| ]{1,32}\.?$",
    )
    return any(re.fullmatch(pattern, compact, flags=re.IGNORECASE) for pattern in patterns)


def _is_short_final_line_candidate(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped or _is_prefix_only_answer(stripped):
        return False
    clean = stripped.replace("**", "").replace("*", "").strip()
    if re.search(r"^\s*(?:[-]|\d+\.)\s+", clean):
        return False
    return _is_short_answer_like(stripped)


def analyze_completion_text(completion: str) -> Dict[str, object]:
    text = completion or ""
    stripped = text.strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    final_line = lines[-1] if lines else ""
    lower = stripped.lower()
    has_final_answer_tag = bool(re.search(rf"^\s*{FINAL_ANSWER_PATTERN}", text, flags=re.IGNORECASE | re.MULTILINE))
    parsed_final_answer = extract_presented_answer(text, allow_meta_recursion=False)
    prefix_only = bool(has_final_answer_tag and parsed_final_answer is None and _is_prefix_only_answer(stripped))
    has_boxed = "\\boxed" in text
    has_latex = bool(has_boxed or "$" in text or "\\frac" in text or "\\text{" in text or "\\(" in text or "\\[" in text)
    has_markdown = bool("###" in text or "**" in text or re.search(r"^\s*[-*]\s+", text, flags=re.MULTILINE) or re.search(r"^\s*\d+\.\s+", text, flags=re.MULTILINE))
    has_multiple_choice = bool(re.search(r"(?:^|\n)\s*(?:\(?[A-D]\)?[\.\):])", text) or re.search(r"\b[A-D]\)\s", text))
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
        ("answer line", "write exactly", "do not use", "use at most", "stop immediately", "provide a clear")
    )
    final_answer_on_last_line = bool(final_line) and bool(re.match(rf"^\s*{FINAL_ANSWER_PATTERN}", final_line, flags=re.IGNORECASE))
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
        parsed_final_answer is not None and _is_short_final_line_candidate(final_line) and not has_multiple_choice
    )
    valid_answer_presentation = bool(
        ((has_final_answer_tag or has_boxed) and parsed_final_answer is not None)
        or final_line_answer_candidate
        or compact_answer_candidate
    )
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
    }


def completion_quality_metrics(completion_or_meta, extra_flags: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    meta = dict(completion_or_meta) if isinstance(completion_or_meta, dict) else analyze_completion_text(str(completion_or_meta or ""))
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
    return {"quality": float(max(0.0, 1.0 - total_penalty)), "penalty": total_penalty, "penalties": penalties}


def _normalize_text_answer(text: str) -> str:
    t = (text or "").strip().lower()
    if not t:
        return ""
    t = re.sub(r"^\s*(?:the\s+)?final answer(?:\s+is)?\s*:?\s*", "", t, flags=re.IGNORECASE)
    t = t.replace("**", "").replace("$", "")
    t = t.replace("\\left", "").replace("\\right", "").replace("\\%", "%")
    t = t.replace("−", "-").replace(",", "").strip()
    boxed = _extract_boxed_candidates(t)
    if boxed:
        t = boxed[-1][1]
    # Canonicalise simple fractions to the documented ``a/b`` form.  Nested
    # fractions remain in LaTeX and are handled safely by the recursive numeric
    # parser below; no ``eval`` or broad symbolic matching is used.
    frac_re = re.compile(
        r"\\(?:dfrac|tfrac|frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}")
    # One pass is intentional.  For ``\frac{\frac{1}{2}}{3}`` it rewrites
    # only the inner fraction, leaving ``\frac{1/2}{3}`` for the balanced,
    # recursive parser instead of creating the ambiguous ``1/2/3``.
    t = frac_re.sub(r"\1/\2", t)
    t = re.sub(r"\s+", "", t)
    t = t.rstrip(".")
    for opening, closing in ((r"\[", r"\]"), (r"\(", r"\)")):
        if t.startswith(opening) and t.endswith(closing):
            t = t[len(opening):-len(closing)]
    t = t.removesuffix("%")  # paper convention: 50% compares as 50
    fraction = re.fullmatch(r"([+-]?\d+)/([+-]?\d+)", t)
    if fraction and int(fraction[2]) != 0:
        t = str(Fraction(int(fraction[1]), int(fraction[2])))
    return t


def _strip_balanced_outer_pair(text: str) -> str:
    t = text
    while len(t) >= 2 and t[0] in "({" and t[-1] in ")}":
        closer = ")" if t[0] == "(" else "}"
        if t[-1] != closer:
            break
        group = _balanced_group(t, 0, t[0], closer)
        if group is None or group[1] != len(t):
            break
        t = group[0]
    return t


def _top_level_slashes(text: str) -> list[int]:
    stack: list[str] = []
    pairs = {")": "(", "}": "{"}
    out: list[int] = []
    for i, char in enumerate(text):
        if char in "({":
            stack.append(char)
        elif char in ")}":
            if not stack or stack[-1] != pairs[char]:
                return []
            stack.pop()
        elif char == "/" and not stack:
            out.append(i)
    return out if not stack else []


def _decimal_literal_to_fraction(text: str) -> Optional[Fraction]:
    """Convert one bounded decimal/scientific literal without float rounding."""
    if len(text) > _MAX_NUMERIC_LITERAL_CHARS or \
            sum(char.isdigit() for char in text) > _MAX_NUMERIC_DIGITS:
        return None
    exponent_match = re.search(r"[eE]([+-]?\d+)$", text)
    if exponent_match is not None:
        try:
            exponent = int(exponent_match.group(1))
        except ValueError:
            return None
        if abs(exponent) > _MAX_ABS_DECIMAL_EXPONENT:
            return None
    try:
        value = Decimal(text)
    except (InvalidOperation, ValueError):
        return None
    if not value.is_finite():
        return None
    return Fraction(value)


def _parse_numeric_expression(text: str, *, depth: int = 0) -> Optional[Fraction]:
    """Parse the exact, bounded numeric grammar used by the verifier."""
    if depth > 12:
        return None
    t = (text or "").strip()
    t = t.replace("\\left", "").replace("\\right", "")
    t = t.replace("−", "-").replace(",", "").replace("\\%", "%")
    t = re.sub(r"\s+", "", t)
    t = t.removeprefix("\\(").removesuffix("\\)")
    t = t.removeprefix("\\[").removesuffix("\\]")
    t = t.strip("$")
    if t.endswith("%"):
        t = t[:-1]
    t = _strip_balanced_outer_pair(t)
    if not t:
        return None

    if t[0] in "+-" and len(t) > 1:
        inner = _parse_numeric_expression(t[1:], depth=depth + 1)
        if inner is None:
            return None
        return inner if t[0] == "+" else -inner

    latex_fraction = _latex_fraction_at(t)
    if latex_fraction is not None and latex_fraction[2] == len(t):
        numerator = _parse_numeric_expression(
            latex_fraction[0], depth=depth + 1)
        denominator = _parse_numeric_expression(
            latex_fraction[1], depth=depth + 1)
        if numerator is None or denominator is None or denominator == 0:
            return None
        return numerator / denominator

    slashes = _top_level_slashes(t)
    if len(slashes) == 1:
        split = slashes[0]
        numerator = _parse_numeric_expression(t[:split], depth=depth + 1)
        denominator = _parse_numeric_expression(t[split + 1:], depth=depth + 1)
        if numerator is None or denominator is None or denominator == 0:
            return None
        return numerator / denominator
    if len(slashes) > 1:
        return None

    scientific = _LATEX_SCIENTIFIC_PATTERN.fullmatch(t)
    if scientific is not None:
        coefficient = _parse_numeric_expression(
            scientific.group("coefficient"), depth=depth + 1)
        exponent = scientific.group("braced_exp") or scientific.group("plain_exp")
        if coefficient is None or exponent is None:
            return None
        try:
            exponent_value = int(exponent)
        except ValueError:
            return None
        if abs(exponent_value) > _MAX_ABS_DECIMAL_EXPONENT:
            return None
        if exponent_value >= 0:
            return coefficient * (10 ** exponent_value)
        return coefficient / (10 ** (-exponent_value))

    if re.fullmatch(SIMPLE_NUMBER_PATTERN, t) is None:
        return None
    return _decimal_literal_to_fraction(t)


def _parse_simple_numeric(text: str) -> Optional[Fraction]:
    direct = _parse_numeric_expression(text)
    if direct is not None:
        return direct
    return _parse_numeric_expression(_normalize_text_answer(text))


def _numeric_answers_equal(left: Fraction, right: Fraction) -> bool:
    """Apply the documented absolute 1e-6 tolerance without float collapse."""
    return abs(left - right) <= NUMERIC_ABSOLUTE_TOLERANCE


def _numeric_prefix(source: str) -> Optional[str]:
    """Extract one complete numeric answer immediately following a cue.

    The boundary check is load-bearing: malformed values such as ``4.5e`` or
    ``1/2/3`` must not be silently truncated to a valid numeric prefix.
    """
    text = source.lstrip()
    while text.startswith("**"):
        text = text[2:].lstrip()
    math_prefix = ""
    if text.startswith("$"):
        math_prefix = "$"
        text = text[1:].lstrip()
    elif text.startswith(r"\("):
        math_prefix = r"\)"
        text = text[2:].lstrip()

    candidates: list[str] = []
    sign_end = 1 if text.startswith(("+", "-", "−")) else 0
    fraction = _latex_fraction_at(text, sign_end)
    if fraction is not None:
        candidates.append(text[:fraction[2]])
    scientific = _LATEX_SCIENTIFIC_PATTERN.match(text)
    if scientific is not None:
        candidates.append(scientific.group(0))
    bare_fraction = re.match(
        rf"{SIMPLE_NUMBER_PATTERN}\s*/\s*{SIMPLE_NUMBER_PATTERN}%?", text,
        flags=re.IGNORECASE,
    )
    if bare_fraction is not None:
        candidates.append(bare_fraction.group(0))
    decimal = re.match(rf"{SIMPLE_NUMBER_PATTERN}%?", text, flags=re.IGNORECASE)
    if decimal is not None:
        candidates.append(decimal.group(0))
    if not candidates:
        return None

    for candidate in sorted(candidates, key=len, reverse=True):
        remainder = text[len(candidate):]
        stripped_remainder = remainder.lstrip()
        if math_prefix and stripped_remainder.startswith(math_prefix):
            stripped_remainder = stripped_remainder[len(math_prefix):].lstrip()
        elif math_prefix == "$" and stripped_remainder.startswith("$"):
            stripped_remainder = stripped_remainder[1:].lstrip()
        # Only terminal punctuation or a clean end may follow.  This rejects
        # numeric prefixes of symbolic/malformed answers without searching
        # arbitrary prose for a convenient number.
        if stripped_remainder and stripped_remainder[0] not in ".,;:!?)]}":
            continue
        if _parse_numeric_expression(candidate) is not None:
            return candidate
    return None


def extract_presented_answer(completion: str, meta: Optional[Dict[str, object]] = None, *, allow_meta_recursion: bool = True) -> Optional[str]:
    text = completion or ""
    boxed = _extract_boxed_candidates(text)
    if boxed:
        return _normalize_text_answer(boxed[-1][1]) or None
    candidates: list[tuple[int, str]] = []
    saw_answer_cue = False
    for cue_pattern in (FINAL_ANSWER_PATTERN, ANSWER_CUE_PATTERN, r"####"):
        for match in re.finditer(cue_pattern, text, flags=re.IGNORECASE):
            saw_answer_cue = True
            candidate = _numeric_prefix(text[match.end():])
            if candidate is not None:
                candidates.append((match.start(), candidate))
    if candidates:
        candidates.sort(key=lambda x: x[0])
        return _normalize_text_answer(candidates[-1][1]) or None
    if saw_answer_cue:
        return None
    if allow_meta_recursion:
        meta = dict(meta or analyze_completion_text(text))
        if not bool(meta.get("compact_answer_candidate", False)):
            return None
    stripped = text.strip()
    if _is_short_answer_like(stripped):
        return _normalize_text_answer(stripped) or None
    return None


def is_correct_completion(completion: str, ground_truth: str) -> bool:
    pred = extract_presented_answer(completion)
    gt = _normalize_text_answer(ground_truth)
    if not pred or not gt:
        return False
    pred_num = _parse_simple_numeric(pred)
    gt_num = _parse_simple_numeric(gt)
    if pred_num is not None and gt_num is not None:
        return _numeric_answers_equal(pred_num, gt_num)
    return pred == gt


def normalize_generated_completion(completion: str, *, answer_prefix: Optional[str] = None) -> Dict[str, object]:
    raw_completion = completion or ""
    raw_stripped = raw_completion.lstrip()
    prefix = (answer_prefix or "").strip()
    if prefix and not re.match(rf"^\s*{FINAL_ANSWER_PATTERN}", raw_stripped, flags=re.IGNORECASE):
        full_raw = f"{prefix} {raw_stripped}".strip()
    else:
        full_raw = raw_completion.strip()
    lines = [line.strip() for line in full_raw.splitlines() if line.strip()]
    first_line = lines[0] if lines else full_raw
    first_line_canonical = extract_presented_answer(first_line) if first_line else None
    full_raw_canonical = extract_presented_answer(full_raw) if full_raw else None
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
    return value or str(normalized.get("completion_scored", fallback_completion or "") or "").strip()


def normalize_values(values: Sequence[float], mode: str, state: Dict[str, float]) -> Tuple[list[float], float, float]:
    vals = [float(v) if math.isfinite(float(v)) else 0.0 for v in values]
    if not vals:
        return [], 0.0, 1.0
    mean = float(sum(vals) / len(vals))
    std = float(max((sum((v - mean) ** 2 for v in vals) / max(1, len(vals))) ** 0.5, 1e-8))
    mode = (mode or "off").lower()
    if mode == "off":
        return vals, mean, std
    if mode == "batch":
        return [(v - mean) / std for v in vals], mean, std
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
    return [(v - ref_mean) / ref_std for v in vals], ref_mean, ref_std
