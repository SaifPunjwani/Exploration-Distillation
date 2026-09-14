from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence


_BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_NUMERIC_RE = re.compile(r"-?\d+(?:\.\d+)?(?:\s*/\s*-?\d+(?:\.\d+)?)?%?")


@dataclass(frozen=True)
class RewardRow:
    completion: str
    ground_truth: str
    presented_answer: str
    is_correct: bool
    is_clipped: bool
    format_quality: float
    format_penalty: float
    correctness_reward: float
    novelty_reward: float
    novelty_used: float
    blended_reward: float

    def to_dict(self) -> dict[str, object]:
        return {
            "completion": self.completion,
            "ground_truth": self.ground_truth,
            "presented_answer": self.presented_answer,
            "is_correct": self.is_correct,
            "is_clipped": self.is_clipped,
            "format_quality": self.format_quality,
            "format_penalty": self.format_penalty,
            "correctness_reward": self.correctness_reward,
            "novelty_reward": self.novelty_reward,
            "novelty_used": self.novelty_used,
            "blended_reward": self.blended_reward,
        }


def _strip_answer_prefix(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^\s*(?:the\s+)?final answer(?:\s+is)?\s*:?\s*", "", text, flags=re.I)
    text = re.sub(r"^\s*answer\s*:?\s*", "", text, flags=re.I)
    return text.strip().strip(".")


def normalize_answer(text: str) -> str:
    value = _strip_answer_prefix(text)
    boxed = list(_BOXED_RE.finditer(value))
    if boxed:
        value = boxed[-1].group(1)
    value = value.replace("$", "").replace(",", "").replace("\\left", "").replace("\\right", "")
    value = value.replace("\\%", "%").replace("−", "-")
    value = re.sub(r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", r"\1/\2", value)
    value = re.sub(r"\s+", "", value.lower())
    value = value.strip("{}.")
    return value


def _to_float(text: str) -> float | None:
    value = normalize_answer(text).rstrip("%")
    try:
        if "/" in value and re.fullmatch(r"-?\d+(?:\.\d+)?/-?\d+(?:\.\d+)?", value):
            num, den = value.split("/", 1)
            den_f = float(den)
            if den_f == 0.0:
                return None
            return float(num) / den_f
        return float(value)
    except Exception:
        return None


def extract_presented_answer(completion: str) -> str:
    text = completion or ""
    boxed = list(_BOXED_RE.finditer(text))
    if boxed:
        return normalize_answer(boxed[-1].group(1))
    candidates: list[tuple[int, str]] = []
    for pattern in (
        r"(?:final answer\s*:|the final answer is|answer\s*:)\s*([^.\n]+)",
        r"####\s*([^\n]+)",
    ):
        for match in re.finditer(pattern, text, flags=re.I):
            candidates.append((match.start(), match.group(1)))
    if candidates:
        candidates.sort(key=lambda item: item[0])
        return normalize_answer(candidates[-1][1])
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and _NUMERIC_RE.fullmatch(_strip_answer_prefix(lines[-1])):
        return normalize_answer(lines[-1])
    return ""


def is_correct(completion: str, ground_truth: str) -> bool:
    pred = extract_presented_answer(completion)
    gold = normalize_answer(ground_truth)
    if not pred or not gold:
        return False
    pred_num = _to_float(pred)
    gold_num = _to_float(gold)
    if pred_num is not None and gold_num is not None:
        return abs(pred_num - gold_num) <= 1e-6
    return pred == gold


def format_quality(completion: str, *, clipped: bool = False) -> tuple[float, float]:
    text = completion or ""
    penalty = 0.0
    if not extract_presented_answer(text):
        penalty += 0.30
    if "\\boxed" not in text and not re.search(r"final answer|answer\s*:", text, flags=re.I):
        penalty += 0.10
    if re.search(r"^\s*(?:[-*]|\d+\.)\s+", text, flags=re.M):
        penalty += 0.10
    if clipped:
        penalty += 0.35
    penalty = min(1.0, penalty)
    return max(0.0, 1.0 - penalty), penalty


def score_math_completions(
    completions: Sequence[str],
    ground_truths: Sequence[str],
    *,
    novelty_scores: Iterable[float] | None = None,
    clipped: Iterable[bool] | None = None,
    reward_correct: float = 1.0,
    reward_incorrect: float = -1.0,
    lambda_novelty: float = 0.0,
    incorrect_novelty_scale: float = 0.0,
    apply_format_penalty: bool = False,
    format_penalty_weight: float = 0.0,
) -> list[dict[str, object]]:
    """Score completions for TMX Explorer/Actual reward adapters.

    Format quality is reported for filtering/diagnostics, but is not subtracted
    unless explicitly requested. Production DAPO runs should normally leave
    `apply_format_penalty=False` and use soft-overlong + correctness/novelty.
    """

    novelty_list = list(novelty_scores if novelty_scores is not None else [0.0] * len(completions))
    clipped_list = list(clipped if clipped is not None else [False] * len(completions))
    rows: list[dict[str, object]] = []
    for idx, completion in enumerate(completions):
        gt = ground_truths[idx] if idx < len(ground_truths) else ""
        novelty = float(novelty_list[idx]) if idx < len(novelty_list) else 0.0
        is_clipped = bool(clipped_list[idx]) if idx < len(clipped_list) else False
        correct = is_correct(completion, gt)
        quality, penalty = format_quality(completion, clipped=is_clipped)
        correctness_reward = float(reward_correct) if correct else float(reward_incorrect)
        novelty_used = novelty if correct else float(incorrect_novelty_scale) * novelty
        blended = correctness_reward + float(lambda_novelty) * novelty_used
        if apply_format_penalty:
            blended -= float(format_penalty_weight) * penalty
        row = RewardRow(
            completion=completion,
            ground_truth=gt,
            presented_answer=extract_presented_answer(completion),
            is_correct=correct,
            is_clipped=is_clipped,
            format_quality=quality,
            format_penalty=penalty,
            correctness_reward=correctness_reward,
            novelty_reward=novelty,
            novelty_used=novelty_used,
            blended_reward=blended,
        )
        rows.append(row.to_dict())
    return rows
