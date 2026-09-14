from __future__ import annotations

import os
import re

from expdis_jax.prompting import MATH_INSTRUCTION, math_messages


DEFAULT_GENERATION_ANSWER_PREFIX = "Final answer:"


def _normalize_response_style(style: str) -> str:
    style = str(style or "").strip().lower()
    if style in {"answer_only", "reasoning_final"}:
        return style
    return "reasoning_final"


def _resolved_response_style(cfg, phase: str = "eval") -> str:
    phase_key = "exploration_response_style" if phase == "explore" else "eval_response_style"
    return _normalize_response_style(getattr(cfg, phase_key, ""))


def _resolved_supervised_response_style(cfg, target_contract: str | None = None) -> str:
    contract = str(target_contract or getattr(cfg, "actual_target_contract", "canonical") or "canonical").strip().lower()
    if contract == "full_trace":
        return "reasoning_final"
    return "answer_only"


def resolved_max_completion_len(cfg, phase: str = "eval") -> int:
    phase_key = "exploration_max_completion_len" if phase == "explore" else "eval_max_completion_len"
    phase_value = getattr(cfg, phase_key, None)
    if phase_value is not None:
        try:
            parsed = int(phase_value)
            if parsed > 0:
                return parsed
        except Exception:
            pass
    return max(1, int(getattr(cfg, "max_completion_len", 32) or 32))


def _dataset_answer_instruction_for_style(dataset_name: str, style: str) -> str:
    dataset_name = str(dataset_name or "").lower()
    if style == "reasoning_final":
        return MATH_INSTRUCTION
    if dataset_name in {"gsm8k"}:
        return "Give only the final integer."
    return r"Give only the final answer within \boxed{}."


def _dataset_answer_instruction(cfg, phase: str = "eval") -> str:
    return _dataset_answer_instruction_for_style(
        getattr(cfg, "dataset_name", ""),
        _resolved_response_style(cfg, phase=phase),
    )


def _prepare_user_prompt_for_style(prompt: str, style: str) -> str:
    text = (prompt or "").strip()
    if _normalize_response_style(style) != "reasoning_final":
        return text
    lines = [line.rstrip() for line in text.splitlines()]
    filtered = [
        line for line in lines
        if line.strip().lower() not in {
            "return only the final integer.",
            "return only the final answer.",
        }
    ]
    return "\n".join(filtered).strip()


def _prepare_user_prompt(prompt: str, cfg, phase: str = "eval") -> str:
    return _prepare_user_prompt_for_style(prompt, _resolved_response_style(cfg, phase=phase))


def _resolved_thinking_mode(cfg, phase: str = "eval") -> str:
    phase_key_map = {"explore": "exploration_thinking_mode", "eval": "eval_thinking_mode", "actual": "actual_thinking_mode"}
    phase_key = phase_key_map.get(phase, "eval_thinking_mode")
    mode = str(getattr(cfg, phase_key, "auto") or "auto").strip().lower()
    if mode in {"think", "no_think"}:
        return mode
    return "auto"


def _apply_thinking_mode_to_user_prompt(text: str, cfg, phase: str = "eval") -> str:
    mode = _resolved_thinking_mode(cfg, phase=phase)
    if mode == "auto":
        return text
    switch = "/think" if mode == "think" else "/no_think"
    stripped = (text or "").lstrip()
    if stripped.startswith("/think") or stripped.startswith("/no_think"):
        return text
    return f"{switch}\n{text}".strip()


def _build_system_prompt_for_style(dataset_name: str, style: str) -> str:
    instruction = _dataset_answer_instruction_for_style(dataset_name, style)
    if _normalize_response_style(style) == "reasoning_final":
        return instruction
    return f"{instruction} No explanation."


def build_system_prompt(cfg, phase: str = "eval") -> str:
    return _build_system_prompt_for_style(
        getattr(cfg, "dataset_name", ""),
        _resolved_response_style(cfg, phase=phase),
    )


def generation_answer_prefix(cfg, phase: str = "eval") -> str:
    return ""


def generation_answer_prompt_prefix(cfg, phase: str = "eval") -> str:
    prefix = generation_answer_prefix(cfg, phase=phase).strip()
    if not prefix:
        return ""
    return f"{prefix} "


_BOXED_REASONING_FEW_SHOTS: list[tuple[str, str]] = [
    (
        "What is the sum of 47 and 86?",
        "47 + 86 = 133.\n\n\\boxed{133}",
    ),
    (
        "If a rectangle has length 12 and width 5, what is its area?",
        "Area = length × width = 12 × 5 = 60.\n\n\\boxed{60}",
    ),
    (
        "Find the value of x if 3x + 7 = 22.",
        "3x + 7 = 22\n3x = 15\nx = 5.\n\n\\boxed{5}",
    ),
]

_GSM8K_REASONING_FEW_SHOTS: list[tuple[str, str]] = [
    (
        "What is the sum of 47 and 86?",
        "47 + 86 = 133.\n133",
    ),
    (
        "If a rectangle has length 12 and width 5, what is its area?",
        "Area = length × width = 12 × 5 = 60.\n60",
    ),
    (
        "Find the value of x if 3x + 7 = 22.",
        "3x + 7 = 22\n3x = 15\nx = 5.\n5",
    ),
]

_BOXED_ANSWER_ONLY_FEW_SHOTS: list[tuple[str, str]] = [
    (
        "What is the sum of 47 and 86?",
        "\\boxed{133}",
    ),
    (
        "If a rectangle has length 12 and width 5, what is its area?",
        "\\boxed{60}",
    ),
    (
        "Find the value of x if 3x + 7 = 22.",
        "\\boxed{5}",
    ),
]

_GSM8K_ANSWER_ONLY_FEW_SHOTS: list[tuple[str, str]] = [
    (
        "What is the sum of 47 and 86?",
        "133",
    ),
    (
        "If a rectangle has length 12 and width 5, what is its area?",
        "60",
    ),
    (
        "Find the value of x if 3x + 7 = 22.",
        "5",
    ),
]


def _few_shot_math_examples(cfg, phase: str = "eval") -> list[tuple[str, str]]:
    dataset_name = str(getattr(cfg, "dataset_name", "") or "").lower()
    style = _resolved_response_style(cfg, phase=phase)
    if dataset_name == "gsm8k":
        if style == "reasoning_final":
            return _GSM8K_REASONING_FEW_SHOTS
        return _GSM8K_ANSWER_ONLY_FEW_SHOTS
    if style == "reasoning_final":
        return _BOXED_REASONING_FEW_SHOTS
    return _BOXED_ANSWER_ONLY_FEW_SHOTS


def build_generation_messages(prompt: str, cfg, phase: str = "eval") -> list[dict[str, str]]:
    prompt = _prepare_user_prompt(prompt, cfg, phase=phase)
    prompt = _apply_thinking_mode_to_user_prompt(prompt, cfg, phase=phase)
    reasoning = _resolved_response_style(cfg, phase=phase) == "reasoning_final"
    messages = [] if reasoning else [
        {"role": "system", "content": build_system_prompt(cfg, phase=phase)},
    ]
    # Few-shot examples help base (non-instruct) models learn the answer format.
    few_shot = getattr(cfg, "few_shot_examples", None)
    if few_shot is None:
        import os
        few_shot = int(os.environ.get("EXPDIS_FEW_SHOT_EXAMPLES", "0") or "0")
    if few_shot and few_shot > 0:
        for q, a in _few_shot_math_examples(cfg, phase=phase)[:int(few_shot)]:
            messages.extend(math_messages(q) if reasoning else [{"role": "user", "content": q}])
            messages.append({"role": "assistant", "content": a})
    messages.extend(math_messages(prompt) if reasoning else [{"role": "user", "content": prompt}])
    prefix = generation_answer_prompt_prefix(cfg, phase=phase)
    if prefix:
        messages.append({"role": "assistant", "content": prefix})
    return messages


def resolved_answer_constraint_mode(cfg, phase: str = "eval") -> str:
    mode_key = "exploration_answer_constraint_mode" if phase == "explore" else "eval_answer_constraint_mode"
    mode = str(getattr(cfg, mode_key, "") or "").lower()
    if not mode:
        mode = str(getattr(cfg, "answer_constraint_mode", "auto") or "auto").lower()
    if mode != "auto":
        return mode
    if _resolved_response_style(cfg, phase=phase) == "reasoning_final":
        return "none"
    dataset_name = str(getattr(cfg, "dataset_name", "") or "").lower()
    if dataset_name in {"gsm8k", "aime_2024", "aime_2025"}:
        return "integer"
    if dataset_name in {"dapo_math_17k", "deepscaler_preview"}:
        return "expression"
    return "none"


def get_answer_token_constraints(tokenizer, cfg, phase: str = "eval"):
    mode = resolved_answer_constraint_mode(cfg, phase=phase)
    if mode == "none":
        return None

    cached = getattr(tokenizer, "_expdis_answer_constraints", {})
    cache_key = f"{phase}:{mode}"
    if cache_key in cached:
        return cached[cache_key]

    if mode == "integer":
        allow_re = re.compile(r"^[+\-]?[0-9][0-9,.\-]*$")
        continue_re = re.compile(r"^[0-9,.\-]+$")
    elif mode == "expression":
        def _expression_piece_allowed(piece: str) -> bool:
            compact = (piece or "").strip()
            if not compact:
                return False
            if re.fullmatch(r"[+\-*/^().,=%:|/]+", compact):
                return True
            if re.fullmatch(r"[+\-]?[0-9][0-9,.\-]*", compact):
                return True
            if re.fullmatch(r"[a-zA-Z]{1,3}", compact):
                return True
            if re.fullmatch(r"[a-zA-Z][0-9]", compact):
                return True
            if compact.lower() in {"pi", "sqrt", "frac", "boxed", "infty"}:
                return True
            if re.fullmatch(r"\\(?:pi|sqrt|frac|boxed|infty)", compact, flags=re.IGNORECASE):
                return True
            return False
    else:
        return None

    special_ids = set()
    for token_id in (
        getattr(tokenizer, "pad_token_id", None),
        getattr(tokenizer, "bos_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
    ):
        if token_id is None:
            continue
        if isinstance(token_id, (list, tuple)):
            special_ids.update(int(v) for v in token_id)
        else:
            special_ids.add(int(token_id))

    initial_ids = []
    continuation_ids = []
    vocab_size = len(tokenizer)
    for token_id in range(vocab_size):
        if token_id in special_ids:
            continue
        piece = tokenizer.decode([token_id], skip_special_tokens=False)
        if any(ord(ch) < 32 for ch in piece):
            continue
        compact = piece.strip()
        if not compact:
            continue
        if mode == "expression":
            if _expression_piece_allowed(compact):
                continuation_ids.append(int(token_id))
                initial_ids.append(int(token_id))
            continue
        if continue_re.fullmatch(compact):
            continuation_ids.append(int(token_id))
        if allow_re.fullmatch(compact):
            initial_ids.append(int(token_id))

    constraints = {
        "mode": mode,
        "initial_token_ids": initial_ids,
        "allowed_token_ids": continuation_ids,
    }
    cached = dict(cached)
    cached[cache_key] = constraints
    tokenizer._expdis_answer_constraints = cached
    return constraints


def uses_chat_template(tokenizer) -> bool:
    template = getattr(tokenizer, "chat_template", None)
    return bool(template)


def _chat_template_kwargs(cfg=None, phase: str = "eval") -> dict:
    """Extra kwargs for apply_chat_template (e.g. thinking mode control)."""
    kwargs: dict = {}
    if cfg is not None:
        mode = _resolved_thinking_mode(cfg, phase=phase)
        if mode == "think":
            kwargs["enable_thinking"] = True
        elif mode == "no_think":
            kwargs["enable_thinking"] = False
    if "enable_thinking" not in kwargs:
        thinking_env = os.environ.get("EXPDIS_ENABLE_THINKING", "").strip().lower()
        if thinking_env in ("0", "false", "no", "off"):
            kwargs["enable_thinking"] = False
        elif thinking_env in ("1", "true", "yes", "on"):
            kwargs["enable_thinking"] = True
    return kwargs


def render_generation_prompt(tokenizer, prompt: str, cfg, phase: str = "eval") -> str:
    messages = build_generation_messages(prompt, cfg, phase=phase)
    prefix = generation_answer_prompt_prefix(cfg, phase=phase)
    extra_kwargs = _chat_template_kwargs(cfg, phase=phase)
    if uses_chat_template(tokenizer):
        if prefix:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                continue_final_message=True,
                **extra_kwargs,
            )
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **extra_kwargs,
        )
    return "\n".join(message["content"] for message in messages).rstrip()


def render_supervised_prompt(tokenizer, prompt: str, cfg, target_contract: str | None = None) -> str:
    style = _resolved_supervised_response_style(cfg, target_contract=target_contract)
    prompt = _prepare_user_prompt_for_style(prompt, style)
    messages = math_messages(prompt) if style == "reasoning_final" else [
        {"role": "system", "content": _build_system_prompt_for_style(
            getattr(cfg, "dataset_name", ""), style)},
        {"role": "user", "content": prompt},
    ]
    extra_kwargs = _chat_template_kwargs(cfg, phase="actual")
    if uses_chat_template(tokenizer):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **extra_kwargs,
        )
    return "\n".join(message["content"] for message in messages)


def render_supervised_text(tokenizer, prompt: str, completion: str, cfg, target_contract: str | None = None) -> str:
    style = _resolved_supervised_response_style(cfg, target_contract=target_contract)
    prompt = _prepare_user_prompt_for_style(prompt, style)
    completion = (completion or "").strip()
    messages = math_messages(prompt) if style == "reasoning_final" else [
        {"role": "system", "content": _build_system_prompt_for_style(
            getattr(cfg, "dataset_name", ""), style)},
        {"role": "user", "content": prompt},
    ]
    if uses_chat_template(tokenizer):
        extra_kwargs = _chat_template_kwargs(cfg, phase="actual")
        return tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": completion}],
            tokenize=False,
            add_generation_prompt=False,
            **extra_kwargs,
        )
    return "\n".join([message["content"] for message in messages] + [completion]).strip()
