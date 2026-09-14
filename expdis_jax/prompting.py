"""Qwen's math instruction, appended to the question in the user turn."""

MATH_INSTRUCTION = r"Please reason step by step, and put your final answer within \boxed{}."
PROMPT_PROTOCOL = "qwen3_math_user_suffix_v1"


def math_messages(question: str) -> list[dict[str, str]]:
    text = str(question).rstrip()
    if not text.endswith(MATH_INSTRUCTION):
        text += "\n\n" + MATH_INSTRUCTION
    return [{"role": "user", "content": text}]


def prompt_metadata() -> dict[str, str | None]:
    return {"prompt_protocol": PROMPT_PROTOCOL, "system_prompt": None,
            "user_instruction": MATH_INSTRUCTION}
