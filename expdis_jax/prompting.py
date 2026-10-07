"""System prompts for training and evaluation; the problem is the user message."""

TRAIN_SYSTEM_PROMPT = (
    "You are a helpful mathematician. Solve the problem step by step. "
    "Put your final numerical answer inside \\boxed{} at the end."
)
EVAL_SYSTEM_PROMPT = r"Please reason step by step, and put your final answer within \boxed{}."
PROMPT_PROTOCOL = "system_prompt_v1"


def math_messages(question: str, system_prompt: str = TRAIN_SYSTEM_PROMPT) -> list[dict[str, str]]:
    return [{"role": "system", "content": system_prompt},
            {"role": "user", "content": str(question)}]


def prompt_metadata(system_prompt: str = EVAL_SYSTEM_PROMPT) -> dict[str, str | None]:
    return {"prompt_protocol": PROMPT_PROTOCOL, "system_prompt": system_prompt,
            "user_instruction": None}
