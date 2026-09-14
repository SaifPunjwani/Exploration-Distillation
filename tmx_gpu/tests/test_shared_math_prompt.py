"""Check the actual train/eval entry points, including the local-data route."""
import json

from tmx_jax.data import render_chat_prompt
from tmx_jax.prompting import MATH_INSTRUCTION, math_messages
from tmx_gpu.eval_gpu import render_prompt
from tmx_gpu.grpo_gpu import load_training_examples


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return json.dumps(messages)


def test_training_and_evaluation_use_the_same_user_turn(tmp_path):
    expected = [{"role": "user", "content": "2+2?\n\n" + MATH_INSTRUCTION}]
    tokenizer = Tokenizer()
    assert json.loads(render_chat_prompt("2+2?", tokenizer)) == expected
    assert json.loads(render_prompt(tokenizer, "2+2?", "auto")) == expected
    assert math_messages(expected[0]["content"]) == expected
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps({"problem": "2+2?", "answer": "4"}) + "\n")
    example = load_training_examples(str(path), tokenizer, 0)[0]
    assert json.loads(example.prompt_text) == expected


def test_plain_text_training_keeps_instruction_after_question(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps({"problem": "2+2?", "answer": "4"}) + "\n")
    example = load_training_examples(str(path), object(), 0)[0]
    assert example.prompt_text == "2+2?\n\n" + MATH_INSTRUCTION
