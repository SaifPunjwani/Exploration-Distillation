import unittest

try:
    import torch  # noqa: F401
except ImportError:  # machine without torch: skip cleanly instead of erroring
    raise unittest.SkipTest("torch not installed") from None


from expdis_torch.prompting import (
    build_system_prompt,
    render_generation_prompt,
    render_supervised_text,
    resolved_answer_constraint_mode,
    resolved_max_completion_len,
)
from expdis_torch.reward import (
    canonicalize_completion_for_training,
    extract_presented_answer,
    is_correct_completion,
    normalize_generated_completion,
)


class DummyCfg:
    def __init__(
        self,
        dataset_name: str,
        answer_constraint_mode: str = "auto",
        eval_response_style: str = "reasoning_final",
        exploration_response_style: str = "reasoning_final",
        eval_answer_constraint_mode: str = "auto",
        exploration_answer_constraint_mode: str = "none",
    ):
        self.dataset_name = dataset_name
        self.answer_constraint_mode = answer_constraint_mode
        self.eval_response_style = eval_response_style
        self.exploration_response_style = exploration_response_style
        self.eval_answer_constraint_mode = eval_answer_constraint_mode
        self.exploration_answer_constraint_mode = exploration_answer_constraint_mode
        self.actual_target_contract = "canonical"
        self.max_completion_len = 96
        self.exploration_max_completion_len = None
        self.eval_max_completion_len = None


class FakeTokenizer:
    chat_template = (
        "{% for message in messages %}"
        "{{ '<|' + message['role'] + '|>' + message['content'] }}"
        "{% endfor %}"
    )

    def apply_chat_template(self, messages, tokenize=False, continue_final_message=False, add_generation_prompt=False):
        text = "".join(f"<|{message['role']}|>{message['content']}" for message in messages)
        if add_generation_prompt:
            text += "<|assistant|>"
        return text


class AimePromptingTests(unittest.TestCase):
    def test_aime_uses_integer_constraint_in_auto_mode(self):
        cfg = DummyCfg("aime_2024", eval_response_style="answer_only")
        self.assertEqual(resolved_answer_constraint_mode(cfg, phase="eval"), "integer")

    def test_dapo_uses_expression_constraint_in_auto_answer_only_mode(self):
        cfg = DummyCfg("dapo_math_17k", eval_response_style="answer_only")
        self.assertEqual(resolved_answer_constraint_mode(cfg, phase="eval"), "expression")

    def test_reasoning_eval_disables_auto_answer_constraint(self):
        cfg = DummyCfg("aime_2024", eval_response_style="reasoning_final")
        self.assertEqual(resolved_answer_constraint_mode(cfg, phase="eval"), "none")

    def test_aime_system_prompt_requests_boxed_answer(self):
        cfg = DummyCfg("aime_2024")
        prompt = build_system_prompt(cfg)
        self.assertIn("boxed", prompt.lower())

    def test_explore_prompt_does_not_prefill_answer_prefix_in_reasoning_mode(self):
        cfg = DummyCfg("gsm8k", exploration_response_style="reasoning_final")
        tokenizer = FakeTokenizer()
        rendered = render_generation_prompt(tokenizer, "Solve 1+1.", cfg, phase="explore")
        self.assertIn("<|assistant|>", rendered)
        self.assertFalse(rendered.endswith("Final answer: "))
        self.assertNotIn("<|assistant|>Final answer:", rendered)

    def test_normalizer_prefers_later_boxed_answer_over_early_number(self):
        raw = "The final answer is 1. Therefore, \\(k + m = 2\\). The answer is \\boxed{2}."
        normalized = normalize_generated_completion(raw, answer_prefix="")
        self.assertEqual(normalized["completion_canonical"], "2")
        self.assertEqual(extract_presented_answer(raw), "2")
        self.assertEqual(canonicalize_completion_for_training(raw), "2")

    def test_normalizer_prefers_full_trace_answer_over_first_line_fraction(self):
        raw = "The final answer is 1/4.\n\nk + m = 1 + 4 = 5\n\nTherefore, the answer is \\boxed{5}."
        normalized = normalize_generated_completion(raw, answer_prefix="")
        self.assertEqual(normalized["completion_canonical"], "5")
        self.assertEqual(normalized["completion_scored"], "5")

    def test_numeric_answers_with_commas_are_preserved_and_canonicalized(self):
        raw = "Final answer: 70,000"
        self.assertEqual(extract_presented_answer(raw), "70000")
        self.assertEqual(canonicalize_completion_for_training(raw), "70000")
        self.assertTrue(is_correct_completion(raw, "70000"))

    def test_negative_decimal_answers_with_commas_score_correctly(self):
        raw = "Final answer: -1,250.5"
        self.assertEqual(extract_presented_answer(raw), "-1250.5")
        self.assertEqual(canonicalize_completion_for_training(raw), "-1250.5")
        self.assertTrue(is_correct_completion(raw, "-1250.5"))

    def test_supervised_canonical_uses_answer_only_prompting(self):
        cfg = DummyCfg("gsm8k", eval_response_style="reasoning_final")
        tokenizer = FakeTokenizer()
        rendered = render_supervised_text(
            tokenizer,
            "Solve 1+1.",
            "Final answer: 2",
            cfg,
            target_contract="canonical",
        )
        self.assertIn("No explanation.", rendered)
        # canonical must not leak the reasoning_final instruction (expdis_torch/prompting.py:46)
        self.assertNotIn("show your reasoning", rendered)

    def test_supervised_full_trace_uses_reasoning_prompting(self):
        cfg = DummyCfg("gsm8k", eval_response_style="answer_only")
        tokenizer = FakeTokenizer()
        rendered = render_supervised_text(
            tokenizer,
            "Solve 1+1.\nReturn only the final integer.",
            "We compute 1+1=2.\nFinal answer: 2",
            cfg,
            target_contract="full_trace",
        )
        self.assertIn(r"Please reason step by step, and put your final answer within \boxed{}.", rendered)
        self.assertNotIn("<|system|>", rendered)
        self.assertNotIn("final integer on the last line", rendered)
        self.assertNotIn("Return only the final integer.", rendered)
        self.assertNotIn("No explanation.", rendered)

    def test_phase_specific_completion_lengths(self):
        cfg = DummyCfg("gsm8k")
        cfg.max_completion_len = 96
        cfg.exploration_max_completion_len = 64
        cfg.eval_max_completion_len = 24
        self.assertEqual(resolved_max_completion_len(cfg, phase="explore"), 64)
        self.assertEqual(resolved_max_completion_len(cfg, phase="eval"), 24)


if __name__ == "__main__":
    unittest.main()
