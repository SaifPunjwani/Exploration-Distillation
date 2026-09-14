import unittest

from tmx.orchestration.reward_plugins import score_math_completions
from tmx.reward import GRPOReward


class _DummyNovelty:
    def __init__(self, value: float):
        self.value = float(value)
        self.last_feature_stats = {}

    def score_batch_with_details(self, completions, train: bool, phase: str = "default"):
        return [self.value for _ in completions], [{} for _ in completions]


class RewardGeometryTests(unittest.TestCase):
    def test_linear_reward_with_zero_incorrect_reward_and_incorrect_novelty_can_make_wrong_answers_positive(self):
        reward = GRPOReward(
            novelty_scorer=_DummyNovelty(7.5),
            lambda_novelty=0.1,
            reward_correct=1.0,
            reward_incorrect=0.0,
            reward_shape="linear",
            incorrect_novelty_scale=1.0,
            piecewise_bounds=[-1.0, -0.5, 0.0, 0.5, 1.0],
            novelty_zscore="off",
            correctness_zscore="off",
            format_penalty_weight=0.0,
            gate_novelty_by_quality=False,
        )

        values, details = reward.compute_batch(
            ["Final answer: 8"],
            ["4"],
            return_details=True,
        )

        self.assertGreater(values[0], 0.0)
        self.assertFalse(details[0]["is_correct"])

    def test_linear_reward_with_negative_incorrect_reward_keeps_wrong_answers_negative(self):
        reward = GRPOReward(
            novelty_scorer=_DummyNovelty(7.5),
            lambda_novelty=0.03,
            reward_correct=1.0,
            reward_incorrect=-1.0,
            reward_shape="linear",
            piecewise_bounds=[-1.0, -0.5, 0.0, 0.5, 1.0],
            novelty_zscore="off",
            correctness_zscore="off",
            format_penalty_weight=0.0,
            gate_novelty_by_quality=False,
        )

        values, details = reward.compute_batch(
            ["Final answer: 8", "Final answer: 4"],
            ["4", "4"],
            return_details=True,
        )

        self.assertLess(values[0], 0.0)
        self.assertGreater(values[1], 0.0)
        self.assertFalse(details[0]["is_correct"])
        self.assertTrue(details[1]["is_correct"])

    def test_orchestration_reward_plugin_keeps_wrong_novelty_below_correctness(self):
        rows = score_math_completions(
            ["reasoning... \\boxed{8}", "reasoning... \\boxed{4}"],
            ["4", "4"],
            novelty_scores=[1.0, 0.0],
            lambda_novelty=0.75,
            incorrect_novelty_scale=1.0,
            reward_correct=1.0,
            reward_incorrect=-1.0,
            apply_format_penalty=False,
        )

        self.assertFalse(rows[0]["is_correct"])
        self.assertTrue(rows[1]["is_correct"])
        self.assertLess(rows[0]["blended_reward"], rows[1]["blended_reward"])
        self.assertEqual(rows[0]["format_penalty"], 0.0)


if __name__ == "__main__":
    unittest.main()
