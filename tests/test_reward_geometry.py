import unittest

try:
    import torch  # noqa: F401
except ImportError:  # machine without torch: skip cleanly instead of erroring
    raise unittest.SkipTest("torch not installed") from None


from expdis_torch.reward import GRPOReward


class _DummyNovelty:
    def __init__(self, value: float):
        self.value = float(value)
        self.last_feature_stats = {}

    def score_batch_with_details(self, completions, train: bool, phase: str = "default"):
        return [self.value for _ in completions], [{} for _ in completions]


class RewardGeometryTests(unittest.TestCase):
    def test_linear_reward_with_zero_incorrect_reward_never_makes_wrong_answers_positive(self):
        reward = GRPOReward(
            novelty_scorer=_DummyNovelty(7.5),
            lambda_novelty=0.1,
            reward_correct=1.0,
            reward_incorrect=0.0,
            reward_shape="linear",
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

        # Novelty must never make an incorrect completion profitable — even with
        # quality gating disabled. (Novelty on wrong answers is the measured
        # reward-hacking failure mode this method exists to prevent.)
        self.assertLessEqual(values[0], 0.0)
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


if __name__ == "__main__":
    unittest.main()
