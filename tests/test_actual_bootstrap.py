import unittest

try:
    import torch  # noqa: F401
except ImportError:  # machine without torch: skip cleanly instead of erroring
    raise unittest.SkipTest("torch not installed") from None


from expdis_torch.data import (
    choose_actual_bootstrap_trajectories,
    choose_actual_policy_trajectories,
)


class ActualBootstrapTests(unittest.TestCase):
    def test_uses_accepted_trajectories_when_threshold_is_met(self):
        accepted = [{"id": "a1"}, {"id": "a2"}, {"id": "a3"}, {"id": "a4"}]
        all_rows = accepted + [{"id": "b1"}]

        selected, info = choose_actual_bootstrap_trajectories(accepted, all_rows, min_examples=4)

        self.assertEqual(selected, accepted)
        self.assertEqual(info["policy"], "accepted_only")
        self.assertFalse(info["used_fallback"])
        self.assertEqual(info["selected_count"], 4)

    def test_oversamples_accepted_trajectories_when_sparse(self):
        accepted = [{"id": "a1"}]
        all_rows = accepted + [{"id": "b1"}, {"id": "b2"}]

        selected, info = choose_actual_bootstrap_trajectories(accepted, all_rows, min_examples=4, seed=123)

        self.assertEqual(len(selected), 4)
        self.assertEqual(info["policy"], "accepted_oversampled")
        self.assertFalse(info["used_fallback"])
        self.assertEqual(info["reason"], "accepted_below_minimum")
        self.assertEqual(info["selected_count"], 4)
        self.assertEqual(info["oversampled_count"], 3)
        self.assertTrue(all(row["id"] == "a1" for row in selected))

    def test_falls_back_to_all_trajectories_when_none_are_accepted(self):
        accepted = []
        all_rows = [{"id": "b1"}, {"id": "b2"}]

        selected, info = choose_actual_bootstrap_trajectories(accepted, all_rows, min_examples=4)

        self.assertEqual(selected, all_rows)
        self.assertEqual(info["policy"], "all_trajectories_fallback")
        self.assertTrue(info["used_fallback"])
        self.assertEqual(info["reason"], "no_accepted_trajectories")
        self.assertEqual(info["accepted_count"], 0)

    def test_actual_policy_prefers_accepted_and_caps_incorrect(self):
        accepted = [
            {"id": "a1", "is_correct": True, "format_quality": 1.0, "valid_answer_presentation": True},
            {"id": "a2", "is_correct": True, "format_quality": 1.0, "valid_answer_presentation": True},
        ]
        incorrect = [
            {"id": "b1", "is_correct": False, "format_quality": 1.0, "valid_answer_presentation": True, "r_novel": 3.0},
            {"id": "b2", "is_correct": False, "format_quality": 0.9, "valid_answer_presentation": True, "r_novel": 2.0},
            {"id": "b3", "is_correct": False, "format_quality": 0.1, "valid_answer_presentation": True, "r_novel": 10.0},
        ]

        selected, info = choose_actual_policy_trajectories(
            accepted,
            accepted + incorrect,
            policy="accepted_balanced",
            max_incorrect_ratio=0.5,
            seed=123,
        )

        self.assertEqual(info["positive_source"], "accepted")
        self.assertEqual(info["positive_count"], 2)
        self.assertEqual(info["incorrect_selected_count"], 1)
        self.assertEqual(len(selected), 3)
        self.assertEqual(sum(1 for row in selected if row["id"].startswith("a")), 2)
        self.assertEqual(sum(1 for row in selected if row["id"].startswith("b")), 1)

    def test_actual_policy_falls_back_when_no_positive_examples_exist(self):
        all_rows = [
            {"id": "b1", "is_correct": False, "format_quality": 1.0, "valid_answer_presentation": True},
            {"id": "b2", "is_correct": False, "format_quality": 0.8, "valid_answer_presentation": True},
        ]

        selected, info = choose_actual_policy_trajectories(
            [],
            all_rows,
            policy="accepted_balanced",
            max_incorrect_ratio=1.0,
            seed=123,
        )

        self.assertEqual(info["policy"], "all_trajectories_fallback")
        self.assertTrue(info["used_fallback"])
        self.assertEqual(selected, all_rows)


if __name__ == "__main__":
    unittest.main()
