"""Pure tests that bind the executable JAX budget to the paper's Table 3."""

import unittest

from expdis_jax.lineage import (
    fixed_total_geometry,
    novelty_weight_for_round,
    parse_round_schedule,
    split_budget,
)


class FixedTotalBudgetTests(unittest.TestCase):
    def test_split_budget_is_exact_and_deterministic(self):
        self.assertEqual(split_budget(200, 3), [67, 67, 66])
        self.assertEqual(split_budget(50, 3), [17, 17, 16])
        self.assertEqual(split_budget(40, 3), [14, 13, 13])
        self.assertEqual(split_budget(200, 7), [29, 29, 29, 29, 28, 28, 28])

    def test_every_standard_table_row_has_same_rl_total(self):
        for scouts, rounds in (
            (1, 1),
            (2, 1),
            (3, 1),
            (5, 1),
            (7, 1),
            (1, 4),
            (1, 5),
            (3, 4),
            (3, 5),
            (5, 4),
        ):
            with self.subTest(scouts=scouts, rounds=rounds):
                geometry = fixed_total_geometry(
                    scouts_per_round=scouts,
                    rounds=rounds,
                )
                self.assertEqual(geometry.total_scout_updates, 200)
                self.assertEqual(geometry.total_central_updates, 100)
                self.assertEqual(geometry.total_rl_updates, 300)
                self.assertEqual(geometry.update_consumed_rollout_rows, 19_200)

    def test_exact_table_allocations(self):
        self.assertEqual(
            fixed_total_geometry(scouts_per_round=3, rounds=4).scout_updates_by_round,
            ((17, 17, 16),) * 4,
        )
        self.assertEqual(
            fixed_total_geometry(scouts_per_round=3, rounds=4).central_updates_by_round,
            (25, 25, 25, 25),
        )
        self.assertEqual(
            fixed_total_geometry(scouts_per_round=3, rounds=5).scout_updates_by_round,
            ((14, 13, 13),) * 5,
        )
        self.assertEqual(
            fixed_total_geometry(scouts_per_round=5, rounds=4).scout_updates_by_round,
            ((10, 10, 10, 10, 10),) * 4,
        )

    def test_long_dapo_row_count(self):
        self.assertEqual(1_200 * 4 * 16, 76_800)

    def test_invalid_partitions_fail_closed(self):
        with self.assertRaises(ValueError):
            split_budget(-1, 3)
        with self.assertRaises(ValueError):
            split_budget(10, 0)

    def test_annealed_round_schedule_is_exact(self):
        schedule = "0.75,0.50,0.35,0.25"
        self.assertEqual(parse_round_schedule(schedule, 4), (0.75, 0.5, 0.35, 0.25))
        self.assertEqual(
            novelty_weight_for_round(
                scalar_weight=9.0,
                schedule=schedule,
                rounds=4,
                round_index=3,
            ),
            0.35,
        )
        with self.assertRaises(ValueError):
            parse_round_schedule("0.75,0.5", 4)


if __name__ == "__main__":
    unittest.main()
