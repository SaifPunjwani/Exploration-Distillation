import sys
import unittest
from unittest import mock

from expdis_torch.config import build_configs_from_args, parse_args


class BenchmarkConfigTests(unittest.TestCase):
    def test_benchmark_dataset_args_parse(self):
        argv = [
            "run_expdis_torch.py",
            "--dataset-name", "dapo_math_17k",
            "--benchmark-dataset-name", "aime_2024",
            "--benchmark-dataset-split-eval", "train",
            "--benchmark-dataset-split-policy", "first_available",
            "--max-benchmark-examples", "30",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_args()
        cfg = build_configs_from_args(args)[0]
        self.assertEqual(cfg.dataset_name, "dapo_math_17k")
        self.assertEqual(cfg.benchmark_dataset_name, "aime_2024")
        self.assertEqual(cfg.benchmark_dataset_split_eval, "train")
        self.assertEqual(cfg.benchmark_dataset_split_policy, "first_available")
        self.assertEqual(cfg.max_benchmark_examples, 30)

    def test_actual_target_compare_expands(self):
        argv = [
            "run_expdis_torch.py",
            "--actual-target-contract", "compare",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_args()
        cfgs = build_configs_from_args(args)
        self.assertEqual({cfg.actual_target_contract for cfg in cfgs}, {"canonical", "full_trace"})

    def test_eval_and_trajectory_batch_size_args_apply(self):
        argv = [
            "run_expdis_torch.py",
            "--eval-batch-size", "2",
            "--trajectory-batch-size", "1",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_args()
        cfg = build_configs_from_args(args)[0]
        self.assertEqual(cfg.eval_batch_size, 2)
        self.assertEqual(cfg.trajectory_batch_size, 1)


if __name__ == "__main__":
    unittest.main()
