import unittest

try:
    import torch  # noqa: F401
except ImportError:  # machine without torch: skip cleanly instead of erroring
    raise unittest.SkipTest("torch not installed") from None


from expdis_torch.pipeline import _log_wandb_artifact, _resolve_actual_training_gate


class PipelineArtifactTests(unittest.TestCase):
    def test_log_wandb_artifact_ignores_missing_optional_path(self):
        _log_wandb_artifact(object(), None, "missing_optional_artifact")
        _log_wandb_artifact(object(), "", "empty_optional_artifact")

    def test_actual_training_gate_skips_when_no_accepted_trajectories(self):
        gate = _resolve_actual_training_gate({"accepted_count": 0})
        self.assertTrue(gate["skip"])
        self.assertEqual(gate["reason"], "no_accepted_trajectories")

    def test_actual_training_gate_allows_training_with_accepted_positives(self):
        gate = _resolve_actual_training_gate(
            {"accepted_count": 2},
            {"positive_count": 2},
        )
        self.assertFalse(gate["skip"])
        self.assertEqual(gate["reason"], "ok")


if __name__ == "__main__":
    unittest.main()
