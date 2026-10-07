"""Focused invariants for cross-stage/round checkpoint handoff."""

from types import SimpleNamespace
import os
import tempfile
import unittest

try:
    import jax
except ImportError:  # machine without jax: skip cleanly instead of erroring
    raise unittest.SkipTest("jax not installed") from None
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from expdis_jax.distill import _sft_init_weights_checkpoint
from expdis_jax.train import (
    _CHECKPOINT_LEGACY,
    _CHECKPOINT_NONE,
    _CHECKPOINT_RESUME,
    _CHECKPOINT_WEIGHTS,
    _checkpoint_request,
    _fresh_state_with_params,
    _maybe_restore_state_from_checkpoint,
    _require_exact_resume_metadata,
)


def _cfg(**overrides):
    values = {
        "init_weights_checkpoint": "",
        "resume_checkpoint": "",
        "init_checkpoint": "",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class CheckpointRequestTest(unittest.TestCase):
    def test_empty_source_preserves_base_initialization(self):
        self.assertEqual(_checkpoint_request(_cfg()), (_CHECKPOINT_NONE, ""))

    def test_explicit_modes_are_distinct(self):
        self.assertEqual(
            _checkpoint_request(_cfg(init_weights_checkpoint="/weights")),
            (_CHECKPOINT_WEIGHTS, "/weights"),
        )
        self.assertEqual(
            _checkpoint_request(_cfg(resume_checkpoint="/resume")),
            (_CHECKPOINT_RESUME, "/resume"),
        )
        self.assertEqual(
            _checkpoint_request(_cfg(init_checkpoint="/legacy")),
            (_CHECKPOINT_LEGACY, "/legacy"),
        )

    def test_multiple_sources_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            _checkpoint_request(
                _cfg(
                    init_weights_checkpoint="/weights",
                    resume_checkpoint="/resume",
                )
            )


class FreshStageStateTest(unittest.TestCase):
    def test_weights_handoff_resets_step_and_optimizer(self):
        tx = optax.adamw(learning_rate=1e-3)
        original = train_state.TrainState.create(
            apply_fn=lambda *_args, **_kwargs: None,
            params={"w": jnp.asarray([1.0, -1.0])},
            tx=tx,
        )
        advanced = original.apply_gradients(
            grads={"w": jnp.asarray([0.5, -0.25])}
        )
        new_params = {"w": jnp.asarray([3.0, 4.0])}

        fresh = _fresh_state_with_params(advanced, new_params)
        reference = train_state.TrainState.create(
            apply_fn=advanced.apply_fn,
            params=new_params,
            tx=tx,
        )

        self.assertEqual(int(np.asarray(fresh.step)), 0)
        np.testing.assert_array_equal(
            np.asarray(fresh.params["w"]), np.asarray(new_params["w"])
        )
        for actual_leaf, reference_leaf in zip(
            jax.tree_util.tree_leaves(fresh.opt_state),
            jax.tree_util.tree_leaves(reference.opt_state),
        ):
            np.testing.assert_array_equal(
                np.asarray(actual_leaf), np.asarray(reference_leaf)
            )

    def test_weights_checkpoint_ignores_saved_optimizer_and_step(self):
        import orbax.checkpoint as ocp

        mesh = Mesh(np.asarray(jax.devices()), ("fsdp",))
        sharding = NamedSharding(mesh, P("fsdp"))
        tx = optax.adamw(learning_rate=1e-3)
        source = train_state.TrainState.create(
            apply_fn=lambda *_args, **_kwargs: None,
            params={"w": jax.device_put(jnp.asarray([2.0, 3.0]), sharding)},
            tx=tx,
        ).apply_gradients(grads={"w": jnp.asarray([0.25, -0.5])})
        target = train_state.TrainState.create(
            apply_fn=source.apply_fn,
            params={"w": jax.device_put(jnp.asarray([0.0, 0.0]), sharding)},
            tx=tx,
        ).apply_gradients(grads={"w": jnp.asarray([1.0, 1.0])})
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "weights")
            ocp.PyTreeCheckpointer().save(
                path,
                {
                    "params": source.params,
                    "opt_state": source.opt_state,
                    "step": 99,
                },
            )
            restored, logical_step = _maybe_restore_state_from_checkpoint(
                target,
                {"w": sharding},
                _cfg(init_weights_checkpoint=path),
                mesh,
            )

        reference = train_state.TrainState.create(
            apply_fn=source.apply_fn,
            params=source.params,
            tx=tx,
        )
        self.assertEqual(logical_step, 0)
        self.assertEqual(int(np.asarray(restored.step)), 0)
        np.testing.assert_allclose(
            np.asarray(restored.params["w"]), np.asarray(source.params["w"])
        )
        for actual_leaf, reference_leaf in zip(
            jax.tree_util.tree_leaves(restored.opt_state),
            jax.tree_util.tree_leaves(reference.opt_state),
        ):
            np.testing.assert_array_equal(
                np.asarray(actual_leaf), np.asarray(reference_leaf)
            )


class ExactResumeContractTest(unittest.TestCase):
    class _Checkpointer:
        def __init__(self, metadata):
            self._metadata = metadata

        def metadata(self, _path):
            return self._metadata

    def test_complete_metadata_is_accepted(self):
        _require_exact_resume_metadata(
            self._Checkpointer(
                {"params": object(), "opt_state": object(), "step": object()}
            ),
            "/resume",
        )

    def test_current_orbax_step_metadata_shape_is_accepted(self):
        metadata = SimpleNamespace(
            item_metadata=SimpleNamespace(
                tree={
                    "params": object(),
                    "opt_state": object(),
                    "step": object(),
                }
            )
        )
        _require_exact_resume_metadata(
            self._Checkpointer(metadata), "/resume"
        )

    def test_exact_resume_restores_optimizer_and_step(self):
        import orbax.checkpoint as ocp

        mesh = Mesh(np.asarray(jax.devices()), ("fsdp",))
        sharding = NamedSharding(mesh, P("fsdp"))
        tx = optax.adamw(learning_rate=1e-3)
        source = train_state.TrainState.create(
            apply_fn=lambda *_args, **_kwargs: None,
            params={"w": jax.device_put(jnp.asarray([2.0, 3.0]), sharding)},
            tx=tx,
        ).apply_gradients(grads={"w": jnp.asarray([0.25, -0.5])})
        target = train_state.TrainState.create(
            apply_fn=source.apply_fn,
            params={"w": jax.device_put(jnp.asarray([0.0, 0.0]), sharding)},
            tx=tx,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "resume")
            ocp.PyTreeCheckpointer().save(
                path,
                {
                    "params": source.params,
                    "opt_state": source.opt_state,
                    "step": 7,
                },
            )
            restored, logical_step = _maybe_restore_state_from_checkpoint(
                target,
                {"w": sharding},
                _cfg(resume_checkpoint=path),
                mesh,
            )

        self.assertEqual(logical_step, 7)
        self.assertEqual(int(np.asarray(restored.step)), 7)
        np.testing.assert_allclose(
            np.asarray(restored.params["w"]), np.asarray(source.params["w"])
        )
        for actual_leaf, source_leaf in zip(
            jax.tree_util.tree_leaves(restored.opt_state),
            jax.tree_util.tree_leaves(source.opt_state),
        ):
            np.testing.assert_array_equal(
                np.asarray(actual_leaf), np.asarray(source_leaf)
            )

    def test_missing_optimizer_state_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "missing: opt_state"):
            _require_exact_resume_metadata(
                self._Checkpointer({"params": object(), "step": object()}),
                "/resume",
            )


class SftSourceTest(unittest.TestCase):
    def test_sft_explicit_weights_source(self):
        self.assertEqual(
            _sft_init_weights_checkpoint(_cfg(), "/central"), "/central"
        )

    def test_sft_rejects_resume_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "does not implement exact"):
            _sft_init_weights_checkpoint(
                _cfg(resume_checkpoint="/mid-sft"), None
            )

    def test_sft_rejects_conflicting_weight_sources(self):
        with self.assertRaisesRegex(ValueError, "conflicting"):
            _sft_init_weights_checkpoint(
                _cfg(init_weights_checkpoint="/a"), "/b"
            )


if __name__ == "__main__":
    unittest.main()
