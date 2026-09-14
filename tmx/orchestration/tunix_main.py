"""Programmatic Tunix/MaxText entry for TMX DAPO + Dr.GRPO runs.

This is the new-architecture path:

* Tunix owns the RL loop and vLLM-on-TPU rollout engine.
* MaxText owns the distributed JAX model/sharding path.
* TMX owns only the research logic: +1/-1 correctness, novelty/RND,
  DAPO/Dr.GRPO contract guards, HF/W&B metadata.

The standard Tunix CLI is enough for λ=0 correctness baselines. Novelty runs
need a programmatic learner adapter because Tunix's reward manager does not see
arbitrary rollout-output metadata; novelty must be injected into reward kwargs
inside the learner after ``rl_cluster.generate`` returns.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from typing import Any

from .optimization import apply_runtime_env, maybe_init_distributed_jax
from .spec import (
    AlgorithmSpec,
    BackendSpec,
    ExperimentSpec,
    LoggingSpec,
    NetworkSpec,
    ResumeSpec,
    StorageSpec,
    TopologySpec,
)
from .tunix_config import emit_programmatic_argv
from .tunix_learner import (
    HiddenStateNoveltyScorer,
    build_dapo_drgrpo_config,
    build_tmx_dapo_learner_class,
    novelty_config_from_algorithm,
)


def _load_spec_from_env() -> ExperimentSpec:
    raw_b64 = os.environ.get("TMX_EXPERIMENT_SPEC_JSON_B64", "").strip()
    raw_json = os.environ.get("TMX_EXPERIMENT_SPEC_JSON", "").strip()
    if raw_b64:
        data = json.loads(base64.b64decode(raw_b64).decode("utf-8"))
    elif raw_json:
        data = json.loads(raw_json)
    else:
        raise RuntimeError(
            "TMX_EXPERIMENT_SPEC_JSON_B64 or TMX_EXPERIMENT_SPEC_JSON is required"
        )
    return ExperimentSpec(
        name=data["name"],
        network=NetworkSpec(**data.get("network", {})),
        topology=TopologySpec(
            **{
                **data.get("topology", {}),
                "train_workers": tuple(data.get("topology", {}).get("train_workers", ())),
                "rollout_workers": tuple(data.get("topology", {}).get("rollout_workers", ())),
                "train_chips": tuple(data.get("topology", {}).get("train_chips", ())),
                "rollout_chips": tuple(data.get("topology", {}).get("rollout_chips", ())),
            }
        ),
        storage=StorageSpec(**data.get("storage", {})),
        logging=LoggingSpec(**data.get("logging", {})),
        resume=ResumeSpec(**data.get("resume", {})),
        algorithm=AlgorithmSpec(**data.get("algorithm", {})),
        backend=BackendSpec(**data.get("backend", {})),
    )


def _set_contract_env(spec: ExperimentSpec) -> None:
    os.environ.setdefault("TMX_LAMBDA_NOVELTY", str(spec.algorithm.lambda_novelty))
    os.environ.setdefault(
        "TMX_INCORRECT_NOVELTY_SCALE", str(spec.algorithm.incorrect_novelty_scale)
    )
    os.environ.setdefault("TMX_DAPO_DRGRPO_FULL_CONTRACT", "1")
    os.environ.setdefault(
        "TMX_DAPO_DRGRPO_CONTRACT_COMPLETION_LEN",
        str(spec.algorithm.max_completion_len),
    )
    os.environ.setdefault(
        "TMX_DAPO_DRGRPO_CONTRACT_SOFT_OVERLONG_EXPECTED_LEN",
        str(spec.algorithm.soft_overlong_expected_len),
    )
    os.environ.setdefault(
        "TMX_DAPO_DRGRPO_CONTRACT_SOFT_OVERLONG_CACHE_LEN",
        str(spec.algorithm.soft_overlong_cache_len),
    )


def _build_novelty_scorer(spec: ExperimentSpec, rl_cluster: Any) -> HiddenStateNoveltyScorer | None:
    novelty_cfg = novelty_config_from_algorithm(spec.algorithm)
    if novelty_cfg.lambda_novelty <= 0.0:
        return None

    # The first valid production implementation must expose real hidden-state
    # features from the actor. We deliberately do not substitute token-hash
    # novelty here because that would change the paper contract.
    feature_extractor = getattr(rl_cluster.train_actor, "tmx_extract_pooled_layers", None)
    novelty_state = getattr(rl_cluster.train_actor, "tmx_novelty_state", None)
    return HiddenStateNoveltyScorer(
        feature_extractor=feature_extractor,
        novelty_state=novelty_state,
        config=novelty_cfg,
    )


def _build_pipeline_class(spec: ExperimentSpec) -> type:
    """Create a GrpoPipeline subclass that swaps in DAPO + novelty learner."""
    from tunix.cli import grpo_main
    from tunix.cli.utils import data as data_lib

    class TMXGrpoPipeline(grpo_main.GrpoPipeline):  # type: ignore[misc, valid-type]
        def _run(self, mode: str = "grpo") -> None:
            if mode != "grpo":
                raise ValueError("TMX programmatic path supports standard grpo only")

            self._setup_kubernetes()
            tokenizer = self._get_tokenizer()
            raw_dataset, custom_batch_fn = self._load_raw_dataset(tokenizer)
            self.compute_params(raw_dataset)

            dataset, _ = data_lib.post_init_dataset(
                raw_dataset,
                tokenizer,
                batch_size=self.config.get("batch_size", 1),
                num_batches=self.config.get("num_batches"),
                max_prompt_length=self.config["rollout_config"].get("max_prompt_length"),
                fraction=self.config.get("train_fraction", 1.0),
                num_epochs=self.config.get("num_train_epochs", 1),
                prompt_key=self.config.get("prompt_key", "prompts"),
                custom_batch_fn=custom_batch_fn,
            )

            rl_cluster = self.create_rl_cluster(tokenizer)
            algo_config = build_dapo_drgrpo_config(spec)
            learner_cls = build_tmx_dapo_learner_class()
            novelty_scorer = _build_novelty_scorer(spec, rl_cluster)
            learner = learner_cls(
                rl_cluster=rl_cluster,
                reward_fns=self.obtain_reward_fn(),
                algo_config=algo_config,
                novelty_scorer=novelty_scorer,
                novelty_config=novelty_config_from_algorithm(spec.algorithm),
            )
            learner.train(dataset)

    return TMXGrpoPipeline


def _print_payload(payload: dict[str, object]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


def describe() -> int:
    spec = _load_spec_from_env()
    result = spec.validate()
    cmd = [os.sys.executable, "-m", "tmx.orchestration.tunix_main", "run"]
    _print_payload(
        {
            "status": "ok" if result.ok else "invalid",
            "engine": "tunix_maxtext_programmatic_dapo_drgrpo",
            "run_name": spec.name,
            "command": cmd,
            "tunix_argv": emit_programmatic_argv(spec),
            "lambda_novelty": spec.algorithm.lambda_novelty,
            "max_completion_len": spec.algorithm.max_completion_len,
            "resume_checkpoint": (
                spec.resume.init_checkpoint or spec.backend.maxtext_load_parameters_path
            ),
            "errors": list(result.errors),
            "warnings": list(result.warnings),
        }
    )
    return 0 if result.ok else 2


def run() -> int:
    spec = _load_spec_from_env()
    result = spec.validate()
    if not result.ok:
        _print_payload({"status": "invalid", "errors": list(result.errors)})
        return 2

    if (
        os.environ.get("TMX_MAXTEXT_TUNIX_EXPERIMENTAL_ACK", "") != "1"
        and not spec.backend.experimental_adapter_ack
    ):
        _print_payload(
            {
                "status": "blocked",
                "reason": "set TMX_MAXTEXT_TUNIX_EXPERIMENTAL_ACK=1 after TPU smoke gates pass",
            }
        )
        return 4

    apply_runtime_env()
    _set_contract_env(spec)
    dist_info = maybe_init_distributed_jax()
    tunix_argv = emit_programmatic_argv(spec)
    _print_payload(
        {
            "status": "starting",
            "engine": "tunix_maxtext_programmatic_dapo_drgrpo",
            "run_name": spec.name,
            "lambda_novelty": spec.algorithm.lambda_novelty,
            "distributed": dist_info,
            "tunix_argv": tunix_argv,
        }
    )
    if os.environ.get("TMX_DRY_RUN_WORKLOAD", "0") == "1":
        return 0

    pipeline_cls = _build_pipeline_class(spec)
    pipeline = pipeline_cls(tunix_argv)
    pipeline.run_grpo_trainer()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Programmatic Tunix DAPO/Dr.GRPO entry for TMX")
    parser.add_argument("action", nargs="?", default="describe", choices=("describe", "run"))
    args = parser.parse_args()
    return describe() if args.action == "describe" else run()


if __name__ == "__main__":
    raise SystemExit(main())
