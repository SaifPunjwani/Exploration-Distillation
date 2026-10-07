# Exploration-Distillation (ExpDis)

Code for *Decoupling Exploration from Optimization in RLVR* (Punjwani and
Goldblum). Checkpoints are at
[SaifPunjwani/expdis-checkpoints](https://huggingface.co/SaifPunjwani/expdis-checkpoints).

ExpDis trains exploration and optimization in separate policies, both
initialized from the same base model:

1. **Explorer RL.** One or more explorers are trained with DAPO on a reward of
   correctness plus λ times a novelty bonus. The bonus is Random Network
   Distillation (RND) on the explorer's hidden states and is added only to
   verifier-correct completions.
2. **Rejection sampling.** Explorer trajectories are kept only if the final
   `\boxed{}` answer is correct, the completion terminates within the
   32,768-token budget, and it has no repetition loop. One shortest trajectory
   is kept per problem, and the set is capped at 500 per round.
3. **Student training.** The student is fine-tuned on the filtered
   trajectories (SFT) and then trained with DAPO on the correctness-only
   reward, with no novelty term.

Two extensions use the same stages. With K parallel explorers, the explorers
start from the same model with different seeds, and their trajectories are
pooled before filtering. With R rounds, the training prompts are split into R
disjoint shards, round r uses shard r, the round-r student initializes both
models in round r+1, and λ follows a schedule across rounds (0.75, 0.50, 0.35,
0.25 for four rounds). The update budget is fixed: 200 explorer updates and 100
student updates in total, divided evenly across explorers and rounds.

## Repository layout

| Path | Contents |
|---|---|
| `expdis_jax/` | JAX/Flax implementation for TPU. See [`expdis_jax/README.md`](expdis_jax/README.md). |
| `expdis_jax/scripts/` | TPU setup, vLLM serving and reload, and launch scripts |
| `tests/` | CPU tests, plus optional numerical checks against Hugging Face Transformers |
| `requirements-dev.lock` | Pinned CPU test environment |

## Method to code

| Paper | Code |
|---|---|
| Explorer RL and student RL (DAPO update) | `grpo.py`, `train.py::run_training` |
| Reward and verifier | `rewarding.py`, `train.py::_score_rollouts` |
| RND novelty bonus; kNN and elliptical bonuses (Appendix C.1) | `novelty.py`, `model.py` |
| Rejection sampling (quality filter) | `filtering.py`, `pipeline.py::collect_accepted` |
| Student SFT | `distill.py::run_sft` |
| Parallel explorers and multiple rounds | `pipeline.py::multi_round_main`, `lineage.py`, `parallel_pipeline.py` |
| Evaluation (avg@n, pass@k, InterDistinct-4, distinct answers) | `eval.py` |
| General-capability evaluation (MMLU-Pro, MMLU-Redux, IFEval, GPQA-Diamond, ZebraLogic) | `general_eval.py` |
| Baselines and ablations | `scripts/launch_jax_pipeline.sh` (`BASELINE`, `ACCEPTED_SELECTION_POLICY`) |

Some identifiers in the code predate the paper's terminology. `scout` and
`explorer_*` refer to the explorer. `actual_*`, `central_*`, and the `--main-*`
command-line flags refer to the student. `pipeline_mode="two_model"` is a
single round, and `multi_round` with `--explorers-per-round K` is the
multi-explorer, multi-round configuration. Saved checkpoint paths use the same
names.

## Run the tests on CPU

With Python 3.11:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.lock
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 JAX_PLATFORMS=cpu \
  .venv/bin/python -m pytest tests/test_expdis_jax_pipeline_cpu.py -q
```

This test runs explorer RL, filtering, student SFT, and student RL on a small
local Flax model, with fixtures in place of the dataset and the vLLM server.
It needs no network access, credentials, or accelerator. Run the full suite
with `.venv/bin/python -m pytest tests -q`. Tests that need a PyTorch reference
environment are skipped when it is not installed.

## Training on TPU

Training needs a TPU VM for the trainer, vLLM servers for rollouts, and a way
to send updated weights to the servers after every step. The setup and launch
commands are in [`expdis_jax/README.md`](expdis_jax/README.md#training-on-tpu),
and the commands for the paper's ablations (Table 2, Table 8, Table 9, fixed
versus annealed λ, NaivePool) are in its [Ablations](expdis_jax/README.md#ablations)
section.
For Ministral, use the
[BF16 instruct checkpoint](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-BF16);
the FP8 checkpoint is not supported for training.

## Prompt

The problem is the user message. Training uses the system message

> You are a helpful mathematician. Solve the problem step by step. Put your final numerical answer inside \boxed{} at the end.

and evaluation uses the system message

> Please reason step by step, and put your final answer within \boxed{}.

Evaluation can also use the training system message: pass `--prompt train` to
`expdis_jax.eval`, or set `EXPDIS_EVAL_PROMPT=train` for every evaluation in a
run. Both use the model's chat template, with thinking enabled for Qwen3.

## Notes

- Hugging Face caches default to `.hf/` in the repository unless `HF_HOME`,
  `HF_HUB_CACHE`, or `HF_DATASETS_CACHE` is set.
- Runs stop at startup without W&B credentials. Set `EXPDIS_REQUIRE_WANDB=0` to
  run without W&B. The W&B project is `expdis`.

## License

Apache License 2.0. See [LICENSE](LICENSE).
