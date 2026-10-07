# expdis_jax

JAX/Flax implementation of Exploration-Distillation for TPU (v4, v5e, v6e). It
uses 1-D FSDP sharding and Pallas flash attention, and samples rollouts from
vLLM servers.

## Pipeline

`pipeline.py::main` runs one round:

1. **Explorer RL** (`train.py::run_training`, λ > 0, 200 updates). Each update
   sends the current weights to the vLLM servers, samples 4 prompts × 16
   completions, scores them (`_score_rollouts`), applies dynamic sampling, and
   takes one optimizer step.
2. **Rejection sampling** (`filtering.py`, `pipeline.py::collect_accepted`).
   The candidates are the explorer rollouts used in training updates. A
   trajectory is kept if it is verifier-correct, ends with a stop token, has a
   valid boxed answer that matches the reference, is non-empty and within
   32,768 tokens, and has no 40-character span repeated four times in a row.
   One shortest trajectory is kept per problem. If more than 500 problems
   remain, the first 500 in a fixed prompt-hash order are kept.
3. **Student SFT** (`distill.py::run_sft`). Starts from the same model as the
   explorer. Two epochs, batch size 1, learning rate 5e-6, at most 500
   examples. The loss is token-mean cross-entropy on completion tokens only.
4. **Student RL** (`train.py::run_training`, λ = 0, 100 updates, learning rate
   1e-6). Correctness reward plus the soft overlong penalty, no novelty.
5. **Evaluation** (`eval.py`). AIME24 with 64 samples per problem by default.

`pipeline.py::multi_round_main` runs R rounds with K explorers per round. The
prompt set is split into R disjoint shards. The 200 explorer and 100 student
updates are divided across rounds and explorers by `lineage.py` (for example,
17/17/16 explorer updates and 25 student updates per round for R=4, K=3). The
explorers in a round start from the same model with different seeds, and their
trajectories are pooled before filtering. Round r+1 starts both models from the
round-r student. The RND networks and optimizer state are re-initialized for
every explorer and every round. λ follows `--round-novelty-schedule`.

The native driver trains a round's explorers one after another.
`parallel_pipeline.py` runs them concurrently on separate allocations; see
[PARALLEL.md](PARALLEL.md).

## RL update

Explorer RL, student RL, and the DAPO baselines use the same update
(`grpo.py`):

- 4 prompt groups × 16 completions = 64 rows per update, one optimizer step per
  batch, and fresh rollouts from the current weights for every batch.
- Advantages are the reward minus the group mean, with no division by the
  standard deviation.
- The policy-gradient loss is summed over tokens and divided by a fixed
  64 × 32,768 (the Dr. GRPO normalization).
- The clipped ratio uses ε_low = 0.2 and ε_high = 0.28. The old
  log-probabilities are those of the current weights, so with one step per
  batch the ratio is 1 and the clip does not change the gradient.
- KL coefficient 0. Truncated completions are masked from the loss.
- Dynamic sampling keeps a group when the total rewards in the group are not
  all equal, and samples new prompts otherwise. After 8 attempts, the batch is
  filled with the remaining groups.
- The DAPO baselines use this update unchanged. The GRPO and Dr. GRPO
  baselines (`BASELINE=grpo` and `BASELINE=drgrpo` below) change only the
  items listed there.
- Rewards: +1 if correct, −1 otherwise. The soft overlong penalty grows
  linearly from 0 at 26,214 tokens to −1 at 32,768 tokens. The explorer adds λ
  times the novelty bonus on correct completions.
- AdamW with β = (0.9, 0.95), ε = 1e-8, no weight decay, and gradient clipping
  at norm 1.0. Learning rate 5e-6 for explorer RL and the baselines, 1e-6 for
  student RL.

Prompts are drawn at random from the full training set at each step. The
answer checker (`rewarding.py`) extracts the last boxed answer, removes commas
and dollar signs, and compares numbers as exact rationals with tolerance 1e-6;
other answers are compared as normalized strings.

## Novelty bonus (RND)

`novelty.py` keeps one frozen random target MLP and one trained predictor MLP
per layer, at layers ⌊L/4⌋, ⌊L/2⌋, and ⌊3L/4⌋ (7, 14, 21 for Qwen3-1.7B; 9, 18,
27 for Qwen3-4B; 6, 13, 19 for Ministral-3-3B). Both MLPs are
Linear–ReLU–Linear–ReLU–Linear with width and output 512. The input is the
policy's hidden state mean-pooled over the completion tokens; the completion
is encoded without the prompt. The bonus for each layer is
`sqrt(mean((target - predictor)^2) + 1e-8)`, averaged over the three layers,
with no normalization. After each policy step the predictor takes one Adam
step (learning rate 1e-4) on the features of the rollouts in that batch.

## Alternative novelty bonuses

`--novelty-method` (`NOVELTY_METHOD` in the launchers) selects the bonus:
`rnd` (default), `knn`, or `elliptical`, the two alternatives of Appendix C.1.
Both use one feature per completion, φ: the final hidden state after the last
RMSNorm, mean-pooled over the completion tokens (same completion-only encoding
as RND) and L2-normalized.

- `knn`: the bonus is the mean cosine distance 1 − φᵀψ from φ to its k = 16
  nearest neighbors ψ in a FIFO buffer of the 4,096 most recent correct
  completions (`--knn-k`, `--knn-buffer-size`). With fewer than k entries the
  available ones are used; an empty buffer gives 0.
- `elliptical`: the bonus is sqrt(φᵀ Σ⁻¹ φ) with
  Σ = λ_ridge·I + Σᵢ φᵢφᵢᵀ over past correct completions, λ_ridge = 1.0
  (`--elliptical-ridge`). The bonus is computed with a linear solve.

As with RND, all rollouts of a step are scored against the state from before
the step, and the state is updated after the policy step. Only completions
that are verifier-correct and part of the training batch enter the buffer or
Σ; groups rejected by dynamic sampling do not. The bonus is raw (no
normalization), multiplied by λ and credited only on correct completions, as
in Eq. 1. The state is re-initialized for every explorer and every round, and
exact-resume checkpoints store it under `novelty_state`. Under the full
contract `knn` requires k = 16 and a 4,096-entry buffer.

```bash
NOVELTY_METHOD=knn LAMBDA_NOVELTY=0.5 RUN_NAME=expdis_knn \
bash expdis_jax/scripts/launch_jax_pipeline.sh
NOVELTY_METHOD=elliptical LAMBDA_NOVELTY=0.5 RUN_NAME=expdis_elliptical \
bash expdis_jax/scripts/launch_jax_pipeline.sh
```

## Configuration

`config.py` defines `TrainConfig` and the command-line flags.
`config.py::validate_contract` checks the settings above at startup and stops
if they differ. Note that `TrainConfig` sets `lambda_novelty=0.0`; the launch
scripts set λ = 0.5.

## Training on TPU

Requirements:

- a Linux x86_64 TPU VM for training;
- one or more vLLM servers with the OpenAI `/v1/completions` API and
  `return_token_ids` support, serving the same base model;
- a Hugging Face dataset repository you can write to
  (`EXPDIS_HF_CHECKPOINT_REPO`). It is used to send weights to the vLLM servers
  after every step and to mirror checkpoints;
- W&B credentials, or `EXPDIS_REQUIRE_WANDB=0`.

The paper's runs used one TPU v5litepod-64 slice: 4 hosts for training and 12
hosts for vLLM.

```bash
# On the training TPU VM:
bash expdis_jax/scripts/setup_jax_tpu.sh

# From a machine with gcloud access to the serving TPU slice:
TPU_NAME="<serving-slice>" ZONE="<serving-zone>" WORKERS="0 1 2 3" \
MAX_MODEL_LEN=40960 \
bash expdis_jax/scripts/bootstrap_vllm_slice.sh

# On the same machine, in a separate terminal, run the reload watcher.
# REMOTE_HOST is an SSH alias for the training VM.
REMOTE_HOST="<trainer-ssh-alias>" \
REMOTE_RUNS_ROOT="/home/<trainer-user>/expdis/runs" \
bash expdis_jax/scripts/external_vllm_reload_watcher.sh

# On the training VM:
export EXPDIS_VLLM_SERVER_URLS="http://<vllm_ip>:8000/v1"
export EXPDIS_HF_CHECKPOINT_REPO="<you>/<dataset-repo>"
export EXPDIS_VLLM_RELOAD_TPU_NAME="<serving-slice>"
export EXPDIS_VLLM_RELOAD_ZONE="<serving-zone>"
export EXPDIS_VLLM_RELOAD_WORKERS="0 1 2 3"
export EXPDIS_REQUIRE_WANDB=0   # or run `wandb login`

# ExpDis (single-round): one explorer, one round.
LAMBDA_NOVELTY=0.5 RUN_NAME=expdis_single_round \
bash expdis_jax/scripts/launch_jax_pipeline.sh

# ExpDis: three explorers, four rounds, annealed λ.
NUM_ROUNDS=4 EXPLORERS_PER_ROUND=3 ROUND_NOVELTY_SCHEDULE=0.75,0.50,0.35,0.25 \
RUN_NAME=expdis bash expdis_jax/scripts/launch_jax_pipeline.sh

# Baselines: BASELINE=dapo, dapo_4x, dapo_novelty, drgrpo, or grpo.
BASELINE=dapo bash expdis_jax/scripts/launch_jax_pipeline.sh
```

`BASELINE` runs only the RL stage with a correctness-only reward: `dapo` (300
updates), `dapo_4x` (1,200 updates), `dapo_novelty` (λ = 0.5, 300 updates),
`drgrpo` (the DAPO update with a symmetric 0.2 clip and no dynamic sampling or
overlong filtering, 300 updates), and `grpo` (as `drgrpo`, with
standard-deviation-normalized advantages and a per-sequence mean loss). All
baselines keep the shared settings above, including the soft overlong penalty.

Replace the placeholders in angle brackets. If the watcher cannot reach both
slices, training waits for the reload acknowledgment. If no reload transport is
configured, training stops rather than sampling from stale weights.

Multi-host runs need `--checkpoint-root` (`CHECKPOINT_ROOT` in the launchers)
on storage that every trainer host can read: a shared filesystem mounted at
the same path, or a `gs://` prefix. Only the student's weights carry over to
the next round.

Other launchers:

- `launch_jax_pipeline_multihost_v5lite.sh`: one v5litepod-64 slice, workers
  0–3 train and 4–15 serve.
- `launch_dapo_drgrpo_tpu.sh`: explorer RL only, on separate training and
  serving slices (`TRAIN_*` and `SERVE_*` variables).
- `launch_vanilla_grpo.sh` and `launch_vanilla_grpo_multihost.sh`: an older
  GRPO setup (8 completions per prompt, 16,384-token completions, 100 steps,
  DeepScaleR) that differs from the paper. Use `BASELINE=grpo` for the paper's
  GRPO baseline.

Outputs are written to `$RUNS_ROOT/$RUN_NAME` (default `$HOME/expdis/runs`):
`console.log`, checkpoints (`step_XXXXXX/`), trajectory JSONL files, the
filtered set `trajectory_library*.jsonl` with a `.funnel.json` summary, and
`run_summary.json`. Set `EXPDIS_SAVE_OPT_STATE=1` to save optimizer state for
resuming within a stage.

## Ablations

Each paper ablation is one launcher call with the environment from the
previous section set. Unless a row says otherwise, runs use Qwen3-1.7B, one
round, one explorer, and λ = 0.5. Evaluate the saved students on the other
benchmarks with `eval.py` as described under Evaluation.

Stage ablations (Table 8):

```bash
# DAPO
BASELINE=dapo bash expdis_jax/scripts/launch_jax_pipeline.sh
# Unfiltered SFT + RL
ACCEPTED_SELECTION_POLICY=unfiltered RUN_NAME=abl_unfiltered_sft_rl \
bash expdis_jax/scripts/launch_jax_pipeline.sh
# Filtered SFT, no RL
EXPDIS_PIPELINE_STOP_AFTER_SFT=1 EXPDIS_PIPELINE_EVAL_AFTER_SFT=1 \
RUN_NAME=abl_filtered_sft_no_rl bash expdis_jax/scripts/launch_jax_pipeline.sh
# Filtered SFT + RL (ExpDis)
RUN_NAME=expdis_single_round bash expdis_jax/scripts/launch_jax_pipeline.sh
```

`ACCEPTED_SELECTION_POLICY=unfiltered` replaces the filter: explorer
trajectories enter SFT regardless of correctness, termination, or repetition.
`EXPDIS_PIPELINE_STOP_AFTER_SFT=1` skips student RL;
`EXPDIS_PIPELINE_EVAL_AFTER_SFT=1` then evaluates the SFT student on AIME24
and writes `final_eval_aime24.json`. The run fails if the vLLM servers were
not reloaded with the SFT weights, unless `EXPDIS_ACTUAL_VLLM_SERVER_URLS`
points at servers that already serve them.

Quality filtering with three explorers (Appendix E):

```bash
# NaivePool
EXPLORERS_PER_ROUND=3 ACCEPTED_SELECTION_POLICY=naive_pool RUN_NAME=abl_naivepool_k3 \
bash expdis_jax/scripts/launch_jax_pipeline.sh
# Quality filter
EXPLORERS_PER_ROUND=3 RUN_NAME=expdis_k3 bash expdis_jax/scripts/launch_jax_pipeline.sh
```

`naive_pool` keeps every verifier-correct trajectory from the pooled
explorers, with no termination, boxed-answer, or repetition checks and no
one-per-problem rule. Both ablation policies keep the 500-trajectory cap and
the SFT settings (two epochs, batch size 1), so every arm distills at most 500
trajectories. When more rows qualify, `naive_pool` takes terminated,
valid-answer, unclipped rows first, then higher total reward, then shorter
completions; `unfiltered` takes a fixed pseudo-random subset ordered by a hash
of each row. `trajectory_library.accepted.jsonl.funnel.json` records the
policy and counts.

Fixed versus annealed λ over four rounds (Appendix E). Add
`EXPDIS_MULTIROUND_EVAL_EACH_ROUND=1` to evaluate the student after every
round; each round writes `round_<r>/final_eval_aime24.json`.

```bash
# Fixed λ = 0.5
NUM_ROUNDS=4 LAMBDA_NOVELTY=0.5 EXPDIS_MULTIROUND_EVAL_EACH_ROUND=1 \
RUN_NAME=abl_fixed_lambda bash expdis_jax/scripts/launch_jax_pipeline.sh
# Annealed λ
NUM_ROUNDS=4 ROUND_NOVELTY_SCHEDULE=0.75,0.50,0.35,0.25 EXPDIS_MULTIROUND_EVAL_EACH_ROUND=1 \
RUN_NAME=abl_annealed_lambda bash expdis_jax/scripts/launch_jax_pipeline.sh
```

Training data (Table 9). Combine one dataset setting with `BASELINE=grpo`,
`BASELINE=drgrpo`, `BASELINE=dapo`, or no `BASELINE` for ExpDis:

```bash
# DAPO-Math-17K (default)
DATASET_NAME=dapo_math_17k bash expdis_jax/scripts/launch_jax_pipeline.sh
# DeepScaleR-17K: a seeded uniform random subset of 17,000 problems
DATASET_NAME=deepscaler MAX_TRAIN_EXAMPLES=17000 TRAIN_SUBSET_POLICY=random TRAIN_SUBSET_SEED=0 \
bash expdis_jax/scripts/launch_jax_pipeline.sh
# DeepScaleR full (40,315 problems)
DATASET_NAME=deepscaler MAX_TRAIN_EXAMPLES=40315 bash expdis_jax/scripts/launch_jax_pipeline.sh
```

`MAX_TRAIN_EXAMPLES` defaults to 20,000, which keeps all of DAPO-Math-17K but
only the first 20,000 rows of DeepScaleR. `TRAIN_SUBSET_POLICY=first` (the
default) keeps the leading rows; `random` draws the subset with
`TRAIN_SUBSET_SEED` and keeps it in dataset order.

Breadth and depth (Table 2). `lineage.py` splits the 200 explorer and 100
student updates as listed in the table:

```bash
# Breadth: K = 2, 3, 5, 7 explorers in one round
EXPLORERS_PER_ROUND=5 bash expdis_jax/scripts/launch_jax_pipeline.sh
# Depth: R = 4 rounds, one explorer
NUM_ROUNDS=4 ROUND_NOVELTY_SCHEDULE=0.75,0.50,0.35,0.25 bash expdis_jax/scripts/launch_jax_pipeline.sh
# Both: R = 4, K = 3 (also R = 4, K = 5)
NUM_ROUNDS=4 EXPLORERS_PER_ROUND=3 ROUND_NOVELTY_SCHEDULE=0.75,0.50,0.35,0.25 \
bash expdis_jax/scripts/launch_jax_pipeline.sh
# R = 5 rows (5×1 and 5×3)
NUM_ROUNDS=5 EXPLORERS_PER_ROUND=3 ROUND_NOVELTY_SCHEDULE=<five values> \
bash expdis_jax/scripts/launch_jax_pipeline.sh
```

The paper gives the λ schedule only for four rounds. For R = 5, pass five
values, or leave `ROUND_NOVELTY_SCHEDULE` unset to use `LAMBDA_NOVELTY` in
every round.

## Environment variables

Defaults in parentheses.

| Variable | Effect |
|---|---|
| `EXPDIS_VLLM_SERVER_URLS` | comma-separated vLLM URLs (required) |
| `LAMBDA_NOVELTY` (0.5) | novelty weight λ |
| `NOVELTY_METHOD` (rnd) | novelty bonus: `rnd`, `knn`, or `elliptical` |
| `GRPO_MAX_STEPS` (200) | explorer RL updates |
| `MAIN_RL_MAX_STEPS` (100) | student RL updates |
| `MAIN_RL_LR` (1e-6) | student RL learning rate |
| `NUM_ROUNDS` (1), `EXPLORERS_PER_ROUND` (1), `ROUND_NOVELTY_SCHEDULE` | rounds, explorers per round, and the λ schedule |
| `SEED` (0) | run seed |
| `EXPDIS_PIPELINE_STOP_AFTER_EXPLORER` (0) | stop after explorer RL |
| `EXPDIS_PIPELINE_STOP_AFTER_SFT` (0), `EXPDIS_PIPELINE_EVAL_AFTER_SFT` (0) | stop after student SFT; evaluate the SFT student |
| `ACCEPTED_SELECTION_POLICY` (`coverage_pool_c8`) | trajectory filter: `coverage_pool_c8` (paper), `naive_pool`, or `unfiltered` |
| `DATASET_NAME` (`dapo_math_17k`), `MAX_TRAIN_EXAMPLES` (20000) | training set (`dapo_math_17k` or `deepscaler`) and its row cap |
| `TRAIN_SUBSET_POLICY` (`first`), `TRAIN_SUBSET_SEED` (0) | which rows the cap keeps: leading rows or a seeded random subset |
| `EXPDIS_MULTIROUND_EVAL_EACH_ROUND` (0) | evaluate the student after every round |
| `SAVE_EVERY_STEPS` (50) | checkpoint interval |
| `EXPDIS_TRAIN_VLLM_RELOAD_MODE` | `external` (reload watcher, set by the launchers) or `direct` (SSH into the serving slice, the default in `train.py`) |
| `EXPDIS_REQUIRE_WANDB` (1) | stop at startup without W&B credentials |
| `EXPDIS_HF_CHECKPOINT_REPO`, `EXPDIS_HF_MIRROR_*` | Hugging Face mirroring of checkpoints, trajectories, reload bundles, and metrics |
| `EXPDIS_ALLOW_GCS_ARTIFACTS` (0), `EXPDIS_GCS_CHECKPOINT_BASE` | optional GCS artifact storage |

## Evaluation

`eval.py` uses the training answer checker and reports avg@n, pass@k for k = 1,
2, 4, ..., 64 (the unbiased estimator of Chen et al., 2021), the number of
distinct answers, answer entropy, and InterDistinct-4 over whitespace tokens.
Sampling uses temperature 0.6, top-p 0.95, top-k 20, min-p 0, and at most
32,768 completion tokens. The system message is the evaluation prompt by
default; `--prompt train` (or `EXPDIS_EVAL_PROMPT=train`) uses the training
prompt instead.

```bash
python -m expdis_jax.eval --model-name /path/to/model \
  --server-urls http://localhost:8000/v1 --which AIME_2024 \
  --output-path aime24.json
```

`--which` accepts `AIME_2024`, `AIME_2025`, `AIME_2026`, `MATH500`, `AMC23`,
`Minerva-Math`, and `GSM8K`. Each is loaded from Hugging Face at the revision
pinned in `eval.BENCHMARK_SOURCES`. To use a local copy instead, pass
`--dataset-jsonl /path/to/benchmark.jsonl` with `problem` and `answer` (or
`question` and `ground_truth`) fields. The default
number of samples is 64 for AIME24, AIME25, AIME26, MATH500, and Minerva-Math,
32 for AMC23, and 8 for GSM8K; `--num-rollouts` overrides it.

To average the five main benchmarks for one checkpoint:

```bash
python -m expdis_jax.eval --summarize-results \
  aime24.json aime25.json aime26.json math500.json minerva.json \
  --output-path mean_accuracy.json
```

## Diversity and faithfulness

`diversity.py` computes the Appendix D metrics and reasoning faithfulness
for one model and writes one JSON file.

- InterDistinct-4, distinct answers, and answer entropy: taken from the
  `eval.py` result.
- Token entropy: 8 rollouts for each AIME24 problem with the evaluation prompt
  and sampler (temperature 0.6, top-p 0.95, top-k 20, min-p 0, at most 32,768
  completion tokens), with seed `260902000 + 1000 * problem + rollout`. The
  value is the mean surprisal -log p of the sampled tokens, pooled over the
  rollouts of each problem and averaged over problems, in nats per token.
  `--entropy-temperature`, `--entropy-top-p`, `--entropy-top-k`, and
  `--entropy-max-tokens` change the sampler. This needs a served policy.
- Semantic diversity: the DARLING math classifier
  (`dogtooth/qwen3-4b-emb-finetuned-step-70-hf`, revision `af8b543`) scores
  every pair of generations of a problem (`--semantic-block-size k` instead
  clusters consecutive blocks of k generations). Each
  response keeps its first 2,046 classifier tokens; a pair uses the same
  approach when the class-1 probability is above 0.5. Union-find gives the
  clusters, and the report gives clusters / n (`semantic_clusters_over_n`)
  and DARLING's mean (n - cluster size) / (n - 1) (`darling_diversity`).
- Reasoning faithfulness: the judge prompt of Rahman et al. (2026, Figure 32)
  labels each response 1, 0.5, or 0 for how well its reasoning supports its
  final answer. The judge is an OpenAI-compatible chat endpoint (`o3` by
  default, as in that paper) and reads the first 16 responses of each problem
  (`--faithfulness-rollouts`, 0 for all). `faithfulness` is the fraction
  labeled 1; `label_rates` and `label_rates_correct` give all three fractions
  over all responses and over correct ones.

Serve the classifier with vLLM 0.11 after allowing token-ID inputs on
`/classify`:

```bash
python expdis_jax/scripts/patch_vllm_classify_token_ids.py
vllm serve dogtooth/qwen3-4b-emb-finetuned-step-70-hf \
  --revision af8b543dc0d22a93e9c6b5aabc501b01b1d4895f --task classify --port 8100
```

```bash
# From an eval.py result (semantic, faithfulness, lexical), plus token entropy
# from the served policy:
OPENAI_API_KEY=... python -m expdis_jax.diversity --eval-json aime24.json \
  --model-name /path/to/model --server-urls http://localhost:8000/v1 \
  --classifier-url http://localhost:8100 --judge-url https://api.openai.com/v1 \
  --output-path aime24_diversity.json
```

Without `--eval-json`, the 64-sample evaluation is generated first. Each part
is optional: `--skip-entropy`, `--classifier none`, and an empty `--judge-url`
leave the corresponding field `null`.

## General-capability evaluation

`general_eval.py` measures prior capabilities on MMLU-Pro, MMLU-Redux, IFEval,
and GPQA-Diamond, plus ZebraLogic. Every benchmark uses the model's
chat template with a single user turn, thinking on for Qwen3 and off for
Ministral (`--enable-thinking auto` reads the chat template), temperature 0.6,
top-p 0.95, top-k 20, min-p 0, at most 32,768 completion tokens (serve with
`--max-model-len 40960`), and per-request seeds from `--seed 1234`. With
thinking on, only the text after the last `</think>` is graded, and a
completion that never closes its thinking counts as wrong. Answers that cannot
be extracted count as wrong; there is no random fallback.

| Benchmark | Data (pinned revision) | Prompt | Score | Samples |
|---|---|---|---|---|
| `mmlu_pro` | TIGER-Lab/MMLU-Pro test, 12,032 questions | five category-matched validation CoT examples, "the answer is (X)" | last explicit answer declaration or `\boxed{}` letter | 1 |
| `mmlu_redux` | WildEval/ZeroEval curated MMLU-Redux, 2,778 questions; text and corrected gold from edinburgh-dawg/mmlu-redux | five subject-matched cais/mmlu dev examples in `<example>` blocks, target in `<target_question>`, answer as `{"answer": "C"}` | last JSON object with an `answer` key | 64 |
| `ifeval` | google-research `instruction_following_eval`, 541 prompts | author prompt, unchanged | all instructions satisfied under the author strict scorer | 64 |
| `gpqa_diamond` | idavidrein/gpqa archive, 198 questions, options shuffled with `random.Random(0)` | zero-shot, letter in `\boxed{X}` | last explicit declaration or `\boxed{}` letter; `\boxed{\text{C}}` accepted | 64 |
| `zebralogic` | WildEval/ZebraLogic grid_mode test, 1,000 puzzles | ZeroEval `ZEBRA_GRID` template | every cell of the last JSON solution correct | 64 |

The reported score is accuracy averaged over all samples (avg@n). Data is
downloaded on first use into `--data-dir` (default
`~/.cache/expdis/general_eval`) and checked against SHA-256 pins where the
original evaluation recorded them. IFEval also needs `absl-py`, `langdetect`,
`nltk`, and `immutabledict` (the original run used 2.3.1, 1.0.9, 3.9.1, and
4.2.1); the author scorer and NLTK `punkt_tab` data are fetched at pinned
revisions. Loading MMLU-Pro, MMLU-Redux, and ZebraLogic needs `pyarrow`.

```bash
for b in mmlu_pro mmlu_redux ifeval gpqa_diamond zebralogic; do
  python -m expdis_jax.general_eval --model-name Qwen/Qwen3-1.7B \
    --server-urls http://localhost:8000/v1 --benchmark $b \
    --output-path base/$b.json
done
# Repeat with --model-name /path/to/checkpoint --output-path method/$b.json, then:
python -m expdis_jax.general_eval --summarize \
  --base-results base/*.json --results method/*.json --output-path change.json
```

`--summarize` checks that both files of each benchmark share the same protocol
and reports, per benchmark, the change from Base in percentage points
("Change from Base (%)"), and the mean change over MMLU-Pro, MMLU-Redux,
IFEval, and GPQA-Diamond ("Prior capabilities Δ"). ZebraLogic is reported but
not included in the mean.

## Modules

- `config.py`: `TrainConfig`, command-line flags, `validate_contract`
- `pipeline.py`: single-round and multi-round drivers
- `parallel_pipeline.py`: concurrent explorers on separate allocations
- `train.py`: RL loop (weight sync, rollouts, scoring, dynamic sampling, update, checkpoints)
- `grpo.py`: policy-gradient loss
- `rewarding.py`: answer extraction, answer checking, reward
- `novelty.py`: multi-layer RND; kNN and elliptical bonuses
- `filtering.py`: rejection sampling
- `distill.py`: student SFT
- `lineage.py`: update budgets across explorers and rounds, λ schedule
- `data.py`: DAPO-Math-17K, DeepScaleR, and JSONL loaders; chat templates; prompt shards
- `generate.py`: vLLM client
- `eval.py`: evaluation
- `general_eval.py`: general-capability evaluation and change from Base
- `diversity.py`: token entropy, semantic diversity, reasoning faithfulness
- `model.py`: Qwen3 and Ministral decoder in Flax
- `weights.py`: conversion between Hugging Face safetensors and Flax parameters
- `mesh.py`: FSDP mesh and sharding
- `lr_schedules.py`: learning-rate schedules
- `scripts/`: TPU setup, vLLM bootstrap and reload, launchers

## Tests

The tests in `tests/` run on CPU and cover the loss and reward, RND, the filter,
budget splits, checkpoint handoffs, learning-rate schedules, padding, and a
small end-to-end pipeline:

```bash
pip install -r requirements-dev.lock
pytest tests -q
XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  pytest tests/test_expdis_jax_pipeline_cpu.py -q
```

`tests/test_expdis_jax_model_reference.py` compares the Flax forward pass with
Hugging Face Transformers. Set `EXPDIS_HF_REFERENCE_PYTHON` to a Python
environment with PyTorch and Transformers to run it.
