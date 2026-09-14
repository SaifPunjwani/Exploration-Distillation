# c8 ExpDis implementation-repair contract

Status: frozen before GPU launch. This document defines the corrected GPU
replication run and separates method-preserving repairs from new-method
experiments.

Scientific label: **c8 corrected/intended ExpDis**. It follows the originally
stated one-trajectory-per-problem method, but it is not a byte-exact rerun of
the historical TPU selector, which globally ranked and capped correct rows.
The fresh c8 Scout trajectory artifact will therefore also feed a matched
`naive_pool` historical-selector control with otherwise identical SFT and
Central stages; that control isolates the selector repair without retraining
the Scout.

## Scientific contract held fixed

- Model: Qwen3-1.7B, initialized from the same pretrained checkpoint for Scout
  and Central.
- Dataset: DAPO-Math-17K, using the existing deterministic shuffle/shard rules.
- One round: 200 Scout DR-GRPO steps, then SFT, then 100 Central DR-GRPO steps.
- The Scout contributes exactly 12,800 training-time trajectories
  (200 steps x 4 prompts x 16 generations). No post-Scout harvest/top-up is
  permitted in the canonical row; a short or malformed pool fails closed.
- Rollout geometry: 4 prompts per step, 16 completions per prompt, temperature
  1.0, top-p 0.95, top-k 20.
- Training completion budget: 16,384 tokens. Evaluation completion budget:
  32,768 tokens.
- Scout reward: correctness plus lambda times multilayer-RND novelty, with
  lambda = 0.5 and novelty credit restricted to correct completions.
- Central reward: correctness-only RLVR plus the contract's DAPO overlong
  shaping; no novelty reward.
- Optimizers and rates: the existing contract values (Scout AdamW 5e-6, SFT
  AdamW 5e-6 for two epochs with per-example optimizer stepping, Central
  AdamW 1e-6), KL beta 0, group size 16,
  DR-GRPO fixed denominator, DAPO asymmetric clipping, and soft-overlong
  reward shaping. Truncated-loss masking is disabled, matching the actually
  run TPU reference (`mask_truncated_completions=False`); the c7 GPU-only
  masking change is not carried into the canonical replication.
- Primary reporting uses the final checkpoint and the frozen six-benchmark
  evaluation protocol. Small AIME probes are health diagnostics, not a model
  selection rule.

## c8 implementation repairs

These changes make the GPU implementation match the intended experiment; they
are not additional ExpDis components.

1. **Coverage-preserving QualityPool.** Apply the existing correctness,
   termination, matching-boxed-answer, unclipped, and exact-duplicate gates;
   retain at most one trajectory per problem; use a deterministic
   quality/length tie-break within a problem; and never globally rank examples
   from different problems by raw nonstationary RND magnitude. These c8 gates
   deliberately do not inherit the experimental QualityPool 128/16000-token
   bounds or looping heuristic, which were not part of the frozen contract.
   For a one-Scout primary run this yields broad problem coverage. MultiScout
   source balancing is logged and evaluated separately. A labeled
   `original_blended_per_problem` control selects the highest blended-reward
   candidate within each problem, then shorter, while preserving the same hard
   gates and coverage cap; it is not substituted for canonical c8.
2. **Termination-aware SFT.** Append exactly one model EOS/chat-end token to
   every accepted completion, include it in the completion-only loss, and
   preserve it when truncating. Restore the TPU's per-example optimizer
   stepping (`grad_accum=1`; c7 used the labeled GPU deviation 8). Log
   appended, already-present, and truncated counts plus actual optimizer-step
   count.
3. **Fresh-policy rollouts.** In c8 validity mode the serving policy is synced
   before the next rollout after every learner update. The primary remains one
   update per rollout; no replay or multi-epoch PPO is introduced.
4. **Arm-invariant group eligibility.** Homogeneous-correctness groups may be
   admitted only when the novelty term that actually enters the reward supplies
   variance. Overlong-penalty variance alone cannot make a lambda arm eligible
   when the correctness control would be skipped. Rejected groups never
   back-fill the c8 learner batch. c8 continues resampling up to a generous
   64-attempt operational ceiling; if even that cannot assemble the declared
   batch geometry, the run fails closed rather than training on an ineligible
   fallback group.
5. **Complete recovery state.** Checkpoints save and restore the learner
   optimizer alongside the model and RND state. A recovered run is not silently
   restarted with a fresh optimizer. Only resumable `step_*` checkpoints carry
   optimizer moments; phase-final/deployment models remain lightweight. One
   local step checkpoint is retained per phase, with durable cadence versions
   stored in W&B.
6. **Fresh diagnostic probes.** Probe generation uses explicitly synchronized
   current weights. Probe values remain diagnostic only.

Legacy modes remain available so c7 can be reproduced exactly; c8 flags are
explicit in every launch and are written to W&B configuration and summaries.

## Launch gates

1. Unit and pipeline tests must pass before publishing the code artifact.
2. A banked-trajectory SFT-only canary may be used to test the selector and EOS
   repairs cheaply. It is labeled `c7-reuse-canary` and is not a canonical ExpDis
   result because its Scout was trained under the earlier 32K c7 configuration.
3. The SFT-only canary must restore healthy termination/format behavior before
   Central RL is allowed: no cap-pinned median, valid-answer rate at least 90%,
   and clipped rate no more than 10 percentage points above the measured base
   under the same diagnostic protocol. Accuracy is reported but is not used to
   tune against AIME.
4. The canonical c8 result is produced only by a fresh 16K Scout-to-Central run
   under the fixed contract above.

### Gate-resume implementation boundary

The currently implemented external-SFT continuation path is deliberately
fail-closed and always records
`contract+scientific-variant:external_sft_exposure_gated` (or the composed
GPU-MAX tier if any underlying flag differs). It is suitable for the
predeclared `c7-reuse-sft-exposure` diagnostic only and cannot produce or be
renamed as the canonical c8 row. Its evidence bundle must bind the selected
model artifact and digest, all four frozen held-out gate artifacts, the
latest-passing selection manifest, the exact frozen accepted-library semantic
digest, and the exact source trajectory artifacts and file digests. For this
diagnostic, the accepted library is copied byte-for-byte from
`gpu-jrl-c8-c7reuse-sft-ga8-checkpoints-20260713-results:v1`: exactly 500 rows
under `sft_prompt_completion_multiset_v1`, SHA-256
`337ab6ebd154c9ef6a3616cc72aa7a4c8ae15e89a732ad929464218945f8712f`.
The external continuation fails closed for any other artifact, row count, or
semantic digest; it does not silently re-filter the c7 bank under repaired c8
rules.

The training-only precursor is separately exposed as
`frozen_c7_library_ga8_sft_retrain`. It downloads that exact results artifact
into a freshly cleaned digest-scoped root, validates the W&B artifact digest
`64b9241d1ba4509395ff6f4473c794ee` and every downloaded member, copies its
`accepted.jsonl` byte-for-byte, and binds the original Scout trajectory
artifact refs/digests/file hashes. It then **runs** two-epoch AdamW SFT from
Base (it does not reuse or skip to an existing model) with gradient
accumulation 8 and preregistered snapshots at updates 16/32/64/126. The route
requires `--stop-after-sft`; no snapshot can enter Central until the four
independent gates and correctness-blind selector complete. It is explicitly
noncanonical (`gpu-max:grad_accum` at the SFT stage) and cannot be renamed as
fresh c8.

Any selected external checkpoint that later enters Central binds a second,
independently recomputed runtime contract: exactly 100 steps, model/init and
resolved learning rate/optimizer, c8 validity, completion and serving lengths,
soft-overlong and fixed-denominator settings, group geometry, sampling and
dynamic-sampling settings, truncation masking, clipping/loss mode, dataset,
seed, and other generation-affecting arguments. That exact object is included
in the signed lineage and every Central checkpoint/final state. Resume and
completed-final reuse require exact equality, and a deployable final whose
recorded step is not exactly 100 is rejected.

Canonical gate-resume support is implemented as the separate
`fresh_c8_single_scout` evidence profile; it does not relax or reinterpret the
frozen c7-reuse path above. A fresh run must opt in before SFT with
`FRESH_C8_SFT_GATE=1`, `STOP_AFTER_SFT=1`, `SFT_GRAD_ACCUM=1`, and
`SFT_SAVE_STEPS=125,250,500,1000`. The launcher and pipeline then prove, from
durable checkpoint provenance, all fixed-contract facts in addition to the
behavioral gate: exact code-artifact membership, Base initialization,
`grad_accum=1`, exactly 1,000 planned and completed optimizer updates, the
200-step fresh c8 Scout/RND/runtime contract, one exact 12,800-trajectory
training-time source file (64 rows at every step), and exactly 500
`coverage_pool_c8` rows from 500 unique problems. Snapshot artifacts are not
uploaded until a signed completed-run record proves that all four
preregistered post-update snapshots exist and the 1,000 updates completed.
The code identity and both byte-level and parsed-semantic trajectory hashes
are recorded in signed evidence at the instant the Scout completes and are
embedded in its final trainer state. A resumed pipeline re-hashes the code,
model state, and trajectory file against those completion-time facts before
reusing the Scout; it may not manufacture provenance later from whatever
bytes happen to be present when SFT starts.

Before Scout update 1, the pipeline writes an atomic code-binding sidecar with
the exact artifact `name:vN`, W&B digest, member hashes, and file-manifest SHA,
then publishes that sidecar in the versioned results artifact.
Mutable aliases such as `latest` are rejected. Every Scout `step_*` checkpoint
and final trainer state embeds the identical binding. Any pre-completion retry
independently re-hashes the current artifact and compares it with every local
checkpoint before entering GRPO or loading checkpoint bytes; an old checkpoint
with no binding, or any version/digest/file-manifest drift, fails closed.
Changing `CODE_ARTIFACT` therefore requires a new run from step zero.

Each snapshot is evaluated independently with the same accepted-library-
disjoint, dual-prompt gate and frozen thresholds described below. Selection
must be run with `--selection-profile fresh_c8_single_scout`,
`--accepted-jsonl`, and `--accepted-download-manifest`. It revalidates the
accepted artifact bytes, recomputes every persisted training-prompt hash and
seed-selection digest from the held-out prompt/problem fields, rejects either
problem-ID or training-prompt-hash overlap, recomputes all four verdicts from
raw rollouts, and applies the same correctness-blind latest-passing rule. If
no checkpoint passes, no Central is launched. Before this selection, the
finished SFT stage is recorded only as `canonical_pending_gate`; it makes no
canonical claim. If
update 1,000 is selected, the continuation preserves the canonical c8 tier
only after independently binding exact code, selected-model, trajectory,
accepted-library, four-gate, selection, and full Central-runtime artifact
identities. The external Central resolves the pipeline sentinel
`DYNAMIC_MAX_ATTEMPTS=0` exactly once to the frozen c8 value 64 and rejects any
explicit or resolved value other than 64 before signing lineage. If update
125, 250, or 500 is selected, the implementation forces
the scientific label `fresh_c8_sft_exposure_gated`; a caller cannot override
that label. Completed SFT/Central stages and GRPO resume checkpoints revalidate
the same signed lineage and runtime objects and fail closed on any drift.

## Predeclared SFT exposure diagnostic after the failed canary

The exact `grad_accum=1` c7-reuse canary completed 1,000 AdamW updates and
failed the launch gate through nontermination (74.27% clipped, 38.18% valid,
median exactly 32,768), even though 87.45% of the generations that did finish
were correct. The selector, EOS tokenization, export, serving, and evaluator
were separately audited and did not explain the failure. Consequently:

The independent repeated gate artifact
`gpu-jrl-c8-sft-gate-c7reuse-step1000-20260713-sft-gate:v0`
(digest `555dd67041e85a94f863b6e38ab1ae77`) confirmed the same failure under
both frozen prompt modes. Under the paper-evaluation prompt it measured
29.69% correctness, 39.06% termination, 60.94% clipping, 39.06% valid boxed
answers, a 32,768-token median, 56.25% repetition, and 60.94% unclosed
`<think>` tags, with zero generation errors. These values are diagnostic and
do not participate in exposure-checkpoint selection.

1. The failed 1,000-update result is retained; it is not replaced or hidden.
2. Before the fresh c8 pool exists, a labeled `c7-reuse-sft-exposure` diagnostic
   preserves the 500 examples, two epochs, AdamW, learning rate 5e-6,
   completion-only cross entropy, and EOS supervision, but uses gradient
   accumulation 8 (126 optimizer updates). Full evaluation checkpoints are
   fixed in advance at optimizer updates 16, 32, 64, and 126.
3. Checkpoints are judged on a deterministic accepted-library-disjoint
   DAPO-Math set under both the training and paper-evaluation system prompts.
   The frozen gate uses termination, clipping, boxed-answer validity, length,
   repetition, think-tag closure, and generation-error rates. Correctness is
   recorded for diagnosis but cannot affect the verdict.
4. If multiple checkpoints pass every health criterion, the **latest passing
   checkpoint** is selected, maximizing exposure to the Scout library subject
   to the predeclared behavior constraint. If none passes, no checkpoint from
   this diagnostic may enter Central RL.
5. The fresh c8 Scout pool still receives the exact 1,000-update canonical SFT
   attempt first. Only if that fresh-data endpoint fails the same independent
   health gate may the predeclared exposure-controlled variant be reported as
   a labeled repair rather than the canonical row.

### Base calibration of the repetition threshold

Before any of the exposure checkpoints was evaluated, the frozen held-out gate
was run on the unmodified Qwen3-1.7B Base with the same 32 prompts, two samples
per prompt, paired seeds, and both prompt modes. Base passed every criterion
except the paper-prompt repetition threshold: 7/64 generations were flagged
(`0.109375`) against the provisional maximum `0.10`; the training-prompt rate
was 5/64. This is a one-generation discretization failure of a gate that a
healthy reference model must itself satisfy. Therefore, before inspecting any
checkpoint-gate output, the sole calibrated change is:

- `max_repetition_rate = 7/64 = 0.109375`, i.e. no exposure checkpoint may
  have a higher repetition rate than the measured Base in either prompt mode.

All other thresholds, prompts, held-out examples, sampling seeds, and the
latest-passing selection rule remain unchanged. Base correctness was recorded
for diagnosis only and was not used for this calibration or for checkpoint
selection. The original Base gate artifact and its provisional failure are
retained rather than overwritten.

## Explicitly out of scope for the primary c8 row

- KL regularization, replay, changed learning rates, more or fewer primary
  steps, extra SFT epochs, larger pools, 24K/32K training, LM-judge filtering,
  semantic embedding filters, novelty re-normalization changes, or AIME-based
  checkpoint cherry-picking. Enabling truncated-loss masking is also a labeled
  Overlong-Filtering ablation rather than the canonical TPU-reference row.
- Those can be studied later as labeled ablations. They cannot replace the
  final-checkpoint c8 primary row.

The historical MD/PDF values are replication targets, not measured evidence
and not constraints used to alter the method.
