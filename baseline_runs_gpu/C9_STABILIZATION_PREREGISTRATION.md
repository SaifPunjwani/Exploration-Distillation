# C9 ExpDis Scout stabilization preregistration

Status: preregistered before any C9 result was observed. This document does
not alter or relabel the frozen C8 runs. C8 failures and all C9 mechanism
controls remain reportable negative evidence.

## Motivation

The exact seed-0 C8 Scout remained plausible at its step-25 diagnostic probe,
then collapsed by steps 47--49: clipping rose from 56.3% to 89.1%, validity
fell from 81.3% to 28.1%, and step 50 could assemble only three of four
required groups after 64 attempts. The exact correctness-only DAPO control on
the same stack remained operationally healthy and reached 52.5% on its
fresh-weight step-75 AIME24 avg@4 diagnostic. The factors isolated here are:

1. Scout learning rate (`5e-6` versus the DAPO `1e-6`);
2. raw versus selected-batch-z-scored RND novelty; and
3. retry-dependent candidate-attempt RND updates versus exactly one RND
   update on the selected 64-row learner batch.

## Step-50 mechanism screen

All cells use seed 0, one immutable code artifact, the same prompt stream, and
a durable training-dtype step-50 checkpoint. C5 is the predetermined C9
primary. No other cell may replace it based on benchmark results.

| Cell | RND update lifecycle | Novelty scale | Scout LR | Role |
|---|---|---:|---:|---|
| C0 | every candidate attempt | raw | `5e-6` | frozen C8 reference |
| C1 | every candidate attempt | raw | `1e-6` | LR control under C8 lifecycle |
| C2 | selected 64 rows once/step | raw | `5e-6` | lifecycle control at original LR |
| C3 | selected 64 rows once/step | raw | `1e-6` | lifecycle + LR control |
| C4 | selected 64 rows once/step | selected-batch z-score | `5e-6` | normalization control |
| **C5** | **selected 64 rows once/step** | **selected-batch z-score** | **`1e-6`** | **predetermined primary** |

C0 is already represented by the exact C8 seed-0 run. Candidate-attempt RND
updates cannot be combined coherently with selected-batch normalization,
because the selected batch does not exist until dynamic sampling finishes.

Every cell otherwise holds fixed Qwen3-1.7B, DAPO-Math-17K, `lambda=0.5`,
correct-only novelty credit, RND layers 7/14/21 and RND LR `1e-4`, four
prompts by 16 generations, training sampling `T=1.0/top_p=.95/top_k=20`,
16,384 training completion tokens, soft-overlong shaping, no truncated-loss
masking, KL zero, one policy update per rollout, current-policy reload after
every update, and at most 64 fail-closed dynamic-sampling attempts.

## Benchmark-blind health gates

Each completed step must have exactly four selected groups and 64 learner
rows, zero fallback groups, policy staleness zero, finite loss/rewards/logps/
gradients/RND values, and generation error rate at most 5%. A transport error
above 5% invalidates and reruns the unchanged cell; a dynamic-assembly failure
with healthy transport is a scientific failure.

For the selected-batch lifecycle, `rnd_update_count == completed_step`, with
exactly one update on exactly 64 rows per step. For z-scoring, the logged
center and scale must be finite, normalized selected-batch novelty must have
mean zero and population standard deviation one within numerical tolerance,
and every incorrect row must receive exactly zero novelty contribution.

After step 10, compute five-step rolling means. Fail an arm when three
consecutive rolling windows simultaneously have clipped/nonterminated rate
at least 0.50 and valid-answer rate at most 0.65. Neither correctness nor any
benchmark score participates in this gate.

At step 50, evaluate a pre-hashed non-benchmark panel of 32 DAPO-Math prompts,
two samples each, under both training and paper-evaluation prompt variants.
Use `T=.6`, `top_p=.95`, `top_k=20`, `min_p=0`, thinking mode, and 32,768
completion tokens. Both prompt modes must satisfy termination >=0.80,
clipping <=0.20, valid boxed answer >=0.90, median length <=24,576,
repetition <=7/64, unclosed-think rate <=0.20, and generation errors <=0.05.
Correctness may be recorded but is hidden from the advancement decision.

Only C5 advances if it passes all gates. If it fails, no other matrix cell is
promoted automatically; the next change must be separately preregistered.
DAPO-consistent survivor-only Overlong Filtering is reserved for a labeled
C10 experiment rather than confounded with C9.

AIME probes are disabled in the step-50 mechanism screen. In the full C5 run,
Scout and Central probes may be logged every 25 steps only as fresh-weight
avg@4 diagnostics; they cannot stop training, choose a checkpoint, or decide
advancement.

## Fixed full C9 pipeline and success criterion

The seed-0 SingleScout primary must:

1. complete exactly 200 Scout updates and 12,800 selected trajectories;
2. pass the final benchmark-blind behavior gate;
3. produce exactly 500 correct, naturally terminated, valid, unclipped,
   deduplicated trajectories from 500 unique problems using
   `coverage_pool_c8`, with no top-up or historical merge;
4. initialize SFT from Base and run EOS-aware completion-only loss for two
   epochs, gradient accumulation one, exactly 1,000 optimizer updates;
5. use the fixed SFT endpoint (not an earlier selected snapshot), which must
   pass the same behavior gate;
6. initialize Central from that endpoint and complete exactly 100
   correctness-only updates at LR `1e-6`, with novelty identically zero; and
7. evaluate the fixed final Central with P1 on AIME24/25 (64 samples/problem),
   MATH500/AMC23 (32), Minerva-Math (64), and GSM8K (8), at
   `T=.6/top_p=.95/top_k=20/min_p=0`, thinking mode, 32,768 tokens, exact
   sample-count guards, and generation error rate at most 5%.

The primary measured performance criterion is the hard-mean ordering

`ExpDis final Central > final DAPO > Base`

over AIME24, AIME25, MATH500, and Minerva-Math. Historical MD/PDF values such
as AIME24 52.10/52.57 and hard mean 49.28 are projection-era targets, not
checkpoint-selection criteria. P2 (38,912-token) AIME24/25 is a secondary
headline sensitivity run after the canonical P1 comparison.

If seed 0 completes, seeds 1 and 2 use the identical C5 Scout contract for
MultiScout. Seed 0 remains the predeclared SingleScout; the best seed is never
chosen post hoc.
