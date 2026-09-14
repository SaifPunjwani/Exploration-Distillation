# C10 survivor-consistent Overlong Filtering preregistration

Date: 2026-07-15. Status: preregistered before any C10 training or behavior-
gate result was observed. C5 remains a completed negative result and is not
relabeled, rerun, or threshold-adjusted by this document.

## Motivation and single experimental change

The predetermined C5 Scout completed all 50 updates with exact selected-batch
RND accounting and passed its rolling training-health gate. On the fixed
32-prompt behavior panel it passed every training-mode and paper-evaluation
criterion except paper-evaluation repetition: 10/64 rows were flagged versus
the Base-calibrated maximum of 7/64. The hard gate therefore failed and C5 did
not advance. Correctness was diagnostic only and did not affect that decision.

C10 retains the complete C5 contract and adds the DAPO Overlong Filtering
mechanism reserved in `C9_STABILIZATION_PREREGISTRATION.md`. The change is
defined as *survivor-consistent* filtering:

1. a completion is a survivor iff it did not hit the 16,384-token training
   completion cap;
2. dynamic-sampling eligibility is evaluated only over survivors and requires
   at least two survivors in the group;
3. correctness counts and novelty-contribution variance used by the
   eligibility rule are computed only over those survivors;
4. each selected group's reward mean is computed only over its survivors;
5. truncated rows receive advantage zero and are excluded from backward loss;
   and
6. the Dr.GRPO loss denominator remains the fixed full sampled geometry
   `64 * 16,384`, never the survivor count.

Thus a truncated row is absent from group-relative policy learning rather than
remaining in the group baseline while its own gradient is masked. The RND
predictor lifecycle and selected-batch z-score population remain exactly C5;
changing them would confound the mechanism screen. Soft-overlong shaping is
also retained for naturally terminating completions in the 80--100% budget
band. C10 is a labeled stabilization experiment, not a claim that the frozen
projection-era implementation already used these semantics.

## Fixed C10 step-50 canary

C10 uses seed 0, Qwen3-1.7B at the pinned revision, DAPO-Math-17K, one Scout,
one round, `lambda=0.5`, correct-only novelty credit, RND layers 7/14/21 and
RND LR `1e-4`, selected 64-row RND update exactly once per completed step,
selected-batch population z-score, Scout LR `1e-6`, four prompts by sixteen
generations, training sampling `T=1.0/top_p=.95/top_k=20`, 16,384 completion
tokens, KL zero, one current-policy update per rollout, and fail-closed dynamic
sampling with at most 64 candidate attempts. All code, image, command, model,
dataset, and orchestrator identities must be immutable and receipt-bound.

The canary runs exactly 50 Scout updates and no SFT, Central, or benchmark
evaluation. Every completed step must contain exactly four selected groups and
64 serialized learner rows, zero fallback groups, zero policy staleness,
finite losses/rewards/log-probabilities/gradients/RND values, and at most 5%
generation errors. RND update count must equal completed step. Every selected
group must record at least two survivors; eligibility statistics, survivor
counts, masked-row counts, and baseline statistics must replay exactly from
the raw rows. The existing rolling collapse rule is unchanged.

At step 50, the exact same pre-hashed behavior panel, prompt variants, paired
seeds, sampling parameters, 32,768-token budget, metric implementation, and
Base-calibrated thresholds used for C5 are applied unchanged. Both prompt
modes must satisfy termination >=0.80, clipping <=0.20, valid boxed answer
>=0.90, median length <=24,576, repetition <=7/64, unclosed-think <=0.20,
and generation errors <=0.05. Correctness is recorded but hidden from the
advancement decision. A transport error above 5% invalidates and reruns the
unchanged canary; any semantic or behavior-gate failure is a scientific
failure.

No threshold may be relaxed and no alternative seed or checkpoint may replace
C10 after results are observed. Only the fixed step-50 C10 endpoint may pass
this screen.

## Advancement and full pipeline

Only if the C10 canary passes all training, provenance, replay, and fixed-panel
gates may the identical C10 semantics advance to the seed-0 SingleScout full
pipeline. That pipeline keeps the already preregistered ExpDis contract:

1. exactly 200 Scout updates and 12,800 selected trajectory rows;
2. the same final Scout behavior gate;
3. deterministic `coverage_pool_c8` selection of exactly 500 correct,
   naturally terminated, valid, unclipped, deduplicated trajectories from 500
   unique problems, with no historical merge or top-up;
4. Base-initialized, EOS-aware completion-only SFT for two epochs, gradient
   accumulation one, and exactly 1,000 optimizer updates;
5. the same fixed SFT-endpoint behavior gate;
6. exactly 100 correctness-only Central updates at LR `1e-6`, novelty exactly
   zero, using the same survivor-consistent Overlong Filtering; and
7. exact final P1 evaluation at AIME24/25 avg@64, MATH500/AMC23 avg@32,
   Minerva-Math avg@64, and GSM8K avg@8.

Mid-training avg@4 probes remain non-headline diagnostics and cannot select a
checkpoint, stop training, or decide advancement. Historical MD/PDF numbers
remain targets rather than evidence. If seed 0 completes, the same frozen C10
contract—not a best-seed choice—extends to MultiScout, multi-round, and MR-ME.

