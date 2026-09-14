# C9 ExpDis selected-batch RND stabilization contract

Status: implementation-only; no C9 job is launched by this change. C9 is a
labeled scientific variant and is not a revision of the frozen C8 evidence.

## Immutable identity

A C9 Scout must set all of the following:

- `--expdis-stabilization-mode c9_selected_batch_update`
- `--scientific-variant expdis_c9_selected_batch_rnd`
- `--rl-validity-mode c8`
- `--novelty-normalization raw` **or**
  `--novelty-normalization selected_batch_zscore`
- `--incorrect-novelty-scale 0`
- an immutable `--code-source-manifest` bound to a versioned W&B code artifact

The resolved Scout learning rate, normalization, RND lifecycle, update scope,
gate order, and code identity are embedded in the runtime contract and every
checkpoint. C9 W&B artifacts and metric rows carry the same variant fields.
Resume additionally requires the persisted C9 RND selected-batch update count
to equal the completed Scout learner step.

## Per-step lifecycle

1. Hold the RND predictor fixed for the entire dynamic-sampling step.
2. Score every error-free candidate against that same predictor state.
3. Decide group eligibility using raw, correct-gated novelty contribution.
   Incorrect completions receive zero novelty credit. Soft-overlong variation
   cannot make a homogeneous group eligible under the C8 validity rule.
4. Once exactly `prompts_per_step * num_generations` rows are selected (64 in
   the headline geometry), recompute every selected learner reward under the
   declared normalization:
   - `raw`: use the selected row's raw novelty;
   - `selected_batch_zscore`: population-z-score all selected raw novelty
     values with epsilon `1e-6`, then apply the correct-only gate.
5. Update the RND predictor exactly once on those final selected feature rows.
6. Compute group-mean advantages from the consistently recomputed selected
   rewards and perform the policy update.

If dynamic sampling cannot assemble the learner batch, neither the RND
predictor nor the policy is updated. Discarded attempts never train RND.

## Backward-compatibility boundary

The default remains `c8_candidate_attempt_update + raw` with an implicit Scout
learning rate of `5e-6`. Its existing `score_and_update` implementation,
runtime-contract schema, protected fresh-C8 profiles, and validators are left
unchanged. A selected-batch normalization or the C9 label on that default path
is rejected. The protected seed-0/seed-1/seed-2 fresh-C8 profiles also reject
explicit C9 lifecycle or learning-rate flags.

## Canary grid (not results)

The pipeline and launcher expose a fail-closed `canary` execution profile and
bind each job to an exact `STABILIZATION_CELL`.  The complete preregistered
mechanism screen is:

| RND lifecycle | normalization | Scout LR |
|---|---:|---:|
| C8 candidate-attempt | raw | `5e-6` (C0; completed frozen reference) |
| C8 candidate-attempt | raw | `1e-6` (C1) |
| C9 selected-batch | raw | `5e-6` (C2) |
| C9 selected-batch | raw | `1e-6` (C3) |
| C9 selected-batch | selected-batch z-score | `5e-6` (C4) |
| C9 selected-batch | selected-batch z-score | `1e-6` (C5 primary) |

The failed frozen-C8 `raw, 5e-6` run is C0 and is not rerun.  Canaries are
seed-0, 50-step, Scout-only runs with AIME probes disabled.  Profile validation
rejects drift in lifecycle, normalization, LR, seed, model/data, group geometry,
retry ceiling, sampling, budget, current-policy reload, masking, or code
identity. Advancement is based on preregistered training-health gates, not a
diagnostic AIME probe.

## Probe integrity in the next artifact

The same patch makes `min_p=0` explicit and logs an AIME24 probe score only
when all 30 problems have exactly four successful generations (120/120). The
primary diagnostic key is `eval/AIME24_avg_at_4`; the historical
`eval/AIME24_avg` alias is emitted only for the same complete protocol. Partial
probes log counts and protocol metadata but no accuracy.
