# GPU branch audit against the supplied ExpDis manuscript

Reviewed source: `gpu` at `1208f2161b282598f56aabc6a8655d6a7491b83d`, followed by
the local fixes described below. Audit date: 2026-09-14. The supplied latest
manuscript is the yardstick; historical campaign contracts are evidence of
different protocols, not a substitute for that manuscript.

## Prompt update — 2026-09-17

Default training and evaluation now append Qwen's math instruction after the
question in one user message (`qwen3_math_user_suffix_v1`). Runtime validators
and evaluation manifests record the new identity and reject the old format.
The suite passes 959 tests (1 skipped), including the tiny two-round pipeline.
This prompt update does not change the campaign-recipe differences below or
establish the provenance of historical paper scores.

## Verdict

The GPU stack has substantial tested training, filtering, orchestration and
receipt infrastructure. **Its default/c8/c9/c10 recipes are not an exact
implementation of the latest paper.** Do not present their checkpoints or test
results as reproducing the paper's JAX configuration. The main branch contains
the corrected reference implementation and its separate `AUDIT_REPORT.md`.

## A. Fixed bug

`tmx_gpu/grpo_gpu.py::_train_update` previously returned `updates=1` without
calling `optimizer.step()` when every advantage was zero or every row was
masked. That changes Adam's momentum decay and step counter, and therefore
subsequent updates. The function now supplies explicit zero gradients and takes
the scheduled optimizer step. It also rejects empty/misaligned row and advantage
arrays. `tmx_gpu/tests/test_zero_gradient_update.py` compares both cases against
an independent AdamW reference after a preceding nonzero update.

The pipeline now accepts Explorer/Main CLI names while retaining the legacy
aliases. The launcher protects both spellings from overriding campaign identity
through `EXTRA_ARGS`. The tiny two-round run uses the new flags; the legacy
commands remain covered by the rest of the suite.

## B. Remaining differences from the manuscript

| Mechanism | Paper | GPU implementation / location |
|---|---|---|
| Completion budget and denominator | 32,768 / 32,768 | Defaults 16,384 / 16,384: `grpo_gpu.py:5014`. `--completion-budget` changes the generation cap and soft window, but does not by itself establish the complete paper protocol. |
| Soft-overlong window | 26,214 + 6,554 | Defaults 13,107 + 3,277: `grpo_gpu.py:5061`. |
| Clipped rows | Excluded from policy loss, retained in all-row advantages and pool | Default `mask_truncated=False`; c10's survivor-only baseline/eligibility also changes which rows determine advantages. See `grpo_gpu.py:2264`, `:3277`, `:5024`. |
| Objective | Asymmetric clipped objective, one step | Default REINFORCE; `--ppo-mode` supplies the clipped objective, but `validate_rl_validity` rejects it for c8 (`grpo_gpu.py:1615`, `:3247`, `:5038`). At ratio one, the first-step gradient agrees; the implementation and runtime contract still differ. |
| Sampling snapshot | Sync before every batch and stage | Standalone legacy reload interval is 10; c8 requires managed serving and every-update sync. Stage reloads exist for a managed pool. Unmanaged endpoints cannot establish that identity. |
| Dynamic sampling | Total-reward spread, eight attempts, zero-variance fallback | Legacy and c8 have different rules; c8 uses novelty-contribution spread for homogeneous groups and excludes fallback. See `group_reason`, `group_eligibility_for_rows` (`grpo_gpu.py:2153`) and collection around `:2980`. |
| RND timing | Score against frozen predictor for the iteration, update selected features after policy | c8 updates on each candidate attempt (`grpo_gpu.py:2868`). c9 freezes candidate scoring but updates on the selected batch before `_train_update` (`:3163`). Some c9 cells standardize novelty. |
| Feature window/layers | Full completion; automatic quarter-depth layers per model | Defaults 16,384 and literal `7,14,21` (`grpo_gpu.py:5065`), appropriate to Qwen3-1.7B's depth only. |
| Pool construction | All selected Explorer training rows, then filter | Campaign paths can harvest additional completions to meet a minimum pool size; c8/c9 profiles bind particular harvest and evidence rules. See `pipeline_gpu.py:4716`, `:5090`. |
| Filter | Strict gates, 128–4,500 tokens, one shortest winner/problem, cap 500 | Generic `quality_pool` has a 16,000-token upper gate (`filter_pool.py:68`) and differs from the separate, stricter `coverage_pool_c8` policy. `filter_pool.py:423` defaults to `quality_pool`; the pipeline's min-accepted default is 500 (`pipeline_gpu.py:7247`) rather than permitting any nonempty pool up to 500. |
| SFT/Main selection | Two epochs, then Main RL on resulting weights | Campaign arms include behaviour gates and checkpoint selection, plus provenance-bound c9/c10 variants. Those are additional algorithmic choices. |
| Checkpoint cadence | 50 updates | Standalone GPU default is 25 updates. |

Shared components that do match include ±1 correctness, correctness-gated raw
novelty in the ordinary raw mode, group-mean advantages without standard
deviation, fixed per-row length normalization, zero KL, B=4/G=16, the default
AdamW parameters and stage learning rates, completion-only SFT, and Main reward
without novelty. Matching individual components does not make the whole
campaign a paper reproduction.

Changing c8/c9/c10 constants in place would mislabel their frozen receipts.
A future GPU paper profile must separately implement the full combination
above, preserve the old identities, and be trained/evaluated again.

## C. Tests and commands actually run

- Full suite: `python -m pytest tests tmx_gpu/tests -o addopts='' -q`:
  **957 passed, 1 skipped**, 78.25 seconds in the final rerun on 2026-09-15,
  using the fresh Python 3.13/pip audit environment. The skip is the opt-in live
  pinned-dataset download audit; the tiny-model training/resume tests ran.
- Runtime used: Python 3.13.13, PyTorch 2.14.0, Transformers 5.17.0, JAX 0.11.1,
  Flax 0.12.9, Optax 0.2.8, NumPy 2.5.3, pytest 9.1.1, datasets 5.0.1.
  This environment is now pinned in `requirements-test.lock`. Installing
  the complete `requirements.txt` environment did not finish; its older
  dependency combination is **not** certified by this result.
- All five `python -m tmx_gpu.{pipeline_gpu,grpo_gpu,sft_gpu,filter_pool,eval_gpu}
  --help` entry points exited 0 (run individually).
- `scripts/tmx_paper.py --help`, `scripts/tmx_submit.py --help`,
  `scripts/tmx_submit.py list-presets`, `scripts/tmx_paper.py validate all`, and
  `scripts/tmx_paper.py plan explorer-novelty --backend raw-tpu-tmx-jax` exited 0.
  Validation and planning are local checks; no submission was executed.

The fresh environment passed `pip check`. Fresh-install validation used an empty
HF cache with credential variables removed. The final suite also ran with
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`. The tiny
Llama and tokenizer are now initialized locally, replacing download-dependent
fixtures that could skip the pipeline tests. Two-round optimization and
crash/resume tests use the real model with a stub serving endpoint. They do not launch CUDA
training or establish real vLLM restart/reload behaviour.

## D. Documentation and hygiene

The README no longer calls the GPU default the paper's DAPO configuration. It
states the different defaults and links this audit. The installation guidance
now includes the JAX dependencies imported by the complete test suite.
The invalid explanation that PPO clipping requires nonzero KL was corrected:
the c8 restriction is a campaign choice, not a PPO requirement.

Removed a deployment hostname from `baseline_runs_gpu/C7_PROVENANCE.md`. The
remaining old project-name strings identify external artifacts, historical jobs
and tests of those identifiers; the README explains that exception. Scans
found no personal filesystem paths, email addresses or common literal API-token
patterns. This is a static scan, not a claim that arbitrary credentials can
always be detected.

The publication branch contains one initial commit authored by Saif Punjwani,
including these fixes. It has no parent commits or development history.

## E. What still requires hardware and run evidence

No accelerator training, seven-benchmark evaluation, three-seed replication,
wall-time matching or reported paper accuracy was reproduced here. The paper's
result tables cannot be verified from source/tests alone. Retain exact source
and dependency versions, model/dataset revisions, seed/configuration, raw
rollout pools, filter receipts, checkpoint lineage and timing when conducting
that reproduction. Use the same evaluation harness for every compared arm.
