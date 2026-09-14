# ExpDis implementation audit

Reviewed 2026-09-14 through 2026-09-17 against the authors' manuscript,
*Exploration-Distillation: Decoupling Exploration from Optimization in RLVR*,
especially Method, Algorithms 1–2, and the training/verifier appendices.
Reviewed starting commits:

- `main`: `3aa3097f5de881f226f691f5c8cb4cf3842e033c`
- `gpu`: `1208f2161b282598f56aabc6a8655d6a7491b83d`

This release includes the audit fixes described below. It does not establish
which code produced historical paper results. The initial audit left the original research working repository untouched.
The 2026-09-17 prompt update also aligns its active prompt renderers. The publication branches each contain one
initial commit, with no parent history.

## Prompt consolidation — 2026-09-17

New training and evaluation use `qwen3_math_user_suffix_v1`: Qwen's math
instruction follows the question in a user message, without a system message.
JAX, GPU, the active TRL implementation and the HF inference loader share this
format. Evaluation metadata distinguishes it from historical system-message
pools. Qwen3-1.7B, Qwen3-4B and Ministral release tokenizers produce identical
training/evaluation prompts. Existing weights and table scores are unchanged;
this update does not establish their original training-stack provenance.

Validation: the JAX suite passes 161 tests and 10 subtests (5 dependency skips),
including actual small-model training and round handoffs. The active TRL
prompt/evaluation checks pass 30 tests. Accelerator reproduction remains open.

## Outcome

The JAX core and complete Explorer → filter → Main SFT → Main RL flow have been
corrected and exercised with actual small-model optimization, checkpoint
restoration, distributed processes and independent Hugging Face forwards.
Full multi-host handoffs now have an explicit shared-checkpoint path. A separate
process launcher supports concurrent Explorers with declared disjoint resources.

**This is not yet an empirical reproduction of the paper.** TPU compilation/execution,
real vLLM reloads, 32k memory/performance, equal elapsed time, and the reported
benchmark gains require accelerator runs. The GPU campaign defaults still
differ from the latest manuscript; see that branch's `GPU_AUDIT.md`.

## A. Bugs fixed and their consequences

| Finding | Fix and evidence |
|---|---|
| Token lengths/rewards could be based on estimates or re-tokenized decoded text, losing sampled action/EOS identity. | `generate.py` requests and validates returned sampled token IDs; `train.py` uses them before rewards, clipping, novelty and policy loss. The selected pool retains them for SFT. Training rejects a server that omits them under the paper contract. Regression tests use deliberately inconsistent text/tokenizations. |
| Training correctness inspected only a response suffix. A valid final boxed answer earlier in the response could be missed. | Full-response grading in `train.py`; long-tail regression. The strict SFT filter still grades the boxed payload itself. |
| Predictor updates occurred during candidate scoring, changing novelty across retries and preceding the policy step. | Candidate scoring freezes the predictor; retained feature batches update it once after policy optimization (`train.py`). The target remains frozen. |
| Typed JAX PRNG keys could fall through to a constant seed. | `novelty.py` extracts key data explicitly; independent keys now produce distinct predictor/target initializations. |
| Hard-coded layer indices did not generalize across the paper's model depths. | Quarter-depth automatic selection; explicit incompatible selections fail the paper contract. Requested hidden layers preserve caller order. |
| Local Qwen configs and Ministral text decoders did not implement all required positional/numerical details. | Config/model fixes cover nested text configs, nested RoPE parameters, Ministral position scaling, padding-aware positions and fp32 RMSNorm arithmetic. Four independent HF forward comparisons cover Qwen3/Ministral3 in fp32 and bf16. |
| The chunked output head used different compute precision from the model forward. | bf16 head computation followed by fp32 log-softmax; numerical comparison against the dense head. |
| SFT could consume a different token sequence or spend the completion window on an oversized prompt. | Preserve sampled completion IDs and cap the prompt separately; completion-only CE on the accepted trajectory. |
| Published Ministral tokenizer metadata could not load in Transformers 4.x. | Require Transformers 5.17+, regenerate the exact CPU lock, and request the Mistral regex compatibility handling in all tokenizer entry points. Public Qwen3-1.7B, Qwen3-4B and Ministral-3-3B BF16 tokenizers render and encode the paper prompt. A local modern-metadata regression runs offline. |
| Weight loading could fetch native Mistral and HF copies together; exporting fetched base weights unnecessarily. | Download only HF `model*.safetensors`, obey the shard index, ignore native duplicates, and fetch only metadata during export. Quantized configs fail before downloading weights. Tests include a deliberately invalid native sidecar. |
| SFT's 6,548-token window was incompatible with Pallas' 128-token tile divisibility. | The attention adapter pads to a whole tile and slices the output back to the original length. The actual Pallas forward and custom backward match dense attention at lengths 128 and 129 in CPU interpretation mode. TPU compilation is still untested. |
| Short novelty/SFT inputs paid for the full padded context; an unused fallback replaced hidden states with text hashes and Adam with manual SGD. | Bucket sequence lengths, trim trailing padding per novelty microbatch, and retain every active token. SFT stays on host arrays until device placement. Removed the incompatible fallback; stale enablement now fails validation. Features, SFT loss and gradients match full padding in tests. |
| A stage could begin without first installing its current weights in serving. | Mandatory stage-start synchronization, including resume; missing reload bundle is fatal. Transport is mocked in offline pipeline tests, so remote acknowledgment still needs a hardware test. |
| An HTTP peer could fetch the previous batch before the next was published. | Generation-specific URLs and atomic publication/acknowledgment in `train.py`. An actual two-process test deliberately requests early and changes the batch values; an HTTP unit regression rejects stale/future IDs. This race caused a failure in the full audit run before the fix. |
| Cross-host SFT/Main and round handoffs were blocked or depended on host-local storage; source-only parameter collection was unsafe. | `checkpointing.py`, distributed SFT input placement, all-host parameter gathering and shared stage/round paths. Two-process tests complete K=1/R=1 and K=2/R=2. Uncommitted checkpoint directories are rejected. |
| Checkpoint retries could silently omit optimizer state or allow training to continue after both saves failed. | Preserve the entire requested payload on retry and raise after failure. Shared writes require a completed Orbax checkpoint before reusing an existing path. |
| Resume advanced prompt RNG as if each update used one draw, despite variable dynamic retries. | Paper-contract iterations derive their prompt and generation-seed stream from stage seed and update index. Serving/runtime nondeterminism can still prevent bitwise reproduction. |
| Multi-Explorer execution was sequential only. | Retained the native sequential driver and added `parallel_pipeline.py`, which launches independent groups concurrently, checks resource separation and pool counts, waits before Main training, and records elapsed times. Remote scheduler/device enforcement remains the deployment's responsibility. |
| Evaluation defaults/aggregation and hard-coded historical baseline deltas could misrepresent a new run. | P1 sample counts, local JSONL benchmark input, shared verifier/answer normalization and same-pool estimators in `eval.py`; removed unrelated fixed baseline deltas. |
| GPU zero-gradient batches reported an update without advancing Adam. | Explicit zero-gradient optimizer step, verified against an independent AdamW reference. Historical GPU protocol identities otherwise remain unchanged. |
| The primary launcher ignored the documented round-count variables, assumed personal environment paths and selected a personal HF destination. | Forward round geometry/schedule and seed, accept activated environments, default Explorer novelty to 0.5, require explicit reload destination/worker coordinates, and leave older runs intact by default. Seven isolated launcher checks validate parsed configs and missing-setting errors without launching training. |
| Scheduled probes imported a removed baseline constant; standalone benchmark tracking used AIME labels for other datasets. | Repair the probe, record model identity, and label each benchmark/sample count correctly. The evaluation CLI also aggregates the five primary benchmarks and reports normalized-answer entropy. Tests reject mixed, partial and duplicate benchmark files. |
| TPU setup could upgrade JAX after installing its TPU dependencies, and the multi-host launcher dropped round settings. | Resolve the pinned JAX/Flax/Optax/Orbax stack together; forward round geometry, schedule, seed and Main settings on every host. Linux x86_64 dependency resolution succeeds for glibc 2.31. Hardware installation remains untested. |

## B. Reproductions and focused tests

All tests below run without downloading pretrained weights or benchmark data.

| Regression / invariant | Test file |
|---|---|
| Long response verifier, token accounting, RND ordering/key identity, failed reload | `tests/test_expdis_jax_rollout_contract.py` |
| Sampled-ID preservation, eight-attempt fallback, fixed denominator, nontrivial clipping derivatives, bf16 head, configuration drift | `tests/test_expdis_jax_sampling_regressions.py` |
| Padding invariance, actual Pallas forward/backward in CPU interpretation, removed fallback | `tests/test_expdis_jax_padding.py` |
| HF shard selection, metadata-only export, early quantization rejection, modern tokenizer metadata | `tests/test_expdis_jax_weights.py` |
| Shared artifact worker, required upload failure, no JAX import in child | `tests/test_expdis_jax_artifacts.py` |
| HTTP batch publication identity | `tests/test_expdis_jax_batch_transport.py` |
| Model forward versus independently initialized HF Qwen3/Ministral3 | `tests/test_expdis_jax_model_reference.py`, `tests/hf_forward_reference.py` |
| Actual RL → accepted JSONL → SFT → Main RL → checkpoint/HF export | `tests/test_expdis_jax_pipeline_cpu.py` |
| Two separate JAX processes, early batch requests, shared SFT/Main checkpoints and two rounds/two Explorers | `tests/test_expdis_jax_multihost_cpu.py` |
| Concurrent processes, disjoint resource declarations, 17/17/16 × four rounds, Main lineage, failed-Explorer containment | `tests/test_expdis_jax_parallel_pipeline.py` |
| Interrupted versus committed shared checkpoint | `tests/test_expdis_jax_shared_checkpoint.py` |
| P1/sample counts, full-response verifier, exact-rational normalization, avg/pass/answer diversity from one pool | `tests/test_expdis_jax_eval_protocol.py` |

Existing filter, novelty, objective, round-sharding and weights-only/resume tests
also pass. The tiny integration tests replace only external data, serving,
logging and mirroring; policy and predictor optimization, filtering, SFT, model
export and checkpoint restore execute real code. Their reduced geometry is
explicitly outside the large-model paper configuration.

## C. Test and command results

The final source passed **161 tests and 10 subtests**, with **5 skips** across
the full-suite invocation and one targeted rerun. The full-suite invocation
passed 160 tests; macOS revoked Desktop access before the K=2/R=2 worker could
start. After access was restored on 2026-09-16, that test passed in 61.17 seconds
without a code change. All four independent HF forward comparisons ran. This includes
real small-model RL/SFT, two-process shared-checkpoint handoffs, K=2/R=2,
padding/gradient equivalence and the actual Pallas kernel interpreted on CPU
in fp32 and bf16. fp32 derivatives use coordinate-wise comparisons; bf16
derivatives use a relative vector-error bound of two machine epsilons because
tiled softmax recomputation rounds differently from dense autodiff.
Four simulated CPU devices also passed both real pipeline variants
(ordinary and host-side gradient accumulation): **2 passed**, 31.82 seconds.

The tiny pipeline tests retain sampled token IDs and assert finite, changed
policy weights after both Explorer RL and Main RL, including host-side gradient
accumulation.

A fresh Python 3.11 virtualenv installed `requirements-dev.lock` with standard
pip and passed `pip check`. With an empty HF cache, model-hub access disabled,
and credential environment variables removed, the README's tiny pipeline command
passed **2 tests in 57.16 seconds**. No pretrained model, dataset download or
accelerator was needed. This checks installation and the offline first-run path;
it does not validate the TPU setup scripts. Seven additional launcher checks
passed with training replaced by the real CLI parser/contract validator.

The cleanup consolidated duplicate artifact upload code and removed the
non-paper hash fallback and obsolete build plan. Sequence bucketing reduces padded work without changing
token selection or the RL denominator. A local CPU microbenchmark (two-layer,
8-hidden-unit model; 97 active tokens; three warmups, ten timed forwards) measured
5.49 ms at width 1,024 versus 0.164 ms at width 128. This isolates padding overhead;
it is not a TPU or end-to-end training speedup. The benchmark receipt is included
in the review bundle.

Reproducible main environment: Python 3.11.15; JAX/jaxlib 0.7.1, Flax 0.12.0,
Optax 0.2.8, Orbax 0.12.4, NumPy 1.26.4, Transformers 5.17.0, pytest 9.1.1.
The exact CPU environment is captured in `requirements-dev.lock`.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.lock
.venv/bin/python -m pytest tests -q

# Optional independent HF forward references: point to a Python with PyTorch
# and Transformers supporting both model families. The audit used 5.17.0.
EXPDIS_HF_REFERENCE_PYTHON=/path/to/reference/python \
  .venv/bin/python -m pytest tests/test_expdis_jax_model_reference.py -q

JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \
  .venv/bin/python -m pytest tests/test_expdis_jax_pipeline_cpu.py -q
```

The audit's full-suite run supplied the independent HF Python, so all four
reference comparisons ran. Without it, those optional tests may skip in the
lightweight CPU environment. The five other skips concern optional Torch/pilot
dependencies. JAX buffer-donation and a test-only Orbax restore-sharding warning
are recorded; they did not fail numerical or checkpoint assertions.

GPU: **957 passed, 1 skipped** in 78.25 seconds with model-hub access disabled,
using the fresh environment recorded in that branch's `requirements-test.lock`.
The full old `requirements.txt` installation did not finish and is not certified.
See `GPU_AUDIT.md` for exact versions and commands.

CLI checks all exited 0: JAX train/pipeline/eval/parallel-pipeline help; pilot
help; GPU pipeline/RL/SFT/filter/eval help; GPU orchestration help, list-presets,
validate-all and a raw-TPU plan. Hardware setup, submit and training launchers
were not executed. Shell syntax checks cover the modified launch scripts.

## D. README and manuscript alignment

The READMEs now distinguish training stages from the `grpo_*` implementation
names, describe full-response/token-ID scoring, correct P1 counts, automatic
novelty layers, deferred predictor updates and shared checkpoints, and provide
an offline small-model test command. The pilot's default model-download/training
command is no longer presented as a tiny offline sanity check. GPU defaults are
identified as a separate campaign protocol. See `expdis_jax/PARALLEL.md` for the
concurrent deployment interface and its limits.

Both training CLIs now accept Explorer/Main role names. Legacy spellings,
checkpoint paths and serialized fields remain compatible. The separate appendix
harnesses for kNN/elliptical novelty, sampled-token entropy and DARLING semantic
classification are absent; the evaluation documentation states that boundary.

The first-run instructions now give an explicit credential-free CPU command and
its expected result. TPU instructions include the reload watcher, trainer/serving
coordinates, a 40,960-token serving window and an explicit writable HF repo.
The primary launcher accepts ordinary activated environments and forwards its
documented round counts; full remote serving remains an integration requirement.

Two manuscript issues require author judgment rather than silently changing
code to satisfy contradictory text:

1. Algorithm 1 admits groups with varying total rewards, while Appendix B also
   says homogeneous-correctness groups at λ=0 are never admitted directly.
   Soft-overlong rewards can vary in such a group. The implementation follows
   Algorithm 1's total-reward rule. Clarify that exception in the appendix.
2. The numeric verifier already parses some recursive numeric fraction forms
   beyond the appendix's stated simple integer fractions. This robustness was
   preserved; it is not symbolic algebra. Specify the parser's actual accepted
   grammar or identify the exact verifier revision used for reported results.

For Ministral training, use the publisher's
[BF16 instruct checkpoint](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-BF16).
The unsuffixed instruct checkpoint is FP8, which this training loader does not
support. This audit downloaded tokenizer/config metadata only; it did not train
those full model weights. Tokenizer probe revisions are retained in the review
bundle. Pin the exact revision used for any reported reproduction.

The new test/implementation fixes change executable behaviour. They cannot
retroactively demonstrate that historical result tables were generated by this
revision. Preserve that distinction in a public release description.

## E. Publication history and hygiene

The publication branches each contain one initial commit authored by Saif
Punjwani. The audited fixes are included in those initial snapshots; development
commits are not ancestors of either branch. Audit patches and validation receipts
are retained separately from Git history.

Static scans of source, tests and docs found no personal filesystem paths,
email addresses or common literal API-token patterns. Both branches retain
documented historical external artifact/job identifiers and GPU retains
tests of those identifiers. One deployment hostname was removed from the GPU
provenance document. Generic loopback/example addresses and public model/artifact
identifiers remain. Dependency files contain no private package endpoints.

## F. Verified mechanism checklist

“PASS” here means source plus offline executable evidence, not full accelerator
or benchmark reproduction.

| Mechanism | Status / implementing location |
|---|---|
| Mandatory snapshot sync at stage start and every iteration | PASS control flow (`train.py`); real serving transport unverified |
| One optimizer step; ratio with clip 0.2/0.28 | PASS `grpo.py`; at one step old log-probabilities are evaluated inline as `stop_gradient(new_lp)` under the same frozen weights; they are not vLLM-provided log-probabilities |
| Group-mean advantage, no std; denominator selected rows × 32,768; KL=0 | PASS `grpo.py`, `config.py` |
| B=4, G=16; eight attempts; total-reward spread; fallback counts denominator | PASS `config.py`, `train.py` |
| ±1 correctness; correct-only raw novelty; soft window 26,214/6,554 | PASS `train.py`, `rewarding.py`, config validation |
| Main reward has no novelty | PASS `pipeline.py` Main-stage config and actual pipeline assertions |
| RND: live completion features, quarter-depth layers, width512, three affine layers, RMS error +1e-8, Adam1e-4, predictor after policy | PASS `model.py`, `novelty.py`, `train.py`; fresh state each Explorer/round |
| All selected Explorer rows enter pool, including clipped rows | PASS `train.py` trajectory writer and integration pool counts |
| Correct/terminated/boxed-gold/repetition/128–4500 filter | PASS `filtering.py` |
| Reward-independent shortest winner/problem; deterministic ties/order; cap500 | PASS `filtering.py` |
| Two epochs, completion-only token-mean SFT | PASS `distill.py` and real optimization test |
| Fresh optimizer across stages; only Main weights carried | PASS checkpoint-handoff and multi-process pipeline tests |
| Shuffled disjoint shards; near-even 200/100 partition; per-round cap500 | PASS `lineage.py`, dataset tests, native and parallel driver plan tests |
| Concurrent Explorers | PASS process-launcher tests; real accelerator groups and scheduler enforcement unverified |
| AdamW .9/.95, eps1e-8, wd0, clip1, stage LRs 5e-6/1e-6/5e-6; fp32 params/bf16 compute | PASS configured source and small-model optimizer/forward tests |
| Checkpoint interval50 separate from sync interval1 | PASS defaults and checkpoint tests; durable remote storage integration unverified |
| P1 and same-pool avg/pass/diversity estimators | PASS `eval.py`, `run_eval`; actual datasets/results unverified |

## G. Remaining reproduction work

1. Run the corrected JAX stack on the intended TPU topology with real serving:
   validate per-iteration reload acknowledgments, sampled IDs, matching model
   revisions, shared storage, compiled Pallas forward/backward and 32k memory use.
2. Run a complete round, then MR-ME with independent serving groups. Inspect
   selected/candidate row counts, optimizer/RND counters, pool/filter receipts,
   and Main-only checkpoint lineage. Measure all stage and synchronization time.
3. Train the matched DAPO control and the requested ExpDis variants for the
   declared seeds. Freeze dataset/model/dependency revisions and evaluate all
   seven benchmarks from retained completion pools under one P1/verifier build.
4. Recompute the five-benchmark mean and pass@k from raw per-problem counts.
   CPU tests cannot promise the reported 51.42 mean or the claimed gains.

No fresh accelerator allocation or full benchmark run was performed in this
audit. Publication can accurately claim an audited implementation with the
validation above; a claim of reproduced paper results needs those run artifacts.
