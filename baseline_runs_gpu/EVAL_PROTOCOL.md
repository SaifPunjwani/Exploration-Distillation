# Evaluation Protocol — GPU Reproduction

Three protocols matter. Every GPU result must state which one it used.

## P1 — Project protocol (matches measured_base_aggregates; primary)

Source: the project's TPU-era evaluation notes and the W&B runs
`baseline_eval_Base_*` (2026-05-04..06).

- Sampling: temperature 0.6, top_p 0.95, top_k 20, min_p 0, thinking mode auto
  (Qwen3 chat template default = thinking on), seeds fixed per sample.
- Max completion tokens: **32,768** (chosen over Qwen's 38,912 because on TPU
  vLLM the larger budget required YaRN rope-factor 2.0, which produced
  degenerate loops; 32k is the model card's default recommendation and keeps
  the AIME24 clip rate low).
- Samples per problem: AIME24 64×30, AIME25 64×30 (I+II), MATH500 32×500,
  AMC23 32×40, Minerva-Math 64×272 for every model family, GSM8K 8×1319.
- Prompt: system "Please reason step by step, and put your final answer
  within \boxed{}." + problem as user turn; chat template rendered
  client-side; served via /v1/completions.
- Metrics: avg@k = mean accuracy over first k samples; pass@k = unbiased
  1−C(n−c,k)/C(n,k); maj@32 = majority vote over first 32 extracted answers;
  valid_answer_rate; clipped_rate (finish_reason==length);
  distinct_answer_mean. Verifier: last \boxed{} (brace-matched) → final-answer
  cues → normalization (numeric LaTeX frac → a/b, bare fractions, decimals,
  and scientific notation) → absolute numeric tolerance 1e-6 using bounded
  exact-rational arithmetic (never binary-float equality)
  (`tmx_jax/rewarding.py`; TPU version was "baseline_common_v1", same family).
- Symbolic golds outside that bounded numeric grammar use normalized exact-string
  comparison; they are **not** claimed as algebraic-equivalence coverage. The
  pinned audit in `GOLD_SYNTAX_AUDIT.json` records 124/500 MATH500 and 82/272
  Minerva-Math golds in exact-string-only categories (radicals, symbolic
  fractions, tuples/functions, units/text, and algebraic expressions).
- Headline aggregation: hard mean over AIME24/AIME25/MATH500/Minerva-Math;
  AMC23 + GSM8K appendix-only.

## P2 — Qwen3 report / model-card protocol (secondary anchor)

Source: Qwen3-1.7B model card (verified 2026-07-07) + Qwen3 Technical Report
(arXiv:2505.09388).

- Same sampling (T 0.6 / top_p 0.95 / top_k 20 / min_p 0, thinking mode).
- Max output length **38,912** for competition benchmarks (= 40,960 context
  − 2,048 prompt budget; no YaRN needed on GPU vLLM when prompt ≤ 2,048).
- Same math prompt ("Please reason step by step, and put your final answer
  within \boxed{}.").
- Qwen3-1.7B reported (thinking): AIME24 avg@64 = 48.3 ("report anchor";
  49.1 appears in the model-card-family tables the project cites), pass@4
  71.1, pass@32 80.0.
- GPU repro runs this as a secondary suite on AIME24/AIME25 to show the
  budget effect explicitly.
- The measured P2-vs-P1 comparison (accuracy and clip rates at both output
  budgets) lives in the results document, which is kept outside this
  repository; the protocol here is what those numbers were produced under.

## P3 — Historical avg@32 protocol (DO NOT USE — documented for provenance)

Pre-May TPU era: T=1.0, top_p=1.0, top_k=0, 16,384 tokens, avg@32.
Not comparable to P1/P2; some old logs/tables reference it.

## Rules for the reproduction campaign

1. Every trained-model eval uses P1 (so deltas vs the measured base anchors
   are apples-to-apples), with P2 run additionally for headline checkpoints.
2. Never mix anchors: the strict base anchor is the P1 AIME24 avg@64 measured
   by this harness on the pinned base model; an anchor that mixes in an
   earlier run must be labeled as such wherever it is used. Base-anchor
   validation gate for the GPU harness: reproduce the project's TPU base
   anchor within the run-to-run spread of the project's own repeated base
   runs.
3. Record in every aggregate JSON: sampling params, budget, num samples,
   verifier version, prompt template hash, model hash, vLLM version
   (GPU = vllm 0.24.0), and protocol id P1/P2.
4. GSM8K gold answers must be parsed after '####'; nested \boxed{} needs
   brace matching (both are historical footguns).
