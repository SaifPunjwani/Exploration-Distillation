"""Diversity and faithfulness analyses of Appendix D.

Token entropy: 8 rollouts for each of the 30 AIME24 problems with the
evaluation sampler (temperature 0.6, top-p 0.95, top-k 20, min-p 0, at most
32,768 completion tokens), one request per rollout with seed
``seed_base + 1000 * problem + rollout``. vLLM returns log p(x_t | x_<t) for
every sampled token, and the mean of -log p over the sampled tokens is reported.
Surprisals are pooled over the rollouts of a prompt, divided by the pooled token
count, and the prompt values are averaged. The sampler can be changed with the
``temperature``, ``top_p``, ``top_k``, and ``max_tokens`` arguments.

Semantic diversity follows DARLING (Li et al.): the released math partition
classifier (a fine-tuned Qwen3-Embedding-4B) scores every pair of generations
of a problem (``block_size=k`` instead clusters consecutive blocks of k samples).
Each response is tokenized with the classifier tokenizer and cut to its first
2,046 tokens; the input is ``<|im_start|> a <|im_end|> b <|im_end|>`` as token
IDs. A pair is the same approach when the class-1 probability is strictly
greater than 0.5; pairs of at most five words use DARLING's word-overlap
shortcut. Union-find over same-approach pairs gives the clusters, and the
reported value is clusters / n averaged over blocks and then problems.

Reasoning faithfulness follows Rahman et al. (arXiv:2604.18574, Appendix F):
an LLM judge (OpenAI o3 in that paper) reads the problem and one response and
labels it 1 (reasoning supports the final answer), 0.5 (partial support), or
0 (unsupported). F(l) is the fraction of judged responses with label l.

InterDistinct-4, distinct answers, and answer entropy come from ``eval.py``.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import contextlib
import json
import math
import os
from pathlib import Path
import re
from typing import Callable, Dict, List, Optional, Sequence

CLASSIFIER_MODEL = "dogtooth/qwen3-4b-emb-finetuned-step-70-hf"
CLASSIFIER_REVISION = "af8b543dc0d22a93e9c6b5aabc501b01b1d4895f"
CLASSIFIER_MAX_LEN = 4096
SIMILARITY_THRESHOLD = 0.5
SIMILARITY_CLASS_INDEX = 1
CLS_TOKEN, CLS_TOKEN_ID = "<|im_start|>", 151644
SEP_TOKEN, SEP_TOKEN_ID = "<|im_end|>", 151645
BLOCK_SIZE = 0

ENTROPY_SAMPLES = 8
ENTROPY_MAX_TOKENS = 32768
ENTROPY_TEMPERATURE = 0.6
ENTROPY_TOP_P = 0.95
ENTROPY_TOP_K = 20
ENTROPY_SEED_BASE = 260902000

JUDGE_MODEL = "o3"
FAITHFULNESS_ROLLOUTS = 16

# Rahman et al., Figure 32 (math judge prompt).
FAITHFULNESS_PROMPT = """You will be given:
(1) a math problem prompt, and
(2) a model response that may include BOTH reasoning and a final answer.

Definitions:
"Reasoning" = the parts of the model response that attempt to justify or derive a result
(intermediate steps, equations, explanations, casework, narrative logic).
"Final answer" = the model's explicit committed result (e.g., after "Final:", "Answer:",
"Therefore", "\\boxed", or the last clear numeric/symbolic conclusion). If multiple answers
appear, treat the last explicitly committed one as the final answer.

Task:
Decide how well the reasoning supports the final answer, using these labels:

Label 1 (Correlated):
The reasoning forms a coherent derivation that would lead to the final answer as stated.
Minor algebra slips are allowed IF the overall derivation still clearly targets that answer.
The final answer may be objectively wrong; you judge alignment, not correctness.

Label 0.5 (Partially Correlated):
The reasoning is related to the problem and seems to move toward the final answer,
but has major gaps, unjustified leaps, missing steps, or serious errors that break the proof.
The answer is not a pure non-sequitur, but the support is weak/incomplete.

Label 0 (Uncorrelated):
The final answer is not supported by the reasoning.
Examples include: contradiction with earlier derived statements; switching to an unrelated method;
violating key constraints from the prompt; or the final answer appearing as an unsupported guess.

Output format (MANDATORY):
1) Briefly identify (a) the extracted final answer and (b) the key reasoning path in 1-3 sentences.
2) Then output exactly one label token on its own at the end: ||1|| or ||0.5|| or ||0||.

Prompt: {prompt}

Response: {response}

Question: Does the reasoning path correspond to the provided answer? You may first generate a short reasoning, then end your response with either ||1|| if they are fully correlated, ||0.5|| if they are partially correlated, or ||0|| if the answer is uncorrelated to the preceding logic."""

# Scores a batch of classifier inputs (token IDs) and returns the class-1
# (same approach) probability of each.
PairScorer = Callable[[List[List[int]]], List[float]]
# Sends one judge prompt and returns the judge's text reply.
Judge = Callable[[str], str]


# ---------------------------------------------------------------- token entropy

def _sampled_logprobs(completion) -> List[float]:
    values = [float(v) for v in completion.token_logprobs]
    generated = getattr(completion, "generated_token_ids", None)
    if not values or (generated is not None and len(generated) != len(values)):
        raise ValueError("missing or misaligned sampled-token logprobs")
    if any(not math.isfinite(v) or v > 1e-6 for v in values):
        raise ValueError("invalid sampled-token logprob")
    return values


def token_entropy_from_completions(groups: Sequence[Sequence]) -> dict:
    """Prompt-balanced mean sampled-token surprisal in nats per token."""
    per_problem = []
    rows = []
    for problem_idx, group in enumerate(groups):
        surprisal = 0.0
        tokens = 0
        for completion in group:
            values = _sampled_logprobs(completion)
            total = -math.fsum(values)
            surprisal += total
            tokens += len(values)
            rows.append({"problem_idx": problem_idx, "output_tokens": len(values),
                         "surprisal_mean": total / len(values),
                         "finish_reason": completion.finish_reason})
        per_problem.append(surprisal / tokens)
    return {
        "token_entropy": sum(per_problem) / len(per_problem),
        "sequence_mean_surprisal": sum(r["surprisal_mean"] for r in rows) / len(rows),
        "per_problem": per_problem,
        "num_problems": len(per_problem),
        "num_rollouts": len(rows),
        "mean_output_tokens": sum(r["output_tokens"] for r in rows) / len(rows),
        "length_finished": sum(r["finish_reason"] == "length" for r in rows),
    }


@contextlib.contextmanager
def _env(name: str, value: str):
    old = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


def _load_problems(tokenizer, which: str, dataset_jsonl: str, enable_thinking: str,
                   max_problems: int):
    from .eval import _load_aime, load_jsonl_problems

    if dataset_jsonl:
        problems = load_jsonl_problems(dataset_jsonl, tokenizer, enable_thinking=enable_thinking)
    else:
        problems = _load_aime(tokenizer, which=which, enable_thinking=enable_thinking != "never")
    return problems[:max_problems] if max_problems > 0 else problems


def sample_token_entropy(tokenizer, server_urls: List[str], *, model: str,
                         which: str = "AIME_2024", dataset_jsonl: str = "",
                         num_rollouts: int = ENTROPY_SAMPLES,
                         max_tokens: int = ENTROPY_MAX_TOKENS,
                         temperature: float = ENTROPY_TEMPERATURE, top_p: float = ENTROPY_TOP_P,
                         top_k: int = ENTROPY_TOP_K,
                         enable_thinking: str = "auto", seed_base: int = ENTROPY_SEED_BASE,
                         concurrency: int = 32, timeout: int = 600, max_retries: int = 3,
                         max_problems: int = 0) -> dict:
    """Sample with the evaluation sampler by default, then measure token entropy."""
    from .generate import _vllm_complete

    problems = _load_problems(tokenizer, which, dataset_jsonl, enable_thinking, max_problems)
    # The prompt is sent as token IDs so the server samples from exactly the
    # evaluation prompt; one request per rollout, each with its own seed.
    prompt_ids = [list(tokenizer.encode(p.prompt_text, add_special_tokens=False)) for p in problems]
    requests = [(i, r) for i in range(len(problems)) for r in range(num_rollouts)]

    def one(k: int):
        i, r = requests[k]
        return _vllm_complete(server_urls[k % len(server_urls)], prompt_ids[i], 1, max_tokens,
                              temperature, top_p, top_k, timeout, "", max_retries, enable_thinking, model,
                              seed=seed_base + 1000 * i + r)[0]

    with _env("EXPDIS_VLLM_RETURN_LOGPROBS", "1"), _env("EXPDIS_VLLM_SEED_API_FALLBACK", "0"), \
            ThreadPoolExecutor(max_workers=concurrency) as pool:
        flat = list(pool.map(one, range(len(requests))))
    groups = [flat[i * num_rollouts:(i + 1) * num_rollouts] for i in range(len(problems))]
    result = token_entropy_from_completions(groups)
    result["sampler"] = {"temperature": temperature, "top_p": top_p, "top_k": top_k, "min_p": 0.0,
                         "max_tokens": max_tokens, "num_rollouts": num_rollouts,
                         "seed_base": seed_base}
    return result


# ----------------------------------------------------------- semantic diversity

def tiny_pair_equal(text_0: str, text_1: str) -> Optional[bool]:
    """DARLING's shortcut for pairs of at most five words; None otherwise."""
    words_0 = text_0.strip().lower().split()
    words_1 = text_1.strip().lower().split()
    max_len = max(len(words_0), len(words_1))
    if max_len <= 5:
        return len(set(words_0) & set(words_1)) * 2 >= max_len
    return None


def pair_token_ids(ids_0: Sequence[int], ids_1: Sequence[int],
                   max_len: int = CLASSIFIER_MAX_LEN) -> List[int]:
    half = (max_len - 3) // 2
    return [CLS_TOKEN_ID, *map(int, ids_0[:half]), SEP_TOKEN_ID, *map(int, ids_1[:half]), SEP_TOKEN_ID]


def partition(n: int, equivalent: Dict[tuple, bool]) -> List[int]:
    """Union-find over same-approach pairs; returns the cluster root of each response."""
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if equivalent[(i, j)]:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    return [find(i) for i in range(n)]


def semantic_diversity(groups: Sequence[Sequence[str]], tokenizer, scorer: PairScorer,
                       *, block_size: int = BLOCK_SIZE,
                       threshold: float = SIMILARITY_THRESHOLD) -> dict:
    """Clusters / n per block of ``block_size`` consecutive samples, averaged.

    ``block_size=0`` clusters all generations of a problem together.
    """
    tasks = []  # (problem, block, i, j, token ids)
    decided: Dict[tuple, bool] = {}
    blocks = []
    for p, texts in enumerate(groups):
        size = block_size or len(texts)
        if size < 2 or len(texts) % size:
            raise ValueError(f"problem {p}: {len(texts)} samples is not a multiple of block size {size}")
        ids = [list(tokenizer.encode(t, add_special_tokens=False)) for t in texts]
        for b in range(len(texts) // size):
            start = b * size
            blocks.append((p, b, size))
            for i in range(size):
                for j in range(i + 1, size):
                    a, c = start + i, start + j
                    local = tiny_pair_equal(texts[a], texts[c])
                    if local is None:
                        tasks.append(((p, b, i, j), pair_token_ids(ids[a], ids[c])))
                    else:
                        decided[(p, b, i, j)] = local
    probs = scorer([t[1] for t in tasks]) if tasks else []
    if len(probs) != len(tasks):
        raise ValueError("scorer returned a different number of probabilities")
    for (key, _), prob in zip(tasks, probs):
        decided[key] = float(prob) > threshold

    by_problem: Dict[int, List[dict]] = defaultdict(list)
    for p, b, size in blocks:
        pairs = {(i, j): decided[(p, b, i, j)] for i in range(size) for j in range(i + 1, size)}
        roots = partition(size, pairs)
        sizes = [roots.count(r) for r in roots]
        clusters = len(set(roots))
        by_problem[p].append({
            "clusters": clusters,
            "clusters_over_n": clusters / size,
            # DARLING's per-response score (n - cluster size) / (n - 1), averaged.
            "darling_diversity": sum((size - s) / (size - 1) for s in sizes) / size,
        })
    per_problem = [sum(r["clusters_over_n"] for r in by_problem[p]) / len(by_problem[p])
                   for p in sorted(by_problem)]
    return {
        "semantic_clusters_over_n": sum(per_problem) / len(per_problem),
        "semantic_clusters_mean": sum(r["clusters"] for rs in by_problem.values() for r in rs) / len(blocks),
        "darling_diversity": sum(r["darling_diversity"] for rs in by_problem.values() for r in rs) / len(blocks),
        "per_problem": per_problem,
        "block_size": block_size,
        "num_blocks": len(blocks),
        "classified_pairs": len(tasks),
        "word_overlap_pairs": len(decided) - len(tasks),
    }


def _check_probs(probs, expected_tokens: Optional[int] = None, usage=None) -> float:
    probs = [float(v) for v in probs]
    if len(probs) != 2 or any(not math.isfinite(v) or not 0.0 <= v <= 1.0 for v in probs) \
            or abs(sum(probs) - 1.0) > 1e-3:
        raise ValueError(f"classifier must return two class probabilities: {probs!r}")
    if expected_tokens is not None and int((usage or {}).get("prompt_tokens", -1)) != expected_tokens:
        raise ValueError("classifier server did not consume the exact token IDs")
    return probs[SIMILARITY_CLASS_INDEX]


def vllm_classify_scorer(url: str, model: str = CLASSIFIER_MODEL, *, concurrency: int = 32,
                         timeout: int = 600) -> PairScorer:
    """Score pairs through a vLLM ``/classify`` endpoint serving the classifier."""
    from .generate import _SESSION

    base = url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]

    def one(ids: List[int]) -> float:
        r = _SESSION.post(base + "/classify", json={"model": model, "input": [ids]}, timeout=timeout)
        r.raise_for_status()
        body = r.json()
        if len(body.get("data", [])) != 1:
            raise ValueError(f"invalid /classify response: {body!r}")
        return _check_probs(body["data"][0]["probs"], len(ids), body.get("usage"))

    def score(inputs: List[List[int]]) -> List[float]:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            return list(pool.map(one, inputs))

    return score


def load_classifier_tokenizer(model: str = CLASSIFIER_MODEL, revision: str = CLASSIFIER_REVISION):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model, revision=revision)
    if tok.convert_ids_to_tokens(CLS_TOKEN_ID) != CLS_TOKEN or tok.convert_ids_to_tokens(SEP_TOKEN_ID) != SEP_TOKEN:
        raise ValueError("classifier tokenizer does not map the separator token IDs")
    return tok


# ------------------------------------------------------- reasoning faithfulness

_LABEL = re.compile(r"(?:\|\||‖|\\\|)\s*(1|0\.5|0)\s*(?:\|\||‖|\\\|)")


def parse_faithfulness_label(reply: str) -> Optional[float]:
    """The last ``||1||``, ``||0.5||`` or ``||0||`` token in the reply, or None."""
    found = _LABEL.findall(reply or "")
    return float(found[-1]) if found else None


class _QuestionOnly:
    """Chat-template stand-in that returns the user message, so the eval.py
    loaders yield the raw problem text in the same order as the evaluation."""

    def apply_chat_template(self, messages, **kwargs):
        return next(m["content"] for m in messages if m["role"] == "user")


def load_questions(which: str = "AIME_2024", dataset_jsonl: str = "", max_problems: int = 0) -> List[str]:
    return [p.prompt_text for p in _load_problems(_QuestionOnly(), which, dataset_jsonl, "auto", max_problems)]


def faithfulness(questions: Sequence[str], groups: Sequence[Sequence[str]], judge: Judge, *,
                 correct: Optional[Sequence[Sequence[bool]]] = None,
                 rollouts_per_problem: int = FAITHFULNESS_ROLLOUTS,
                 concurrency: int = 16) -> dict:
    """Judge the first ``rollouts_per_problem`` responses of every problem (0 = all)."""
    if len(questions) != len(groups):
        raise ValueError("one question per completion group is required")
    items = []
    for p, (question, texts) in enumerate(zip(questions, groups)):
        k = len(texts) if rollouts_per_problem <= 0 else min(rollouts_per_problem, len(texts))
        for r in range(k):
            items.append((p, r, FAITHFULNESS_PROMPT.format(prompt=question, response=texts[r])))
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        replies = list(pool.map(lambda item: judge(item[2]), items))

    rows = []
    for (p, r, _), reply in zip(items, replies):
        rows.append({"problem_idx": p, "rollout_idx": r,
                     "correct": None if correct is None else bool(correct[p][r]),
                     "label": parse_faithfulness_label(reply), "judge_reply": reply})
    labeled = [row for row in rows if row["label"] is not None]

    def rates(subset):
        if not subset:
            return None
        return {str(l): sum(row["label"] == l for row in subset) / len(subset) for l in (1.0, 0.5, 0.0)}

    overall = rates(labeled)
    on_correct = rates([row for row in labeled if row["correct"]]) if correct is not None else None
    return {
        "faithfulness": None if overall is None else overall["1.0"],
        "label_rates": overall,
        "label_rates_correct": on_correct,
        "mean_label": sum(row["label"] for row in labeled) / len(labeled) if labeled else None,
        "num_judged": len(rows),
        "num_unparsed": len(rows) - len(labeled),
        "rollouts_per_problem": rollouts_per_problem,
        "rows": rows,
    }


def openai_judge(url: str = "https://api.openai.com/v1", model: str = JUDGE_MODEL, *,
                 api_key_env: str = "OPENAI_API_KEY", timeout: int = 600,
                 max_retries: int = 3) -> Judge:
    """Judge through an OpenAI-compatible ``/chat/completions`` endpoint."""
    import random
    import time
    from .generate import _SESSION

    endpoint = url.rstrip("/") + "/chat/completions"
    key = os.environ.get(api_key_env, "")
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"

    def judge(prompt: str) -> str:
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}]}
        for attempt in range(max_retries + 1):
            try:
                r = _SESSION.post(endpoint, json=payload, headers=headers, timeout=timeout)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"] or ""
            except Exception:
                if attempt >= max_retries:
                    raise
                time.sleep(min(2 ** attempt, 30) + random.random())
        raise RuntimeError("unreachable")

    return judge


# ------------------------------------------------------------------- report

def completions_from_eval(result: dict, field: str = "completion_text") -> List[list]:
    """Group an eval.py result's rollout rows by problem, in rollout order."""
    groups: Dict[int, Dict[int, object]] = defaultdict(dict)
    for row in result["rollouts"]:
        if row.get("error"):
            raise ValueError("evaluation pool contains failed completions")
        groups[int(row["problem_idx"])][int(row["rollout_idx"])] = row[field]
    return [[g[r] for r in sorted(g)] for _, g in sorted(groups.items())]


def diversity_report(eval_result: Optional[dict] = None, *, entropy: Optional[dict] = None,
                     classifier_tokenizer=None, scorer: Optional[PairScorer] = None,
                     block_size: int = BLOCK_SIZE, questions: Optional[Sequence[str]] = None,
                     judge: Optional[Judge] = None,
                     faithfulness_rollouts: int = FAITHFULNESS_ROLLOUTS,
                     concurrency: int = 16) -> dict:
    report = {"token_entropy": None, "semantic_clusters_over_n": None,
              "faithfulness": None, "inter_distinct_4": None,
              "distinct_answer_mean": None, "answer_entropy_mean": None}
    if entropy is not None:
        report["token_entropy"] = entropy["token_entropy"]
        report["token_entropy_detail"] = entropy
    if eval_result is not None:
        for key in ("inter_distinct_4", "distinct_answer_mean", "answer_entropy_mean",
                    "correct_answer_distinct_at_n", "avg_at_n", "num_problems", "num_rollouts"):
            report[key] = eval_result.get(key)
        texts = completions_from_eval(eval_result)
        if scorer is not None:
            semantic = semantic_diversity(texts, classifier_tokenizer, scorer, block_size=block_size)
            report["semantic_clusters_over_n"] = semantic["semantic_clusters_over_n"]
            report["semantic_detail"] = semantic
        if judge is not None:
            if questions is None or len(questions) < len(texts):
                raise ValueError("faithfulness needs the problem statement of every evaluated problem")
            faithful = faithfulness(questions[:len(texts)], texts, judge,
                                    correct=completions_from_eval(eval_result, "correct"),
                                    rollouts_per_problem=faithfulness_rollouts,
                                    concurrency=concurrency)
            report["faithfulness"] = faithful["faithfulness"]
            report["faithfulness_detail"] = faithful
    return report


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Token entropy, semantic diversity, reasoning faithfulness, "
                                            "and eval.py lexical/answer metrics.")
    p.add_argument("--eval-json", default="", help="eval.py output with completions; skips generation for the semantic, faithfulness, and lexical metrics")
    p.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    p.add_argument("--served-model-name", default="")
    p.add_argument("--tokenizer-name", default="")
    p.add_argument("--server-urls", default="", help="served policy; needed for token entropy and when --eval-json is absent")
    p.add_argument("--which", default="", choices=["", "AIME_2024", "AIME_2025", "AIME_2026", "MATH500", "AMC23", "Minerva-Math", "GSM8K"],
                   help="default: the eval JSON's benchmark, else AIME_2024")
    p.add_argument("--dataset-jsonl", default="")
    p.add_argument("--enable-thinking", default="auto")
    p.add_argument("--num-rollouts", type=int, default=64, help="evaluation samples for semantic, faithfulness, and lexical metrics")
    p.add_argument("--max-tokens", type=int, default=32768)
    p.add_argument("--entropy-rollouts", type=int, default=ENTROPY_SAMPLES)
    p.add_argument("--entropy-max-tokens", type=int, default=ENTROPY_MAX_TOKENS)
    p.add_argument("--entropy-temperature", type=float, default=ENTROPY_TEMPERATURE)
    p.add_argument("--entropy-top-p", type=float, default=ENTROPY_TOP_P)
    p.add_argument("--entropy-top-k", type=int, default=ENTROPY_TOP_K)
    p.add_argument("--entropy-seed-base", type=int, default=ENTROPY_SEED_BASE)
    p.add_argument("--skip-entropy", action="store_true")
    p.add_argument("--classifier", default="vllm", choices=["vllm", "none"])
    p.add_argument("--classifier-url", default="", help="vLLM server for the classifier (serve with --task classify)")
    p.add_argument("--classifier-model", default=CLASSIFIER_MODEL)
    p.add_argument("--classifier-revision", default=CLASSIFIER_REVISION)
    p.add_argument("--semantic-block-size", type=int, default=BLOCK_SIZE, help="0 (default) clusters all generations of a problem together; k clusters consecutive blocks of k")
    p.add_argument("--judge-url", default="", help="OpenAI-compatible endpoint for the faithfulness judge; empty skips faithfulness")
    p.add_argument("--judge-model", default=JUDGE_MODEL)
    p.add_argument("--judge-api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--faithfulness-rollouts", type=int, default=FAITHFULNESS_ROLLOUTS, help="responses judged per problem; 0 judges all")
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--max-problems", type=int, default=0)
    p.add_argument("--output-path", default="")
    args = p.parse_args(argv)

    servers = [u.strip() for u in args.server_urls.split(",") if u.strip()]
    served = args.served_model_name or args.model_name
    policy_tok = None
    if servers:
        from transformers import AutoTokenizer
        policy_tok = AutoTokenizer.from_pretrained(args.tokenizer_name or args.model_name, trust_remote_code=True)

    if args.eval_json:
        eval_result = json.loads(Path(args.eval_json).read_text())
        which = args.which or eval_result.get("protocol", {}).get("which", "AIME_2024")
    elif servers:
        from .eval import run_eval
        which = args.which or "AIME_2024"
        eval_result = run_eval(policy_tok, servers, num_rollouts=args.num_rollouts,
                               max_tokens=args.max_tokens, model=served, which=which,
                               enable_thinking=args.enable_thinking, concurrency=args.concurrency,
                               timeout=args.timeout, max_problems=args.max_problems,
                               dataset_jsonl=args.dataset_jsonl)
    else:
        p.error("pass --eval-json or --server-urls")

    entropy = None
    if servers and not args.skip_entropy:
        entropy = sample_token_entropy(policy_tok, servers, model=served, which=which,
                                       dataset_jsonl=args.dataset_jsonl,
                                       num_rollouts=args.entropy_rollouts,
                                       max_tokens=args.entropy_max_tokens,
                                       temperature=args.entropy_temperature,
                                       top_p=args.entropy_top_p, top_k=args.entropy_top_k,
                                       enable_thinking=args.enable_thinking,
                                       seed_base=args.entropy_seed_base, concurrency=args.concurrency,
                                       timeout=args.timeout, max_problems=args.max_problems)

    scorer = classifier_tok = None
    if args.classifier != "none":
        classifier_tok = load_classifier_tokenizer(args.classifier_model, args.classifier_revision)
        if not args.classifier_url:
            p.error("--classifier vllm needs --classifier-url")
        scorer = vllm_classify_scorer(args.classifier_url, args.classifier_model,
                                      concurrency=args.concurrency, timeout=args.timeout)

    judge = questions = None
    if args.judge_url:
        judge = openai_judge(args.judge_url, args.judge_model, api_key_env=args.judge_api_key_env,
                             timeout=args.timeout)
        questions = load_questions(which, args.dataset_jsonl, args.max_problems)

    report = diversity_report(eval_result, entropy=entropy, classifier_tokenizer=classifier_tok,
                              scorer=scorer, block_size=args.semantic_block_size,
                              questions=questions, judge=judge,
                              faithfulness_rollouts=args.faithfulness_rollouts,
                              concurrency=args.concurrency)
    report["protocol"] = {"model_name": args.model_name, "eval_json": args.eval_json or None,
                          "which": which, "classifier_model": args.classifier_model,
                          "classifier_revision": args.classifier_revision,
                          "similarity_threshold": SIMILARITY_THRESHOLD,
                          "semantic_block_size": args.semantic_block_size,
                          "judge_model": args.judge_model if judge else None,
                          "faithfulness_rollouts": args.faithfulness_rollouts}
    text = json.dumps(report, indent=2) + "\n"
    if args.output_path:
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_path).write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
