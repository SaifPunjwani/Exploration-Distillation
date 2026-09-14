"""DARLING-style diversity metrics for W&B logging.

Implements:
  - Inter-generation distinct-n (dist-3, dist-4) from facebook/darling
  - LM judge for math strategy diversity via DeepInfra Qwen3.5-9B
  - Truncated_ variants (only non-clipped generations)
"""

import os
import itertools
import random
import time
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# dist-n: inter-generation distinct n-grams
# ---------------------------------------------------------------------------

def _get_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def inter_generation_distinct_n(
    generations: List[str],
    n: int = 4,
) -> float:
    """Average fraction of n-grams unique to each generation vs all others.

    For each generation, computes:
        |ngrams unique to this gen| / |total ngrams in this gen|
    Returns the mean over all generations.

    Matches the DARLING ``calculate_inter_generation_diversity`` definition.
    """
    if not generations or len(generations) <= 1:
        return 1.0 if generations else 0.0

    all_ngram_lists = []
    all_ngram_sets = []
    for gen in generations:
        tokens = gen.split()
        ngrams = _get_ngrams(tokens, n) if len(tokens) >= n else []
        all_ngram_lists.append(ngrams)
        all_ngram_sets.append(set(ngrams))

    scores = []
    for i in range(len(generations)):
        total = len(all_ngram_lists[i])
        if total == 0:
            scores.append(0.0)
            continue
        others = set()
        for j, s in enumerate(all_ngram_sets):
            if j != i:
                others.update(s)
        unique = all_ngram_sets[i] - others
        scores.append(len(unique) / total)

    return sum(scores) / len(scores)


def compute_distinct_n_for_groups(
    groups: List[dict],
    ns: Tuple[int, ...] = (3, 4),
    truncated_only: bool = False,
) -> Dict[str, float]:
    """Compute dist-n metrics over a list of rollout groups.

    Args:
        groups: List of rollout group dicts, each with ``rows`` containing
            ``completion_text``, ``terminated``, ``clipped`` fields.
        ns: Which n-gram sizes to compute.
        truncated_only: If True, only include non-clipped (terminated) rows.

    Returns:
        Dict mapping e.g. ``"dist_3"`` -> mean across groups,
        ``"dist_4"`` -> mean, ``"num_groups"`` -> count used.
    """
    results: Dict[str, List[float]] = {f"dist_{n}": [] for n in ns}

    for group in groups:
        rows = group.get("rows", [])
        if truncated_only:
            rows = [r for r in rows if r.get("terminated", False) and not r.get("clipped", False)]
        texts = [r.get("completion_text", "") for r in rows if r.get("completion_text")]
        if len(texts) < 2:
            continue
        for n in ns:
            score = inter_generation_distinct_n(texts, n=n)
            results[f"dist_{n}"].append(score)

    out: Dict[str, float] = {}
    for key, vals in results.items():
        out[key] = float(sum(vals) / max(1, len(vals))) if vals else 0.0
    out["num_groups"] = float(len(results.get(f"dist_{ns[0]}", [])))
    return out


# ---------------------------------------------------------------------------
# LM judge: cognitive strategy diversity via DeepInfra
# ---------------------------------------------------------------------------

_COGNITIVE_JUDGE_PROMPT = """You are given a math problem (Original Prompt) and two model-generated solutions (Generation 0 and Generation 1). Classify each response according to the presence of the following cognitive behaviors:

Verification: Explicitly checks intermediate or final results against the conditions or criteria of the problem (e.g., "Checking if x = 2 satisfies the equation...").

Subgoal Setting: Clearly identifies intermediate steps or goals explicitly set to guide toward the final solution (e.g., "First, I'll simplify the equation to isolate x...").

Backtracking: Explicitly reverses or discards previous steps or attempts to explore a new solution path after realizing a previous approach was incorrect or insufficient (e.g., "This approach leads nowhere, let's try another...").

Backward Chaining: Starts reasoning from the desired goal or final result and logically works backward towards initial conditions (e.g., "To obtain the final value of x, let's see what conditions must hold first...").

Provide your classification separately for each generation using the tags:

Verification: <verification>yes/no</verification>

Subgoal Setting: <subgoal>yes/no</subgoal>

Backtracking: <backtracking>yes/no</backtracking>

Backward Chaining: <backward_chaining>yes/no</backward_chaining>

Original Prompt:
\"\"\"{prompt}\"\"\"

Generation 0:
\"\"\"{gen0}\"\"\"

Generation 1:
\"\"\"{gen1}\"\"\"

After classifying each generation, briefly evaluate if Generation 0 and Generation 1 exhibit exactly the same cognitive behaviors. Respond with "[[Yes]]" if they exhibit exactly the same behaviors, or "[[No]]" if they do not."""


def _judge_pair_sync(
    prompt: str,
    gen0: str,
    gen1: str,
    *,
    api_key: str,
    base_url: str = "https://api.deepinfra.com/v1/openai",
    model: str = "Qwen/Qwen3.5-9B",
    max_retries: int = 2,
    timeout: float = 30.0,
) -> Optional[bool]:
    """Call LM judge to determine if two generations use different strategies.

    Returns True if strategies differ, False if same, None on failure.
    """
    try:
        from openai import OpenAI
    except ImportError:
        return None

    user_message = _COGNITIVE_JUDGE_PROMPT.format(
        prompt=prompt.strip(),
        gen0=gen0.strip()[:4000],
        gen1=gen1.strip()[:4000],
    )

    client = OpenAI(api_key=api_key, base_url=base_url)
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": user_message}],
                max_tokens=1024,
                temperature=0.3,
            )
            content = resp.choices[0].message.content.strip().lower()
            tag = content.split("[[")[-1].split("]]")[0].strip()
            if tag == "yes":
                return False  # same cognitive behaviors
            elif tag == "no":
                return True  # different strategies
        except Exception as e:
            if attempt < max_retries:
                time.sleep(1.5 ** attempt)
            else:
                print(f"[diversity] LM judge failed after {max_retries} attempts: {e}")
    return None


def compute_lm_judge_diversity(
    groups: List[dict],
    *,
    max_pairs_per_group: int = 3,
    max_groups: int = 8,
    truncated_only: bool = False,
    api_key: Optional[str] = None,
    base_url: str = "https://api.deepinfra.com/v1/openai",
    model: str = "Qwen/Qwen3.5-9B",
) -> Dict[str, float]:
    """Compute LM judge diversity over sampled pairs from rollout groups.

    Returns dict with ``judge_diversity_rate`` (fraction of pairs judged as
    using different strategies) and ``judge_pair_count``.
    """
    if api_key is None:
        api_key = os.environ.get("TMX_DEEPINFRA_API_KEY", "").strip()
    if not api_key:
        return {"judge_diversity_rate": 0.0, "judge_pair_count": 0.0}

    sampled_groups = groups[:max_groups] if len(groups) > max_groups else groups
    if len(groups) > max_groups:
        sampled_groups = random.sample(groups, max_groups)

    different_count = 0
    total_count = 0

    for group in sampled_groups:
        prompt = group.get("prompt", "")
        rows = group.get("rows", [])
        if truncated_only:
            rows = [r for r in rows if r.get("terminated", False) and not r.get("clipped", False)]
        texts = [r.get("completion_text", "") for r in rows if r.get("completion_text")]
        if len(texts) < 2:
            continue

        pairs = list(itertools.combinations(range(len(texts)), 2))
        if len(pairs) > max_pairs_per_group:
            pairs = random.sample(pairs, max_pairs_per_group)

        for i, j in pairs:
            result = _judge_pair_sync(
                prompt, texts[i], texts[j],
                api_key=api_key, base_url=base_url, model=model,
            )
            if result is not None:
                total_count += 1
                if result:
                    different_count += 1

    return {
        "judge_diversity_rate": float(different_count / max(1, total_count)),
        "judge_pair_count": float(total_count),
    }


# ---------------------------------------------------------------------------
# Combined entry point for W&B logging
# ---------------------------------------------------------------------------

def compute_all_diversity_metrics(
    groups: List[dict],
    step: int,
    *,
    log_interval: int = 10,
    api_key: Optional[str] = None,
) -> Optional[Dict[str, float]]:
    """Compute all diversity metrics if step is a multiple of log_interval.

    Returns a flat dict ready for W&B logging, or None if not a logging step.
    """
    if step % log_interval != 0:
        return None

    metrics: Dict[str, float] = {}

    # dist-3/4 on all generations
    dist_all = compute_distinct_n_for_groups(groups, ns=(3, 4), truncated_only=False)
    metrics["diversity/dist_3"] = dist_all["dist_3"]
    metrics["diversity/dist_4"] = dist_all["dist_4"]
    metrics["diversity/num_groups"] = dist_all["num_groups"]

    # truncated_ variants (non-clipped only)
    dist_trunc = compute_distinct_n_for_groups(groups, ns=(3, 4), truncated_only=True)
    metrics["diversity/truncated_dist_3"] = dist_trunc["dist_3"]
    metrics["diversity/truncated_dist_4"] = dist_trunc["dist_4"]
    metrics["diversity/truncated_num_groups"] = dist_trunc["num_groups"]

    # LM judge (only if API key available)
    if api_key or os.environ.get("TMX_DEEPINFRA_API_KEY", "").strip():
        judge_all = compute_lm_judge_diversity(
            groups, truncated_only=False, api_key=api_key,
        )
        metrics["diversity/judge_diversity_rate"] = judge_all["judge_diversity_rate"]
        metrics["diversity/judge_pair_count"] = judge_all["judge_pair_count"]

        judge_trunc = compute_lm_judge_diversity(
            groups, truncated_only=True, api_key=api_key,
        )
        metrics["diversity/truncated_judge_diversity_rate"] = judge_trunc["judge_diversity_rate"]
        metrics["diversity/truncated_judge_pair_count"] = judge_trunc["judge_pair_count"]

    return metrics
