"""Diversity and faithfulness metrics with fake servers, classifier, and judge."""
import json
import math

import pytest

from expdis_jax import diversity
from expdis_jax import eval as evaluation
from expdis_jax import generate
from expdis_jax.generate import Completion
from expdis_jax.prompting import EVAL_SYSTEM_PROMPT


class WordTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return [len(w) for w in text.split()]


def text(strategy, i):
    return f"{strategy} approach number {i} with enough words to classify"


def first_word_scorer(calls):
    # Same approach when the first token of both responses (word length) matches.
    def score(inputs):
        calls.extend(inputs)
        out = []
        for ids in inputs:
            sep = ids.index(diversity.SEP_TOKEN_ID)
            out.append(0.9 if ids[1] == ids[sep + 1] else 0.1)
        return out
    return score


def test_token_entropy_is_prompt_balanced_mean_surprisal():
    groups = [
        [Completion("a", [], [-1.0, -1.0], "stop", [1, 2]), Completion("b", [], [-2.0], "stop", [3])],
        [Completion("c", [], [-0.5, -0.5, -0.5, -0.5], "length", [1, 2, 3, 4])],
    ]
    result = diversity.token_entropy_from_completions(groups)
    assert result["per_problem"] == pytest.approx([4.0 / 3.0, 0.5])
    assert result["token_entropy"] == pytest.approx((4.0 / 3.0 + 0.5) / 2)
    assert result["sequence_mean_surprisal"] == pytest.approx((1.0 + 2.0 + 0.5) / 3)
    assert result["num_rollouts"] == 3 and result["length_finished"] == 1


def test_token_entropy_rejects_missing_or_misaligned_logprobs():
    with pytest.raises(ValueError):
        diversity.token_entropy_from_completions([[Completion("a", [], [], "stop", [1])]])
    with pytest.raises(ValueError):
        diversity.token_entropy_from_completions([[Completion("a", [], [-1.0], "stop", [1, 2])]])


def test_entropy_sampler_uses_eval_prompt_seed_lattice_and_eval_sampler(monkeypatch):
    rendered = []

    class ChatTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            rendered.append(messages)
            return json.dumps(messages)

        def encode(self, prompt, add_special_tokens=False):
            return [len(prompt), 7]

    monkeypatch.setattr(evaluation, "load_dataset", lambda *a, **kw: [
        {"Problem": "p0", "Answer": "1"}, {"Problem": "p1", "Answer": "2"}])
    calls = []

    def fake_complete(url, prompt, n, max_tokens, temperature, top_p, top_k, timeout, api_key,
                      max_retries, enable_thinking, model, seed=None):
        import os
        assert os.environ["EXPDIS_VLLM_RETURN_LOGPROBS"] == "1"
        calls.append((prompt, n, max_tokens, temperature, top_p, top_k, seed))
        return [Completion("x", [], [-0.25, -0.75], "stop", [5, 6])]

    monkeypatch.setattr(generate, "_vllm_complete", fake_complete)
    result = diversity.sample_token_entropy(ChatTokenizer(), ["http://a", "http://b"], model="m",
                                            num_rollouts=3, seed_base=100, concurrency=2)
    assert all(m[0] == {"role": "system", "content": EVAL_SYSTEM_PROMPT} for m in rendered)
    assert sorted(c[6] for c in calls) == [100, 101, 102, 1100, 1101, 1102]
    assert {c[1:6] for c in calls} == {(1, 32768, 0.6, 0.95, 20)}
    assert all(isinstance(c[0], list) for c in calls)
    assert result["token_entropy"] == pytest.approx(0.5)
    assert result["num_rollouts"] == 6


def test_pair_inputs_truncate_each_response_to_half_the_budget():
    ids = diversity.pair_token_ids(list(range(5000)), list(range(10)))
    assert ids[0] == diversity.CLS_TOKEN_ID and ids[-1] == diversity.SEP_TOKEN_ID
    assert ids[2047] == diversity.SEP_TOKEN_ID and len(ids) == 2046 + 10 + 3


def test_semantic_clusters_over_n_in_blocks_of_eight():
    # Block 0: two strategies; block 1: four strategies.
    strategies = ["aa", "bbb"] * 4 + ["aa", "bbb", "cccc", "ddddd"] * 2
    texts = [text(s, i) for i, s in enumerate(strategies)]
    calls = []
    result = diversity.semantic_diversity([texts], WordTokenizer(), first_word_scorer(calls), block_size=8)
    assert result["num_blocks"] == 2 and result["classified_pairs"] == 2 * 28 == len(calls)
    assert result["semantic_clusters_over_n"] == pytest.approx((2 / 8 + 4 / 8) / 2)
    # Block 0: every response shares a cluster of 4 -> (8-4)/7; block 1: clusters of 2 -> 6/7.
    assert result["darling_diversity"] == pytest.approx((4 / 7 + 6 / 7) / 2)


def test_semantic_whole_pool_and_strict_threshold():
    texts = [text(s, i) for i, s in enumerate(["aa", "bbb", "aa", "bbb"])]
    result = diversity.semantic_diversity([texts], WordTokenizer(), first_word_scorer([]))
    assert result["semantic_clusters_over_n"] == pytest.approx(2 / 4)
    at_threshold = diversity.semantic_diversity([texts], WordTokenizer(), lambda xs: [0.5] * len(xs),
                                                block_size=0)
    assert at_threshold["semantic_clusters_over_n"] == 1.0
    with pytest.raises(ValueError):
        diversity.semantic_diversity([texts[:3]], WordTokenizer(), first_word_scorer([]), block_size=2)


def test_short_pairs_use_word_overlap_without_the_classifier():
    texts = ["42", "42", "the answer is 7", "answer 7", "x", "y", "z", "w"]
    calls = []
    result = diversity.semantic_diversity([texts], WordTokenizer(), first_word_scorer(calls))
    assert calls == [] and result["word_overlap_pairs"] == 28
    # {42,42}, {the answer is 7, answer 7}, x, y, z, w.
    assert result["semantic_clusters_over_n"] == pytest.approx(6 / 8)


def test_vllm_classifier_requires_exact_token_transport(monkeypatch):
    class Response:
        def __init__(self, body):
            self.body = body

        def raise_for_status(self):
            pass

        def json(self):
            return self.body

    seen = []

    def post(url, json, timeout):
        seen.append((url, json))
        tokens = len(json["input"][0])
        return Response({"data": [{"probs": [0.3, 0.7]}], "usage": {"prompt_tokens": tokens}})

    monkeypatch.setattr(generate._SESSION, "post", post)
    scorer = diversity.vllm_classify_scorer("http://c/v1", concurrency=1)
    assert scorer([[1, 2, 3]]) == [0.7]
    assert seen[0][0] == "http://c/classify" and seen[0][1]["input"] == [[1, 2, 3]]
    monkeypatch.setattr(generate._SESSION, "post", lambda url, json, timeout: Response(
        {"data": [{"probs": [0.3, 0.7]}], "usage": {"prompt_tokens": 99}}))
    with pytest.raises(ValueError):
        scorer([[1, 2, 3]])


@pytest.mark.parametrize("reply,label", [
    ("answer 5, derivation sound. ||1||", 1.0),
    ("first ||0|| then revised ||0.5||", 0.5),
    ("label: ‖0‖", 0.0),
    (r"\|1\|", 1.0),
    ("no label here", None),
])
def test_parse_faithfulness_label(reply, label):
    assert diversity.parse_faithfulness_label(reply) == label


def test_faithfulness_rates_and_prompt():
    prompts = []

    def judge(prompt):
        prompts.append(prompt)
        if "trace-a" in prompt:
            return "fine ||1||"
        if "trace-b" in prompt:
            return "gaps ||0.5||"
        return "no label"

    groups = [["trace-a", "trace-b", "trace-a"], ["trace-a", "trace-c", "trace-b"]]
    correct = [[True, False, True], [False, True, True]]
    result = diversity.faithfulness(["Q0", "Q1"], groups, judge, correct=correct,
                                    rollouts_per_problem=2, concurrency=1)
    assert result["num_judged"] == 4 and result["num_unparsed"] == 1
    assert result["faithfulness"] == pytest.approx(2 / 3)
    assert result["label_rates"] == pytest.approx({"1.0": 2 / 3, "0.5": 1 / 3, "0.0": 0.0})
    assert result["label_rates_correct"] == pytest.approx({"1.0": 1.0, "0.5": 0.0, "0.0": 0.0})
    assert result["mean_label"] == pytest.approx(2.5 / 3)
    assert any("Prompt: Q1\n\nResponse: trace-c" in p for p in prompts)
    assert all(p.rstrip().endswith("uncorrelated to the preceding logic.") for p in prompts)


def test_questions_come_from_the_eval_loader_without_chat_template(monkeypatch):
    monkeypatch.setattr(evaluation, "load_dataset", lambda *a, **kw: [
        {"Problem": "Find x.", "Answer": "3"}, {"Problem": "Find y.", "Answer": "4"}])
    assert diversity.load_questions("AIME_2024") == ["Find x.", "Find y."]


def eval_result():
    rollouts = []
    for p in range(2):
        for r in range(8):
            rollouts.append({"problem_idx": p, "rollout_idx": r, "correct": r % 2 == 0,
                             "error": False, "completion_text": text("aa" if r < 4 else "bbb", r)})
    return {"inter_distinct_4": 0.4, "distinct_answer_mean": 3.0, "answer_entropy_mean": 1.1,
            "correct_answer_distinct_at_n": 1.0, "avg_at_n": 0.5, "num_problems": 2,
            "num_rollouts": 8, "rollouts": rollouts, "protocol": {"which": "AIME_2024"}}


def test_report_from_eval_json(tmp_path, monkeypatch):
    path = tmp_path / "eval.json"
    path.write_text(json.dumps(eval_result()))
    out = tmp_path / "diversity.json"
    diversity.main(["--eval-json", str(path), "--classifier", "none", "--output-path", str(out)])
    report = json.loads(out.read_text())
    assert report["inter_distinct_4"] == 0.4 and report["distinct_answer_mean"] == 3.0
    assert report["token_entropy"] is None and report["semantic_clusters_over_n"] is None
    assert report["faithfulness"] is None and report["protocol"]["which"] == "AIME_2024"

    monkeypatch.setattr(diversity, "openai_judge", lambda *a, **kw: (lambda prompt: "ok ||1||"))
    monkeypatch.setattr(diversity, "load_questions", lambda *a, **kw: ["Q0", "Q1"])
    diversity.main(["--eval-json", str(path), "--classifier", "none", "--judge-url", "http://judge/v1",
                    "--output-path", str(out)])
    report = json.loads(out.read_text())
    assert report["faithfulness"] == 1.0
    assert report["faithfulness_detail"]["num_judged"] == 16  # min(16, 8) per problem


def test_report_semantic_from_eval_result():
    report = diversity.diversity_report(eval_result(), classifier_tokenizer=WordTokenizer(),
                                        scorer=first_word_scorer([]))
    assert report["semantic_clusters_over_n"] == pytest.approx(2 / 8)
    assert math.isclose(report["semantic_detail"]["darling_diversity"], 4 / 7)
