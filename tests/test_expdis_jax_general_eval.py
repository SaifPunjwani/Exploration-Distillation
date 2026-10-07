"""CPU tests for the general-capability evaluation (no network, fake vLLM)."""

import json
import random
from types import SimpleNamespace

import pytest

from expdis_jax import general_eval as ge
from expdis_jax.generate import Completion


def done(text, finish="stop"):
    return Completion(text=text, token_ids=[], token_logprobs=[], finish_reason=finish)


# MMLU-Pro

def mmlu_row(qid, category, answer_index=1, options=None, cot="A: Let's think step by step. Because."):
    options = options or ["w", "x", "y", "z"]
    return {"question_id": qid, "question": f"Q{qid}?", "options": options, "answer": chr(65 + answer_index),
            "answer_index": answer_index, "category": category, "cot_content": cot}


def test_mmlu_pro_prompt_is_five_shot_category_matched():
    validation = [mmlu_row(100 + i, "law") for i in range(6)] + [mmlu_row(200, "math")]
    test = mmlu_row(1, "law", options=["a", "b", "N/A", "N/A"])
    prompt = ge.mmlu_pro_prompt(test, validation)
    assert prompt.startswith("The following are multiple choice questions (with answers) about law. ")
    assert prompt.count("Answer: Let's think step by step.") == 6
    assert "Q105?" not in prompt and "Q200?" not in prompt
    assert prompt.endswith("Question:\nQ1?\nOptions:\nA. a\nB. b\nAnswer: Let's think step by step.")
    record = ge.mmlu_pro_records([test], validation)[0]
    assert record["option_count"] == 2 and record["answer"] == "B"
    with pytest.raises(ValueError):
        ge.mmlu_pro_prompt(mmlu_row(2, "math"), validation)


@pytest.mark.parametrize("response,count,want", [
    ("so the answer is (C).", 4, "C"),
    ("The answer is B. Wait, final answer: D", 4, "D"),
    (r"\boxed{(A)}", 4, "A"),
    ("Reasoning\n(B)", 4, "B"),
    ("the answer is (E)", 4, None),
    ("no letter here", 10, None),
])
def test_mmlu_pro_answer(response, count, want):
    assert ge.mmlu_pro_answer(response, count) == want


# MMLU-Redux

@pytest.mark.parametrize("response,want", [
    ('{"answer":"A"}', "A"),
    ('thinking A; option B. {"answer":"C"}', "C"),
    ('{"note":"\\"answer\\":\\"A\\""} {"answer":"B"}', "B"),
    ('{"answer":"A"} {"answer":"D"}', "D"),
    ('{"answer":"A"} {"answer":"z"}', None),
    ('{"answer":"A"} {"answer":}', None),
    ('{"answer":"A"} {answer:"B"}', None),
    ('{"answer":"A"} {"note":"quoted answer: B"}', "A"),
    ('{"answer":"A","answer":"B"}', None),
    ('{"answer":"A","other":"B"}', "A"),
    ('{"answer":["A","B"]}', None),
    ('{"answer":"a"}', None),
    ("A", None),
])
def test_redux_json_answer(response, want):
    assert ge.redux_answer(response) == want


def test_redux_join_applies_gold_correction_and_prompt_hides_gold():
    publisher = {"anatomy": [
        {"question": "q0", "choices": ["a", "b", "c", "d"], "answer": 0, "error_type": "ok", "correct_answer": None},
        {"question": "q1", "choices": ["a", "b", "c", "d"], "answer": 0, "error_type": "wrong_groundtruth",
         "correct_answer": "C"},
    ]}
    curated = [{"id": "mmlu-redux-anatomy-#1", "config": "anatomy", "choices": ["a", "b", "c", "d"],
                "correct_answer": "c"}]
    test = ge.redux_join(curated, publisher)
    assert test == [{"id": "mmlu-redux-anatomy-#1", "question": "q1", "choices": ["a", "b", "c", "d"],
                     "answer": "C", "subject": "anatomy"}]
    dev = [{"question": f"d{i}", "choices": ["p", "q", "r", "s"], "answer": "B", "subject": "anatomy"}
           for i in range(6)]
    prompt = ge.redux_records(test, dev)[0]["prompt"]
    assert prompt.count("<example>") == 5 and "d5" not in prompt
    assert prompt.endswith("</target_question>\n\n" + ge.REDUX_JSON_INSTRUCTION)
    assert ge.redux_prompt(dict(test[0], answer="Z"), dev) == prompt
    bad = [dict(curated[0], correct_answer="a")]
    with pytest.raises(ValueError):
        ge.redux_join(bad, publisher)


# GPQA-Diamond

def gpqa_rows(count):
    return [{"Record ID": f"r{i}", "Question": f"Question {i}?", "Incorrect Answer 1": f"x{i}",
             "Incorrect Answer 2": f"y{i}", "Incorrect Answer 3": f"z{i}", "Correct Answer": f"gold{i}",
             "Explanation": "HIDDEN"} for i in range(count)]


def test_gpqa_records_use_seed_zero_shuffle_and_hide_explanation():
    rows = gpqa_rows(5)
    state = random.getstate()
    records = ge.gpqa_records(rows, expected=5)
    assert random.getstate() == state
    rng = random.Random(0)
    for row, record in zip(rows, records):
        choices = [row[f] for f in ge.GPQA_OPTION_FIELDS]
        rng.shuffle(choices)
        assert record["choices"] == choices
        assert record["choices"][ge.ABCD.index(record["answer"])] == row["Correct Answer"]
        assert "HIDDEN" not in record["prompt"] and record["prompt"].endswith(r"in \boxed{X}.")
    with pytest.raises(ValueError):
        ge.gpqa_records(rows, expected=6)


@pytest.mark.parametrize("response", [r"Reasoning.\n\boxed{A}", "Final answer: A", "The answer is (A).", "A",
                                      r"\boxed{b}" + "\n" + r"\boxed{A}", r"\boxed{\text{A}}",
                                      r"Final answer: \boxed{\textbf{a}}"])
def test_gpqa_accepts_explicit_final_letter(response):
    assert ge.grade_gpqa_diamond({"answer": "A"}, response) == {"correct": True, "invalid": False, "extracted": "A"}


@pytest.mark.parametrize("response", ["", "No conclusion.", "The answer is", "Final answer: E",
                                      r"\boxed{A} or \boxed{B}", "Final answer: A or B", r"\boxed{E}",
                                      "The answer is (A).\nFinal answer: I abstain.",
                                      r"Final answer: I abstain. \boxed{\text{A}}",
                                      r"\boxed{\textbf{\mathrm{A}}}", r"\textbf{\boxed{\text{A}}}"])
def test_gpqa_invalid_or_ambiguous_never_guesses(response):
    assert ge.grade_gpqa_diamond({"answer": "A"}, response)["invalid"] is True


def test_gpqa_bad_gold_raises():
    with pytest.raises(ValueError):
        ge.grade_gpqa_diamond({"answer": "E"}, "A")


# ZebraLogic

def zebra_row(i=0):
    return {"id": f"z{i}", "size": "2*2", "puzzle": "Two houses.",
            "solution": {"header": ["House", "Name", "Drink"], "rows": [["1", "Alice", "tea"], ["2", "Bob", "water"]]}}


def test_zebra_prompt_and_whole_puzzle_grading():
    record = ge.zebra_records([zebra_row()], "P:{puzzle}\n{json_template}", expected=1)[0]
    assert "Alice" not in record["prompt"] and '"House 2"' in record["prompt"]
    gold = record["gold"]
    good = json.dumps({"reasoning": "r", "solution": gold})
    assert ge.grade_zebralogic(record, good)["correct"] is True
    lists = json.dumps({"solution": {"House 1": {"Name": [" ALICE "], "Drink": "Tea"},
                                     "House 2": {"Name": "bob", "Drink": "water"}}})
    assert ge.grade_zebralogic(record, lists)["correct"] is True
    partial = json.dumps({"solution": {"House 1": {"Name": "Alice", "Drink": "tea"}, "House 2": {"Name": "Bob"}}})
    result = ge.grade_zebralogic(record, partial)
    assert result["correct"] is False and result["invalid"] is False and result["cell_accuracy"] == 0.75
    assert ge.grade_zebralogic(record, "no json")["invalid"] is True
    assert ge.grade_zebralogic(record, json.dumps({"solution": {"House 1": {"Name": 3}}}))["invalid"] is True
    with pytest.raises(ValueError):
        ge.zebra_records([zebra_row(), zebra_row()], "{puzzle}", expected=2)


# IFEval

class FakeIFEvalLib:
    class InputExample:
        def __init__(self, key, instruction_id_list, prompt, kwargs):
            self.key, self.instruction_id_list, self.prompt, self.kwargs = key, instruction_id_list, prompt, kwargs

    @staticmethod
    def test_instruction_following_strict(example, responses):
        ok = ["," not in responses[example.prompt]]
        return SimpleNamespace(follow_all_instructions=all(ok), follow_instruction_list=ok)

    @staticmethod
    def test_instruction_following_loose(example, responses):
        return SimpleNamespace(follow_all_instructions=True, follow_instruction_list=[True])


def test_ifeval_strict_primary_and_empty_invalid():
    record = ge.ifeval_records([{"key": 7, "prompt": "No commas.", "instruction_id_list": ["punctuation:no_comma"],
                                 "kwargs": [{}]}])[0]
    assert record["id"] == "ifeval-7"
    assert ge.grade_ifeval(record, "fine", scorer=FakeIFEvalLib)["correct"] is True
    failed = ge.grade_ifeval(record, "a, b", scorer=FakeIFEvalLib)
    assert failed["correct"] is False and failed["invalid"] is False and failed["loose_prompt_correct"] is True
    assert ge.grade_ifeval(record, "  ", scorer=FakeIFEvalLib)["invalid"] is True
    with pytest.raises(ValueError):
        ge.ifeval_records([{"key": 1, "prompt": "p", "instruction_id_list": ["a"], "kwargs": []}])


# Scoring, generation, and change from Base

def test_final_response_thinking():
    assert ge.final_response("<think>x</think>ans", True) == "ans"
    assert ge.final_response("<think>unfinished", True) is None
    assert ge.final_response("<think>a</think>b<think>c", True, reject_reopened=True) is None
    assert ge.final_response("<think>raw", False) == "<think>raw"


def test_score_completions_counts_invalid_and_pass_at_k():
    records = [{"id": "p", "answer": "A", "subject": "s"}, {"id": "q", "answer": "B", "subject": "s"}]
    completions = [[done('<think>t</think>{"answer":"A"}'), done("<think>never closed")],
                   [done('</think>{"answer":"C"}'), done("", finish="error")]]
    result = ge.score_completions(records, "mmlu_redux", completions, num_samples=2, thinking=True)
    assert result["accuracy"] == 0.25 and result["accuracy_percent"] == 25.0
    assert result["problem_counts"] == {"p": 1, "q": 0}
    assert result["invalid_answers"] == 2 and result["error_total"] == 1
    assert result["pass_at_k"] == {"1": 0.25, "2": 0.5}
    with pytest.raises(RuntimeError):
        ge.score_completions(records, "mmlu_redux", completions[:1], num_samples=2, thinking=True)


class FakeTokenizer:
    chat_template = "{% if enable_thinking %}{% endif %}"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize and add_generation_prompt and len(messages) == 1
        return {"input_ids": [1, len(messages[0]["content"]), int(enable_thinking)]}


def test_run_general_eval_uses_vllm_client_with_paper_sampling(monkeypatch):
    calls = {}

    def fake_generate(prompts, server_urls, **kwargs):
        calls.update(kwargs, prompts=prompts, server_urls=server_urls)
        return [[done(r"<think>x</think>\boxed{A}"), done(r"<think>x</think>\boxed{B}")] for _ in prompts]

    monkeypatch.setattr(ge, "batched_generate", fake_generate)
    records = ge.gpqa_records(gpqa_rows(3), expected=3)
    tokenizer = FakeTokenizer()
    assert ge.thinking_enabled(tokenizer, "auto") is True
    result = ge.run_general_eval(records, "gpqa_diamond", tokenizer, ["http://fake/v1"], num_samples=2,
                                 thinking=True)
    assert calls["prompts"][0][2] == 1
    assert (calls["temperature"], calls["top_p"], calls["top_k"], calls["max_tokens"]) == (0.6, 0.95, 20, 32768)
    assert calls["n_per_prompt"] == 2 and calls["fanout_per_prompt"] == 2 and calls["seed_base"] == 1234
    expected = sum(r["answer"] == "A" for r in records) + sum(r["answer"] == "B" for r in records)
    assert result["num_correct"] == expected and result["invalid_answers"] == 0


def write_result(tmp_path, name, benchmark, accuracy, model):
    path = tmp_path / f"{name}_{benchmark}.json"
    protocol = {"benchmark": benchmark, "model_name": model, "num_samples": 64, "max_tokens": 32768,
                "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                "prompt_version": ge.PROMPT_VERSION[benchmark], "grader_version": ge.GRADER_VERSION[benchmark],
                "dataset": ge.DATASET[benchmark], "max_problems": 0}
    path.write_text(json.dumps({"protocol": protocol, "accuracy_percent": accuracy, "error_total": 0}))
    return str(path)


def test_summarize_change_from_base(tmp_path, capsys):
    base = {"mmlu_pro": 50.0, "mmlu_redux": 60.0, "ifeval": 70.0, "gpqa_diamond": 30.0, "zebralogic": 10.0}
    method = {"mmlu_pro": 51.0, "mmlu_redux": 59.0, "ifeval": 72.0, "gpqa_diamond": 30.0, "zebralogic": 14.0}
    base_paths = [write_result(tmp_path, "base", b, v, "base") for b, v in base.items()]
    method_paths = [write_result(tmp_path, "method", b, v, "method") for b, v in method.items()]
    out_path = tmp_path / "change.json"
    ge.main(["--summarize", "--base-results", *base_paths, "--results", *method_paths,
             "--output-path", str(out_path)])
    out = json.loads(out_path.read_text())
    assert out["benchmarks"]["mmlu_redux"]["change_percent"] == -1.0
    assert out["benchmarks"]["zebralogic"]["change_percent"] == 4.0
    assert out["prior_capabilities_change_percent"] == pytest.approx((1 - 1 + 2 + 0) / 4)
    with pytest.raises(ValueError):
        ge.summarize_change(base_paths[1:], method_paths[1:])
    mismatched = write_result(tmp_path, "other", "mmlu_pro", 51.0, "method")
    data = json.loads(open(mismatched).read())
    data["protocol"]["temperature"] = 1.0
    open(mismatched, "w").write(json.dumps(data))
    with pytest.raises(ValueError):
        ge.summarize_change(base_paths, [mismatched, *method_paths[1:]])
