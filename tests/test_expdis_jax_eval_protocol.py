"""P1 metric tests use fixed local pools; no model, datasets, or network."""
import json
import pytest
from expdis_jax import eval as evaluation
from expdis_jax.generate import Completion
from expdis_jax.prompting import EVAL_SYSTEM_PROMPT, PROMPT_PROTOCOL, TRAIN_SYSTEM_PROMPT


@pytest.mark.parametrize("which", evaluation.PRIMARY_BENCHMARKS)
@pytest.mark.parametrize("model", [
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "mistralai/Ministral-3-3B-Instruct-2512-BF16",
])
def test_september_24_primary_benchmarks_default_to_64_samples(which, model):
    assert evaluation.paper_sample_count(which, model) == 64


def test_paper_sample_counts():
    assert evaluation.paper_sample_count("AIME_2024", "Qwen/Qwen3-1.7B") == 64
    assert evaluation.paper_sample_count("AIME_2026", "Qwen/Qwen3-4B") == 64
    assert evaluation.paper_sample_count("Minerva-Math", "Qwen/Qwen3-4B") == 64
    assert evaluation.paper_sample_count("Minerva-Math", "Ministral-3-3B") == 64
    assert evaluation.paper_sample_count("AMC23", "Qwen/Qwen3-4B") == 32
    assert evaluation.paper_sample_count("GSM8K", "Qwen/Qwen3-1.7B") == 8


def test_training_and_evaluation_use_their_system_prompts(tmp_path):
    from expdis_jax.data import render_chat_prompt
    from expdis_jax.prompting import math_messages
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is True
            return json.dumps(messages)
    tokenizer = Tokenizer()
    training = [{"role": "system", "content": TRAIN_SYSTEM_PROMPT}, {"role": "user", "content": "2+2?"}]
    evaluating = [{"role": "system", "content": EVAL_SYSTEM_PROMPT}, {"role": "user", "content": "2+2?"}]
    path = tmp_path / "questions.jsonl"
    path.write_text(json.dumps({"problem": "2+2?", "answer": "4"}) + "\n")
    assert json.loads(render_chat_prompt("2+2?", tokenizer)) == training
    assert math_messages("2+2?") == training
    problems = evaluation.load_jsonl_problems(str(path), tokenizer)
    assert json.loads(problems[0].prompt_text) == evaluating


def test_verifier_and_pass64_use_same_pool(monkeypatch):
    monkeypatch.setattr(evaluation, "_load_aime", lambda *a, **kw: [
        evaluation.AimeProblem("problem one", "0.5"), evaluation.AimeProblem("problem two", "2")])
    # The first problem has one correct rational-equivalent answer; the other
    # is unsolved. pass@1=avg@64, pass@64=one of two problems solved.
    pool = [[Completion(r"\boxed{\frac{1}{2}}", [], [], "stop")] +
            [Completion(r"\boxed{3}", [], [], "stop") for _ in range(63)],
            [Completion(r"\boxed{3}", [], [], "stop") for _ in range(64)]]
    monkeypatch.setattr(evaluation, "batched_generate", lambda *a, **kw: pool)
    result = evaluation.run_eval(None, ["unused"], num_rollouts=64)
    assert result["avg_at_n"] == 1 / 128
    assert result["pass_at_1"] == result["avg_at_n"]
    assert result["pass_at_64"] == result["any_correct_fraction"] == 0.5
    assert result["correct_answer_distinct_at_n"] == 0.5
    values = [result[f"pass_at_{k}"] for k in (1, 2, 4, 8, 16, 32, 64)]
    assert values == sorted(values)
    assert "delta_vs_base" not in str(evaluation.aime24_wandb_metrics(result))


@pytest.mark.parametrize("pool", [[], [[Completion("", [], [], "error")]]])
def test_missing_or_failed_samples_invalidate_evaluation(monkeypatch, pool):
    monkeypatch.setattr(evaluation, "_load_aime", lambda *a, **kw: [evaluation.AimeProblem("p", "1")])
    monkeypatch.setattr(evaluation, "batched_generate", lambda *a, **kw: pool)
    with pytest.raises(RuntimeError):
        evaluation.run_eval(None, ["unused"], num_rollouts=1)


def test_empty_evaluation_fails_clearly(monkeypatch):
    monkeypatch.setattr(evaluation, "_load_aime", lambda *a, **kw: [])
    with pytest.raises(ValueError, match="empty"):
        evaluation.run_eval(None, ["unused"])


@pytest.mark.parametrize("answer", [r"50\%", r"\[50\]", "100/2"])
def test_answer_normalization_matches_paper_formatting(answer):
    assert evaluation._normalize_text_answer(answer) == "50"
def test_minerva_local_checkpoint_uses_uniform_sample_count(tmp_path):
    import json
    from expdis_jax.eval import paper_sample_count
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "qwen3", "hidden_size": 2560, "num_hidden_layers": 36}))
    assert paper_sample_count("Minerva-Math", str(tmp_path)) == 64


def test_minerva_unknown_model_uses_uniform_sample_count():
    from expdis_jax.eval import paper_sample_count
    assert paper_sample_count("Minerva-Math", "unknown-model-alias") == 64


def test_training_probe_writes_result_without_historical_baseline(tmp_path, monkeypatch):
    from expdis_jax.config import TrainConfig
    from expdis_jax.train import _maybe_run_aime_probe
    monkeypatch.setenv("EXPDIS_AIME_PROBE_EVERY_STEPS", "1")
    monkeypatch.setenv("EXPDIS_AIME_PROBE_REQUIRED", "1")
    monkeypatch.setenv("EXPDIS_HF_MIRROR_EVAL_PROBES", "0")
    monkeypatch.delenv("EXPDIS_GCS_CHECKPOINT_BASE", raising=False)
    monkeypatch.setattr(evaluation, "run_eval", lambda *a, **kw: {"avg_at_n": .5})
    cfg = TrainConfig(output_dir=str(tmp_path), model_name="mistralai/Ministral-3-3B-Instruct-2512-BF16")
    _maybe_run_aime_probe(cfg=cfg, tokenizer=None, server_urls=["unused"],
                         step=1, stage="explorer", use_wandb=False)
    result = json.loads((tmp_path / "eval_probes/aime24_explorer_step_000001.json").read_text())
    assert result["protocol"]["model_name"] == cfg.model_name
    assert result["protocol"]["prompt_protocol"] == PROMPT_PROTOCOL
    assert result["protocol"]["system_prompt"] == EVAL_SYSTEM_PROMPT
    assert "baseline" not in str(result)
    assert result["delta_vs_initial_avg_at_n"] == 0


def test_wandb_labels_use_the_actual_sample_count_and_benchmark():
    result = {"num_rollouts": 64, "avg_at_n": .5}
    metrics = evaluation.aime24_wandb_metrics(result)
    assert metrics["aime24_avg_at_64"] == .5
    assert "aime24_avg_at_32" not in metrics
    result["protocol"] = {"which": "Minerva-Math"}
    metrics = evaluation.aime24_wandb_metrics(result, prefix="eval/minerva_math")
    assert all("aime24" not in key for key in metrics)


def test_five_benchmark_mean_excludes_additional_benchmarks(tmp_path):
    paths = []
    for i, which in enumerate((*evaluation.PRIMARY_BENCHMARKS, "GSM8K")):
        path = tmp_path / f"{i}.json"
        path.write_text(json.dumps({"protocol": {"which": which, "model_name": "checkpoint"},
                                    "avg_at_n": i / 10, "error_total": 0}))
        paths.append(path)
    summary = evaluation.summarize_results(paths)
    assert summary["mean_accuracy_percent"] == 20
    assert summary["benchmark_accuracy_percent"]["GSM8K"] == 50
    with pytest.raises(ValueError, match="missing primary"):
        evaluation.summarize_results(paths[1:])
    with pytest.raises(ValueError, match="duplicate"):
        evaluation.summarize_results(paths + paths[:1])
    changed = json.loads(paths[-1].read_text())
    changed["protocol"]["model_name"] = "different-checkpoint"
    paths[-1].write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="share one model"):
        evaluation.summarize_results(paths)


def test_answer_entropy_uses_normalized_answers(monkeypatch):
    import math
    monkeypatch.setattr(evaluation, "_load_aime", lambda *a, **kw: [evaluation.AimeProblem("p", "1/2")])
    pool = [[Completion(text, [], [], "stop") for text in (
        r"\boxed{1/2}", r"\boxed{\frac{1}{2}}", r"\boxed{3}", r"\boxed{3}")]]
    monkeypatch.setattr(evaluation, "batched_generate", lambda *a, **kw: pool)
    result = evaluation.run_eval(None, ["unused"], num_rollouts=4)
    assert result["answer_entropy_mean"] == pytest.approx(math.log(2))
    assert result["distinct_answer_mean"] == 2


def test_evaluation_can_use_the_training_prompt(tmp_path, monkeypatch):
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return json.dumps(messages)
    path = tmp_path / "questions.jsonl"
    path.write_text(json.dumps({"problem": "2+2?", "answer": "4"}) + "\n")
    problems = evaluation.load_jsonl_problems(str(path), Tokenizer(), prompt="train")
    assert json.loads(problems[0].prompt_text)[0]["content"] == TRAIN_SYSTEM_PROMPT
    monkeypatch.setenv("EXPDIS_EVAL_PROMPT", "train")
    problems = evaluation.load_jsonl_problems(str(path), Tokenizer())
    assert json.loads(problems[0].prompt_text)[0]["content"] == TRAIN_SYSTEM_PROMPT
    monkeypatch.setenv("EXPDIS_EVAL_PROMPT", "eval")
    problems = evaluation.load_jsonl_problems(str(path), Tokenizer())
    assert json.loads(problems[0].prompt_text)[0]["content"] == EVAL_SYSTEM_PROMPT
    with pytest.raises(ValueError):
        evaluation.eval_system_prompt("other")
