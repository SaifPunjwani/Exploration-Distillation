"""Shared fixtures for the tmx_gpu CPU test suite (no GPU, no vLLM).

The e2e tests drive the real pipeline with a tiny random causal LM and a
stub OpenAI-completions server (stub_server.py). The tiny model and tokenizer
are initialized locally; no model download or accelerator is required.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# 6 training problems; the stub server reads the answer from the [gt=..] marker.
TRAIN_PROBLEMS = [(4, "2 + 2"), (7, "3 + 4"), (12, "5 + 7"),
                  (9, "4 + 5"), (23, "11 + 12"), (5, "1 + 4")]
PROBE_PROBLEMS = [(15, "7 + 8"), (30, "13 + 17")]


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory) -> str:
    """Build a small real Llama and tokenizer entirely offline."""
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    target = tmp_path_factory.mktemp("tiny-model")
    words = ["<pad>", "<unk>", "<eos>", "<bos>", "Step", "considers", "quantity",
             "and", "refines", "the", "estimate.", "The", "answer", "is", "Compute",
             "You", "are", "a", "helpful", "mathematician.", "Solve", "problem",
             "step", "by", "step.", "+"]
    backend = Tokenizer(models.WordLevel(dict(zip(words, range(len(words)))), unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>",
        unk_token="<unk>", eos_token="<eos>", bos_token="<bos>")
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\n' }}"
        "{% endfor %}{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}")
    cfg = LlamaConfig(vocab_size=len(words), hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=40960, pad_token_id=0, eos_token_id=2, bos_token_id=3)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        model = LlamaForCausalLM(cfg)
    model.save_pretrained(target)
    tokenizer.save_pretrained(target)
    return str(target)


@pytest.fixture(scope="session")
def novelty_layers(tiny_model_dir) -> str:
    """Deepest valid RND layer for the tiny model (layer <= n_blocks - 2)."""
    from transformers import AutoConfig

    n = int(AutoConfig.from_pretrained(tiny_model_dir).num_hidden_layers)
    return str(max(0, n - 2))


def _write_problems(path: Path, problems) -> str:
    with open(path, "w") as f:
        for i, (answer, expr) in enumerate(problems):
            f.write(json.dumps({
                "problem_id": f"p{i}",
                "problem": f"Compute {expr}. [gt={answer}]",
                "answer": str(answer),
            }) + "\n")
    return str(path)


@pytest.fixture(scope="session")
def train_jsonl(tmp_path_factory) -> str:
    return _write_problems(tmp_path_factory.mktemp("data") / "train.jsonl", TRAIN_PROBLEMS)


@pytest.fixture(scope="session")
def probe_jsonl(tmp_path_factory) -> str:
    return _write_problems(tmp_path_factory.mktemp("data") / "probe.jsonl", PROBE_PROBLEMS)


@pytest.fixture()
def stub_pool():
    """Running stub server; yields (endpoint URL, StubState)."""
    from tmx_gpu.tests.stub_server import start_stub_server

    server, port, state = start_stub_server()
    yield f"http://127.0.0.1:{port}", state
    server.shutdown()
