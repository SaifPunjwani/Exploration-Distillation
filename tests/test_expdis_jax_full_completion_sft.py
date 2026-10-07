"""Full-budget token transport tests, not full-size accelerator-training tests."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from expdis_jax import train
from expdis_jax.config import TrainConfig, validate_contract
from expdis_jax.distill import build_sft_batch_from_pretokenized, pretokenize_sft_examples
from expdis_jax.filtering import select_quality_pool
from expdis_jax.generate import Completion


@pytest.fixture
def tokenizer():
    backend = Tokenizer(models.WordLevel(
        {"[UNK]": 0, "prompt": 1, "x": 2, "[EOS]": 3}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
                                  pad_token="[UNK]", eos_token="[EOS]")


def test_defaults_and_full_contract_allow_full_budget_sft(monkeypatch):
    monkeypatch.setenv("EXPDIS_DAPO_DRGRPO_FULL_CONTRACT", "1")
    cfg = TrainConfig()
    assert cfg.accepted_min_completion_tokens == 1
    assert cfg.accepted_max_completion_tokens == cfg.max_completion_len == 32_768
    assert cfg.distill_max_total_len == cfg.max_prompt_len + 32_768 == 34_816
    validate_contract(cfg, require_eval_contract=True)


def test_full_contract_rejects_accidental_legacy_window(monkeypatch):
    monkeypatch.setenv("EXPDIS_DAPO_DRGRPO_FULL_CONTRACT", "1")
    cfg = replace(TrainConfig(), accepted_min_completion_tokens=128,
                  accepted_max_completion_tokens=4_500, distill_max_total_len=6_548)
    with pytest.raises(ValueError, match="accepted_.*completion_tokens"):
        validate_contract(cfg)


@pytest.mark.parametrize("length", [127, 4_501, 10_000, 32_768])
def test_sampled_completion_and_terminal_label_survive_sft(tokenizer, monkeypatch, length):
    monkeypatch.setenv("EXPDIS_SFT_APPEND_EOS", "0")
    cfg = TrainConfig()
    ids = [2] * (length - 1) + [tokenizer.eos_token_id]
    example = {"prompt_text": "prompt " * 2_048,
               "completion_text": "Text must not replace the sampled token IDs.",
               "completion_token_ids": ids}
    encoded = pretokenize_sft_examples(tokenizer, [example], cfg.distill_max_total_len,
                                      cfg.max_prompt_len)
    assert encoded[0]["c_ids"] == ids
    assert len(encoded[0]["p_ids"]) == 2_048
    batch = build_sft_batch_from_pretokenized(encoded, cfg.distill_max_total_len,
                                             tokenizer.pad_token_id)
    supervised = batch["labels"][0][batch["label_mask"][0].astype(bool)]
    np.testing.assert_array_equal(supervised, ids)
    assert supervised[-1] == tokenizer.eos_token_id
    assert batch["label_mask"].sum() == length


@pytest.mark.parametrize("with_token_ids", [False, True])
def test_preprocessing_raises_instead_of_truncating(tokenizer, monkeypatch, with_token_ids):
    monkeypatch.setenv("EXPDIS_SFT_APPEND_EOS", "0")
    example = {"prompt_text": "prompt " * 2_048, "completion_text": "x " * 10_000}
    if with_token_ids:
        example["completion_token_ids"] = [2] * 9_999 + [3]
    tokenizer.truncation_side = "right"
    with pytest.raises(ValueError, match="SFT.*(capacity|truncat)"):
        pretokenize_sft_examples(tokenizer, [example], 6_548)
    assert tokenizer.truncation_side == "right"


def test_text_fallback_also_keeps_the_full_completion(tokenizer, monkeypatch):
    monkeypatch.setenv("EXPDIS_SFT_APPEND_EOS", "0")
    example = {"prompt_text": "prompt " * 2_048, "completion_text": "x " * 32_768}
    encoded = pretokenize_sft_examples(tokenizer, [example], TrainConfig().distill_max_total_len)
    assert encoded[0]["c_ids"] == [2] * 32_768


def test_batch_builder_cannot_silently_slice_a_completion():
    example = {"p_ids": [1] * 2_048, "c_ids": [2] * 9_999 + [3]}
    with pytest.raises(ValueError, match="SFT.*(capacity|truncat)"):
        build_sft_batch_from_pretokenized([example], 6_548, 0)


def test_append_eos_does_not_drop_existing_full_length_terminal(tokenizer, monkeypatch):
    monkeypatch.setenv("EXPDIS_SFT_APPEND_EOS", "1")
    ids = [2] * 32_767 + [3]
    example = {"prompt_text": "prompt " * 2_048,
               "completion_text": "unused", "completion_token_ids": ids}
    encoded = pretokenize_sft_examples(tokenizer, [example], 34_816)
    assert encoded[0]["c_ids"] == ids


def test_append_eos_raises_if_it_would_exceed_capacity(tokenizer, monkeypatch):
    monkeypatch.setenv("EXPDIS_SFT_APPEND_EOS", "1")
    example = {"prompt_text": "prompt " * 2_048,
               "completion_text": "unused", "completion_token_ids": [2] * 32_768}
    with pytest.raises(ValueError, match="SFT.*(capacity|truncat)"):
        pretokenize_sft_examples(tokenizer, [example], 34_816)


@pytest.mark.parametrize("finish_reason", ["stop", "eos", "eos_token", "stop_sequence"])
def test_natural_end_at_full_budget_reaches_filter_and_sft(tokenizer, monkeypatch, finish_reason):
    monkeypatch.setenv("EXPDIS_SFT_APPEND_EOS", "0")
    cfg = replace(TrainConfig(), grpo_num_generations=1)
    suffix = r"Therefore \boxed{1}."
    suffix_length = len(tokenizer(suffix, add_special_tokens=False)["input_ids"])
    # Unique words avoid the repetition screen; token IDs come from the real
    # fast tokenizer, with the sampled terminal action absent from decoded text.
    text = " ".join(f"term{i}" for i in range(32_767 - suffix_length)) + " " + suffix
    ids = tokenizer(text, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    assert len(ids) == cfg.max_completion_len
    example = SimpleNamespace(problem_id="boundary", prompt_text="prompt " * 2_048,
                              ground_truth="1")
    completion = Completion(text, [], [], finish_reason, ids)
    rows, _, _ = train._score_rollouts(
        tokenizer=tokenizer, examples=[example], completions=[[completion]], cfg=cfg)
    rollout = train._build_rollout_batch(tokenizer, [example], [[completion]], cfg, rows)
    assert rows[0]["completion_token_length"] == 32_768
    assert rows[0]["terminated"] is True
    assert rows[0]["clipped"] is False
    assert rollout["completion_mask"].sum() == 32_768
    assert rollout["terminated"][0] == 1
    assert rollout["clipped"][0] == 0
    np.testing.assert_array_equal(rollout["full_input_ids"][0, 2_048:], ids)

    selected, funnel = select_quality_pool(rows)
    assert len(selected) == funnel["accepted"] == 1
    encoded = pretokenize_sft_examples(tokenizer, selected, cfg.distill_max_total_len,
                                      cfg.max_prompt_len)
    sft = build_sft_batch_from_pretokenized(encoded, cfg.distill_max_total_len,
                                           tokenizer.pad_token_id)
    supervised = sft["labels"][0][sft["label_mask"][0].astype(bool)]
    np.testing.assert_array_equal(supervised, ids)
    assert supervised[-1] == tokenizer.eos_token_id
