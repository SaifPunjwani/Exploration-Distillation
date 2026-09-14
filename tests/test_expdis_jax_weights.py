"""Load only the selected full-precision HF weight files."""
import json
import numpy as np
import pytest
from safetensors.numpy import save_file
from expdis_jax import weights


def test_single_hf_checkpoint_ignores_native_duplicate(tmp_path):
    save_file({"model.norm.weight": np.ones(2, np.float32)}, str(tmp_path / "model.safetensors"))
    (tmp_path / "consolidated.safetensors").write_bytes(b"must not be opened")
    result = weights._load_shards(str(tmp_path))
    assert set(result) == {"model.norm.weight"}
    np.testing.assert_array_equal(result["model.norm.weight"], 1)


def test_shard_index_selects_files_and_tensors(tmp_path):
    save_file({"a": np.ones(2, np.float32), "unlisted": np.zeros(1, np.float32)}, str(tmp_path / "model-1.safetensors"))
    save_file({"b": np.full(2, 3, np.float32)}, str(tmp_path / "model-2.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        "a": "model-1.safetensors", "b": "model-2.safetensors"}}))
    (tmp_path / "consolidated.safetensors").write_bytes(b"unused native format")
    result = weights._load_shards(str(tmp_path))
    assert set(result) == {"a", "b"}
    np.testing.assert_array_equal(result["b"], 3)


def test_export_metadata_download_excludes_model_weights(monkeypatch):
    calls = []
    monkeypatch.setattr(weights, "snapshot_download", lambda **kw: calls.append(kw) or "/snapshot")
    weights._download("test/model", weights=False)
    assert not any("safetensors" in pattern for pattern in calls[0]["allow_patterns"])
    weights._download("test/model")
    assert "model*.safetensors" in calls[1]["allow_patterns"]
    assert "*.safetensors" not in calls[1]["allow_patterns"]


def test_quantized_config_rejected_before_weight_download(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": "fp8"}}))
    def unexpected(*args, **kwargs):
        raise AssertionError("downloaded unsupported quantized weights")
    monkeypatch.setattr(weights, "_download", unexpected)
    with pytest.raises(ValueError, match="full-precision BF16"):
        weights.hf_to_flax_params(str(tmp_path), 1, True)


def test_modern_tokenizer_metadata_loads_offline(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import AutoTokenizer, TokenizersBackend
    backend = Tokenizer(models.WordLevel({"<unk>": 0, "hello": 1, "<eos>": 2}, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = TokenizersBackend(tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>")
    tokenizer.save_pretrained(tmp_path)
    loaded = AutoTokenizer.from_pretrained(tmp_path, local_files_only=True, fix_mistral_regex=True)
    assert loaded("hello", add_special_tokens=False)["input_ids"] == [1]
    assert loaded.eos_token_id == 2
