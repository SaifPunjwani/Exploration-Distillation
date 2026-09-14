import gc
import os

import torch

from .devices import mark_step

_NOVELTY_BATCH_SIZE = int(os.environ.get("TMX_NOVELTY_BATCH_SIZE", "16") or 16)


def auto_layer_selection(num_hidden_layers: int):
    if num_hidden_layers <= 0:
        return [0]
    if num_hidden_layers <= 3:
        return list(range(num_hidden_layers))
    picks = [
        num_hidden_layers // 4,
        num_hidden_layers // 2,
        (3 * num_hidden_layers) // 4,
    ]
    uniq = sorted(set(max(0, min(num_hidden_layers - 1, p)) for p in picks))
    return uniq if uniq else [num_hidden_layers - 1]


def _pool_hidden(hidden: torch.Tensor, attention_mask: torch.Tensor, pool: str) -> torch.Tensor:
    pool = (pool or "mean").lower()
    mask = attention_mask.float()
    if pool == "last_token":
        last_idx = (mask.sum(dim=1).long() - 1).clamp(min=0)
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        return hidden[batch_idx, last_idx]

    # Default mean pooling over non-pad tokens.
    mask = mask.unsqueeze(-1)
    masked_hidden = hidden * mask
    sum_hidden = masked_hidden.sum(dim=1)
    count = mask.sum(dim=1).clamp(min=1.0)
    return sum_hidden / count


def get_completion_features(
    texts,
    model,
    tokenizer,
    device,
    max_length: int = 512,
    use_input_embeddings: bool = False,
    feature_source: str = "last",
    layers=None,
    layer_pool: str = "mean",
    return_stats: bool = False,
):
    """
    Return pooled completion features.
    Output format:
      - {'last': [B, H]} for feature_source='last'
      - {'layer_<idx>': [B, H], ...} for feature_source='multilayer'
    """
    if max_length is None:
        max_length = 512
    max_length = int(max(8, max_length))

    raw_enc = tokenizer(
        texts,
        add_special_tokens=True,
        padding=False,
        truncation=False,
    )
    raw_token_lengths = [len(ids) for ids in raw_enc["input_ids"]]
    feature_stats = {
        "count": int(len(raw_token_lengths)),
        "truncated_count": int(sum(1 for length in raw_token_lengths if int(length) > max_length)),
        "token_length_sum": float(sum(raw_token_lengths)),
        "max_input_tokens": int(max(raw_token_lengths) if raw_token_lengths else 0),
        "max_length": int(max_length),
    }

    is_xla = getattr(device, "type", "") == "xla"

    # tpu_flash attention (Pallas kernel) only works on XLA devices.  If the
    # model was loaded with tpu_flash but we're running on CPU (e.g. novelty
    # scoring after offload), fall back to eager to avoid RuntimeError.
    _orig_attn = None
    if not is_xla and getattr(getattr(model, "config", None), "_attn_implementation", None) == "tpu_flash":
        _orig_attn = "tpu_flash"
        model.config._attn_implementation = "eager"
        if hasattr(model.config, "_attn_implementation_internal"):
            model.config._attn_implementation_internal = "eager"

    # Fixed-shape padding for XLA graph caching: always pad to max_length so
    # the compiled graph can be reused across batches with different natural
    # sequence lengths.
    padding_strategy = "max_length" if is_xla else True
    enc = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=padding_strategy,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    source = (feature_source or "last").lower()
    if source != "multilayer":
        with torch.no_grad():
            if use_input_embeddings:
                emb_layer = model.get_input_embeddings()
                hidden = emb_layer(input_ids)
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                hidden = outputs.hidden_states[-1]
        features = {"last": _pool_hidden(hidden, attention_mask, layer_pool).detach()}
        del hidden
        if not use_input_embeddings:
            del outputs
        del input_ids
        del attention_mask
        gc.collect()
        mark_step(is_xla=is_xla)
        if _orig_attn is not None:
            model.config._attn_implementation = _orig_attn
            if hasattr(model.config, "_attn_implementation_internal"):
                model.config._attn_implementation_internal = _orig_attn
        if return_stats:
            return features, feature_stats
        return features

    # Micro-batch multilayer feature extraction so novelty scoring does not
    # force one giant host-side forward over every completion at once.
    batch_size = max(1, _NOVELTY_BATCH_SIZE)
    num_texts = input_ids.size(0)
    all_features: dict = {}
    selected = None
    import time as _time
    _feat_t0 = _time.monotonic()
    _n_chunks = (num_texts + batch_size - 1) // batch_size
    print(f"[novelty-feat] starting multilayer extraction: {num_texts} texts, batch={batch_size}, chunks={_n_chunks}, device={device}, seq_len={input_ids.size(1)}", flush=True)
    for chunk_start in range(0, num_texts, batch_size):
        chunk_end = min(num_texts, chunk_start + batch_size)
        actual_chunk_size = chunk_end - chunk_start
        chunk_ids = input_ids[chunk_start:chunk_end]
        chunk_mask = attention_mask[chunk_start:chunk_end]
        # Pad last micro-batch to full batch_size for fixed XLA graph shape.
        if is_xla and actual_chunk_size < batch_size:
            pad_rows = batch_size - actual_chunk_size
            chunk_ids = torch.nn.functional.pad(chunk_ids, (0, 0, 0, pad_rows))
            chunk_mask = torch.nn.functional.pad(chunk_mask, (0, 0, 0, pad_rows))
        _chunk_t0 = _time.monotonic()
        with torch.no_grad():
            outputs = model(
                input_ids=chunk_ids,
                attention_mask=chunk_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = outputs.hidden_states
        _chunk_dt = _time.monotonic() - _chunk_t0
        print(f"[novelty-feat] chunk {chunk_start}:{chunk_end} forward done in {_chunk_dt:.1f}s", flush=True)
        if selected is None:
            n_layers = max(1, len(hidden_states) - 1)
            selected = list(layers) if layers else auto_layer_selection(n_layers)
        for li in selected:
            li = int(max(0, min(len(hidden_states) - 2, li)))
            key = f"layer_{li}"
            pooled = _pool_hidden(hidden_states[li + 1], chunk_mask, layer_pool).detach()
            # Slice off padding rows so only real data is kept.
            pooled = pooled[:actual_chunk_size]
            all_features.setdefault(key, []).append(pooled)
            del pooled
        del hidden_states, outputs, chunk_ids, chunk_mask
        gc.collect()
        mark_step(is_xla=is_xla)
    print(f"[novelty-feat] all chunks done in {_time.monotonic() - _feat_t0:.1f}s", flush=True)
    features = {k: torch.cat(v, dim=0) for k, v in all_features.items()}
    del all_features, input_ids, attention_mask
    gc.collect()
    mark_step(is_xla=is_xla)

    # Restore original attention implementation if we switched it.
    if _orig_attn is not None:
        model.config._attn_implementation = _orig_attn
        if hasattr(model.config, "_attn_implementation_internal"):
            model.config._attn_implementation_internal = _orig_attn

    if return_stats:
        return features, feature_stats
    return features


def get_completion_embeddings(
    texts,
    model,
    tokenizer,
    device,
    max_length: int = 128,
    use_input_embeddings: bool = False,
    feature_source: str = "last",
    layers=None,
    layer_pool: str = "mean",
) -> torch.Tensor:
    """
    Backward-compatible helper. Returns a single [B, H] tensor.
    For multilayer features, this returns the mean across selected layers.
    """
    feats = get_completion_features(
        texts=texts,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_length=max_length,
        use_input_embeddings=use_input_embeddings,
        feature_source=feature_source,
        layers=layers,
        layer_pool=layer_pool,
    )
    if "last" in feats:
        return feats["last"]
    stacked = torch.stack(list(feats.values()), dim=0)  # [L, B, H]
    return stacked.mean(dim=0)


