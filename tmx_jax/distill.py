"""SFT distillation (Actual model) in JAX.

Loads the collected "accepted" trajectories from Explorer, trains the
Actual model on full-trace targets (prompt is masked, all completion
tokens are supervised, padded tokens ignored).

Uses AdamW for the current contract path. Loss is token-level cross-entropy
with ``label_mask``.
"""

from __future__ import annotations

import os
import time
import hashlib
from dataclasses import asdict

import jax
import jax.numpy as jnp
import numpy as np
from flax.training import train_state
from jax.sharding import NamedSharding, PartitionSpec as P
from transformers import AutoTokenizer

from .config import TrainConfig
from .grpo import chunked_per_token_logps_from_hidden
from .mesh import FSDP_AXIS, build_mesh, shardings_for_params
from .model import Qwen3Model, set_active_mesh
from .train import _adamw_tx, _lm_head_kernel, _make_qwen3_config, _maybe_upload_checkpoint_to_gcs  # reuse
from .weights import hf_to_flax_params


def _stable_wandb_run_id(cfg: TrainConfig) -> str:
    for key in ("TMX_WANDB_ACTUAL_RUN_ID", "TMX_WANDB_DISTILL_RUN_ID", "TMX_WANDB_RUN_ID"):
        if os.environ.get(key):
            return os.environ[key].strip()
    run_name = cfg.wandb_run_name or "jax_actual_sft"
    digest = hashlib.sha1(f"{run_name}:actual".encode("utf-8")).hexdigest()[:24]
    return f"jaxactual{digest}"


def _env_flag(name: str, default: str = "1") -> bool:
    return (os.environ.get(name, default) or "").strip().lower() not in ("0", "false", "no", "off")


def _wandb_log_kwargs(step: int) -> dict:
    if _env_flag("TMX_WANDB_SINGLE_RUN", "1"):
        return {}
    return {"step": step}


def _ce_loss_from_hidden(hidden, labels, mask, lm_head_kernel, chunk_tokens: int):
    """Token-level cross-entropy from hidden states without full-sequence logits materialization."""
    selected_logps = chunked_per_token_logps_from_hidden(
        hidden, labels, lm_head_kernel, chunk_tokens=chunk_tokens
    )
    nll = -selected_logps
    nll = nll * mask.astype(nll.dtype)
    token_count = jnp.maximum(mask.sum(), 1.0)
    return nll.sum() / token_count


def pretokenize_sft_examples(tokenizer, examples, max_total_len: int):
    """Tokenize all SFT examples once upfront so per-batch building is just
    a numpy array slice. Returns a list of {"p_ids": [...], "c_ids": [...]}.

    Truncation policy: prompts truncate from the LEFT (earlier context lost,
    end-of-prompt question preserved); completions truncate from the RIGHT
    (cut off tail of answer). HF tokenizer defaults to 'right' for both,
    which would discard the actual question for long prompts — wrong for
    thinking-mode math. We swap the side per call.
    """
    out = []
    prev_side = getattr(tokenizer, "truncation_side", "right")
    try:
        for ex in examples:
            tokenizer.truncation_side = "left"
            p = tokenizer(ex["prompt_text"], truncation=True, max_length=max_total_len, add_special_tokens=False)["input_ids"]
            tokenizer.truncation_side = "right"
            c_max = max(0, max_total_len - len(p))
            c = tokenizer(ex["completion_text"], truncation=True, max_length=c_max, add_special_tokens=False)["input_ids"] if c_max > 0 else []
            out.append({"p_ids": list(p), "c_ids": list(c)})
    finally:
        tokenizer.truncation_side = prev_side
    return out


def build_sft_batch_from_pretokenized(pretokenized, max_total_len: int, pad_id: int):
    """Build a batch dict from already-tokenized examples (no tokenizer calls)."""
    B = len(pretokenized)
    T = max_total_len
    input_ids = np.full((B, T), pad_id, dtype=np.int32)
    attention_mask = np.zeros((B, T), dtype=np.int32)
    label_mask = np.zeros((B, T), dtype=np.int32)
    labels = np.full((B, T), pad_id, dtype=np.int32)
    for b, ex in enumerate(pretokenized):
        p = ex["p_ids"]
        c = ex["c_ids"]
        seq = (p + c)[:T]
        input_ids[b, :len(seq)] = seq
        attention_mask[b, :len(seq)] = 1
        # Labels shift-left: predict token at pos t from logits at pos t-1.
        # We feed (p + c) as input; target shifted: labels[b, t] = seq[t+1]
        if len(seq) > 1:
            labels[b, :len(seq) - 1] = seq[1:]
            # Only supervise completion tokens (and not the last pad).
            p_len = len(p)
            # Positions in `labels` corresponding to completion predictions: indices >= p_len - 1
            lm_start = max(0, p_len - 1)
            lm_end = len(seq) - 1
            label_mask[b, lm_start:lm_end] = 1
    return {
        "input_ids": jnp.asarray(input_ids),
        "attention_mask": jnp.asarray(attention_mask),
        "labels": jnp.asarray(labels),
        "label_mask": jnp.asarray(label_mask),
    }


def make_sft_train_step(tie_word_embeddings: bool, logprob_chunk_tokens: int):
    def sft_train_step(state, batch):
        def loss_fn(params):
            hidden = state.apply_fn({"params": params}, batch["input_ids"], batch["attention_mask"], return_hidden=True)
            return _ce_loss_from_hidden(
                hidden,
                batch["labels"],
                batch["label_mask"],
                _lm_head_kernel(params, tie_word_embeddings),
                chunk_tokens=logprob_chunk_tokens,
            )
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss
    return sft_train_step


def run_sft(cfg: TrainConfig, accepted_jsonl_path: str):
    """Train Actual model from `cfg.model_name` on accepted trajectories."""
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_wandb = False
    wandb_mod = None
    try:
        import wandb
        wandb_mod = wandb
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity or None,
            name=cfg.wandb_run_name or "jax_actual_sft",
            id=_stable_wandb_run_id(cfg),
            resume="allow",
            group=os.environ.get("TMX_WANDB_GROUP") or (cfg.wandb_run_name or "jax_actual_sft"),
            job_type="actual",
            config=asdict(cfg),
            reinit=True,
        )
        try:
            wandb.define_metric("distill/step")
            wandb.define_metric("distill/*", step_metric="distill/step")
        except Exception:
            pass
        use_wandb = True
    except Exception as e:
        print(f"[sft] wandb disabled: {e}", flush=True)

    m_cfg = _make_qwen3_config(cfg.model_name)
    set_active_mesh(build_mesh())
    model = Qwen3Model(m_cfg)
    params = hf_to_flax_params(
        cfg.model_name,
        m_cfg.num_hidden_layers,
        m_cfg.tie_word_embeddings,
        dtype=jnp.float32,
        use_qk_norm=bool(m_cfg.use_qk_norm),
        hf_weight_prefix=str(m_cfg.hf_weight_prefix),
    )

    tx = _adamw_tx(cfg, cfg.distill_lr)
    mesh = build_mesh()
    set_active_mesh(mesh)
    param_sh = shardings_for_params(params["params"], mesh)
    sharded_params = jax.tree_util.tree_map(lambda x, s: jax.device_put(x, s), params["params"], param_sh)

    def init_fn(raw_params):
        return train_state.TrainState.create(apply_fn=model.apply, params=raw_params, tx=tx)

    state_shape = jax.eval_shape(init_fn, sharded_params)

    def infer_leaf_sh(leaf):
        if getattr(leaf, "ndim", 0) >= 2:
            return NamedSharding(mesh, P(FSDP_AXIS, *([None] * (leaf.ndim - 1))))
        return NamedSharding(mesh, P())

    state_sh = jax.tree_util.tree_map(infer_leaf_sh, state_shape)
    state_sh = state_sh.replace(params=param_sh)
    state = jax.jit(init_fn, out_shardings=state_sh)(sharded_params)

    # Load trajectories
    import json
    examples = []
    with open(accepted_jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            examples.append({"prompt_text": r["prompt_text"], "completion_text": r["completion_text"]})
    print(f"[sft] loaded {len(examples)} accepted trajectories", flush=True)

    # Pre-tokenize ONCE so per-batch building is just numpy slicing. With multi-
    # epoch SFT this saves N*epochs tokenizer calls (worth ~30-90s for N=500).
    t_pretok = time.perf_counter()
    pretokenized = pretokenize_sft_examples(tokenizer, examples, cfg.max_total_len)
    print(f"[sft] pretokenized {len(pretokenized)} examples in {time.perf_counter()-t_pretok:.1f}s", flush=True)

    step_jit = jax.jit(
        make_sft_train_step(bool(m_cfg.tie_word_embeddings), int(cfg.logprob_chunk_tokens)),
        donate_argnums=(0,),
    )
    bs = cfg.distill_batch_size
    epoch = 0
    total_steps = 0
    train_start = time.perf_counter()
    while epoch < cfg.distill_epochs:
        rng = np.random.default_rng(cfg.seed + epoch)
        order = rng.permutation(len(examples))
        epoch_loss_sum = 0.0
        epoch_steps = 0
        for i in range(0, len(order), bs):
            sel = [pretokenized[j] for j in order[i:i+bs]]
            batch = build_sft_batch_from_pretokenized(sel, cfg.max_total_len, int(tokenizer.pad_token_id))
            label_tokens = float(np.asarray(batch["label_mask"]).sum())
            t0 = time.perf_counter()
            state, loss = step_jit(state, batch)
            total_steps += 1
            epoch_steps += 1
            epoch_loss_sum += float(loss)
            dt = time.perf_counter() - t0
            if use_wandb:
                wandb_mod.log(
                    {
                        "distill/step": total_steps,
                        "distill/epoch": epoch,
                        "distill/batch_loss": float(loss),
                        "distill/loss": float(loss),
                        "distill/step_time": dt,
                        "distill/wall_time_seconds": time.perf_counter() - train_start,
                        "distill/learning_rate": float(cfg.distill_lr),
                        "distill/label_tokens": label_tokens,
                        "distill/tokens_per_sec": label_tokens / max(dt, 1e-6),
                    },
                    **_wandb_log_kwargs(total_steps),
                )
            if total_steps % 10 == 0:
                print(f"[sft] step {total_steps} epoch {epoch} loss={float(loss):.4f} dt={dt:.1f}s", flush=True)
        epoch_loss = epoch_loss_sum / max(1, epoch_steps)
        if use_wandb:
                wandb_mod.log(
                    {
                        "distill/step": total_steps,
                        "distill/epoch": epoch + 1,
                        "distill/epoch_loss": epoch_loss,
                        "distill/loss": epoch_loss,
                        "distill/wall_time_seconds": time.perf_counter() - train_start,
                    },
                    **_wandb_log_kwargs(total_steps),
                )
        print(f"[sft] epoch {epoch + 1}: avg loss={epoch_loss:.4f}", flush=True)
        epoch += 1

    # Save final actual model
    import orbax.checkpoint as ocp
    out = os.path.join(cfg.output_dir, "actual_sft_final")
    ocp.PyTreeCheckpointer().save(os.path.abspath(out), {"params": state.params})
    print(f"[sft] saved actual SFT -> {out}", flush=True)
    _maybe_upload_checkpoint_to_gcs(os.path.abspath(out), cfg)
    if use_wandb and wandb_mod is not None:
        try:
            wandb_mod.finish()
        except Exception:
            pass
    return state
