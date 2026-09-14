import json
import math
import os
import shutil
import torch
import torch.nn as nn
import time
from tqdm.auto import tqdm

from .devices import optimizer_step, mark_step
from .utils import causal_lm_from_pretrained


def _per_token_logprobs(logits: torch.Tensor, labels: torch.Tensor):
    """
    Compute per-token logprobs for completion tokens (labels != -100).
    Applies the standard causal LM shift: logits[t] predicts token t+1.
    Returns (per_token_logps, completion_mask) both of shape [B, T-1].
    """
    logits = logits[:, :-1, :]
    labels = labels[:, 1:]
    log_probs = torch.log_softmax(logits, dim=-1)
    labels_safe = labels.clone()
    labels_safe[labels_safe == -100] = 0
    token_logp = log_probs.gather(-1, labels_safe.unsqueeze(-1)).squeeze(-1)
    mask = (labels != -100).to(token_logp.dtype)
    token_logp = token_logp * mask
    return token_logp, mask


def _sequence_logprobs(logits: torch.Tensor, labels: torch.Tensor):
    """
    Compute per-sequence logprobs for completion tokens (labels != -100).
    Returns (sum_logp, mean_logp, lengths).
    """
    token_logp, mask = _per_token_logprobs(logits, labels)
    lengths = mask.sum(dim=1).clamp(min=1)
    sum_logp = token_logp.sum(dim=1)
    mean_logp = sum_logp / lengths
    return sum_logp, mean_logp, lengths


def _run_root_from_cfg(cfg):
    return os.path.abspath(os.path.dirname(cfg.plots_dir) or ".")


def _stage_checkpoint_dir(cfg, stage_name: str):
    return os.path.join(_run_root_from_cfg(cfg), "checkpoints", stage_name)


def _stage_resume_enabled() -> bool:
    return os.environ.get("TMX_RESUME_TRAINING", "0") not in ("", "0", "false", "False", "no", "NO")


def _load_stage_checkpoint(cfg, stage_name: str, tokenizer, device, is_xla: bool):
    stage_dir = _stage_checkpoint_dir(cfg, stage_name)
    state_path = os.path.join(stage_dir, "resume_state.json")
    if not _stage_resume_enabled():
        return None, 0, stage_dir
    if not (os.path.isdir(stage_dir) and os.path.isfile(state_path)):
        return None, 0, stage_dir
    try:
        with open(state_path, "r", encoding="utf-8") as handle:
            state = json.load(handle)
        completed_epochs = int(state.get("completed_epochs", 0) or 0)
    except Exception:
        completed_epochs = 0
    print(f"[resume] Loading {stage_name} checkpoint from {stage_dir} (completed_epochs={completed_epochs})")
    model = causal_lm_from_pretrained(stage_dir, is_xla=is_xla).to(device)
    if tokenizer is not None:
        try:
            tokenizer.pad_token_id = model.config.pad_token_id or tokenizer.pad_token_id
        except Exception:
            pass
    return model, completed_epochs, stage_dir


def _save_stage_checkpoint(model, tokenizer, stage_dir: str, completed_epochs: int):
    os.makedirs(stage_dir, exist_ok=True)
    state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    try:
        model.save_pretrained(stage_dir, state_dict=state_dict, safe_serialization=True)
    except TypeError:
        model.save_pretrained(stage_dir, state_dict=state_dict)
    if tokenizer is not None:
        tokenizer.save_pretrained(stage_dir)
    with open(os.path.join(stage_dir, "resume_state.json"), "w", encoding="utf-8") as handle:
        json.dump({"completed_epochs": int(completed_epochs), "save_time_unix": float(time.time())}, handle, indent=2)


def _save_periodic_step_checkpoint(model, tokenizer, cfg, stage_name: str, global_step: int):
    save_steps = max(0, int(os.environ.get("TMX_ACTUAL_GRPO_SAVE_STEPS", "0") or 0))
    if save_steps <= 0 or global_step <= 0 or (global_step % save_steps) != 0:
        return
    save_limit = max(1, int(os.environ.get("TMX_ACTUAL_GRPO_SAVE_TOTAL_LIMIT", "4") or 4))
    root = os.path.join(_run_root_from_cfg(cfg), "checkpoints")
    step_dir = os.path.join(root, f"{stage_name}_step_{int(global_step):06d}")
    os.makedirs(step_dir, exist_ok=True)
    state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    try:
        model.save_pretrained(step_dir, state_dict=state_dict, safe_serialization=True)
    except TypeError:
        model.save_pretrained(step_dir, state_dict=state_dict)
    if tokenizer is not None:
        tokenizer.save_pretrained(step_dir)
    with open(os.path.join(step_dir, "resume_state.json"), "w", encoding="utf-8") as handle:
        json.dump({"global_step": int(global_step), "save_time_unix": float(time.time())}, handle, indent=2)

    prefix = f"{stage_name}_step_"
    candidates = []
    try:
        for entry in os.listdir(root):
            if not entry.startswith(prefix):
                continue
            entry_path = os.path.join(root, entry)
            if not os.path.isdir(entry_path):
                continue
            try:
                step_value = int(entry.split("_step_", 1)[1])
            except Exception:
                continue
            candidates.append((step_value, entry_path))
    except FileNotFoundError:
        candidates = []
    candidates.sort(key=lambda item: item[0], reverse=True)
    for _, old_path in candidates[save_limit:]:
        shutil.rmtree(old_path, ignore_errors=True)


def _resolve_training_world_size(is_xla: bool) -> int:
    try:
        world_size = int(os.environ.get("WORLD_SIZE") or os.environ.get("LOCAL_WORLD_SIZE") or "1")
    except ValueError:
        world_size = 1
    if world_size <= 0:
        world_size = 1
    if is_xla and os.environ.get("TMX_USE_XLA_WORLD_SIZE", "0") == "1":
        try:
            import torch_xla.runtime as xr  # type: ignore
            runtime_count = None
            for attr_name in ("global_runtime_device_count", "addressable_runtime_device_count"):
                fn = getattr(xr, attr_name, None)
                if fn is None:
                    continue
                try:
                    runtime_count = int(fn())
                except Exception:
                    runtime_count = None
                if runtime_count and runtime_count > 0:
                    break
            if runtime_count and runtime_count > 0:
                world_size = runtime_count
            else:
                world_size = int(xr.world_size())
        except Exception:
            try:
                import torch_xla.core.xla_model as xm  # type: ignore
                world_size = int(xm.xrt_world_size())
            except Exception:
                world_size = max(1, world_size)
    return max(1, world_size)


def train_actual_model(distill_dataset, tokenizer, cfg, device, is_xla: bool, use_wandb: bool, wandb_run=None, init_model=None):
    """
    Train ACTUAL model by supervised distillation on Explorer's traces.
    """
    from transformers import AutoModelForCausalLM

    resumed_model, start_epoch, stage_dir = _load_stage_checkpoint(cfg, "actual_distill_last", tokenizer, device, is_xla=is_xla)
    if resumed_model is not None:
        actual_model = resumed_model
        print("Using resumed Actual distillation checkpoint...")
    elif init_model is None:
        print("Loading Actual model (fresh base copy)...")
        actual_model = causal_lm_from_pretrained(cfg.model_name, is_xla=is_xla).to(device)
    else:
        print("Using warm-started Actual model for distillation...")
        actual_model = init_model.to(device)
    actual_model.config.use_cache = False
    actual_model.train()

    # Flash attention handling for SFT.
    # - Pallas backward needs ~22G HBM.  On ≤16G chips, swap to eager.
    # - On ≥32G chips, flash backward fits; keep flash for O(N) memory.
    # - Eager attention at 10k tokens uses O(N²)=88G across all layers —
    #   even gradient checkpointing can't help because XLA compiles
    #   the full backward graph at once.
    # Control via TMX_DISTILL_EAGER_ATTN: "1" force eager, "0" keep flash,
    # unset = auto (keep flash if loaded, since most v6e chips are ≥32G).
    _distill_orig_attn = getattr(actual_model.config, "_attn_implementation", None)
    _distill_flash_swapped = False
    _force_eager = os.environ.get("TMX_DISTILL_EAGER_ATTN", "").strip().lower()
    if _distill_orig_attn == "tpu_flash" and _force_eager in ("1", "true", "yes"):
        actual_model.config._attn_implementation = "eager"
        if hasattr(actual_model.config, "_attn_implementation_internal"):
            actual_model.config._attn_implementation_internal = "eager"
        _distill_flash_swapped = True
        print("[distill] Flash attn → eager (TMX_DISTILL_EAGER_ATTN=1)")
    elif _distill_orig_attn == "tpu_flash":
        print("[distill] Keeping flash attention for SFT (Pallas backward fits ≥32G chips)")
    _distill_flash_nan_fix = (_distill_orig_attn == "tpu_flash" and not _distill_flash_swapped)

    grad_ckpt_env = (os.environ.get("TMX_DISTILL_GRAD_CKPT", "1" if is_xla else "0") or "0").strip().lower()
    use_grad_ckpt = grad_ckpt_env not in ("0", "false", "no")
    if use_grad_ckpt:
        try:
            from .explorer import _enable_model_grad_checkpointing

            _enable_model_grad_checkpointing(
                actual_model,
                preserve_rng_state=False,
                verbose=True,
                label="distill",
            )
            print("[distill] gradient checkpointing enabled")
        except Exception as e:
            print(f"[distill] WARN: could not enable gradient checkpointing: {e}")
    elif hasattr(actual_model, "gradient_checkpointing_disable"):
        try:
            actual_model.gradient_checkpointing_disable()
        except Exception:
            pass

    optim_override = (
        os.environ.get("TMX_DISTILL_OPTIM", "").strip()
        or os.environ.get("TMX_OPTIM", "").strip()
        or "adamw_torch"
    ).lower()
    if optim_override in {"adamw", "adamw_torch", "adamw_hf"}:
        optimizer = torch.optim.AdamW(actual_model.parameters(), lr=cfg.distill_lr)
    elif optim_override in {"adafactor", "transformers_adafactor"}:
        try:
            from transformers.optimization import Adafactor
        except Exception:
            from transformers import Adafactor  # type: ignore

        optimizer = Adafactor(
            actual_model.parameters(),
            lr=cfg.distill_lr,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
        )
    else:
        raise ValueError(
            f"Unsupported TMX_DISTILL_OPTIM/TMX_OPTIM={optim_override!r}. "
            "Supported values: adamw_torch, adafactor."
        )
    print(f"[distill] optimizer={optim_override} lr={cfg.distill_lr}")

    loader = torch.utils.data.DataLoader(
        distill_dataset,
        batch_size=cfg.distill_batch_size,
        shuffle=True,
    )

    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    epoch_losses = []
    global_step = 0
    train_start = time.perf_counter()

    # Flush XLA caches and Python garbage before training to maximize
    # free HBM for the backward graph (flash backward needs ~22G).
    import gc
    gc.collect()
    mark_step(is_xla=is_xla)

    if start_epoch >= int(cfg.distill_epochs):
        print(f"[resume] Distillation already complete at epoch {start_epoch}/{cfg.distill_epochs}.")
        return actual_model, epoch_losses

    for epoch in range(start_epoch, cfg.distill_epochs):
        total_loss = 0.0
        steps = 0

        for batch in tqdm(loader, desc=f"Distill epoch {epoch+1}"):
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            optimizer.zero_grad()

            # Flash attention (Pallas) ignores attention_mask and uses only
            # causal masking.  With right-padded data, padding tokens produce
            # garbage KV that causes NaN in the kernel.  Fix: replace padding
            # input_ids with the first real token so the model sees "real"
            # embeddings everywhere.  Labels already have -100 for padding,
            # so loss is unaffected.
            if is_xla and attention_mask.sum() < attention_mask.numel():
                _fill_id = input_ids[0, 0]
                input_ids = torch.where(attention_mask.bool(), input_ids, _fill_id)
                attention_mask = torch.ones_like(attention_mask)

            # Chunked LM-head forward: run transformer backbone to get hidden
            # states (~40MB), then apply lm_head + CE in chunks along the
            # sequence dimension.  This avoids materializing the full
            # [seq_len, vocab_size] logits tensor (~3GB for 10k×152k) which
            # can OOM on a single TPU chip.
            _CHUNK = max(64, int(os.environ.get("TMX_DISTILL_CE_CHUNK", "512") or 512))
            _use_chunked_head = is_xla and hasattr(actual_model, "model") and hasattr(actual_model, "lm_head")
            if _use_chunked_head:
                _backbone_out = actual_model.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                _hidden = _backbone_out[0]  # [batch, seq, hidden_dim]
                _seq_len = _hidden.size(1)
                _total_tokens = int((labels != -100).sum().item())
                if _total_tokens == 0:
                    del _hidden, _backbone_out
                    mark_step(is_xla=is_xla)
                    global_step += 1
                    continue
                # Chunked CE: process lm_head + cross-entropy in chunks.
                # Flash attention (Pallas) may produce NaN in some hidden
                # states → NaN logits → NaN per-token CE.  Handle at the
                # per-token level (like GRPO's nan_to_num on per_token_logps)
                # which adds minimal XLA graph overhead compared to
                # nan_to_num on the full hidden states tensor.
                _total_loss = torch.tensor(0.0, device=_hidden.device, dtype=torch.float32)
                _valid_tokens = torch.tensor(0, device=_hidden.device, dtype=torch.long)
                for _ci in range(0, _seq_len, _CHUNK):
                    _ce = min(_ci + _CHUNK, _seq_len)
                    _chunk_logits = actual_model.lm_head(_hidden[:, _ci:_ce, :]).float()
                    _chunk_labels = labels[:, _ci:_ce].reshape(-1)
                    if _distill_flash_nan_fix:
                        # Per-token CE, then zero out NaN tokens.
                        _per_tok = nn.functional.cross_entropy(
                            _chunk_logits.reshape(-1, _chunk_logits.size(-1)),
                            _chunk_labels,
                            ignore_index=-100,
                            reduction="none",
                        )
                        _tok_valid = ~torch.isnan(_per_tok) & (_chunk_labels != -100)
                        _per_tok = torch.where(_tok_valid, _per_tok, torch.zeros_like(_per_tok))
                        _total_loss = _total_loss + _per_tok.sum()
                        _valid_tokens = _valid_tokens + _tok_valid.sum()
                    else:
                        _chunk_loss = nn.functional.cross_entropy(
                            _chunk_logits.reshape(-1, _chunk_logits.size(-1)),
                            _chunk_labels,
                            ignore_index=-100,
                            reduction="sum",
                        )
                        _total_loss = _total_loss + _chunk_loss
                        _valid_tokens = _valid_tokens + (_chunk_labels != -100).sum()
                    del _chunk_logits
                del _hidden, _backbone_out
                _valid_count = max(int(_valid_tokens.item()), 1)
                if _distill_flash_nan_fix and (global_step < 10 or global_step % 200 == 0):
                    print(f"[distill] valid tokens for loss: {_valid_count}/{_total_tokens}", flush=True)
                loss = _total_loss / _valid_count
            else:
                outputs = actual_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits
                loss = loss_fct(
                    logits.float().view(-1, logits.size(-1)),
                    labels.view(-1),
                )
                del logits

            _loss_val = loss.item()
            if global_step < 10 or global_step % 200 == 0:
                print(f"[distill] step={global_step} loss={_loss_val:.4f}", flush=True)

            # Skip NaN losses to prevent weight corruption from NaN gradients.
            if math.isnan(_loss_val) or math.isinf(_loss_val):
                if global_step < 50 or global_step % 100 == 0:
                    _valid_labels = (labels != -100).sum().item()
                    print(f"[distill-WARN] NaN/inf loss at step={global_step}, "
                          f"valid_labels={_valid_labels}, skipping backward", flush=True)
                mark_step(is_xla=is_xla)
                global_step += 1
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(actual_model.parameters(), max_norm=1.0)
            optimizer_step(optimizer, is_xla=is_xla)
            mark_step(is_xla=is_xla)

            total_loss += _loss_val
            steps += 1
            global_step += 1

            if use_wandb and wandb_run is not None:
                wandb_run.log(
                    {
                        "distill/step": global_step,
                        "distill/batch_loss": loss.item(),
                        "distill/wall_time_seconds": time.perf_counter() - train_start,
                    }
                )

        avg = total_loss / max(1, steps)
        print(f"Distill epoch {epoch+1}: avg loss = {avg:.4f}")
        epoch_losses.append(avg)

        if use_wandb and wandb_run is not None:
            wandb_run.log(
                {
                    "distill/epoch": epoch + 1,
                    "distill/loss": avg,
                    "distill/wall_time_seconds": time.perf_counter() - train_start,
                }
            )
        _save_stage_checkpoint(actual_model, tokenizer, stage_dir, epoch + 1)

    # Restore flash attention for subsequent GRPO stages
    if _distill_flash_swapped:
        actual_model.config._attn_implementation = "tpu_flash"
        if hasattr(actual_model.config, "_attn_implementation_internal"):
            actual_model.config._attn_implementation_internal = "tpu_flash"
        print("[distill] Flash attention restored for subsequent stages")

    return actual_model, epoch_losses


def train_actual_with_offline_grpo(actual_dataset,
                                   tokenizer,
                                   cfg,
                                   device,
                                   is_xla: bool,
                                   use_wandb: bool,
                                   wandb_run=None,
                                   init_model=None):
    """
    Train ACTUAL model with offline GRPO-style updates on Explorer trajectories.
    """
    from transformers import AutoModelForCausalLM

    resumed_model, start_epoch, stage_dir = _load_stage_checkpoint(cfg, "actual_offline_grpo_last", tokenizer, device, is_xla=is_xla)
    if resumed_model is not None:
        actual_model = resumed_model
        print("Using resumed Actual offline GRPO checkpoint...")
    elif init_model is None:
        print("Loading Actual model (fresh base copy) for offline GRPO...")
        actual_model = causal_lm_from_pretrained(cfg.model_name, is_xla=is_xla).to(device)
    else:
        print("Using warm-started Actual model for offline GRPO...")
        actual_model = init_model.to(device)
    actual_model.config.use_cache = False
    actual_model.train()

    ref_model = None
    if cfg.actual_grpo_kl_coef and cfg.actual_grpo_kl_coef > 0:
        print("Loading reference model for KL regularization...")
        ref_model = causal_lm_from_pretrained(cfg.model_name, is_xla=is_xla).to(device)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

    optim_override = (
        os.environ.get("TMX_ACTUAL_GRPO_OPTIM", "").strip()
        or os.environ.get("TMX_OPTIM", "").strip()
        or "adamw_torch"
    ).lower()
    if optim_override in {"adamw", "adamw_torch", "adamw_hf"}:
        optimizer = torch.optim.AdamW(actual_model.parameters(), lr=cfg.actual_grpo_lr)
    elif optim_override in {"adafactor", "transformers_adafactor"}:
        try:
            from transformers.optimization import Adafactor
        except Exception:
            from transformers import Adafactor  # type: ignore
        optimizer = Adafactor(
            actual_model.parameters(),
            lr=cfg.actual_grpo_lr,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
        )
    else:
        raise ValueError(f"Unsupported optimizer for Actual GRPO: {optim_override!r}")
    print(f"[actual-grpo] optimizer={optim_override} lr={cfg.actual_grpo_lr}")

    loader = torch.utils.data.DataLoader(
        actual_dataset,
        batch_size=cfg.actual_grpo_batch_size,
        shuffle=True,
    )

    grad_accum = max(1, cfg.actual_grpo_grad_accum)
    global_step = 0
    log_history = []
    train_start = time.perf_counter()

    if start_epoch >= int(cfg.actual_grpo_epochs):
        print(f"[resume] Offline GRPO already complete at epoch {start_epoch}/{cfg.actual_grpo_epochs}.")
        return actual_model, log_history

    clip_eps = 0.2
    for epoch in range(start_epoch, cfg.actual_grpo_epochs):
        # Precompute old (behavior) and reference logprobs for the entire
        # dataset once per epoch.  This avoids a second forward pass per
        # batch during training and keeps the "old" policy fixed within
        # each epoch (standard offline PPO).
        print(f"[actual-grpo] Precomputing old/ref logprobs for epoch {epoch+1}...")
        _precomp_loader = torch.utils.data.DataLoader(
            actual_dataset, batch_size=cfg.actual_grpo_batch_size, shuffle=False,
        )
        all_old_logps = []
        all_ref_logps = []
        actual_model.eval()
        with torch.no_grad():
            for _pb in _precomp_loader:
                _ids, _mask, _labs = _pb[0].to(device), _pb[1].to(device), _pb[2].to(device)
                _old_out = actual_model(input_ids=_ids, attention_mask=_mask)
                _old_tl, _ = _per_token_logprobs(_old_out.logits, _labs)
                all_old_logps.append(_old_tl.cpu())
                if ref_model is not None:
                    _ref_out = ref_model(input_ids=_ids, attention_mask=_mask)
                    _ref_tl, _ = _per_token_logprobs(_ref_out.logits, _labs)
                    all_ref_logps.append(_ref_tl.cpu())
                mark_step(is_xla=is_xla)
        actual_model.train()
        all_old_logps = torch.cat(all_old_logps, dim=0)
        if ref_model is not None:
            all_ref_logps = torch.cat(all_ref_logps, dim=0)
        print(f"[actual-grpo] Precomputed logprobs for {all_old_logps.size(0)} samples.")

        # Build an index-aware loader so we can look up precomputed logprobs.
        _epoch_indices = torch.randperm(len(actual_dataset))
        optimizer.zero_grad()
        last_batch_idx = -1
        bs = cfg.actual_grpo_batch_size
        num_batches = (len(actual_dataset) + bs - 1) // bs
        for batch_idx in tqdm(range(num_batches), desc=f"Actual offline GRPO epoch {epoch+1}"):
            last_batch_idx = batch_idx
            idx = _epoch_indices[batch_idx * bs : (batch_idx + 1) * bs]
            batch = [actual_dataset[i] for i in idx]
            # Stack into tensors (dataset returns tuples of tensors).
            input_ids = torch.stack([b[0] for b in batch]).to(device)
            attention_mask = torch.stack([b[1] for b in batch]).to(device)
            labels = torch.stack([b[2] for b in batch]).to(device)
            rewards = torch.stack([b[3] for b in batch]).to(device)
            r_correct = torch.stack([b[4] for b in batch]).to(device)
            r_novel = torch.stack([b[5] for b in batch]).to(device)
            is_correct = torch.stack([b[6] for b in batch]).to(device)

            old_per_token_logps = all_old_logps[idx].to(device)
            completion_mask = (labels != -100).to(old_per_token_logps.dtype)

            # Current policy forward pass (with grad).
            outputs = actual_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            per_token_logps, _ = _per_token_logprobs(outputs.logits, labels)
            per_token_logps = torch.nan_to_num(per_token_logps, nan=0.0)

            # KL penalty against base reference model.
            if ref_model is not None:
                ref_per_token_logps = all_ref_logps[idx].to(device)
                per_token_kl = (
                    torch.exp(ref_per_token_logps - per_token_logps)
                    - (ref_per_token_logps - per_token_logps)
                    - 1.0
                )
                per_token_kl = torch.nan_to_num(per_token_kl, nan=0.0)
            else:
                per_token_kl = torch.zeros_like(per_token_logps)

            # Per-sequence metrics for logging.
            lengths = completion_mask.sum(dim=1).clamp(min=1)
            mean_logp = (per_token_logps * completion_mask).sum(dim=1) / lengths

            rewards = rewards.float() * cfg.actual_grpo_reward_scale
            if cfg.actual_grpo_reward_clip is not None:
                rewards = rewards.clamp(
                    min=-cfg.actual_grpo_reward_clip,
                    max=cfg.actual_grpo_reward_clip,
                )
            reward_mean = rewards.mean()
            reward_std = rewards.std(unbiased=False)
            reward_std_value = float(reward_std.item())
            reward_norm_active = bool(cfg.actual_grpo_reward_norm and reward_std_value > 1e-6)
            if reward_norm_active:
                adv = (rewards - reward_mean) / (reward_std + 1e-8)
            else:
                adv = rewards
            if cfg.actual_grpo_adv_clip is not None:
                adv = adv.clamp(
                    min=-cfg.actual_grpo_adv_clip,
                    max=cfg.actual_grpo_adv_clip,
                )

            # PPO-clip objective with DR-GRPO token-level normalization.
            ratio = torch.exp(per_token_logps - old_per_token_logps)
            clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
            per_token_loss1 = ratio * adv.detach().unsqueeze(1)
            per_token_loss2 = clipped_ratio * adv.detach().unsqueeze(1)
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

            if cfg.actual_grpo_kl_coef:
                per_token_loss = per_token_loss + cfg.actual_grpo_kl_coef * per_token_kl

            # DR-GRPO: normalize by total completion tokens across the batch.
            total_tokens = max(1.0, float(completion_mask.sum().item()))
            loss = (per_token_loss * completion_mask).sum() / total_tokens

            loss_scaled = loss / grad_accum
            loss_scaled.backward()

            # Logging scalars.
            kl_mean = (per_token_kl * completion_mask).sum() / total_tokens
            clip_frac = ((ratio != clipped_ratio).float() * completion_mask).sum() / total_tokens

            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(actual_model.parameters(), max_norm=1.0)
                optimizer_step(optimizer, is_xla=is_xla)
                mark_step(is_xla=is_xla)
                optimizer.zero_grad()
                _save_periodic_step_checkpoint(
                    actual_model,
                    tokenizer,
                    cfg,
                    "actual_offline_grpo_last",
                    global_step,
                )

            global_step += 1
            log_row = {
                "step": global_step,
                "loss": float(loss.item()),
                "loss_pg": float(loss.item()),
                "loss_kl": float(kl_mean.item()),
                "clip_frac": float(clip_frac.item()),
                "reward_mean": float(reward_mean.item()),
                "reward_std": reward_std_value,
                "reward_norm_active": 1.0 if reward_norm_active else 0.0,
                "adv_mean": float(adv.mean().item()),
                "adv_std": float(adv.std(unbiased=False).item()),
                "kl_mean": float(kl_mean.item()),
                "mean_logp": float(mean_logp.mean().item()),
                "mean_len": float(lengths.float().mean().item()),
                "batch_correct_frac": float(is_correct.mean().item()),
                "batch_r_correct": float(r_correct.mean().item()),
                "batch_r_novel": float(r_novel.mean().item()),
                "epoch": epoch + 1,
                "wall_time_seconds": time.perf_counter() - train_start,
            }
            log_history.append(log_row)

            if use_wandb and wandb_run is not None:
                wandb_run.log(
                    {f"actual_grpo/{k}": v for k, v in log_row.items()}
                )

        if last_batch_idx >= 0 and (last_batch_idx + 1) % grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(actual_model.parameters(), max_norm=1.0)
            optimizer_step(optimizer, is_xla=is_xla)
            mark_step(is_xla=is_xla)
            optimizer.zero_grad()
            _save_periodic_step_checkpoint(
                actual_model,
                tokenizer,
                cfg,
                "actual_offline_grpo_last",
                global_step,
            )
        _save_stage_checkpoint(actual_model, tokenizer, stage_dir, epoch + 1)

    if ref_model is not None:
        del ref_model

    return actual_model, log_history


def train_actual_with_online_grpo(train_ds, tokenizer, cfg, device, is_xla,
                                   use_wandb, wandb_run, init_model):
    """
    Train Actual model with online GRPO on dataset prompts (correctness reward).

    Unlike offline GRPO which replays pre-collected trajectory rows, this runs
    the full GRPOTrainer loop: generate completions from prompts, compute
    correctness rewards, and do policy updates.  Reuses the Explorer GRPO
    infrastructure with Actual-specific hyperparameters and lambda_novelty=0.
    """
    from copy import deepcopy
    from .explorer import train_explorer_with_grpo

    actual_cfg = deepcopy(cfg)
    actual_cfg.lambda_novelty = 0.0
    actual_cfg.incorrect_novelty_scale = 0.0
    actual_cfg.gate_novelty_by_quality = False
    actual_cfg.novelty_zscore = "off"
    actual_cfg.correctness_zscore = "off"
    actual_cfg.anchor_metric = "none"
    actual_cfg.anchor_weight = 0.0
    actual_cfg.grpo_batch_size = cfg.actual_grpo_batch_size
    actual_cfg.grpo_grad_accum = max(1, cfg.actual_grpo_grad_accum)
    actual_cfg.grpo_lr = cfg.actual_grpo_lr

    world_size = _resolve_training_world_size(is_xla)
    effective_batch = actual_cfg.grpo_batch_size * actual_cfg.grpo_grad_accum * world_size
    steps_per_epoch = max(1, math.ceil(len(train_ds) / max(1, effective_batch)))
    if cfg.actual_online_grpo_max_steps is not None:
        actual_cfg.grpo_max_steps = max(1, int(cfg.actual_online_grpo_max_steps))
    else:
        actual_cfg.grpo_max_steps = steps_per_epoch * max(1, cfg.actual_grpo_epochs)

    actual_cfg.explorer_output_dir = os.path.join(
        _run_root_from_cfg(cfg) or "runs",
        "actual_online_grpo",
    )
    os.makedirs(actual_cfg.explorer_output_dir, exist_ok=True)

    save_steps_env = os.environ.get("TMX_ACTUAL_GRPO_SAVE_STEPS", "25")
    save_limit_env = os.environ.get("TMX_ACTUAL_GRPO_SAVE_TOTAL_LIMIT", "4")
    prev_save_steps = os.environ.get("TMX_GRPO_SAVE_STEPS")
    prev_save_limit = os.environ.get("TMX_GRPO_SAVE_TOTAL_LIMIT")
    os.environ["TMX_GRPO_SAVE_STEPS"] = save_steps_env
    os.environ["TMX_GRPO_SAVE_TOTAL_LIMIT"] = save_limit_env

    # Reset Explorer resume state so Actual GRPO starts from step 0,
    # unless TMX_ACTUAL_GRPO_RESUME_STEP / TMX_ACTUAL_GRPO_RESUME_CHECKPOINT
    # are set for resuming Actual GRPO from a checkpoint.
    prev_resume_step = os.environ.get("TMX_RESUME_STEP")
    prev_resume_ckpt = os.environ.get("TMX_RESUME_CHECKPOINT")
    actual_resume_step = os.environ.get("TMX_ACTUAL_GRPO_RESUME_STEP", "0")
    actual_resume_ckpt = os.environ.get("TMX_ACTUAL_GRPO_RESUME_CHECKPOINT", "")
    os.environ["TMX_RESUME_STEP"] = actual_resume_step
    if actual_resume_ckpt:
        os.environ["TMX_RESUME_CHECKPOINT"] = actual_resume_ckpt

    # Wire actual_grpo_kl_coef through to decoupled training path via env var
    prev_grpo_beta = os.environ.get("TMX_GRPO_BETA")
    kl_coef = getattr(cfg, "actual_grpo_kl_coef", 0.0) or 0.0
    if kl_coef > 0:
        os.environ["TMX_GRPO_BETA"] = str(kl_coef)
        print(f"[actual-online-grpo] KL regularization beta={kl_coef}")

    print(f"[actual-online-grpo] {len(train_ds)} prompts, "
          f"world_size={world_size}, effective_batch={effective_batch}, "
          f"~{steps_per_epoch} steps/epoch × {cfg.actual_grpo_epochs} epochs "
          f"= {actual_cfg.grpo_max_steps} total steps")

    try:
        actual_model, _tok, _ns, log_history = train_explorer_with_grpo(
            train_ds,
            cfg=actual_cfg,
            device=device,
            is_xla=is_xla,
            use_wandb=use_wandb,
            wandb_run=wandb_run,
            init_model=init_model,
            grpo_metric_prefix="actual_grpo",
            train_metric_prefix="actual_grpo",
            run_label="Actual online GRPO",
        )
    finally:
        if prev_resume_step is not None:
            os.environ["TMX_RESUME_STEP"] = prev_resume_step
        elif "TMX_RESUME_STEP" in os.environ:
            del os.environ["TMX_RESUME_STEP"]
        if prev_resume_ckpt is not None:
            os.environ["TMX_RESUME_CHECKPOINT"] = prev_resume_ckpt
        elif "TMX_RESUME_CHECKPOINT" in os.environ:
            del os.environ["TMX_RESUME_CHECKPOINT"]
        if prev_save_steps is not None:
            os.environ["TMX_GRPO_SAVE_STEPS"] = prev_save_steps
        elif "TMX_GRPO_SAVE_STEPS" in os.environ:
            del os.environ["TMX_GRPO_SAVE_STEPS"]
        if prev_save_limit is not None:
            os.environ["TMX_GRPO_SAVE_TOTAL_LIMIT"] = prev_save_limit
        elif "TMX_GRPO_SAVE_TOTAL_LIMIT" in os.environ:
            del os.environ["TMX_GRPO_SAVE_TOTAL_LIMIT"]
        if prev_grpo_beta is not None:
            os.environ["TMX_GRPO_BETA"] = prev_grpo_beta
        elif "TMX_GRPO_BETA" in os.environ:
            del os.environ["TMX_GRPO_BETA"]

    return actual_model, log_history
