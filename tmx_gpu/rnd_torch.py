#!/usr/bin/env python3
"""Multilayer RND novelty scorer in PyTorch, faithful to tmx_jax/novelty.py.

Contract decisions (mirrors the JAX TPU implementation):
- One (frozen target, trainable predictor) pair per policy layer in {7, 14, 21}.
  Each block is an MLP Linear(input_dim, 512) - ReLU - Linear(512, 512) - ReLU
  - Linear(512, 512), input_dim = policy hidden size (2048 for Qwen3-1.7B).
- Features: mean-pool over non-pad tokens of the policy's post-block-`layer`
  hidden state (hidden_states[layer + 1]; see DEVIATIONS) for the re-encoded
  full raw completion text, truncated to novelty_feature_max_length tokens
  (18432 = max_total_len on GPU).
- Per-layer novelty = sqrt(mean((pred - target)^2) + 1e-8) ("sqrt_mse"),
  final novelty = mean over the 3 layers, computed BEFORE the predictor
  update. One Adam(lr=1e-4) predictor step per scoring batch; the predictor
  trains on ALL scored candidates of the batch.
- Parameter init replays the exact numpy stream of tmx_jax.novelty
  (_init_dense_numpy: glorot-uniform kernels, zero biases; per layer target
  then predictor) seeded with seed + 17 so GPU and TPU RND targets match for
  the same base seed.

DEVIATIONS:
- The JAX path keeps one optax.adam state per layer; here a single
  torch.optim.Adam covers all predictor parameters. Adam state is
  element-wise and every layer is stepped exactly once per scoring batch, so
  the update trajectory is identical.
- HF models return hidden states with the embedding output FIRST, so entry k
  is the output of 0-indexed block k-1 (and the final entry is the
  post-final-norm state). We therefore index hidden_states[layer + 1]: the
  raw residual stream after block `layer` runs, matching the JAX model's
  return_pooled_layers=(7,14,21) semantics (tmx_jax/model.py pools AFTER
  layers_{i} executes, before the final norm).
- Gradient checkpointing is temporarily disabled on the policy during feature
  extraction (it runs under no_grad; the flag would only produce warnings).
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

DEFAULT_LAYERS = (7, 14, 21)
RND_SEED_OFFSET = 17


def _init_dense_numpy(rng: np.random.Generator, in_dim: int, out_dim: int) -> Dict[str, np.ndarray]:
    """Glorot-uniform kernel + zero bias; same draw order as tmx_jax.novelty."""
    limit = np.sqrt(6.0 / float(in_dim + out_dim))
    return {
        "kernel": rng.uniform(-limit, limit, size=(int(in_dim), int(out_dim))).astype(np.float32),
        "bias": np.zeros((int(out_dim),), dtype=np.float32),
    }


class RNDBlock(nn.Module):
    """Linear-ReLU-Linear-ReLU-Linear, fp32."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        return self.fc3(h)

    @torch.no_grad()
    def load_numpy_params(self, params: Dict[str, Dict[str, np.ndarray]]) -> None:
        for name, mod in (("fc1", self.fc1), ("fc2", self.fc2), ("fc3", self.fc3)):
            # numpy kernel is (in, out); torch Linear weight is (out, in).
            mod.weight.copy_(torch.from_numpy(params[name]["kernel"].T.copy()))
            mod.bias.copy_(torch.from_numpy(params[name]["bias"].copy()))


class MultilayerRND:
    """One frozen-target/trainable-predictor RND pair per policy layer."""

    def __init__(
        self,
        *,
        input_dim: int = 2048,
        hidden_dim: int = 512,
        layers: Sequence[int] = DEFAULT_LAYERS,
        lr: float = 1e-4,
        seed: int = 0,
        device: str = "cuda",
        metric: str = "sqrt_mse",
        clip_value: float = 0.0,
    ):
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.layers = tuple(int(x) for x in layers)
        self.lr = float(lr)
        self.seed = int(seed)
        self.device = torch.device(device)
        self.metric = str(metric or "sqrt_mse").lower()
        self.clip_value = float(clip_value or 0.0)
        self.targets: Dict[int, RNDBlock] = {}
        self.predictors: Dict[int, RNDBlock] = {}
        self.optimizer: torch.optim.Adam | None = None
        # C9 uses this counter as a crash/resume invariant: exactly one RND
        # update must accompany each completed Scout learner step.  Frozen C8
        # continues through score_and_update(), which deliberately does not
        # touch this new C9-only counter or alter its update semantics.
        self.selected_batch_update_count = 0
        self._build(self.seed)

    # ------------------------------------------------------------- build ----

    def _build(self, seed: int) -> None:
        rng = np.random.default_rng(int(seed) + RND_SEED_OFFSET)
        self.targets, self.predictors = {}, {}
        for layer in self.layers:
            # Same draw order as tmx_jax.novelty.init_rnd_map: per layer,
            # target(fc1,fc2,fc3) then predictor(fc1,fc2,fc3).
            t_np = {k: _init_dense_numpy(rng, self.input_dim if k == "fc1" else self.hidden_dim, self.hidden_dim)
                    for k in ("fc1", "fc2", "fc3")}
            p_np = {k: _init_dense_numpy(rng, self.input_dim if k == "fc1" else self.hidden_dim, self.hidden_dim)
                    for k in ("fc1", "fc2", "fc3")}
            target = RNDBlock(self.input_dim, self.hidden_dim).to(self.device)
            predictor = RNDBlock(self.input_dim, self.hidden_dim).to(self.device)
            target.load_numpy_params(t_np)
            predictor.load_numpy_params(p_np)
            target.requires_grad_(False)
            target.eval()
            self.targets[layer] = target
            self.predictors[layer] = predictor
        params = [p for layer in self.layers for p in self.predictors[layer].parameters()]
        self.optimizer = torch.optim.Adam(params, lr=self.lr)

    def reset(self, seed: int | None = None) -> None:
        """Fresh target/predictor/optimizer (multi-round: new RND each round)."""
        if seed is not None:
            self.seed = int(seed)
        self.selected_batch_update_count = 0
        self._build(self.seed)

    # ------------------------------------------------------------- score ----

    def _novelty_values(self, diff: torch.Tensor) -> torch.Tensor:
        mse = torch.mean(diff ** 2, dim=-1)
        if self.metric == "sqrt_mse":
            return torch.sqrt(mse + 1e-8)
        if self.metric == "l2":
            return torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-8)
        if self.metric == "l1":
            return torch.mean(torch.abs(diff), dim=-1)
        return mse

    def score(
        self,
        features_by_layer: Dict[int, torch.Tensor],
    ) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        """Score a batch without mutating predictor or optimizer state.

        C9 calls this for every dynamic-sampling candidate while holding the
        predictor fixed for the entire learner step.  Keeping this separate
        from :meth:`score_and_update` is intentional: the latter remains the
        frozen C8 implementation and therefore preserves historical behavior.
        """
        per_layer: Dict[int, np.ndarray] = {}
        with torch.no_grad():
            for layer in self.layers:
                feats = features_by_layer.get(layer)
                if feats is None:
                    continue
                feats = feats.to(self.device, dtype=torch.float32)
                target = self.targets[layer](feats)
                prediction = self.predictors[layer](feats)
                values = self._novelty_values(prediction - target)
                if self.clip_value > 0.0:
                    values = torch.minimum(
                        values, torch.full_like(values, self.clip_value))
                per_layer[layer] = (
                    values.detach().cpu().numpy().astype(np.float32))
        if not per_layer:
            return np.zeros((0,), dtype=np.float32), {}
        stacked = np.stack(
            [per_layer[layer] for layer in self.layers if layer in per_layer],
            axis=1,
        )
        return stacked.mean(axis=1).astype(np.float32), per_layer

    def update_selected_batch(
        self,
        features_by_layer: Dict[int, torch.Tensor],
    ) -> float:
        """Fit the predictor once on the final selected learner batch.

        Returns the summed per-layer MSE optimized by this single Adam step.
        The C9-only counter is persisted so a resumed Scout can prove that its
        predictor has advanced exactly once per completed policy update.
        """
        losses: List[torch.Tensor] = []
        for layer in self.layers:
            feats = features_by_layer.get(layer)
            if feats is None:
                continue
            feats = feats.to(self.device, dtype=torch.float32)
            with torch.no_grad():
                target = self.targets[layer](feats)
            prediction = self.predictors[layer](feats)
            losses.append(torch.mean((prediction - target) ** 2))
        if not losses or self.optimizer is None:
            raise ValueError("selected-batch RND update requires feature rows")
        total_loss = torch.stack(losses).sum()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        self.optimizer.step()
        self.selected_batch_update_count += 1
        return float(total_loss.detach().cpu())

    def score_and_update(
        self,
        features_by_layer: Dict[int, torch.Tensor],
        *,
        train: bool = True,
    ) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        """Return (final novelty (B,), per-layer novelty) for a scoring batch.

        Novelty is computed from the CURRENT prediction error, then a single
        Adam step fits the predictor to this batch (all candidates).
        """
        per_layer: Dict[int, np.ndarray] = {}
        losses: List[torch.Tensor] = []
        for layer in self.layers:
            feats = features_by_layer.get(layer)
            if feats is None:
                continue
            feats = feats.to(self.device, dtype=torch.float32)
            with torch.no_grad():
                t = self.targets[layer](feats)
            p = self.predictors[layer](feats)
            diff = p - t
            values = self._novelty_values(diff.detach())
            if self.clip_value > 0.0:
                values = torch.minimum(values, torch.full_like(values, self.clip_value))
            per_layer[layer] = values.detach().cpu().numpy().astype(np.float32)
            losses.append(torch.mean(diff ** 2))
        if not per_layer:
            return np.zeros((0,), dtype=np.float32), {}
        if train and losses and self.optimizer is not None:
            # Disjoint parameter sets per layer -> summing losses yields the
            # same per-layer gradients as stepping each layer independently.
            self.optimizer.zero_grad(set_to_none=True)
            torch.stack(losses).sum().backward()
            self.optimizer.step()
        stacked = np.stack([per_layer[layer] for layer in self.layers if layer in per_layer], axis=1)
        return stacked.mean(axis=1).astype(np.float32), per_layer

    # --------------------------------------------------------- state i/o ----

    def state_dict(self) -> Dict[str, Any]:
        meta: Dict[str, Any] = {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "layers": list(self.layers),
            "lr": self.lr,
            "seed": self.seed,
            "metric": self.metric,
            "clip_value": self.clip_value,
        }
        # Do not perturb frozen C8 state dictionaries.  This key is emitted
        # only after the opt-in C9 API has actually performed an update.
        if self.selected_batch_update_count:
            meta["selected_batch_update_count"] = int(
                self.selected_batch_update_count
            )
        return {
            "meta": meta,
            "targets": {str(k): v.state_dict() for k, v in self.targets.items()},
            "predictors": {str(k): v.state_dict() for k, v in self.predictors.items()},
            "optimizer": self.optimizer.state_dict() if self.optimizer is not None else {},
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        for key, sd in state.get("targets", {}).items():
            self.targets[int(key)].load_state_dict(sd)
        for key, sd in state.get("predictors", {}).items():
            self.predictors[int(key)].load_state_dict(sd)
        if state.get("optimizer") and self.optimizer is not None:
            self.optimizer.load_state_dict(state["optimizer"])
        self.selected_batch_update_count = int(
            state.get("meta", {}).get("selected_batch_update_count", 0)
        )

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path, map_location=self.device, weights_only=False))


# ------------------------------------------------------ feature extraction ----

def extract_novelty_features(
    model,
    tokenizer,
    texts: List[str],
    *,
    layers: Sequence[int] = DEFAULT_LAYERS,
    max_length: int = 18432,
    batch_size: int = 2,
    device: str = "cuda",
    autocast_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[Dict[int, torch.Tensor], Dict[str, float]]:
    """Mean-pooled hidden features of `texts` at `layers` from the live policy.

    Re-encodes the full raw completion text (add_special_tokens=True,
    truncation to max_length), runs the policy trunk under no_grad + bf16
    autocast, and mean-pools each requested layer over non-pad tokens.
    Returns ({layer: (N, H) fp32 tensor on device}, feature stats).
    """
    layers = tuple(int(x) for x in layers)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    trunk = getattr(model, "model", model)
    gc_was_enabled = bool(getattr(trunk, "gradient_checkpointing", False))
    if gc_was_enabled and hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    feats: Dict[int, List[torch.Tensor]] = {layer: [] for layer in layers}
    lengths: List[int] = []
    try:
        for start in range(0, len(texts), max(1, int(batch_size))):
            chunk = [t if t else " " for t in texts[start:start + max(1, int(batch_size))]]
            enc = tokenizer(
                chunk,
                add_special_tokens=True,
                truncation=True,
                max_length=int(max_length),
                padding=True,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            lengths.extend(int(x) for x in attention_mask.sum(dim=1).tolist())
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=autocast_dtype,
                                                 enabled=(torch.device(device).type == "cuda")):
                out = trunk(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                )
            # index 0 = embeddings, index L+1 = output of 0-indexed block L,
            # last index = post-final-norm state (never a valid RND feature).
            hidden_states = out.hidden_states
            mask = attention_mask.to(torch.float32).unsqueeze(-1)
            count = mask.sum(dim=1).clamp(min=1.0)
            for layer in layers:
                if layer + 1 >= len(hidden_states) - 1:
                    raise ValueError(
                        f"novelty layer {layer} out of range: model exposes "
                        f"{len(hidden_states) - 1} blocks and the final entry is "
                        "post-final-norm; pick layer <= n_blocks - 2"
                    )
                h = hidden_states[layer + 1].to(torch.float32)
                feats[layer].append((h * mask).sum(dim=1) / count)
    finally:
        if gc_was_enabled and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    stats = {
        "count": float(len(lengths)),
        "truncated_count": float(sum(1 for n in lengths if n >= int(max_length))),
        "token_length_sum": float(sum(lengths)),
        "max_input_tokens": float(max(lengths) if lengths else 0.0),
        "max_length": float(max_length),
    }
    return {layer: torch.cat(chunks, dim=0) for layer, chunks in feats.items()}, stats
