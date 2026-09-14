import os
import random
from pathlib import Path
import torch


def _is_xla_device(device) -> bool:
    if isinstance(device, torch.device):
        return device.type == "xla"
    if isinstance(device, str):
        return "xla" in device
    return False


def _single_worker_xla_collective_passthrough() -> bool:
    if os.environ.get("PJRT_DEVICE", "").upper() != "TPU":
        return False
    worker_hosts = (os.environ.get("TPU_WORKER_HOSTNAMES") or "").strip().lower()
    process_bounds = (os.environ.get("TPU_PROCESS_BOUNDS") or "").strip()
    chips_per_process = (os.environ.get("TPU_CHIPS_PER_PROCESS_BOUNDS") or "").strip()
    if worker_hosts == "localhost" and process_bounds == "1,1,1" and chips_per_process == "1,1,1":
        return True
    try:
        import torch_xla.runtime as xr  # type: ignore

        for attr_name in ("process_count", "local_process_count"):
            fn = getattr(xr, attr_name, None)
            if fn is None:
                continue
            try:
                process_count = int(fn())
            except Exception:
                process_count = 0
            if process_count > 1:
                return False
            if process_count == 1:
                return True
    except Exception:
        pass
    try:
        world_size = int(os.environ.get("WORLD_SIZE") or os.environ.get("LOCAL_WORLD_SIZE") or "1")
    except ValueError:
        world_size = 1
    return world_size <= 1


def _xla_metric_cpu_max_numel() -> int:
    raw = (os.environ.get("EXPDIS_XLA_METRIC_CPU_MAX_NUMEL") or "").strip()
    if not raw:
        return 65536
    try:
        return max(1, int(raw))
    except Exception:
        return 65536


def _xla_runtime_device_count() -> int:
    if os.environ.get("PJRT_DEVICE", "").upper() != "TPU":
        return 1
    try:
        import torch_xla.runtime as xr  # type: ignore

        for attr_name in ("global_runtime_device_count", "addressable_runtime_device_count", "world_size"):
            fn = getattr(xr, attr_name, None)
            if fn is None:
                continue
            try:
                value = int(fn())
            except Exception:
                value = 0
            if value > 0:
                return value
    except Exception:
        pass
    return 1


def _move_small_metrics_to_cpu(data, max_numel: int):
    if torch.is_tensor(data):
        if data.device.type != "xla":
            return data
        if int(data.numel()) > max_numel:
            return data
        return data.detach().to("cpu")
    if isinstance(data, dict):
        return {key: _move_small_metrics_to_cpu(value, max_numel) for key, value in data.items()}
    if isinstance(data, tuple):
        return tuple(_move_small_metrics_to_cpu(value, max_numel) for value in data)
    if isinstance(data, list):
        return [_move_small_metrics_to_cpu(value, max_numel) for value in data]
    return data


def _all_xla_metric_tensors_fit_cpu_handoff(data, max_numel: int) -> bool:
    if torch.is_tensor(data):
        return data.device.type != "xla" or int(data.numel()) <= max_numel
    if isinstance(data, dict):
        return all(_all_xla_metric_tensors_fit_cpu_handoff(value, max_numel) for value in data.values())
    if isinstance(data, tuple):
        return all(_all_xla_metric_tensors_fit_cpu_handoff(value, max_numel) for value in data)
    if isinstance(data, list):
        return all(_all_xla_metric_tensors_fit_cpu_handoff(value, max_numel) for value in data)
    return True


def _xla_cpu_input_handoff_enabled() -> bool:
    return (os.environ.get("EXPDIS_XLA_CPU_INPUT_HANDOFF", "1") or "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _materialize_tensor_for_cpu_handoff(tensor):
    if tensor is None or not torch.is_tensor(tensor):
        return tensor
    return tensor.detach().cpu().clone()


def _rehydrate_tensor_to_device(tensor, device):
    if tensor is None or not torch.is_tensor(tensor):
        return tensor
    if isinstance(device, str):
        device = torch.device(device)
    if tensor.device == device:
        return tensor
    return tensor.to(device)


def _hf_hub_cache_root_candidates():
    roots = []
    hub_cache = (os.environ.get("HF_HUB_CACHE") or "").strip()
    if hub_cache:
        roots.append(Path(hub_cache))
    hf_home = (os.environ.get("HF_HOME") or "").strip()
    if hf_home:
        roots.append(Path(hf_home) / "hub")
    roots.append(Path(".hf") / "hub")
    seen = set()
    unique = []
    for root in roots:
        resolved = root.expanduser()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        unique.append(resolved)
    return unique


def _hf_repo_cached(repo_id: str, repo_type: str = "model") -> bool:
    if not repo_id or os.path.isdir(repo_id):
        return False
    prefix = "models" if repo_type == "model" else "datasets"
    repo_dirname = f"{prefix}--{repo_id.replace('/', '--')}"
    for root in _hf_hub_cache_root_candidates():
        repo_dir = root / repo_dirname
        if not repo_dir.exists():
            continue
        refs_main = repo_dir / "refs" / "main"
        snapshots = repo_dir / "snapshots"
        if refs_main.exists() or any(snapshots.glob("*")):
            return True
    return False


def hf_from_pretrained_kwargs(repo_id: str, repo_type: str = "model"):
    if (os.environ.get("EXPDIS_HF_LOCAL_ONLY") or "").strip().lower() in ("1", "true", "yes"):
        return {"local_files_only": True}
    if (os.environ.get("EXPDIS_HF_LOCAL_IF_CACHED", "1") or "").strip().lower() in ("0", "false", "no"):
        return {}
    if not _hf_repo_cached(repo_id, repo_type=repo_type):
        return {}
    print(f"[hf-cache] Using local cache for {repo_type} {repo_id}.")
    return {"local_files_only": True}


def patch_torch_autocast_enabled_signature():
    """
    Compatibility shim for older torch versions where is_autocast_enabled()
    does not accept a device_type argument.
    """
    fn = getattr(torch, "is_autocast_enabled", None)
    if fn is None:
        return
    if getattr(fn, "_expdis_accepts_device_type", False):
        return

    accepts_device_type = False
    try:
        fn("cpu")
        accepts_device_type = True
    except TypeError:
        accepts_device_type = False
    except Exception:
        # If it accepts a device argument but runtime state is unusual, avoid patching.
        accepts_device_type = True

    if accepts_device_type:
        return

    original = fn

    def _wrapped_is_autocast_enabled(device_type=None):  # noqa: ARG001
        return original()

    _wrapped_is_autocast_enabled._expdis_accepts_device_type = True  # type: ignore[attr-defined]
    torch.is_autocast_enabled = _wrapped_is_autocast_enabled  # type: ignore[assignment]
    print("[patch] torch.is_autocast_enabled patched for backward compatibility.")


def causal_lm_from_pretrained(model_name: str, *, is_xla: bool = False, **kwargs):
    """
    TPU/XLA runs can OOM in the SDPA masking path for some decoder models.
    Prefer eager attention on XLA when available, and fall back cleanly if the
    installed Transformers build or model class does not accept the flag.
    """
    from transformers import AutoModelForCausalLM

    load_kwargs = {**hf_from_pretrained_kwargs(model_name, repo_type="model"), **dict(kwargs)}
    # TPU flash attention: eliminates O(N²) memory, enabling 8k+ training on single chip.
    use_flash = is_xla and os.environ.get("EXPDIS_XLA_FLASH_ATTENTION", "0") not in ("0", "false", "no", "")
    if use_flash and _register_tpu_flash_attention():
        # Transformers 5.x validates attn_implementation against known names before
        # checking ALL_ATTENTION_FUNCTIONS, so we load with "eager" then swap to
        # "tpu_flash" in the config so the forward pass dispatches to our kernel.
        load_kwargs.setdefault("attn_implementation", "eager")
        load_kwargs.setdefault("torch_dtype", torch.bfloat16)
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
            model.config._attn_implementation = "tpu_flash"
            model.config._attn_implementation_internal = "tpu_flash"
            _patch_model_bf16_for_flash_attention(model)
            print("[patch] Loading causal LM with attn_implementation=tpu_flash, dtype=bf16 for XLA.")
            return model
        except Exception as e:
            load_kwargs.pop("attn_implementation", None)
            load_kwargs.pop("dtype", None)
            print(f"[WARN] TPU flash attention load failed; falling back. ({e})")

    force_eager = is_xla and os.environ.get("EXPDIS_XLA_FORCE_EAGER_ATTN", "1") != "0"
    if force_eager:
        load_kwargs.setdefault("attn_implementation", "eager")
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
            print("[patch] Loading causal LM with attn_implementation=eager for XLA.")
            return model
        except TypeError:
            load_kwargs.pop("attn_implementation", None)
        except Exception as e:
            load_kwargs.pop("attn_implementation", None)
            print(f"[WARN] Eager attention load failed on XLA; retrying default attention. ({e})")
    return AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)


_TPU_FLASH_REGISTERED = False


def _register_tpu_flash_attention():
    """Register TPU flash attention in HuggingFace's ALL_ATTENTION_FUNCTIONS.

    Uses torch_xla Pallas flash attention kernel to eliminate O(N²) attention
    memory, enabling 8k+ training on a single TPU chip.  Requires JAX.
    """
    global _TPU_FLASH_REGISTERED
    if _TPU_FLASH_REGISTERED:
        return True
    try:
        from torch_xla.experimental.custom_kernel import flash_attention as _xla_flash
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except ImportError as e:
        print(f"[WARN] TPU flash attention unavailable: {e}")
        return False

    _FLASH_BLOCK = 1024  # Pallas kernel requires seq_len divisible by this (fwd=512, bwd=1024)

    def tpu_flash_attention_forward(
        module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs
    ):
        # GQA: expand KV heads to match query heads
        num_kv_groups = getattr(module, "num_key_value_groups", 1)
        if num_kv_groups > 1:
            bsz, num_kv_heads, slen, head_dim = key.shape
            key = key[:, :, None, :, :].expand(
                bsz, num_kv_heads, num_kv_groups, slen, head_dim
            ).reshape(bsz, num_kv_heads * num_kv_groups, slen, head_dim)
            value = value[:, :, None, :, :].expand(
                bsz, num_kv_heads, num_kv_groups, slen, head_dim
            ).reshape(bsz, num_kv_heads * num_kv_groups, slen, head_dim)

        q = query.to(torch.bfloat16).contiguous()
        k = key.to(torch.bfloat16).contiguous()
        v = value.to(torch.bfloat16).contiguous()

        q_len = q.shape[2]
        kv_len = k.shape[2]

        # During generation with KV cache, Q and K have different lengths.
        # Fall back to eager SDPA — Pallas kernel requires both dimensions
        # block-aligned and doesn't handle the mismatch well.
        if q_len != kv_len:
            out = torch.nn.functional.scaled_dot_product_attention(
                q.float(), k.float(), v.float(), is_causal=(q_len > 1),
                scale=scaling,
            ).to(torch.bfloat16)
            return out.transpose(1, 2).contiguous(), None

        if q_len % _FLASH_BLOCK != 0:
            pad_len = (-q_len) % _FLASH_BLOCK
            _z = torch.zeros(q.shape[0], q.shape[1], pad_len, q.shape[3],
                             dtype=torch.bfloat16, device=q.device)
            q = torch.cat([q, _z], dim=2)
            k = torch.cat([k, _z], dim=2)
            v = torch.cat([v, _z], dim=2)
            out = _xla_flash(q, k, v, causal=True, sm_scale=scaling)
            out = out[:, :, :q_len, :]
        else:
            out = _xla_flash(q, k, v, causal=True, sm_scale=scaling)

        return out.transpose(1, 2).contiguous(), None

    ALL_ATTENTION_FUNCTIONS["tpu_flash"] = tpu_flash_attention_forward
    _TPU_FLASH_REGISTERED = True
    print("[patch] TPU flash attention registered (Pallas kernel).")
    return True


def _patch_model_bf16_for_flash_attention(model):
    """Force RMSNorm and RotaryEmbedding to stay in bf16.

    The Pallas flash attention backward kernel requires ALL operands in bf16.
    Standard HF implementations promote to f32 for numerical stability
    (RMSNorm.forward casts to f32, RotaryEmbedding computes freqs in f32).
    These f32 intermediates create f32 gradients in the backward graph that
    cause the Pallas kernel to fail with 'XLA layout does not match MLIR layout'.

    Patching these to stay in bf16 eliminates f32 from the entire graph.
    """
    patched = []

    # Patch RMSNorm: remove .to(torch.float32) promotion
    try:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
        def _bf16_rms_norm_forward(self, hidden_states):
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            return self.weight * hidden_states

        for module in model.modules():
            if isinstance(module, Qwen3RMSNorm):
                module.forward = _bf16_rms_norm_forward.__get__(module, type(module))
        patched.append("RMSNorm")
    except ImportError:
        pass

    # Patch RotaryEmbedding: compute frequencies in bf16 instead of f32
    try:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
        _orig_rotary_forward = Qwen3RotaryEmbedding.forward

        def _bf16_rotary_forward(self, x, position_ids):
            inv_freq_expanded = self.inv_freq[None, :, None].to(x.dtype).expand(
                position_ids.shape[0], -1, 1
            ).to(x.device)
            position_ids_expanded = position_ids[:, None, :].to(x.dtype)
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
            return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

        Qwen3RotaryEmbedding.forward = _bf16_rotary_forward
        patched.append("RotaryEmbedding")
    except ImportError:
        pass

    if patched:
        print(f"[patch] {', '.join(patched)} forced to bf16 for flash attention.", flush=True)


def enable_tpu_flash_attention_on_model(model):
    """Switch an already-loaded model to use TPU flash attention."""
    if _register_tpu_flash_attention():
        model.config._attn_implementation = "tpu_flash"
        print("[patch] Model attention switched to tpu_flash.")
    return model


def patch_torch_checkpoint_autocast_for_xla():
    """
    torch.utils.checkpoint on some torch builds tries to inspect an `xla` device
    module that does not exist under `torch`, causing checkpointed XLA forwards
    to fail before the actual model step.

    For TPU runs we do not need XLA autocast kwargs here, so return `None` for
    the device-specific autocast block and preserve the CPU block.
    """
    try:
        import torch.utils.checkpoint as checkpoint_mod
    except Exception as e:
        print(f"[WARN] Could not patch torch checkpoint autocast for XLA: {e}")
        return

    if getattr(checkpoint_mod, "_expdis_xla_autocast_patched", False):
        return

    original_get_autocast_kwargs = getattr(checkpoint_mod, "_get_autocast_kwargs", None)
    if original_get_autocast_kwargs is not None:

        def _get_autocast_kwargs_xla_safe(device_type="cuda"):
            if str(device_type).lower() == "xla":
                device_autocast_kwargs = {
                    "enabled": False,
                    "dtype": None,
                    "cache_enabled": False,
                }
                cpu_autocast_kwargs = {
                    "enabled": torch.is_autocast_enabled("cpu"),
                    "dtype": torch.get_autocast_dtype("cpu"),
                    "cache_enabled": torch.is_autocast_cache_enabled(),
                }
                return device_autocast_kwargs, cpu_autocast_kwargs
            return original_get_autocast_kwargs(device_type)

        checkpoint_mod._get_autocast_kwargs = _get_autocast_kwargs_xla_safe

    original_get_device_module = getattr(checkpoint_mod, "_get_device_module", None)
    if original_get_device_module is not None:

        class _XLADeviceModuleStub:
            _initialized = False

        def _get_device_module_xla_safe(device="cuda"):
            if str(device).lower() == "xla":
                # torch.utils.checkpoint expects a torch.<device> module, but
                # torch has no torch.xla attribute. Returning an uninitialized
                # stub keeps RNG-state preservation on the CPU path only.
                return _XLADeviceModuleStub()
            return original_get_device_module(device)

        checkpoint_mod._get_device_module = _get_device_module_xla_safe

    original_supports_autocast = getattr(checkpoint_mod, "_supports_autocast", None)
    if original_supports_autocast is not None:

        def _supports_autocast_xla_safe(device):
            if str(device).lower() == "xla":
                return False
            return original_supports_autocast(device)

        checkpoint_mod._supports_autocast = _supports_autocast_xla_safe

    checkpoint_mod._expdis_xla_autocast_patched = True
    print("[patch] torch.utils.checkpoint autocast patched for XLA.")


def patch_torch_xla_device_module_for_rng():
    """
    torch.random.fork_rng() expects a registered torch.<device> module.
    Some torch/torch_xla builds expose XLA RNG helpers via torch_xla.core.xla_model
    but do not register torch.xla, which breaks activation checkpoint backward.
    """
    if getattr(getattr(torch, "xla", None), "_expdis_rng_registered", False):
        return

    try:
        import torch_xla.core.xla_model as xm
    except Exception as e:
        print(f"[WARN] Could not register torch.xla RNG module: {e}")
        return

    def _normalize_xla_device(device):
        if device is None:
            return None
        if isinstance(device, int):
            return f"xla:{device}"
        return str(device)

    def _visible_xla_device_count() -> int:
        visible = (os.environ.get("TPU_VISIBLE_CHIPS") or "").strip()
        if visible:
            chips = [chip.strip() for chip in visible.split(",") if chip.strip()]
            if chips:
                return len(chips)
        for env_key in ("LOCAL_WORLD_SIZE", "WORLD_SIZE"):
            raw = (os.environ.get(env_key) or "").strip()
            if raw:
                try:
                    value = int(raw)
                except ValueError:
                    value = 0
                if value > 0:
                    return value
        try:
            return max(1, int(xm.xrt_world_size()))
        except Exception:
            return 1

    class _TorchXLADeviceModule:
        _initialized = True
        _expdis_rng_registered = True

        @staticmethod
        def device_count():
            return _visible_xla_device_count()

        @staticmethod
        def get_rng_state(device=None):
            return xm.get_rng_state(device=_normalize_xla_device(device))

        @staticmethod
        def set_rng_state(seed, device=None):
            return xm.set_rng_state(int(seed), device=_normalize_xla_device(device))

    device_module = _TorchXLADeviceModule()
    try:
        torch.xla = device_module  # type: ignore[attr-defined]
    except Exception:
        pass

    register_fn = getattr(torch, "_register_device_module", None)
    if callable(register_fn):
        try:
            register_fn("xla", device_module)
        except RuntimeError:
            pass
        except Exception as e:
            print(f"[WARN] torch._register_device_module('xla', ...) failed: {e}")

    print("[patch] torch.xla RNG device module registered for checkpoint support.")


def patch_transformers_isin_for_xla():
    """
    Avoid torch.isin on XLA (TPU) since it can trigger CPU fallback/compile OOM.
    Uses broadcasted equality instead.
    """
    try:
        import importlib
        import transformers.pytorch_utils as pu
    except Exception as e:
        print(f"[WARN] Could not patch transformers isin (import failed): {e}")
        return

    if not hasattr(pu, "isin_mps_friendly"):
        return

    original = pu.isin_mps_friendly

    def isin_xla_friendly(elements: torch.Tensor, test_elements):
        if isinstance(elements, torch.Tensor) and elements.device.type == "xla":
            if not torch.is_tensor(test_elements):
                test_elements = torch.tensor(test_elements, device=elements.device)
            else:
                test_elements = test_elements.to(elements.device)

            if test_elements.numel() <= 1:
                return elements == test_elements
            # On XLA, avoid large membership reductions (can blow vmem in compile).
            return torch.zeros_like(elements, dtype=torch.bool)
        return original(elements, test_elements)

    pu.isin_mps_friendly = isin_xla_friendly

    # Patch any modules that imported isin_mps_friendly directly.
    for module_name in (
        "transformers.generation.utils",
        "transformers.generation.stopping_criteria",
        "transformers.generation.logits_process",
        "transformers.generation.candidate_generator",
    ):
        try:
            mod = importlib.import_module(module_name)
        except Exception:
            continue
        if hasattr(mod, "isin_mps_friendly"):
            setattr(mod, "isin_mps_friendly", isin_xla_friendly)

    print("[patch] transformers.isin_mps_friendly patched for XLA.")


def patch_torch_isin_for_xla():
    """
    Monkey-patch torch.isin to avoid XLA reduce-window lowering.
    """
    original = torch.isin

    def isin_xla(elements, test_elements):
        if isinstance(elements, torch.Tensor) and elements.device.type == "xla":
            if not torch.is_tensor(test_elements):
                test_elements = torch.tensor(test_elements, device=elements.device)
            else:
                test_elements = test_elements.to(elements.device)

            if test_elements.numel() <= 1:
                return elements == test_elements
            return torch.zeros_like(elements, dtype=torch.bool)
        return original(elements, test_elements)

    torch.isin = isin_xla
    print("[patch] torch.isin patched for XLA.")


def patch_transformers_attention_mask_for_xla():
    """
    Replace GenerationMixin._prepare_attention_mask_for_generation on XLA to avoid
    torch.isin + boolean conversion (can trigger CPU fallback + compile OOM).
    """
    try:
        from transformers.generation.utils import GenerationMixin
    except Exception as e:
        print(f"[WARN] Could not patch attention mask for XLA: {e}")
        return

    original = GenerationMixin._prepare_attention_mask_for_generation

    def _prepare_attention_mask_for_generation_xla(self, inputs_tensor, generation_config, model_kwargs):
        if not isinstance(inputs_tensor, torch.Tensor) or inputs_tensor.device.type != "xla":
            return original(self, inputs_tensor, generation_config, model_kwargs)

        pad_token_id = generation_config._pad_token_tensor
        if "input_ids" in model_kwargs and model_kwargs["input_ids"].shape[1] > 0:
            inputs_tensor = model_kwargs["input_ids"]

        default_attention_mask = torch.ones(inputs_tensor.shape[:2], dtype=torch.long, device=inputs_tensor.device)
        if pad_token_id is None:
            return default_attention_mask

        is_input_ids = len(inputs_tensor.shape) == 2 and inputs_tensor.dtype in [torch.int, torch.long]
        if not is_input_ids:
            return default_attention_mask

        if torch.is_tensor(pad_token_id):
            pad_token_id = pad_token_id.to(inputs_tensor.device)
        return inputs_tensor.ne(pad_token_id).long()

    GenerationMixin._prepare_attention_mask_for_generation = _prepare_attention_mask_for_generation_xla
    print("[patch] GenerationMixin._prepare_attention_mask_for_generation patched for XLA.")


def patch_transformers_logits_processor_for_xla():
    """
    Disable suppress/bad-words processors on XLA to avoid large `isin` membership masks.
    """
    try:
        from transformers.generation.utils import GenerationMixin
    except Exception as e:
        print(f"[WARN] Could not patch logits processor for XLA: {e}")
        return

    original = GenerationMixin._get_logits_processor

    def _get_logits_processor_xla(self, *args, **kwargs):
        device = kwargs.get("device")
        if device is None and len(args) > 5:
            device = args[5]

        if not _is_xla_device(device):
            return original(self, *args, **kwargs)

        generation_config = kwargs.get("generation_config")
        if generation_config is None and args:
            generation_config = args[0]
        if generation_config is not None:
            generation_config.suppress_tokens = None
            generation_config.begin_suppress_tokens = None
            generation_config.bad_words_ids = None

        from transformers.generation.logits_process import LogitsProcessorList

        logits_processor = kwargs.get("logits_processor")
        if logits_processor is None and len(args) > 4:
            logits_processor = args[4]

        if logits_processor is None:
            return LogitsProcessorList()
        if isinstance(logits_processor, LogitsProcessorList):
            return logits_processor
        return LogitsProcessorList(logits_processor)

    GenerationMixin._get_logits_processor = _get_logits_processor_xla
    print("[patch] GenerationMixin._get_logits_processor patched for XLA.")


def patch_transformers_stopping_criteria_for_xla():
    """
    Drop EOS-based stopping on XLA to avoid `isin`-style membership checks.
    """
    try:
        from transformers.generation.utils import GenerationMixin
        from transformers.generation.stopping_criteria import (
            EosTokenCriteria,
            StoppingCriteriaList,
        )
    except Exception as e:
        print(f"[WARN] Could not patch stopping criteria for XLA: {e}")
        return

    original = GenerationMixin._get_stopping_criteria

    def _get_stopping_criteria_xla(self, generation_config, stopping_criteria=None, tokenizer=None):
        criteria = original(self, generation_config, stopping_criteria, tokenizer)
        pad_tensor = getattr(generation_config, "_pad_token_tensor", None)
        eos_tensor = getattr(generation_config, "_eos_token_tensor", None)
        if _is_xla_device(getattr(pad_tensor, "device", None)) or _is_xla_device(getattr(eos_tensor, "device", None)):
            filtered = StoppingCriteriaList(
                [c for c in criteria if not isinstance(c, EosTokenCriteria)]
            )
            return filtered
        return criteria

    GenerationMixin._get_stopping_criteria = _get_stopping_criteria_xla
    print("[patch] GenerationMixin._get_stopping_criteria patched for XLA.")


def patch_accelerate_gather_object_for_xla():
    """
    TRL GRPO uses accelerate.gather_object for prompt/completion logging.
    On TPU this raises NotImplementedError. For single-process TPU runs, gathering
    is a no-op, so return the local object directly.
    """
    try:
        import accelerate.utils.operations as accel_ops
        from accelerate import Accelerator
    except Exception as e:
        print(f"[WARN] Could not patch accelerate.gather_object for XLA: {e}")
        return

    current = getattr(accel_ops, "gather_object", None)
    if current is None:
        return
    if getattr(current, "_expdis_xla_safe", False):
        return

    original = current

    def gather_object_xla_safe(obj):
        if _single_worker_xla_collective_passthrough():
            # Single-process TPU execution doesn't need a cross-process gather.
            return obj
        try:
            return original(obj)
        except NotImplementedError as e:
            if "TPU" in str(e):
                return obj
            raise

    gather_object_xla_safe._expdis_xla_safe = True  # type: ignore[attr-defined]
    accel_ops.gather_object = gather_object_xla_safe

    current_gather = getattr(Accelerator, "gather", None)
    if current_gather is not None and not getattr(current_gather, "_expdis_xla_single_worker_safe", False):

        def gather_xla_safe(self, tensor, *args, **kwargs):
            if _single_worker_xla_collective_passthrough():
                return tensor
            return current_gather(self, tensor, *args, **kwargs)

        gather_xla_safe._expdis_xla_single_worker_safe = True  # type: ignore[attr-defined]
        Accelerator.gather = gather_xla_safe

    # TRL imports gather_object directly; patch its module-level alias too.
    try:
        import trl.trainer.grpo_trainer as grpo_mod

        if hasattr(grpo_mod, "gather_object"):
            grpo_mod.gather_object = gather_object_xla_safe
    except Exception:
        pass

    current_gather_for_metrics = getattr(Accelerator, "gather_for_metrics", None)
    if current_gather_for_metrics is not None and not getattr(current_gather_for_metrics, "_expdis_xla_single_worker_safe", False):

        def _metric_placeholder_cpu(data):
            if torch.is_tensor(data):
                return torch.zeros(tuple(data.shape), dtype=data.dtype, device="cpu")
            if isinstance(data, dict):
                return {key: _metric_placeholder_cpu(value) for key, value in data.items()}
            if isinstance(data, tuple):
                return tuple(_metric_placeholder_cpu(value) for value in data)
            if isinstance(data, list):
                return [_metric_placeholder_cpu(value) for value in data]
            return data

        def gather_for_metrics_xla_safe(self, input_data, use_gather_object=False):
            max_numel = _xla_metric_cpu_max_numel()
            local_metric_handoff = (
                not use_gather_object
                and os.environ.get("EXPDIS_XLA_LOCAL_METRIC_HANDOFF", "1") in ("1", "true", "True", "yes", "YES")
                and _all_xla_metric_tensors_fit_cpu_handoff(input_data, max_numel)
            )
            if local_metric_handoff:
                # These tensors are used for logging / bookkeeping only. On TPU,
                # even scalar metric gathers can force large executable materialization
                # on the multi-process DDP path. Keep them local and move them to CPU.
                return _move_small_metrics_to_cpu(input_data, max_numel)
            if _single_worker_xla_collective_passthrough() and not use_gather_object:
                # Default to passthrough so GRPO logging records real completion
                # lengths and token counts, but move small metric tensors to the
                # host so later `.item()` / reductions do not re-enter XLA.
                # Retain the old placeholder behavior behind an env flag for
                # emergency TPU debugging only.
                if os.environ.get("EXPDIS_XLA_PLACEHOLDER_METRICS", "0") in ("1", "true", "True", "yes", "YES"):
                    return _metric_placeholder_cpu(input_data)
                return _move_small_metrics_to_cpu(input_data, max_numel)
            return current_gather_for_metrics(self, input_data, use_gather_object=use_gather_object)

        gather_for_metrics_xla_safe._expdis_xla_single_worker_safe = True  # type: ignore[attr-defined]
        Accelerator.gather_for_metrics = gather_for_metrics_xla_safe

    print("[patch] accelerate gather helpers patched for XLA.")


def patch_transformers_nested_xla_mesh_reduce_for_single_worker():
    """
    transformers.Trainer still routes metric/logging gathers through
    nested_xla_mesh_reduce on TPU. For single-worker TPU runs this collective is
    unnecessary and can trigger an avoidable XLA sync/allocation spike during
    logging and checkpoint save steps.
    """
    try:
        import transformers.trainer_pt_utils as trainer_pt_utils
        import transformers.trainer as trainer_mod
    except Exception as e:
        print(f"[WARN] Could not patch transformers nested_xla_mesh_reduce for XLA: {e}")
        return

    original = getattr(trainer_pt_utils, "nested_xla_mesh_reduce", None)
    if original is None:
        return
    if getattr(original, "_expdis_xla_single_worker_safe", False):
        return

    def nested_xla_mesh_reduce_single_worker(tensors, name):
        if _single_worker_xla_collective_passthrough():
            return tensors
        return original(tensors, name)

    nested_xla_mesh_reduce_single_worker._expdis_xla_single_worker_safe = True  # type: ignore[attr-defined]
    trainer_pt_utils.nested_xla_mesh_reduce = nested_xla_mesh_reduce_single_worker
    if hasattr(trainer_mod, "nested_xla_mesh_reduce"):
        trainer_mod.nested_xla_mesh_reduce = nested_xla_mesh_reduce_single_worker

    print("[patch] transformers nested_xla_mesh_reduce patched for single-worker XLA.")


def patch_accelerate_gradient_state_for_xla():
    """
    Patch Trainer.training_step to call mark_step() after every step on the
    single-worker XLA path.

    GRPOTrainer creates many XLA tensors during generation/scoring, then the
    loss backward frees intermediate autograd tensors. If mark_step() is delayed
    until the next iteration (accelerate's _set_sync_gradients), the XLA lazy
    graph references deleted tensors → ``Check failed: tensor_data``.

    Fix: flush the XLA graph right after each training_step returns, while all
    tensors from that step are still alive.

    Important: this is only safe/necessary on the single-worker XLA path. On
    multi-process DDP TPU runs, flushing inside training_step leaves the outer
    Trainer loss accumulator holding stale XLA scalars before
    `_inner_training_loop` does `tr_loss = tr_loss + tr_loss_step`.
    """
    try:
        from transformers import Trainer
    except Exception:
        return
    try:
        import accelerate.state as accelerate_state
    except Exception:
        accelerate_state = None

    original = getattr(Trainer, "training_step", None)
    if original is None or getattr(original, "_expdis_patched", False):
        pass
    else:
        def training_step_with_mark_step(self, model, inputs, num_items_in_batch=None):
            loss = original(self, model, inputs, num_items_in_batch=num_items_in_batch)
            if not _single_worker_xla_collective_passthrough():
                return loss
            placeholder_step_loss = False
            placeholder_dtype = None
            placeholder_device = None
            if (
                torch.is_tensor(loss)
                and loss.device.type == "xla"
                and os.environ.get("EXPDIS_XLA_PLACEHOLDER_STEP_LOSS", "0") in ("1", "true", "True", "yes", "YES")
            ):
                # Trainer expects a tensor on args.device for host-side bookkeeping
                # after backward. We cannot safely return the pre-mark_step XLA loss
                # handle, so record enough metadata to create a fresh placeholder
                # scalar after the graph flush.
                placeholder_step_loss = True
                placeholder_dtype = loss.dtype
                placeholder_device = loss.device
            # Flush the XLA graph immediately so no stale tensor handles
            # survive to the next iteration's mark_step.
            try:
                import torch_xla.core.xla_model as _xm
                _xm.mark_step()
            except Exception:
                pass
            if placeholder_step_loss:
                return torch.zeros((), device=placeholder_device, dtype=placeholder_dtype)
            return loss

        training_step_with_mark_step._expdis_patched = True  # type: ignore[attr-defined]
        Trainer.training_step = training_step_with_mark_step
        print("[patch] Trainer.training_step patched with post-step mark_step for XLA.")

    if accelerate_state is not None:
        original_set_sync_gradients = getattr(accelerate_state.GradientState, "_set_sync_gradients", None)
        if original_set_sync_gradients is not None and not getattr(original_set_sync_gradients, "_expdis_patched", False):
            def _set_sync_gradients_xla_safe(self, sync_gradients):
                skip_xla_mark_step = (
                    _single_worker_xla_collective_passthrough()
                    and sync_gradients
                    and os.environ.get("USE_TORCH_XLA") == "1"
                    and os.environ.get("EXPDIS_XLA_SKIP_ACCELERATE_SYNC_MARK_STEP", "0")
                    in ("1", "true", "True", "yes", "YES")
                )
                if skip_xla_mark_step:
                    self.sync_gradients = sync_gradients
                    return
                return original_set_sync_gradients(self, sync_gradients)

            _set_sync_gradients_xla_safe._expdis_patched = True  # type: ignore[attr-defined]
            accelerate_state.GradientState._set_sync_gradients = _set_sync_gradients_xla_safe
            print("[patch] accelerate GradientState._set_sync_gradients patched to skip duplicate XLA mark_step.")


def patch_trl_grpo_compute_loss_for_xla():
    """
    On the single-process XLA SPMD path, GRPOTrainer buffers the tensors
    returned by _generate_and_score_completions() across accumulation steps.
    Keeping those buffered tensors resident on XLA can leave TRL with stale
    handles by the time _compute_loss() first touches prompt_ids/completion_ids.

    Fix: hand those buffered tensors back as CPU tensors, then rehydrate them
    onto the live XLA device immediately before TRL computes the loss.
    """
    try:
        from trl.trainer.grpo_trainer import GRPOTrainer
    except Exception as e:
        print(f"[WARN] Could not patch TRL GRPO compute_loss for XLA: {e}")
        return

    original = getattr(GRPOTrainer, "_compute_loss", None)
    if original is None or getattr(original, "_expdis_xla_cpu_handoff", False):
        return

    def _compute_loss_xla_safe(self, model, inputs):
        if (
            _single_worker_xla_collective_passthrough()
            and _is_xla_device(getattr(self.accelerator, "device", None))
            and _xla_cpu_input_handoff_enabled()
        ):
            target_device = self.accelerator.device
            remapped = dict(inputs)
            for key in (
                "prompt_ids",
                "prompt_mask",
                "completion_ids",
                "completion_mask",
                "advantages",
                "old_per_token_logps",
                "ref_per_token_logps",
            ):
                if key in remapped:
                    remapped[key] = _rehydrate_tensor_to_device(remapped[key], target_device)
            inputs = remapped
        return original(self, model, inputs)

    _compute_loss_xla_safe._expdis_xla_cpu_handoff = True  # type: ignore[attr-defined]
    GRPOTrainer._compute_loss = _compute_loss_xla_safe
    print("[patch] GRPOTrainer._compute_loss patched for XLA CPU->device handoff.")


def patch_transformers_accelerator_num_processes_for_xla_spmd():
    """
    In single-process XLA SPMD, Accelerate may still report num_processes=1 even
    though the runtime exposes multiple addressable TPU devices. TRL's GRPO
    constructor validates `num_generations` against `accelerator.num_processes`,
    so override the Trainer-owned Accelerator state to match the XLA runtime
    device count before GRPOTrainer performs that divisibility check.
    """
    try:
        from transformers import Trainer
    except Exception as e:
        print(f"[WARN] Could not patch Trainer accelerator num_processes for XLA SPMD: {e}")
        return

    original = getattr(Trainer, "create_accelerator_and_postprocess", None)
    if original is None or getattr(original, "_expdis_xla_spmd_num_processes", False):
        return

    def create_accelerator_and_postprocess_xla_spmd(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if _single_worker_xla_collective_passthrough() and (os.environ.get("EXPDIS_FSDP") or "").strip().lower() in ("1", "true", "full_shard"):
            runtime_count = _xla_runtime_device_count()
            if runtime_count > 1:
                try:
                    self.accelerator.state.num_processes = runtime_count
                    print(
                        "[patch] Trainer accelerator num_processes forced to "
                        f"XLA runtime device count={runtime_count} for single-process SPMD."
                    )
                except Exception as exc:
                    print(f"[WARN] Failed to override Trainer accelerator num_processes for XLA SPMD: {exc}")
        return result

    create_accelerator_and_postprocess_xla_spmd._expdis_xla_spmd_num_processes = True  # type: ignore[attr-defined]
    Trainer.create_accelerator_and_postprocess = create_accelerator_and_postprocess_xla_spmd
    print("[patch] Trainer.create_accelerator_and_postprocess patched for XLA SPMD num_processes.")


def patch_trl_grpo_sampler_for_xla_spmd():
    """
    TRL's GRPO sampler recomputes its effective batch size at dataloader-build
    time from `self.accelerator.num_processes`. On single-process XLA SPMD that
    field can slip back to 1 even after constructor-time validation, which makes
    `effective_batch_size // num_generations` become zero and crashes
    `RepeatSampler.__iter__` with `range(..., step=0)`.

    Force the sampler path to use the XLA runtime device count consistently and
    re-write the accelerator state on the trainer instance before the sampler is
    instantiated.
    """
    try:
        from trl.trainer.grpo_trainer import GRPOTrainer, RepeatSampler
    except Exception as e:
        print(f"[WARN] Could not patch TRL GRPO sampler for XLA SPMD: {e}")
        return

    original = getattr(GRPOTrainer, "_get_train_sampler", None)
    if original is None or getattr(original, "_expdis_xla_spmd_sampler", False):
        return

    def _spmd_effective_num_processes(trainer) -> int:
        try:
            num_processes = int(getattr(trainer.accelerator, "num_processes", 1) or 1)
        except Exception:
            num_processes = 1
        if (
            _single_worker_xla_collective_passthrough()
            and (os.environ.get("EXPDIS_FSDP") or "").strip().lower() in ("1", "true", "full_shard")
        ):
            runtime_count = _xla_runtime_device_count()
            if runtime_count > num_processes:
                num_processes = runtime_count
                try:
                    trainer.accelerator.state.num_processes = runtime_count
                except Exception:
                    pass
                print(
                    "[patch] GRPO sampler using XLA runtime device count="
                    f"{runtime_count} for single-process SPMD."
                )
        return max(1, num_processes)

    def _get_train_sampler_xla_spmd(self, dataset=None):
        if not _single_worker_xla_collective_passthrough():
            return original(self) if not dataset else original(self, dataset)

        effective_num_processes = _spmd_effective_num_processes(self)
        effective_batch_size = (
            int(self.args.per_device_train_batch_size)
            * effective_num_processes
            * int(self.args.gradient_accumulation_steps)
        )
        sampler_batch_size = effective_batch_size // int(self.num_generations)
        if sampler_batch_size <= 0:
            raise ValueError(
                "Invalid GRPO sampler batch size under XLA SPMD: "
                f"effective_batch_size={effective_batch_size}, "
                f"num_generations={self.num_generations}, "
                f"per_device_train_batch_size={self.args.per_device_train_batch_size}, "
                f"gradient_accumulation_steps={self.args.gradient_accumulation_steps}, "
                f"effective_num_processes={effective_num_processes}."
            )
        return RepeatSampler(
            data_source=self.train_dataset,
            mini_repeat_count=self.num_generations,
            batch_size=sampler_batch_size,
            repeat_count=self.num_iterations * self.args.gradient_accumulation_steps,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )

    _get_train_sampler_xla_spmd._expdis_xla_spmd_sampler = True  # type: ignore[attr-defined]
    GRPOTrainer._get_train_sampler = _get_train_sampler_xla_spmd
    print("[patch] GRPOTrainer._get_train_sampler patched for XLA SPMD.")


def patch_transformers_trainer_logging_for_single_worker():
    """
    On single-worker TPU runs, Trainer logging does not need a distributed gather.
    Some TRL/XLA combinations surface `tr_loss` as a non-scalar tensor, and the
    default `_nested_gather(...).mean().item()` path can trigger an avoidable HBM
    allocation spike during `_maybe_log_save_evaluate`.
    """
    try:
        from transformers import Trainer
        import transformers.trainer as trainer_mod
    except Exception as e:
        print(f"[WARN] Could not patch Trainer logging for XLA: {e}")
        return

    original = getattr(Trainer, "_maybe_log_save_evaluate", None)
    if original is None:
        return
    if getattr(original, "_expdis_xla_single_worker_safe", False):
        return

    SaveStrategy = getattr(trainer_mod, "SaveStrategy", None)

    def _maybe_log_save_evaluate_xla_safe(
        self,
        tr_loss,
        grad_norm,
        model,
        trial,
        epoch,
        ignore_keys_for_eval,
        start_time,
        learning_rate=None,
    ):
        if not _single_worker_xla_collective_passthrough():
            return original(
                self,
                tr_loss,
                grad_norm,
                model,
                trial,
                epoch,
                ignore_keys_for_eval,
                start_time,
                learning_rate=learning_rate,
            )

        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            try:
                import torch_xla.core.xla_model as xm  # type: ignore

                xm.mark_step()
            except Exception:
                pass

            logs: dict[str, float] = {}

            if torch.is_tensor(tr_loss):
                with torch.no_grad():
                    tr_loss.zero_()
            else:
                tr_loss -= tr_loss

            # Avoid materializing TPU-resident loss / grad tensors on CPU.
            # On this single-worker XLA path, that transfer itself can trigger
            # the HBM OOM that kills the run. Leave generic Trainer loss /
            # grad_norm logging empty here and rely on TRL GRPO metrics for
            # the useful W&B signal.
            if learning_rate is not None:
                logs["learning_rate"] = learning_rate
            else:
                logs["learning_rate"] = self._get_learning_rate()

            self._globalstep_last_logged = self.state.global_step
            self.store_flos()
            if logs:
                self.log(logs, start_time)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)
            is_new_best_metric = self._determine_best_metric(metrics=metrics, trial=trial)

            if SaveStrategy is not None and self.args.save_strategy == SaveStrategy.BEST:
                self.control.should_save = is_new_best_metric

        if self.control.should_save:
            self._save_checkpoint(model, trial)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

        return metrics

    _maybe_log_save_evaluate_xla_safe._expdis_xla_single_worker_safe = True  # type: ignore[attr-defined]
    Trainer._maybe_log_save_evaluate = _maybe_log_save_evaluate_xla_safe
    print("[patch] Trainer._maybe_log_save_evaluate patched for single-worker XLA.")


def patch_torch_xla_parallel_loader_for_single_worker():
    """
    On single-worker TPU runs, the default XLA parallel loader prefetch settings
    can keep too many future batches resident around the step boundary. This is
    exactly where our Explorer GRPO runs are OOMing (`parallel_loader -> xm.mark_step()`).

    Force smaller prefetch queues by default on this path, while still allowing
    explicit overrides via env vars.
    """
    try:
        import torch_xla.distributed.parallel_loader as pl  # type: ignore
    except Exception as e:
        print(f"[WARN] Could not patch torch_xla ParallelLoader for XLA: {e}")
        return

    original = getattr(pl.MpDeviceLoader, "__init__", None)
    original_per_device_next = getattr(pl.PerDeviceLoader, "next", None)
    if original is None or original_per_device_next is None:
        return
    if getattr(original, "_expdis_xla_single_worker_safe", False):
        return

    def _env_int(name: str, default: int) -> int:
        raw = os.environ.get(name)
        if raw is None or str(raw).strip() == "":
            return int(default)
        try:
            return max(1, int(raw))
        except Exception:
            return int(default)

    def mp_device_loader_init_xla_safe(self, loader, device, **kwargs):
        if _single_worker_xla_collective_passthrough():
            kwargs.setdefault(
                "loader_prefetch_size",
                _env_int("EXPDIS_XLA_LOADER_PREFETCH_SIZE", 1),
            )
            kwargs.setdefault(
                "device_prefetch_size",
                _env_int("EXPDIS_XLA_DEVICE_PREFETCH_SIZE", 1),
            )
            kwargs.setdefault(
                "host_to_device_transfer_threads",
                _env_int("EXPDIS_XLA_HOST_TO_DEVICE_TRANSFER_THREADS", 1),
            )
        return original(self, loader, device, **kwargs)

    def per_device_loader_next_xla_safe(self):
        if not _single_worker_xla_collective_passthrough():
            return original_per_device_next(self)

        skip_loader_mark_step = os.environ.get("EXPDIS_XLA_SKIP_LOADER_MARK_STEP", "0") not in (
            "",
            "0",
            "false",
            "False",
            "no",
            "NO",
        )
        if not skip_loader_mark_step:
            return original_per_device_next(self)

        import torch_xla.core.xla_model as xm  # type: ignore
        import torch_xla.debug.profiler as xp  # type: ignore

        if xp.get_tracer_marked_step():
            xp.set_tracer_marked_step(False)
            self._batches_yielded += 1
        else:
            if self._mark_step_batch_count <= self._batches_yielded:
                self._batches_yielded = 0
            else:
                self._batches_yielded += 1

        item = self._loader.next_item(self._device)
        if item is None:
            if not self._loader._exception_queue.empty():
                raise self._loader._exception_queue.get()
            xm.mark_step()
            raise StopIteration
        return item

    mp_device_loader_init_xla_safe._expdis_xla_single_worker_safe = True  # type: ignore[attr-defined]
    per_device_loader_next_xla_safe._expdis_xla_single_worker_safe = True  # type: ignore[attr-defined]
    pl.MpDeviceLoader.__init__ = mp_device_loader_init_xla_safe
    pl.PerDeviceLoader.next = per_device_loader_next_xla_safe
    print("[patch] torch_xla MpDeviceLoader prefetch patched for single-worker XLA.")


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(model) -> torch.device:
    return next(model.parameters()).device


def _normalize_token_id(token_id):
    if token_id is None:
        return None
    if isinstance(token_id, (list, tuple)):
        return int(token_id[0]) if token_id else None
    if torch.is_tensor(token_id):
        if token_id.numel() == 0:
            return None
        return int(token_id.flatten()[0].item())
    return int(token_id)


def _normalize_eos_token_ids(token_id):
    """Return a list of all EOS token IDs (handles single int, list, tensor)."""
    if token_id is None:
        return []
    if isinstance(token_id, (list, tuple)):
        return [int(v) for v in token_id if v is not None]
    if torch.is_tensor(token_id):
        return [int(v) for v in token_id.flatten().tolist()]
    return [int(token_id)]


def _normalize_token_ids(token_ids):
    if token_ids is None:
        return None
    if torch.is_tensor(token_ids):
        token_ids = token_ids.detach().cpu().flatten().tolist()
    elif isinstance(token_ids, (set, tuple)):
        token_ids = list(token_ids)
    elif not isinstance(token_ids, list):
        token_ids = [token_ids]
    normalized = []
    for token_id in token_ids:
        value = _normalize_token_id(token_id)
        if value is not None:
            normalized.append(int(value))
    return sorted(set(normalized))


def xla_safe_generate(model, *args, **kwargs):
    """
    Minimal generation loop for XLA to avoid HF generate internals that trigger large compilations.
    Supports input_ids/attention_mask + GenerationConfig for max_new_tokens/temperature/top_p/top_k.
    """
    # Flash attention (Pallas kernel) crashes during autoregressive generation
    # on XLA because sequence lengths change each step and the kernel requires
    # fixed shapes padded to multiples of 1024.  Switch to eager attention for
    # the duration of generation, then restore.
    _orig_attn = None
    if getattr(getattr(model, "config", None), "_attn_implementation", None) == "tpu_flash":
        _orig_attn = "tpu_flash"
        model.config._attn_implementation = "eager"
        if hasattr(model.config, "_attn_implementation_internal"):
            model.config._attn_implementation_internal = "eager"

    def _restore_flash_attn():
        if _orig_attn is not None:
            model.config._attn_implementation = _orig_attn
            if hasattr(model.config, "_attn_implementation_internal"):
                model.config._attn_implementation_internal = _orig_attn

    input_ids = kwargs.pop("input_ids", None)
    attention_mask = kwargs.pop("attention_mask", None)
    if input_ids is None:
        input_ids = kwargs.pop("inputs", None)
    if input_ids is None and args:
        input_ids = args[0]
    if input_ids is None:
        raise ValueError("xla_safe_generate requires input_ids or inputs.")
    if attention_mask is None:
        attention_mask = kwargs.pop("attention_mask", None)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

    generation_config = kwargs.pop("generation_config", None)

    max_new_tokens = kwargs.pop("max_new_tokens", None)
    if max_new_tokens is None and generation_config is not None:
        max_new_tokens = getattr(generation_config, "max_new_tokens", None)
    if max_new_tokens is None and generation_config is not None:
        max_length = getattr(generation_config, "max_length", None)
        if max_length is not None:
            max_new_tokens = max(0, int(max_length) - int(input_ids.shape[1]))
    if max_new_tokens is None:
        max_new_tokens = 32

    do_sample = kwargs.pop("do_sample", None)
    if do_sample is None and generation_config is not None:
        do_sample = getattr(generation_config, "do_sample", None)
    if do_sample is None:
        do_sample = True

    temperature = kwargs.pop("temperature", None)
    if temperature is None and generation_config is not None:
        temperature = getattr(generation_config, "temperature", None)
    if temperature is None or temperature == 0:
        temperature = 1.0

    top_p = kwargs.pop("top_p", None)
    if top_p is None and generation_config is not None:
        top_p = getattr(generation_config, "top_p", None)
    if top_p is None:
        top_p = 1.0

    top_k = kwargs.pop("top_k", None)
    if top_k is None and generation_config is not None:
        top_k = getattr(generation_config, "top_k", None)
    if top_k is None:
        top_k = 0

    raw_eos = kwargs.pop("eos_token_id", None)
    if raw_eos is None and generation_config is not None:
        raw_eos = getattr(generation_config, "eos_token_id", None)
    eos_token_ids = _normalize_eos_token_ids(raw_eos)
    eos_token_id = eos_token_ids[0] if eos_token_ids else None

    pad_token_id = kwargs.pop("pad_token_id", None)
    if pad_token_id is None and generation_config is not None:
        pad_token_id = getattr(generation_config, "pad_token_id", None)
    pad_token_id = _normalize_token_id(pad_token_id)
    if pad_token_id is None and eos_token_id is not None:
        pad_token_id = eos_token_id
    if pad_token_id is None:
        pad_token_id = 0

    debug_generate = os.environ.get("EXPDIS_DEBUG_GENERATE", "0") == "1"
    try:
        progress_every = int(os.environ.get("EXPDIS_XLA_GENERATE_PROGRESS_EVERY", "0") or "0")
    except Exception:
        progress_every = 0
    if progress_every < 0:
        progress_every = 0
    progress_rank = (
        os.environ.get("RANK")
        or os.environ.get("LOCAL_RANK")
        or os.environ.get("XRT_SHARD_ORDINAL")
        or "0"
    ).strip()
    if _is_xla_device(input_ids.device):
        try:
            import torch_xla.core.xla_model as _xm_progress  # type: ignore

            progress_rank = str(_xm_progress.get_ordinal())
        except Exception:
            pass
    report_progress = progress_every > 0 and progress_rank == "0"
    if debug_generate:
        print(
            "[xla_safe_generate] "
            f"base_len={int(input_ids.shape[1])} "
            f"max_new_tokens={int(max_new_tokens)} "
            f"do_sample={bool(do_sample)} "
            f"temperature={float(temperature)} "
            f"top_p={float(top_p) if top_p is not None else 'None'} "
            f"top_k={int(top_k) if top_k is not None else 0} "
            f"eos_token_ids={eos_token_ids} "
            f"pad_token_id={pad_token_id}"
        )
    if report_progress:
        print(
            "[xla_safe_generate] "
            f"rank={progress_rank} batch={int(input_ids.shape[0])} "
            f"base_len={int(input_ids.shape[1])} max_new_tokens={int(max_new_tokens)} "
            f"progress_every={progress_every}"
        )

    allowed_token_ids = _normalize_token_ids(kwargs.pop("allowed_token_ids", None))
    initial_token_ids = _normalize_token_ids(kwargs.pop("initial_token_ids", None))

    use_cache = kwargs.pop("use_cache", None)
    if use_cache is None and generation_config is not None:
        use_cache = getattr(generation_config, "use_cache", None)
    if use_cache is None:
        use_cache = True

    device = input_ids.device
    use_incremental_xla = _is_xla_device(device) and os.environ.get(
        "EXPDIS_XLA_INCREMENTAL_GENERATE", "0"
    ) not in ("", "0", "false", "False", "no", "NO")
    if use_incremental_xla:
        use_cache = True
        if debug_generate:
            print("[xla_safe_generate] using incremental cached XLA decode path")

    def _neg_inf_like(t: torch.Tensor) -> torch.Tensor:
        if t.dtype.is_floating_point:
            return torch.tensor(torch.finfo(t.dtype).min, device=t.device, dtype=t.dtype)
        return torch.tensor(-1e9, device=t.device, dtype=torch.float32)

    def _sample_next(logits_tensor: torch.Tensor, sample: bool) -> torch.Tensor:
        if not sample:
            return torch.argmax(logits_tensor, dim=-1, keepdim=True)
        if _is_xla_device(logits_tensor.device):
            # XLA-friendly sampling to avoid multinomial CPU fallback.
            gumbel = -torch.log(-torch.log(torch.rand_like(logits_tensor)))
            return torch.argmax(logits_tensor + gumbel, dim=-1, keepdim=True)
        probs = torch.softmax(logits_tensor, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def _build_constraint_masks(vocab_size: int, dtype: torch.dtype, device_: torch.device):
        if not allowed_token_ids:
            return None, None
        neg_inf = torch.tensor(torch.finfo(dtype).min, device=device_, dtype=dtype)
        cont_mask = torch.full((1, vocab_size), neg_inf, device=device_, dtype=dtype)
        cont_mask[:, torch.tensor(allowed_token_ids, device=device_, dtype=torch.long)] = 0
        for _eos_id in eos_token_ids:
            cont_mask[:, int(_eos_id)] = 0

        init_ids = initial_token_ids if initial_token_ids else allowed_token_ids
        init_mask = torch.full((1, vocab_size), neg_inf, device=device_, dtype=dtype)
        init_mask[:, torch.tensor(init_ids, device=device_, dtype=torch.long)] = 0
        return init_mask, cont_mask

    if _is_xla_device(device) and not use_incremental_xla:
        xm = None
        try:
            import torch_xla.core.xla_model as xm  # type: ignore
        except Exception:
            xm = None

        with torch.no_grad():
            max_new_tokens = int(max_new_tokens)
            if max_new_tokens <= 0:
                _restore_flash_attn()
                return input_ids

            mark_step_every = int(os.environ.get("EXPDIS_XLA_MARK_STEP_EVERY", "1"))
            if mark_step_every <= 0:
                mark_step_every = 1

            batch_size, base_len = input_ids.shape
            max_len = base_len + max_new_tokens

            if attention_mask is None:
                attn = input_ids.ne(pad_token_id).long()
            else:
                attn = attention_mask

            attn_sum = attn.sum(dim=-1, keepdim=True).to(torch.long)
            left_padded = attn[:, 0].eq(0)
            next_pos = torch.where(
                left_padded.unsqueeze(-1),
                torch.full_like(attn_sum, base_len),
                attn_sum,
            )

            if attn.shape[1] < max_len:
                pad_len = max_len - attn.shape[1]
                attn = torch.cat(
                    [attn, torch.zeros((batch_size, pad_len), device=device, dtype=attn.dtype)],
                    dim=-1,
                )

            if input_ids.shape[1] < max_len:
                pad_len = max_len - input_ids.shape[1]
                pad = torch.full((batch_size, pad_len), pad_token_id, device=device, dtype=input_ids.dtype)
                generated = torch.cat([input_ids, pad], dim=-1)
            else:
                generated = input_ids

            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
            answer_started = torch.zeros(batch_size, dtype=torch.bool, device=device)
            constraint_masks = None

            for step in range(max_new_tokens):
                outputs = model(input_ids=generated, attention_mask=attn, use_cache=False)
                logits = outputs.logits

                last_pos = (next_pos - 1).clamp(min=0)
                vocab_size = logits.shape[-1]
                logits = logits.gather(1, last_pos.unsqueeze(-1).expand(-1, 1, vocab_size)).squeeze(1)

                if constraint_masks is None:
                    constraint_masks = _build_constraint_masks(vocab_size, logits.dtype, logits.device)
                init_mask, cont_mask = constraint_masks
                if cont_mask is not None:
                    active_mask = torch.where(answer_started.unsqueeze(-1), cont_mask, init_mask)
                    logits = logits + active_mask

                if temperature != 1.0:
                    logits = logits / temperature

                if top_k and top_k > 0:
                    k = min(int(top_k), logits.shape[-1])
                    values, _ = torch.topk(logits, k, dim=-1)
                    min_values = values[:, -1].unsqueeze(-1)
                    neg_inf = _neg_inf_like(logits)
                    logits = torch.where(logits < min_values, neg_inf, logits)

                if top_p is not None and float(top_p) < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                    probs = torch.softmax(sorted_logits, dim=-1)
                    cumprobs = torch.cumsum(probs, dim=-1)
                    mask = cumprobs > float(top_p)
                    mask[..., 0] = False
                    neg_inf = _neg_inf_like(sorted_logits)
                    sorted_logits = torch.where(mask, neg_inf, sorted_logits)
                    logits = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)

                next_tokens = _sample_next(logits, do_sample)
                if debug_generate and step < 20:
                    tok_ids = next_tokens.detach().cpu().flatten().tolist()[:8]
                    print(
                        f"[xla_safe_generate] step{step}_next_tokens={tok_ids} "
                        f"finished={finished.detach().cpu().tolist()[:8]}"
                    )

                was_finished = finished
                if eos_token_ids:
                    pad_tensor = torch.full_like(next_tokens, pad_token_id)
                    next_tokens = torch.where(was_finished.unsqueeze(-1), pad_tensor, next_tokens)
                    hit_eos = torch.zeros_like(finished)
                    for _eos_id in eos_token_ids:
                        hit_eos = hit_eos | (next_tokens.squeeze(-1) == _eos_id)
                    finished = finished | hit_eos
                answer_started = answer_started | (~was_finished)

                current_tokens = generated.gather(1, next_pos)
                next_tokens = torch.where(was_finished.unsqueeze(-1), current_tokens, next_tokens)
                generated = generated.scatter(1, next_pos, next_tokens)

                attn_update = (~was_finished).unsqueeze(-1).to(attn.dtype)
                attn = attn.scatter(1, next_pos, attn_update)
                next_pos = (next_pos + attn_update.to(next_pos.dtype)).clamp(max=max_len - 1)

                if xm is not None and ((step + 1) % mark_step_every == 0 or step == max_new_tokens - 1):
                    # Keep tensors alive through the XLA barrier. On TPU, the
                    # live graph can still hold handles through the returned
                    # output object at mark_step time.
                    xm.mark_step()
                    del outputs
                    del logits
                    del last_pos
                    del current_tokens
                    del next_tokens
                    del attn_update
                    if report_progress and (((step + 1) % progress_every == 0) or step == max_new_tokens - 1):
                        finished_count = int(finished.detach().sum().item())
                        print(
                            "[xla_safe_generate] "
                            f"rank={progress_rank} generated={step + 1}/{max_new_tokens} "
                            f"finished={finished_count}/{finished.numel()}"
                        )

            _restore_flash_attn()
            return generated

    xm = None
    xla_mark_step_every = 1
    if _is_xla_device(device):
        try:
            import torch_xla.core.xla_model as xm  # type: ignore
        except Exception:
            xm = None
        xla_mark_step_every = int(os.environ.get("EXPDIS_XLA_MARK_STEP_EVERY", "1"))
        if xla_mark_step_every <= 0:
            xla_mark_step_every = 1

    with torch.no_grad():
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=device)
        answer_started = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=device)
        generated = input_ids
        attn = attention_mask
        past = None
        constraint_masks = None

        for step in range(int(max_new_tokens)):
            if report_progress and step == 0:
                print(
                    "[xla_safe_generate] "
                    f"rank={progress_rank} first_token_forward_start "
                    f"batch={int(generated.shape[0])} seq_len={int(generated.shape[1])}"
                )
            if past is None or not use_cache:
                model_inputs = {"input_ids": generated, "attention_mask": attn}
            else:
                model_inputs = {"input_ids": generated[:, -1:], "attention_mask": attn, "past_key_values": past}

            outputs = model(**model_inputs, use_cache=use_cache)
            logits = outputs.logits[:, -1, :]
            if report_progress and step == 0:
                print(
                    "[xla_safe_generate] "
                    f"rank={progress_rank} first_token_forward_done "
                    f"logits_shape={tuple(logits.shape)}"
                )

            if constraint_masks is None:
                constraint_masks = _build_constraint_masks(logits.shape[-1], logits.dtype, logits.device)
            init_mask, cont_mask = constraint_masks
            if cont_mask is not None:
                active_mask = torch.where(answer_started.unsqueeze(-1), cont_mask, init_mask)
                logits = logits + active_mask
            if temperature != 1.0:
                logits = logits / temperature

            if top_k and top_k > 0:
                k = min(int(top_k), logits.shape[-1])
                values, _ = torch.topk(logits, k, dim=-1)
                min_values = values[:, -1].unsqueeze(-1)
                neg_inf = _neg_inf_like(logits)
                logits = torch.where(logits < min_values, neg_inf, logits)

            if top_p is not None and float(top_p) < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                probs = torch.softmax(sorted_logits, dim=-1)
                cumprobs = torch.cumsum(probs, dim=-1)
                mask = cumprobs > float(top_p)
                mask[..., 0] = False
                neg_inf = _neg_inf_like(sorted_logits)
                sorted_logits = torch.where(mask, neg_inf, sorted_logits)
                logits = torch.zeros_like(logits).scatter(1, sorted_indices, sorted_logits)

            next_tokens = _sample_next(logits, do_sample)

            if eos_token_ids:
                pad_tensor = torch.full_like(next_tokens, pad_token_id)
                next_tokens = torch.where(finished.unsqueeze(-1), pad_tensor, next_tokens)
                hit_eos = torch.zeros_like(finished)
                for _eos_id in eos_token_ids:
                    hit_eos = hit_eos | (next_tokens.squeeze(-1) == _eos_id)
                finished = finished | hit_eos
            answer_started = answer_started | (~finished)

            generated = torch.cat([generated, next_tokens], dim=-1)
            if attn is not None:
                attn = torch.cat([attn, torch.ones_like(next_tokens, dtype=attn.dtype)], dim=-1)
            past = outputs.past_key_values if use_cache else None

            if xm is not None and ((step + 1) % xla_mark_step_every == 0 or step == int(max_new_tokens) - 1):
                # Keep outputs alive through the XLA barrier because past_key_values
                # can share storage/handles with the returned model output object.
                xm.mark_step()
                del model_inputs
                del outputs
                del logits
                del next_tokens
                if report_progress and step == 0:
                    finished_count = int(finished.detach().sum().item())
                    print(
                        "[xla_safe_generate] "
                        f"rank={progress_rank} first_token_mark_step_done "
                        f"finished={finished_count}/{finished.numel()}"
                    )
                if report_progress and (((step + 1) % progress_every == 0) or step == int(max_new_tokens) - 1):
                    finished_count = int(finished.detach().sum().item())
                    print(
                        "[xla_safe_generate] "
                        f"rank={progress_rank} generated={step + 1}/{int(max_new_tokens)} "
                        f"finished={finished_count}/{finished.numel()}"
                    )

            if xm is None and finished.all():
                break

        _restore_flash_attn()
        return generated


def patch_model_generate_for_xla(model):
    """
    Replace model.generate with xla_safe_generate to avoid HF generate internals on TPU.
    """
    import types

    def _generate(self, *args, **kwargs):
        constraints = getattr(self, "_expdis_answer_constraints", None)
        if constraints:
            kwargs.setdefault("allowed_token_ids", constraints.get("allowed_token_ids"))
            kwargs.setdefault("initial_token_ids", constraints.get("initial_token_ids"))
        return xla_safe_generate(self, *args, **kwargs)

    model.generate = types.MethodType(_generate, model)
    print("[patch] model.generate patched for XLA.")


def patch_tokenizer_for_xla_fixed_padding(tokenizer, max_length: int, warning_max_length: "int | None" = None):
    """
    Force fixed-length padding on tokenizer calls to stabilize XLA shapes.
    Overrides padding=True/longest to padding='max_length' for TPU runs.
    `warning_max_length` controls the tokenizer's public length ceiling so valid
    prompt+completion flows do not emit misleading "sequence longer than model"
    warnings while prompt tokenization still pads to `max_length`.
    """
    import types

    tokenizer._expdis_xla_fixed_padding_max_length = int(max_length)
    tokenizer.model_max_length = int(warning_max_length or max_length)

    tokenizer_cls = tokenizer.__class__
    if not getattr(tokenizer_cls, "_expdis_xla_fixed_padding_patched", False):
        original_call = tokenizer_cls.__call__

        def _call(self, *args, **kwargs):
            fixed_max_length = getattr(self, "_expdis_xla_fixed_padding_max_length", None)
            if fixed_max_length is not None:
                padding = kwargs.get("padding")
                if padding is None or padding is True or padding == "longest":
                    kwargs["padding"] = "max_length"
                kwargs.setdefault("max_length", fixed_max_length)
                kwargs.setdefault("truncation", True)
            return original_call(self, *args, **kwargs)

        tokenizer_cls.__call__ = _call
        tokenizer_cls._expdis_xla_fixed_padding_patched = True

    if hasattr(tokenizer_cls, "apply_chat_template") and not getattr(tokenizer_cls, "_expdis_xla_chat_template_patched", False):
        original_apply = tokenizer_cls.apply_chat_template

        def _apply(self, *args, **kwargs):
            fixed_max_length = getattr(self, "_expdis_xla_fixed_padding_max_length", None)
            if fixed_max_length is not None:
                padding = kwargs.get("padding")
                if padding is None or padding is True or padding == "longest":
                    kwargs["padding"] = "max_length"
                kwargs.setdefault("max_length", fixed_max_length)
                kwargs.setdefault("truncation", True)
            return original_apply(self, *args, **kwargs)

        tokenizer_cls.apply_chat_template = _apply
        tokenizer_cls._expdis_xla_chat_template_patched = True
