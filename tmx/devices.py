import glob
import os
import urllib.request

import torch

xm = None
xr = None

_METADATA_BASE = "http://metadata.google.internal/computeMetadata/v1/instance/attributes/"


def _tpu_device_nodes():
    return sorted(glob.glob("/dev/accel*"))


def _metadata_attr(name: str) -> str | None:
    req = urllib.request.Request(
        f"{_METADATA_BASE}{name}",
        headers={"Metadata-Flavor": "Google"},
    )
    try:
        with urllib.request.urlopen(req, timeout=0.5) as resp:
            value = resp.read().decode("utf-8").strip()
            return value or None
    except Exception:
        return None


def _worker_endpoints() -> list[str]:
    raw = _metadata_attr("worker-network-endpoints")
    if not raw:
        return []
    endpoints: list[str] = []
    for item in raw.split(","):
        parts = [part.strip() for part in item.split(":") if part.strip()]
        if parts:
            endpoints.append(parts[-1])
    return endpoints


def _is_multi_worker_slice() -> bool:
    return len(_worker_endpoints()) > 1


def _allow_single_worker_slice_override() -> bool:
    if os.environ.get("TMX_ALLOW_SINGLE_WORKER_TPU") == "1":
        return True
    worker_hosts = (os.environ.get("TPU_WORKER_HOSTNAMES") or "").strip().lower()
    process_bounds = (os.environ.get("TPU_PROCESS_BOUNDS") or "").strip()
    chips_per_process = (os.environ.get("TPU_CHIPS_PER_PROCESS_BOUNDS") or "").strip()
    return worker_hosts == "localhost" and process_bounds == "1,1,1" and chips_per_process == "1,1,1"


def _tpu_runtime_present() -> bool:
    return bool(_tpu_device_nodes() or _metadata_attr("accelerator-type"))


def _infer_tpu_accelerator_type() -> str | None:
    if os.environ.get("TPU_ACCELERATOR_TYPE"):
        return os.environ["TPU_ACCELERATOR_TYPE"]
    metadata_accelerator = _metadata_attr("accelerator-type")
    if metadata_accelerator:
        return metadata_accelerator
    nodes = _tpu_device_nodes()
    node_count = len(nodes)
    if os.path.exists("/sys/module/tpu_v4"):
        if node_count >= 4:
            return "v4-8"
        if node_count > 0:
            return f"v4-{max(1, node_count * 2)}"
    return None


def clear_torch_xla_env():
    if os.environ.get("TMX_TPU_SLICE_LAUNCHED") == "1" and os.environ.get("TMX_REQUIRE_TPU") == "1":
        os.environ["PJRT_DEVICE"] = "TPU"
        os.environ["USE_TORCH_XLA"] = "1"
        return
    os.environ.pop("PJRT_DEVICE", None)
    os.environ["USE_TORCH_XLA"] = "0"


def _env_indicates_tpu() -> bool:
    pjrt_device = os.environ.get("PJRT_DEVICE", "").upper()
    if pjrt_device == "TPU":
        return True
    for key in (
        "TPU_NAME",
        "TPU_CHIPS_PER_HOST_BOUNDS",
        "TPU_ACCELERATOR_TYPE",
        "XRT_TPU_CONFIG",
        "COLAB_TPU_ADDR",
    ):
        if os.environ.get(key):
            return True
    return _tpu_runtime_present()


def _slice_launch_guard_message() -> str:
    endpoints = _worker_endpoints()
    endpoint_msg = ", ".join(endpoints) if endpoints else "unknown"
    return (
        "[devices] Multi-worker TPU slice detected. "
        "Launch via the slice-aware scripts so all workers participate. "
        f"Detected worker endpoints: {endpoint_msg}."
    )


def set_torch_xla_env(preference: str = "auto") -> bool:
    pref = (preference or "auto").lower()
    if pref == "tpu":
        if _tpu_runtime_present():
            os.environ.setdefault("PJRT_DEVICE", "TPU")
            os.environ["USE_TORCH_XLA"] = "1"
            inferred = _infer_tpu_accelerator_type()
            if inferred:
                os.environ.setdefault("TPU_ACCELERATOR_TYPE", inferred)
            os.environ.setdefault("TPU_SKIP_MDS_QUERY", "1")
            return True
        clear_torch_xla_env()
        return False
    if pref == "auto" and _env_indicates_tpu():
        if _tpu_runtime_present():
            os.environ.setdefault("PJRT_DEVICE", "TPU")
            os.environ["USE_TORCH_XLA"] = "1"
            inferred = _infer_tpu_accelerator_type()
            if inferred:
                os.environ.setdefault("TPU_ACCELERATOR_TYPE", inferred)
            os.environ.setdefault("TPU_SKIP_MDS_QUERY", "1")
            return True
    clear_torch_xla_env()
    return False


def maybe_import_xla():
    """Try importing torch_xla lazily so CPU/GPU runs still work."""
    global xm, xr
    if xm is not None or xr is not None:
        return xm, xr
    if os.environ.get("USE_TORCH_XLA") != "1":
        return None, None
    try:
        import torch_xla.core.xla_model as xm_mod
        import torch_xla.runtime as xr_mod
        current_mark_step = getattr(xm_mod, "mark_step", None)
        if current_mark_step is not None and not getattr(current_mark_step, "_tmx_env_guard", False):
            original_mark_step = current_mark_step

            def mark_step_env_guard(*args, **kwargs):
                os.environ.setdefault("PJRT_DEVICE", "TPU")
                os.environ["USE_TORCH_XLA"] = "1"
                return original_mark_step(*args, **kwargs)

            mark_step_env_guard._tmx_env_guard = True  # type: ignore[attr-defined]
            xm_mod.mark_step = mark_step_env_guard
        xm = xm_mod
        xr = xr_mod
    except Exception:
        xm, xr = None, None
    return xm, xr


def resolve_device(preference: str = "auto"):
    """
    Pick device with TPU-first preference when requested.
    Returns (torch.device, is_xla: bool, label: str).
    """
    pref = (preference or "auto").lower()

    if (
        _is_multi_worker_slice()
        and os.environ.get("TMX_TPU_SLICE_LAUNCHED") != "1"
        and not _allow_single_worker_slice_override()
    ):
        msg = _slice_launch_guard_message()
        if pref == "tpu":
            if os.environ.get("TMX_REQUIRE_TPU") == "1":
                raise RuntimeError(msg)
            print(f"{msg} Falling back to CPU.")
            return torch.device("cpu"), False, "cpu"
        if pref == "auto":
            print(f"{msg} Auto mode will not attempt local TPU init from a single worker shell.")
    elif _is_multi_worker_slice() and _allow_single_worker_slice_override():
        print("[devices] Multi-worker TPU slice detected, but single-worker local TPU override is active.")

    use_xla = set_torch_xla_env(pref)
    if use_xla:
        xm_mod, xr_mod = maybe_import_xla()
        if xm_mod is not None and xr_mod is not None:
            try:
                dev = xm_mod.xla_device()
                try:
                    supported = xm_mod.get_xla_supported_devices()
                except Exception:
                    supported = []
                print(f"[devices] XLA device acquired: {dev}; supported={supported}")
                return dev, True, "tpu"
            except Exception as e:
                print(f"[devices] XLA device failed: {e}")
                if os.environ.get("TMX_REQUIRE_TPU") == "1" and pref == "tpu":
                    raise RuntimeError(
                        "[devices] TPU initialization failed in a slice-launched run. "
                        "Refusing CPU fallback because TMX_REQUIRE_TPU=1."
                    ) from e
        clear_torch_xla_env()
    elif pref == "tpu":
        if not _tpu_runtime_present():
            msg = "[devices] TPU requested but no TPU runtime or accelerator metadata was found."
        else:
            msg = "[devices] TPU requested but torch_xla could not initialize the TPU runtime."
        if os.environ.get("TMX_REQUIRE_TPU") == "1":
            raise RuntimeError(f"{msg} Refusing to fall back because TMX_REQUIRE_TPU=1.")
        print(f"{msg} Falling back to CPU.")

    if pref in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda"), False, "cuda"

    return torch.device("cpu"), False, "cpu"


def optimizer_step(optimizer, is_xla: bool):
    """Use XLA-friendly optimizer step when on TPU."""
    if is_xla and xm is not None:
        xm.optimizer_step(optimizer, barrier=True)
        xm.mark_step()
    else:
        optimizer.step()


def mark_step(is_xla: bool, *, wait: bool = False, reset_scope: bool = True):
    if is_xla and xm is not None:
        xm.mark_step(wait=wait, reset_scope=reset_scope)
