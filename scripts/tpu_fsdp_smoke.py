#!/usr/bin/env python3
import json
import os
import socket
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tmx.devices import maybe_import_xla, resolve_device


def _env_int(name: str):
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return raw


def main() -> None:
    device, is_xla, label = resolve_device("tpu")
    xm, xr = maybe_import_xla()
    payload = {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "rank": _env_int("RANK"),
        "local_rank": _env_int("LOCAL_RANK"),
        "world_size_env": _env_int("WORLD_SIZE"),
        "local_world_size_env": _env_int("LOCAL_WORLD_SIZE"),
        "pjrt_device": os.environ.get("PJRT_DEVICE"),
        "tpu_visible_chips": os.environ.get("TPU_VISIBLE_CHIPS"),
        "tpu_process_bounds": os.environ.get("TPU_PROCESS_BOUNDS"),
        "device": str(device),
        "label": label,
        "is_xla": bool(is_xla),
    }

    if not is_xla or xm is None or xr is None:
        print(json.dumps(payload, sort_keys=True))
        return

    try:
        payload["xla_world_size"] = int(xr.world_size())
    except Exception as exc:
        payload["xla_world_size_error"] = str(exc)
    try:
        payload["xla_global_ordinal"] = int(xr.global_ordinal())
    except Exception as exc:
        payload["xla_global_ordinal_error"] = str(exc)
    try:
        payload["xla_local_ordinal"] = int(xm.get_local_ordinal())
    except Exception as exc:
        payload["xla_local_ordinal_error"] = str(exc)

    xm.rendezvous("tmx_fsdp_smoke_start")
    ordinal_value = float(payload.get("xla_global_ordinal", 0) or 0)
    tensor = torch.tensor([ordinal_value], device=device)
    xm.mark_step()
    payload["tensor_device"] = str(tensor.device)
    payload["tensor_value"] = tensor.cpu().tolist()
    xm.rendezvous("tmx_fsdp_smoke_done")

    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
