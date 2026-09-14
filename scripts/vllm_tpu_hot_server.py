"""Custom vLLM TPU server with in-process /update_weights endpoint.

Replaces `vllm serve` when EXPDIS_VLLM_HOT_RELOAD=1. Exposes the standard
OpenAI-compatible routes plus POST /update_weights that hot-swaps weights
via engine collective_rpc, avoiding the 60-90s XLA recompile from full restart.

The training side calls POST /update_weights with JSON {"path": "<local_dir>"};
each worker must already have the weights copied to that path (orchestrator's
reload_vllm_slice_with_model.sh does the parallel copy from GCS or training TPU).

Falls back to full restart if collective_rpc fails (e.g., API mismatch).
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

logger = logging.getLogger("vllm_hot_server")


def _apply_update_weights_patch() -> None:
    """Monkey-patch vllm's OpenAI app to register POST /update_weights.

    vLLM's FastAPI app is module-level. We import it, add a route, then let
    vllm's CLI entrypoint take over. The patched app is used automatically.
    """
    try:
        from vllm.entrypoints.openai import api_server  # type: ignore
    except Exception as exc:
        logger.error("could not import vllm.entrypoints.openai.api_server: %s", exc)
        raise

    app: FastAPI = api_server.app

    @app.post("/update_weights")
    async def update_weights(req: Request) -> Response:
        """Hot-swap weights in-place via vLLM engine collective_rpc.

        Request body: {"path": "<local_path_to_hf_model_dir>"}
        Returns 200 on success, 4xx/5xx on error.
        """
        try:
            payload = await req.json()
        except Exception:
            raise HTTPException(400, "invalid JSON body")
        path = (payload or {}).get("path", "").strip()
        if not path:
            raise HTTPException(400, "'path' field required")
        if not os.path.isdir(path):
            raise HTTPException(404, f"path not found or not a directory: {path}")

        # Pull the running async engine client off the app state. The attr name
        # varies between vLLM versions: try the known locations.
        engine = None
        for attr in ("engine_client", "engine", "async_engine"):
            engine = getattr(app.state, attr, None)
            if engine is not None:
                break
        if engine is None:
            raise HTTPException(500, "vLLM engine not found on app.state")

        # Try a sequence of known weight-update RPC names. vLLM evolved this API
        # over releases; we try the modern name first, then the older one.
        rpc_candidates = (
            "load_weights_from_path",
            "load_weights",
            "reload_weights",
        )
        last_err: Optional[Exception] = None
        for method_name in rpc_candidates:
            try:
                if hasattr(engine, "collective_rpc_async"):
                    await engine.collective_rpc_async(method_name, args=(path,))
                elif hasattr(engine, "collective_rpc"):
                    result = engine.collective_rpc(method_name, args=(path,))
                    if asyncio.iscoroutine(result):
                        await result
                else:
                    raise AttributeError("engine has no collective_rpc{_async}")
                logger.info("update_weights OK via %s (path=%s)", method_name, path)
                return JSONResponse(
                    status_code=200,
                    content={
                        "status": "ok",
                        "loaded_from": path,
                        "rpc": method_name,
                    },
                )
            except Exception as exc:  # try next method name
                last_err = exc
                logger.warning("update_weights %s failed: %s", method_name, exc)
                continue

        raise HTTPException(
            500,
            f"all collective_rpc methods failed; last error: {last_err}",
        )

    @app.get("/update_weights/health")
    async def update_weights_health() -> Response:
        return JSONResponse(status_code=200, content={"endpoint": "ready"})

    logger.info("registered POST /update_weights route on vLLM FastAPI app")


def main() -> int:
    logging.basicConfig(
        level=os.environ.get("VLLM_HOT_SERVER_LOG_LEVEL", "INFO"),
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    )
    _apply_update_weights_patch()
    # Delegate to vllm's CLI main. It'll start the OpenAI API server using the
    # patched app (because we mutated the module-level object).
    try:
        from vllm.entrypoints.openai.api_server import run_server  # type: ignore
        from vllm.entrypoints.openai.cli_args import (  # type: ignore
            make_arg_parser,
            validate_parsed_serve_args,
        )
    except Exception as exc:
        logger.error("could not import vllm server entrypoints: %s", exc)
        return 1

    # vLLM's argparser. We pass through sys.argv[1:] so all CLI args are honored.
    import argparse as _argparse
    parser = make_arg_parser(_argparse.ArgumentParser())
    args = parser.parse_args()
    try:
        validate_parsed_serve_args(args)
    except Exception as exc:
        logger.warning("validate_parsed_serve_args failed (non-fatal): %s", exc)
    # Register SIGTERM handler to clean shutdown
    def _handle_term(signum, frame):
        logger.info("received signal %s, exiting", signum)
        sys.exit(0)
    signal.signal(signal.SIGTERM, _handle_term)
    signal.signal(signal.SIGINT, _handle_term)

    try:
        asyncio.run(run_server(args))
    except Exception as exc:
        logger.error("run_server raised: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
