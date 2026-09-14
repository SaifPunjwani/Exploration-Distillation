#!/usr/bin/env python3
"""Local vLLM OpenAI-server pool: launch, health-wait, generate, restart-reload.

Contract decisions:
- One vLLM OpenAI api_server subprocess per GPU id (CUDA_VISIBLE_DEVICES=<id>),
  bf16, configurable --max-model-len, constant --served-model-name across
  reloads (clients cache the model name; changing it 404s mid-run).
- reload(model_dir) = kill all servers, relaunch pointing at the new weights
  dir with the SAME served-model-name, wait healthy. Restart-based reload was
  the TPU-proven weight-sync pattern (sleep/wake was flaky).
- generate() always POSTs pre-rendered prompts to /v1/completions (never
  /chat/completions; prompts are chat-template-rendered client side).
  Training sampling contract: temperature=1.0, top_p=0.95, top_k=20,
  max_tokens=16384. Each (prompt, sample) is its own request with a distinct
  seed (seed collapse across a fanout group was a measured TPU failure mode),
  round-robined across servers with per-server concurrency limits.

Failure semantics (mirrors tmx_jax/generate.py, which raised on error
completions by default):
- A request that exhausts all endpoints x 3 retry rounds degrades to
  Completion(finish_reason="error") — callers MUST filter these with
  completion_failed(); grpo_gpu drops whole error-bearing groups so error
  rows never reach reward scoring or trajectory files.
- generate() counts failed requests per call: any-error calls increment a
  consecutive-failure counter (reset by a clean call) and raise RuntimeError
  once max_consecutive_failures (default 3) is reached, so a wedged pool
  aborts the run instead of silently training on garbage.
- health_check() (called by the trainer between steps and between harvest
  passes) polls managed server processes and relaunches any that died — or
  that HUNG: >= hung_timeout_threshold (2) consecutive request timeouts on
  one server mark it wedged; its whole process group is SIGKILLed and it is
  relaunched. generate() deprioritizes wedged servers meanwhile.

DEVIATIONS:
- The TPU client sent one request per prompt with n=G; here every sample is a
  separate seeded n=1 request so the group fans out across the whole pool
  (same sampling distribution, better GPU utilization, no seed collapse).
- The TPU gate TMX_VLLM_ALLOW_ERROR_COMPLETIONS is replaced by the
  consecutive-failure threshold above: transient errors are tolerated (their
  groups are discarded upstream), sustained failure raises.

Pure Python (subprocess + requests); no torch import so eval-only /
pipeline-driver environments can import this module.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import requests


@dataclass
class Completion:
    text: str
    finish_reason: str
    completion_tokens: int = 0
    error: Optional[str] = None


def completion_failed(c: Completion) -> bool:
    """True for retry-exhausted requests (must never be scored or persisted)."""
    return c.finish_reason == "error" or c.error is not None


@dataclass
class VLLMPool:
    model_dir: str = ""
    # Hub revision for the initial remote base launch only. ``reload`` always
    # targets locally exported learner/SFT bytes and clears this field.
    model_revision: str = ""
    gpu_ids: Sequence[int] = ()
    served_model_name: str = ""
    port_base: int = 8000
    max_model_len: int = 18432
    gpu_memory_utilization: float = 0.90
    max_num_seqs: int = 32
    log_dir: str = "/tmp"
    extra_args: Sequence[str] = ()
    endpoints: List[str] = field(default_factory=list)  # ".../v1" URLs
    procs: List[subprocess.Popen] = field(default_factory=list)
    log_files: List[Any] = field(default_factory=list)
    managed: bool = True
    # generate() calls with >=1 failed request before aborting the run.
    max_consecutive_failures: int = 3
    # Seconds to wait after killing server processes before reusing the GPU /
    # port (CUDA memory release); tests zero this out.
    settle_seconds: float = 5.0
    # Hung-server detection: a wedged-but-alive vLLM answers /health but lets
    # requests hit the full per-request timeout. After this many CONSECUTIVE
    # request timeouts on one server it is treated as dead (health_check kills
    # its process group and relaunches it) and generate() prefers the other
    # servers meanwhile.
    hung_timeout_threshold: int = 2
    consecutive_failures: int = field(default=0, init=False)
    timeout_strikes: List[int] = field(default_factory=list, init=False)
    _strike_lock: threading.Lock = field(default_factory=threading.Lock,
                                         init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.served_model_name:
            self.served_model_name = self.model_dir
        if self.managed and not self.endpoints:
            self.endpoints = [
                f"http://127.0.0.1:{self.port_base + i}/v1" for i in range(len(self.gpu_ids))
            ]
        self.timeout_strikes = [0] * len(self.endpoints)

    # -------------------------------------------------------------- attach ----

    @classmethod
    def attach(cls, endpoints: Sequence[str], served_model_name: str) -> "VLLMPool":
        """Wrap already-running servers (no launch/reload management)."""
        eps = [e.rstrip("/") for e in endpoints if e.strip()]
        eps = [e if e.endswith("/v1") else e + "/v1" for e in eps]
        return cls(served_model_name=served_model_name, endpoints=eps, managed=False)

    # -------------------------------------------------------------- launch ----

    def launch(self) -> None:
        if not self.managed:
            raise RuntimeError("attach()-mode pool cannot launch servers")
        os.makedirs(self.log_dir, exist_ok=True)
        self.procs = [None] * len(self.gpu_ids)  # type: ignore[list-item]
        self.log_files = [None] * len(self.gpu_ids)
        self.timeout_strikes = [0] * len(self.endpoints)
        try:
            for i in range(len(self.gpu_ids)):
                self._launch_one(i)
        except Exception:
            # A partial launch must not leave None placeholders in procs
            # (they would break _kill/reload later) or orphan the servers
            # that DID start holding GPU memory: clean up, then re-raise.
            print("[vllm-pool] launch failed part-way; killing already-started "
                  "servers before re-raising", flush=True)
            self._kill()
            raise

    def _launch_one(self, i: int) -> None:
        """(Re)launch the server for pool slot i on its GPU/port."""
        gpu = self.gpu_ids[i]
        port = self.port_base + i
        # A stale server (orphan from a dead pipeline process) answering on
        # this port would mask our own server's bind failure and serve OLD
        # weights forever — refuse to adopt it.
        stale = False
        try:
            stale = requests.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code == 200
        except requests.RequestException:
            stale = False
        if stale:
            raise RuntimeError(
                f"port {port} already answers /health — a stale vLLM server is "
                "running (orphan from a previous run?); kill it before launching"
            )
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", str(self.model_dir),
            "--served-model-name", str(self.served_model_name),
            "--port", str(port),
            "--dtype", "bfloat16",
            "--max-model-len", str(int(self.max_model_len)),
            "--gpu-memory-utilization", str(float(self.gpu_memory_utilization)),
            "--max-num-seqs", str(int(self.max_num_seqs)),
        ]
        if self.model_revision:
            cmd += ["--revision", str(self.model_revision)]
        cmd += [str(a) for a in self.extra_args]
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        old_log = self.log_files[i] if i < len(self.log_files) else None
        if old_log is not None:
            try:
                old_log.close()
            except Exception:  # noqa: BLE001
                pass
        log_path = os.path.join(self.log_dir, f"vllm_gpu{gpu}_port{port}.log")
        log_f = open(log_path, "a")
        log_f.write(f"\n===== launch {time.strftime('%Y-%m-%d %H:%M:%S')} model={self.model_dir} =====\n")
        log_f.flush()
        proc = subprocess.Popen(
            cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.procs[i] = proc
        self.log_files[i] = log_f
        print(f"[vllm-pool] launched gpu={gpu} port={port} pid={proc.pid} "
              f"model={self.model_dir} served-as={self.served_model_name}", flush=True)

    def wait_healthy(self, timeout: float = 1200.0, poll: float = 5.0) -> None:
        deadline = time.time() + float(timeout)
        for i, ep in enumerate(self.endpoints):
            base = ep[:-3] if ep.endswith("/v1") else ep
            while True:
                proc = self.procs[i] if (self.managed and i < len(self.procs)) else None
                if proc is not None and proc.poll() is not None:
                    raise RuntimeError(
                        f"vLLM server for {ep} exited rc={proc.returncode}; "
                        f"see logs in {self.log_dir}"
                    )
                try:
                    r = requests.get(base + "/health", timeout=5)
                    if r.status_code == 200:
                        # A stale orphan could answer while our process died on
                        # a bind failure — require our own process to be alive.
                        if proc is not None and proc.poll() is not None:
                            raise RuntimeError(
                                f"vLLM server for {ep} exited rc={proc.returncode} "
                                f"but the port still answers /health (stale server?); "
                                f"see logs in {self.log_dir}"
                            )
                        break
                except requests.RequestException:
                    pass
                if time.time() > deadline:
                    raise RuntimeError(f"vLLM server {ep} not healthy after {timeout:.0f}s")
                time.sleep(poll)
        print(f"[vllm-pool] all {len(self.endpoints)} servers healthy", flush=True)

    def health_check(self) -> int:
        """Between-steps liveness check: relaunch dead or hung managed servers.

        Dead = the server process exited. Hung = >= hung_timeout_threshold
        CONSECUTIVE request timeouts on that server (wedged-but-alive vLLM).
        Either way the server's WHOLE process group is killed (vLLM spawns
        workers that can outlive a dead leader and keep holding GPU memory /
        the port) and we settle before relaunching. Returns the number of
        servers restarted (0 for attach-mode pools, which cannot be managed
        from here). Raises if a relaunched server fails to become healthy.
        """
        if not self.managed or not self.procs:
            return 0
        restarted = 0
        threshold = max(1, int(self.hung_timeout_threshold))
        for i, proc in enumerate(self.procs):
            dead = proc is None or proc.poll() is not None
            with self._strike_lock:
                strikes = self.timeout_strikes[i] if i < len(self.timeout_strikes) else 0
            hung = strikes >= threshold
            if not dead and not hung:
                continue
            why = (f"died rc={proc.returncode}" if dead and proc is not None
                   else "vanished" if dead
                   else f"hung ({strikes} consecutive request timeouts)")
            print(f"[vllm-pool] server {i} (port {self.port_base + i}) {why}; "
                  f"killing its process group and relaunching from {self.model_dir}",
                  flush=True)
            self._reap_one(i)
            self._launch_one(i)
            with self._strike_lock:
                if i < len(self.timeout_strikes):
                    self.timeout_strikes[i] = 0
            restarted += 1
        if restarted:
            self.wait_healthy()
        return restarted

    def _reap_one(self, i: int) -> None:
        """SIGKILL server slot i's whole process group, wait for the leader,
        and settle so relaunching cannot race leaked GPU memory / the port."""
        proc = self.procs[i] if i < len(self.procs) else None
        if proc is None:
            return
        try:
            # start_new_session=True at launch => pgid == leader pid; killpg
            # reaps surviving worker children even after the leader died.
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:  # noqa: BLE001 — group already gone
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass
        deadline = time.time() + 30
        while proc.poll() is None and time.time() < deadline:
            time.sleep(0.5)
        time.sleep(self.settle_seconds)

    # -------------------------------------------------------------- reload ----

    def reload(self, model_dir: str, *, timeout: float = 1200.0) -> None:
        """Kill servers and relaunch on new weights, same served-model-name."""
        if not self.managed:
            print("[vllm-pool] WARNING: attach()-mode pool cannot reload; "
                  "servers keep serving stale weights", flush=True)
            return
        t0 = time.time()
        print(f"[vllm-pool] reload -> {model_dir} (restart-based)", flush=True)
        for attempt in (1, 2):  # one retry: a flaky reload must not kill a multi-day run
            self._kill()
            self.model_dir = str(model_dir)
            # Stage handoffs and learner sync exports are local directories;
            # an upstream Hub commit must never be forwarded to those loads.
            self.model_revision = ""
            try:
                self.launch()
                self.wait_healthy(timeout=timeout)
                break
            except Exception as exc:  # noqa: BLE001 — launch/wait fail many ways
                # launch() cleans up its partial state on failure and _kill()
                # tolerates dead/missing procs, so the retry is always safe.
                if attempt == 2:
                    raise
                print(f"[vllm-pool] reload attempt {attempt} failed ({exc}); "
                      "retrying once", flush=True)
                time.sleep(10.0)
        print(f"[vllm-pool] reload complete in {time.time() - t0:.1f}s "
              f"(served-model-name={self.served_model_name})", flush=True)

    def _kill(self) -> None:
        # Tolerate None placeholders (partial launch) and already-dead procs:
        # _kill must always be safe to call so reload's retry can never crash.
        live = [p for p in self.procs if p is not None]
        for proc in live:
            if proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except Exception:  # noqa: BLE001
                    try:
                        proc.terminate()
                    except Exception:  # noqa: BLE001
                        pass
        deadline = time.time() + 60
        for proc in live:
            while proc.poll() is None and time.time() < deadline:
                time.sleep(1.0)
            if proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:  # noqa: BLE001
                    try:
                        proc.kill()
                    except Exception:  # noqa: BLE001
                        pass
                # Wait for the SIGKILL to land: relaunching while the old
                # server still holds GPU memory / the port makes the new
                # server OOM or fail to bind.
                kill_deadline = time.time() + 30
                while proc.poll() is None and time.time() < kill_deadline:
                    time.sleep(0.5)
        self.procs = []
        for log_f in self.log_files:
            if log_f is None:
                continue
            try:
                log_f.close()
            except Exception:  # noqa: BLE001
                pass
        self.log_files = []
        if live:
            time.sleep(self.settle_seconds)  # let CUDA memory release before relaunch

    def shutdown(self) -> None:
        if self.managed:
            self._kill()

    # ------------------------------------------------------------ generate ----

    def _endpoint_order(self, rid: int) -> List[int]:
        """Round-robin primary + failover order over endpoint indices,
        deprioritizing servers at/over the consecutive-timeout threshold so a
        wedged server stops absorbing (and timing out) fresh requests while
        it waits for health_check to kill + relaunch it."""
        n = len(self.endpoints)
        base = [(int(rid) + j) % n for j in range(n)]
        threshold = max(1, int(self.hung_timeout_threshold))
        with self._strike_lock:
            strikes = (list(self.timeout_strikes)
                       if len(self.timeout_strikes) == n else [0] * n)
        return sorted(base, key=lambda i: strikes[i] >= threshold)  # stable

    def _note_timeout(self, ei: int) -> None:
        with self._strike_lock:
            if ei < len(self.timeout_strikes):
                self.timeout_strikes[ei] += 1

    def _note_success(self, ei: int) -> None:
        with self._strike_lock:
            if ei < len(self.timeout_strikes):
                self.timeout_strikes[ei] = 0

    def _post_one(
        self,
        prompt: str,
        rid: int,
        sampling: Dict[str, Any],
        timeout: float,
    ) -> Completion:
        payload = {
            "model": self.served_model_name,
            "prompt": prompt,
            "n": 1,
            "stream": False,
            "seed": int(rid),
            **sampling,
        }
        last_error = None
        for attempt in range(3):
            for ei in self._endpoint_order(rid):
                ep = self.endpoints[ei]
                try:
                    r = requests.post(f"{ep}/completions", json=payload, timeout=timeout)
                    r.raise_for_status()
                    data = r.json()
                    ch = data["choices"][0]
                    usage = data.get("usage", {})
                    self._note_success(ei)
                    return Completion(
                        text=ch.get("text", ""),
                        finish_reason=str(ch.get("finish_reason") or ""),
                        completion_tokens=int(usage.get("completion_tokens", 0) or 0),
                    )
                except requests.exceptions.Timeout as exc:
                    # Wedged-but-alive server: count consecutive timeouts so
                    # health_check can treat it as dead (kill + relaunch).
                    last_error = f"{type(exc).__name__}: {exc}"
                    self._note_timeout(ei)
                except Exception as exc:  # noqa: BLE001
                    last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(min(5 * (attempt + 1), 15))
        return Completion(text="", finish_reason="error", error=last_error)

    def generate(
        self,
        prompts: List[str],
        n: int,
        *,
        max_tokens: int = 16384,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 20,
        min_p: float = 0.0,
        seed_base: int = 0,
        timeout: float = 3600.0,
        concurrency_per_server: int = 8,
        enforce_consecutive_failure_limit: bool = True,
    ) -> List[List[Completion]]:
        """Fan (prompt x n) requests out over all servers; return grouped.

        Failure accounting: a call with >= 1 retry-exhausted request counts as
        a failed call; max_consecutive_failures failed calls in a row raise
        RuntimeError (a clean call resets the counter). Individual error
        completions are returned for the caller to filter via
        completion_failed() — they must never be scored or persisted.
        """
        if not self.endpoints:
            raise RuntimeError("vLLM pool has no endpoints")
        sampling = {
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "min_p": float(min_p),
        }
        results: Dict[tuple, Completion] = {}
        workers = max(1, len(self.endpoints) * max(1, int(concurrency_per_server)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {}
            for pi, prompt in enumerate(prompts):
                for si in range(int(n)):
                    rid = int(seed_base) + pi * 100003 + si
                    futs[pool.submit(self._post_one, prompt, rid, sampling, timeout)] = (pi, si)
            for fut in as_completed(futs):
                results[futs[fut]] = fut.result()

        failed = [c for c in results.values() if completion_failed(c)]
        if failed:
            if enforce_consecutive_failure_limit:
                self.consecutive_failures += 1
            else:
                # Protected callers (C10) own a preregistered row-rate
                # transport rule.  Do not let this legacy call-count circuit
                # breaker preempt or leak across that classifier.
                self.consecutive_failures = 0
            dead = [i for i, p in enumerate(self.procs) if p.poll() is not None] if self.managed else []
            print(f"[vllm-pool] WARNING: {len(failed)}/{len(results)} requests failed "
                  f"after retries (consecutive failed calls: "
                  f"{self.consecutive_failures}/{self.max_consecutive_failures}; "
                  f"caller-owned-rate-policy={not enforce_consecutive_failure_limit}; "
                  f"dead servers: {dead or 'none'}); "
                  f"first error: {failed[0].error}", flush=True)
            if enforce_consecutive_failure_limit and \
                    self.consecutive_failures >= max(
                        1, int(self.max_consecutive_failures)):
                raise RuntimeError(
                    f"vLLM pool failing persistently: {len(failed)}/{len(results)} requests "
                    f"failed in the {self.consecutive_failures}th consecutive failing "
                    f"generate() call (dead servers: {dead or 'none'}; last error: "
                    f"{failed[0].error}); see logs in {self.log_dir}"
                )
        else:
            self.consecutive_failures = 0
        return [[results[(pi, si)] for si in range(int(n))] for pi in range(len(prompts))]
