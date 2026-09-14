"""Optional artifact mirroring in a child process that never imports JAX."""

import hashlib
import os
import subprocess
import sys
import tempfile
import time


def _hf_artifact_repo() -> str:
    return (
        os.environ.get("EXPDIS_HF_CHECKPOINT_REPO")
        or os.environ.get("EXPDIS_HF_ARTIFACT_REPO")
        or ""
    ).strip()


def _hf_upload_log_path(cfg, path_in_repo: str, source: str) -> str:
    log_dir = os.environ.get("EXPDIS_HF_UPLOAD_LOG_DIR", "").strip()
    if not log_dir:
        log_dir = os.path.join(cfg.output_dir, "hf_upload_logs")
    os.makedirs(log_dir, exist_ok=True)
    basename = os.path.basename(path_in_repo.rstrip("/")) or "artifact"
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in basename)[:80]
    digest = hashlib.sha1(f"{source}:{path_in_repo}".encode("utf-8")).hexdigest()[:12]
    return os.path.join(log_dir, f"{int(time.time())}_{digest}_{safe}.log")


def _run_hf_upload_child(env: dict, cfg, path_in_repo: str, source: str) -> bool:
    require = str(env.get("EXPDIS_REQUIRE_HF_ARTIFACT_UPLOAD", os.environ.get("EXPDIS_REQUIRE_HF_ARTIFACT_UPLOAD", "0"))).lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    async_upload = str(env.get("EXPDIS_HF_MIRROR_ASYNC", os.environ.get("EXPDIS_HF_MIRROR_ASYNC", "1"))).lower() not in (
        "0",
        "false",
        "no",
        "off",
    ) and not require
    log_path = _hf_upload_log_path(cfg, path_in_repo, source)
    if async_upload:
        try:
            with open(log_path, "ab") as log_f:
                subprocess.Popen(
                    [sys.executable, os.path.abspath(__file__)],
                    env=env,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            print(f"[hf] scheduled background upload -> {env['EXPDIS_HF_UPLOAD_REPO_ID']}/{path_in_repo} log={log_path}", flush=True)
            return True
        except Exception as exc:
            print(f"[hf] WARN failed to schedule upload for {path_in_repo}: {type(exc).__name__}: {exc}", flush=True)
            if require:
                raise
            return False
    ret = subprocess.run(
        [sys.executable, os.path.abspath(__file__)],
        env=env,
        capture_output=True,
        text=True,
        timeout=float(os.environ.get("EXPDIS_HF_UPLOAD_TIMEOUT_SECONDS", "7200") or "7200"),
    )
    if ret.returncode == 0:
        print(f"[hf] uploaded -> {env['EXPDIS_HF_UPLOAD_REPO_ID']}/{path_in_repo}", flush=True)
        return True
    print(f"[hf] WARN upload failed for {path_in_repo}: {(ret.stderr or ret.stdout)[-1000:]}", flush=True)
    if require:
        raise RuntimeError(f"HF upload failed for {path_in_repo}")
    return False


def _maybe_upload_path_to_hf(
    local_path: str,
    cfg,
    path_in_repo: str,
    *,
    is_dir: bool,
    force_sync: bool = False,
) -> bool:
    repo_id = _hf_artifact_repo()
    if not repo_id or not os.path.exists(local_path):
        return False
    env = os.environ.copy()
    env.pop("EXPDIS_HF_UPLOAD_GCS_URI", None)
    env.pop("EXPDIS_HF_UPLOAD_LOCAL_PATH", None)
    env.update(
        {
            "EXPDIS_HF_UPLOAD_REPO_ID": repo_id,
            "EXPDIS_HF_UPLOAD_REPO_TYPE": "dataset",
            "EXPDIS_HF_UPLOAD_LOCAL_PATH": os.path.abspath(local_path),
            "EXPDIS_HF_UPLOAD_PATH_IN_REPO": path_in_repo.strip("/"),
            "EXPDIS_HF_UPLOAD_IS_DIR": "1" if is_dir else "0",
        }
    )
    if force_sync:
        env["EXPDIS_HF_MIRROR_ASYNC"] = "0"
    return _run_hf_upload_child(env, cfg, path_in_repo, os.path.abspath(local_path))


def _maybe_upload_gcs_uri_to_hf(
    gcs_uri: str,
    cfg,
    path_in_repo: str,
    *,
    is_dir: bool,
    force_sync: bool = False,
) -> bool:
    repo_id = _hf_artifact_repo()
    if not repo_id or not gcs_uri:
        return False
    env = os.environ.copy()
    env.pop("EXPDIS_HF_UPLOAD_GCS_URI", None)
    env.pop("EXPDIS_HF_UPLOAD_LOCAL_PATH", None)
    env.update(
        {
            "EXPDIS_HF_UPLOAD_REPO_ID": repo_id,
            "EXPDIS_HF_UPLOAD_REPO_TYPE": "dataset",
            "EXPDIS_HF_UPLOAD_GCS_URI": gcs_uri,
            "EXPDIS_HF_UPLOAD_PATH_IN_REPO": path_in_repo.strip("/"),
            "EXPDIS_HF_UPLOAD_IS_DIR": "1" if is_dir else "0",
        }
    )
    if force_sync:
        env["EXPDIS_HF_MIRROR_ASYNC"] = "0"
    return _run_hf_upload_child(env, cfg, path_in_repo, gcs_uri)


def _upload_local(api, local_path, path_in_repo, *, repo_id, repo_type, is_dir):
    common = dict(repo_id=repo_id, repo_type=repo_type, path_in_repo=path_in_repo,
                  commit_message=f"mirror {path_in_repo}")
    if is_dir:
        api.upload_folder(folder_path=local_path, **common)
    else:
        api.upload_file(path_or_fileobj=local_path, **common)


def _upload_worker():
    from huggingface_hub import HfApi

    repo_id = os.environ["EXPDIS_HF_UPLOAD_REPO_ID"]
    repo_type = os.environ.get("EXPDIS_HF_UPLOAD_REPO_TYPE") or "dataset"
    path = os.environ["EXPDIS_HF_UPLOAD_PATH_IN_REPO"].strip("/")
    is_dir = os.environ.get("EXPDIS_HF_UPLOAD_IS_DIR", "0") == "1"
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=True, exist_ok=True)
    kwargs = dict(repo_id=repo_id, repo_type=repo_type, is_dir=is_dir)
    gcs_uri = os.environ.get("EXPDIS_HF_UPLOAD_GCS_URI", "").rstrip("/")
    if gcs_uri:
        with tempfile.TemporaryDirectory(prefix="expdis_hf_gcs_") as tmp:
            local = os.path.join(tmp, "artifact")
            if is_dir:
                os.makedirs(local)
                command = ["gcloud", "storage", "cp", "--recursive", gcs_uri + "/*", local + "/"]
            else:
                command = ["gcloud", "storage", "cp", gcs_uri, local]
            subprocess.run(command, check=True, text=True)
            _upload_local(api, local, path, **kwargs)
    else:
        _upload_local(api, os.environ["EXPDIS_HF_UPLOAD_LOCAL_PATH"], path, **kwargs)
    print(f"[hf] uploaded {repo_id}/{path}", flush=True)


if __name__ == "__main__":
    _upload_worker()
