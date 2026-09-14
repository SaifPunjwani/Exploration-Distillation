import hashlib
import os
import subprocess
import sys
import time


DEFAULT_HF_ARTIFACT_REPO = "SaifPunjwani/two-model-exploration-checkpoints"


def env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def artifact_backend() -> str:
    return os.environ.get("TMX_ARTIFACT_BACKEND", "hf").strip().lower() or "hf"


def gcs_artifacts_enabled() -> bool:
    """GCS writes are opt-in only; TPU allocation is fine, bucket churn is not."""
    return env_flag("TMX_ALLOW_GCS_ARTIFACTS", "0") and artifact_backend() in {
        "gcs",
        "dual",
        "both",
        "gcs+hf",
    }


def hf_repo_id() -> str:
    return (
        os.environ.get("TMX_HF_CHECKPOINT_REPO")
        or os.environ.get("TMX_HF_ARTIFACT_REPO")
        or DEFAULT_HF_ARTIFACT_REPO
    ).strip()


def hf_repo_type() -> str:
    return os.environ.get("TMX_HF_REPO_TYPE", "dataset").strip() or "dataset"


def hf_artifact_prefix(run_name: str | None = None) -> str:
    prefix = os.environ.get("TMX_HF_ARTIFACT_PREFIX", "").strip().strip("/")
    if prefix:
        return prefix
    safe = safe_artifact_name(run_name or os.environ.get("RUN_NAME", "") or "run")
    return safe


def safe_artifact_name(value: str) -> str:
    import re

    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "").strip())
    cleaned = cleaned.strip("-.")
    return cleaned or "artifact"


_HF_UPLOAD_CODE = r"""
import os
import traceback

from huggingface_hub import HfApi

repo_id = os.environ["TMX_HF_UPLOAD_REPO_ID"]
repo_type = os.environ.get("TMX_HF_UPLOAD_REPO_TYPE") or "dataset"
local_path = os.environ["TMX_HF_UPLOAD_LOCAL_PATH"]
path_in_repo = os.environ["TMX_HF_UPLOAD_PATH_IN_REPO"].strip("/")
is_dir = os.environ.get("TMX_HF_UPLOAD_IS_DIR", "0") == "1"

print(f"[hf-child] upload {local_path} -> {repo_id}/{path_in_repo}", flush=True)
try:
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=True, exist_ok=True)
    if is_dir:
        api.upload_folder(
            repo_id=repo_id,
            repo_type=repo_type,
            folder_path=local_path,
            path_in_repo=path_in_repo,
            commit_message=f"tmx artifact {path_in_repo}",
        )
    else:
        api.upload_file(
            repo_id=repo_id,
            repo_type=repo_type,
            path_or_fileobj=local_path,
            path_in_repo=path_in_repo,
            commit_message=f"tmx artifact {path_in_repo}",
        )
    print("[hf-child] done", flush=True)
except Exception:
    traceback.print_exc()
    raise
"""


def _upload_log_path(path_in_repo: str, source: str) -> str:
    log_dir = os.environ.get("TMX_HF_UPLOAD_LOG_DIR", "").strip() or os.path.join(
        os.getcwd(), "hf_upload_logs"
    )
    os.makedirs(log_dir, exist_ok=True)
    basename = os.path.basename(path_in_repo.rstrip("/")) or "artifact"
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in basename)[:80]
    digest = hashlib.sha1(f"{source}:{path_in_repo}".encode("utf-8")).hexdigest()[:12]
    return os.path.join(log_dir, f"{int(time.time())}_{digest}_{safe}.log")


def upload_path_to_hf(local_path: str, path_in_repo: str, *, is_dir: bool) -> str | None:
    """Upload an artifact to Hugging Face and return an hf:// URI on success."""
    repo_id = hf_repo_id()
    if not repo_id or not os.path.exists(local_path):
        return None
    path_in_repo = path_in_repo.strip("/")
    env = os.environ.copy()
    env.update(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": env.get("HF_HUB_ENABLE_HF_TRANSFER", "1"),
            "TMX_HF_UPLOAD_REPO_ID": repo_id,
            "TMX_HF_UPLOAD_REPO_TYPE": hf_repo_type(),
            "TMX_HF_UPLOAD_LOCAL_PATH": os.path.abspath(local_path),
            "TMX_HF_UPLOAD_PATH_IN_REPO": path_in_repo,
            "TMX_HF_UPLOAD_IS_DIR": "1" if is_dir else "0",
        }
    )
    require = env_flag("TMX_REQUIRE_HF_ARTIFACT_UPLOAD", "0")
    async_upload = env_flag("TMX_HF_MIRROR_ASYNC", "1") and not require
    log_path = _upload_log_path(path_in_repo, os.path.abspath(local_path))
    if async_upload:
        try:
            with open(log_path, "ab") as log_f:
                subprocess.Popen(
                    [sys.executable, "-c", _HF_UPLOAD_CODE],
                    env=env,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            print(f"[hf] scheduled upload -> {repo_id}/{path_in_repo} log={log_path}", flush=True)
            return f"hf://{repo_id}/{path_in_repo}"
        except Exception as exc:
            print(f"[hf] WARN failed to schedule upload for {path_in_repo}: {exc}", flush=True)
            if require:
                raise
            return None
    ret = subprocess.run(
        [sys.executable, "-c", _HF_UPLOAD_CODE],
        env=env,
        capture_output=True,
        text=True,
        timeout=float(os.environ.get("TMX_HF_UPLOAD_TIMEOUT_SECONDS", "7200") or "7200"),
    )
    if ret.returncode == 0:
        print(f"[hf] uploaded -> {repo_id}/{path_in_repo}", flush=True)
        return f"hf://{repo_id}/{path_in_repo}"
    print(f"[hf] WARN upload failed for {path_in_repo}: {(ret.stderr or ret.stdout)[-1000:]}", flush=True)
    if require:
        raise RuntimeError(f"HF upload failed for {path_in_repo}")
    return None
