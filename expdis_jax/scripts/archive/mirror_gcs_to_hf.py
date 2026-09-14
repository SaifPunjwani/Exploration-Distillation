#!/usr/bin/env python3
"""Mirror GCS checkpoint artifacts into a Hugging Face dataset repo.

The script preserves the path below the GCS bucket, downloads one object at a
time, uploads it to HF, and removes the local temp file. It is intentionally
simple and resumable: existing HF files are skipped unless --force is passed.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, capture_output=True, check=check)


def iter_gcs_objects(prefixes: list[str]) -> list[str]:
    objects: list[str] = []
    for prefix in prefixes:
        target = prefix.rstrip("/")
        if not target.startswith("gs://"):
            raise ValueError(f"GCS prefix must start with gs://, got {prefix!r}")
        if not target.endswith("/**"):
            target = target.rstrip("/") + "/**"
        proc = run(["gcloud", "storage", "ls", "--recursive", target])
        for line in proc.stdout.splitlines():
            line = line.strip()
            if line.startswith("gs://") and not line.endswith("/"):
                objects.append(line)
    return sorted(set(objects))


def relpath_for_uri(uri: str, bucket: str) -> str:
    base = f"gs://{bucket}/"
    if not uri.startswith(base):
        raise ValueError(f"{uri!r} is not under bucket {bucket!r}")
    return uri[len(base):]


def should_exclude(rel: str, patterns: list[str]) -> bool:
    rel = rel.lstrip("/")
    for pat in patterns:
        pat = pat.strip().lstrip("/")
        if not pat:
            continue
        if pat.endswith("/"):
            if rel.startswith(pat):
                return True
        elif rel == pat or rel.startswith(pat.rstrip("/") + "/"):
            return True
    return False


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", default="two-model-exploration-checkpoints")
    p.add_argument("--repo-id", default=os.environ.get("EXPDIS_HF_CHECKPOINT_REPO", "SaifPunjwani/two-model-exploration-checkpoints"))
    p.add_argument("--repo-type", default="dataset")
    p.add_argument("--prefix", action="append", default=[], help="GCS prefix to mirror. May be repeated.")
    p.add_argument("--exclude", action="append", default=["secrets/"], help="Relative path prefix to exclude. May be repeated.")
    p.add_argument("--manifest", default="", help="Optional path to write full source/destination manifest.")
    p.add_argument("--force", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    prefixes = args.prefix or [f"gs://{args.bucket}/"]
    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type=args.repo_type, private=True, exist_ok=True)

    print(f"[mirror] listing HF files for {args.repo_id}", flush=True)
    existing = set(api.list_repo_files(args.repo_id, repo_type=args.repo_type))
    print(f"[mirror] existing HF files: {len(existing)}", flush=True)

    print(f"[mirror] listing GCS objects: {prefixes}", flush=True)
    objects = iter_gcs_objects(prefixes)
    print(f"[mirror] source GCS objects: {len(objects)}", flush=True)

    todo: list[tuple[str, str]] = []
    for uri in objects:
        rel = relpath_for_uri(uri, args.bucket)
        if should_exclude(rel, args.exclude):
            continue
        if not args.force and rel in existing:
            continue
        todo.append((uri, rel))
        if args.limit and len(todo) >= args.limit:
            break

    if args.manifest:
        manifest_path = Path(args.manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w") as f:
            for uri, rel in todo:
                f.write(f"{uri}\t{args.repo_id}/{rel}\n")
        print(f"[mirror] wrote manifest: {manifest_path}", flush=True)

    print(f"[mirror] upload todo: {len(todo)}", flush=True)
    failures = 0
    with tempfile.TemporaryDirectory(prefix="expdis_gcs_to_hf_") as tmp:
        tmpdir = Path(tmp)
        for idx, (uri, rel) in enumerate(todo, start=1):
            local = tmpdir / rel
            local.parent.mkdir(parents=True, exist_ok=True)
            print(f"[mirror] {idx}/{len(todo)} download {uri}", flush=True)
            try:
                run(["gcloud", "storage", "cp", uri, str(local)])
                print(f"[mirror] {idx}/{len(todo)} upload {rel}", flush=True)
                api.upload_file(
                    repo_id=args.repo_id,
                    repo_type=args.repo_type,
                    path_or_fileobj=str(local),
                    path_in_repo=rel,
                    commit_message=f"mirror {rel}",
                )
            except Exception as exc:
                failures += 1
                print(f"[mirror] ERROR {uri}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            finally:
                try:
                    local.unlink(missing_ok=True)
                except Exception:
                    pass
    print(f"[mirror] complete failures={failures}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
