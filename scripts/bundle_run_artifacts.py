#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tarfile


PATTERNS = [
    "analysis/benchmark_summary.json",
    "analysis/benchmark_eval_samples.json",
    "analysis/benchmark_rollouts_*.json",
    "analysis/embedding_novelty_summary.json",
    "analysis/permutation_tests.json",
    "explorer_trajectories*.jsonl",
    "plots/*",
    "console.log",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bundle benchmark and trajectory artifacts from a run dir.")
    parser.add_argument("run_dir", help="Run directory under runs/ or an absolute path.")
    parser.add_argument("--output", default="", help="Optional output tar.gz path.")
    return parser.parse_args()


def _resolve_run_dir(value: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    repo_runs = Path("runs") / value
    if repo_runs.is_dir():
        return repo_runs.resolve()
    raise SystemExit(f"Run directory not found: {value}")


def _collect(run_dir: Path) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    for pattern in PATTERNS:
        for path in run_dir.glob(pattern):
            if path.is_file():
                resolved = path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    files.append(resolved)
    files.sort()
    return files


def main() -> None:
    args = _parse_args()
    run_dir = _resolve_run_dir(args.run_dir)
    files = _collect(run_dir)
    if not files:
        raise SystemExit(f"No bundleable artifacts found in {run_dir}")

    output = Path(args.output).expanduser() if args.output else run_dir / "artifacts_bundle.tar.gz"
    output.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_dir": str(run_dir),
        "artifact_count": len(files),
        "files": [str(path.relative_to(run_dir)) for path in files],
    }
    manifest_path = run_dir / "analysis" / "artifacts_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    with tarfile.open(output, "w:gz") as archive:
        archive.add(manifest_path, arcname=str(manifest_path.relative_to(run_dir)))
        for path in files:
            archive.add(path, arcname=str(path.relative_to(run_dir)))

    print(str(output.resolve()))


if __name__ == "__main__":
    main()
