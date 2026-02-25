#!/usr/bin/env python3
"""Create local git worktrees for model branches."""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
from pathlib import Path


def _slugify(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip())
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "branch"


def _run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if check and proc.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            f"  {' '.join(shlex.quote(c) for c in cmd)}\n"
            f"  exit={proc.returncode}\n"
            f"  stdout={proc.stdout.strip()}\n"
            f"  stderr={proc.stderr.strip()}"
        )
    return proc


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare worktrees from remote branches.")
    parser.add_argument("--repo", default=".", help="Git repository root")
    parser.add_argument(
        "--output-dir",
        default="../model-worktrees",
        help="Directory to place worktrees",
    )
    parser.add_argument("--remote", default="origin", help="Remote name")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip git fetch")
    parser.add_argument("branches", nargs="+", help="Remote branch names")
    args = parser.parse_args()

    repo = Path(args.repo).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (repo / ".git").exists():
        raise FileNotFoundError(f"Not a git repository: {repo}")

    if not args.skip_fetch:
        print("[INFO] git fetch --all --prune")
        _run(["git", "-C", str(repo), "fetch", "--all", "--prune"])

    for branch in args.branches:
        remote_ref = f"{args.remote}/{branch}"
        probe = _run(["git", "-C", str(repo), "rev-parse", "--verify", remote_ref], check=False)
        if probe.returncode != 0:
            print(f"[WARN] Skip (missing remote ref): {remote_ref}")
            continue

        worktree_path = out_dir / _slugify(branch)
        if worktree_path.exists():
            print(f"[INFO] Skip (already exists): {worktree_path}")
            continue

        print(f"[INFO] Create worktree for {remote_ref} -> {worktree_path}")
        _run(["git", "-C", str(repo), "worktree", "add", "--detach", str(worktree_path), remote_ref])

    print("[INFO] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
