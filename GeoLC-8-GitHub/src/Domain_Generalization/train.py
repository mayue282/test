#!/usr/bin/env python3
"""Task-specific entry point for GeoLC-8 domain generalization."""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    command = [
        sys.executable,
        str(ROOT / "pipeline/scripts/train_benchmark.py"),
        "--workspace",
        str(ROOT),
        "--task",
        "dg",
        *sys.argv[1:],
    ]
    raise SystemExit(subprocess.call(command))

