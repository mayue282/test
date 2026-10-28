#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNS = [
    ("segmentation", "unet_resnet50"),
    ("segmentation", "segformer_b2"),
    ("segmentation", "rs3mamba"),
    ("segmentation", "dinov2_b14_mask2former"),
    ("da", "daformer"),
    ("da", "mic"),
    ("da", "mrm"),
    ("dg", "rein"),
    ("dg", "crossearth"),
    ("dg", "depthforge"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    python = sys.executable
    train = args.workspace / "pipeline/scripts/train_benchmark.py"
    summary = []
    for task, method in RUNS:
        result_path = args.workspace / "pipeline/experiments/runs" / task / method / "result.json"
        if result_path.exists():
            summary.append(json.loads(result_path.read_text()))
            print(f"SKIP completed {task}/{method}", flush=True)
            continue
        command = [python, str(train), "--workspace", str(args.workspace), "--task", task, "--method", method, "--epochs", str(args.epochs), "--input-size", str(args.input_size), "--batch-size", str(args.batch_size), "--workers", str(args.workers), "--device", args.device]
        if args.resume:
            command.append("--resume")
        print("RUN " + " ".join(command), flush=True)
        subprocess.run(command, check=True)
        summary.append(json.loads(result_path.read_text()))
        (args.workspace / "pipeline/experiments/runs/summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()

