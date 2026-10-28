#!/usr/bin/env python3
"""Evaluate a prepared GeoLC-8 checkpoint on its fixed protocol test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.experiments.models import make_model
from pipeline.scripts.train_benchmark import METHODS, evaluate, loaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=METHODS, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit-eval-batches", type=int)
    args = parser.parse_args()
    if args.method not in METHODS[args.task]:
        parser.error(f"{args.method} is not configured for {args.task}")
    return args


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    loader_args = SimpleNamespace(
        workspace=args.workspace.resolve(),
        task=args.task,
        input_size=args.input_size,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    _, _, test_loader, _, counts = loaders(loader_args)
    model = make_model(args.method, pretrained=False).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    metrics = evaluate(
        model,
        test_loader,
        device,
        limit=args.limit_eval_batches,
        per_region=args.task == "dg",
    )
    output = {
        "task": args.task,
        "method": args.method,
        "checkpoint": str(args.checkpoint),
        "dataset_counts": counts,
        "test": metrics,
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
