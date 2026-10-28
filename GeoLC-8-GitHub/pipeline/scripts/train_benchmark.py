#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pipeline.experiments.data import OpticalDataset, choose_samples, read_manifest
from pipeline.experiments.metrics import SegmentationMetrics
from pipeline.experiments.models import ema_copy, make_model, update_ema


METHODS = {
    "segmentation": ["unet_resnet50", "segformer_b2", "rs3mamba", "dinov2_b14_mask2former"],
    "da": ["daformer", "mic", "mrm"],
    "dg": ["rein", "crossearth", "depthforge"],
}

# Global reference-label frequencies from the audited 4.275-billion-pixel corpus.
CLASS_FREQUENCIES = torch.tensor([0.05143, 0.12860, 0.08628, 0.02296, 0.29135, 0.11444, 0.22590, 0.07903])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=METHODS, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--limit-train-batches", type=int)
    parser.add_argument("--limit-eval-batches", type=int)
    args = parser.parse_args()
    if args.method not in METHODS[args.task]:
        parser.error(f"{args.method} is not configured for {args.task}")
    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def dice_loss(logits, target, classes=8):
    probs = logits.softmax(1)
    one_hot = F.one_hot(target, classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    score = (2 * (probs * one_hot).sum(dims) + 1.0) / (probs.sum(dims) + one_hot.sum(dims) + 1.0)
    return 1.0 - score.mean()


def supervised_loss(logits, target):
    weights = CLASS_FREQUENCIES.to(logits.device).rsqrt()
    weights = weights / weights.mean()
    return F.cross_entropy(logits, target, weight=weights) + dice_loss(logits, target)


def style_mix(x):
    if x.shape[0] < 2:
        return x
    perm = torch.randperm(x.shape[0], device=x.device)
    mean = x.mean((2, 3), keepdim=True)
    std = x.std((2, 3), keepdim=True).clamp_min(1e-5)
    other_mean, other_std = mean[perm], std[perm]
    alpha = torch.rand(x.shape[0], 1, 1, 1, device=x.device) * 0.8
    return ((x - mean) / std) * (alpha * other_std + (1 - alpha) * std) + alpha * other_mean + (1 - alpha) * mean


def loaders(args):
    base = args.workspace / "pipeline/data"
    samples = read_manifest(base / "experiment_splits.csv")
    image_dir = base / "tiles/images"
    label_dir = base / "pseudo_labels/labels"
    common = dict(image_dir=image_dir, label_dir=label_dir, input_size=args.input_size)
    if args.task == "segmentation":
        train = OpticalDataset(choose_samples(samples, args.task, "train"), train=True, **common)
        val = OpticalDataset(choose_samples(samples, args.task, "validation"), train=False, **common)
        test = OpticalDataset(choose_samples(samples, args.task, "test"), train=False, **common)
        target = None
    elif args.task == "da":
        train = OpticalDataset(choose_samples(samples, args.task, "source_train"), train=True, **common)
        val = OpticalDataset(choose_samples(samples, args.task, "source_val"), train=False, **common)
        test = OpticalDataset(choose_samples(samples, args.task, "target_test"), train=False, **common)
        target = OpticalDataset(choose_samples(samples, args.task, "target_train"), train=True, labeled=False, strong_view=True, **common)
    else:
        train = OpticalDataset(choose_samples(samples, args.task, "source_train"), train=True, **common)
        val = OpticalDataset(choose_samples(samples, args.task, "source_val"), train=False, **common)
        test = OpticalDataset(choose_samples(samples, args.task, "unseen_test"), train=False, **common)
        target = None
    kwargs = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=False, persistent_workers=args.workers > 0)
    return (
        DataLoader(train, shuffle=True, drop_last=True, **kwargs),
        DataLoader(val, shuffle=False, **kwargs),
        DataLoader(test, shuffle=False, **kwargs),
        DataLoader(target, shuffle=True, drop_last=True, **kwargs) if target is not None else None,
        {"train": len(train), "validation": len(val), "test": len(test), "target_unlabeled": len(target) if target else 0},
    )


@torch.no_grad()
def evaluate(model, loader, device, limit=None, per_region=False):
    model.eval()
    overall = SegmentationMetrics(device=device)
    regions = defaultdict(lambda: SegmentationMetrics(device=device))
    total_loss = 0.0
    seen = 0
    for step, batch in enumerate(loader):
        if limit is not None and step >= limit:
            break
        image, label = batch["image"].to(device), batch["label"].to(device)
        logits = model(image)
        total_loss += supervised_loss(logits, label).item() * image.shape[0]
        seen += image.shape[0]
        overall.update(logits, label)
        if per_region:
            for i, region in enumerate(batch["region"]):
                regions[region].update(logits[i : i + 1], label[i : i + 1])
    metrics = overall.compute()
    metrics["loss"] = round(total_loss / max(seen, 1), 6)
    if per_region:
        metrics["regions"] = {key: value.compute() for key, value in sorted(regions.items())}
    return metrics


def train_epoch(args, model, teacher, optimizer, source_loader, target_loader, device):
    model.train()
    total = defaultdict(float)
    target_iter = iter(target_loader) if target_loader else None
    start = time.time()
    steps = 0
    for step, source in enumerate(source_loader):
        if args.limit_train_batches is not None and step >= args.limit_train_batches:
            break
        image = source["image"].to(device)
        label = source["label"].to(device)
        if args.task == "dg" and args.method in {"crossearth", "depthforge"}:
            image = style_mix(image)
        logits, feats = model(image, return_features=True)
        sup = supervised_loss(logits, label)
        loss = sup
        unsup = torch.zeros((), device=device)
        rebuild = torch.zeros((), device=device)

        if target_iter is not None:
            try:
                target = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target = next(target_iter)
            weak = target["image"].to(device)
            strong = target["strong_image"].to(device)
            with torch.no_grad():
                pseudo_logits = teacher(weak)
                confidence, pseudo = pseudo_logits.softmax(1).max(1)
                valid = confidence >= (0.90 if args.method == "daformer" else 0.80)
            if args.method in {"mic", "mrm"}:
                mask = (torch.rand(strong.shape[0], 1, strong.shape[2], strong.shape[3], device=device) > 0.4).float()
                strong = strong * mask
            target_logits, target_feats = model(strong, return_features=True)
            pixel_loss = F.cross_entropy(target_logits, pseudo, reduction="none")
            unsup = (pixel_loss * valid).sum() / valid.sum().clamp_min(1)
            if args.method == "daformer":
                entropy = -(target_logits.softmax(1) * target_logits.log_softmax(1)).sum(1).mean()
                unsup = unsup + 0.01 * entropy
            if args.method == "mrm":
                with torch.no_grad():
                    teacher_feats = teacher(weak, return_features=True)[1][-1]
                rebuild = F.smooth_l1_loss(target_feats[-1], teacher_feats)
            loss = sup + 0.5 * unsup + 0.1 * rebuild

        if args.task == "dg" and args.method == "crossearth":
            original_logits = model(source["image"].to(device))
            loss = loss + 0.1 * F.kl_div(logits.log_softmax(1), original_logits.softmax(1).detach(), reduction="batchmean") / (logits.shape[-1] * logits.shape[-2])

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if teacher is not None:
            update_ema(teacher, model)
        total["loss"] += loss.item()
        total["supervised"] += sup.item()
        total["unsupervised"] += unsup.item()
        total["reconstruction"] += rebuild.item()
        steps += 1
    elapsed = time.time() - start
    return {key: round(value / max(steps, 1), 6) for key, value in total.items()} | {"steps": steps, "seconds": round(elapsed, 2), "samples_per_second": round(steps * args.batch_size / max(elapsed, 1e-6), 3)}


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def main():
    args = parse_args()
    seed_everything(args.seed)
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable. Run outside the filesystem sandbox with MPS permission.")
    device = torch.device(args.device)
    output = args.output_dir or args.workspace / "pipeline/experiments/runs" / args.task / args.method
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "history.json"
    history = json.loads(log_path.read_text()) if args.resume and log_path.exists() else []

    train_loader, val_loader, test_loader, target_loader, counts = loaders(args)
    model = make_model(args.method, pretrained=not args.no_pretrained).to(device)
    teacher = ema_copy(model).to(device) if args.task == "da" else None
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best = -1.0
    start_epoch = 1
    checkpoint = output / "last.pt"
    if args.resume and checkpoint.exists():
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if teacher is not None and state.get("teacher"):
            teacher.load_state_dict(state["teacher"])
        start_epoch = state["epoch"] + 1
        best = state.get("best", best)

    config = vars(args).copy()
    config["workspace"] = str(config["workspace"])
    config["output_dir"] = str(output)
    config["dataset_counts"] = counts
    config["parameter_count"] = sum(p.numel() for p in model.parameters())
    config["trainable_parameter_count"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    config["implementation_note"] = "Pure-PyTorch/MPS port; method-specific public objectives/modules retained; not a bit-identical CUDA/mmseg reproduction."
    atomic_json(output / "config.json", config)
    print(json.dumps(config, sort_keys=True), flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        train_stats = train_epoch(args, model, teacher, optimizer, train_loader, target_loader, device)
        val_stats = evaluate(model, val_loader, device, args.limit_eval_batches)
        scheduler.step()
        record = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "train": train_stats, "validation": val_stats}
        history.append(record)
        atomic_json(log_path, history)
        state = {"epoch": epoch, "model": model.state_dict(), "teacher": teacher.state_dict() if teacher else None, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "best": max(best, val_stats["miou"])}
        torch.save(state, checkpoint)
        if val_stats["miou"] > best:
            best = val_stats["miou"]
            torch.save({"epoch": epoch, "model": model.state_dict(), "validation": val_stats}, output / "best.pt")
        print(json.dumps(record, sort_keys=True), flush=True)

    best_state = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_state["model"])
    test = evaluate(model, test_loader, device, args.limit_eval_batches, per_region=args.task == "dg")
    result = {"task": args.task, "method": args.method, "seed": args.seed, "epochs": args.epochs, "best_epoch": best_state["epoch"], "validation": best_state["validation"], "test": test, "dataset_counts": counts}
    atomic_json(output / "result.json", result)
    print("FINAL " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
