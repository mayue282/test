#!/usr/bin/env python3
"""Verify GeoLC-8 manifests, raster pairs, and pretrained file hashes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--skip-hashes", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.workspace.resolve()
    data = root / "pipeline/data"
    manifest = data / "experiment_splits.csv"
    with manifest.open(newline="", encoding="utf-8") as stream:
        expected = {row["tile_name"] for row in csv.DictReader(stream)}

    raster_dirs = {
        "images": data / "tiles/images",
        "labels": data / "pseudo_labels/labels",
    }
    confidence_dir = data / "pseudo_labels/confidence"
    if confidence_dir.exists():
        raster_dirs["confidence"] = confidence_dir
    report: dict[str, object] = {"manifest_rows": len(expected), "rasters": {}}
    for name, directory in raster_dirs.items():
        actual = {path.name for path in directory.glob("*.tif")}
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        report["rasters"][name] = {
            "count": len(actual),
            "missing": missing[:10],
            "extra": extra[:10],
        }
        if missing or extra:
            raise RuntimeError(f"{name} does not match the split manifest")

    if not args.skip_hashes:
        checked = 0
        sums = root / "pretrained/SHA256SUMS"
        for line in sums.read_text(encoding="utf-8").splitlines():
            expected_hash, relative = line.split(maxsplit=1)
            relative = relative.strip()
            actual_hash = sha256(root / "pretrained" / relative)
            if actual_hash != expected_hash:
                raise RuntimeError(f"Checksum mismatch: {relative}")
            checked += 1
        report["verified_model_hashes"] = checked

    report["status"] = "ok"
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
