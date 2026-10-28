#!/usr/bin/env python3
"""Audit OpenEarthMap pseudo-label and confidence GeoTIFF outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--confidence-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected = {path.name for path in args.image_dir.glob("*.tif")}
    labels = {path.name for path in args.label_dir.glob("*.tif")}
    confidence = {path.name for path in args.confidence_dir.glob("*.tif")}

    errors: list[str] = []
    if labels != expected:
        errors.append(
            f"label filenames differ: missing={len(expected - labels)}, "
            f"extra={len(labels - expected)}"
        )
    if confidence != expected:
        errors.append(
            f"confidence filenames differ: missing={len(expected - confidence)}, "
            f"extra={len(confidence - expected)}"
        )

    histogram = np.zeros(9, dtype=np.int64)
    confidence_sum = 0
    pixel_count = 0
    confidence_min = 255
    confidence_max = 0

    for name in sorted(expected & labels & confidence):
        with (
            rasterio.open(args.image_dir / name) as image_source,
            rasterio.open(args.label_dir / name) as label_source,
            rasterio.open(args.confidence_dir / name) as confidence_source,
        ):
            for kind, source in (
                ("label", label_source),
                ("confidence", confidence_source),
            ):
                if source.width != 1024 or source.height != 1024:
                    errors.append(f"{name}: {kind} shape is not 1024x1024")
                if source.count != 1 or source.dtypes[0] != "uint8":
                    errors.append(f"{name}: {kind} is not single-band uint8")
                if source.crs != image_source.crs or source.transform != image_source.transform:
                    errors.append(f"{name}: {kind} georeferencing differs from image")

            label = label_source.read(1)
            conf = confidence_source.read(1)
            if label.min() < 1 or label.max() > 8:
                errors.append(
                    f"{name}: label range {int(label.min())}..{int(label.max())}"
                )
            histogram += np.bincount(label.ravel(), minlength=9)[:9]
            confidence_sum += int(conf.sum(dtype=np.uint64))
            pixel_count += conf.size
            confidence_min = min(confidence_min, int(conf.min()))
            confidence_max = max(confidence_max, int(conf.max()))

    report = {
        "status": "ok" if not errors else "failed",
        "expected_images": len(expected),
        "label_files": len(labels),
        "confidence_files": len(confidence),
        "label_histogram": {str(index): int(value) for index, value in enumerate(histogram)},
        "mean_confidence": confidence_sum / pixel_count / 255 if pixel_count else None,
        "confidence_min_uint8": confidence_min if pixel_count else None,
        "confidence_max_uint8": confidence_max if pixel_count else None,
        "errors": errors[:100],
        "error_count": len(errors),
    }
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
