#!/usr/bin/env python3
"""Create deterministic segmentation, DA, and DG split manifests."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unseen-regions", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.tiles.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))

    by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_source[row["source_id"]].append(row)

    ranked_sources = sorted(by_source, key=lambda key: (len(by_source[key]), key))
    unseen_sources = set(ranked_sources[: args.unseen_regions])

    segmentation_counts: dict[str, Counter[str]] = {}
    output_rows: list[dict[str, str]] = []

    for source_id in sorted(by_source):
        source_rows = sorted(by_source[source_id], key=lambda row: row["tile_name"])
        random.Random(f"{args.seed}:{source_id}").shuffle(source_rows)

        count = len(source_rows)
        train_end = math.floor(count * 0.60)
        validation_end = train_end + math.floor(count * 0.10)

        for index, row in enumerate(source_rows):
            if index < train_end:
                segmentation_split = "train"
            elif index < validation_end:
                segmentation_split = "validation"
            else:
                segmentation_split = "test"

            domain = "target" if source_id in unseen_sources else "source"
            if domain == "target":
                da_usage = (
                    "evaluation"
                    if segmentation_split == "test"
                    else "unlabeled_adaptation"
                )
                dg_usage = "unseen_evaluation"
            else:
                da_usage = {
                    "train": "labeled_train",
                    "validation": "source_validation",
                    "test": "source_test_unused",
                }[segmentation_split]
                dg_usage = da_usage

            output_rows.append(
                {
                    "tile_name": row["tile_name"],
                    "source_id": source_id,
                    "country": row["country"],
                    "city": row["city"],
                    "segmentation_split": segmentation_split,
                    "da_domain": domain,
                    "da_usage": da_usage,
                    "dg_domain": "unseen" if domain == "target" else "source",
                    "dg_usage": dg_usage,
                }
            )

        segmentation_counts[source_id] = Counter(
            row["segmentation_split"]
            for row in output_rows
            if row["source_id"] == source_id
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(output_rows[0])
    with args.output.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(output_rows, key=lambda row: row["tile_name"]))

    region_summary = []
    for source_id in ranked_sources:
        first = by_source[source_id][0]
        counts = segmentation_counts[source_id]
        region_summary.append(
            {
                "source_id": source_id,
                "country": first["country"],
                "city": first["city"],
                "patches": len(by_source[source_id]),
                "segmentation_train": counts["train"],
                "segmentation_validation": counts["validation"],
                "segmentation_test": counts["test"],
                "da_domain": "target" if source_id in unseen_sources else "source",
                "dg_domain": "unseen" if source_id in unseen_sources else "source",
            }
        )

    report = {
        "seed": args.seed,
        "segmentation_ratio": [6, 1, 3],
        "patch_count": len(rows),
        "source_region_count": len(by_source) - len(unseen_sources),
        "target_unseen_region_count": len(unseen_sources),
        "source_patch_count": sum(
            len(by_source[source_id])
            for source_id in by_source
            if source_id not in unseen_sources
        ),
        "target_unseen_patch_count": sum(
            len(by_source[source_id]) for source_id in unseen_sources
        ),
        "regions": region_summary,
    }
    args.summary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
