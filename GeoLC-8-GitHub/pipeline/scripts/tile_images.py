#!/usr/bin/env python3
"""Create full 1024x1024 RGB GeoTIFF tiles and reject zero-heavy windows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window


TILE_SIZE = 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tile-manifest", type=Path, required=True)
    parser.add_argument("--max-zero-fraction", type=float, default=0.0)
    parser.add_argument("--max-cloud-fraction", type=float, default=0.10)
    parser.add_argument("--cloud-min-mean", type=float, default=185.0)
    parser.add_argument("--cloud-min-channel", type=float, default=150.0)
    parser.add_argument("--cloud-max-saturation", type=float, default=0.30)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.max_zero_fraction <= 1:
        raise ValueError("--max-zero-fraction must be between 0 and 1")
    if not 0 <= args.max_cloud_fraction <= 1:
        raise ValueError("--max-cloud-fraction must be between 0 and 1")

    with args.manifest.open(encoding="utf-8", newline="") as stream:
        sources = list(csv.DictReader(stream))

    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        args.tile_manifest.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "tile_name",
        "source_id",
        "country",
        "city",
        "source_image",
        "row_offset",
        "column_offset",
        "zero_fraction",
        "cloud_fraction",
        "left",
        "bottom",
        "right",
        "top",
        "crs",
        "gsd_m",
        "cloud_percent",
        "quality",
    ]
    records: list[dict[str, object]] = []

    for source_number, source in enumerate(sources, start=1):
        source_image = args.source_dir / f'{source["country"]}_{source["city"]}.tif'
        accepted = 0
        rejected_zero = 0
        rejected_cloud = 0
        with rasterio.open(source_image) as dataset:
            full_rows = dataset.height // TILE_SIZE
            full_columns = dataset.width // TILE_SIZE
            for tile_row in range(full_rows):
                row_offset = tile_row * TILE_SIZE
                strip_window = Window(0, row_offset, dataset.width, TILE_SIZE)
                strip = dataset.read(window=strip_window)

                for tile_column in range(full_columns):
                    column_offset = tile_column * TILE_SIZE
                    patch = strip[
                        :,
                        :,
                        column_offset : column_offset + TILE_SIZE,
                    ]
                    zero_fraction = float(np.all(patch == 0, axis=0).mean())
                    if zero_fraction > args.max_zero_fraction:
                        rejected_zero += 1
                        continue

                    rgb = patch[:3].astype(np.float32)
                    channel_max = rgb.max(axis=0)
                    channel_min = rgb.min(axis=0)
                    channel_mean = rgb.mean(axis=0)
                    saturation = (channel_max - channel_min) / np.maximum(
                        channel_max, 1.0
                    )
                    cloud_mask = (
                        (channel_mean >= args.cloud_min_mean)
                        & (channel_min >= args.cloud_min_channel)
                        & (saturation <= args.cloud_max_saturation)
                    )
                    cloud_fraction = float(cloud_mask.mean())
                    if cloud_fraction > args.max_cloud_fraction:
                        rejected_cloud += 1
                        continue

                    accepted += 1
                    tile_name = (
                        f'{source["country"]}_{source["city"]}_{accepted:06d}.tif'
                    )
                    tile_window = Window(
                        column_offset, row_offset, TILE_SIZE, TILE_SIZE
                    )
                    bounds = rasterio.windows.bounds(tile_window, dataset.transform)
                    records.append(
                        {
                            "tile_name": tile_name,
                            "source_id": source["source_id"],
                            "country": source["country"],
                            "city": source["city"],
                            "source_image": source_image.name,
                            "row_offset": row_offset,
                            "column_offset": column_offset,
                            "zero_fraction": f"{zero_fraction:.8f}",
                            "cloud_fraction": f"{cloud_fraction:.8f}",
                            "left": bounds[0],
                            "bottom": bounds[1],
                            "right": bounds[2],
                            "top": bounds[3],
                            "crs": str(dataset.crs),
                            "gsd_m": source["gsd_m"],
                            "cloud_percent": source["cloud_percent"],
                            "quality": source["quality"],
                        }
                    )

                    if not args.dry_run:
                        profile = dataset.profile.copy()
                        profile.update(
                            width=TILE_SIZE,
                            height=TILE_SIZE,
                            transform=dataset.window_transform(tile_window),
                            count=3,
                            compress="zstd",
                            zstd_level=1,
                            predictor=2,
                            tiled=True,
                            blockxsize=512,
                            blockysize=512,
                            num_threads="ALL_CPUS",
                        )
                        with rasterio.open(args.output_dir / tile_name, "w", **profile) as out:
                            out.write(patch[:3])

                print(
                    f'[{source_number}/{len(sources)}] {source["country"]}_'
                    f'{source["city"]}: row {tile_row + 1}/{full_rows}, '
                    f"kept={accepted}, rejected_zero={rejected_zero}, "
                    f"rejected_cloud={rejected_cloud}"
                )

        print(
            f'completed {source["source_id"]}: kept={accepted}, '
            f"rejected_zero={rejected_zero}, rejected_cloud={rejected_cloud}, "
            f"edge_pixels=({dataset.width % TILE_SIZE}, {dataset.height % TILE_SIZE})"
        )

    if not args.dry_run:
        with args.tile_manifest.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

    print(f"total accepted tiles: {len(records)}")


if __name__ == "__main__":
    main()
