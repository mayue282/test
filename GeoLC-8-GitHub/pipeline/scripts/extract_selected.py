#!/usr/bin/env python3
"""Extract the single RGB GeoTIFF from each selected source archive."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from zipfile import ZipFile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace = args.workspace.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with args.manifest.open(encoding="utf-8", newline="") as stream:
        sources = list(csv.DictReader(stream))

    for index, source in enumerate(sources, start=1):
        archive = workspace / source["archive_relpath"]
        output = output_dir / f'{source["country"]}_{source["city"]}.tif'
        with ZipFile(archive) as bundle:
            members = [
                member
                for member in bundle.infolist()
                if member.filename.lower().endswith((".tif", ".tiff"))
            ]
            if len(members) != 1:
                raise RuntimeError(
                    f"Expected one TIFF in {archive}, found {len(members)}"
                )
            member = members[0]
            if output.exists() and output.stat().st_size == member.file_size:
                print(f"[{index}/{len(sources)}] already extracted: {output.name}")
                continue

            temporary = output.with_suffix(output.suffix + ".part")
            with bundle.open(member) as source_stream, temporary.open("wb") as target:
                shutil.copyfileobj(source_stream, target, length=16 * 1024 * 1024)
            if temporary.stat().st_size != member.file_size:
                raise RuntimeError(f"Size mismatch while extracting {archive}")
            temporary.replace(output)
            print(f"[{index}/{len(sources)}] extracted: {output.name}")


if __name__ == "__main__":
    main()
