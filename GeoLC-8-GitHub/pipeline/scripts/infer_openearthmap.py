#!/usr/bin/env python3
"""Generate OpenEarthMap pseudo labels for RGB GeoTIFF patches."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import rasterio
import segmentation_models_pytorch as smp
import torch


DEFAULT_WEIGHTS = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "openearthmap_original"
    / "u-efficientnet-b4_s0_CELoss_pretrained.pth"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--confidence-dir", type=Path, required=True)
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help=(
            "Original OpenEarthMap WACV 2023 U-Net/EfficientNet-B4 checkpoint "
            f"(default: {DEFAULT_WEIGHTS})"
        ),
    )
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--tta", choices=["none", "flip4"], default="flip4")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--threads", type=int, default=0)
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(weights: Path, device: torch.device) -> torch.nn.Module:
    if not weights.is_file():
        raise FileNotFoundError(f"OpenEarthMap checkpoint not found: {weights}")
    print(f"weights={weights.resolve()}")
    model = smp.Unet(
        classes=9,
        in_channels=3,
        activation=None,
        encoder_weights=None,
        encoder_name="efficientnet-b4",
        decoder_attention_type="scse",
    )
    state = torch.load(weights, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def predict(model: torch.nn.Module, image: torch.Tensor, tta: str) -> tuple[np.ndarray, np.ndarray]:
    with torch.inference_mode():
        if tta == "none":
            probabilities = torch.softmax(model(image), dim=1)
        else:
            augmented = torch.cat(
                [
                    image,
                    torch.flip(image, dims=(-1,)),
                    torch.flip(image, dims=(-2,)),
                    torch.flip(image, dims=(-2, -1)),
                ],
                dim=0,
            )
            original, flip_x, flip_y, flip_xy = torch.softmax(
                model(augmented), dim=1
            ).chunk(4, dim=0)
            probabilities = (
                original
                + torch.flip(flip_x, dims=(-1,))
                + torch.flip(flip_y, dims=(-2,))
                + torch.flip(flip_xy, dims=(-2, -1))
            ) / 4
    # The official OpenEarthMap demo suppresses class 0 before argmax for
    # valid imagery. Our input patches contain no NoData pixels, so retain
    # only the eight semantic land-cover classes (1..8).
    probabilities = torch.cat(
        [torch.zeros_like(probabilities[:, :1]), probabilities[:, 1:]], dim=1
    )
    confidence, labels = probabilities.max(dim=1)
    return (
        labels.to("cpu").numpy().astype(np.uint8),
        np.rint(confidence.to("cpu").numpy() * 255).astype(np.uint8),
    )


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    print(f"device={device}, torch={torch.__version__}, tta={args.tta}")
    model = load_model(args.weights, device)

    args.label_dir.mkdir(parents=True, exist_ok=True)
    args.confidence_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(args.image_dir.glob("*.tif"))
    if args.limit is not None:
        images = images[: args.limit]

    started = time.time()
    pending: list[tuple[int, Path]] = []
    for index, image_path in enumerate(images, start=1):
        label_path = args.label_dir / image_path.name
        confidence_path = args.confidence_dir / image_path.name
        if args.resume and label_path.exists() and confidence_path.exists():
            print(f"[{index}/{len(images)}] already complete: {image_path.name}")
        else:
            pending.append((index, image_path))

    processed = 0
    for offset in range(0, len(pending), args.batch_size):
        batch_started = time.time()
        batch = pending[offset : offset + args.batch_size]
        rgbs: list[np.ndarray] = []
        profiles: list[dict] = []
        for _, image_path in batch:
            with rasterio.open(image_path) as source:
                rgbs.append(source.read((1, 2, 3)).astype(np.float32) / 255.0)
                profiles.append(source.profile.copy())

        tensor = torch.from_numpy(np.stack(rgbs)).to(device)
        labels, confidence = predict(model, tensor, args.tta)
        elapsed = time.time() - batch_started

        for batch_index, ((index, image_path), profile) in enumerate(zip(batch, profiles)):
            label_path = args.label_dir / image_path.name
            confidence_path = args.confidence_dir / image_path.name
            profile.update(
                count=1,
                dtype="uint8",
                nodata=None,
                compress="zstd",
                zstd_level=1,
                predictor=2,
                tiled=True,
                blockxsize=512,
                blockysize=512,
            )
            with rasterio.open(label_path, "w", **profile) as output:
                output.write(labels[batch_index], 1)
            with rasterio.open(confidence_path, "w", **profile) as output:
                output.write(confidence[batch_index], 1)

            mean_confidence = float(confidence[batch_index].mean() / 255.0)
            print(
                f"[{index}/{len(images)}] {image_path.name}: batch={len(batch)}, "
                f"batch_elapsed={elapsed:.2f}s, mean_confidence={mean_confidence:.3f}"
            )
        processed += len(batch)

    total = time.time() - started
    print(f"processed={processed}, elapsed_seconds={total:.1f}")


if __name__ == "__main__":
    main()
