from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


@dataclass(frozen=True)
class Sample:
    tile_name: str
    source_id: str
    country: str
    city: str
    segmentation_split: str
    da_domain: str
    da_usage: str
    dg_domain: str
    dg_usage: str

    @property
    def region(self) -> str:
        return f"{self.country}_{self.city}"


def read_manifest(path: Path) -> list[Sample]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [Sample(**row) for row in csv.DictReader(handle)]


def choose_samples(samples: list[Sample], task: str, split: str) -> list[Sample]:
    if task == "segmentation":
        return [s for s in samples if s.segmentation_split == split]
    if task == "da":
        if split == "source_train":
            return [s for s in samples if s.da_usage == "labeled_train"]
        if split == "source_val":
            return [s for s in samples if s.da_usage == "source_validation"]
        if split == "target_train":
            return [s for s in samples if s.da_usage == "unlabeled_adaptation"]
        if split == "target_test":
            return [s for s in samples if s.da_usage == "evaluation"]
    if task == "dg":
        if split == "source_train":
            return [s for s in samples if s.dg_usage == "labeled_train"]
        if split == "source_val":
            return [s for s in samples if s.dg_usage == "source_validation"]
        if split == "unseen_test":
            return [s for s in samples if s.dg_usage == "unseen_evaluation"]
    raise ValueError(f"Unsupported task/split: {task}/{split}")


def _read_rgb(path: Path) -> torch.Tensor:
    with rasterio.open(path) as src:
        array = src.read((1, 2, 3), out_dtype="uint8")
    return torch.from_numpy(array.astype(np.float32) / 255.0)


def _read_label(path: Path) -> torch.Tensor:
    with rasterio.open(path) as src:
        array = src.read(1, out_dtype="uint8")
    # OpenEarthMap reference masks use IDs 1--8; PyTorch losses use 0--7.
    return torch.from_numpy(array.astype(np.int64) - 1)


def normalize(image: torch.Tensor) -> torch.Tensor:
    return (image - IMAGENET_MEAN) / IMAGENET_STD


def strong_augment(image: torch.Tensor) -> torch.Tensor:
    """Tensor-only strong augmentation used by MIC-style consistency."""
    out = image.clone()
    gain = 0.65 + 0.7 * torch.rand(3, 1, 1)
    bias = (torch.rand(3, 1, 1) - 0.5) * 0.25
    out = (out * gain + bias).clamp(0, 1)
    if random.random() < 0.5:
        out = F.avg_pool2d(out[None], 3, 1, 1)[0]
    return out


class OpticalDataset(Dataset):
    def __init__(
        self,
        samples: list[Sample],
        image_dir: Path,
        label_dir: Path,
        input_size: int,
        train: bool,
        labeled: bool = True,
        strong_view: bool = False,
    ) -> None:
        self.samples = samples
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.input_size = input_size
        self.train = train
        self.labeled = labeled
        self.strong_view = strong_view

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = _read_rgb(self.image_dir / sample.tile_name)
        label = _read_label(self.label_dir / sample.tile_name) if self.labeled else None

        if self.train and image.shape[-1] > self.input_size:
            top = random.randint(0, image.shape[-2] - self.input_size)
            left = random.randint(0, image.shape[-1] - self.input_size)
            image = image[:, top : top + self.input_size, left : left + self.input_size]
            if label is not None:
                label = label[top : top + self.input_size, left : left + self.input_size]
        elif image.shape[-2:] != (self.input_size, self.input_size):
            image = F.interpolate(image[None], (self.input_size, self.input_size), mode="bilinear", align_corners=False)[0]
            if label is not None:
                label = F.interpolate(label[None, None].float(), (self.input_size, self.input_size), mode="nearest")[0, 0].long()

        if self.train:
            if random.random() < 0.5:
                image = image.flip(-1)
                if label is not None:
                    label = label.flip(-1)
            if random.random() < 0.5:
                image = image.flip(-2)
                if label is not None:
                    label = label.flip(-2)

        weak = normalize(image)
        result = {"image": weak, "name": sample.tile_name, "region": sample.region}
        if label is not None:
            result["label"] = label
        if self.strong_view:
            result["strong_image"] = normalize(strong_augment(image))
        return result
