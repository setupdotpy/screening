"""PyTorch dataset utilities for UAVid SegFormer fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from uavid_labels import uavid_label_to_class_ids


@dataclass(frozen=True)
class UAVidSample:
    image_path: Path
    label_path: Path


class UAVidSegmentationDataset(Dataset):
    """Dataset for UAVid folders with seq*/Images and seq*/Labels."""

    def __init__(
        self,
        root_dir: Path,
        image_size: int = 512,
        color_tolerance: int = 5,
        max_samples: int | None = None,
        augment: bool = False,
    ):
        self.root_dir = Path(root_dir)
        self.image_size = max(1, int(image_size))
        self.color_tolerance = int(color_tolerance)
        self.augment = bool(augment)
        self.samples = discover_uavid_samples(self.root_dir)
        if max_samples is not None:
            self.samples = self.samples[: max(int(max_samples), 0)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        image_bgr = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
        label = cv2.imread(str(sample.label_path), cv2.IMREAD_UNCHANGED)
        if image_bgr is None:
            raise ValueError(f"Could not read image: {sample.image_path}")
        if label is None:
            raise ValueError(f"Could not read label: {sample.label_path}")

        image_bgr = cv2.resize(image_bgr, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        label_ids = uavid_label_to_class_ids(
            label,
            target_shape=(self.image_size, self.image_size),
            tolerance=self.color_tolerance,
            ignore_index=255,
        )

        if self.augment and np.random.random() < 0.5:
            image_bgr = np.ascontiguousarray(image_bgr[:, ::-1])
            label_ids = np.ascontiguousarray(label_ids[:, ::-1])

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_rgb = (image_rgb - mean) / std
        pixel_values = torch.from_numpy(image_rgb.transpose(2, 0, 1))
        labels = torch.from_numpy(label_ids.astype(np.int64))
        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "image_name": sample.image_path.name,
        }


def discover_uavid_samples(root_dir: Path) -> list[UAVidSample]:
    samples: list[UAVidSample] = []
    for seq_dir in sorted(Path(root_dir).glob("seq*")):
        image_dir = seq_dir / "Images"
        label_dir = seq_dir / "Labels"
        if not image_dir.exists() or not label_dir.exists():
            continue
        for image_path in sorted(image_dir.iterdir()):
            if not image_path.is_file():
                continue
            label_path = label_dir / image_path.name
            if label_path.exists():
                samples.append(UAVidSample(image_path=image_path, label_path=label_path))
    return samples


def collate_segmentation_batch(batch: list[dict[str, torch.Tensor | str]]) -> dict[str, torch.Tensor | list[str]]:
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),  # type: ignore[index]
        "labels": torch.stack([item["labels"] for item in batch]),  # type: ignore[index]
        "image_names": [str(item["image_name"]) for item in batch],
    }
