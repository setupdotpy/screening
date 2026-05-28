"""Select images with annotated roadside vegetation.

This utility uses Mapillary-style semantic label masks. It selects images where
the label mask contains both road pixels and vegetation pixels close to the road,
then copies the corresponding images to a pipeline input directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from config import IMAGE_EXTENSIONS


@dataclass(frozen=True)
class LabelColors:
    road: tuple[tuple[int, int, int], ...]
    vegetation: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class Candidate:
    image_path: Path
    label_path: Path
    road_pixels: int
    vegetation_pixels: int
    roadside_vegetation_pixels: int
    roadside_ratio: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select images with vegetation close to annotated road pixels.")
    parser.add_argument("--image_dir", required=True, help="Directory containing source images.")
    parser.add_argument("--label_dir", required=True, help="Directory containing semantic label PNG masks.")
    parser.add_argument("--config", required=True, help="Mapillary-style config JSON with label colors.")
    parser.add_argument("--output_dir", default="data/roadside_tree_images", help="Directory to copy selected images into.")
    parser.add_argument("--report_csv", default=None, help="Optional CSV report path.")
    parser.add_argument("--max_images", type=int, default=100, help="Maximum selected images to copy.")
    parser.add_argument("--max_scan", type=int, default=None, help="Optional maximum number of label masks to scan.")
    parser.add_argument(
        "--resize_width",
        type=int,
        default=1024,
        help="Resize label masks to this width before scoring. Use 0 to keep original resolution.",
    )
    parser.add_argument("--road_dilate_px", type=int, default=80, help="Road dilation radius for roadside contact.")
    parser.add_argument("--min_road_pixels", type=int, default=20_000, help="Minimum road area required.")
    parser.add_argument("--min_vegetation_pixels", type=int, default=20_000, help="Minimum vegetation area required.")
    parser.add_argument(
        "--min_roadside_vegetation_pixels",
        type=int,
        default=2_500,
        help="Minimum vegetation pixels within the dilated road region.",
    )
    return parser.parse_args()


def load_label_colors(config_path: Path) -> LabelColors:
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    road_colors: list[tuple[int, int, int]] = []
    vegetation_colors: list[tuple[int, int, int]] = []
    for label in config.get("labels", []):
        text = f"{label.get('readable', '')} {label.get('name', '')}".lower()
        color = tuple(int(value) for value in label["color"])
        if "vegetation" in text or "tree" in text:
            vegetation_colors.append(color)
        elif "road" in text:
            road_colors.append(color)

    if not road_colors:
        raise ValueError(f"No road labels found in {config_path}")
    if not vegetation_colors:
        raise ValueError(f"No vegetation/tree labels found in {config_path}")

    return LabelColors(road=tuple(road_colors), vegetation=tuple(vegetation_colors))


def color_mask(label_rgb: np.ndarray, colors: Iterable[tuple[int, int, int]]) -> np.ndarray:
    mask = np.zeros(label_rgb.shape[:2], dtype=bool)
    for color in colors:
        mask |= np.all(label_rgb == np.array(color, dtype=np.uint8), axis=2)
    return mask


def matching_image(label_path: Path, image_dir: Path) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        image_path = image_dir / f"{label_path.stem}{extension}"
        if image_path.exists():
            return image_path
    return None


def score_candidate(
    image_path: Path,
    label_path: Path,
    colors: LabelColors,
    road_dilate_px: int,
    resize_width: int,
) -> Candidate | None:
    label_bgr = cv2.imread(str(label_path), cv2.IMREAD_COLOR)
    if label_bgr is None:
        return None

    scale_factor = 1.0
    if resize_width > 0 and label_bgr.shape[1] > resize_width:
        scale_factor = resize_width / float(label_bgr.shape[1])
        resize_height = max(1, round(label_bgr.shape[0] * scale_factor))
        label_bgr = cv2.resize(label_bgr, (resize_width, resize_height), interpolation=cv2.INTER_NEAREST)

    label_rgb = cv2.cvtColor(label_bgr, cv2.COLOR_BGR2RGB)
    road_mask = color_mask(label_rgb, colors.road)
    vegetation_mask = color_mask(label_rgb, colors.vegetation)

    road_pixels = int(np.count_nonzero(road_mask))
    vegetation_pixels = int(np.count_nonzero(vegetation_mask))
    if road_pixels == 0 or vegetation_pixels == 0:
        return Candidate(image_path, label_path, road_pixels, vegetation_pixels, 0, 0.0)

    scaled_dilate_px = max(1, round(road_dilate_px * scale_factor))
    kernel_size = max(1, scaled_dilate_px * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    road_neighborhood = cv2.dilate(road_mask.astype(np.uint8), kernel) > 0
    roadside_vegetation_pixels = int(np.count_nonzero(vegetation_mask & road_neighborhood))
    roadside_ratio = roadside_vegetation_pixels / float(max(vegetation_pixels, 1))
    area_scale = 1.0 / max(scale_factor * scale_factor, 1e-6)

    return Candidate(
        image_path=image_path,
        label_path=label_path,
        road_pixels=round(road_pixels * area_scale),
        vegetation_pixels=round(vegetation_pixels * area_scale),
        roadside_vegetation_pixels=round(roadside_vegetation_pixels * area_scale),
        roadside_ratio=roadside_ratio,
    )


def candidate_passes(
    candidate: Candidate,
    min_road_pixels: int,
    min_vegetation_pixels: int,
    min_roadside_vegetation_pixels: int,
) -> bool:
    return (
        candidate.road_pixels >= min_road_pixels
        and candidate.vegetation_pixels >= min_vegetation_pixels
        and candidate.roadside_vegetation_pixels >= min_roadside_vegetation_pixels
    )


def write_report(candidates: list[Candidate], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_name",
                "label_name",
                "road_pixels",
                "vegetation_pixels",
                "roadside_vegetation_pixels",
                "roadside_ratio",
            ],
        )
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(
                {
                    "image_name": candidate.image_path.name,
                    "label_name": candidate.label_path.name,
                    "road_pixels": candidate.road_pixels,
                    "vegetation_pixels": candidate.vegetation_pixels,
                    "roadside_vegetation_pixels": candidate.roadside_vegetation_pixels,
                    "roadside_ratio": round(candidate.roadside_ratio, 4),
                }
            )


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir)
    label_dir = Path(args.label_dir)
    output_dir = Path(args.output_dir)
    report_csv = Path(args.report_csv) if args.report_csv else output_dir / "selection_report.csv"

    colors = load_label_colors(Path(args.config))
    scored: list[Candidate] = []
    label_paths = sorted(label_dir.glob("*.png"))
    if args.max_scan is not None:
        label_paths = label_paths[: max(args.max_scan, 0)]

    for index, label_path in enumerate(label_paths, start=1):
        image_path = matching_image(label_path, image_dir)
        if image_path is None:
            continue
        candidate = score_candidate(image_path, label_path, colors, args.road_dilate_px, args.resize_width)
        if candidate and candidate_passes(
            candidate,
            args.min_road_pixels,
            args.min_vegetation_pixels,
            args.min_roadside_vegetation_pixels,
        ):
            scored.append(candidate)
        if index % 100 == 0:
            print(f"Scanned {index}/{len(label_paths)} labels; selected candidates: {len(scored)}")

    scored.sort(key=lambda item: item.roadside_vegetation_pixels, reverse=True)
    selected = scored[: max(args.max_images, 0)]

    output_dir.mkdir(parents=True, exist_ok=True)
    for candidate in selected:
        shutil.copy2(candidate.image_path, output_dir / candidate.image_path.name)

    write_report(selected, report_csv)
    print(f"Selected {len(selected)} images into {output_dir}")
    print(f"Wrote report: {report_csv}")


if __name__ == "__main__":
    main()
