"""Command-line entry point for roadside tree risk screening."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from config import DetectorConfig, PipelineConfig, RoadConfig, RiskConfig, ensure_output_dirs
from detector import TreeDetector
from feature_extraction import extract_features
from risk_scoring import add_risk_score
from road_region import estimate_road_region
from utils import list_images, read_image, write_csv
from visualization import save_masks, save_visualization


CSV_FIELDS = [
    "image_name",
    "tree_id",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "confidence",
    "mask_area_px",
    "bbox_height_px",
    "canopy_width_px",
    "distance_to_road_px",
    "lean_dx_px",
    "lean_toward_road",
    "overhang_ratio",
    "normalized_tree_size",
    "uncertainty",
    "inverse_distance_to_road",
    "risk_score",
    "risk_level",
    "detection_source",
    "road_source",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Roadside tree detection and risk scoring pipeline.")
    parser.add_argument("--image_dir", default="data/images", help="Directory containing input images.")
    parser.add_argument("--output_dir", default="outputs", help="Directory for CSV, masks, and visualizations.")
    parser.add_argument("--force_fallback", action="store_true", help="Skip YOLO and use HSV vegetation detection.")
    parser.add_argument("--max_images", type=int, default=None, help="Optional maximum number of images to process.")
    parser.add_argument(
        "--road_polygon",
        default=None,
        help='Optional manual road polygon as "x1,y1;x2,y2;x3,y3". Uses image pixel coordinates.',
    )
    return parser.parse_args()


def parse_polygon(value: str | None) -> list[tuple[int, int]] | None:
    if not value:
        return None

    polygon: list[tuple[int, int]] = []
    for point in value.split(";"):
        parts = point.split(",")
        if len(parts) != 2:
            raise ValueError(f"Invalid polygon point '{point}'. Expected x,y.")
        polygon.append((int(parts[0]), int(parts[1])))

    if len(polygon) < 3:
        raise ValueError("Manual road polygon must contain at least three points.")
    return polygon


def build_config(force_fallback: bool = False, road_polygon: str | None = None) -> PipelineConfig:
    return PipelineConfig(
        detector=DetectorConfig(force_fallback=force_fallback),
        road=RoadConfig(manual_polygon=parse_polygon(road_polygon)),
        risk=RiskConfig(),
    )


def run_pipeline(
    image_dir: Path,
    output_dir: Path,
    force_fallback: bool = False,
    road_polygon: str | None = None,
    max_images: int | None = None,
) -> List[dict]:
    config = build_config(force_fallback=force_fallback, road_polygon=road_polygon)

    ensure_output_dirs(output_dir)
    detector = TreeDetector(config.detector)
    rows: List[dict] = []

    image_paths = list_images(image_dir)
    if max_images is not None:
        image_paths = image_paths[: max(max_images, 0)]
    if not image_paths:
        print(f"No images found in {image_dir}. Add images and rerun the command.")

    for image_path in image_paths:
        image = read_image(image_path)
        detections = detector.detect(image)
        road = estimate_road_region(image, config.road)

        feature_rows = []
        distance_lines = []
        lean_arrows = []
        for tree_id, detection in enumerate(detections, start=1):
            feature_result = extract_features(image_path.name, tree_id, detection, road, image.shape)
            scored_row = add_risk_score(feature_result.row, config.risk)
            rows.append(scored_row)
            feature_rows.append(scored_row)
            distance_lines.append(feature_result.distance_line)
            lean_arrows.append(feature_result.lean_arrow)

        save_masks(image_path.name, detections, output_dir)
        save_visualization(
            image=image,
            image_name=image_path.name,
            road=road,
            detections=detections,
            feature_results=feature_rows,
            distance_lines=distance_lines,
            lean_arrows=lean_arrows,
            output_dir=output_dir,
        )
        print(f"Processed {image_path.name}: {len(detections)} vegetation components")

    write_csv(rows, output_dir / "csv" / "tree_features.csv", CSV_FIELDS)
    print(f"Wrote CSV: {output_dir / 'csv' / 'tree_features.csv'}")
    return rows


def main() -> None:
    args = parse_args()
    run_pipeline(Path(args.image_dir), Path(args.output_dir), args.force_fallback, args.road_polygon, args.max_images)


if __name__ == "__main__":
    main()
