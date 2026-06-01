"""Command-line entry point for UAV roadside canopy inspection-priority screening."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import List

import numpy as np

from config import CSV_FIELDS, PipelineConfig, RoadConfig, SamConfig, SegmentationConfig, ensure_output_dirs
from candidate_extraction import build_connected_component_candidates, normalize_candidate_masks
from feature_extraction import extract_features
from risk_scoring import score_inspection_priority
from road_region import extract_road_context
from segformer_segmenter import SegFormerSegmenter, mean_confidence
from structural_filter import apply_structural_filter, distance_buffer_from_map, distance_map_to_mask
from sam_splitter import split_vegetation_with_sam
from segformer_segmenter import mask_for_labels
from uavid_labels import load_uavid_segmentation
from utils import list_images, mask_boundary_pixels, read_image, write_csv
from visualization import save_masks, save_visualization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Roadside tree screening from RGB images.")
    parser.add_argument("--image_dir", default="data/images", help="Directory containing input images.")
    parser.add_argument("--label_dir", default="data/labels", help="Directory containing UAVid label masks.")
    parser.add_argument("--output_dir", default="outputs", help="Directory for CSV, masks, and visualizations.")
    parser.add_argument("--use_uavid_labels", action="store_true", help="Use UAVid ground-truth label masks when available.")
    parser.add_argument("--max_images", type=int, default=None, help="Optional maximum number of images to process.")
    parser.add_argument(
        "--road_polygon",
        default=None,
        help='Optional manual road polygon as "x1,y1;x2,y2;x3,y3". Uses image pixel coordinates.',
    )
    parser.add_argument(
        "--inference_size",
        type=int,
        default=1024,
        help="Maximum SegFormer inference dimension in pixels.",
    )
    parser.add_argument(
        "--segformer_model",
        default=None,
        help="Optional SegFormer checkpoint path/name for model fallback prediction.",
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


def build_config(
    inference_size: int,
    road_polygon: str | None,
    use_uavid_labels: bool = False,
    segformer_model: str | None = None,
) -> PipelineConfig:
    segmentation = replace(
        SegmentationConfig(),
        inference_max_size=max(1, int(inference_size)),
        model_name=segformer_model or SegmentationConfig().model_name,
    )
    road = RoadConfig(manual_polygon=parse_polygon(road_polygon))
    return PipelineConfig(segmentation=segmentation, road=road, sam=SamConfig(), use_uavid_labels=use_uavid_labels)


def run_pipeline(
    image_dir: Path,
    output_dir: Path,
    label_dir: Path | None = None,
    road_polygon: str | None = None,
    max_images: int | None = None,
    inference_size: int = 1024,
    use_uavid_labels: bool = False,
    segformer_model: str | None = None,
) -> List[dict]:
    config = build_config(
        inference_size=inference_size,
        road_polygon=road_polygon,
        use_uavid_labels=use_uavid_labels,
        segformer_model=segformer_model,
    )

    ensure_output_dirs(output_dir)
    segmenter = None if use_uavid_labels else SegFormerSegmenter(config.segmentation)
    rows: List[dict] = []

    image_paths = list_images(image_dir)
    if max_images is not None:
        image_paths = image_paths[: max(max_images, 0)]

    if not image_paths:
        print(f"No images found in {image_dir}. Add images and rerun the command.")
        write_csv(rows, output_dir / "csv" / "tree_features.csv", CSV_FIELDS)
        return rows

    for image_path in image_paths:
        try:
            image = read_image(image_path)
        except Exception as exc:  # noqa: BLE001 - continue processing other images.
            print(f"WARNING: skipping {image_path.name}: {exc}")
            continue

        try:
            segmentation = None
            used_uavid_labels = False
            if use_uavid_labels:
                try:
                    segmentation = load_uavid_segmentation(
                        image_path=image_path,
                        label_dir=label_dir or Path("data/labels"),
                        image_shape=image.shape,
                        tolerance=config.label_color_tolerance,
                    )
                    used_uavid_labels = True
                except Exception as exc:  # noqa: BLE001 - requested fallback behavior.
                    print(f"WARNING: UAVid label unavailable for {image_path.name}: {exc}. Falling back to model prediction.")
                    if segmenter is None:
                        segmenter = SegFormerSegmenter(config.segmentation)

            if segmentation is None:
                if segmenter is None:
                    segmenter = SegFormerSegmenter(config.segmentation)
                segmentation = segmenter.segment(image)

            road_context = extract_road_context(segmentation, config.road, image.shape)
            tree_mask = mask_for_labels(segmentation.class_masks, config.segmentation.tree_classes)
            low_vegetation_mask = mask_for_labels(segmentation.class_masks, config.segmentation.low_vegetation_classes)

            if used_uavid_labels:
                vegetation_candidates = build_connected_component_candidates(
                    vegetation_mask=tree_mask,
                    candidate_config=config.candidates,
                    vegetation_overlap=1.0,
                    used_sam_splitting=False,
                )
                for candidate in vegetation_candidates:
                    candidate["source"] = "uavid_label"
                    candidate["segmentation_confidence"] = 1.0
                split_stats = {
                    "used_sam": False,
                    "sam_masks_generated": 0,
                    "sam_masks_kept_after_overlap": 0,
                    "post_split_candidates": len(vegetation_candidates),
                    "connected_components_count": len(vegetation_candidates),
                    "fallback_used": False,
                    "fallback_reason": None,
                }
            else:
                vegetation_candidates = split_vegetation_with_sam(
                    image_bgr=image,
                    vegetation_mask=tree_mask,
                    road_context_mask=road_context.combined_mask,
                    config=config.sam,
                )
                vegetation_candidates = normalize_candidate_masks(vegetation_candidates)
                for candidate in vegetation_candidates:
                    candidate["segmentation_confidence"] = mean_confidence(segmentation.confidence_map, candidate["mask"])
                split_stats = getattr(split_vegetation_with_sam, "last_stats", {})

            accepted_candidates, rejected_candidates = apply_structural_filter(
                vegetation_candidates,
                road_context,
                image.shape,
                config.structural,
            )

            feature_results = []
            image_rows = []
            road_distance_map = distance_map_to_mask(road_context.combined_mask)
            road_buffer_mask = distance_buffer_from_map(road_distance_map, config.risk.road_buffer_radius_px)
            road_edge_pixels = mask_boundary_pixels(road_context.combined_mask)
            uncertainty_source = "label_mask_no_entropy" if used_uavid_labels else "model_probabilities"
            for tree_id, candidate in enumerate(accepted_candidates, start=1):
                feature_result = extract_features(
                    image_path.name,
                    tree_id,
                    candidate,
                    road_context,
                    image.shape,
                    all_candidates=accepted_candidates,
                    low_vegetation_mask=low_vegetation_mask,
                    risk_config=config.risk,
                    road_distance_map=road_distance_map,
                    road_buffer_mask=road_buffer_mask,
                    road_edge_pixels=road_edge_pixels,
                    image_bgr=image,
                    segmentation_entropy_map=segmentation.entropy_map,
                    tree_probability_map=segmentation.tree_probability_map,
                    uncertainty_source=uncertainty_source,
                )
                feature_results.append(feature_result)

            canopy_areas = [float(result.row["canopy_area_px"]) for result in feature_results]
            median_canopy_area = float(np.median(canopy_areas)) if canopy_areas else 0.0
            for feature_result in feature_results:
                scored_row = score_inspection_priority(
                    feature_result.row,
                    config.risk,
                    median_canopy_area=median_canopy_area,
                )
                rows.append(scored_row)
                image_rows.append(scored_row)

            save_masks(image_path.name, accepted_candidates, output_dir)
            sam_used = bool(split_stats.get("used_sam", False))
            if used_uavid_labels:
                splitting_label = "UAVid labels + canopy components"
            elif sam_used:
                splitting_label = "SegFormer + SAM canopy proposals"
            else:
                splitting_label = "SegFormer + connected components fallback"
            save_visualization(
                image=image,
                image_name=image_path.name,
                segmentation=segmentation,
                road_context=road_context,
                tree_mask=tree_mask,
                low_vegetation_mask=low_vegetation_mask,
                accepted_candidates=accepted_candidates,
                rejected_candidates=rejected_candidates,
                feature_results=feature_results,
                splitting_label=splitting_label,
                output_dir=output_dir,
            )

            average_priority = (
                sum(float(row["final_priority_score"]) for row in image_rows) / len(image_rows)
                if image_rows
                else 0.0
            )
            print(
                f"Processed {image_path.name} | "
                f"segmentation={segmentation.elapsed_seconds:.2f}s | "
                f"segmentation_source={segmentation.source} | "
                f"sam_masks_generated={int(split_stats.get('sam_masks_generated', 0))} | "
                f"sam_masks_kept={int(split_stats.get('sam_masks_kept_after_overlap', 0))} | "
                f"post_split_candidates={int(split_stats.get('post_split_candidates', 0))} | "
                f"connected_components={int(split_stats.get('connected_components_count', 0))} | "
                f"accepted_canopies={len(accepted_candidates)} | "
                f"avg_priority={average_priority:.3f}"
            )
        except Exception as exc:  # noqa: BLE001 - keep pipeline moving.
            print(f"WARNING: failed on {image_path.name}: {exc}")
            continue

    write_csv(rows, output_dir / "csv" / "tree_features.csv", CSV_FIELDS)
    print(f"Wrote CSV: {output_dir / 'csv' / 'tree_features.csv'}")
    return rows


def main() -> None:
    args = parse_args()
    run_pipeline(
        Path(args.image_dir),
        Path(args.output_dir),
        label_dir=Path(args.label_dir),
        road_polygon=args.road_polygon,
        max_images=args.max_images,
        inference_size=args.inference_size,
        use_uavid_labels=args.use_uavid_labels,
        segformer_model=args.segformer_model,
    )


if __name__ == "__main__":
    main()
