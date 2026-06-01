"""Configuration defaults for the roadside tree screening pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, Tuple


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

USE_SAM_SPLITTING = True
USE_UAVID_LABELS = False
SAM_MODEL_NAME = "sam_b.pt"
SAM_IMAGE_SIZE = 1024
SAM_MIN_MASK_AREA = 500
SAM_MAX_MASK_AREA_RATIO = 0.80
VEGETATION_OVERLAP_THRESHOLD = 0.60
MASK_DUPLICATE_IOU_THRESHOLD = 0.85
POST_SPLIT_MERGED_VEGETATION = True
POST_SPLIT_MIN_WIDTH_PX = 180
POST_SPLIT_MIN_AREA_RATIO = 0.015
POST_SPLIT_VALLEY_RATIO = 0.86
POST_SPLIT_MIN_CHILD_AREA = 500
POST_SPLIT_MIN_CHILD_WIDTH_PX = 40
POST_SPLIT_MAX_CHILDREN = 4
MIN_TREE_HEIGHT_RATIO = 0.15
MIN_TREE_AREA_RATIO = 0.005
MIN_BBOX_HEIGHT_PX = 80
TREE_SCORE_THRESHOLD = 0.45
SPLIT_TREE_SCORE_THRESHOLD = 0.40
ROAD_BUFFER_RADIUS = 25
MIN_TREE_AREA_PX = 150
MIN_CANOPY_WIDTH_PX = 10
MIN_CANOPY_HEIGHT_PX = 10
MAX_TREE_DISTANCE_TO_ROAD_PX = 250
NEARBY_TREE_RADIUS_PX = 150
LABEL_COLOR_TOLERANCE = 5
EDGE_ROUGHNESS_MIN = 3.0
EDGE_ROUGHNESS_MAX = 15.0
USE_INSPECTION_PRIORITY_TERMINOLOGY = True


@dataclass(frozen=True)
class SegmentationConfig:
    model_name: str = "nvidia/segformer-b1-finetuned-cityscapes-1024-1024"
    inference_max_size: int = 1024
    default_confidence: float = 0.5
    vegetation_classes: Tuple[str, ...] = ("vegetation", "tree")
    tree_classes: Tuple[str, ...] = ("tree", "vegetation")
    low_vegetation_classes: Tuple[str, ...] = ("low vegetation", "terrain")
    road_context_classes: Tuple[str, ...] = ("road", "sidewalk")
    ignored_classes: Tuple[str, ...] = (
        "building",
        "wall",
        "fence",
        "pole",
        "traffic light",
        "traffic sign",
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
        "sky",
    )


@dataclass(frozen=True)
class CandidateConfig:
    min_component_area_px: int = MIN_TREE_AREA_PX
    morph_kernel_size: int = 5
    morph_close_iterations: int = 2


@dataclass(frozen=True)
class StructuralFilterConfig:
    tree_score_threshold: float = TREE_SCORE_THRESHOLD
    split_tree_score_threshold: float = SPLIT_TREE_SCORE_THRESHOLD
    min_height_ratio: float = MIN_TREE_HEIGHT_RATIO
    min_area_ratio: float = MIN_TREE_AREA_RATIO
    min_vertical_continuity: float = 0.20
    min_canopy_spread: float = 0.15
    min_bbox_height_px: int = MIN_BBOX_HEIGHT_PX
    min_tree_area_px: int = MIN_TREE_AREA_PX
    min_canopy_width_px: int = MIN_CANOPY_WIDTH_PX
    min_canopy_height_px: int = MIN_CANOPY_HEIGHT_PX
    max_tree_distance_to_road_px: float = MAX_TREE_DISTANCE_TO_ROAD_PX


@dataclass(frozen=True)
class RiskConfig:
    distance_scale_px: float = 250.0
    road_buffer_radius_px: int = ROAD_BUFFER_RADIUS
    nearby_tree_radius_px: int = NEARBY_TREE_RADIUS_PX
    low_vegetation_buffer_radius_px: int = NEARBY_TREE_RADIUS_PX
    edge_roughness_min: float = EDGE_ROUGHNESS_MIN
    edge_roughness_max: float = EDGE_ROUGHNESS_MAX
    use_inspection_priority_terminology: bool = USE_INSPECTION_PRIORITY_TERMINOLOGY
    weights: Tuple[float, float, float, float, float, float] = (0.30, 0.20, 0.15, 0.15, 0.10, 0.10)


@dataclass(frozen=True)
class RoadConfig:
    manual_polygon: Sequence[tuple[int, int]] | None = None
    heuristic_top_fraction: float = 0.64
    heuristic_side_margin_fraction: float = 0.18


@dataclass(frozen=True)
class PipelineConfig:
    sam: "SamConfig" = field(default_factory=lambda: SamConfig())
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    candidates: CandidateConfig = field(default_factory=CandidateConfig)
    structural: StructuralFilterConfig = field(default_factory=StructuralFilterConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    road: RoadConfig = field(default_factory=RoadConfig)
    use_uavid_labels: bool = USE_UAVID_LABELS
    label_color_tolerance: int = LABEL_COLOR_TOLERANCE


@dataclass(frozen=True)
class SamConfig:
    use_sam_splitting: bool = USE_SAM_SPLITTING
    model_name: str = SAM_MODEL_NAME
    image_size: int = SAM_IMAGE_SIZE
    min_mask_area: int = SAM_MIN_MASK_AREA
    max_mask_area_ratio: float = SAM_MAX_MASK_AREA_RATIO
    vegetation_overlap_threshold: float = VEGETATION_OVERLAP_THRESHOLD
    duplicate_iou_threshold: float = MASK_DUPLICATE_IOU_THRESHOLD
    post_split_merged_vegetation: bool = POST_SPLIT_MERGED_VEGETATION
    post_split_min_width_px: int = POST_SPLIT_MIN_WIDTH_PX
    post_split_min_area_ratio: float = POST_SPLIT_MIN_AREA_RATIO
    post_split_valley_ratio: float = POST_SPLIT_VALLEY_RATIO
    post_split_min_child_area: int = POST_SPLIT_MIN_CHILD_AREA
    post_split_min_child_width_px: int = POST_SPLIT_MIN_CHILD_WIDTH_PX
    post_split_max_children: int = POST_SPLIT_MAX_CHILDREN


@dataclass(frozen=True)
class DetectorConfig:
    """Compatibility config retained for older imports."""

    model_name: str = SegmentationConfig.model_name
    inference_max_size: int = SegmentationConfig.inference_max_size
    min_component_area_px: int = CandidateConfig.min_component_area_px


CSV_FIELDS = [
    "image_name",
    "tree_id",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "canopy_area_px",
    "canopy_width_px",
    "canopy_height_px",
    "canopy_diameter_px",
    "canopy_compactness",
    "canopy_circularity",
    "canopy_asymmetry_score",
    "canopy_gap_ratio",
    "canopy_edge_roughness",
    "canopy_irregularity",
    "rgb_green_ratio",
    "rgb_brightness_mean",
    "rgb_brightness_std",
    "distance_to_road_px",
    "inverse_distance_to_road",
    "road_overlap_ratio",
    "road_buffer_overlap_ratio",
    "normalized_canopy_size",
    "inspection_priority_score",
    "inspection_priority_level",
    "candidate_source",
]


def ensure_output_dirs(output_dir: Path) -> None:
    """Create the output subdirectories used by the pipeline."""
    (output_dir / "csv").mkdir(parents=True, exist_ok=True)
    (output_dir / "visualizations").mkdir(parents=True, exist_ok=True)
    (output_dir / "masks").mkdir(parents=True, exist_ok=True)
