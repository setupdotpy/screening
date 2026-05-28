"""Configuration defaults for the roadside tree screening pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class SegmentationConfig:
    model_name: str = "nvidia/segformer-b1-finetuned-cityscapes-1024-1024"
    inference_max_size: int = 1024
    fallback_confidence: float = 0.5
    vegetation_classes: Tuple[str, ...] = ("vegetation", "terrain")
    road_context_classes: Tuple[str, ...] = ("road", "sidewalk")


@dataclass(frozen=True)
class CandidateConfig:
    min_component_area_px: int = 500
    morph_kernel_size: int = 5
    morph_close_iterations: int = 2


@dataclass(frozen=True)
class StructuralFilterConfig:
    tree_score_threshold: float = 0.45
    min_height_ratio: float = 0.04
    min_area_ratio: float = 0.00005


@dataclass(frozen=True)
class RiskConfig:
    distance_normalizer_px: float = 250.0
    weights: Tuple[float, float, float, float, float] = (0.35, 0.25, 0.25, 0.10, 0.05)


@dataclass(frozen=True)
class PipelineConfig:
    segmentation: SegmentationConfig = SegmentationConfig()
    candidates: CandidateConfig = CandidateConfig()
    structural: StructuralFilterConfig = StructuralFilterConfig()
    risk: RiskConfig = RiskConfig()


CSV_FIELDS = [
    "image_name",
    "tree_id",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "mask_area_px",
    "bbox_height_px",
    "bbox_width_px",
    "canopy_width_px",
    "height_ratio",
    "aspect_ratio",
    "lower_narrowness",
    "upper_canopy_spread",
    "vertical_continuity",
    "tree_score",
    "distance_to_road_px",
    "inverse_distance_to_road",
    "lean_dx_px",
    "lean_toward_road",
    "overhang_ratio",
    "normalized_tree_size",
    "segmentation_confidence",
    "uncertainty",
    "risk_score",
    "risk_level",
    "segmentation_source",
]


def ensure_output_dirs(output_dir: Path) -> None:
    """Create all output subdirectories used by the pipeline."""
    (output_dir / "csv").mkdir(parents=True, exist_ok=True)
    (output_dir / "visualizations").mkdir(parents=True, exist_ok=True)
    (output_dir / "masks").mkdir(parents=True, exist_ok=True)
