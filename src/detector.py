"""Compatibility wrapper for older imports."""

from __future__ import annotations

from dataclasses import replace
from typing import List, Optional

import numpy as np

from config import CandidateConfig, DetectorConfig, PipelineConfig, RoadConfig, SamConfig, SegmentationConfig, StructuralFilterConfig
from road_region import RoadContext, extract_road_context
from sam_splitter import split_vegetation_with_sam
from segformer_segmenter import SegFormerSegmenter
from structural_filter import TreeCandidate, apply_structural_filter
from segformer_segmenter import mask_for_labels


Detection = TreeCandidate


class TreeDetector:
    """Retained for compatibility with earlier code paths."""

    def __init__(self, config: DetectorConfig):
        segmentation = replace(
            SegmentationConfig(),
            model_name=config.model_name,
            inference_max_size=config.inference_max_size,
        )
        candidates = replace(CandidateConfig(), min_component_area_px=config.min_component_area_px)
        self.pipeline = PipelineConfig(
            segmentation=segmentation,
            candidates=candidates,
            structural=StructuralFilterConfig(),
            road=RoadConfig(),
        )
        self.segmenter = SegFormerSegmenter(self.pipeline.segmentation)

    def detect(self, image: np.ndarray, road_context: Optional[RoadContext] = None) -> List[TreeCandidate]:
        segmentation = self.segmenter.segment(image)
        context = road_context or extract_road_context(segmentation, self.pipeline.road, image.shape)
        vegetation_mask = mask_for_labels(segmentation.class_masks, self.pipeline.segmentation.vegetation_classes)
        vegetation_components = split_vegetation_with_sam(
            image_bgr=image,
            vegetation_mask=vegetation_mask,
            road_context_mask=context.combined_mask,
            config=SamConfig(),
        )
        accepted, _ = apply_structural_filter(vegetation_components, context, image.shape, self.pipeline.structural)
        return accepted
