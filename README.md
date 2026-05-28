# Preliminary Roadside Tree Detection, Feature Extraction, and Risk Scoring Pipeline

## Project Summary

This repository implements a reproducible feasibility pipeline for preliminary roadside tree and vegetation risk screening from monocular roadside imagery. The system detects vegetation candidates, estimates a road region, extracts interpretable image-space features, assigns a transparent risk score, and exports tabular and visual outputs for review.

The project is designed as an engineering screening deliverable rather than a production hazard assessment tool. It prioritizes modular implementation, explainable assumptions, deterministic outputs, and documented failure modes over custom model training or maximum detection accuracy.

## Technical Scope

The pipeline supports three operational modes:

- Pretrained Ultralytics YOLO segmentation when model inference is available.
- HSV vegetation fallback using color thresholding, morphology, and connected components.
- Manual road polygon override for images where the bottom-of-image road assumption is weak.

No custom training is required. The pretrained YOLO weights are downloaded automatically by Ultralytics on first use.

## Repository Layout

```text
roadside_tree_risk/
├── data/
│   └── images/
├── outputs/
│   ├── csv/
│   ├── visualizations/
│   └── masks/
├── src/
│   ├── main.py
│   ├── config.py
│   ├── detector.py
│   ├── road_region.py
│   ├── feature_extraction.py
│   ├── risk_scoring.py
│   ├── visualization.py
│   └── utils.py
├── requirements.txt
├── README.md
└── report_template.md
```

## Installation

From the project directory:

```bash
cd ~/screening/roadside_tree_risk
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If the system exposes Python as `python3` instead of `python`, create the environment with `python3 -m venv .venv`. After activation, the expected command remains `python src/main.py ...`.

## Dataset Sources

Recommended dataset:

- Mapillary Vistas: https://www.mapillary.com/dataset/vistas

Backup datasets:

- Cityscapes: https://www.cityscapes-dataset.com/
- BDD100K: https://bdd-data.berkeley.edu/

The code also accepts arbitrary roadside or custom user images. For a screening submission, a small curated sample of representative roadside images is sufficient if the limitations and image source are clearly documented.

## Input Data

Place images in:

```text
data/images/
```

Supported image extensions:

```text
.jpg, .jpeg, .png, .bmp, .tif, .tiff, .webp
```

## Execution

Primary command:

```bash
cd ~/screening/roadside_tree_risk
python src/main.py --image_dir data/images --output_dir outputs
```

Fallback-only command:

```bash
python src/main.py --image_dir data/images --output_dir outputs --force_fallback
```

Quick subset command:

```bash
python src/main.py --image_dir data/images --output_dir outputs --max_images 25
```

Manual road polygon command:

```bash
python src/main.py --image_dir data/images --output_dir outputs --road_polygon "120,420;820,420;960,719;0,719"
```

The manual polygon uses image pixel coordinates in `x,y` order.

## Outputs

The pipeline writes:

- `outputs/csv/tree_features.csv`
- `outputs/visualizations/*_risk.jpg`
- `outputs/masks/*_tree_*.png`

The CSV is intended to support inspection, ranking, and failure analysis. The visualizations are intended to verify whether the road approximation, vegetation mask, lean estimate, and risk level are plausible for each image.

## Methodology

The processing flow is:

```text
input image
-> tree or vegetation detection
-> road region estimation
-> feature extraction
-> risk scoring
-> CSV export
-> visualization and mask export
```

Detection first attempts to use a pretrained YOLO segmentation model. If YOLO is unavailable or does not produce useful tree or vegetation detections, the pipeline falls back to HSV thresholding for green vegetation. The fallback assigns a fixed confidence value of `0.5`, reflecting moderate uncertainty.

Road estimation uses the following priority:

1. Road labels, if a segmentation mask and road label values are provided programmatically.
2. Manual road polygon from `--road_polygon`, if supplied.
3. Bottom-image polygon heuristic.

## Extracted Features

For each detected component, the following features are exported:

- `image_name`
- `tree_id`
- `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2`
- `confidence`
- `mask_area_px`
- `bbox_height_px`
- `canopy_width_px`
- `distance_to_road_px`
- `lean_dx_px`
- `lean_toward_road`
- `overhang_ratio`
- `normalized_tree_size`
- `uncertainty`
- `risk_score`
- `risk_level`

Lean is estimated by comparing the horizontal center of the upper 40 percent of mask pixels with the lower 30 percent of mask pixels. If the horizontal displacement points toward the estimated road centroid, `lean_toward_road` is set to `True`.

## Risk Scoring

The score is intentionally simple, bounded, and auditable. Each risk factor is converted to a value in `[0, 1]`, then combined with fixed weights:

```text
risk_score =
0.35 * inverse_distance_to_road
+ 0.25 * lean_toward_road
+ 0.25 * overhang_ratio
+ 0.10 * normalized_tree_size
+ 0.05 * uncertainty
```

The distance term uses a smooth exponential proximity function:

```text
inverse_distance_to_road = exp(-distance_to_road_px / distance_scale_px)
```

With the default `distance_scale_px = 250`, a tree touching or overlapping the road receives a distance term near `1.0`; farther trees decay smoothly toward `0.0`. This avoids a discontinuous cutoff while preserving the expected monotonic behavior: closer vegetation receives higher risk contribution.

Risk levels:

- Low: `risk_score < 0.33`
- Medium: `0.33 <= risk_score < 0.66`
- High: `risk_score >= 0.66`

The result should be interpreted as a prioritization score for review, not as an engineering determination of tree stability.

## Mathematical Reliability Rationale

The scoring logic is designed to be mathematically reasonable for a preliminary image-based ranking system:

- All feature contributions are bounded in `[0, 1]`, preventing a single raw pixel measurement from dominating because of image resolution.
- The weighted sum is also clamped to `[0, 1]`, making risk levels stable and interpretable.
- Distance-to-road is monotonic: smaller distance always increases risk contribution.
- Overhang ratio is a true ratio: tree pixels overlapping the road divided by total tree pixels.
- Normalized size is scale-aware within the image: mask area divided by image area.
- Uncertainty is tied directly to detector confidence: `uncertainty = 1 - confidence`.
- Lean is intentionally binary because monocular imagery does not provide reliable metric trunk angle.

Reliability still depends on detection quality and road-region quality. The visualization output is therefore part of the method, not only a presentation artifact: it allows each scored component to be inspected against the image evidence.

## Reproducibility Notes

- All paths are relative to the project directory.
- The pipeline does not require custom training.
- The fallback detector allows the project to run even when model weights cannot be downloaded.
- Output folders are created automatically.
- CSV column order is fixed in `src/main.py`.
- `--max_images` can be used for deterministic subset runs during report iteration.

## Assumptions

- Vegetation masks are treated as candidate roadside tree or vegetation components.
- In the absence of road labels, the lower image region is used as an approximate road proxy.
- Pixel-space measurements are useful for preliminary ranking but do not represent metric distances.
- The fallback detector is most effective when vegetation has visible green color.
- A manual road polygon can improve results for unusual camera viewpoints.

## Limitations

- Image-space approximation only
- No true depth estimation
- No metric tree height
- Perspective distortion
- Occlusion
- Segmentation uncertainty
- Shadows and lighting sensitivity
- Green non-tree objects may be detected as vegetation
- Leafless trees, dead branches, and trunks without foliage may be missed

## Failure Cases

Known failure cases include:

- Winter imagery or dry vegetation with weak green signal
- Dense vegetation where individual trees merge into one component
- Roads hidden by parked vehicles, shadows, or foreground objects
- Camera viewpoints where the road is not near the lower image region
- Overhanging branches with little visible green foliage
- Segmentation masks that confuse shrubs, grass, and trees

## Suggested Evaluation Protocol

For a formal screening submission:

1. Use a small, documented image set with varied viewpoints, lighting, and tree-road geometry.
2. Run the default pipeline and inspect every visualization.
3. Record clear successes, partial successes, and failures.
4. Compare YOLO-assisted output with `--force_fallback` on at least a few images.
5. Include the CSV summary and representative visualizations in the report.

## Future Improvements

- Use a segmentation model trained on vegetation, tree, road, sidewalk, and curb classes.
- Add dataset-specific label ingestion for Mapillary or Cityscapes annotations.
- Estimate camera calibration and perspective geometry.
- Add monocular depth or stereo depth where available.
- Separate trunks, canopies, shrubs, and grass.
- Add human review tools for uncertain detections.
- Validate risk scores against field inspection labels.

## How UAV LiDAR Could Improve the System

UAV LiDAR would address the largest technical limitation: reliance on 2D image-space approximations. LiDAR could provide true 3D geometry, accurate tree height, trunk orientation, canopy volume, real-world distance estimation, structural analysis, and improved robustness to occlusion. A LiDAR-enhanced workflow could distinguish overhanging canopy from background vegetation and support risk thresholds in meters rather than pixels.
