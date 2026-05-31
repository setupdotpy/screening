# UAV Roadside Tree/Canopy Risk Screening from RGB Images

## Overview

This project is a preliminary UAV-based roadside tree and canopy screening pipeline. It targets UAVid-style oblique aerial imagery with roads, roadside trees, low vegetation, buildings, cars, humans, and background clutter.

The pipeline uses UAVid semantic labels directly when available. If labels are not available, it falls back to the existing SegFormer/SAM segmentation path. The output is an explainable image-space risk ranking, not a real hazard probability estimate.

Because the system uses a single RGB UAV image, extracted features are image-space proxies. True tree height, trunk diameter, lean angle, and physical distance cannot be estimated reliably without camera calibration, multi-view reconstruction, or UAV LiDAR.

## Setup

```bash
cd ~/screening/roadside_tree_risk
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If using the project conda environment:

```bash
cd ~/screening/roadside_tree_risk
conda activate screening
```

## Data Layout

Place UAV RGB images in:

```text
data/images/
```

If UAVid label masks are available, place matching labels in:

```text
data/labels/
```

The label loader matches files by image stem and supports common suffixes such as `_label`, `_labelTrainIds`, `_gt`, `_gtFine_labelIds`, `_gtFine_color`, and `_color`.

## Running

Use UAVid labels when available:

```bash
python src/main.py \
  --image_dir data/images \
  --label_dir data/labels \
  --output_dir outputs \
  --use_uavid_labels
```

Run model fallback without labels:

```bash
python src/main.py --image_dir data/images --output_dir outputs
```

Optional:

```bash
python src/main.py --image_dir data/images --output_dir outputs --max_images 25
python src/main.py --image_dir data/images --output_dir outputs --inference_size 1024
```

## UAVid Classes

The UAVid label path supports:

- Building
- Road
- Static car
- Tree
- Low vegetation
- Human
- Moving car
- Background clutter

Labels are converted to semantic masks with a configurable color tolerance. The default tolerance is `5`.

## Pipeline

```text
Input UAV RGB image
-> UAVid semantic label loading or SegFormer/SAM fallback
-> road mask
-> tree mask
-> low vegetation mask
-> connected components on tree mask
-> roadside canopy filtering
-> canopy feature extraction
-> preliminary risk score
-> CSV and visualization export
```

## Why UAVid

UAVid contains UAV oblique imagery and semantic classes that are directly useful for roadside canopy analysis: `Road`, `Tree`, and `Low vegetation`. This is a better target than Cityscapes for UAV roadside imagery because Cityscapes is street-view and does not provide the same aerial viewpoint.

## Why Canopy-Level Features

Street-view assumptions such as trunk evidence and lean angle are not reliable in UAV imagery. Tree trunks are often hidden by canopy, camera angle, and occlusion. This pipeline therefore uses canopy-level image features:

- canopy area
- canopy width and height
- canopy compactness
- canopy circularity
- distance to road
- overlap with road and road buffer
- nearby tree density
- low vegetation context
- canopy asymmetry

Canopy asymmetry is only a weak 2D proxy. It is not a true lean angle.

## Model Fallback

When UAVid labels are not used or not found, the pipeline falls back to SegFormer plus SAM:

- SegFormer provides semantic context.
- SAM provides instance-like mask proposals.
- SAM masks are clipped to the semantic vegetation/tree mask.
- If SAM is unavailable or produces no usable masks, connected components are used.

This fallback is useful for exploratory testing, but UAVid labels are preferred for controlled UAVid experiments.

## Risk Score

The score is a bounded weighted ranking:

```text
risk_score =
0.35 * inverse_distance_to_road
+ 0.25 * road_buffer_overlap_ratio
+ 0.15 * normalized_canopy_size
+ 0.10 * nearby_tree_density
+ 0.10 * canopy_asymmetry_score
+ 0.05 * uncertainty
```

Risk levels:

- Low: `risk_score < 0.33`
- Medium: `0.33 <= risk_score < 0.66`
- High: `risk_score >= 0.66`

This score is for preliminary prioritization only.

## Outputs

CSV:

```text
outputs/csv/tree_features.csv
```

Visualizations:

```text
outputs/visualizations/*_risk.jpg
```

Canopy masks:

```text
outputs/masks/*_canopy_*.png
```

Visualizations draw road masks in blue, tree masks in green, low vegetation in light green, canopy boxes, rejected components, distance lines, risk labels, canopy area, and road distance.

## Limitations

- Single RGB UAV image only
- No true 3D geometry
- No metric tree height
- No real depth
- No true trunk diameter
- No reliable lean angle
- No physical clearance measurement
- Perspective distortion
- Occlusion and overlapping canopies
- UAVid labels provide semantic classes, not guaranteed individual tree instances
- SAM is not tree-specific when fallback mode is used

LiDAR or multi-view reconstruction would be needed for true tree height, lean angle, trunk geometry, canopy volume, and 3D road clearance.
