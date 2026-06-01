# UAV Roadside Canopy Inspection Priority Screening

## Overview

This project performs preliminary inspection-priority screening for roadside tree/canopy regions in UAV RGB imagery. It targets UAVid-style oblique aerial scenes containing roads, roadside trees, low vegetation, buildings, vehicles, humans, and background clutter.

The proposed score should not be interpreted as a direct estimate of tree failure probability. It is a preliminary image-based ranking score for selecting roadside canopy regions that may require further inspection.

The pipeline uses UAVid semantic labels directly when available. If labels are not available, it falls back to the existing SegFormer/SAM segmentation path.

## Setup

```bash
cd ~/screening/roadside_tree_risk
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

With the project conda environment:

```bash
cd ~/screening/roadside_tree_risk
conda activate screening
```

## Data Layout

Place UAV RGB images in:

```text
data/images/
```

Place matching UAVid label masks in:

```text
data/labels/
```

The label loader matches files by image stem and supports common suffixes such as `_label`, `_labelTrainIds`, `_gt`, `_gtFine_labelIds`, `_gtFine_color`, and `_color`.

## Running

Use UAVid labels:

```bash
python src/main.py \
  --image_dir data/images \
  --label_dir data/labels \
  --output_dir outputs \
  --use_uavid_labels
```

Use model fallback:

```bash
python src/main.py --image_dir data/images --output_dir outputs
```

## Fine-Tuning SegFormer On UAVid

This project includes a lightweight fine-tuning script for the labeled UAVid split:

```text
data/archive/uavid_train/seq1  images=600 labels=600
data/archive/uavid_val/seq16   images=70  labels=70
```

The local GPU is suitable for SegFormer-B0 with conservative settings: `512` image size, batch size `1`, gradient accumulation, and CUDA mixed precision.

Smoke test:

```bash
conda run -n screening python -u src/train_segformer_uavid.py \
  --train_dir data/archive/uavid_train \
  --val_dir data/archive/uavid_val \
  --output_dir models/segformer-uavid-smoke \
  --image_size 256 \
  --epochs 1 \
  --batch_size 1 \
  --gradient_accumulation_steps 1 \
  --max_train_samples 2 \
  --max_val_samples 2 \
  --num_workers 0 \
  --eval_every_steps 0
```

Recommended full fine-tuning command for this machine:

```bash
conda run -n screening python -u src/train_segformer_uavid.py \
  --train_dir data/archive/uavid_train \
  --val_dir data/archive/uavid_val \
  --output_dir models/segformer-uavid \
  --model_name nvidia/segformer-b0-finetuned-ade-512-512 \
  --image_size 512 \
  --epochs 8 \
  --batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 6e-5 \
  --num_workers 2 \
  --eval_every_steps 100
```

Run prediction with the fine-tuned checkpoint:

```bash
python src/main.py \
  --image_dir data/uavid_test_one_per_seq \
  --output_dir outputs_finetuned_uavid \
  --segformer_model models/segformer-uavid/best \
  --inference_size 768
```

## Pipeline

```text
Input UAV image
-> load UAVid label or predicted segmentation
-> extract road mask
-> extract tree mask
-> extract low vegetation mask
-> connected components on tree mask
-> filter tree/canopy candidates
-> compute road-context features
-> compute canopy-structure features
-> compute inspection_priority_score
-> save CSV and visualization
```

## Canopy Indicators

The system uses image-space canopy proxies, including:

- canopy area, width, height, and diameter
- canopy compactness
- canopy circularity
- canopy asymmetry
- canopy gap ratio
- canopy edge roughness
- canopy irregularity
- RGB green ratio
- RGB brightness mean and standard deviation
- distance to road
- road and road-buffer overlap

Large, irregular, sparse, or asymmetric canopies near roads may deserve further inspection. These indicators do not prove structural instability.

## Inspection Priority Score

```text
inspection_priority_score =
0.30 * inverse_distance_to_road
+ 0.20 * road_buffer_overlap_ratio
+ 0.15 * normalized_canopy_size
+ 0.15 * canopy_irregularity
+ 0.10 * canopy_gap_ratio
+ 0.10 * canopy_asymmetry_score
```

Levels:

- Low: `inspection_priority_score < 0.33`
- Medium: `0.33 <= inspection_priority_score < 0.66`
- High: `inspection_priority_score >= 0.66`

This is a ranking score for inspection order, not a hazard probability.

## Outputs

CSV:

```text
outputs/csv/tree_features.csv
```

Visualizations:

```text
outputs/visualizations/*_priority.jpg
```

Canopy masks:

```text
outputs/masks/*_canopy_*.png
```

Visualizations show road masks, tree/canopy masks, low vegetation masks, inspection-priority candidates, bounding boxes, priority labels, canopy area, canopy irregularity, and distance lines.

## Limits Of RGB-Only UAV Imagery

Single UAV RGB images cannot determine:

- tree lean angle
- trunk decay
- root damage
- internal disease
- true tree height
- trunk diameter
- physical road clearance
- metric distance without calibration

Canopy features are image-space proxies. UAV LiDAR, calibrated multi-view reconstruction, or field inspection is required for real hazard assessment.
