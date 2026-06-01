# UAV Roadside Canopy Inspection Priority Screening

## Overview

This project performs preliminary inspection-priority screening for roadside tree and canopy regions in UAV RGB imagery. It targets UAVid-style oblique aerial scenes containing roads, roadside trees, low vegetation, buildings, vehicles, humans, and background clutter.

The output is not a tree-failure probability. It is an image-based ranking score for selecting roadside canopy regions that may require further human inspection.

The pipeline supports two segmentation modes:

- UAVid label mode: use ground-truth UAVid label masks directly for controlled testing.
- Model mode: use a SegFormer semantic segmentation model, optionally fine-tuned on UAVid, with SAM used only as a mask proposal splitter.

## Setup

Recommended conda environment:

```bash
cd ~/screening/roadside_tree_risk
conda activate screening
pip install -r requirements.txt
```

Virtual environment alternative:

```bash
cd ~/screening/roadside_tree_risk
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If PyTorch must be installed for a specific CUDA version, install the matching PyTorch build first, then install the remaining requirements.

## Data Layout

For ordinary runs, place UAV RGB images in:

```text
data/images/
```

Place matching UAVid label masks in:

```text
data/labels/
```

The label loader matches files by image stem and supports common suffixes such as `_label`, `_labelTrainIds`, `_gt`, `_gtFine_labelIds`, `_gtFine_color`, and `_color`.

The local UAVid archive used during development has this layout:

```text
data/archive/uavid_train/seq1/Images
data/archive/uavid_train/seq1/Labels
data/archive/uavid_val/seq16/Images
data/archive/uavid_val/seq16/Labels
data/archive/uavid_test/<seq>/Images
```

## Running The Pipeline

Use UAVid label masks:

```bash
python src/main.py \
  --image_dir data/images \
  --label_dir data/labels \
  --output_dir outputs \
  --use_uavid_labels
```

Use the default SegFormer model:

```bash
python src/main.py \
  --image_dir data/images \
  --output_dir outputs
```

Use the fine-tuned UAVid checkpoint:

```bash
python src/main.py \
  --image_dir data/uavid_test_one_per_seq \
  --output_dir outputs_finetuned_uavid \
  --segformer_model models/segformer-uavid-continued-10/best \
  --inference_size 768
```

Run a bounded seq1-only test from the archive:

```bash
python src/main.py \
  --image_dir data/archive/uavid_train/seq1/Images \
  --output_dir outputs_uncertainty_seq1_archive20_continued10 \
  --segformer_model models/segformer-uavid-continued-10/best \
  --inference_size 768 \
  --max_images 20
```

## Pipeline

```text
Input UAV RGB image
-> load UAVid label mask or predict SegFormer segmentation
-> extract road, tree/canopy, and low vegetation masks
-> split predicted vegetation with SAM proposals when available
-> fall back to connected components when SAM splitting is unavailable
-> filter roadside canopy candidates
-> compute road-context features
-> compute canopy-structure and RGB proxy features
-> compute base_inspection_score
-> compute uncertainty_score
-> compute final_priority_score
-> save CSV, canopy masks, and visualizations
```

## SegFormer And SAM

SegFormer provides semantic context: road, tree/canopy, low vegetation, buildings, cars, humans, and background clutter. For UAVid work, the useful classes are Road, Tree, Low vegetation, Building, Static car, Moving car, Human, and Background clutter.

SegFormer masks can merge nearby trees into large connected canopy blobs. SAM is therefore used only as an instance-like mask proposal splitter. SAM proposals are intersected with the SegFormer tree/canopy mask so that cars, buildings, people, and roads are not accepted only because SAM proposed them.

If SAM is unavailable, fails to load, returns no masks, or no SAM masks pass vegetation overlap filtering, the pipeline falls back to connected components on the tree/canopy mask.

## Fine-Tuning SegFormer On UAVid

The training script fine-tunes SegFormer on UAVid-style image/label pairs:

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

The continued checkpoint used in recent tests is:

```text
models/segformer-uavid-continued-10/best
```

Recent validation metrics for that checkpoint:

```text
mean_iou = 0.5107
pixel_accuracy = 0.8297
road_iou = 0.6833
tree_iou = 0.7141
low_vegetation_iou = 0.6087
```

## Canopy Features

For each accepted canopy candidate, the CSV includes:

- canopy area, width, height, and diameter
- canopy compactness
- canopy circularity
- canopy asymmetry score
- canopy gap ratio
- canopy edge roughness
- canopy irregularity
- RGB green ratio
- RGB brightness mean and standard deviation
- distance to road
- road overlap and road-buffer overlap
- normalized canopy size

These are image-space proxies. Large, irregular, sparse, or asymmetric canopy regions near roads may deserve earlier inspection, but these features do not diagnose tree health or structural failure.

## Uncertainty-Aware Inspection Priority

The final priority score includes a base inspection score and an uncertainty score.

```text
base_inspection_score =
0.30 * inverse_distance_to_road
+ 0.20 * road_buffer_overlap_ratio
+ 0.15 * normalized_canopy_size
+ 0.15 * canopy_irregularity
+ 0.10 * canopy_gap_ratio
+ 0.10 * canopy_asymmetry_score

uncertainty_score =
0.50 * segmentation_entropy_uncertainty
+ 0.30 * tree_probability_uncertainty
+ 0.20 * instance_merge_uncertainty

final_priority_score =
clip(base_inspection_score + 0.20 * uncertainty_score, 0, 1)
```

Priority levels:

- Low: `final_priority_score < 0.33`
- Medium: `0.33 <= final_priority_score < 0.66`
- High: `final_priority_score >= 0.66`

Uncertainty terms:

- `segmentation_entropy_uncertainty`: mean normalized SegFormer entropy inside the canopy mask.
- `mean_tree_probability`: mean SegFormer tree probability inside the canopy mask.
- `tree_probability_uncertainty`: `1 - mean_tree_probability`.
- `instance_merge_uncertainty`: high when a canopy component is much larger than the median accepted canopy area in the same image.

If UAVid ground-truth labels are used, entropy uncertainty is set to zero and tree probability is set to one because label masks do not provide model probability distributions. The uncertainty source is recorded as `label_mask_no_entropy`. For model predictions, the uncertainty source is `model_probabilities`.

The final score is not a tree-failure probability. It is an image-based inspection-priority score that increases when a canopy region is both potentially relevant to the roadway and uncertain enough to require further human inspection.

## Outputs

CSV:

```text
outputs/csv/tree_features.csv
```

Visualizations:

```text
outputs/visualizations/*_priority.jpg
```

Accepted canopy masks:

```text
outputs/masks/*_canopy_*.png
```

Visualization labels use:

```text
T{id} B{base_score} U{uncertainty_score} P{final_score} {level}
```

## Recent Seq1 Test

A bounded 20-image seq1 archive test with `models/segformer-uavid-continued-10/best` produced:

```text
images processed = 20
accepted canopies = 510
visualizations = 20
mask files = 510
Low priority = 112
Medium priority = 398
High priority = 0
mean base_inspection_score = 0.3365
mean uncertainty_score = 0.2658
mean final_priority_score = 0.3897
```

The uncertainty term changed the ranking by increasing final scores by `0.20 * uncertainty_score`. It improves inspection-priority ranking behavior, not segmentation accuracy.

## Limitations

Because the system uses a single RGB UAV image, extracted features are image-space proxies. True tree height, trunk diameter, lean angle, and physical distance cannot be estimated reliably without camera calibration, multi-view reconstruction, or UAV LiDAR.

Additional limitations:

- no true 3D geometry
- no metric tree height
- no physical clearance estimate
- no reliable trunk location
- no trunk decay detection
- no root damage detection
- no internal disease detection
- canopy asymmetry is not tree lean
- SAM is not tree-specific
- UAVid semantic labels are not individual tree ground truth
- overlapping canopies may still merge
- uncertainty is an engineering approximation, not Bayesian posterior uncertainty

## Future Work

Future work should include UAV LiDAR, calibrated multi-view reconstruction, individual tree instance segmentation, temporal monitoring, field validation, and better probabilistic uncertainty estimation.
