# UAV Roadside Canopy Inspection Priority Screening

This repository contains a prototype pipeline for ranking roadside tree-canopy
regions in UAV RGB imagery for follow-up human inspection.

The system **does not identify hazardous trees directly** and it **does not
predict tree-failure probability**. A single RGB UAV image cannot reliably
measure tree height, trunk diameter, lean angle, root damage, or internal decay.
The output is an inspection-priority score for deciding which roadside canopy
regions should be reviewed first.

## What The Pipeline Does

The pipeline combines semantic segmentation, mask splitting, road-context
features, canopy-shape proxies, and uncertainty-aware scoring:

```text
UAV RGB image
-> SegFormer semantic probability map
-> road and tree/canopy masks
-> SAM mask proposals clipped by the tree mask
-> connected-component fallback if SAM fails
-> roadside canopy candidate filtering
-> road-context and canopy-proxy features
-> base inspection-priority score
-> uncertainty score
-> final Low / Medium / High priority label
```

SegFormer answers **what each pixel is**: road, tree, low vegetation, building,
car, human, background clutter, etc. SAM is used only as an instance-like mask
splitter: it helps separate adjacent canopy blobs, but its proposals are clipped
by the SegFormer tree mask so that road, car, and building pixels are not
accepted as tree candidates. If SAM returns no useful masks, the pipeline falls
back to connected components on the tree mask.

## Current Model And Dataset

The project uses UAVid-style oblique aerial scenes. The dataset used during
development is UAVid v1:

- official site: <https://uavid.nl/>
- Kaggle mirror used for download:
  <https://www.kaggle.com/datasets/dasmehdixtr/uavid-v1>

The current fine-tuned checkpoint is:

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

The model was fine-tuned for only 15 epochs, so the current outputs should be
treated as preliminary. Longer training, stronger augmentation, and
validation-based checkpoint selection are expected to improve segmentation and
downstream ranking.

## Setup

Recommended conda environment:

```bash
conda activate screening
pip install -r requirements.txt
```

Virtual environment alternative:

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If PyTorch must be installed for a specific CUDA version, install the matching
PyTorch build first, then install the remaining requirements.

## Data Layout

For ordinary runs, place UAV RGB images in:

```text
data/images/
```

Place matching UAVid label masks in:

```text
data/labels/
```

The label loader matches files by image stem and supports common suffixes such
as `_label`, `_labelTrainIds`, `_gt`, `_gtFine_labelIds`, `_gtFine_color`, and
`_color`.

The local UAVid archive used during development has this layout:

```text
data/archive/uavid_train/seq1/Images
data/archive/uavid_train/seq1/Labels
data/archive/uavid_val/seq16/Images
data/archive/uavid_val/seq16/Labels
data/archive/uavid_test/<seq>/Images
```

## Running The Pipeline

Use UAVid label masks for controlled testing:

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

Run a bounded train-sequence sanity check:

```bash
python src/main.py \
  --image_dir data/archive/uavid_train/seq1/Images \
  --output_dir outputs_uncertainty_seq1_20_continued10 \
  --segformer_model models/segformer-uavid-continued-10/best \
  --inference_size 768 \
  --max_images 20
```

The train-derived seq1 run is useful for debugging because it checks whether a
preliminary 15-epoch model can produce masks, candidates, scores, and
visualizations under familiar conditions. It is not evidence of generalization.
Use unseen one-per-sequence outputs for the main failure analysis.

## Inspection-Priority Score

For each accepted canopy candidate, the pipeline computes road-context features
and canopy-proxy features:

- distance to road
- road-buffer overlap
- canopy area, width, height, and diameter
- compactness and circularity
- asymmetry
- gap ratio
- edge roughness
- RGB green ratio and brightness statistics

These are image-space proxies. They can support triage, but they do not diagnose
tree health or structural failure.

The base score is:

```text
base_inspection_score =
0.30 * inverse_distance_to_road
+ 0.20 * road_buffer_overlap_ratio
+ 0.15 * normalized_canopy_size
+ 0.15 * canopy_irregularity
+ 0.10 * canopy_gap_ratio
+ 0.10 * canopy_asymmetry_score
```

The inverse distance term is computed from road distance:

```text
inverse_distance_to_road = exp(-distance_to_road / lambda)
```

This makes the score larger when the canopy is closer to the road. Road
proximity receives the largest weight because the goal is roadside inspection
priority, not general tree health classification.

## Uncertainty Handling

The uncertainty score is:

```text
uncertainty_score =
0.50 * segmentation_entropy_uncertainty
+ 0.30 * tree_probability_uncertainty
+ 0.20 * instance_merge_uncertainty
```

The terms mean:

- `segmentation_entropy_uncertainty`: SegFormer is unsure between classes.
- `tree_probability_uncertainty`: mean tree probability inside the candidate is
  low.
- `instance_merge_uncertainty`: the canopy is much larger than the median
  candidate in the image and may contain multiple merged trees.

The final priority score is:

```text
final_priority_score =
clip(base_inspection_score + 0.20 * uncertainty_score, 0, 1)
```

Priority labels:

- Low: `final_priority_score < 0.33`
- Medium: `0.33 <= final_priority_score < 0.66`
- High: `final_priority_score >= 0.66`

The uncertainty term can raise ambiguous road-adjacent candidates for human
review, but it is capped so that it cannot dominate the road-context and canopy
features.

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

## Current Results

The main generalization evidence comes from unseen one-per-sequence UAVid test
images. The train-derived seq1 outputs are kept as a debugging sanity check.

```text
Train-derived seq1 uncertainty run:
images = 20
accepted canopies = 510
Low = 112
Medium = 398
High = 0
mean base score = 0.3365
mean uncertainty score = 0.2658
mean final score = 0.3897

Unseen one-per-sequence uncertainty run:
images = 13
accepted canopies = 209
Low = 72
Medium = 137
High = 0
mean base score = 0.3238
mean uncertainty score = 0.1604
mean final score = 0.3559
mean tree probability = 0.8671
```

No High labels appeared in these runs because the pipeline does not observe
physical failure evidence. This is expected for an RGB-only inspection-priority
prototype.

## Failure Modes

The main failure cases in the unseen outputs are:

- small, distant, sparse, or low-contrast trees can be missed
- adjacent tree crowns can remain merged into one candidate
- SAM can over-split or under-split canopy regions
- wrong road masks can distort distance and road-buffer overlap
- road-adjacent fragments can be promoted as false positives
- scene and viewpoint shifts reduce reliability

Practical fixes include lower size thresholds, multi-scale inference, longer
training, more unseen validation sequences, stronger tree-instance supervision,
better SAM prompting/filtering, manual or calibrated road geometry, and expert
review for high-uncertainty candidates.

## Limitations

This project is useful for triage, not deployment-level tree risk assessment.
Important limitations:

- no true 3D geometry
- no metric tree height
- no physical road clearance estimate
- no reliable trunk location
- no trunk decay detection
- no root damage detection
- no internal disease detection
- canopy asymmetry is not tree lean
- SAM is not tree-specific
- UAVid semantic labels are not individual-tree ground truth
- uncertainty is an engineering approximation, not Bayesian posterior
  uncertainty

## LiDAR Extension

UAV LiDAR is the most important future extension because it adds 3D structure.
The point-cloud paper included in this project suggests extracting tree points
and recognizing individual street-tree instances using centroid prediction and
clustering. A similar extension could:

- split merged RGB canopy blobs into 3D tree instances
- measure tree height and crown volume
- estimate true road clearance
- locate trunks and separate neighboring crowns
- compute metric distance from each tree to the road edge
- validate high-uncertainty RGB candidates against geometry
- detect crown loss, leaning change, or growth toward the roadway over time

## References

- Xie et al., "SegFormer," NeurIPS 2021:
  <https://papers.neurips.cc/paper/2021/hash/64f1f27bf1b4ec22924fd0acb550c235-Abstract.html>
- Kirillov et al., "Segment Anything," ICCV 2023:
  <https://arxiv.org/abs/2304.02643>
- Lyu et al., "UAVid," ISPRS JPRS 2020:
  <https://uavid.nl/>
- UAVid Kaggle mirror used during development:
  <https://www.kaggle.com/datasets/dasmehdixtr/uavid-v1>
- Liu et al., "Instance recognition of street trees from urban point clouds,"
  ISPRS JPRS 2023:
  <https://doi.org/10.1016/j.isprsjprs.2023.04.010>
