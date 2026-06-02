# UAV Roadside Canopy Inspection Priority Screening

This repository runs a UAV RGB-image pipeline that finds roadside tree/canopy
regions and ranks them for human inspection.

Important: the score is **not** tree-failure probability. The code does not
detect internal decay, root damage, trunk condition, true lean, or real tree
height. It only produces an image-based inspection-priority ranking.

## What This Code Does

Given UAV images, the pipeline:

1. segments the scene into classes such as road, tree, low vegetation, building,
   car, human, and background clutter;
2. extracts road and tree/canopy masks;
3. uses SAM mask proposals to split large canopy blobs when possible;
4. falls back to connected components if SAM is unavailable or not useful;
5. filters candidates to keep roadside canopy regions;
6. computes road-context and canopy-shape features;
7. assigns Low, Medium, or High inspection priority;
8. saves a CSV, candidate masks, and visualization images.

The detailed mathematical explanation is in the report. This README focuses on
how to run the code and what each part of the codebase does.

## Repository Structure

```text
src/
  main.py                     Main pipeline entry point
  segformer_segmenter.py       SegFormer semantic segmentation
  sam_splitter.py              SAM-based mask proposal splitting
  candidate_extraction.py      Connected-component canopy candidates
  road_region.py               Road mask and road-buffer handling
  feature_extraction.py        Canopy and road-context feature extraction
  risk_scoring.py              Inspection-priority scoring
  visualization.py             Priority overlay and mask outputs
  train_segformer_uavid.py     SegFormer fine-tuning script
  uavid_dataset.py             UAVid dataset loader
  uavid_labels.py              UAVid label utilities

data/                          Input data location
models/                        Fine-tuned model checkpoints
outputs*/                      Generated CSVs, masks, and visualizations
requirements.txt               Python dependencies
```

## Setup

Using conda:

```bash
conda activate screening
pip install -r requirements.txt
```

Using a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you need a specific CUDA build, install the matching PyTorch version first,
then install the remaining requirements.

## Data Layout

The dataset is not committed to this repository. During development, UAVid v1
was downloaded from Kaggle and unpacked locally. The expected local layout is:

```text
data/archive/
  uavid_train/
    seq1/
      Images/
      Labels/
    ...
  uavid_val/
    seq16/
      Images/
      Labels/
    ...
  uavid_test/
    <seq>/
      Images/
```

How the data is used:

- `data/archive/uavid_train/` is used for fine-tuning SegFormer.
- `data/archive/uavid_val/` is used for validation during fine-tuning and
  checkpoint selection.
- `data/archive/uavid_train/seq1/Images` is also used for a small train-derived
  debugging run. This verifies that the preliminary model and downstream
  pipeline can produce masks, candidates, scores, and visualizations under
  familiar conditions. It is not used as generalization evidence.
- `data/uavid_test_one_per_seq/` contains one selected image from each UAVid
  test sequence. This is the main unseen test set used for result discussion and
  failure analysis because these scenes were not used during fine-tuning.

For custom runs outside the UAVid archive, place RGB images here:

```text
data/images/
```

If using semantic label masks for controlled testing, place matching labels
here:

```text
data/labels/
```

The label loader matches files by image stem and supports suffixes such as
`_label`, `_labelTrainIds`, `_gt`, `_gtFine_labelIds`, `_gtFine_color`, and
`_color`.

UAVid links:

- official site: <https://uavid.nl/>
- Kaggle mirror used for download:
  <https://www.kaggle.com/datasets/dasmehdixtr/uavid-v1>

## Run The Pipeline

### 1. Run With UAVid Ground-Truth Labels

Use this mode when you want to test the downstream candidate extraction,
features, scoring, and visualization without relying on model predictions.

```bash
python src/main.py \
  --image_dir data/images \
  --label_dir data/labels \
  --output_dir outputs \
  --use_uavid_labels
```

In this mode, uncertainty from model probabilities is not available because
ground-truth labels are hard masks.

### 2. Run With The Default SegFormer Model

```bash
python src/main.py \
  --image_dir data/images \
  --output_dir outputs
```

### 3. Run With The Fine-Tuned UAVid Checkpoint

```bash
python src/main.py \
  --image_dir data/uavid_test_one_per_seq \
  --output_dir outputs_finetuned_uavid \
  --segformer_model models/segformer-uavid-continued-10/best \
  --inference_size 768
```

Current fine-tuned checkpoint:

```text
models/segformer-uavid-continued-10/best
```

Recent validation metrics:

```text
mean_iou = 0.5107
pixel_accuracy = 0.8297
road_iou = 0.6833
tree_iou = 0.7141
low_vegetation_iou = 0.6087
```

This checkpoint was fine-tuned for only 15 epochs, so results are preliminary.

### 4. Run A Small Debugging Test

This is useful for quickly checking that the pipeline runs end to end.

```bash
python src/main.py \
  --image_dir data/archive/uavid_train/seq1/Images \
  --output_dir outputs_uncertainty_seq1_20_continued10 \
  --segformer_model models/segformer-uavid-continued-10/best \
  --inference_size 768 \
  --max_images 20
```

Testing on train-derived images is normal for debugging the pipeline, especially
with a preliminary 15-epoch model. It checks whether masks, candidates, scores,
and visualizations are produced under familiar conditions. It should not be
used as generalization evidence. Use unseen one-per-sequence outputs for the
main result and failure analysis.

### Optional: Manual Road Polygon

If road segmentation is weak, you can pass a manual road polygon in image pixel
coordinates:

```bash
python src/main.py \
  --image_dir data/images \
  --output_dir outputs_manual_road \
  --road_polygon "100,400;500,350;900,700;50,700"
```

## Outputs

The pipeline writes:

```text
<output_dir>/csv/tree_features.csv
<output_dir>/visualizations/*_priority.jpg
<output_dir>/masks/*_canopy_*.png
```

The CSV contains one row per accepted canopy candidate. Important columns
include:

```text
image_path
tree_id
canopy_area_px
canopy_width_px
canopy_height_px
distance_to_road_px
road_buffer_overlap_ratio
canopy_irregularity
canopy_gap_ratio
canopy_asymmetry_score
base_inspection_score
uncertainty_score
final_priority_score
inspection_priority_level
```

Visualization labels use:

```text
T{id} B{base_score} U{uncertainty_score} P{final_score} {level}
```

## How The Pipeline Works

### Segmentation

`src/segformer_segmenter.py` loads a SegFormer model and predicts semantic
classes for each image. For UAVid, the important classes are road, tree, low
vegetation, building, static car, moving car, human, and clutter.

The code extracts:

- road context mask
- tree/canopy mask
- low-vegetation mask
- model confidence and entropy maps when model probabilities are available

### Candidate Splitting

`src/sam_splitter.py` uses SAM proposals to split large tree/canopy regions.
Each SAM proposal is intersected with the SegFormer tree mask, so SAM only helps
split pixels already predicted as tree/canopy.

If SAM cannot load, returns no masks, or no masks pass filtering, the code uses
connected components from `src/candidate_extraction.py`.

### Roadside Filtering

`src/road_region.py` and `src/structural_filter.py` keep candidates that are
large enough and relevant to the road. Candidates can pass if they are close to
the road mask or overlap the road buffer.

This step is important because the project is about roadside inspection
priority, not all trees in the image.

### Feature Extraction

`src/feature_extraction.py` computes image-space features for each accepted
candidate, including:

- area and bounding box size
- compactness and circularity
- gap ratio
- asymmetry
- edge roughness
- RGB green/brightness statistics
- distance to road
- road-buffer overlap
- mean tree probability and entropy-based uncertainty

These features are proxies for triage only. They are not physical measurements.

### Scoring

`src/risk_scoring.py` computes:

- `base_inspection_score`
- `uncertainty_score`
- `final_priority_score`
- `inspection_priority_level`

The base score gives the most importance to road proximity and road-buffer
overlap. Canopy size, irregularity, gap ratio, and asymmetry contribute less.
The uncertainty score can raise ambiguous candidates for review, but it is
capped so it cannot dominate the base road-context score.

Priority levels:

- Low: `final_priority_score < 0.33`
- Medium: `0.33 <= final_priority_score < 0.66`
- High: `final_priority_score >= 0.66`

## Fine-Tune SegFormer On UAVid

Example training command:

```bash
python -u src/train_segformer_uavid.py \
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

The current continued checkpoint used for experiments is:

```text
models/segformer-uavid-continued-10/best
```

## Current Result Folders

The important output folders in this workspace are:

```text
outputs_uncertainty_seq1_20_continued10
outputs_uncertainty_one_per_seq_continued10
```

`outputs_uncertainty_seq1_20_continued10` contains train-derived images and is
mainly for debugging. `outputs_uncertainty_one_per_seq_continued10` contains
unseen one-per-sequence test images and should be used for result discussion and
failure analysis.

## Known Failure Modes

Common issues:

- small or distant trees may be missed
- sparse crowns may be classified as low vegetation or background
- adjacent crowns can remain merged
- SAM may over-split or under-split canopy blobs
- wrong road masks can distort the priority score
- road-adjacent non-tree fragments can become false positives

Possible fixes:

- train SegFormer longer and with more small-tree examples
- use multi-scale inference
- tune candidate size thresholds
- improve SAM prompting/filtering
- add manual or calibrated road geometry
- use tree-instance labels if available
- add LiDAR point-cloud clustering for true 3D tree instances

## LiDAR Future Work

LiDAR would improve this task because it provides 3D information that RGB cannot
measure. It could help:

- split merged tree crowns into individual trees
- measure tree height and crown volume
- locate trunks
- compute true distance and clearance to the road
- validate uncertain RGB candidates
- detect structural change over repeated surveys

The point-cloud paper included in this repository is a useful reference for this
future direction:

- Liu et al., "Instance recognition of street trees from urban point clouds,"
  ISPRS JPRS 2023:
  <https://doi.org/10.1016/j.isprsjprs.2023.04.010>

## References

- SegFormer: <https://papers.neurips.cc/paper/2021/hash/64f1f27bf1b4ec22924fd0acb550c235-Abstract.html>
- Segment Anything: <https://arxiv.org/abs/2304.02643>
- UAVid: <https://uavid.nl/>
- UAVid Kaggle mirror: <https://www.kaggle.com/datasets/dasmehdixtr/uavid-v1>
