"""Fine-tune SegFormer on UAVid semantic segmentation labels."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader
from tqdm import tqdm

from uavid_dataset import UAVidSegmentationDataset, collate_segmentation_batch
from uavid_labels import UAVID_CLASS_NAMES, UAVID_ID_TO_LABEL, UAVID_LABEL_TO_ID


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune SegFormer on UAVid.")
    parser.add_argument("--train_dir", default="data/archive/uavid_train", help="UAVid train split root.")
    parser.add_argument("--val_dir", default="data/archive/uavid_val", help="UAVid validation split root.")
    parser.add_argument("--output_dir", default="models/segformer-uavid", help="Directory for fine-tuned checkpoint.")
    parser.add_argument(
        "--model_name",
        default="nvidia/segformer-b0-finetuned-ade-512-512",
        help="Base SegFormer checkpoint. B0 is recommended for this 8GB VRAM GPU.",
    )
    parser.add_argument("--image_size", type=int, default=512, help="Square resize used for training.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=6e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--eval_every_steps", type=int, default=200)
    parser.add_argument("--save_every_epochs", type=int, default=1)
    parser.add_argument("--label_tolerance", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no_amp", action="store_true", help="Disable CUDA mixed precision.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = UAVidSegmentationDataset(
        Path(args.train_dir),
        image_size=args.image_size,
        color_tolerance=args.label_tolerance,
        max_samples=args.max_train_samples,
        augment=True,
    )
    val_dataset = UAVidSegmentationDataset(
        Path(args.val_dir),
        image_size=args.image_size,
        color_tolerance=args.label_tolerance,
        max_samples=args.max_val_samples,
        augment=False,
    )
    if len(train_dataset) == 0:
        raise RuntimeError(f"No training image/label pairs found in {args.train_dir}")
    if len(val_dataset) == 0:
        raise RuntimeError(f"No validation image/label pairs found in {args.val_dir}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_segmentation_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_segmentation_batch,
    )

    model = build_model(args.model_name)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and not args.no_amp))

    print(f"device={device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"gpu={props.name} vram_gb={props.total_memory / 1024**3:.2f}")
    print(f"train_samples={len(train_dataset)} val_samples={len(val_dataset)}")
    print(f"model={args.model_name} image_size={args.image_size} batch_size={args.batch_size} accum={args.gradient_accumulation_steps}")

    global_step = 0
    best_miou = -1.0
    history: list[dict[str, float | int]] = []
    start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", dynamic_ncols=True)
        for step, batch in enumerate(progress, start=1):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and not args.no_amp)):
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss / max(args.gradient_accumulation_steps, 1)

            scaler.scale(loss).backward()
            epoch_loss += float(loss.detach().cpu()) * max(args.gradient_accumulation_steps, 1)

            if step % max(args.gradient_accumulation_steps, 1) == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                progress.set_postfix(loss=f"{epoch_loss / step:.4f}")
                if args.eval_every_steps > 0 and global_step % args.eval_every_steps == 0:
                    metrics = evaluate(model, val_loader, device)
                    print(format_metrics(global_step, epoch, epoch_loss / step, metrics))
                    if metrics["mean_iou"] > best_miou:
                        best_miou = metrics["mean_iou"]
                        save_checkpoint(model, output_dir / "best", args, metrics)

        metrics = evaluate(model, val_loader, device)
        avg_loss = epoch_loss / max(len(train_loader), 1)
        history.append({"epoch": epoch, "train_loss": avg_loss, **metrics})
        print(format_metrics(global_step, epoch, avg_loss, metrics))

        if metrics["mean_iou"] > best_miou:
            best_miou = metrics["mean_iou"]
            save_checkpoint(model, output_dir / "best", args, metrics)
        if args.save_every_epochs > 0 and epoch % args.save_every_epochs == 0:
            save_checkpoint(model, output_dir / f"epoch-{epoch:03d}", args, metrics)

    final_metrics = evaluate(model, val_loader, device)
    save_checkpoint(model, output_dir / "final", args, final_metrics)
    with (output_dir / "training_history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    print(f"done elapsed_min={(time.perf_counter() - start) / 60.0:.2f} best_miou={best_miou:.4f}")
    print(f"final_checkpoint={output_dir / 'final'}")


def build_model(model_name: str):
    from transformers import SegformerForSemanticSegmentation

    return SegformerForSemanticSegmentation.from_pretrained(
        model_name,
        num_labels=len(UAVID_CLASS_NAMES),
        id2label={int(k): v for k, v in UAVID_ID_TO_LABEL.items()},
        label2id={k: int(v) for k, v in UAVID_LABEL_TO_ID.items()},
        ignore_mismatched_sizes=True,
    )


@torch.inference_mode()
def evaluate(model, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    intersections = np.zeros(len(UAVID_CLASS_NAMES), dtype=np.float64)
    unions = np.zeros(len(UAVID_CLASS_NAMES), dtype=np.float64)
    correct = 0.0
    total = 0.0
    losses: list[float] = []

    for batch in loader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        outputs = model(pixel_values=pixel_values, labels=labels)
        losses.append(float(outputs.loss.detach().cpu()))
        logits = functional.interpolate(
            outputs.logits,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        predictions = logits.argmax(dim=1)
        valid = labels != 255
        correct += float(((predictions == labels) & valid).sum().detach().cpu())
        total += float(valid.sum().detach().cpu())

        pred_np = predictions.detach().cpu().numpy()
        label_np = labels.detach().cpu().numpy()
        valid_np = label_np != 255
        for class_id in range(len(UAVID_CLASS_NAMES)):
            pred_c = (pred_np == class_id) & valid_np
            label_c = (label_np == class_id) & valid_np
            intersections[class_id] += np.count_nonzero(pred_c & label_c)
            unions[class_id] += np.count_nonzero(pred_c | label_c)

    ious = np.divide(intersections, unions, out=np.full_like(intersections, np.nan), where=unions > 0)
    return {
        "val_loss": float(np.mean(losses)) if losses else 0.0,
        "pixel_accuracy": float(correct / max(total, 1.0)),
        "mean_iou": float(np.nanmean(ious)) if np.any(~np.isnan(ious)) else 0.0,
        "road_iou": class_iou(ious, "road"),
        "tree_iou": class_iou(ious, "tree"),
        "low_vegetation_iou": class_iou(ious, "low vegetation"),
    }


def class_iou(ious: np.ndarray, class_name: str) -> float:
    index = UAVID_LABEL_TO_ID[class_name]
    value = ious[index]
    return 0.0 if np.isnan(value) else float(value)


def save_checkpoint(model, output_dir: Path, args: argparse.Namespace, metrics: dict[str, float]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    save_image_processor(output_dir)
    metadata = {
        "base_model": args.model_name,
        "image_size": args.image_size,
        "classes": list(UAVID_CLASS_NAMES),
        "metrics": metrics,
    }
    with (output_dir / "uavid_training_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def save_image_processor(output_dir: Path) -> None:
    from transformers import SegformerImageProcessor

    processor = SegformerImageProcessor(
        do_resize=False,
        do_rescale=False,
        do_normalize=False,
    )
    processor.save_pretrained(output_dir)


def format_metrics(global_step: int, epoch: int, loss: float, metrics: dict[str, float]) -> str:
    return (
        f"step={global_step} epoch={epoch} train_loss={loss:.4f} "
        f"val_loss={metrics['val_loss']:.4f} acc={metrics['pixel_accuracy']:.4f} "
        f"miou={metrics['mean_iou']:.4f} road_iou={metrics['road_iou']:.4f} "
        f"tree_iou={metrics['tree_iou']:.4f} lowveg_iou={metrics['low_vegetation_iou']:.4f}"
    )


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
