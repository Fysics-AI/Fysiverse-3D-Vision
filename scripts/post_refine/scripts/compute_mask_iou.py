#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute per object and mean mask IoU from rendered metric masks.")
    parser.add_argument("--label-mask", required=True, type=Path)
    parser.add_argument("--before-mask-dir", required=True, type=Path)
    parser.add_argument("--after-mask-dir", required=True, type=Path)
    parser.add_argument("--before-penetration", required=True, type=Path)
    parser.add_argument("--after-penetration", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_label(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.int32)


def load_render_mask(path: Path) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("RGBA"))
    rgb = arr[..., :3]
    alpha = arr[..., 3]
    return (rgb.max(axis=2) > 127) & (alpha > 0)


def compute_split(label: np.ndarray, ids: list[int], mask_dir: Path) -> dict:
    per_object = []
    for obj_id in ids:
        gt = label == obj_id
        pred_path = mask_dir / f"mask_{obj_id:03d}.png"
        pred = load_render_mask(pred_path)
        if pred.shape != gt.shape:
            pred_img = Image.fromarray(pred.astype(np.uint8) * 255)
            pred = np.asarray(pred_img.resize((gt.shape[1], gt.shape[0]), Image.Resampling.NEAREST)) > 0
        intersection = int(np.logical_and(gt, pred).sum())
        union = int(np.logical_or(gt, pred).sum())
        iou = float(intersection / union) if union else 0.0
        per_object.append(
            {
                "id": int(obj_id),
                "iou": iou,
                "intersection": intersection,
                "union": union,
                "gt_pixels": int(gt.sum()),
                "pred_pixels": int(pred.sum()),
            }
        )
    return {
        "mIoU": float(np.mean([item["iou"] for item in per_object])) if per_object else 0.0,
        "per_object": per_object,
    }


def main() -> int:
    args = parse_args()
    label = load_label(args.label_mask)
    ids = [int(v) for v in sorted(np.unique(label)) if int(v) != 0]
    result = {
        "schema": "fysiverse_external_dataset_refine_metrics.v1",
        "label_mask": str(args.label_mask),
        "ids": ids,
        "before": compute_split(label, ids, args.before_mask_dir),
        "after": compute_split(label, ids, args.after_mask_dir),
    }
    result["before"]["penetration"] = json.loads(args.before_penetration.read_text(encoding="utf-8"))
    result["after"]["penetration"] = json.loads(args.after_penetration.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

