#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


Y_UP_TO_Z_UP = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mask-dir-name", default="")
    return parser.parse_args()


def object_indices(case_dir: Path, mask_dir_name: str) -> tuple[Path, list[int]]:
    if mask_dir_name:
        mask_dir = case_dir / mask_dir_name
    elif (case_dir / "masks").is_dir():
        mask_dir = case_dir / "masks"
    else:
        mask_dir = case_dir / "generated_masks"
    indices = sorted(int(path.stem) for path in mask_dir.glob("*.png") if path.stem.isdigit())
    if not indices:
        raise RuntimeError(f"No object masks found in {mask_dir}")
    return mask_dir, indices


def mask_foreground(path: Path) -> np.ndarray:
    image = Image.open(path)
    arr = np.asarray(image)
    if arr.ndim == 2:
        return arr > 0
    if arr.shape[-1] == 4:
        alpha = arr[..., 3]
        rgb = arr[..., :3]
        return (alpha > 0) & (rgb.max(axis=-1) > 0)
    return arr[..., :3].max(axis=-1) > 0


def write_label_mask(mask_dir: Path, indices: list[int], output_path: Path) -> tuple[int, int]:
    first = mask_foreground(mask_dir / f"{indices[0]}.png")
    label = np.zeros(first.shape, dtype=np.uint8)
    for idx in indices:
        fg = mask_foreground(mask_dir / f"{idx}.png")
        if fg.shape != label.shape:
            fg = np.asarray(Image.fromarray(fg.astype(np.uint8) * 255).resize((label.shape[1], label.shape[0]), Image.Resampling.NEAREST)) > 0
        label[fg] = int(idx) + 1
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(label).save(output_path)
    height, width = label.shape
    return width, height


def load_stage_objects(case_dir: Path) -> dict[int, dict]:
    for name in ("stage3_predictions.json", "stage3_inputs.json"):
        path = case_dir / name
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        return {int(obj.get("edit_index", obj.get("object_index", idx))): obj for idx, obj in enumerate(data.get("objects") or [])}
    return {}


def write_manifest(case_dir: Path, indices: list[int], output_path: Path) -> None:
    stage_objects = load_stage_objects(case_dir)
    objects = []
    for idx in indices:
        stage_obj = stage_objects.get(idx, {})
        objects.append(
            {
                "mask_id": int(idx) + 1,
                "mask_name": f"mask_{int(idx) + 1:03d}",
                "final_3d_object_name": f"mask_{int(idx) + 1:03d}_object",
                "semantic_label": stage_obj.get("category") or f"object_{idx:03d}",
                "bbox_2d_xywh": stage_obj.get("bbox"),
                "source_object_index": int(idx),
            }
        )
    payload = {
        "schema": "fysiverse_external_dataset_alignment_manifest.v1",
        "status": "ok",
        "case_dir": str(case_dir),
        "objects": objects,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_camera_optimization(case_dir: Path, label_size: tuple[int, int], output_path: Path) -> None:
    camera_payload = json.loads((case_dir / "camera.json").read_text(encoding="utf-8"))
    camera = camera_payload["camera"]
    c2w_yup = np.asarray(camera["camera_to_world"], dtype=np.float64).reshape(4, 4)
    c2w_zup = Y_UP_TO_Z_UP @ c2w_yup
    intrinsic = np.asarray(camera["intrinsic"], dtype=np.float64).reshape(3, 3)
    cam_width = float(camera.get("width") or label_size[0])
    cam_height = float(camera.get("height") or label_size[1])
    intr_norm = [
        [float(intrinsic[0, 0] / cam_width), 0.0, float(intrinsic[0, 2] / cam_width)],
        [0.0, float(intrinsic[1, 1] / cam_height), float(intrinsic[1, 2] / cam_height)],
        [0.0, 0.0, 1.0],
    ]
    payload = {
        "schema": "fysiverse_external_dataset_fixed_camera.v1",
        "status": "ok",
        "source_camera_json": str(case_dir / "camera.json"),
        "source_camera_coordinate_system": camera_payload.get("coordinate_system"),
        "source_camera_kind": camera_payload.get("source"),
        "note": "Camera converted from source Y-up scene frame to Blender/SAPIEN Z-up frame. Used as a fixed camera for object RTS refinement.",
        "optimized_camera_to_world": c2w_zup.tolist(),
        "camera_to_world": c2w_zup.tolist(),
        "intrinsics_normalized": intr_norm,
        "source_image_size": [int(label_size[0]), int(label_size[1])],
        "image_size": [int(label_size[0]), int(label_size[1])],
        "metrics": {},
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir, indices = object_indices(args.case_dir, args.mask_dir_name)
    label_size = write_label_mask(mask_dir, indices, args.output_dir / "label_mask.png")
    write_manifest(args.case_dir, indices, args.output_dir / "alignment_manifest.json")
    write_camera_optimization(args.case_dir, label_size, args.output_dir / "fixed_camera_optimization.json")
    summary = {
        "case_dir": str(args.case_dir),
        "mask_dir": str(mask_dir),
        "object_indices": indices,
        "label_mask": str(args.output_dir / "label_mask.png"),
        "label_size": list(label_size),
        "alignment_manifest": str(args.output_dir / "alignment_manifest.json"),
        "fixed_camera_optimization": str(args.output_dir / "fixed_camera_optimization.json"),
    }
    (args.output_dir / "prepare_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
