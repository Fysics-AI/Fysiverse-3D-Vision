#!/usr/bin/env python3
"""Prepare external 3D-FUTURE validation inputs for the public Layout worker."""

from __future__ import annotations

import argparse
import errno
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_SCENES = ("0000000", "0000008", "0000035")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def matrix_list(matrix: np.ndarray) -> list[list[float]]:
    return [[float(value) for value in row] for row in np.asarray(matrix).tolist()]


def intrinsics_from_fov_y(width: int, height: int, fov_y_degrees: float) -> np.ndarray:
    focal = 0.5 * float(height) / math.tan(math.radians(fov_y_degrees) * 0.5)
    return np.asarray(
        [[focal, 0.0, width * 0.5], [0.0, focal, height * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def derive_input_camera(scene_id: str, source_metadata: dict[str, Any]) -> dict[str, Any]:
    normalization = source_metadata.get("layout_blender_norm_meta")
    if not isinstance(normalization, dict):
        raise KeyError("source metadata has no layout_blender_norm_meta")
    layout = normalization.get("layout_normalization")
    if not isinstance(layout, dict):
        raise KeyError("source metadata has no layout_normalization")

    layout_offset = np.asarray(layout["offset_xyz"], dtype=np.float64).reshape(3)
    scene_post = np.asarray(normalization["scene_post_matrix"], dtype=np.float64).reshape(4, 4)
    layout_transform = np.eye(4, dtype=np.float64)
    layout_transform[:3, 3] = -layout_offset
    camera_to_world = scene_post @ layout_transform
    world_to_camera = np.linalg.inv(camera_to_world)

    scene = source_metadata.get("scene", {})
    image = scene.get("image", {}) if isinstance(scene, dict) else {}
    width = int(image.get("width", 1200))
    height = int(image.get("height", 1200))
    fovs = [
        float(item["fov"])
        for item in source_metadata.get("objects", [])
        if isinstance(item, dict) and isinstance(item.get("fov"), (int, float))
    ]
    if not fovs:
        raise ValueError("source metadata contains no object camera FOV")
    fov_y_degrees = math.degrees(float(np.median(fovs)))

    return {
        "schema_version": "fysicsmagic.scene_camera.v1",
        "scene_id": scene_id,
        "source": "dataset_metadata_derived_input_camera",
        "is_input_camera_ground_truth": True,
        "is_model_prediction": False,
        "camera": {
            "camera_to_world": matrix_list(camera_to_world),
            "world_to_camera": matrix_list(world_to_camera),
            "intrinsic": matrix_list(intrinsics_from_fov_y(width, height, fov_y_degrees)),
            "width": width,
            "height": height,
            "fov_y_degrees": fov_y_degrees,
            "near": None,
            "far": None,
        },
        "coordinate_system": {
            "final_scene_frame": "Stage 3 Y-up layout frame used by scene.glb",
            "scene_up_axis": "+Y",
            "camera_convention": "OpenGL/Blender: local +X right, +Y up, -Z forward",
            "matrix_semantics": "column vectors; p_world = camera_to_world @ p_camera",
            "matrix_storage": "row-major JSON arrays",
            "units": str(normalization.get("units", "meters")),
        },
        "derivation": {
            "layout_offset_xyz": [float(value) for value in layout_offset],
            "layout_transform": matrix_list(layout_transform),
            "scene_post_matrix": matrix_list(scene_post),
            "formula": "camera_to_world = scene_post_matrix @ layout_transform @ I",
            "note": (
                "3D-FUTURE object poses use the input-camera frame. The camera is derived "
                "from dataset preprocessing metadata; the Layout model does not predict it."
            ),
        },
    }


def link_or_copy(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        try:
            os.link(source, destination)
            return
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
            print(f"[prepare] cross-device hardlink; copying {source}")
    shutil.copy2(source, destination)


def prepare_scene(dataset_root: Path, output_root: Path, scene_id: str, asset_mode: str) -> Path:
    source_dir = dataset_root / scene_id
    stage_path = source_dir / f"{scene_id}.json"
    image_path = source_dir / "image.png"
    if not stage_path.is_file() or not image_path.is_file():
        raise FileNotFoundError(f"missing 3D-FUTURE validation input for {scene_id}: {source_dir}")
    stage = read_json(stage_path)
    source_metadata_path = Path(stage["source"]["scene_norm_json"])
    if not source_metadata_path.is_file():
        raise FileNotFoundError(f"source metadata not found: {source_metadata_path}")

    case_root = output_root / scene_id
    if case_root.exists():
        raise FileExistsError(f"output already exists: {case_root}")
    inputs = case_root / "inputs"
    masks_dir = inputs / "masks"
    assets_dir = inputs / "object_assets"
    masks_dir.mkdir(parents=True)
    assets_dir.mkdir(parents=True)
    shutil.copy2(image_path, inputs / "image.png")
    shutil.copy2(stage_path, inputs / "source_scene_metadata.json")

    manifest_objects = []
    for expected_id, item in enumerate(stage.get("objects", [])):
        object_id = int(item["edit_index"])
        if object_id != expected_id:
            raise ValueError(
                f"{scene_id} object indices must be contiguous: expected {expected_id}, got {object_id}"
            )
        mask_source = source_dir / f"{object_id}.png"
        asset_source = Path(item["mesh_retrieval"]["trellis_model_dir"]) / "model.glb"
        if not mask_source.is_file() or not asset_source.is_file():
            raise FileNotFoundError(f"missing mask or object asset for {scene_id} object {object_id}")
        shutil.copy2(mask_source, masks_dir / f"{object_id}.png")
        link_or_copy(asset_source, assets_dir / str(object_id) / "model.glb", asset_mode)
        manifest_objects.append(
            {
                "id": str(object_id),
                "category": str(item.get("category") or "object"),
                "mask": f"inputs/masks/{object_id}.png",
                "asset": f"inputs/object_assets/{object_id}/model.glb",
            }
        )

    camera = derive_input_camera(scene_id, read_json(source_metadata_path))
    write_json(inputs / "camera.json", camera)
    manifest = {
        "version": "1.0",
        "scene_id": scene_id,
        "image": "inputs/image.png",
        "mask_backend": "provided",
        "coordinate_system": {
            "up_axis": "y",
            "handedness": "right",
            "units": "normalized_scene_units",
        },
        "objects": manifest_objects,
        "stages": {
            "mask": "reused",
            "prepare": "reused",
            "flux": "reused",
            "trellis2": "reused",
            "layout": "pending",
            "post_refine": "pending",
        },
        "outputs": {
            "poses": "reports/poses.json",
            "scene": "scenes/scene.glb",
            "scene_refined": "scenes/scene_refined.glb",
            "post_refine_summary": "post_refine/post_refine_summary.json",
        },
    }
    write_json(case_root / "manifest.json", manifest)
    return case_root / "manifest.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scene", action="append", dest="scenes")
    parser.add_argument("--asset-mode", choices=("hardlink", "copy"), default="hardlink")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    scenes = tuple(args.scenes or DEFAULT_SCENES)
    args.output_root.mkdir(parents=True, exist_ok=True)
    for scene_id in scenes:
        manifest = prepare_scene(
            args.dataset_root.resolve(), args.output_root.resolve(), scene_id, args.asset_mode
        )
        print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
