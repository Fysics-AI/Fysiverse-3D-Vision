"""Assemble object GLBs with explicit translation, rotation, and scaling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from fysiverse.manifest import load_manifest, relative_artifact, resolve_artifact, save_manifest


def pose_transform(pose: dict[str, Any]) -> np.ndarray:
    translation = np.asarray(pose.get("translation"), dtype=np.float64)
    rotation = np.asarray(pose.get("rotation"), dtype=np.float64)
    scaling = float(pose.get("scaling"))
    if translation.shape != (3,):
        raise ValueError(f"translation must have shape (3,), got {translation.shape}")
    if rotation.shape != (3, 3):
        raise ValueError(f"rotation must have shape (3, 3), got {rotation.shape}")
    if not np.all(np.isfinite(translation)) or not np.all(np.isfinite(rotation)) or not np.isfinite(scaling):
        raise ValueError("pose contains a non-finite value")
    if scaling <= 0:
        raise ValueError(f"scaling must be positive, got {scaling}")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation * scaling
    transform[:3, 3] = translation
    return transform


def load_scene(path: Path) -> trimesh.Scene:
    loaded = trimesh.load(str(path), force="scene", process=False)
    if isinstance(loaded, trimesh.Trimesh):
        loaded = trimesh.Scene(loaded)
    if not isinstance(loaded, trimesh.Scene) or not loaded.geometry:
        raise RuntimeError(f"asset is not a non-empty mesh scene: {path}")
    return loaded


def normalize_scene(scene: trimesh.Scene, margin: float) -> np.ndarray:
    if not 0 <= margin < 1:
        raise ValueError(f"normalization margin must be in [0, 1), got {margin}")
    bounds = scene.bounds
    if bounds is None:
        raise RuntimeError("cannot normalize an empty scene")
    center = (bounds[0] + bounds[1]) / 2.0
    max_extent = float(np.max(bounds[1] - bounds[0]))
    if not np.isfinite(max_extent) or max_extent <= 0:
        raise RuntimeError(f"invalid scene extent: {max_extent}")
    scale = 2.0 * (1.0 - margin) / max_extent
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] *= scale
    transform[:3, 3] = -center * scale
    scene.apply_transform(transform)
    return transform


def assemble(
    manifest_path: Path,
    poses: list[dict[str, Any]],
    *,
    normalize: bool = False,
    normalization_margin: float = 0.02,
) -> tuple[trimesh.Scene, np.ndarray]:
    manifest_path = manifest_path.resolve()
    manifest = load_manifest(manifest_path)
    pose_by_id = {str(item["id"]): item for item in poses}
    output = trimesh.Scene()
    for item in manifest["objects"]:
        object_id = str(item["id"])
        if object_id not in pose_by_id:
            raise ValueError(f"missing pose for object {object_id}")
        asset_path = resolve_artifact(manifest_path, item["asset"])
        if not asset_path.is_file():
            raise FileNotFoundError(f"object {object_id} asset not found: {asset_path}")
        transform = pose_transform(pose_by_id[object_id])
        source = load_scene(asset_path)
        for node_name in source.graph.nodes_geometry:
            node_transform, geometry_name = source.graph.get(frame_to=node_name)
            output.add_geometry(
                source.geometry[geometry_name].copy(),
                geom_name=f"object_{object_id}_{geometry_name}",
                node_name=f"object_{object_id}_{node_name}",
                transform=transform @ node_transform,
            )
    if not output.geometry:
        raise RuntimeError("no object geometry was assembled")
    scene_transform = normalize_scene(output, normalization_margin) if normalize else np.eye(4, dtype=np.float64)
    return output, scene_transform


def export_scene(
    manifest_path: Path,
    poses: list[dict[str, Any]],
    *,
    normalize: bool = False,
    normalization_margin: float = 0.02,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    manifest = load_manifest(manifest_path)
    scene, scene_transform = assemble(
        manifest_path,
        poses,
        normalize=normalize,
        normalization_margin=normalization_margin,
    )
    scene_path = resolve_artifact(manifest_path, manifest["outputs"]["scene"])
    poses_path = resolve_artifact(manifest_path, manifest["outputs"]["poses"])
    scene_path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(scene_path))

    payload = {
        "version": "1.0",
        "coordinate_system": manifest["coordinate_system"],
        "scene_transform": scene_transform.tolist(),
        "scene_transform_applied": bool(normalize),
        "objects": poses,
    }
    poses_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest["stages"]["layout"] = "complete"
    manifest["outputs"]["scene"] = relative_artifact(manifest_path, scene_path)
    manifest["outputs"]["poses"] = relative_artifact(manifest_path, poses_path)
    save_manifest(manifest_path, manifest)
    return payload
