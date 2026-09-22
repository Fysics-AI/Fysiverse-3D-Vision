#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import bpy


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import render_original_camera_animation as render_utils  # noqa: E402


MASK_OBJECT_RE = re.compile(r"^mask_(\d+)_object(?:[._].*)?$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render one RGB foreground and per-instance silhouettes.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--camera-optimization", required=True, type=Path)
    parser.add_argument("--rgb-output", required=True, type=Path)
    parser.add_argument("--mask-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--compute-backend", default="CUDA")
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    return parser.parse_args(argv)


def set_if_present(owner: Any, name: str, value: Any) -> None:
    if hasattr(owner, name):
        setattr(owner, name, value)


def instance_objects() -> dict[int, list[bpy.types.Object]]:
    result: dict[int, list[bpy.types.Object]] = {}
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        match = MASK_OBJECT_RE.match(obj.name)
        if match is None:
            obj.hide_render = True
            continue
        result.setdefault(int(match.group(1)), []).append(obj)
    if not result:
        raise RuntimeError("No mask_NNN_object meshes were found in the final scene")
    return result


def configure_rgb(scene: bpy.types.Scene, args: argparse.Namespace, width: int, height: int) -> None:
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    scene.cycles.samples = int(args.samples)
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 8
    scene.cycles.diffuse_bounces = 3
    scene.cycles.glossy_bounces = 3
    scene.render.resolution_x = int(width)
    scene.render.resolution_y = int(height)
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.compression = 15
    scene.render.film_transparent = True
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0


def configure_masks(scene: bpy.types.Scene) -> bpy.types.Material:
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    scene.cycles.samples = 1
    scene.cycles.use_denoising = False
    scene.cycles.max_bounces = 0
    scene.cycles.diffuse_bounces = 0
    scene.cycles.glossy_bounces = 0
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.compression = 15
    scene.render.film_transparent = True
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    material = bpy.data.materials.get("__metric_instance_mask")
    if material is None:
        material = bpy.data.materials.new("__metric_instance_mask")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    object_info = nodes.new(type="ShaderNodeObjectInfo")
    emission = nodes.new(type="ShaderNodeEmission")
    output = nodes.new(type="ShaderNodeOutputMaterial")
    emission.inputs["Strength"].default_value = 1.0
    links.new(object_info.outputs["Color"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def render_png(scene: bpy.types.Scene, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(path.resolve())
    bpy.ops.render.render(write_still=True)
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"Render was not created: {path}")
    return {"path": str(path.resolve()), "bytes": int(path.stat().st_size)}


def main() -> int:
    args = parse_args()
    bpy.ops.wm.open_mainfile(filepath=str(args.input.resolve()))
    scene = bpy.context.scene
    scene.frame_set(1)

    devices = render_utils.configure_gpu(args.compute_backend)
    render_utils.configure_world(0.05, True)
    camera = render_utils.add_optimized_input_camera(
        optimization_path=args.camera_optimization.resolve(),
        camera_name="OriginalInputCamera",
    )
    width, height = (int(value) for value in camera["image_size"])
    objects_by_id = instance_objects()

    bmin, bmax = render_utils.collect_bbox(1, 1)
    render_utils.add_camera_relative_lights(
        scene.camera,
        bmin,
        bmax,
        key_energy=220.0,
        fill_energy=25.0,
    )
    configure_rgb(scene, args, width, height)
    rgb = render_png(scene, args.rgb_output)

    all_instances = [obj for group in objects_by_id.values() for obj in group]
    mask_material = configure_masks(scene)
    for obj in all_instances:
        obj.hide_render = False
        obj.data.materials.clear()
        obj.data.materials.append(mask_material)
    masks: list[dict[str, Any]] = []
    args.mask_dir.mkdir(parents=True, exist_ok=True)
    for mask_id, selected in sorted(objects_by_id.items()):
        for obj in all_instances:
            obj.color = (1.0, 1.0, 1.0, 1.0) if obj in selected else (0.0, 0.0, 0.0, 1.0)
        output = args.mask_dir / f"mask_{mask_id:03d}.png"
        masks.append({"mask_id": mask_id, "objects": [obj.name for obj in selected], **render_png(scene, output)})

    payload = {
        "schema": "fysiverse_metric_renders.v1",
        "status": "ok",
        "input_blend": str(args.input.resolve()),
        "camera_optimization": str(args.camera_optimization.resolve()),
        "image_size": [width, height],
        "samples": int(args.samples),
        "compute_backend": args.compute_backend,
        "devices": devices,
        "rgb": rgb,
        "mask_encoding": "visible target instance is white; all other foreground instances are opaque black",
        "masks": masks,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(render_utils.as_json(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rgb": rgb["path"], "mask_count": len(masks), "image_size": [width, height]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
