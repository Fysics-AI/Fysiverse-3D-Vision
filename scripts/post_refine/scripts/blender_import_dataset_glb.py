#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import bpy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", required=True, type=Path)
    parser.add_argument("--output-blend", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    return parser.parse_args(argv)


def clean_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for block in list(bpy.data.meshes):
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in list(bpy.data.materials):
        if block.users == 0:
            bpy.data.materials.remove(block)


def mesh_objects() -> list[bpy.types.Object]:
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def object_sort_key(obj: bpy.types.Object) -> tuple[int, str]:
    text = obj.name
    match = re.search(r"object[-_. ]*(\d+)", text, re.IGNORECASE)
    if match:
        return int(match.group(1)), text
    nums = re.findall(r"\d+", text)
    if nums:
        return int(nums[-1]), text
    return 10**9, text


def bounds(obj: bpy.types.Object) -> list[list[float]]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()
    try:
        pts = [eval_obj.matrix_world @ vertex.co for vertex in mesh.vertices]
    finally:
        eval_obj.to_mesh_clear()
    return [
        [float(min(p.x for p in pts)), float(min(p.y for p in pts)), float(min(p.z for p in pts))],
        [float(max(p.x for p in pts)), float(max(p.y for p in pts)), float(max(p.z for p in pts))],
    ]


def main() -> int:
    args = parse_args()
    clean_scene()
    glb_path = args.case_dir / "scene.glb"
    bpy.ops.import_scene.gltf(filepath=str(glb_path))
    bpy.context.view_layer.update()

    objects = sorted(mesh_objects(), key=object_sort_key)
    renamed = []
    for out_idx, obj in enumerate(objects, start=1):
        original_name = obj.name
        obj.name = f"mask_{out_idx:03d}_object"
        obj.data.name = f"mask_{out_idx:03d}_mesh"
        obj["mask_id"] = out_idx
        obj["source_object_index"] = out_idx - 1
        renamed.append(
            {
                "mask_id": out_idx,
                "source_object_index": out_idx - 1,
                "original_name": original_name,
                "name": obj.name,
                "bbox": bounds(obj),
            }
        )

    bpy.context.scene.frame_set(1)
    args.output_blend.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output_blend))
    report = {
        "schema": "fysiverse_external_dataset_import_blend.v1",
        "case_dir": str(args.case_dir),
        "scene_glb": str(glb_path),
        "output_blend": str(args.output_blend),
        "object_count": len(renamed),
        "objects": renamed,
        "note": "Blender glTF import converts the glTF Y-up scene to Blender Z-up. Objects are renamed to match label-mask ids.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
