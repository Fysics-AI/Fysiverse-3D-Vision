#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export visible Blend meshes as a single world-space triangle soup for nvdiffrast.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frame", type=int, default=1)
    parser.add_argument("--include-hidden-render", action="store_true")
    parser.add_argument("--include-helper-meshes", action="store_true")
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    return parser.parse_args(argv)


def should_export_object(obj: bpy.types.Object, args: argparse.Namespace) -> bool:
    if obj.type != "MESH":
        return False
    if obj.hide_render and not args.include_hidden_render:
        return False
    if not args.include_helper_meshes:
        if obj.name.startswith("__"):
            return False
        if obj.name in {"OriginalInputCamera"}:
            return False
    return True


def triangulate_polygon(indices: list[int]) -> list[tuple[int, int, int]]:
    if len(indices) < 3:
        return []
    if len(indices) == 3:
        return [(indices[0], indices[1], indices[2])]
    return [(indices[0], indices[i], indices[i + 1]) for i in range(1, len(indices) - 1)]


def main() -> int:
    args = parse_args()
    bpy.ops.wm.open_mainfile(filepath=str(args.input))
    scene = bpy.context.scene
    scene.frame_set(int(args.frame))
    depsgraph = bpy.context.evaluated_depsgraph_get()
    depsgraph.update()

    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    objects: list[dict[str, object]] = []

    for obj in scene.objects:
        if not should_export_object(obj, args):
            continue
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        if mesh is None:
            continue
        try:
            if not mesh.vertices or not mesh.polygons:
                continue
            start_v = len(vertices)
            start_f = len(faces)
            matrix = eval_obj.matrix_world.copy()
            bmin = Vector((float("inf"), float("inf"), float("inf")))
            bmax = Vector((float("-inf"), float("-inf"), float("-inf")))
            for vertex in mesh.vertices:
                point = matrix @ vertex.co
                vertices.append((float(point.x), float(point.y), float(point.z)))
                bmin.x = min(bmin.x, point.x)
                bmin.y = min(bmin.y, point.y)
                bmin.z = min(bmin.z, point.z)
                bmax.x = max(bmax.x, point.x)
                bmax.y = max(bmax.y, point.y)
                bmax.z = max(bmax.z, point.z)
            for poly in mesh.polygons:
                local_indices = [start_v + int(idx) for idx in poly.vertices]
                faces.extend(triangulate_polygon(local_indices))
            objects.append(
                {
                    "name": obj.name,
                    "vertices_start": start_v,
                    "vertices_count": len(vertices) - start_v,
                    "faces_start": start_f,
                    "faces_count": len(faces) - start_f,
                    "bbox_min": [float(bmin.x), float(bmin.y), float(bmin.z)],
                    "bbox_max": [float(bmax.x), float(bmax.y), float(bmax.z)],
                }
            )
        finally:
            eval_obj.to_mesh_clear()

    if not vertices or not faces:
        raise RuntimeError(f"No exportable mesh triangles found in {args.input}")

    verts_np = np.asarray(vertices, dtype=np.float32)
    faces_np = np.asarray(faces, dtype=np.int32)
    metadata = {
        "schema": "fysiverse_nvdiffrast_mesh.v1",
        "input": str(args.input),
        "frame": int(args.frame),
        "num_vertices": int(verts_np.shape[0]),
        "num_faces": int(faces_np.shape[0]),
        "objects": objects,
        "coordinate_system": "Blender world, Z-up",
        "notes": [
            "Vertices are already transformed to world space at the selected frame.",
            "Faces are triangle indices into the combined vertex array.",
            "Helper meshes whose names start with '__' are excluded by default.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, vertices=verts_np, faces=faces_np, metadata=json.dumps(metadata, ensure_ascii=False))
    print(f"Wrote nvdiffrast mesh: {args.output}")
    print(f"vertices={verts_np.shape[0]} faces={faces_np.shape[0]} objects={len(objects)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
