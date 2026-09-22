#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import bmesh
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree


MASK_OBJECT_RE = re.compile(r"^mask_(\d+)_object(?:[._].*)?$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute PAT3D-style inter-object penetration ratio.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-edge-ratio", type=float, default=0.05)
    parser.add_argument("--min-faces", type=int, default=32)
    parser.add_argument("--max-faces", type=int, default=20000)
    parser.add_argument("--max-remeshed-faces", type=int, default=100000)
    parser.add_argument("--long-edge-factor", type=float, default=1.25)
    parser.add_argument("--subdivision-passes", type=int, default=4)
    parser.add_argument("--crossing-epsilon-ratio", type=float, default=1e-6)
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    return parser.parse_args(argv)


def metric_objects() -> list[bpy.types.Object]:
    objects = [
        obj
        for obj in bpy.context.scene.objects
        if obj.type == "MESH" and MASK_OBJECT_RE.match(obj.name) is not None and not obj.hide_render
    ]
    if not objects:
        raise RuntimeError("No visible mask_NNN_object meshes found")
    return sorted(objects, key=lambda obj: obj.name)


def scene_bounds(objects: list[bpy.types.Object]) -> tuple[np.ndarray, np.ndarray]:
    points = [np.asarray(obj.matrix_world @ Vector(corner), dtype=np.float64) for obj in objects for corner in obj.bound_box]
    stacked = np.stack(points)
    return stacked.min(axis=0), stacked.max(axis=0)


def evaluated_world_mesh(obj: bpy.types.Object) -> bpy.types.Mesh:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = bpy.data.meshes.new_from_object(evaluated, depsgraph=depsgraph)
    mesh.transform(evaluated.matrix_world)
    mesh.update()
    return mesh


def decimate_mesh(mesh: bpy.types.Mesh, target_faces: int, name: str) -> tuple[bpy.types.Mesh, int]:
    mesh.calc_loop_triangles()
    original_faces = len(mesh.loop_triangles)
    if original_faces <= target_faces:
        return mesh, original_faces

    temp = bpy.data.objects.new(f"__metric_decimate_{name}", mesh)
    bpy.context.collection.objects.link(temp)
    modifier = temp.modifiers.new(name="metric_target_edge_decimate", type="DECIMATE")
    modifier.decimate_type = "COLLAPSE"
    modifier.ratio = max(1e-6, min(1.0, float(target_faces) / float(original_faces)))
    modifier.use_collapse_triangulate = True
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = temp.evaluated_get(depsgraph)
    reduced = bpy.data.meshes.new_from_object(evaluated, depsgraph=depsgraph)
    reduced.update()
    bpy.data.objects.remove(temp, do_unlink=True)
    bpy.data.meshes.remove(mesh)
    return reduced, original_faces


def enforce_maximum_edge(
    mesh: bpy.types.Mesh,
    maximum_edge: float,
    passes: int,
    maximum_faces: int,
) -> tuple[bpy.types.Mesh, int, bool]:
    bm = bmesh.new()
    bm.from_mesh(mesh)
    used_passes = 0
    capped = False
    for _ in range(max(0, int(passes))):
        bm.edges.ensure_lookup_table()
        long_edges = [edge for edge in bm.edges if edge.calc_length() > maximum_edge]
        if not long_edges:
            break
        if len(bm.faces) * 4 > int(maximum_faces):
            capped = True
            break
        bmesh.ops.subdivide_edges(bm, edges=long_edges, cuts=1, use_grid_fill=True)
        used_passes += 1
    bmesh.ops.triangulate(bm, faces=list(bm.faces))
    refined = bpy.data.meshes.new(mesh.name + "_target_edge")
    bm.to_mesh(refined)
    refined.update()
    bm.free()
    bpy.data.meshes.remove(mesh)
    return refined, used_passes, capped


def mesh_arrays(mesh: bpy.types.Mesh) -> tuple[np.ndarray, np.ndarray]:
    mesh.calc_loop_triangles()
    vertices = np.empty((len(mesh.vertices), 3), dtype=np.float64)
    for index, vertex in enumerate(mesh.vertices):
        vertices[index] = vertex.co
    faces = np.empty((len(mesh.loop_triangles), 3), dtype=np.int64)
    for index, triangle in enumerate(mesh.loop_triangles):
        faces[index] = triangle.vertices
    return vertices, faces


def unique_edge_lengths(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=0)
    edges.sort(axis=1)
    edges = np.unique(edges, axis=0)
    return np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)


def triangle_straddles_plane(triangle: np.ndarray, plane_triangle: np.ndarray, epsilon: float) -> bool:
    normal = np.cross(plane_triangle[1] - plane_triangle[0], plane_triangle[2] - plane_triangle[0])
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-15:
        return False
    distances = (triangle - plane_triangle[0]) @ (normal / norm)
    return bool(float(distances.min()) < -epsilon and float(distances.max()) > epsilon)


def strict_crossing(a: np.ndarray, b: np.ndarray, epsilon: float) -> bool:
    return triangle_straddles_plane(a, b, epsilon) and triangle_straddles_plane(b, a, epsilon)


def remesh_object(obj: bpy.types.Object, target_edge: float, args: argparse.Namespace) -> dict[str, Any]:
    mesh = evaluated_world_mesh(obj)
    mesh.calc_loop_triangles()
    original_faces = len(mesh.loop_triangles)
    surface_area = float(sum(polygon.area for polygon in mesh.polygons))
    ideal_triangle_area = math.sqrt(3.0) * target_edge * target_edge / 4.0
    target_faces = int(round(surface_area / max(ideal_triangle_area, 1e-15)))
    target_faces = min(original_faces, max(int(args.min_faces), min(int(args.max_faces), target_faces)))
    mesh, original_faces = decimate_mesh(mesh, target_faces, obj.name)
    mesh, subdivision_passes, subdivision_capped = enforce_maximum_edge(
        mesh,
        target_edge * float(args.long_edge_factor),
        int(args.subdivision_passes),
        int(args.max_remeshed_faces),
    )
    vertices, faces = mesh_arrays(mesh)
    edge_lengths = unique_edge_lengths(vertices, faces)
    tree = BVHTree.FromPolygons(
        [Vector(vertex) for vertex in vertices],
        [tuple(int(value) for value in face) for face in faces],
        all_triangles=True,
        epsilon=0.0,
    )
    bpy.data.meshes.remove(mesh)
    return {
        "name": obj.name,
        "mask_id": int(MASK_OBJECT_RE.match(obj.name).group(1)),
        "vertices": vertices,
        "faces": faces,
        "tree": tree,
        "edge_lengths": edge_lengths,
        "summary": {
            "name": obj.name,
            "mask_id": int(MASK_OBJECT_RE.match(obj.name).group(1)),
            "surface_area": surface_area,
            "original_face_count": int(original_faces),
            "target_face_count": int(target_faces),
            "remeshed_vertex_count": int(len(vertices)),
            "remeshed_face_count": int(len(faces)),
            "mean_edge_length": float(np.mean(edge_lengths)),
            "max_edge_length": float(np.max(edge_lengths)),
            "subdivision_passes": int(subdivision_passes),
            "subdivision_capped": bool(subdivision_capped),
            "bbox_min": vertices.min(axis=0).tolist(),
            "bbox_max": vertices.max(axis=0).tolist(),
        },
    }


def main() -> int:
    args = parse_args()
    if args.target_edge_ratio <= 0.0:
        raise ValueError("--target-edge-ratio must be positive")
    bpy.ops.wm.open_mainfile(filepath=str(args.input.resolve()))
    bpy.context.scene.frame_set(1)
    objects = metric_objects()
    bounds_min, bounds_max = scene_bounds(objects)
    diagonal = float(np.linalg.norm(bounds_max - bounds_min))
    if diagonal <= 0.0:
        raise RuntimeError("Scene diagonal is zero")
    target_edge = float(args.target_edge_ratio) * diagonal
    crossing_epsilon = float(args.crossing_epsilon_ratio) * diagonal

    remeshed = [remesh_object(obj, target_edge, args) for obj in objects]
    all_edge_lengths = np.concatenate([entry["edge_lengths"] for entry in remeshed])
    mean_edge = float(np.mean(all_edge_lengths))

    pair_results: list[dict[str, Any]] = []
    penetrating_pairs = 0
    raw_overlaps = 0
    for left_index, left in enumerate(remeshed):
        for right in remeshed[left_index + 1 :]:
            if left["mask_id"] == right["mask_id"]:
                continue
            overlaps = left["tree"].overlap(right["tree"])
            strict_count = 0
            for left_face_index, right_face_index in overlaps:
                left_triangle = left["vertices"][left["faces"][left_face_index]]
                right_triangle = right["vertices"][right["faces"][right_face_index]]
                if strict_crossing(left_triangle, right_triangle, crossing_epsilon):
                    strict_count += 1
            raw_count = len(overlaps)
            raw_overlaps += raw_count
            penetrating_pairs += strict_count
            pair_results.append(
                {
                    "left": left["name"],
                    "right": right["name"],
                    "raw_bvh_overlaps": int(raw_count),
                    "strict_penetrating_triangle_pairs": int(strict_count),
                }
            )

    ratio = float(penetrating_pairs * mean_edge / diagonal)
    payload = {
        "schema": "fysiverse_penetration_ratio.v1",
        "status": "ok",
        "input_blend": str(args.input.resolve()),
        "scene_bounds_min": bounds_min.tolist(),
        "scene_bounds_max": bounds_max.tolist(),
        "scene_diagonal": diagonal,
        "target_edge_ratio": float(args.target_edge_ratio),
        "target_edge_length": target_edge,
        "mean_remeshed_edge_length": mean_edge,
        "strict_crossing_epsilon": crossing_epsilon,
        "self_intersections_included": False,
        "raw_inter_object_bvh_overlaps": int(raw_overlaps),
        "inter_object_penetrating_triangle_pairs": int(penetrating_pairs),
        "r_pen": ratio,
        "objects": [entry["summary"] for entry in remeshed],
        "pairs": pair_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"r_pen": ratio, "penetrating_pairs": penetrating_pairs, "mean_edge": mean_edge}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
