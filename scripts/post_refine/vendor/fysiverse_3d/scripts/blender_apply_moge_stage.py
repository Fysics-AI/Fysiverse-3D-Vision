#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import bmesh
import bpy
from mathutils import Matrix, Vector


SKIP_PREFIXES = (
    "__gravity_ground__",
    "__rigidbody_preview_ground__",
    "__sapien_ground__",
)


def log_stage(message: str) -> None:
    print(f"[moge-stage {time.strftime('%H:%M:%S')}] {message}", flush=True)


def elapsed_sec(start: float) -> float:
    return float(time.monotonic() - start)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply a SAM3D MoGe postprocess stage directly in Blender while preserving materials/textures.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--pack-textures", action="store_true")
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    return parser.parse_args(argv)


def numpy_to_json(value: Any) -> Any:
    if isinstance(value, Vector):
        return [float(v) for v in value]
    if isinstance(value, Matrix):
        return [[float(v) for v in row] for row in value]
    if isinstance(value, dict):
        return {str(k): numpy_to_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [numpy_to_json(v) for v in value]
    return value


def matrix_from_list(values: list[list[float]]) -> Matrix:
    return Matrix([[float(v) for v in row] for row in values])


def vector_from_list(values: list[float]) -> Vector:
    vec = Vector((float(values[0]), float(values[1]), float(values[2])))
    if vec.length < 1e-10:
        raise ValueError("expected non-zero vector")
    vec.normalize()
    return vec


def image_is_packed(image: bpy.types.Image) -> bool:
    if getattr(image, "packed_file", None) is not None:
        return True
    packed_files = getattr(image, "packed_files", None)
    return bool(packed_files)


def pack_external_images() -> list[str]:
    packed: list[str] = []
    for image in bpy.data.images:
        if image.source != "FILE" or not image.filepath or image_is_packed(image):
            continue
        try:
            image.pack()
            packed.append(image.name)
        except Exception as exc:
            print(f"Warning: failed to pack image texture {image.name}: {exc}")
    try:
        bpy.ops.file.pack_all()
    except Exception as exc:
        print(f"Warning: bpy.ops.file.pack_all failed: {exc}")
    return packed


def should_skip(obj: bpy.types.Object) -> bool:
    if obj.type != "MESH":
        return True
    if any(obj.name.startswith(prefix) for prefix in SKIP_PREFIXES):
        return True
    return False


def mesh_objects() -> list[bpy.types.Object]:
    return [obj for obj in bpy.context.scene.objects if not should_skip(obj)]


def world_bounds(obj: bpy.types.Object) -> list[Vector]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()
    try:
        if mesh is not None and mesh.vertices:
            matrix_world = eval_obj.matrix_world
            first = matrix_world @ mesh.vertices[0].co
            min_x = max_x = float(first.x)
            min_y = max_y = float(first.y)
            min_z = max_z = float(first.z)
            for vertex in mesh.vertices[1:]:
                world = matrix_world @ vertex.co
                min_x = min(min_x, float(world.x))
                min_y = min(min_y, float(world.y))
                min_z = min(min_z, float(world.z))
                max_x = max(max_x, float(world.x))
                max_y = max(max_y, float(world.y))
                max_z = max(max_z, float(world.z))
            return [Vector((min_x, min_y, min_z)), Vector((max_x, max_y, max_z))]
    finally:
        if mesh is not None:
            eval_obj.to_mesh_clear()

    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    return [
        Vector((min(v.x for v in corners), min(v.y for v in corners), min(v.z for v in corners))),
        Vector((max(v.x for v in corners), max(v.y for v in corners), max(v.z for v in corners))),
    ]


def bounds_to_json(bounds: list[Vector]) -> list[list[float]]:
    return [[float(v.x), float(v.y), float(v.z)] for v in bounds]


def bounds_center(bounds: list[Vector]) -> Vector:
    return (bounds[0] + bounds[1]) * 0.5


def bounds_extent(bounds: list[Vector]) -> Vector:
    return bounds[1] - bounds[0]


def apply_world_translation(obj: bpy.types.Object, delta: Vector) -> None:
    obj.matrix_world = Matrix.Translation(delta) @ obj.matrix_world


def set_origin_to_world_point_preserve_geometry(obj: bpy.types.Object, world_point: Vector) -> Vector:
    old_world = obj.matrix_world.copy()
    local_point = old_world.inverted() @ world_point
    new_world = old_world.copy()
    new_world.translation = world_point
    vertex_transform = new_world.inverted() @ old_world
    for vertex in obj.data.vertices:
        vertex.co = vertex_transform @ vertex.co
    obj.data.update()
    obj.matrix_world = new_world
    return local_point


def recenter_origins_to_bbox_centers(objects: list[bpy.types.Object]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    bpy.context.view_layer.update()
    for idx, obj in enumerate(objects):
        before = world_bounds(obj)
        center = bounds_center(before)
        old_origin = obj.matrix_world.translation.copy()
        old_local_center = set_origin_to_world_point_preserve_geometry(obj, center)
        bpy.context.view_layer.update()
        after = world_bounds(obj)
        reports.append(
            {
                "index": idx,
                "name": obj.name,
                "old_origin_world": old_origin,
                "new_origin_world": obj.matrix_world.translation.copy(),
                "bbox_center_world": center,
                "old_local_bbox_center": old_local_center,
                "bbox_before": bounds_to_json(before),
                "bbox_after": bounds_to_json(after),
            }
        )
    return reports


def apply_global_matrix(objects: list[bpy.types.Object], matrix: Matrix) -> None:
    for obj in objects:
        obj.matrix_world = matrix @ obj.matrix_world
    bpy.context.view_layer.update()


def snap_to_ground(objects: list[bpy.types.Object], ground_y: float, up: Vector) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for idx, obj in enumerate(objects):
        before = world_bounds(obj)
        min_ground = min(float(corner.dot(up)) for corner in before)
        delta = up * (float(ground_y) - min_ground)
        apply_world_translation(obj, delta)
        bpy.context.view_layer.update()
        after = world_bounds(obj)
        reports.append(
            {
                "index": idx,
                "name": obj.name,
                "up_vector": up,
                "min_ground_before": min_ground,
                "translation": delta,
                "translation_along_up": float(delta.dot(up)),
                "min_ground_after": min(float(corner.dot(up)) for corner in after),
                "bbox_before": bounds_to_json(before),
                "bbox_after": bounds_to_json(after),
            }
        )
    return reports


def up_axis_index(up: Vector) -> int:
    values = [abs(float(up.x)), abs(float(up.y)), abs(float(up.z))]
    return int(max(range(3), key=lambda idx: values[idx]))


def bounds_min_axis(bounds: list[Vector], axis: int) -> float:
    return vector_axis_value(bounds[0], axis)


def bounds_max_axis(bounds: list[Vector], axis: int) -> float:
    return vector_axis_value(bounds[1], axis)


def bounds_center_axis(bounds: list[Vector], axis: int) -> float:
    return 0.5 * (bounds_min_axis(bounds, axis) + bounds_max_axis(bounds, axis))


def bounds_area_on_axes(bounds: list[Vector], axes: list[int]) -> float:
    width = max(bounds_max_axis(bounds, axes[0]) - bounds_min_axis(bounds, axes[0]), 0.0)
    depth = max(bounds_max_axis(bounds, axes[1]) - bounds_min_axis(bounds, axes[1]), 0.0)
    return max(width * depth, 1e-12)


def overlap_area_on_axes(bounds_i: list[Vector], bounds_j: list[Vector], axes: list[int]) -> float:
    overlap_a = overlap_1d(
        bounds_min_axis(bounds_i, axes[0]),
        bounds_max_axis(bounds_i, axes[0]),
        bounds_min_axis(bounds_j, axes[0]),
        bounds_max_axis(bounds_j, axes[0]),
    )
    overlap_b = overlap_1d(
        bounds_min_axis(bounds_i, axes[1]),
        bounds_max_axis(bounds_i, axes[1]),
        bounds_min_axis(bounds_j, axes[1]),
        bounds_max_axis(bounds_j, axes[1]),
    )
    return max(float(overlap_a * overlap_b), 0.0)


def scene_height(entries: list[dict[str, Any]], up_axis: int) -> float:
    if not entries:
        return 1.0
    bottom = min(bounds_min_axis(entry["bounds"], up_axis) for entry in entries)
    top = max(bounds_max_axis(entry["bounds"], up_axis) for entry in entries)
    return max(top - bottom, 1e-8)


def resolve_support_limit(value: float, ratio: float, height: float) -> float:
    if float(value) >= 0.0:
        return float(value)
    return max(float(ratio) * float(height), 1e-8)


def mask_id_from_object_name(name: str) -> int | None:
    match = re.search(r"mask_(\d+)", str(name))
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def first_present(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def would_create_cycle(parents: dict[int, int], child: int, parent: int) -> bool:
    seen = {child}
    current = parent
    while current in parents:
        if current in seen:
            return True
        seen.add(current)
        current = parents[current]
    return current == child


def scene_graph_supports(
    entries: list[dict[str, Any]],
    scene_graph: dict[str, Any] | None,
    *,
    up: Vector,
    eps: float,
) -> dict[str, Any] | None:
    if not isinstance(scene_graph, dict) or scene_graph.get("status") != "ok":
        return None
    raw_relations = list(scene_graph.get("support_relations") or [])
    up_axis = up_axis_index(up)
    axes = horizontal_axis_indices(up)
    id_to_indices: dict[int, list[int]] = {}
    for idx, entry in enumerate(entries):
        mask_id = mask_id_from_object_name(entry["name"])
        if mask_id is not None:
            id_to_indices.setdefault(mask_id, []).append(idx)

    candidates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for rel in raw_relations:
        upper_id = safe_int(first_present(rel, ("upper_mask_id", "child_mask_id", "subject_mask_id")))
        lower_id = safe_int(first_present(rel, ("lower_mask_id", "parent_mask_id", "object_mask_id")))
        if upper_id is None or lower_id is None or upper_id == lower_id:
            rejected.append({"relation": rel, "reason": "invalid_mask_ids"})
            continue
        upper_indices = id_to_indices.get(upper_id) or []
        lower_indices = id_to_indices.get(lower_id) or []
        if not upper_indices or not lower_indices:
            rejected.append({"relation": rel, "reason": "mask_id_not_found_in_blend_objects"})
            continue
        upper_idx = upper_indices[0]
        lower_idx = lower_indices[0]
        upper = entries[upper_idx]
        lower = entries[lower_idx]
        upper_bounds = upper["bounds"]
        lower_bounds = lower["bounds"]
        upper_area = bounds_area_on_axes(upper_bounds, axes)
        lower_area = bounds_area_on_axes(lower_bounds, axes)
        overlap_area = overlap_area_on_axes(upper_bounds, lower_bounds, axes)
        lower_top = bounds_max_axis(lower_bounds, up_axis)
        upper_bottom = bounds_min_axis(upper_bounds, up_axis)
        confidence = safe_float(rel.get("confidence"), 0.0)
        candidates.append(
            {
                "upper_index": upper_idx,
                "upper_name": upper["name"],
                "upper_mask_id": upper_id,
                "lower_index": lower_idx,
                "lower_name": lower["name"],
                "lower_mask_id": lower_id,
                "overlap_area": overlap_area,
                "overlap_ratio_to_upper": overlap_area / max(upper_area, 1e-12),
                "upper_area_xy": upper_area,
                "lower_area_xy": lower_area,
                "bottom_to_lower_top_gap": upper_bottom - lower_top,
                "confidence": confidence,
                "reason": rel.get("reason") or rel.get("evidence"),
                "source": "scene_graph",
                "score_tuple": [-confidence, abs(upper_bottom - lower_top)],
            }
        )

    parents: dict[int, int] = {}
    selected: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: (float(item["score_tuple"][0]), float(item["score_tuple"][1]))):
        child = int(candidate["upper_index"])
        parent = int(candidate["lower_index"])
        if child in parents:
            rejected.append({"relation": candidate, "reason": "child_already_has_support_parent"})
            continue
        if would_create_cycle(parents, child, parent):
            rejected.append({"relation": candidate, "reason": "support_cycle_rejected"})
            continue
        parents[child] = parent
        selected.append(candidate)

    return {
        "source": "scene_graph",
        "up_axis": up_axis,
        "horizontal_axes": axes,
        "parents": parents,
        "selected": selected,
        "candidates": candidates,
        "rejected": rejected,
        "scene_graph_status": scene_graph.get("status"),
        "scene_graph_scope": scene_graph.get("scope"),
        "scene_graph_support_count": len(raw_relations),
    }


def infer_supports(
    entries: list[dict[str, Any]],
    *,
    up: Vector,
    xy_overlap_ratio: float,
    max_gap: float,
    max_penetration: float,
    min_lower_area_ratio: float,
    eps: float,
) -> dict[str, Any]:
    up_axis = up_axis_index(up)
    axes = horizontal_axis_indices(up)
    candidates_by_child: dict[int, list[dict[str, Any]]] = {}
    all_candidates: list[dict[str, Any]] = []
    for upper_idx, upper in enumerate(entries):
        upper_bounds = upper["bounds"]
        upper_area = bounds_area_on_axes(upper_bounds, axes)
        upper_bottom = bounds_min_axis(upper_bounds, up_axis)
        upper_center = bounds_center_axis(upper_bounds, up_axis)
        for lower_idx, lower in enumerate(entries):
            if upper_idx == lower_idx:
                continue
            lower_bounds = lower["bounds"]
            lower_center = bounds_center_axis(lower_bounds, up_axis)
            if upper_center <= lower_center + eps:
                continue
            lower_area = bounds_area_on_axes(lower_bounds, axes)
            if lower_area + eps < upper_area * float(min_lower_area_ratio):
                continue
            overlap_area = overlap_area_on_axes(upper_bounds, lower_bounds, axes)
            overlap_ratio = overlap_area / upper_area
            if overlap_ratio + eps < float(xy_overlap_ratio):
                continue
            lower_top = bounds_max_axis(lower_bounds, up_axis)
            bottom_to_top_gap = upper_bottom - lower_top
            if bottom_to_top_gap < -float(max_penetration) - eps:
                continue
            if bottom_to_top_gap > float(max_gap) + eps:
                continue
            candidate = {
                "upper_index": upper_idx,
                "upper_name": upper["name"],
                "lower_index": lower_idx,
                "lower_name": lower["name"],
                "overlap_area": overlap_area,
                "overlap_ratio_to_upper": overlap_ratio,
                "upper_area_xy": upper_area,
                "lower_area_xy": lower_area,
                "bottom_to_lower_top_gap": bottom_to_top_gap,
                "score_tuple": [abs(bottom_to_top_gap), -overlap_ratio],
            }
            candidates_by_child.setdefault(upper_idx, []).append(candidate)
            all_candidates.append(candidate)

    parents: dict[int, int] = {}
    selected: list[dict[str, Any]] = []
    for child_idx, candidates in candidates_by_child.items():
        best = sorted(candidates, key=lambda item: (float(item["score_tuple"][0]), float(item["score_tuple"][1])))[0]
        parents[child_idx] = int(best["lower_index"])
        selected.append(best)
    return {
        "up_axis": up_axis,
        "horizontal_axes": axes,
        "parents": parents,
        "selected": selected,
        "candidates": all_candidates,
    }


def support_subtree(children: dict[int, list[int]], root: int) -> list[int]:
    out: list[int] = []
    stack = [root]
    seen: set[int] = set()
    while stack:
        idx = stack.pop()
        if idx in seen:
            continue
        seen.add(idx)
        out.append(idx)
        stack.extend(children.get(idx, []))
    return out


def translated_entry(entry: dict[str, Any], delta: Vector) -> dict[str, Any]:
    out = dict(entry)
    out["translation"] = (entry.get("translation") or Vector((0.0, 0.0, 0.0))) + delta
    out["bounds"] = shift_bounds(entry["bounds"], delta)
    return out


def support_bbox_vertical_delta(
    entries: list[dict[str, Any]],
    *,
    child: int,
    parent: int,
    up_axis: int,
    support_gap: float,
) -> dict[str, Any]:
    parent_top = bounds_max_axis(entries[parent]["bounds"], up_axis)
    child_bottom = bounds_min_axis(entries[child]["bounds"], up_axis)
    current_gap = child_bottom - parent_top
    delta_scalar = float(support_gap) - current_gap
    return {
        "method": "bbox",
        "parent_top_before": parent_top,
        "child_bottom_before": child_bottom,
        "current_gap_before": current_gap,
        "required_gap": float(support_gap),
        "translation_along_up": max(delta_scalar, 0.0),
        "needs_translation": delta_scalar > 0.0,
    }


def support_convex_vertical_delta(
    entries: list[dict[str, Any]],
    *,
    child: int,
    parent: int,
    up: Vector,
    up_axis: int,
    support_gap: float,
    eps: float,
    collision_eps: float,
    scene_height_value: float,
) -> dict[str, Any]:
    bbox_info = support_bbox_vertical_delta(entries, child=child, parent=parent, up_axis=up_axis, support_gap=support_gap)
    support_start = time.monotonic()
    log_stage(f"support sat start parent={entries[parent]['name']!r} child={entries[child]['name']!r}")
    current_sat = convex_sat_collision_report(entries[parent], entries[child], eps=collision_eps)
    current_elapsed = elapsed_sec(support_start)
    log_stage(
        "support sat current "
        f"parent={entries[parent]['name']!r} child={entries[child]['name']!r} "
        f"colliding={bool(current_sat.get('colliding'))} reason={current_sat.get('reason')} elapsed={current_elapsed:.2f}s"
    )
    if not bool(current_sat.get("colliding")):
        bbox_info.update(
            {
                "method": "convex_hull_sat",
                "convex_current_collision": current_sat,
                "convex_current_elapsed_sec": current_elapsed,
                "convex_total_elapsed_sec": elapsed_sec(support_start),
                "translation_along_up": 0.0,
                "needs_translation": False,
                "reason": "convex_support_pair_not_colliding",
            }
        )
        return bbox_info

    up_dir = Vector(up)
    if up_dir.length < 1e-10:
        up_dir = Vector((0.0, 0.0, 1.0))
    else:
        up_dir.normalize()

    high = max(
        float(bbox_info["translation_along_up"]),
        float(support_gap),
        float(scene_height_value) * 0.005,
        float(eps) * 10.0,
        1e-5,
    )
    high_sat = current_sat
    max_high = max(float(scene_height_value) * 8.0, high * 16.0, 1.0)
    high_iterations = 0
    high_start = time.monotonic()
    for _ in range(64):
        high_iterations += 1
        moved_child = translated_entry(entries[child], up_dir * high)
        high_sat = convex_sat_collision_report(entries[parent], moved_child, eps=collision_eps)
        if not bool(high_sat.get("colliding")):
            break
        high *= 2.0
        if high > max_high:
            break
    else:
        high_sat = {"colliding": True, "reason": "convex_vertical_search_iteration_limit"}
    high_elapsed = elapsed_sec(high_start)
    log_stage(
        "support sat high-search "
        f"parent={entries[parent]['name']!r} child={entries[child]['name']!r} "
        f"iters={high_iterations} high={high:.6g} colliding={bool(high_sat.get('colliding'))} elapsed={high_elapsed:.2f}s"
    )

    if bool(high_sat.get("colliding")):
        fallback = float(bbox_info["translation_along_up"])
        bbox_info.update(
            {
                "method": "convex_hull_sat",
                "convex_current_collision": current_sat,
                "convex_high_collision": high_sat,
                "convex_current_elapsed_sec": current_elapsed,
                "convex_high_search_elapsed_sec": high_elapsed,
                "convex_total_elapsed_sec": elapsed_sec(support_start),
                "translation_along_up": fallback,
                "needs_translation": fallback > eps,
                "reason": "convex_vertical_search_failed_fallback_bbox",
            }
        )
        return bbox_info

    low = 0.0
    binary_start = time.monotonic()
    binary_iterations = 0
    for _ in range(36):
        binary_iterations += 1
        mid = 0.5 * (low + high)
        moved_child = translated_entry(entries[child], up_dir * mid)
        sat = convex_sat_collision_report(entries[parent], moved_child, eps=collision_eps)
        if bool(sat.get("colliding")):
            low = mid
        else:
            high = mid
    delta_scalar = high + float(support_gap)
    binary_elapsed = elapsed_sec(binary_start)
    log_stage(
        "support sat binary-search "
        f"parent={entries[parent]['name']!r} child={entries[child]['name']!r} "
        f"iters={binary_iterations} min_non_colliding={high:.6g} elapsed={binary_elapsed:.2f}s "
        f"total={elapsed_sec(support_start):.2f}s"
    )
    return {
        **bbox_info,
        "method": "convex_hull_sat",
        "convex_current_collision": current_sat,
        "convex_resolved_collision": high_sat,
        "convex_min_non_colliding_translation": high,
        "convex_current_elapsed_sec": current_elapsed,
        "convex_high_search_elapsed_sec": high_elapsed,
        "convex_binary_search_elapsed_sec": binary_elapsed,
        "convex_total_elapsed_sec": elapsed_sec(support_start),
        "translation_along_up": delta_scalar,
        "needs_translation": delta_scalar > eps,
        "reason": "convex_support_pair_vertical_separation",
    }


def resolve_support_info(
    entries: list[dict[str, Any]],
    *,
    up: Vector,
    scene_graph: dict[str, Any] | None,
    support_adjust: bool,
    support_require_scene_graph: bool,
    support_xy_overlap_ratio: float,
    support_max_gap: float,
    support_max_gap_ratio: float,
    support_max_penetration: float,
    support_max_penetration_ratio: float,
    support_min_lower_area_ratio: float,
    eps: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    up_axis = up_axis_index(up)
    height = scene_height(entries, up_axis)
    max_gap = resolve_support_limit(support_max_gap, support_max_gap_ratio, height)
    max_penetration = resolve_support_limit(support_max_penetration, support_max_penetration_ratio, height)
    if not support_adjust:
        support_info = {
            "source": "disabled",
            "parents": {},
            "selected": [],
            "candidates": [],
            "up_axis": up_axis,
            "horizontal_axes": horizontal_axis_indices(up),
        }
    else:
        graph_support_info = scene_graph_supports(entries, scene_graph, up=up, eps=eps)
        if graph_support_info is not None:
            support_info = graph_support_info
        elif support_require_scene_graph:
            support_info = {
                "source": "missing_scene_graph",
                "parents": {},
                "selected": [],
                "candidates": [],
                "rejected": [],
                "up_axis": up_axis,
                "horizontal_axes": horizontal_axis_indices(up),
            }
        else:
            support_info = infer_supports(
                entries,
                up=up,
                xy_overlap_ratio=support_xy_overlap_ratio,
                max_gap=max_gap,
                max_penetration=max_penetration,
                min_lower_area_ratio=support_min_lower_area_ratio,
                eps=eps,
            )
            support_info["source"] = "bbox_geometry"
    limits = {
        "up_axis": up_axis,
        "scene_height_along_up": height,
        "support_max_gap": max_gap,
        "support_max_penetration": max_penetration,
    }
    return support_info, limits


def separate_support_relations_vertically(
    objects: list[bpy.types.Object],
    *,
    up: Vector,
    scene_graph: dict[str, Any] | None,
    support_adjust: bool,
    support_require_scene_graph: bool,
    support_gap: float,
    support_xy_overlap_ratio: float,
    support_max_gap: float,
    support_max_gap_ratio: float,
    support_max_penetration: float,
    support_max_penetration_ratio: float,
    support_min_lower_area_ratio: float,
    eps: float,
    support_collision_method: str,
    hull_max_vertices: int,
    collision_eps: float,
    convex_decomposition_method: str,
    coacd_config: dict[str, Any],
    collision_cache: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    stage_start = time.monotonic()
    entries: list[dict[str, Any]] = [
        {
            "index": idx,
            "name": obj.name,
            "object": obj,
            "bounds": world_bounds(obj),
            "translation": Vector((0.0, 0.0, 0.0)),
            "collision": {},
        }
        for idx, obj in enumerate(objects)
    ]
    if not entries:
        return {"mode": "support_vertical_min_gap", "objects": [], "support_relations": [], "support_actions": []}

    support_info, limits = resolve_support_info(
        entries,
        up=up,
        scene_graph=scene_graph,
        support_adjust=support_adjust,
        support_require_scene_graph=support_require_scene_graph,
        support_xy_overlap_ratio=support_xy_overlap_ratio,
        support_max_gap=support_max_gap,
        support_max_gap_ratio=support_max_gap_ratio,
        support_max_penetration=support_max_penetration,
        support_max_penetration_ratio=support_max_penetration_ratio,
        support_min_lower_area_ratio=support_min_lower_area_ratio,
        eps=eps,
    )
    up_axis = int(limits["up_axis"])
    height = float(limits["scene_height_along_up"])
    parents: dict[int, int] = {int(k): int(v) for k, v in (support_info.get("parents") or {}).items()}
    children: dict[int, list[int]] = {}
    for child, parent in parents.items():
        if child == parent:
            continue
        children.setdefault(parent, []).append(child)

    support_actions: list[dict[str, Any]] = []
    skipped_actions: list[dict[str, Any]] = []
    stage_collision_cache: dict[str, dict[str, Any]] = collision_cache if collision_cache is not None else {}
    log_stage(
        "support stage start "
        f"objects={len(entries)} selected_relations={len(support_info.get('selected') or [])} "
        f"parents={len(parents)} method={support_collision_method} decomposition={convex_decomposition_method}"
    )

    def ensure_collision(entry: dict[str, Any]) -> None:
        ensure_entry_collision(
            entry,
            max_vertices=int(hull_max_vertices),
            decomposition_method=convex_decomposition_method,
            coacd_config=coacd_config,
            cache=stage_collision_cache,
        )

    prebuild_report: dict[str, Any] = {}
    if support_collision_method == "convex_hull_sat":
        required_indices = {idx for pair in parents.items() for idx in (int(pair[0]), int(pair[1]))}
        prebuild_report = prebuild_entry_collisions(
            entries,
            required_indices,
            max_vertices=int(hull_max_vertices),
            decomposition_method=convex_decomposition_method,
            coacd_config=coacd_config,
            cache=stage_collision_cache,
            phase="support",
        )

    ordered_children = sorted(parents, key=lambda idx: bounds_center_axis(entries[idx]["bounds"], up_axis))
    for child in ordered_children:
        parent = parents[child]
        if parent == child or parent < 0 or parent >= len(entries):
            continue
        if support_collision_method == "convex_hull_sat":
            ensure_collision(entries[parent])
            ensure_collision(entries[child])
            separation_info = support_convex_vertical_delta(
                entries,
                child=child,
                parent=parent,
                up=up,
                up_axis=up_axis,
                support_gap=support_gap,
                eps=eps,
                collision_eps=collision_eps,
                scene_height_value=height,
            )
        else:
            separation_info = support_bbox_vertical_delta(entries, child=child, parent=parent, up_axis=up_axis, support_gap=support_gap)
        delta_scalar = float(separation_info.get("translation_along_up") or 0.0)
        if not bool(separation_info.get("needs_translation")) or delta_scalar <= eps:
            skipped_actions.append(
                {
                    "child_index": child,
                    "child_name": entries[child]["name"],
                    "parent_index": parent,
                    "parent_name": entries[parent]["name"],
                    "reason": separation_info.get("reason") or "already_separated",
                    "separation": separation_info,
                }
            )
            continue
        subtree = support_subtree(children, child)
        delta = up * delta_scalar
        for idx in subtree:
            entries[idx]["bounds"] = shift_bounds(entries[idx]["bounds"], delta)
            entries[idx]["translation"] += delta
        support_actions.append(
            {
                "child_index": child,
                "child_name": entries[child]["name"],
                "parent_index": parent,
                "parent_name": entries[parent]["name"],
                "subtree_indices": subtree,
                "subtree_names": [entries[idx]["name"] for idx in subtree],
                "parent_top_before": separation_info.get("parent_top_before"),
                "child_bottom_before": separation_info.get("child_bottom_before"),
                "current_gap_before": separation_info.get("current_gap_before"),
                "required_gap": separation_info.get("required_gap"),
                "translation": delta,
                "translation_along_up": delta_scalar,
                "separation": separation_info,
            }
        )

    log_stage(
        "support stage apply "
        f"actions={len(support_actions)} skipped={len(skipped_actions)} elapsed={elapsed_sec(stage_start):.2f}s"
    )
    for entry in entries:
        if entry["translation"].length > 1e-12:
            apply_world_translation(entry["object"], entry["translation"])
    bpy.context.view_layer.update()
    objects_report = [
        {
            "index": entry["index"],
            "name": entry["name"],
            "total_translation": entry["translation"],
            "bbox_after": bounds_to_json(world_bounds(entry["object"])),
        }
        for entry in entries
    ]
    return {
        "mode": "support_vertical_min_gap",
        "coordinate_system": "Blender/SAPIEN Z-up when target_up is [0, 0, 1].",
        "up_vector": up,
        "up_axis": up_axis,
        "support_adjust": bool(support_adjust),
        "support_gap": float(support_gap),
        "support_collision_method": str(support_collision_method),
        "convex_hull_max_vertices": int(hull_max_vertices),
        "convex_decomposition_method": str(convex_decomposition_method),
        "coacd_config": dict(coacd_config),
        "collision_prebuild": prebuild_report,
        "convex_collision_eps": float(collision_eps),
        "elapsed_sec": elapsed_sec(stage_start),
        "support_relations": support_info.get("selected") or [],
        "support_candidates": support_info.get("candidates") or [],
        "support_rejected": support_info.get("rejected") or [],
        "support_relation_source": support_info.get("source"),
        "scene_graph_scope": support_info.get("scene_graph_scope"),
        "scene_graph_support_count": support_info.get("scene_graph_support_count"),
        "support_actions": support_actions,
        "support_skipped_actions": skipped_actions,
        "objects": objects_report,
        **limits,
    }


def global_ground_and_support_adjust(
    objects: list[bpy.types.Object],
    *,
    ground_y: float,
    up: Vector,
    scene_graph: dict[str, Any] | None,
    support_adjust: bool,
    support_require_scene_graph: bool,
    support_gap: float,
    support_xy_overlap_ratio: float,
    support_max_gap: float,
    support_max_gap_ratio: float,
    support_max_penetration: float,
    support_max_penetration_ratio: float,
    support_min_lower_area_ratio: float,
    eps: float,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = [
        {"index": idx, "name": obj.name, "object": obj, "bounds": world_bounds(obj), "translation": Vector((0.0, 0.0, 0.0))}
        for idx, obj in enumerate(objects)
    ]
    if not entries:
        return {"mode": "global", "objects": [], "support_relations": [], "support_actions": []}

    support_info, limits = resolve_support_info(
        entries,
        up=up,
        scene_graph=scene_graph,
        support_adjust=support_adjust,
        support_require_scene_graph=support_require_scene_graph,
        support_xy_overlap_ratio=support_xy_overlap_ratio,
        support_max_gap=support_max_gap,
        support_max_gap_ratio=support_max_gap_ratio,
        support_max_penetration=support_max_penetration,
        support_max_penetration_ratio=support_max_penetration_ratio,
        support_min_lower_area_ratio=support_min_lower_area_ratio,
        eps=eps,
    )
    up_axis = int(limits["up_axis"])
    height = float(limits["scene_height_along_up"])
    max_gap = float(limits["support_max_gap"])
    max_penetration = float(limits["support_max_penetration"])
    parents: dict[int, int] = {int(k): int(v) for k, v in (support_info.get("parents") or {}).items()}
    root_indices = [idx for idx in range(len(entries)) if idx not in parents]
    if not root_indices:
        root_indices = list(range(len(entries)))
    root_min = min(bounds_min_axis(entries[idx]["bounds"], up_axis) for idx in root_indices)
    global_delta_scalar = float(ground_y) - float(root_min)
    global_delta = up * global_delta_scalar
    for entry in entries:
        entry["bounds"] = shift_bounds(entry["bounds"], global_delta)
        entry["translation"] += global_delta

    children: dict[int, list[int]] = {}
    for child, parent in parents.items():
        if child == parent:
            continue
        children.setdefault(parent, []).append(child)

    support_actions: list[dict[str, Any]] = []
    if support_adjust and parents:
        ordered_children = sorted(parents, key=lambda idx: bounds_center_axis(entries[idx]["bounds"], up_axis))
        for child in ordered_children:
            parent = parents[child]
            if parent == child or parent < 0 or parent >= len(entries):
                continue
            parent_top = bounds_max_axis(entries[parent]["bounds"], up_axis)
            child_bottom = bounds_min_axis(entries[child]["bounds"], up_axis)
            desired_bottom = parent_top + float(support_gap)
            delta_scalar = desired_bottom - child_bottom
            if abs(delta_scalar) <= eps:
                continue
            subtree = support_subtree(children, child)
            delta = up * delta_scalar
            for idx in subtree:
                entries[idx]["bounds"] = shift_bounds(entries[idx]["bounds"], delta)
                entries[idx]["translation"] += delta
            support_actions.append(
                {
                    "child_index": child,
                    "child_name": entries[child]["name"],
                    "parent_index": parent,
                    "parent_name": entries[parent]["name"],
                    "subtree_indices": subtree,
                    "subtree_names": [entries[idx]["name"] for idx in subtree],
                    "parent_top_before": parent_top,
                    "child_bottom_before": child_bottom,
                    "desired_child_bottom": desired_bottom,
                    "translation": delta,
                    "translation_along_up": delta_scalar,
                    "support_gap": float(support_gap),
                }
            )

    for entry in entries:
        if entry["translation"].length > 1e-12:
            apply_world_translation(entry["object"], entry["translation"])
    bpy.context.view_layer.update()
    objects_report = [
        {
            "index": entry["index"],
            "name": entry["name"],
            "total_translation": entry["translation"],
            "bbox_after": bounds_to_json(world_bounds(entry["object"])),
        }
        for entry in entries
    ]
    return {
        "mode": "global_support_aware" if support_adjust else "global",
        "coordinate_system": "Blender/SAPIEN Z-up when target_up is [0, 0, 1].",
        "up_vector": up,
        "up_axis": up_axis,
        "z_up_target": abs(float(up.z)) >= 0.99,
        "ground_y": float(ground_y),
        "scene_height_along_up": height,
        "root_indices": root_indices,
        "root_names": [entries[idx]["name"] for idx in root_indices],
        "root_min_before_global_translation": root_min,
        "global_translation": global_delta,
        "global_translation_along_up": global_delta_scalar,
        "support_adjust": bool(support_adjust),
        "support_gap": float(support_gap),
        "support_xy_overlap_ratio": float(support_xy_overlap_ratio),
        "support_max_gap": float(max_gap),
        "support_max_penetration": float(max_penetration),
        "support_min_lower_area_ratio": float(support_min_lower_area_ratio),
        "support_relations": support_info.get("selected") or [],
        "support_candidates": support_info.get("candidates") or [],
        "support_rejected": support_info.get("rejected") or [],
        "support_relation_source": support_info.get("source"),
        "scene_graph_scope": support_info.get("scene_graph_scope"),
        "scene_graph_support_count": support_info.get("scene_graph_support_count"),
        "support_actions": support_actions,
        "objects": objects_report,
    }


def overlap_1d(a_min: float, a_max: float, b_min: float, b_max: float) -> float:
    return max(0.0, min(a_max, b_max) - max(a_min, b_min))


def bbox_volume(bounds: list[Vector]) -> float:
    extent = bounds_extent(bounds)
    return max(float(extent.x), 0.0) * max(float(extent.y), 0.0) * max(float(extent.z), 0.0)


def bbox_overlap_report(bounds_i: list[Vector], bounds_j: list[Vector]) -> dict[str, Any]:
    overlaps = [
        overlap_1d(float(bounds_i[0].x), float(bounds_i[1].x), float(bounds_j[0].x), float(bounds_j[1].x)),
        overlap_1d(float(bounds_i[0].y), float(bounds_i[1].y), float(bounds_j[0].y), float(bounds_j[1].y)),
        overlap_1d(float(bounds_i[0].z), float(bounds_i[1].z), float(bounds_j[0].z), float(bounds_j[1].z)),
    ]
    volume = float(overlaps[0] * overlaps[1] * overlaps[2])
    min_volume = max(min(bbox_volume(bounds_i), bbox_volume(bounds_j)), 1e-12)
    return {
        "overlap_xyz": overlaps,
        "overlap_volume": volume,
        "overlap_volume_ratio_to_smaller": volume / min_volume,
        "bounds_i": bounds_to_json(bounds_i),
        "bounds_j": bounds_to_json(bounds_j),
    }


def horizontal_axis_indices(up: Vector) -> list[int]:
    values = [abs(float(up.x)), abs(float(up.y)), abs(float(up.z))]
    vertical = int(max(range(3), key=lambda idx: values[idx]))
    return [idx for idx in range(3) if idx != vertical]


def vector_axis_value(vec: Vector, axis: int) -> float:
    return float((vec.x, vec.y, vec.z)[axis])


def set_vector_axis(vec: Vector, axis: int, value: float) -> Vector:
    out = Vector(vec)
    if axis == 0:
        out.x = value
    elif axis == 1:
        out.y = value
    else:
        out.z = value
    return out


def shift_bounds(bounds: list[Vector], delta: Vector) -> list[Vector]:
    return [bounds[0] + delta, bounds[1] + delta]


def pair_reports(
    entries: list[dict[str, Any]],
    *,
    eps: float,
    min_overlap_volume_ratio: float,
    skip_pairs: set[tuple[int, int]] | None = None,
) -> list[dict[str, Any]]:
    skip_pairs = skip_pairs or set()
    reports: list[dict[str, Any]] = []
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            if (i, j) in skip_pairs or (j, i) in skip_pairs:
                continue
            info = bbox_overlap_report(entries[i]["bounds"], entries[j]["bounds"])
            overlaps = info["overlap_xyz"]
            if any(float(v) <= eps for v in overlaps):
                continue
            if float(info["overlap_volume_ratio_to_smaller"]) < float(min_overlap_volume_ratio):
                continue
            item = dict(info)
            item["indices"] = [i, j]
            item["pair"] = [entries[i]["name"], entries[j]["name"]]
            reports.append(item)
    return reports


def connected_components_from_pairs(reports: list[dict[str, Any]], count: int) -> list[list[int]]:
    parent = list(range(count))

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for report in reports:
        i, j = [int(v) for v in report["indices"]]
        union(i, j)
    groups: dict[int, list[int]] = {}
    for idx in range(count):
        groups.setdefault(find(idx), []).append(idx)
    return [members for members in groups.values() if len(members) > 1]


def pack_indices_along_primary_axis(
    entries: list[dict[str, Any]],
    indices: list[int],
    *,
    margin: float,
    reason: str,
    iteration: int,
    up: Vector,
) -> list[dict[str, Any]]:
    axes = horizontal_axis_indices(up)
    primary_axis = axes[0]
    secondary_axis = axes[1]
    ordered = sorted(
        set(int(i) for i in indices),
        key=lambda idx: (
            vector_axis_value(bounds_center(entries[idx]["bounds"]), primary_axis),
            vector_axis_value(bounds_center(entries[idx]["bounds"]), secondary_axis),
            entries[idx]["name"],
        ),
    )
    if len(ordered) < 2:
        return []
    mins = {idx: vector_axis_value(entries[idx]["bounds"][0], primary_axis) for idx in ordered}
    maxs = {idx: vector_axis_value(entries[idx]["bounds"][1], primary_axis) for idx in ordered}
    centers = {idx: 0.5 * (mins[idx] + maxs[idx]) for idx in ordered}
    widths = {idx: max(maxs[idx] - mins[idx], 1e-8) for idx in ordered}
    group_center = 0.5 * (min(mins.values()) + max(maxs.values()))
    total_width = sum(widths[idx] for idx in ordered) + float(margin) * float(len(ordered) - 1)
    cursor = group_center - 0.5 * total_width
    actions: list[dict[str, Any]] = []
    for idx in ordered:
        desired_center = cursor + 0.5 * widths[idx]
        cursor += widths[idx] + float(margin)
        shift = desired_center - centers[idx]
        if abs(shift) <= 1e-12:
            continue
        delta = set_vector_axis(Vector((0.0, 0.0, 0.0)), primary_axis, shift)
        entries[idx]["bounds"] = shift_bounds(entries[idx]["bounds"], delta)
        entries[idx]["translation"] += delta
        actions.append(
            {
                "iteration": iteration,
                "reason": reason,
                "name": entries[idx]["name"],
                "translation": delta,
                "primary_axis": primary_axis,
                "center_before": centers[idx],
                "center_after": desired_center,
            }
        )
    return actions


def finite_vector(vec: Vector) -> bool:
    return all(math.isfinite(float(value)) for value in (vec.x, vec.y, vec.z))


def bbox_collision_geometry(bounds: list[Vector], *, reason: str) -> dict[str, Any]:
    min_v, max_v = bounds
    vertices = [
        Vector((x, y, z))
        for x in (float(min_v.x), float(max_v.x))
        for y in (float(min_v.y), float(max_v.y))
        for z in (float(min_v.z), float(max_v.z))
    ]
    faces = [
        [0, 1, 3, 2],
        [4, 6, 7, 5],
        [0, 4, 5, 1],
        [2, 3, 7, 6],
        [0, 2, 6, 4],
        [1, 5, 7, 3],
    ]
    edges: set[tuple[int, int]] = set()
    for face in faces:
        for idx, a in enumerate(face):
            b = face[(idx + 1) % len(face)]
            edges.add(tuple(sorted((int(a), int(b)))))
    return {
        "status": "bbox_fallback",
        "reason": reason,
        "vertices": vertices,
        "faces": faces,
        "edges": sorted(edges),
        "source_vertex_count": 8,
        "sampled_vertex_count": 8,
        "hull_vertex_count": len(vertices),
        "hull_face_count": len(faces),
        "hull_edge_count": len(edges),
        "center": bounds_center(bounds),
        "bbox_min": min_v.copy(),
        "bbox_max": max_v.copy(),
    }


def sampled_vertices_for_hull(vertices: list[Vector], max_vertices: int) -> tuple[list[Vector], dict[str, Any]]:
    if max_vertices <= 0 or len(vertices) <= max_vertices:
        return vertices, {"source_vertex_count": len(vertices), "sampled_vertex_count": len(vertices), "sampling": "none"}
    directions = [
        Vector((1.0, 0.0, 0.0)),
        Vector((-1.0, 0.0, 0.0)),
        Vector((0.0, 1.0, 0.0)),
        Vector((0.0, -1.0, 0.0)),
        Vector((0.0, 0.0, 1.0)),
        Vector((0.0, 0.0, -1.0)),
        Vector((1.0, 1.0, 1.0)),
        Vector((1.0, 1.0, -1.0)),
        Vector((1.0, -1.0, 1.0)),
        Vector((-1.0, 1.0, 1.0)),
        Vector((-1.0, -1.0, 1.0)),
        Vector((-1.0, 1.0, -1.0)),
        Vector((1.0, -1.0, -1.0)),
        Vector((-1.0, -1.0, -1.0)),
    ]
    selected: set[int] = set()
    for direction in directions:
        direction.normalize()
        selected.add(max(range(len(vertices)), key=lambda idx: float(vertices[idx].dot(direction))))
    stride = max(1, len(vertices) // max(1, max_vertices - len(selected)))
    for idx in range(0, len(vertices), stride):
        selected.add(idx)
        if len(selected) >= max_vertices:
            break
    sampled = [vertices[idx] for idx in sorted(selected)[:max_vertices]]
    return sampled, {
        "source_vertex_count": len(vertices),
        "sampled_vertex_count": len(sampled),
        "sampling": "extrema_plus_stride",
        "stride": stride,
    }


def evaluated_world_vertices(obj: bpy.types.Object, max_vertices: int) -> tuple[list[Vector], dict[str, Any]]:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()
    try:
        vertices = [eval_obj.matrix_world @ vertex.co for vertex in (mesh.vertices if mesh is not None else [])]
    finally:
        if mesh is not None:
            eval_obj.to_mesh_clear()
    vertices = [Vector((float(v.x), float(v.y), float(v.z))) for v in vertices if finite_vector(v)]
    if not vertices:
        bounds = world_bounds(obj)
        return bbox_collision_geometry(bounds, reason="no_evaluated_vertices")["vertices"], {
            "source_vertex_count": 0,
            "sampled_vertex_count": 8,
            "sampling": "bbox_no_evaluated_vertices",
        }
    return sampled_vertices_for_hull(vertices, max_vertices)


def write_evaluated_world_obj(
    obj: bpy.types.Object,
    path: Path,
    *,
    max_faces: int,
    blender_decimate: bool = False,
) -> dict[str, Any]:
    start = time.monotonic()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = bpy.data.meshes.new_from_object(eval_obj, depsgraph=depsgraph)
    temp_obj: bpy.types.Object | None = None
    try:
        if mesh is None or not mesh.vertices:
            raise RuntimeError("object has no evaluated mesh vertices")
        for vertex in mesh.vertices:
            vertex.co = eval_obj.matrix_world @ vertex.co
        mesh.update()
        temp_obj = bpy.data.objects.new(f"__coacd_source_{obj.name}", mesh)
        bpy.context.collection.objects.link(temp_obj)
        temp_obj.matrix_world.identity()
        bpy.context.view_layer.update()
        mesh.calc_loop_triangles()
        source_vertex_count = len(mesh.vertices)
        source_face_count = len(mesh.loop_triangles)
        decimate_ratio = 1.0
        decimated = False
        target_faces = int(max_faces)
        if bool(blender_decimate) and target_faces > 0 and source_face_count > target_faces:
            decimate_ratio = max(float(target_faces) / max(float(source_face_count), 1.0), 0.001)
            bpy.ops.object.select_all(action="DESELECT")
            temp_obj.select_set(True)
            bpy.context.view_layer.objects.active = temp_obj
            modifier = temp_obj.modifiers.new("coacd_source_decimate", "DECIMATE")
            modifier.ratio = float(decimate_ratio)
            try:
                modifier.use_collapse_triangulate = True
            except Exception:
                pass
            bpy.ops.object.modifier_apply(modifier=modifier.name)
            mesh = temp_obj.data
            mesh.calc_loop_triangles()
            decimated = True
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            f.write("# Blender evaluated world-space mesh for CoACD\n")
            for vertex in mesh.vertices:
                co = vertex.co
                f.write(f"v {float(co.x):.9g} {float(co.y):.9g} {float(co.z):.9g}\n")
            for tri in mesh.loop_triangles:
                indices = [int(idx) + 1 for idx in tri.vertices]
                f.write(f"f {indices[0]} {indices[1]} {indices[2]}\n")
        return {
            "source_vertex_count": int(source_vertex_count),
            "source_face_count": int(source_face_count),
            "vertex_count": int(len(mesh.vertices)),
            "face_count": int(len(mesh.loop_triangles)),
            "decimated": bool(decimated),
            "decimate_ratio": float(decimate_ratio),
            "target_faces": int(target_faces),
            "path": str(path),
            "elapsed_sec": elapsed_sec(start),
        }
    finally:
        if temp_obj is not None:
            bpy.data.objects.remove(temp_obj, do_unlink=True)
        if mesh is not None and mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def read_obj_collision_geometry(path: Path, *, status: str, reason: str) -> dict[str, Any]:
    vertices: list[Vector] = []
    faces: list[list[int]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append(Vector((float(parts[1]), float(parts[2]), float(parts[3]))))
            elif parts[0] == "f" and len(parts) >= 4:
                face: list[int] = []
                for raw in parts[1:]:
                    token = raw.split("/")[0]
                    if not token:
                        continue
                    idx = int(token)
                    if idx < 0:
                        idx = len(vertices) + idx
                    else:
                        idx -= 1
                    face.append(idx)
                if len(face) >= 3:
                    faces.append(face)
    edges: set[tuple[int, int]] = set()
    for face in faces:
        for idx, a in enumerate(face):
            b = face[(idx + 1) % len(face)]
            if a != b:
                edges.add(tuple(sorted((int(a), int(b)))))
    center = Vector((0.0, 0.0, 0.0))
    for vertex in vertices:
        center += vertex
    if vertices:
        center /= float(len(vertices))
    if vertices:
        bbox_min = Vector((min(float(v.x) for v in vertices), min(float(v.y) for v in vertices), min(float(v.z) for v in vertices)))
        bbox_max = Vector((max(float(v.x) for v in vertices), max(float(v.y) for v in vertices), max(float(v.z) for v in vertices)))
    else:
        bbox_min = Vector((0.0, 0.0, 0.0))
        bbox_max = Vector((0.0, 0.0, 0.0))
    return {
        "status": status,
        "reason": reason,
        "vertices": vertices,
        "faces": faces,
        "edges": sorted(edges),
        "source_vertex_count": len(vertices),
        "sampled_vertex_count": len(vertices),
        "hull_vertex_count": len(vertices),
        "hull_face_count": len(faces),
        "hull_edge_count": len(edges),
        "center": center,
        "bbox_min": bbox_min,
        "bbox_max": bbox_max,
        "path": str(path),
    }


def build_convex_collision_geometry(obj: bpy.types.Object, *, max_vertices: int) -> dict[str, Any]:
    bounds = world_bounds(obj)
    vertices, sample_info = evaluated_world_vertices(obj, max_vertices)
    if len(vertices) < 4:
        geometry = bbox_collision_geometry(bounds, reason="fewer_than_four_vertices")
        geometry.update(sample_info)
        return geometry

    bm = bmesh.new()
    try:
        for vertex in vertices:
            bm.verts.new((float(vertex.x), float(vertex.y), float(vertex.z)))
        bm.verts.ensure_lookup_table()
        result = bmesh.ops.convex_hull(bm, input=list(bm.verts), use_existing_faces=False)
        hull_faces = [item for item in result.get("geom", []) if isinstance(item, bmesh.types.BMFace)]
        if not hull_faces:
            hull_faces = list(bm.faces)
        used_verts: list[bmesh.types.BMVert] = []
        seen: set[bmesh.types.BMVert] = set()
        for face in hull_faces:
            for vertex in face.verts:
                if vertex not in seen:
                    seen.add(vertex)
                    used_verts.append(vertex)
        if len(used_verts) < 4 or not hull_faces:
            geometry = bbox_collision_geometry(bounds, reason="convex_hull_degenerate")
            geometry.update(sample_info)
            return geometry
        index_by_vert = {vertex: idx for idx, vertex in enumerate(used_verts)}
        hull_vertices = [Vector((float(vertex.co.x), float(vertex.co.y), float(vertex.co.z))) for vertex in used_verts]
        faces: list[list[int]] = []
        edges: set[tuple[int, int]] = set()
        for face in hull_faces:
            face_indices = [index_by_vert[vertex] for vertex in face.verts if vertex in index_by_vert]
            if len(face_indices) < 3:
                continue
            faces.append(face_indices)
            for idx, a in enumerate(face_indices):
                b = face_indices[(idx + 1) % len(face_indices)]
                if a != b:
                    edges.add(tuple(sorted((int(a), int(b)))))
        if len(hull_vertices) < 4 or not faces or not edges:
            geometry = bbox_collision_geometry(bounds, reason="convex_hull_empty_topology")
            geometry.update(sample_info)
            return geometry
        center = Vector((0.0, 0.0, 0.0))
        for vertex in hull_vertices:
            center += vertex
        center /= float(len(hull_vertices))
        return {
            "status": "ok",
            "reason": "convex_hull",
            "vertices": hull_vertices,
            "faces": faces,
            "edges": sorted(edges),
            **sample_info,
            "hull_vertex_count": len(hull_vertices),
            "hull_face_count": len(faces),
            "hull_edge_count": len(edges),
            "center": center,
            "bbox_min": Vector((min(float(v.x) for v in hull_vertices), min(float(v.y) for v in hull_vertices), min(float(v.z) for v in hull_vertices))),
            "bbox_max": Vector((max(float(v.x) for v in hull_vertices), max(float(v.y) for v in hull_vertices), max(float(v.z) for v in hull_vertices))),
        }
    except Exception as exc:
        geometry = bbox_collision_geometry(bounds, reason=f"convex_hull_failed:{exc}")
        geometry.update(sample_info)
        return geometry
    finally:
        bm.free()


def bool_config(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def log_coacd_source(name: str, source_info: dict[str, Any]) -> None:
    log_stage(
        "coacd source "
        f"name={name!r} faces={source_info.get('source_face_count')}->{source_info.get('face_count')} "
        f"verts={source_info.get('source_vertex_count')}->{source_info.get('vertex_count')} "
        f"blender_decimated={source_info.get('decimated')} elapsed={float(source_info.get('elapsed_sec') or 0.0):.2f}s"
    )


def coacd_decompose_command(
    *,
    source_obj: Path,
    output_dir: Path,
    manifest_path: Path,
    part_prefix: str,
    max_vertices: int,
    coacd_config: dict[str, Any],
) -> list[str]:
    cmd = [
        str(coacd_config.get("conda_bin") or "conda"),
        "run",
        "--no-capture-output",
        "-n",
        str(coacd_config.get("env") or "fysiverse-refine"),
        "python",
        str(Path(__file__).resolve().parent / "coacd_decompose_mesh.py"),
        "--input",
        str(source_obj),
        "--output-dir",
        str(output_dir),
        "--manifest",
        str(manifest_path),
        "--part-prefix",
        str(part_prefix),
        "--save-simplified-source",
        str(output_dir / f"{part_prefix}_simplified_source.obj"),
        "--max-source-faces",
        str(int(coacd_config.get("source_max_faces", max_vertices))),
        "--simplification-backend",
        str(coacd_config.get("source_simplification_backend", "fast_simplification")),
        "--simplification-agg",
        str(float(coacd_config.get("source_simplification_agg", 7.0))),
        "--max-convex-parts",
        str(int(coacd_config.get("max_convex_parts", 8))),
        "--threshold",
        str(float(coacd_config.get("threshold", 0.05))),
        "--preprocess-mode",
        str(coacd_config.get("preprocess_mode", "auto")),
        "--preprocess-resolution",
        str(int(coacd_config.get("preprocess_resolution", 50))),
        "--resolution",
        str(int(coacd_config.get("resolution", 2000))),
        "--mcts-nodes",
        str(int(coacd_config.get("mcts_nodes", 20))),
        "--mcts-iterations",
        str(int(coacd_config.get("mcts_iterations", 80))),
        "--mcts-max-depth",
        str(int(coacd_config.get("mcts_max_depth", 3))),
        "--max-ch-vertex",
        str(int(coacd_config.get("max_ch_vertex", 256))),
        "--apx-mode",
        str(coacd_config.get("apx_mode", "ch")),
        "--seed",
        str(int(coacd_config.get("seed", 0))),
    ]
    if bool_config(coacd_config.get("real_metric"), False):
        cmd.append("--real-metric")
    if not bool_config(coacd_config.get("merge"), True):
        cmd.append("--no-merge")
    if bool_config(coacd_config.get("decimate"), False):
        cmd.append("--decimate")
    return cmd


def run_coacd_collision_from_source(
    name: str,
    *,
    source_obj: Path,
    output_dir: Path,
    manifest_path: Path,
    source_info: dict[str, Any],
    max_vertices: int,
    coacd_config: dict[str, Any],
    total_start: float,
) -> dict[str, Any]:
    env = str(coacd_config.get("env") or "fysiverse-refine")
    log_stage(
        "coacd run start "
        f"name={name!r} env={env} max_parts={int(coacd_config.get('max_convex_parts', 8))} "
        f"threshold={float(coacd_config.get('threshold', 0.05))} "
        f"source_max_faces={int(coacd_config.get('source_max_faces', max_vertices))} "
        f"simplify={coacd_config.get('source_simplification_backend', 'fast_simplification')} "
        f"workers={int(coacd_config.get('workers', 1))} "
        f"mcts_iterations={int(coacd_config.get('mcts_iterations', 80))} "
        f"timeout={int(coacd_config.get('timeout', 600))}s"
    )
    coacd_start = time.monotonic()
    proc = subprocess.run(
        coacd_decompose_command(
            source_obj=source_obj,
            output_dir=output_dir,
            manifest_path=manifest_path,
            part_prefix="coacd_part",
            max_vertices=max_vertices,
            coacd_config=coacd_config,
        ),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=max(1, int(coacd_config.get("timeout", 600))),
    )
    coacd_elapsed = elapsed_sec(coacd_start)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    part_entries = manifest.get("parts") or []
    parts: list[dict[str, Any]] = []
    for part_entry in part_entries:
        part_path = Path(str(part_entry.get("path")))
        if not part_path.is_file():
            continue
        part_geometry = read_obj_collision_geometry(part_path, status="ok", reason="coacd_part")
        if len(part_geometry.get("vertices") or []) >= 4 and part_geometry.get("faces") and part_geometry.get("edges"):
            part_geometry["part_index"] = int(part_entry.get("index", len(parts)))
            parts.append(part_geometry)
    if not parts:
        raise RuntimeError("CoACD produced no usable convex SAT parts")
    simplification = manifest.get("source_simplification") or {}
    log_stage(
        "coacd run done "
        f"name={name!r} parts={len(parts)} simplify_backend={simplification.get('backend')} "
        f"input_faces={manifest.get('coacd_input_face_count')} coacd_elapsed={coacd_elapsed:.2f}s "
        f"total_elapsed={elapsed_sec(total_start):.2f}s"
    )
    return {
        "status": "ok",
        "reason": "coacd",
        "parts": parts,
        "part_count": len(parts),
        "source_vertex_count": int(manifest.get("coacd_input_vertex_count", source_info.get("vertex_count", 0))),
        "source_face_count": int(manifest.get("coacd_input_face_count", source_info.get("face_count", 0))),
        "original_source_vertex_count": int(source_info.get("source_vertex_count", source_info.get("vertex_count", 0))),
        "original_source_face_count": int(source_info.get("source_face_count", source_info.get("face_count", 0))),
        "source_decimated": bool(simplification.get("simplified", source_info.get("decimated", False))),
        "source_decimate_ratio": float(source_info.get("decimate_ratio", 1.0)),
        "source_export_elapsed_sec": float(source_info.get("elapsed_sec") or 0.0),
        "coacd_elapsed_sec": coacd_elapsed,
        "total_elapsed_sec": elapsed_sec(total_start),
        "coacd_stdout_tail": proc.stdout[-4000:],
        "coacd_config": manifest.get("config") or {},
        "source_simplification": simplification,
        "fallback_max_vertices": int(max_vertices),
    }


def run_coacd_collision_decomposition(
    obj: bpy.types.Object,
    *,
    max_vertices: int,
    coacd_config: dict[str, Any],
) -> dict[str, Any]:
    total_start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="moge_stage_coacd_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        source_obj = tmp_path / "source_world.obj"
        source_info = write_evaluated_world_obj(
            obj,
            source_obj,
            max_faces=int(coacd_config.get("source_max_faces", max_vertices)),
            blender_decimate=bool_config(coacd_config.get("blender_source_decimate"), False),
        )
        log_coacd_source(obj.name, source_info)
        return run_coacd_collision_from_source(
            obj.name,
            source_obj=source_obj,
            output_dir=tmp_path / "parts",
            manifest_path=tmp_path / "coacd_manifest.json",
            source_info=source_info,
            max_vertices=max_vertices,
            coacd_config=coacd_config,
            total_start=total_start,
        )


def build_collision_geometry(
    obj: bpy.types.Object,
    *,
    max_vertices: int,
    decomposition_method: str,
    coacd_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    start = time.monotonic()
    if str(decomposition_method) == "coacd":
        try:
            return run_coacd_collision_decomposition(obj, max_vertices=max_vertices, coacd_config=coacd_config or {})
        except Exception as exc:
            log_stage(f"coacd failed name={obj.name!r} elapsed={elapsed_sec(start):.2f}s error={exc}; fallback=single_convex_hull")
            fallback = build_convex_collision_geometry(obj, max_vertices=max_vertices)
            fallback["decomposition_status"] = "fallback_single_convex_hull"
            fallback["decomposition_method"] = "coacd"
            fallback["decomposition_error"] = str(exc)
            fallback["total_elapsed_sec"] = elapsed_sec(start)
            return fallback
    geometry = build_convex_collision_geometry(obj, max_vertices=max_vertices)
    geometry["decomposition_method"] = "single_convex_hull"
    geometry["total_elapsed_sec"] = elapsed_sec(start)
    log_stage(f"single convex hull done name={obj.name!r} elapsed={geometry['total_elapsed_sec']:.2f}s vertices={geometry.get('hull_vertex_count')}")
    return geometry


def matrix_linear_close(a: Matrix, b: Matrix, *, eps: float = 1e-6) -> bool:
    for row in range(3):
        for col in range(3):
            if abs(float(a[row][col]) - float(b[row][col])) > eps:
                return False
    return True


def translated_collision_geometry(collision: dict[str, Any], offset: Vector) -> dict[str, Any]:
    if float(offset.length) <= 1e-12:
        return collision
    out = dict(collision)
    if isinstance(out.get("vertices"), list):
        out["vertices"] = [Vector(vertex) + offset for vertex in out.get("vertices", [])]
    if isinstance(out.get("center"), Vector):
        out["center"] = out["center"] + offset
    if isinstance(out.get("bbox_min"), Vector):
        out["bbox_min"] = out["bbox_min"] + offset
    if isinstance(out.get("bbox_max"), Vector):
        out["bbox_max"] = out["bbox_max"] + offset
    if isinstance(out.get("parts"), list):
        out["parts"] = [translated_collision_geometry(part, offset) if isinstance(part, dict) else part for part in out["parts"]]
    out["cache_translation_offset"] = offset
    return out


def collision_cache_key(entry: dict[str, Any], *, max_vertices: int, decomposition_method: str, coacd_config: dict[str, Any]) -> str:
    config_key = json.dumps(numpy_to_json(coacd_config), ensure_ascii=False, sort_keys=True)
    return f"{entry.get('index')}::{entry.get('name')}::{decomposition_method}::{int(max_vertices)}::{config_key}"


def canonical_axis(axis: Vector, eps: float) -> Vector | None:
    length = float(axis.length)
    if not math.isfinite(length) or length <= eps:
        return None
    out = axis / length
    values = [float(out.x), float(out.y), float(out.z)]
    for value in values:
        if abs(value) > eps:
            if value < 0.0:
                out = -out
            break
    return out


def unique_axes(axes: list[Vector], *, eps: float, max_axes: int) -> list[Vector]:
    out: list[Vector] = []
    seen: set[tuple[int, int, int]] = set()
    quant = max(float(eps) * 10.0, 1e-5)
    for axis in axes:
        normalized = canonical_axis(axis, eps)
        if normalized is None:
            continue
        key = tuple(int(round(float(value) / quant)) for value in (normalized.x, normalized.y, normalized.z))
        if key in seen:
            continue
        seen.add(key)
        out.append(normalized)
        if len(out) >= max_axes:
            break
    return out


def face_normal(vertices: list[Vector], face: list[int], eps: float) -> Vector | None:
    if len(face) < 3:
        return None
    origin = vertices[face[0]]
    for a_idx in range(1, len(face) - 1):
        edge_a = vertices[face[a_idx]] - origin
        edge_b = vertices[face[a_idx + 1]] - origin
        normal = edge_a.cross(edge_b)
        normalized = canonical_axis(normal, eps)
        if normalized is not None:
            return normalized
    return None


def entry_collision_parts(entry: dict[str, Any]) -> list[dict[str, Any]]:
    collision = entry.get("collision") or {}
    parts = [part for part in (collision.get("parts") or []) if isinstance(part, dict)]
    if parts:
        return parts
    return [collision] if collision else []


def collision_part_vertices(entry: dict[str, Any], collision: dict[str, Any]) -> list[Vector]:
    translation = entry.get("translation") or Vector((0.0, 0.0, 0.0))
    return [vertex + translation for vertex in collision.get("vertices", [])]


def collision_part_center(entry: dict[str, Any], collision: dict[str, Any]) -> Vector:
    translation = entry.get("translation") or Vector((0.0, 0.0, 0.0))
    return (collision.get("center") or bounds_center(entry["bounds"])) + translation


def collision_part_bounds(entry: dict[str, Any], collision: dict[str, Any]) -> list[Vector]:
    translation = entry.get("translation") or Vector((0.0, 0.0, 0.0))
    bbox_min = collision.get("bbox_min")
    bbox_max = collision.get("bbox_max")
    if isinstance(bbox_min, Vector) and isinstance(bbox_max, Vector):
        return [bbox_min + translation, bbox_max + translation]
    vertices = collision_part_vertices(entry, collision)
    if not vertices:
        return entry["bounds"]
    return [
        Vector((min(float(v.x) for v in vertices), min(float(v.y) for v in vertices), min(float(v.z) for v in vertices))),
        Vector((max(float(v.x) for v in vertices), max(float(v.y) for v in vertices), max(float(v.z) for v in vertices))),
    ]


def bounds_overlap(bounds_i: list[Vector], bounds_j: list[Vector], eps: float) -> bool:
    return all(
        min(vector_axis_value(bounds_i[1], axis), vector_axis_value(bounds_j[1], axis))
        - max(vector_axis_value(bounds_i[0], axis), vector_axis_value(bounds_j[0], axis))
        > float(eps)
        for axis in range(3)
    )


def collision_vertices(entry: dict[str, Any]) -> list[Vector]:
    vertices: list[Vector] = []
    for part in entry_collision_parts(entry):
        vertices.extend(collision_part_vertices(entry, part))
    return vertices


def collision_center(entry: dict[str, Any]) -> Vector:
    parts = entry_collision_parts(entry)
    if not parts:
        return bounds_center(entry["bounds"])
    center = Vector((0.0, 0.0, 0.0))
    total = 0
    for part in parts:
        vertices = collision_part_vertices(entry, part)
        if not vertices:
            continue
        center += collision_part_center(entry, part) * float(len(vertices))
        total += len(vertices)
    if total <= 0:
        return bounds_center(entry["bounds"])
    return center / float(total)


def collision_face_axes(collision: dict[str, Any], vertices: list[Vector], eps: float) -> list[Vector]:
    axes: list[Vector] = []
    for face in collision.get("faces", []):
        normal = face_normal(vertices, face, eps)
        if normal is not None:
            axes.append(normal)
    return axes


def collision_edge_axes(collision: dict[str, Any], vertices: list[Vector], eps: float, max_edges: int) -> list[Vector]:
    axes: list[Vector] = []
    for a, b in collision.get("edges", []):
        if int(a) >= len(vertices) or int(b) >= len(vertices):
            continue
        axis = canonical_axis(vertices[int(b)] - vertices[int(a)], eps)
        if axis is not None:
            axes.append(axis)
        if len(axes) >= max_edges:
            break
    return unique_axes(axes, eps=eps, max_axes=max_edges)


def project_vertices(vertices: list[Vector], axis: Vector) -> tuple[float, float]:
    values = [float(vertex.dot(axis)) for vertex in vertices]
    return min(values), max(values)


def convex_sat_collision_report_for_parts(
    entry_i: dict[str, Any],
    collision_i: dict[str, Any],
    entry_j: dict[str, Any],
    collision_j: dict[str, Any],
    *,
    eps: float,
    max_face_axes: int = 128,
    max_edge_axes: int = 96,
    max_cross_axes: int = 384,
) -> dict[str, Any]:
    vertices_i = collision_part_vertices(entry_i, collision_i)
    vertices_j = collision_part_vertices(entry_j, collision_j)
    if len(vertices_i) < 4 or len(vertices_j) < 4:
        return {"colliding": False, "reason": "insufficient_convex_vertices"}

    face_axes_i = unique_axes(collision_face_axes(collision_i, vertices_i, eps), eps=eps, max_axes=max_face_axes)
    face_axes_j = unique_axes(collision_face_axes(collision_j, vertices_j, eps), eps=eps, max_axes=max_face_axes)
    edge_axes_i = collision_edge_axes(collision_i, vertices_i, eps, max_edge_axes)
    edge_axes_j = collision_edge_axes(collision_j, vertices_j, eps, max_edge_axes)
    cross_axes: list[Vector] = []
    for edge_i in edge_axes_i:
        for edge_j in edge_axes_j:
            axis = edge_i.cross(edge_j)
            normalized = canonical_axis(axis, eps)
            if normalized is not None:
                cross_axes.append(normalized)
            if len(cross_axes) >= max_cross_axes:
                break
        if len(cross_axes) >= max_cross_axes:
            break
    axes = unique_axes([*face_axes_i, *face_axes_j, *cross_axes], eps=eps, max_axes=max_face_axes * 2 + max_cross_axes)
    if not axes:
        return {"colliding": False, "reason": "no_sat_axes"}

    best_overlap = float("inf")
    best_axis = axes[0]
    separating_axis: Vector | None = None
    separating_gap = 0.0
    for axis in axes:
        min_i, max_i = project_vertices(vertices_i, axis)
        min_j, max_j = project_vertices(vertices_j, axis)
        overlap = min(max_i, max_j) - max(min_i, min_j)
        if overlap <= eps:
            separating_axis = axis
            separating_gap = -float(overlap)
            break
        if overlap < best_overlap:
            best_overlap = float(overlap)
            best_axis = axis

    if separating_axis is not None:
        return {
            "colliding": False,
            "reason": "separating_axis_found",
            "separating_axis": separating_axis,
            "separating_gap": separating_gap,
            "axis_count": len(axes),
            "face_axis_count_i": len(face_axes_i),
            "face_axis_count_j": len(face_axes_j),
            "edge_axis_count_i": len(edge_axes_i),
            "edge_axis_count_j": len(edge_axes_j),
            "cross_axis_count": len(cross_axes),
        }

    center_delta = collision_part_center(entry_j, collision_j) - collision_part_center(entry_i, collision_i)
    if float(center_delta.dot(best_axis)) < 0.0:
        best_axis = -best_axis
    horizontal_length = math.sqrt(float(best_axis.x) ** 2 + float(best_axis.y) ** 2)
    return {
        "colliding": True,
        "reason": "sat_overlap",
        "penetration_depth": best_overlap,
        "mtv_axis_i_to_j": best_axis,
        "mtv_horizontal_length": horizontal_length,
        "axis_count": len(axes),
        "face_axis_count_i": len(face_axes_i),
        "face_axis_count_j": len(face_axes_j),
        "edge_axis_count_i": len(edge_axes_i),
        "edge_axis_count_j": len(edge_axes_j),
        "cross_axis_count": len(cross_axes),
        "collision_i_status": collision_i.get("status"),
        "collision_j_status": collision_j.get("status"),
    }


def convex_sat_collision_report(
    entry_i: dict[str, Any],
    entry_j: dict[str, Any],
    *,
    eps: float,
    max_face_axes: int = 128,
    max_edge_axes: int = 96,
    max_cross_axes: int = 384,
) -> dict[str, Any]:
    parts_i = entry_collision_parts(entry_i)
    parts_j = entry_collision_parts(entry_j)
    if not parts_i or not parts_j:
        return {"colliding": False, "reason": "missing_convex_collision_geometry"}

    checked = 0
    best_collision: dict[str, Any] | None = None
    first_non_collision: dict[str, Any] | None = None
    skipped = 0
    bbox_skipped = 0
    for part_i_index, collision_i in enumerate(parts_i):
        for part_j_index, collision_j in enumerate(parts_j):
            if not bounds_overlap(collision_part_bounds(entry_i, collision_i), collision_part_bounds(entry_j, collision_j), eps):
                bbox_skipped += 1
                if first_non_collision is None:
                    first_non_collision = {
                        "colliding": False,
                        "reason": "convex_part_bbox_separated",
                        "part_i": int(part_i_index),
                        "part_j": int(part_j_index),
                    }
                continue
            sat = convex_sat_collision_report_for_parts(
                entry_i,
                collision_i,
                entry_j,
                collision_j,
                eps=eps,
                max_face_axes=max_face_axes,
                max_edge_axes=max_edge_axes,
                max_cross_axes=max_cross_axes,
            )
            checked += 1
            if bool(sat.get("colliding")):
                sat["part_i"] = int(part_i_index)
                sat["part_j"] = int(part_j_index)
                if best_collision is None or float(sat.get("penetration_depth") or 0.0) > float(best_collision.get("penetration_depth") or 0.0):
                    best_collision = sat
            else:
                if sat.get("reason") == "insufficient_convex_vertices":
                    skipped += 1
                if first_non_collision is None:
                    sat["part_i"] = int(part_i_index)
                    sat["part_j"] = int(part_j_index)
                    first_non_collision = sat

    if best_collision is not None:
        best_collision["part_pair_count"] = checked
        best_collision["part_count_i"] = len(parts_i)
        best_collision["part_count_j"] = len(parts_j)
        best_collision["skipped_part_pair_count"] = skipped
        best_collision["bbox_skipped_part_pair_count"] = bbox_skipped
        best_collision["collision_i_status"] = (entry_i.get("collision") or {}).get("status")
        best_collision["collision_j_status"] = (entry_j.get("collision") or {}).get("status")
        best_collision["decomposition_method_i"] = (entry_i.get("collision") or {}).get("reason")
        best_collision["decomposition_method_j"] = (entry_j.get("collision") or {}).get("reason")
        return best_collision

    report = first_non_collision or {"colliding": False, "reason": "no_convex_part_pairs_checked"}
    report["part_pair_count"] = checked
    report["part_count_i"] = len(parts_i)
    report["part_count_j"] = len(parts_j)
    report["skipped_part_pair_count"] = skipped
    report["bbox_skipped_part_pair_count"] = bbox_skipped
    if report.get("reason") == "separating_axis_found":
        report["reason"] = "all_convex_part_pairs_separated"
    return report


def ensure_entry_collision(
    entry: dict[str, Any],
    *,
    max_vertices: int,
    decomposition_method: str,
    coacd_config: dict[str, Any],
    cache: dict[str, dict[str, Any]],
) -> None:
    if entry.get("collision"):
        return
    key = collision_cache_key(entry, max_vertices=max_vertices, decomposition_method=decomposition_method, coacd_config=coacd_config)
    current_matrix = entry["object"].matrix_world.copy()
    if key not in cache:
        build_start = time.monotonic()
        log_stage(
            "collision build start "
            f"name={entry.get('name')!r} method={decomposition_method} max_vertices={int(max_vertices)} "
            f"cache=miss"
        )
        cache[key] = {
            "matrix_world": current_matrix,
            "collision": build_collision_geometry(
                entry["object"],
                max_vertices=int(max_vertices),
                decomposition_method=decomposition_method,
                coacd_config=coacd_config,
            ),
        }
        collision = cache[key]["collision"]
        log_stage(
            "collision build done "
            f"name={entry.get('name')!r} status={collision.get('status')} reason={collision.get('reason')} "
            f"parts={len(collision.get('parts') or []) or 1} elapsed={elapsed_sec(build_start):.2f}s"
        )
    cached = cache[key]
    cached_matrix = cached.get("matrix_world")
    if isinstance(cached_matrix, Matrix) and matrix_linear_close(cached_matrix, current_matrix):
        offset = current_matrix.translation - cached_matrix.translation
        entry["collision"] = translated_collision_geometry(cached["collision"], offset)
        log_stage(
            "collision cache hit "
            f"name={entry.get('name')!r} offset_length={float(offset.length):.6g} "
            f"parts={len(entry_collision_parts(entry))}"
        )
        return
    build_start = time.monotonic()
    log_stage(f"collision cache stale name={entry.get('name')!r}; rebuilding")
    cache[key] = {
        "matrix_world": current_matrix,
        "collision": build_collision_geometry(
            entry["object"],
            max_vertices=int(max_vertices),
            decomposition_method=decomposition_method,
            coacd_config=coacd_config,
        ),
    }
    entry["collision"] = cache[key]["collision"]
    log_stage(
        "collision rebuild done "
        f"name={entry.get('name')!r} parts={len(entry_collision_parts(entry))} elapsed={elapsed_sec(build_start):.2f}s"
    )


def prebuild_entry_collisions(
    entries: list[dict[str, Any]],
    indices: set[int],
    *,
    max_vertices: int,
    decomposition_method: str,
    coacd_config: dict[str, Any],
    cache: dict[str, dict[str, Any]],
    phase: str,
) -> dict[str, Any]:
    valid_indices = sorted({int(idx) for idx in indices if 0 <= int(idx) < len(entries)})
    if not valid_indices:
        return {"phase": phase, "requested": 0, "built": 0, "cache_hits": 0, "failures": []}

    if str(decomposition_method) != "coacd" or int(coacd_config.get("workers", 1)) <= 1:
        built = 0
        cache_hits = 0
        failures: list[dict[str, Any]] = []
        for idx in valid_indices:
            before = len(cache)
            try:
                ensure_entry_collision(
                    entries[idx],
                    max_vertices=max_vertices,
                    decomposition_method=decomposition_method,
                    coacd_config=coacd_config,
                    cache=cache,
                )
                if len(cache) == before:
                    cache_hits += 1
                else:
                    built += 1
            except Exception as exc:
                failures.append({"index": idx, "name": entries[idx]["name"], "error": str(exc)})
                raise
        return {"phase": phase, "requested": len(valid_indices), "built": built, "cache_hits": cache_hits, "failures": failures}

    workers = max(1, int(coacd_config.get("workers", 1)))
    tasks: list[dict[str, Any]] = []
    cache_hits = 0
    prebuild_start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="moge_stage_coacd_batch_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        for idx in valid_indices:
            entry = entries[idx]
            key = collision_cache_key(entry, max_vertices=max_vertices, decomposition_method=decomposition_method, coacd_config=coacd_config)
            current_matrix = entry["object"].matrix_world.copy()
            cached = cache.get(key)
            cached_matrix = cached.get("matrix_world") if isinstance(cached, dict) else None
            if isinstance(cached_matrix, Matrix) and matrix_linear_close(cached_matrix, current_matrix):
                cache_hits += 1
                continue
            source_dir = tmp_path / f"entry_{idx:03d}"
            source_obj = source_dir / "source_world.obj"
            source_start = time.monotonic()
            source_info = write_evaluated_world_obj(
                entry["object"],
                source_obj,
                max_faces=int(coacd_config.get("source_max_faces", max_vertices)),
                blender_decimate=bool_config(coacd_config.get("blender_source_decimate"), False),
            )
            log_coacd_source(entry["name"], source_info)
            tasks.append(
                {
                    "index": idx,
                    "entry": entry,
                    "key": key,
                    "matrix_world": current_matrix,
                    "source_obj": source_obj,
                    "output_dir": source_dir / "parts",
                    "manifest_path": source_dir / "coacd_manifest.json",
                    "source_info": source_info,
                    "source_elapsed_sec": elapsed_sec(source_start),
                }
            )

        log_stage(
            f"{phase}: coacd prebuild start requested={len(valid_indices)} tasks={len(tasks)} "
            f"cache_hits={cache_hits} workers={workers} source_export_elapsed={elapsed_sec(prebuild_start):.2f}s"
        )
        failures: list[dict[str, Any]] = []
        if tasks:
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(workers, len(tasks))) as executor:
                future_to_task = {
                    executor.submit(
                        run_coacd_collision_from_source,
                        task["entry"]["name"],
                        source_obj=task["source_obj"],
                        output_dir=task["output_dir"],
                        manifest_path=task["manifest_path"],
                        source_info=task["source_info"],
                        max_vertices=max_vertices,
                        coacd_config=coacd_config,
                        total_start=time.monotonic(),
                    ): task
                    for task in tasks
                }
                for future in concurrent.futures.as_completed(future_to_task):
                    task = future_to_task[future]
                    entry = task["entry"]
                    try:
                        cache[task["key"]] = {
                            "matrix_world": task["matrix_world"],
                            "collision": future.result(),
                        }
                    except Exception as exc:
                        log_stage(f"{phase}: coacd prebuild failed name={entry['name']!r} error={exc}; fallback=single_convex_hull")
                        fallback = build_convex_collision_geometry(entry["object"], max_vertices=max_vertices)
                        fallback["decomposition_status"] = "fallback_single_convex_hull"
                        fallback["decomposition_method"] = "coacd"
                        fallback["decomposition_error"] = str(exc)
                        cache[task["key"]] = {
                            "matrix_world": task["matrix_world"],
                            "collision": fallback,
                        }
                        failures.append({"index": int(task["index"]), "name": entry["name"], "error": str(exc)})

    for idx in valid_indices:
        entries[idx]["collision"] = {}
        ensure_entry_collision(
            entries[idx],
            max_vertices=max_vertices,
            decomposition_method=decomposition_method,
            coacd_config=coacd_config,
            cache=cache,
        )
    report = {
        "phase": phase,
        "requested": len(valid_indices),
        "built": len(tasks),
        "cache_hits": cache_hits,
        "workers": workers,
        "failures": failures,
        "elapsed_sec": elapsed_sec(prebuild_start),
    }
    log_stage(
        f"{phase}: coacd prebuild done requested={report['requested']} built={report['built']} "
        f"cache_hits={cache_hits} failures={len(failures)} elapsed={report['elapsed_sec']:.2f}s"
    )
    return report


def convex_pair_reports(
    entries: list[dict[str, Any]],
    *,
    eps: float,
    min_overlap_volume_ratio: float,
    collision_eps: float,
    skip_pairs: set[tuple[int, int]] | None = None,
    ensure_collision: Any | None = None,
    phase: str = "convex_pair_reports",
) -> list[dict[str, Any]]:
    start = time.monotonic()
    skip_pairs = skip_pairs or set()
    reports: list[dict[str, Any]] = []
    broad_phase = pair_reports(entries, eps=eps, min_overlap_volume_ratio=min_overlap_volume_ratio, skip_pairs=skip_pairs)
    log_stage(f"{phase}: broad_phase_candidates={len(broad_phase)} skip_pairs={len(skip_pairs)}")
    for item in broad_phase:
        i, j = [int(v) for v in item["indices"]]
        pair_start = time.monotonic()
        if len(broad_phase) <= 64:
            log_stage(f"{phase}: sat pair start {i}:{entries[i]['name']!r} vs {j}:{entries[j]['name']!r}")
        if ensure_collision is not None:
            ensure_collision(entries[i])
            ensure_collision(entries[j])
        sat = convex_sat_collision_report(entries[i], entries[j], eps=collision_eps)
        pair_elapsed = elapsed_sec(pair_start)
        if len(broad_phase) <= 64 or pair_elapsed > 1.0:
            log_stage(
                f"{phase}: sat pair done {i}:{entries[i]['name']!r} vs {j}:{entries[j]['name']!r} "
                f"colliding={bool(sat.get('colliding'))} reason={sat.get('reason')} "
                f"checked_parts={sat.get('part_pair_count')} bbox_skipped={sat.get('bbox_skipped_part_pair_count')} "
                f"elapsed={pair_elapsed:.2f}s"
            )
        if not bool(sat.get("colliding")):
            continue
        report = dict(item)
        report["collision_method"] = "convex_hull_sat"
        report["convex_collision"] = sat
        reports.append(report)
    log_stage(f"{phase}: narrow_phase_collisions={len(reports)} elapsed={elapsed_sec(start):.2f}s")
    return reports


def convex_separation_delta(
    collision_report: dict[str, Any],
    *,
    margin: float,
    up: Vector,
    min_horizontal_axis: float,
) -> tuple[Vector | None, Vector | None, dict[str, Any]]:
    sat = collision_report.get("convex_collision") or {}
    axis = sat.get("mtv_axis_i_to_j")
    if not isinstance(axis, Vector):
        return None, None, {"reason": "missing_mtv_axis"}
    up_dir = Vector(up)
    if up_dir.length < 1e-10:
        up_dir = Vector((0.0, 0.0, 1.0))
    else:
        up_dir.normalize()
    horizontal = axis - up_dir * float(axis.dot(up_dir))
    horizontal_length = float(horizontal.length)
    if horizontal_length < float(min_horizontal_axis):
        return None, None, {
            "reason": "mtv_axis_is_mostly_vertical",
            "mtv_axis_i_to_j": axis,
            "mtv_horizontal_length": horizontal_length,
            "min_horizontal_axis": float(min_horizontal_axis),
        }
    horizontal.normalize()
    penetration = max(float(sat.get("penetration_depth") or 0.0), 0.0)
    amount = 0.5 * ((penetration / max(horizontal_length, 1e-8)) + float(margin))
    delta_i = -horizontal * amount
    delta_j = horizontal * amount
    return delta_i, delta_j, {
        "reason": "convex_mtv_horizontal_projection",
        "mtv_axis_i_to_j": axis,
        "mtv_horizontal_length": horizontal_length,
        "penetration_depth": penetration,
        "separation_amount_per_object": amount,
        "horizontal_axis": horizontal,
    }


def separate_overlaps_convex(
    objects: list[bpy.types.Object],
    *,
    margin: float,
    max_iters: int,
    min_overlap_volume_ratio: float,
    eps: float,
    up: Vector,
    skip_pairs: set[tuple[int, int]] | None,
    hull_max_vertices: int,
    collision_eps: float,
    min_horizontal_axis: float,
    convex_decomposition_method: str,
    coacd_config: dict[str, Any],
    collision_cache: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    stage_start = time.monotonic()
    entries: list[dict[str, Any]] = []
    for idx, obj in enumerate(objects):
        bounds = world_bounds(obj)
        entries.append(
            {
                "index": idx,
                "name": obj.name,
                "object": obj,
                "bounds": bounds,
                "translation": Vector((0.0, 0.0, 0.0)),
                "collision": {},
            }
        )
    skip_pairs = {tuple(sorted((int(a), int(b)))) for a, b in (skip_pairs or set()) if int(a) != int(b)}
    if len(entries) < 2:
        return [{"mode": "convex_hull_sat", "initial_bbox_candidates": [], "initial_collisions": [], "pairs": [], "fallback_actions": [], "final_collisions": [], "objects": [], "skipped_support_pairs": []}]

    initial_bbox_candidates = pair_reports(entries, eps=eps, min_overlap_volume_ratio=min_overlap_volume_ratio, skip_pairs=skip_pairs)
    log_stage(
        "overlap stage start "
        f"objects={len(entries)} initial_bbox_candidates={len(initial_bbox_candidates)} max_iters={int(max_iters)} "
        f"method=convex_hull_sat decomposition={convex_decomposition_method}"
    )
    stage_collision_cache: dict[str, dict[str, Any]] = collision_cache if collision_cache is not None else {}

    def ensure_collision(entry: dict[str, Any]) -> None:
        ensure_entry_collision(
            entry,
            max_vertices=int(hull_max_vertices),
            decomposition_method=convex_decomposition_method,
            coacd_config=coacd_config,
            cache=stage_collision_cache,
        )

    prebuild_indices = {int(idx) for report in initial_bbox_candidates for idx in (report.get("indices") or [])}
    prebuild_report = prebuild_entry_collisions(
        entries,
        prebuild_indices,
        max_vertices=int(hull_max_vertices),
        decomposition_method=convex_decomposition_method,
        coacd_config=coacd_config,
        cache=stage_collision_cache,
        phase="overlap",
    )

    initial_collisions = convex_pair_reports(
        entries,
        eps=eps,
        min_overlap_volume_ratio=min_overlap_volume_ratio,
        collision_eps=collision_eps,
        skip_pairs=skip_pairs,
        ensure_collision=ensure_collision,
        phase="overlap.initial",
    )
    actions: list[dict[str, Any]] = []
    skipped_actions: list[dict[str, Any]] = []
    skipped_reports: list[dict[str, Any]] = []
    for i, j in sorted(skip_pairs):
        if i < 0 or j < 0 or i >= len(entries) or j >= len(entries):
            continue
        skipped_reports.append({"indices": [i, j], "pair": [entries[i]["name"], entries[j]["name"]], "reason": "support_relation_vertical_only"})

    for iteration in range(max(0, int(max_iters))):
        moved = False
        current = convex_pair_reports(
            entries,
            eps=eps,
            min_overlap_volume_ratio=min_overlap_volume_ratio,
            collision_eps=collision_eps,
            skip_pairs=skip_pairs,
            ensure_collision=ensure_collision,
            phase=f"overlap.iter{iteration}",
        )
        if not current:
            break
        for info in current:
            i, j = [int(v) for v in info["indices"]]
            delta_i, delta_j, move_info = convex_separation_delta(info, margin=margin, up=up, min_horizontal_axis=min_horizontal_axis)
            if delta_i is None or delta_j is None:
                skipped = dict(info)
                skipped.update({"iteration": iteration, "skip": move_info})
                skipped_actions.append(skipped)
                continue
            entries[i]["bounds"] = shift_bounds(entries[i]["bounds"], delta_i)
            entries[j]["bounds"] = shift_bounds(entries[j]["bounds"], delta_j)
            entries[i]["translation"] += delta_i
            entries[j]["translation"] += delta_j
            moved = True
            item = dict(info)
            item.update(
                {
                    "iteration": iteration,
                    "pair": [entries[i]["name"], entries[j]["name"]],
                    "margin": float(margin),
                    "translation_i": delta_i,
                    "translation_j": delta_j,
                    "separation": move_info,
                }
            )
            actions.append(item)
        if not moved:
            break

    fallback_actions: list[dict[str, Any]] = []
    for fallback_iteration in range(10):
        remaining = convex_pair_reports(
            entries,
            eps=eps,
            min_overlap_volume_ratio=min_overlap_volume_ratio,
            collision_eps=collision_eps,
            skip_pairs=skip_pairs,
            ensure_collision=ensure_collision,
            phase=f"overlap.fallback{fallback_iteration}",
        )
        if not remaining:
            break
        before_count = len(fallback_actions)
        components = connected_components_from_pairs(
            [
                report
                for report in remaining
                if float(((report.get("convex_collision") or {}).get("mtv_horizontal_length") or 0.0)) >= float(min_horizontal_axis)
            ],
            len(entries),
        )
        for component in components:
            fallback_actions.extend(
                pack_indices_along_primary_axis(
                    entries,
                    component,
                    margin=margin,
                    reason="component_pack_residual_convex_collision",
                    iteration=fallback_iteration,
                    up=up,
                )
            )
        if len(fallback_actions) == before_count:
            break

    final_collisions = convex_pair_reports(
        entries,
        eps=eps,
        min_overlap_volume_ratio=min_overlap_volume_ratio,
        collision_eps=collision_eps,
        skip_pairs=skip_pairs,
        ensure_collision=ensure_collision,
        phase="overlap.final",
    )
    for entry in entries:
        if entry["translation"].length > 1e-12:
            apply_world_translation(entry["object"], entry["translation"])
    bpy.context.view_layer.update()
    objects_report = [
        {
            "index": entry["index"],
            "name": entry["name"],
            "total_translation": entry["translation"],
            "bbox_after": bounds_to_json(world_bounds(entry["object"])),
            "convex_collision_geometry": {
                key: value
                for key, value in (entry.get("collision") or {}).items()
                if key not in {"vertices", "faces", "edges", "center", "parts"}
            },
            "convex_collision_part_count": len(entry_collision_parts(entry)),
        }
        for entry in entries
    ]
    log_stage(
        "overlap stage done "
        f"actions={len(actions)} skipped_actions={len(skipped_actions)} fallback_actions={len(fallback_actions)} "
        f"remaining={len(final_collisions)} elapsed={elapsed_sec(stage_start):.2f}s"
    )
    return [
        {
            "mode": "convex_hull_sat",
            "broad_phase": "bbox_overlap",
            "narrow_phase": "convex_hull_sat",
            "convex_hull_max_vertices": int(hull_max_vertices),
            "convex_decomposition_method": str(convex_decomposition_method),
            "coacd_config": dict(coacd_config),
            "collision_prebuild": prebuild_report,
            "convex_collision_eps": float(collision_eps),
            "convex_min_horizontal_axis": float(min_horizontal_axis),
            "initial_bbox_candidates": initial_bbox_candidates,
            "initial_overlaps": initial_collisions,
            "pairs": actions,
            "skipped_support_pairs": skipped_reports,
            "skipped_collision_actions": skipped_actions,
            "fallback_actions": fallback_actions,
            "final_overlaps": final_collisions,
            "remaining_overlap_count": len(final_collisions),
            "elapsed_sec": elapsed_sec(stage_start),
            "objects": objects_report,
        }
    ]


def separate_overlaps(
    objects: list[bpy.types.Object],
    *,
    margin: float,
    max_iters: int,
    min_overlap_volume_ratio: float,
    eps: float,
    up: Vector,
    skip_pairs: set[tuple[int, int]] | None = None,
    collision_method: str = "bbox",
    hull_max_vertices: int = 8000,
    collision_eps: float = 1e-6,
    min_horizontal_axis: float = 0.25,
    convex_decomposition_method: str = "single_convex_hull",
    coacd_config: dict[str, Any] | None = None,
    collision_cache: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if str(collision_method) == "convex_hull_sat":
        return separate_overlaps_convex(
            objects,
            margin=margin,
            max_iters=max_iters,
            min_overlap_volume_ratio=min_overlap_volume_ratio,
            eps=eps,
            up=up,
            skip_pairs=skip_pairs,
            hull_max_vertices=hull_max_vertices,
            collision_eps=collision_eps,
            min_horizontal_axis=min_horizontal_axis,
            convex_decomposition_method=convex_decomposition_method,
            coacd_config=coacd_config or {},
            collision_cache=collision_cache,
        )

    entries: list[dict[str, Any]] = [
        {"index": idx, "name": obj.name, "object": obj, "bounds": world_bounds(obj), "translation": Vector((0.0, 0.0, 0.0))}
        for idx, obj in enumerate(objects)
    ]
    skip_pairs = {tuple(sorted((int(a), int(b)))) for a, b in (skip_pairs or set()) if int(a) != int(b)}
    if len(entries) < 2:
        return [{"initial_overlaps": [], "pairs": [], "fallback_actions": [], "final_overlaps": [], "objects": [], "skipped_support_pairs": []}]
    axes = horizontal_axis_indices(up)
    initial = pair_reports(entries, eps=eps, min_overlap_volume_ratio=min_overlap_volume_ratio, skip_pairs=skip_pairs)
    actions: list[dict[str, Any]] = []
    skipped_reports: list[dict[str, Any]] = []
    for i, j in sorted(skip_pairs):
        if i < 0 or j < 0 or i >= len(entries) or j >= len(entries):
            continue
        skipped_reports.append({"indices": [i, j], "pair": [entries[i]["name"], entries[j]["name"]], "reason": "support_relation_vertical_only"})
    for iteration in range(max(0, int(max_iters))):
        moved = False
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                if (i, j) in skip_pairs:
                    continue
                info = bbox_overlap_report(entries[i]["bounds"], entries[j]["bounds"])
                overlaps = info["overlap_xyz"]
                if any(float(v) <= eps for v in overlaps):
                    continue
                if float(info["overlap_volume_ratio_to_smaller"]) < float(min_overlap_volume_ratio):
                    continue
                ci = bounds_center(entries[i]["bounds"])
                cj = bounds_center(entries[j]["bounds"])
                axis = min(axes, key=lambda a: float(overlaps[a]))
                direction = 1.0 if vector_axis_value(ci, axis) >= vector_axis_value(cj, axis) else -1.0
                amount = 0.5 * (float(overlaps[axis]) + float(margin))
                delta_i = set_vector_axis(Vector((0.0, 0.0, 0.0)), axis, direction * amount)
                delta_j = -delta_i
                entries[i]["bounds"] = shift_bounds(entries[i]["bounds"], delta_i)
                entries[j]["bounds"] = shift_bounds(entries[j]["bounds"], delta_j)
                entries[i]["translation"] += delta_i
                entries[j]["translation"] += delta_j
                moved = True
                item = dict(info)
                item.update(
                    {
                        "iteration": iteration,
                        "pair": [entries[i]["name"], entries[j]["name"]],
                        "axis": "xyz"[axis],
                        "separation_axis_overlap": float(overlaps[axis]),
                        "margin": float(margin),
                        "translation_i": delta_i,
                        "translation_j": delta_j,
                    }
                )
                actions.append(item)
        if not moved:
            break
    fallback_actions: list[dict[str, Any]] = []
    for fallback_iteration in range(10):
        remaining = pair_reports(entries, eps=eps, min_overlap_volume_ratio=min_overlap_volume_ratio, skip_pairs=skip_pairs)
        if not remaining:
            break
        before_count = len(fallback_actions)
        for component in connected_components_from_pairs(remaining, len(entries)):
            fallback_actions.extend(
                pack_indices_along_primary_axis(
                    entries,
                    component,
                    margin=margin,
                    reason="component_pack_residual_overlap",
                    iteration=fallback_iteration,
                    up=up,
                )
            )
        if len(fallback_actions) == before_count:
            break
    final = pair_reports(entries, eps=eps, min_overlap_volume_ratio=min_overlap_volume_ratio, skip_pairs=skip_pairs)
    if final:
        involved = sorted({int(idx) for report in final for idx in (report.get("indices") or [])})
        fallback_actions.extend(
            pack_indices_along_primary_axis(
                entries,
                involved,
                margin=margin,
                reason="global_pack_residual_overlap",
                iteration=0,
                up=up,
            )
        )
        final = pair_reports(entries, eps=eps, min_overlap_volume_ratio=min_overlap_volume_ratio, skip_pairs=skip_pairs)
    for entry in entries:
        if entry["translation"].length > 1e-12:
            apply_world_translation(entry["object"], entry["translation"])
    bpy.context.view_layer.update()
    objects_report = [
        {
            "index": entry["index"],
            "name": entry["name"],
            "total_translation": entry["translation"],
            "bbox_after": bounds_to_json(world_bounds(entry["object"])),
        }
        for entry in entries
    ]
    return [
        {
            "initial_overlaps": initial,
            "pairs": actions,
            "skipped_support_pairs": skipped_reports,
            "fallback_actions": fallback_actions,
            "final_overlaps": final,
            "remaining_overlap_count": len(final),
            "objects": objects_report,
        }
    ]


def coacd_config_from_plan(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "conda_bin": str(plan.get("coacd_conda_bin") or "conda"),
        "env": str(plan.get("coacd_env") or "fysiverse-refine"),
        "timeout": int(plan.get("coacd_timeout", 120)),
        "workers": int(plan.get("coacd_workers", 1)),
        "source_max_faces": int(plan.get("coacd_source_max_faces", 5000)),
        "source_simplification_backend": str(plan.get("coacd_source_simplification_backend", "fast_simplification")),
        "source_simplification_agg": float(plan.get("coacd_source_simplification_agg", 7.0)),
        "blender_source_decimate": bool_config(plan.get("coacd_blender_source_decimate"), False),
        "max_convex_parts": int(plan.get("coacd_max_convex_parts", 8)),
        "threshold": float(plan.get("coacd_threshold", 0.05)),
        "preprocess_mode": str(plan.get("coacd_preprocess_mode", "auto")),
        "preprocess_resolution": int(plan.get("coacd_preprocess_resolution", 50)),
        "resolution": int(plan.get("coacd_resolution", 2000)),
        "mcts_nodes": int(plan.get("coacd_mcts_nodes", 20)),
        "mcts_iterations": int(plan.get("coacd_mcts_iterations", 80)),
        "mcts_max_depth": int(plan.get("coacd_mcts_max_depth", 3)),
        "max_ch_vertex": int(plan.get("coacd_max_ch_vertex", 256)),
        "apx_mode": str(plan.get("coacd_apx_mode", "ch")),
        "seed": int(plan.get("coacd_seed", 0)),
        "real_metric": bool_config(plan.get("coacd_real_metric"), False),
        "merge": bool_config(plan.get("coacd_merge"), True),
        "decimate": bool_config(plan.get("coacd_decimate"), False),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_start = time.monotonic()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    convex_decomposition_method = str(plan.get("convex_decomposition_method", "single_convex_hull"))
    coacd_config = coacd_config_from_plan(plan)
    log_stage(
        "run start "
        f"stage={plan.get('stage', {}).get('key')} input={args.input} output={args.output} "
        f"decomposition={convex_decomposition_method} coacd_parts={coacd_config.get('max_convex_parts')} "
        f"coacd_threshold={coacd_config.get('threshold')} source_max_faces={coacd_config.get('source_max_faces')}"
    )
    scene_graph: dict[str, Any] | None = None
    scene_graph_path = plan.get("scene_graph")
    if isinstance(scene_graph_path, str) and scene_graph_path and Path(scene_graph_path).is_file():
        scene_graph = json.loads(Path(scene_graph_path).read_text(encoding="utf-8"))
    elif isinstance(scene_graph_path, dict):
        scene_graph = scene_graph_path
    bpy.ops.wm.open_mainfile(filepath=str(args.input))
    objects = mesh_objects()
    if not objects:
        raise RuntimeError(f"No mesh objects found in {args.input}")
    log_stage(f"opened blend objects={len(objects)} elapsed={elapsed_sec(run_start):.2f}s")

    transform = plan["transform"]
    matrix = matrix_from_list(transform["matrix_4x4"])
    target_up = vector_from_list(transform["target_up"])
    apply_global_matrix(objects, matrix)
    log_stage(f"applied global matrix elapsed={elapsed_sec(run_start):.2f}s")

    support_adjust_reports: Any = {}
    support_skip_pairs: set[tuple[int, int]] = set()
    stage_collision_cache: dict[str, dict[str, Any]] = {}
    if bool(plan.get("support_adjust", False)):
        support_adjust_reports = separate_support_relations_vertically(
            objects,
            up=target_up,
            scene_graph=scene_graph,
            support_adjust=bool(plan.get("support_adjust", False)),
            support_require_scene_graph=bool(plan.get("support_require_scene_graph", False)),
            support_gap=float(plan.get("support_gap", 0.001)),
            support_xy_overlap_ratio=float(plan.get("support_xy_overlap_ratio", 0.15)),
            support_max_gap=float(plan.get("support_max_gap", -1.0)),
            support_max_gap_ratio=float(plan.get("support_max_gap_ratio", 0.03)),
            support_max_penetration=float(plan.get("support_max_penetration", -1.0)),
            support_max_penetration_ratio=float(plan.get("support_max_penetration_ratio", 0.01)),
            support_min_lower_area_ratio=float(plan.get("support_min_lower_area_ratio", 0.25)),
            eps=float(plan.get("bbox_overlap_eps", 1e-8)),
            support_collision_method=str(plan.get("support_collision_method") or plan.get("overlap_collision_method", "convex_hull_sat")),
            hull_max_vertices=int(plan.get("convex_hull_max_vertices", 8000)),
            collision_eps=float(plan.get("convex_collision_eps", 1e-6)),
            convex_decomposition_method=convex_decomposition_method,
            coacd_config=coacd_config,
            collision_cache=stage_collision_cache,
        )
        for rel in support_adjust_reports.get("support_relations") or []:
            upper_idx = safe_int(first_present(rel, ("upper_index", "child_index")))
            lower_idx = safe_int(first_present(rel, ("lower_index", "parent_index")))
            if upper_idx is None or lower_idx is None or upper_idx == lower_idx:
                continue
            support_skip_pairs.add(tuple(sorted((upper_idx, lower_idx))))

    overlap_reports: list[dict[str, Any]] = []
    if bool(plan.get("separate_overlaps")):
        overlap_reports = separate_overlaps(
            objects,
            margin=float(plan.get("bbox_overlap_margin", 0.01)),
            max_iters=int(plan.get("bbox_overlap_iters", 32)),
            min_overlap_volume_ratio=float(plan.get("bbox_min_overlap_volume_ratio", 0.0)),
            eps=float(plan.get("bbox_overlap_eps", 1e-8)),
            up=target_up,
            skip_pairs=support_skip_pairs,
            collision_method=str(plan.get("overlap_collision_method", "bbox")),
            hull_max_vertices=int(plan.get("convex_hull_max_vertices", 8000)),
            collision_eps=float(plan.get("convex_collision_eps", 1e-6)),
            min_horizontal_axis=float(plan.get("convex_min_horizontal_axis", 0.25)),
            convex_decomposition_method=convex_decomposition_method,
            coacd_config=coacd_config,
            collision_cache=stage_collision_cache,
        )

    bbox_snap_reports: Any = []
    if bool(plan.get("snap_bbox_to_ground")):
        bbox_snap_mode = str(plan.get("bbox_snap_mode") or "per_object")
        if bbox_snap_mode == "global_support_aware":
            bbox_snap_reports = global_ground_and_support_adjust(
                objects,
                ground_y=float(plan.get("ground_y", 0.0)),
                up=target_up,
                scene_graph=scene_graph,
                support_adjust=False,
                support_require_scene_graph=bool(plan.get("support_require_scene_graph", False)),
                support_gap=float(plan.get("support_gap", 0.001)),
                support_xy_overlap_ratio=float(plan.get("support_xy_overlap_ratio", 0.15)),
                support_max_gap=float(plan.get("support_max_gap", -1.0)),
                support_max_gap_ratio=float(plan.get("support_max_gap_ratio", 0.03)),
                support_max_penetration=float(plan.get("support_max_penetration", -1.0)),
                support_max_penetration_ratio=float(plan.get("support_max_penetration_ratio", 0.01)),
                support_min_lower_area_ratio=float(plan.get("support_min_lower_area_ratio", 0.25)),
                eps=float(plan.get("bbox_overlap_eps", 1e-8)),
            )
        else:
            bbox_snap_reports = snap_to_ground(objects, float(plan.get("ground_y", 0.0)), target_up)

    origin_reports = recenter_origins_to_bbox_centers(objects)
    log_stage(f"recentered origins elapsed={elapsed_sec(run_start):.2f}s")

    transform_report = dict(transform)
    transform_report.update(
        {
            "bbox_snap_to_ground": bool(plan.get("snap_bbox_to_ground")),
            "bbox_snap_mode": str(plan.get("bbox_snap_mode") or "per_object"),
            "bbox_snap_objects": bbox_snap_reports,
            "support_adjust_objects": support_adjust_reports,
            "support_adjust": bool(plan.get("support_adjust", False)),
            "support_require_scene_graph": bool(plan.get("support_require_scene_graph", False)),
            "support_gap": float(plan.get("support_gap", 0.001)),
            "support_xy_overlap_ratio": float(plan.get("support_xy_overlap_ratio", 0.15)),
            "support_max_gap": float(plan.get("support_max_gap", -1.0)),
            "support_max_gap_ratio": float(plan.get("support_max_gap_ratio", 0.03)),
            "support_max_penetration": float(plan.get("support_max_penetration", -1.0)),
            "support_max_penetration_ratio": float(plan.get("support_max_penetration_ratio", 0.01)),
            "support_min_lower_area_ratio": float(plan.get("support_min_lower_area_ratio", 0.25)),
            "bbox_separate_overlaps": bool(plan.get("separate_overlaps")),
            "bbox_overlap_margin": float(plan.get("bbox_overlap_margin", 0.01)),
            "bbox_overlap_iters": int(plan.get("bbox_overlap_iters", 32)),
            "bbox_min_overlap_volume_ratio": float(plan.get("bbox_min_overlap_volume_ratio", 0.0)),
            "bbox_overlap_eps": float(plan.get("bbox_overlap_eps", 1e-8)),
            "overlap_collision_method": str(plan.get("overlap_collision_method", "bbox")),
            "convex_hull_max_vertices": int(plan.get("convex_hull_max_vertices", 8000)),
            "convex_decomposition_method": str(convex_decomposition_method),
            "coacd_config": dict(coacd_config),
            "convex_collision_eps": float(plan.get("convex_collision_eps", 1e-6)),
            "convex_min_horizontal_axis": float(plan.get("convex_min_horizontal_axis", 0.25)),
            "bbox_overlap_reports": overlap_reports,
            "scene_graph": str(scene_graph_path) if isinstance(scene_graph_path, str) else None,
            "scene_graph_status": scene_graph.get("status") if isinstance(scene_graph, dict) else None,
            "origin_recentered_to_bbox_center": True,
            "origin_recenter_objects": origin_reports,
            "elapsed_sec": elapsed_sec(run_start),
        }
    )

    packed_images: list[str] = []
    if args.pack_textures:
        bpy.data.use_autopack = True
        packed_images = pack_external_images()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output))
    log_stage(f"saved blend elapsed={elapsed_sec(run_start):.2f}s")

    report = {
        "status": "ok",
        "stage": plan.get("stage", {}),
        "input": plan.get("input", {"blend": str(args.input)}),
        "moge": plan.get("moge", {}),
        "transform": transform_report,
        "outputs": {"blend": str(args.output), "report": str(args.report)},
        "objects": [obj.name for obj in objects],
        "packed_images": packed_images,
        "elapsed_sec": elapsed_sec(run_start),
        "note": "This stage was applied directly to the source Blend object transforms, preserving Blender materials and image textures.",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(numpy_to_json(report), ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    report = run(parse_args())
    print(f"Applied MoGe stage {report.get('stage', {}).get('key')} to {len(report.get('objects', []))} mesh objects")
    print(f"Wrote Blend: {report['outputs']['blend']}")
    print(f"Wrote report: {report['outputs']['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
