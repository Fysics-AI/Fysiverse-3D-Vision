#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import bpy
from mathutils import Matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply per-object nvdiffrast pose deltas to a Blend scene.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--object-pose-optimization", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--pack-textures", action="store_true")
    parser.add_argument("--include-rejected", action="store_true", help="Apply deltas even when accepted=false.")
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    return parser.parse_args(argv)


def matrix4(values: Any) -> Matrix:
    return Matrix([[float(v) for v in row] for row in values]).to_4x4()


def main() -> int:
    args = parse_args()
    report = json.loads(args.object_pose_optimization.read_text(encoding="utf-8"))
    bpy.ops.wm.open_mainfile(filepath=str(args.input))

    applied: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for item in report.get("objects") or []:
        name = str(item.get("name") or "")
        if not name:
            skipped.append({"name": name, "reason": "missing_name"})
            continue
        obj = bpy.data.objects.get(name)
        if obj is None:
            skipped.append({"name": name, "reason": "object_not_found"})
            continue
        if obj.type != "MESH":
            skipped.append({"name": name, "reason": f"not_mesh:{obj.type}"})
            continue
        if (not args.include_rejected) and item.get("accepted") is False:
            skipped.append({"name": name, "reason": "not_accepted"})
            continue
        if item.get("delta_transform_world") is None:
            skipped.append({"name": name, "reason": "missing_delta_transform_world"})
            continue
        delta = matrix4(item["delta_transform_world"])
        obj.matrix_world = delta @ obj.matrix_world
        obj["object_pose_optimization_report"] = str(args.object_pose_optimization)
        obj["object_pose_optimization_status"] = str(item.get("status"))
        obj["object_pose_optimization_mask_id"] = int(item.get("mask_id")) if item.get("mask_id") is not None else -1
        obj["object_pose_optimization_delta_translation"] = item.get("delta_translation") or [0.0, 0.0, 0.0]
        obj["object_pose_optimization_delta_yaw_degrees"] = float(item.get("delta_yaw_degrees") or 0.0)
        obj["object_pose_optimization_delta_scale"] = float(item.get("delta_scale") or 0.0)
        obj["object_pose_optimization_scale"] = float(item.get("scale") or 1.0)
        applied.append(
            {
                "name": name,
                "mask_id": item.get("mask_id"),
                "status": item.get("status"),
                "delta_translation": item.get("delta_translation"),
                "delta_yaw_degrees": item.get("delta_yaw_degrees"),
                "delta_scale": item.get("delta_scale"),
                "scale": item.get("scale"),
            }
        )

    if args.pack_textures:
        try:
            bpy.ops.file.pack_all()
        except Exception as exc:  # pragma: no cover - Blender runtime only
            print(f"WARNING: failed to pack textures: {exc}", file=sys.stderr)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output))
    summary = {
        "schema": "fysiverse_blender_apply_object_pose_optimization.v1",
        "status": "ok",
        "input": str(args.input),
        "output": str(args.output),
        "object_pose_optimization": str(args.object_pose_optimization),
        "applied": applied,
        "skipped": skipped,
    }
    summary_path = args.output.with_suffix(".object_pose_apply_report.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Applied object pose optimization to {len(applied)} object(s)")
    print(f"Saved optimized blend: {args.output}")
    print(f"Wrote apply report: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
