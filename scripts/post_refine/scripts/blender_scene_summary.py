#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    bpy.ops.wm.open_mainfile(filepath=str(args.input))
    depsgraph = bpy.context.evaluated_depsgraph_get()
    objects = []
    global_min = [float("inf"), float("inf"), float("inf")]
    global_max = [float("-inf"), float("-inf"), float("-inf")]
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH" or obj.name.startswith("__"):
            continue
        eval_obj = obj.evaluated_get(depsgraph)
        mesh = eval_obj.to_mesh()
        try:
            if not mesh.vertices:
                continue
            mins = [float("inf"), float("inf"), float("inf")]
            maxs = [float("-inf"), float("-inf"), float("-inf")]
            for vertex in mesh.vertices:
                p = eval_obj.matrix_world @ vertex.co
                vals = [float(p.x), float(p.y), float(p.z)]
                for axis in range(3):
                    mins[axis] = min(mins[axis], vals[axis])
                    maxs[axis] = max(maxs[axis], vals[axis])
                    global_min[axis] = min(global_min[axis], vals[axis])
                    global_max[axis] = max(global_max[axis], vals[axis])
            objects.append({"name": obj.name, "bbox_min": mins, "bbox_max": maxs})
        finally:
            eval_obj.to_mesh_clear()
    payload = {
        "input": str(args.input),
        "object_count": len(objects),
        "bbox_min": global_min,
        "bbox_max": global_max,
        "min_z": global_min[2],
        "objects": objects,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
