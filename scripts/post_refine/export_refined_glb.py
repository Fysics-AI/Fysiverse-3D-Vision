#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import bpy


MASK_OBJECT_RE = re.compile(r"^mask_(\d+)_object(?:[._].*)?$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a post-refined Blend scene as a Y-up GLB.")
    parser.add_argument("--input-blend", required=True, type=Path)
    parser.add_argument("--output-glb", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument(
        "--export-normals",
        action="store_true",
        help="Export explicit vertex normals. Disabled by default to avoid flat-normal vertex expansion.",
    )
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    input_blend = args.input_blend.resolve()
    output_glb = args.output_glb.resolve()
    report_path = args.report.resolve()
    bpy.ops.wm.open_mainfile(filepath=str(input_blend))

    objects = sorted(
        (
            obj
            for obj in bpy.context.scene.objects
            if obj.type == "MESH" and MASK_OBJECT_RE.match(obj.name) is not None and not obj.hide_render
        ),
        key=lambda obj: int(MASK_OBJECT_RE.match(obj.name).group(1)),
    )
    if not objects:
        raise RuntimeError(f"no visible mask_NNN_object meshes found in {input_blend}")
    for obj in bpy.context.scene.objects:
        obj.select_set(obj in objects)

    output_glb.parent.mkdir(parents=True, exist_ok=True)
    result = bpy.ops.export_scene.gltf(
        filepath=str(output_glb),
        export_format="GLB",
        use_selection=True,
        export_yup=True,
        export_apply=False,
        export_animations=False,
        export_cameras=False,
        export_lights=False,
        export_extras=True,
        export_normals=args.export_normals,
    )
    if "FINISHED" not in result or not output_glb.is_file() or output_glb.stat().st_size <= 0:
        raise RuntimeError(f"GLB export failed: result={result}, output={output_glb}")

    payload = {
        "schema": "fysicsmagic.optional_post_refine_glb_export.v1",
        "status": "ok",
        "input_blend": str(input_blend),
        "output_glb": str(output_glb),
        "output_bytes": int(output_glb.stat().st_size),
        "coordinate_system": "glTF Y-up",
        "export_normals": args.export_normals,
        "object_count": len(objects),
        "objects": [
            {
                "name": obj.name,
                "mask_id": int(MASK_OBJECT_RE.match(obj.name).group(1)),
            }
            for obj in objects
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
