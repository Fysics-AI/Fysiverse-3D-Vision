#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import bpy
from mathutils import Matrix, Vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add the original input-view camera to a SAM3D/MoGe Blender scene."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None, help="Output blend. Defaults to overwriting --input.")
    parser.add_argument("--camera-report", required=True, type=Path, help="Rotated-stage plan/report containing MoGe intrinsics and upright matrix.")
    parser.add_argument("--stage-report", type=Path, default=None, help="Optional current-stage report; used to carry global grounding translation.")
    parser.add_argument("--camera-optimization", type=Path, default=None, help="Optional optimized camera report from optimize_camera_pose_nvdiffrast.py.")
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--camera-name", default="OriginalInputCamera")
    parser.add_argument("--set-active", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--set-render-resolution", action=argparse.BooleanOptionalAction, default=True)
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    return parser.parse_args(argv)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def matrix3_from_list(values: Any, fallback: Matrix | None = None) -> Matrix:
    if values is None:
        if fallback is None:
            raise ValueError("missing 3x3 matrix")
        return fallback.copy()
    return Matrix([[float(v) for v in row] for row in values]).to_3x3()


def matrix4_from_list(values: Any, fallback: Matrix | None = None) -> Matrix:
    if values is None:
        if fallback is None:
            raise ValueError("missing 4x4 matrix")
        return fallback.copy()
    return Matrix([[float(v) for v in row] for row in values]).to_4x4()


def vector3(values: Any, fallback: tuple[float, float, float] | None = None) -> Vector:
    if values is None:
        if fallback is None:
            raise ValueError("missing 3-vector")
        values = fallback
    return Vector((float(values[0]), float(values[1]), float(values[2])))


def as_json(value: Any) -> Any:
    if isinstance(value, Matrix):
        return [[float(v) for v in row] for row in value]
    if isinstance(value, Vector):
        return [float(v) for v in value]
    if isinstance(value, dict):
        return {str(k): as_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_json(v) for v in value]
    return value


def load_image_size(report: dict[str, Any]) -> tuple[int, int]:
    moge = report.get("moge") or {}
    sample = moge.get("sample_info") or {}
    size = sample.get("image_size")
    if isinstance(size, list) and len(size) >= 2:
        return int(size[0]), int(size[1])

    image_path = ((report.get("input") or {}).get("image")) or ((report.get("stage") or {}).get("input") or {}).get("image")
    if image_path:
        image = bpy.data.images.load(str(image_path), check_existing=True)
        return int(image.size[0]), int(image.size[1])
    raise ValueError("Cannot determine input image size from camera report")


def normalized_intrinsics(report: dict[str, Any]) -> tuple[float, float, float, float]:
    intrinsics = ((report.get("moge") or {}).get("intrinsics"))
    if intrinsics is None:
        raise ValueError("camera report does not contain moge.intrinsics")
    fx = float(intrinsics[0][0])
    fy = float(intrinsics[1][1])
    cx = float(intrinsics[0][2])
    cy = float(intrinsics[1][2])
    return fx, fy, cx, cy


def raw_camera_to_blender_matrix(report: dict[str, Any]) -> Matrix:
    moge = report.get("moge") or {}
    scene_to_blender = matrix3_from_list(moge.get("scene_to_blender_matrix"), Matrix.Identity(3))
    cv_to_scene = matrix3_from_list(
        moge.get("opencv_camera_to_pytorch3d_scene_matrix"),
        Matrix(((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0))),
    )

    # OpenCV camera: +X image right, +Y image down, +Z forward.
    # Blender camera local axes: +X image right, +Y image up, -Z forward.
    right_scene = cv_to_scene @ Vector((1.0, 0.0, 0.0))
    up_scene = cv_to_scene @ Vector((0.0, -1.0, 0.0))
    back_scene = cv_to_scene @ Vector((0.0, 0.0, -1.0))

    right = scene_to_blender @ right_scene
    up = scene_to_blender @ up_scene
    back = scene_to_blender @ back_scene
    right.normalize()
    up.normalize()
    back.normalize()

    return Matrix(
        (
            (right.x, up.x, back.x, 0.0),
            (right.y, up.y, back.y, 0.0),
            (right.z, up.z, back.z, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    )


def stage_global_translation(stage_report: dict[str, Any] | None) -> Vector:
    if not stage_report:
        return Vector((0.0, 0.0, 0.0))
    transform = stage_report.get("transform") or {}
    bbox = transform.get("bbox_snap_objects")
    if isinstance(bbox, dict) and bbox.get("global_translation") is not None:
        return vector3(bbox.get("global_translation"))
    return Vector((0.0, 0.0, 0.0))


def configure_camera_data(cam: bpy.types.Object, report: dict[str, Any], image_size: tuple[int, int]) -> dict[str, Any]:
    fx, fy, cx, cy = normalized_intrinsics(report)
    width, height = image_size
    fx_px = fx * float(width)
    fy_px = fy * float(height)
    cx_px = cx * float(width)
    cy_px = cy * float(height)

    cam.data.type = "PERSP"
    cam.data.sensor_fit = "HORIZONTAL"
    cam.data.sensor_width = 36.0
    cam.data.lens = fx * cam.data.sensor_width
    cam.data.clip_start = 0.001
    cam.data.clip_end = 10000.0
    cam.data.shift_x = 0.5 - cx
    cam.data.shift_y = cy - 0.5

    angle_x = 2.0 * math.atan(0.5 / max(fx, 1e-12))
    angle_y = 2.0 * math.atan(0.5 / max(fy, 1e-12))
    return {
        "intrinsics_normalized": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        "intrinsics_pixels": [[fx_px, 0.0, cx_px], [0.0, fy_px, cy_px], [0.0, 0.0, 1.0]],
        "image_size": [width, height],
        "fov_degrees": {"x": math.degrees(angle_x), "y": math.degrees(angle_y)},
        "blender_camera": {
            "sensor_fit": cam.data.sensor_fit,
            "sensor_width": float(cam.data.sensor_width),
            "lens": float(cam.data.lens),
            "shift_x": float(cam.data.shift_x),
            "shift_y": float(cam.data.shift_y),
            "clip_start": float(cam.data.clip_start),
            "clip_end": float(cam.data.clip_end),
        },
    }


def configure_camera_data_from_normalized(cam: bpy.types.Object, intrinsics: Any, image_size: Any) -> dict[str, Any]:
    fx = float(intrinsics[0][0])
    fy = float(intrinsics[1][1])
    cx = float(intrinsics[0][2])
    cy = float(intrinsics[1][2])
    width, height = int(image_size[0]), int(image_size[1])
    cam.data.type = "PERSP"
    cam.data.sensor_fit = "HORIZONTAL"
    cam.data.sensor_width = 36.0
    cam.data.lens = fx * cam.data.sensor_width
    cam.data.clip_start = 0.001
    cam.data.clip_end = 10000.0
    cam.data.shift_x = 0.5 - cx
    cam.data.shift_y = cy - 0.5
    return {
        "intrinsics_normalized": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        "intrinsics_pixels": [
            [fx * float(width), 0.0, cx * float(width)],
            [0.0, fy * float(height), cy * float(height)],
            [0.0, 0.0, 1.0],
        ],
        "image_size": [width, height],
        "fov_degrees": {
            "x": math.degrees(2.0 * math.atan(0.5 / max(fx, 1e-12))),
            "y": math.degrees(2.0 * math.atan(0.5 / max(fy, 1e-12))),
        },
        "blender_camera": {
            "sensor_fit": cam.data.sensor_fit,
            "sensor_width": float(cam.data.sensor_width),
            "lens": float(cam.data.lens),
            "shift_x": float(cam.data.shift_x),
            "shift_y": float(cam.data.shift_y),
            "clip_start": float(cam.data.clip_start),
            "clip_end": float(cam.data.clip_end),
        },
    }


def main() -> int:
    args = parse_args()
    output_blend = args.output or args.input
    camera_report = read_json(args.camera_report)
    stage_report = read_json(args.stage_report) if args.stage_report and args.stage_report.is_file() else None

    bpy.ops.wm.open_mainfile(filepath=str(args.input))

    camera_optimization = read_json(args.camera_optimization) if args.camera_optimization and args.camera_optimization.is_file() else None
    if camera_optimization is not None:
        camera_to_world = matrix4_from_list(camera_optimization.get("optimized_camera_to_world"))
        extra_translation = Vector((0.0, 0.0, 0.0))
        camera_source = "nvdiffrast_optimized"
    else:
        raw_c2w = raw_camera_to_blender_matrix(camera_report)
        upright = matrix4_from_list((camera_report.get("transform") or {}).get("matrix_4x4"), Matrix.Identity(4))
        extra_translation = stage_global_translation(stage_report)
        extra = Matrix.Translation(extra_translation)
        camera_to_world = extra @ upright @ raw_c2w
        camera_source = "moge_reconstructed"

    cam_data = bpy.data.cameras.get(args.camera_name)
    if cam_data is None:
        cam_data = bpy.data.cameras.new(args.camera_name)
    cam = bpy.data.objects.get(args.camera_name)
    if cam is None:
        cam = bpy.data.objects.new(args.camera_name, cam_data)
        bpy.context.scene.collection.objects.link(cam)
    else:
        cam.data = cam_data
    cam.matrix_world = camera_to_world

    if camera_optimization is not None:
        image_size = [int(v) for v in camera_optimization.get("source_image_size")]
        camera_settings = configure_camera_data_from_normalized(cam, camera_optimization.get("intrinsics_normalized"), image_size)
        cam["camera_optimization_report"] = str(args.camera_optimization)
    else:
        image_size = load_image_size(camera_report)
        camera_settings = configure_camera_data(cam, camera_report, image_size)
    if args.set_render_resolution:
        bpy.context.scene.render.resolution_x = int(image_size[0])
        bpy.context.scene.render.resolution_y = int(image_size[1])
        bpy.context.scene.render.resolution_percentage = 100
    if args.set_active:
        bpy.context.scene.camera = cam

    output_blend.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output_blend))

    world_to_camera = camera_to_world.inverted()
    payload = {
        "schema": "fysiverse_original_input_camera.v1",
        "status": "ok",
        "camera_name": args.camera_name,
        "blend": str(output_blend),
        "camera_report": str(args.camera_report),
        "stage_report": str(args.stage_report) if args.stage_report else None,
        "camera_optimization": str(args.camera_optimization) if args.camera_optimization else None,
        "camera_source": camera_source,
        "coordinate_notes": {
            "raw_sam3d_camera": "Camera center is the origin of the SAM3D/MoGe pointmap scene.",
            "opencv_camera_axes": "+X image right, +Y image down, +Z forward.",
            "blender_camera_axes": "+X image right, +Y image up, -Z forward.",
            "camera_to_world": "Blender camera local-to-world matrix after SAM3D scene-to-Blender conversion and MoGe upright transform.",
            "extra_stage_translation": "Applied only when the current stage report contains a shared global grounding translation.",
        },
        "camera_to_world": camera_to_world,
        "world_to_camera": world_to_camera,
        "location": cam.matrix_world.translation,
        "rotation_euler_xyz": [float(v) for v in cam.rotation_euler],
        "extra_stage_translation": extra_translation,
        "optimization_metrics": camera_optimization.get("metrics") if camera_optimization is not None else None,
        **camera_settings,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(as_json(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Added camera {args.camera_name} to {output_blend}")
    print(f"Wrote camera metadata: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
