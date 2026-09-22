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


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import blender_add_original_camera as camera_utils  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a Blender animation from the estimated original input camera."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--camera-report", type=Path, default=None)
    parser.add_argument("--stage-report", type=Path, default=None)
    parser.add_argument("--camera-optimization", type=Path, default=None, help="Optional optimized camera report from optimize_camera_pose_nvdiffrast.py.")
    parser.add_argument("--camera-name", default="OriginalInputCamera")
    parser.add_argument("--camera-blend-output", type=Path, default=None)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means render the full selected frame range.")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=0, help="<=0 preserves the original input image aspect ratio from MoGe intrinsics metadata.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--compute-backend", default="CUDA")
    parser.add_argument("--world-strength", type=float, default=0.05)
    parser.add_argument("--key-light-energy", type=float, default=220.0)
    parser.add_argument("--fill-light-energy", type=float, default=25.0)
    parser.add_argument("--render-still", action="store_true")
    parser.add_argument("--transparent-background", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    return parser.parse_args(argv)


def as_json(value: Any) -> Any:
    return camera_utils.as_json(value)


def configure_gpu(compute_backend: str) -> list[dict[str, Any]]:
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"

    prefs = bpy.context.preferences.addons["cycles"].preferences
    if hasattr(prefs, "compute_device_type"):
        prefs.compute_device_type = compute_backend
    if hasattr(prefs, "get_devices"):
        prefs.get_devices()

    devices: list[dict[str, Any]] = []
    for device in getattr(prefs, "devices", []):
        dtype = getattr(device, "type", "")
        device.use = dtype != "CPU"
        devices.append(
            {
                "name": getattr(device, "name", ""),
                "type": dtype,
                "use": bool(getattr(device, "use", False)),
            }
        )
    return devices


def configure_world(strength: float, transparent: bool) -> None:
    scene = bpy.context.scene
    world = scene.world or bpy.data.worlds.new("original_camera_render_world")
    scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    background = nodes.new(type="ShaderNodeBackground")
    background.inputs["Color"].default_value = (0.78, 0.80, 0.84, 1.0)
    background.inputs["Strength"].default_value = float(strength)
    output = nodes.new(type="ShaderNodeOutputWorld")
    links.new(background.outputs["Background"], output.inputs["Surface"])
    scene.render.film_transparent = bool(transparent)


def visible_meshes() -> list[bpy.types.Object]:
    objects: list[bpy.types.Object] = []
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        if obj.hide_render or obj.name.startswith("__render_"):
            continue
        objects.append(obj)
    return objects


def collect_bbox(start: int, end: int, max_samples: int = 12) -> tuple[Vector, Vector]:
    scene = bpy.context.scene
    objects = visible_meshes()
    if not objects:
        return Vector((-1.0, -1.0, -1.0)), Vector((1.0, 1.0, 1.0))

    frame_count = max(1, int(end) - int(start) + 1)
    stride = max(1, int(math.ceil(frame_count / max(1, max_samples))))
    frames = list(range(int(start), int(end) + 1, stride))
    if frames[-1] != int(end):
        frames.append(int(end))

    bmin = Vector((math.inf, math.inf, math.inf))
    bmax = Vector((-math.inf, -math.inf, -math.inf))
    found = False
    for frame in frames:
        scene.frame_set(frame)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        depsgraph.update()
        for obj in objects:
            eval_obj = obj.evaluated_get(depsgraph)
            for corner in eval_obj.bound_box:
                point = eval_obj.matrix_world @ Vector(corner)
                bmin.x = min(bmin.x, point.x)
                bmin.y = min(bmin.y, point.y)
                bmin.z = min(bmin.z, point.z)
                bmax.x = max(bmax.x, point.x)
                bmax.y = max(bmax.y, point.y)
                bmax.z = max(bmax.z, point.z)
                found = True
    if not found:
        return Vector((-1.0, -1.0, -1.0)), Vector((1.0, 1.0, 1.0))
    return bmin, bmax


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    direction = target - obj.location
    if direction.length < 1e-8:
        return
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def ensure_light(name: str, light_type: str) -> bpy.types.Object:
    data = bpy.data.lights.get(name)
    if data is None:
        data = bpy.data.lights.new(name, type=light_type)
    obj = bpy.data.objects.get(name)
    if obj is None:
        obj = bpy.data.objects.new(name, data)
        bpy.context.collection.objects.link(obj)
    else:
        obj.data = data
    return obj


def add_camera_relative_lights(
    camera: bpy.types.Object,
    bmin: Vector,
    bmax: Vector,
    *,
    key_energy: float,
    fill_energy: float,
) -> None:
    center = (bmin + bmax) * 0.5
    dims = bmax - bmin
    scale = max(float(dims.length), 1.0)
    cam_quat = camera.matrix_world.to_quaternion()
    cam_right = cam_quat @ Vector((1.0, 0.0, 0.0))
    cam_up = cam_quat @ Vector((0.0, 1.0, 0.0))
    cam_back = cam_quat @ Vector((0.0, 0.0, 1.0))

    key = ensure_light("__original_view_key_area", "AREA")
    key.location = center + cam_back * (0.8 * scale) + cam_right * (0.55 * scale) + cam_up * (0.85 * scale)
    key.data.energy = float(key_energy)
    key.data.size = max(2.5, 1.7 * scale)
    look_at(key, center)

    fill = ensure_light("__original_view_fill_area", "AREA")
    fill.location = center + cam_back * (0.65 * scale) - cam_right * (0.75 * scale) + cam_up * (0.35 * scale)
    fill.data.energy = float(fill_energy)
    fill.data.size = max(3.0, 2.3 * scale)
    look_at(fill, center)


def add_original_input_camera(
    *,
    camera_report_path: Path,
    stage_report_path: Path | None,
    camera_name: str,
) -> dict[str, Any]:
    camera_report = camera_utils.read_json(camera_report_path)
    stage_report = camera_utils.read_json(stage_report_path) if stage_report_path and stage_report_path.is_file() else None

    raw_c2w = camera_utils.raw_camera_to_blender_matrix(camera_report)
    upright = camera_utils.matrix4_from_list((camera_report.get("transform") or {}).get("matrix_4x4"), Matrix.Identity(4))
    extra_translation = camera_utils.stage_global_translation(stage_report)
    camera_to_world = Matrix.Translation(extra_translation) @ upright @ raw_c2w

    cam_data = bpy.data.cameras.get(camera_name)
    if cam_data is None:
        cam_data = bpy.data.cameras.new(camera_name)
    cam = bpy.data.objects.get(camera_name)
    if cam is None:
        cam = bpy.data.objects.new(camera_name, cam_data)
        bpy.context.scene.collection.objects.link(cam)
    else:
        cam.data = cam_data
    cam.matrix_world = camera_to_world

    image_size = camera_utils.load_image_size(camera_report)
    camera_settings = camera_utils.configure_camera_data(cam, camera_report, image_size)
    bpy.context.scene.camera = cam

    return {
        "camera_name": camera_name,
        "camera_report": str(camera_report_path),
        "stage_report": str(stage_report_path) if stage_report_path else None,
        "camera_to_world": camera_to_world,
        "world_to_camera": camera_to_world.inverted(),
        "location": cam.matrix_world.translation,
        "rotation_euler_xyz": [float(v) for v in cam.rotation_euler],
        "extra_stage_translation": extra_translation,
        **camera_settings,
    }


def matrix4_from_json(values: Any) -> Matrix:
    return Matrix([[float(v) for v in row] for row in values]).to_4x4()


def configure_camera_from_normalized(
    cam: bpy.types.Object,
    intrinsics: list[list[float]],
    image_size: list[int],
) -> dict[str, Any]:
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


def add_optimized_input_camera(
    *,
    optimization_path: Path,
    camera_name: str,
) -> dict[str, Any]:
    report = camera_utils.read_json(optimization_path)
    camera_to_world = matrix4_from_json(report["optimized_camera_to_world"])
    cam_data = bpy.data.cameras.get(camera_name)
    if cam_data is None:
        cam_data = bpy.data.cameras.new(camera_name)
    cam = bpy.data.objects.get(camera_name)
    if cam is None:
        cam = bpy.data.objects.new(camera_name, cam_data)
        bpy.context.scene.collection.objects.link(cam)
    else:
        cam.data = cam_data
    cam.matrix_world = camera_to_world
    camera_settings = configure_camera_from_normalized(cam, report["intrinsics_normalized"], report["source_image_size"])
    cam["camera_optimization_report"] = str(optimization_path)
    bpy.context.scene.camera = cam
    return {
        "camera_name": camera_name,
        "camera_optimization": str(optimization_path),
        "camera_to_world": camera_to_world,
        "world_to_camera": camera_to_world.inverted(),
        "location": cam.matrix_world.translation,
        "rotation_euler_xyz": [float(v) for v in cam.rotation_euler],
        "optimization_metrics": report.get("metrics"),
        **camera_settings,
    }


def configure_render(args: argparse.Namespace, start: int, end: int) -> None:
    scene = bpy.context.scene
    scene.frame_start = int(start)
    scene.frame_end = int(end)
    scene.frame_set(int(start))
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    scene.cycles.samples = int(args.samples)
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 8
    scene.cycles.diffuse_bounces = 3
    scene.cycles.glossy_bounces = 3
    scene.cycles.transparent_max_bounces = 4

    scene.render.resolution_x = int(args.width)
    scene.render.resolution_y = int(args.height)
    scene.render.resolution_percentage = 100
    scene.render.fps = int(args.fps)

    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    if args.render_still:
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_mode = "RGBA"
        scene.render.image_settings.compression = 15
    else:
        scene.render.image_settings.file_format = "FFMPEG"
        scene.render.ffmpeg.format = "MPEG4"
        scene.render.ffmpeg.codec = "H264"
        scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
        scene.render.ffmpeg.ffmpeg_preset = "GOOD"
        scene.render.ffmpeg.audio_codec = "NONE"
    scene.render.filepath = str(args.output)


def validate_render_output(path: Path, *, render_still: bool) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Render output was not created: {path}")
    size = int(path.stat().st_size)
    if size <= 0:
        raise RuntimeError(f"Render output is empty: {path}")
    if render_still:
        return {"path": str(path), "bytes": size, "validated": True}

    # Blender/FFmpeg can leave a tiny MP4 with only ftyp/free/mdat when rendering
    # is interrupted before the moov atom is written. Treat that as a hard error.
    min_video_bytes = 4096
    if size < min_video_bytes:
        raise RuntimeError(f"Rendered video is too small to be valid ({size} bytes): {path}")
    with path.open("rb") as handle:
        head = handle.read(min(1024 * 1024, size))
        if size > 1024 * 1024:
            handle.seek(max(0, size - 1024 * 1024))
            tail = handle.read(1024 * 1024)
        else:
            tail = b""
    if b"moov" not in head and b"moov" not in tail:
        raise RuntimeError(f"Rendered MP4 is missing a moov atom and is likely incomplete: {path}")
    return {"path": str(path), "bytes": size, "validated": True}


def selected_frame_range(scene: bpy.types.Scene, args: argparse.Namespace) -> tuple[int, int]:
    start = int(args.start_frame if args.start_frame is not None else scene.frame_start or 1)
    end = int(args.end_frame if args.end_frame is not None else scene.frame_end or start)
    if end < start:
        end = start
    if int(args.max_frames) > 0:
        end = min(end, start + int(args.max_frames) - 1)
    return start, end


def even_dimension(value: float, minimum: int = 2) -> int:
    out = max(int(round(value)), int(minimum))
    if out % 2:
        out += 1
    return out


def main() -> int:
    args = parse_args()
    args.output = args.output.resolve()
    args.output_json = args.output_json.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.open_mainfile(filepath=str(args.input))
    scene = bpy.context.scene
    start, end = selected_frame_range(scene, args)

    devices = configure_gpu(args.compute_backend)
    configure_world(args.world_strength, args.transparent_background)
    if args.camera_optimization is not None:
        camera_payload = add_optimized_input_camera(
            optimization_path=args.camera_optimization.resolve(),
            camera_name=args.camera_name,
        )
    else:
        if args.camera_report is None:
            raise ValueError("--camera-report is required when --camera-optimization is not provided")
        camera_payload = add_original_input_camera(
            camera_report_path=args.camera_report.resolve(),
            stage_report_path=args.stage_report.resolve() if args.stage_report else None,
            camera_name=args.camera_name,
        )
    if int(args.height) <= 0:
        image_width, image_height = camera_payload["image_size"]
        args.height = even_dimension(float(args.width) * float(image_height) / max(float(image_width), 1.0))
    bmin, bmax = collect_bbox(start, end)
    add_camera_relative_lights(
        scene.camera,
        bmin,
        bmax,
        key_energy=args.key_light_energy,
        fill_energy=args.fill_light_energy,
    )
    configure_render(args, start, end)

    camera_blend = None
    if args.camera_blend_output:
        camera_blend = args.camera_blend_output.resolve()
        camera_blend.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(camera_blend))

    payload = {
        "schema": "fysiverse_original_camera_animation_render.v1",
        "status": "configured" if args.skip_render else "ok",
        "input": str(args.input.resolve()),
        "output": str(args.output),
        "camera_blend": str(camera_blend) if camera_blend else None,
        "frame_start": int(start),
        "frame_end": int(end),
        "width": int(args.width),
        "height": int(args.height),
        "fps": int(args.fps),
        "samples": int(args.samples),
        "compute_backend": args.compute_backend,
        "devices": devices,
        "world_strength": float(args.world_strength),
        "transparent_background": bool(args.transparent_background),
        "bbox_min": bmin,
        "bbox_max": bmax,
        "camera": camera_payload,
        "note": "Rendered from the optimized original input camera when --camera-optimization is provided; otherwise from the estimated SAM3D/MoGe input camera.",
    }

    print("Original-camera render configuration:", flush=True)
    print(f"  input={args.input}", flush=True)
    print(f"  output={args.output}", flush=True)
    print(f"  camera={scene.camera.name if scene.camera else None}", flush=True)
    print(f"  frames={start}-{end}", flush=True)
    print(f"  resolution={args.width}x{args.height}", flush=True)
    print(f"  samples={args.samples}", flush=True)
    print(f"  devices={devices}", flush=True)
    print(f"  bbox_min={tuple(round(v, 5) for v in bmin)}", flush=True)
    print(f"  bbox_max={tuple(round(v, 5) for v in bmax)}", flush=True)

    if not args.skip_render:
        if not any(d.get("use") and d.get("type") != "CPU" for d in devices):
            raise RuntimeError(f"No non-CPU Cycles device enabled: {devices}")
        if args.render_still:
            bpy.ops.render.render(write_still=True)
            payload["output_validation"] = validate_render_output(args.output, render_still=True)
            print(f"Wrote still: {args.output}", flush=True)
        else:
            bpy.ops.render.render(animation=True)
            payload["output_validation"] = validate_render_output(args.output, render_still=False)
            print(f"Wrote video: {args.output}", flush=True)
    else:
        payload["output_validation"] = {"path": str(args.output), "validated": False, "reason": "skip_render"}

    args.output_json.write_text(json.dumps(as_json(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote render report: {args.output_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
