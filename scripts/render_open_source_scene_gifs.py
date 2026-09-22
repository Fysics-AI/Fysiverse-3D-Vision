#!/usr/bin/env python3
"""Render textured turntable frames for the public 3D-FUTURE examples."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-glb", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument(
        "--max-faces-per-object",
        type=int,
        default=100_000,
        help="Preview-only decimation limit; use 0 to keep the imported meshes unchanged.",
    )
    return parser.parse_args(argv)


def look_at(camera: bpy.types.Object, target: Vector) -> None:
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()


def scene_bounds(objects: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    corners = []
    for obj in objects:
        corners.extend(obj.matrix_world @ Vector(corner) for corner in obj.bound_box)
    if not corners:
        raise RuntimeError("GLB contains no mesh objects")
    minimum = Vector((min(point.x for point in corners), min(point.y for point in corners), min(point.z for point in corners)))
    maximum = Vector((max(point.x for point in corners), max(point.y for point in corners), max(point.z for point in corners)))
    return minimum, maximum


def main() -> int:
    args = parse_args()
    if args.frames < 4 or args.size < 64 or args.samples < 1:
        raise ValueError("frames must be >= 4, size must be >= 64, and samples must be >= 1")
    if args.max_faces_per_object < 0:
        raise ValueError("max-faces-per-object must be >= 0")
    output_dir = args.output_dir.resolve()
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for stale_frame in frames_dir.glob("frame_*.png"):
        stale_frame.unlink()

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(args.scene_glb.resolve()))
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    # README media is a lightweight capability preview. Decimation is applied
    # only to Blender's imported copy; the source GLB and its PBR materials are
    # never modified.
    if args.max_faces_per_object:
        for obj in meshes:
            face_count = len(obj.data.polygons)
            if face_count <= args.max_faces_per_object:
                continue
            bpy.ops.object.select_all(action="DESELECT")
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            modifier = obj.modifiers.new("README_preview_decimation", "DECIMATE")
            modifier.ratio = args.max_faces_per_object / face_count
            bpy.ops.object.modifier_apply(modifier=modifier.name)
            obj.select_set(False)

    minimum, maximum = scene_bounds(meshes)
    center = (minimum + maximum) * 0.5
    dimensions = maximum - minimum
    extent = max(dimensions.length, 1.0)
    horizontal_radius = max(math.hypot(dimensions.x, dimensions.y) * 0.5, 0.5)

    scene = bpy.context.scene
    # Eevee evaluates the imported glTF material nodes and embedded textures;
    # Workbench's MATERIAL mode shows viewport colors rather than PBR texture
    # maps and therefore produces gray previews for these assets.
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.image_settings.compression = 30
    scene.render.resolution_x = args.size
    scene.render.resolution_y = args.size
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.render.fps = 8
    scene.render.fps_base = 1.0
    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = args.samples
    elif hasattr(scene, "eevee_next"):
        scene.eevee_next.taa_render_samples = args.samples
    scene.view_settings.look = "AgX - Medium High Contrast"

    if scene.world is None:
        scene.world = bpy.data.worlds.new("TurntableWorld")
    scene.world.use_nodes = True
    background = scene.world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.055, 0.065, 0.08, 1.0)
    background.inputs["Strength"].default_value = 0.7

    camera_data = bpy.data.cameras.new("TurntableCamera")
    camera = bpy.data.objects.new("TurntableCamera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = max(horizontal_radius * 2.35, dimensions.z * 1.3, 1.0)
    camera.data.clip_start = 0.01
    camera.data.clip_end = extent * 20

    for name, location, energy, size in (
        ("Key", center + Vector((extent, -extent, extent * 1.6)), 1400.0, extent),
        ("Fill", center + Vector((-extent, -extent * 0.4, extent)), 900.0, extent * 0.8),
        ("Rim", center + Vector((0.0, extent, extent * 1.4)), 1100.0, extent * 0.7),
    ):
        data = bpy.data.lights.new(name, type="AREA")
        data.energy = energy
        data.shape = "DISK"
        data.size = size
        lamp = bpy.data.objects.new(name, data)
        scene.collection.objects.link(lamp)
        lamp.location = location
        look_at(lamp, center)

    orbit_radius = max(extent * 1.8, 2.0)
    orbit_height = max(dimensions.z * 0.65, horizontal_radius * 0.75, 0.75)
    for frame in range(args.frames):
        angle = 2.0 * math.pi * frame / args.frames - math.pi * 0.25
        camera.location = center + Vector(
            (orbit_radius * math.cos(angle), orbit_radius * math.sin(angle), orbit_height)
        )
        look_at(camera, center)
        scene.render.filepath = str(frames_dir / f"frame_{frame:03d}.png")
        bpy.ops.render.render(write_still=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
