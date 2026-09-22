#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
from pathlib import Path
from typing import Any


OBJECT_NAME_RE = re.compile(r"object[-_. ]*(\d+)", re.IGNORECASE)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is missing: {resolved}")
    return resolved


def require_dir(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise NotADirectoryError(f"{label} is missing: {resolved}")
    return resolved


def reset_dir(path: Path, *, overwrite: bool) -> None:
    resolved = path.expanduser().resolve()
    if resolved in {Path("/"), Path.home().resolve()} or len(resolved.parts) < 4:
        raise ValueError(f"refusing to reset broad path: {resolved}")
    if resolved.exists():
        if not overwrite:
            raise FileExistsError(f"output exists; pass --overwrite: {resolved}")
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True)


def materialize_file(source: Path, destination: Path, mode: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        try:
            os.link(source, destination)
            return "hardlink"
        except OSError:
            pass
    shutil.copy2(source, destination)
    return "copy"


def parse_glb_json(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        header = stream.read(12)
        if len(header) != 12:
            raise ValueError(f"truncated GLB header: {path}")
        magic, version, _total_length = struct.unpack("<4sII", header)
        if magic != b"glTF" or version != 2:
            raise ValueError(f"expected GLB v2: {path}")
        chunk_header = stream.read(8)
        if len(chunk_header) != 8:
            raise ValueError(f"missing GLB JSON chunk: {path}")
        chunk_length, chunk_type = struct.unpack("<II", chunk_header)
        if chunk_type != 0x4E4F534A:
            raise ValueError(f"first GLB chunk is not JSON: {path}")
        raw = stream.read(chunk_length).rstrip(b" \t\r\n\x00")
    return json.loads(raw.decode("utf-8"))


def validate_glb_object_contract(path: Path, mask_indices: list[int]) -> dict[str, Any]:
    gltf = parse_glb_json(path)
    mesh_nodes = [
        {"node_index": index, "name": str(node.get("name") or ""), "mesh": int(node["mesh"])}
        for index, node in enumerate(gltf.get("nodes") or [])
        if isinstance(node, dict) and node.get("mesh") is not None
    ]
    parsed: list[dict[str, Any]] = []
    for node in mesh_nodes:
        match = OBJECT_NAME_RE.search(node["name"])
        if match is None:
            raise ValueError(
                "post-refine requires every mesh node name to contain object_<index>; "
                f"unmatched node={node['name']!r} in {path}"
            )
        parsed.append({**node, "object_index": int(match.group(1))})

    expected = mask_indices
    actual = sorted(item["object_index"] for item in parsed)
    if actual != expected:
        raise ValueError(
            "GLB mesh-node/object mapping does not match numbered masks: "
            f"expected={expected}, actual={actual}, glb={path}"
        )
    return {"mesh_node_count": len(parsed), "mesh_nodes": sorted(parsed, key=lambda item: item["object_index"])}


def validate_camera(path: Path, *, allow_preview_camera: bool) -> dict[str, Any]:
    payload = read_json(path)
    camera = payload.get("camera")
    if not isinstance(camera, dict):
        raise ValueError(f"camera.json has no camera object: {path}")
    c2w = camera.get("camera_to_world")
    intrinsic = camera.get("intrinsic")
    if not (
        isinstance(c2w, list)
        and len(c2w) == 4
        and all(isinstance(row, list) and len(row) == 4 for row in c2w)
    ):
        raise ValueError(f"camera_to_world must be 4x4: {path}")
    if not (
        isinstance(intrinsic, list)
        and len(intrinsic) == 3
        and all(isinstance(row, list) and len(row) == 3 for row in intrinsic)
    ):
        raise ValueError(f"intrinsic must be 3x3: {path}")
    if int(camera.get("width") or 0) <= 0 or int(camera.get("height") or 0) <= 0:
        raise ValueError(f"camera width/height must be positive: {path}")

    source = str(payload.get("source") or "")
    if source == "generated_preview" and not allow_preview_camera:
        raise ValueError(
            "generated_preview camera is not aligned with the input photograph; "
            "pass --allow-preview-camera only for an operational, non-accuracy test"
        )
    coordinate = payload.get("coordinate_system") or {}
    up_axis = str(coordinate.get("scene_up_axis") or "").lower().replace(" ", "")
    if up_axis not in {"+y", "y", "y-up", "yup"}:
        raise ValueError(
            "The refinement runner applies a fixed Y-up to Z-up conversion; "
            f"camera scene_up_axis must be +Y, got {up_axis!r}"
        )
    return {
        "source": source,
        "is_input_camera_ground_truth": payload.get("is_input_camera_ground_truth"),
        "scene_up_axis": coordinate.get("scene_up_axis"),
        "width": int(camera["width"]),
        "height": int(camera["height"]),
    }


def numbered_masks(mask_dir: Path) -> list[Path]:
    masks = sorted(
        (path for path in mask_dir.glob("*.png") if path.stem.isdigit()),
        key=lambda path: int(path.stem),
    )
    if not masks:
        raise RuntimeError(f"no numbered PNG masks found in {mask_dir}")
    indices = [int(path.stem) for path in masks]
    if indices != list(range(len(indices))):
        raise ValueError(f"mask names must be contiguous and zero-based, got {indices}")
    return masks


def resolve_prepare_inputs(args: argparse.Namespace) -> dict[str, Any]:
    manifest: dict[str, Any] | None = None
    manifest_path: Path | None = None
    if args.pipeline_manifest:
        manifest_path = require_file(args.pipeline_manifest, "pipeline manifest")
        manifest = read_json(manifest_path)

    def selected(
        explicit: Path | None,
        manifest_keys: tuple[str, ...],
        label: str,
        *,
        directory: bool = False,
    ) -> Path:
        raw: Path | None = explicit
        if raw is None and manifest is not None:
            for manifest_key in manifest_keys:
                value = manifest.get(manifest_key)
                if value:
                    raw = Path(str(value))
                    break
        if raw is None:
            raise ValueError(f"{label} is required")
        if not raw.is_absolute() and manifest_path is not None:
            raw = manifest_path.parent / raw
        return require_dir(raw, label) if directory else require_file(raw, label)

    if manifest is not None:
        outputs = manifest.get("outputs") or {}
        objects = manifest.get("objects") or []
        # Public manifests store paths relative to the manifest and list each
        # object mask; Magic's legacy manifest stores flattened path keys.
        if args.scene_glb is None and outputs.get("scene"):
            args.scene_glb = Path(str(outputs["scene"]))
        if args.image is None and manifest.get("image"):
            args.image = Path(str(manifest["image"]))
        if args.mask_dir is None and objects:
            mask_values = [item.get("mask") for item in objects if isinstance(item, dict)]
            if mask_values:
                first = Path(str(mask_values[0]))
                args.mask_dir = first.parent

    return {
        "pipeline_manifest": manifest_path,
        "pipeline_data": manifest,
        "scene_glb": selected(args.scene_glb, ("final_glb",), "scene GLB"),
        "image": selected(args.image, ("image_path", "image"), "input image"),
        "mask_dir": selected(args.mask_dir, ("mask_dir",), "mask directory", directory=True),
        "camera_json": require_file(args.camera_json, "camera JSON"),
        "stage_json": require_file(args.stage_json, "stage JSON") if args.stage_json else None,
    }


def prepare(args: argparse.Namespace) -> int:
    inputs = resolve_prepare_inputs(args)
    case_dir = args.case_dir.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    reset_dir(case_dir, overwrite=args.overwrite)
    reset_dir(run_dir, overwrite=args.overwrite)

    masks = numbered_masks(inputs["mask_dir"])
    mask_indices = [int(path.stem) for path in masks]
    camera_summary = validate_camera(inputs["camera_json"], allow_preview_camera=args.allow_preview_camera)
    glb_summary = validate_glb_object_contract(inputs["scene_glb"], mask_indices)

    materialized: dict[str, Any] = {}
    materialized["scene_glb"] = materialize_file(inputs["scene_glb"], case_dir / "scene.glb", args.copy_mode)
    image_suffix = ".png" if inputs["image"].suffix.lower() == ".png" else ".jpg"
    image_name = f"input_rgb{image_suffix}"
    materialized["input_rgb"] = materialize_file(inputs["image"], case_dir / image_name, args.copy_mode)
    materialized["camera_json"] = materialize_file(inputs["camera_json"], case_dir / "camera.json", args.copy_mode)

    mask_modes = []
    for source in masks:
        mask_modes.append(materialize_file(source, case_dir / "masks" / source.name, args.copy_mode))
    materialized["masks"] = mask_modes

    if inputs["stage_json"] is not None:
        materialized["stage3_inputs"] = materialize_file(
            inputs["stage_json"], case_dir / "stage3_inputs.json", args.copy_mode
        )
    elif inputs["pipeline_data"] is not None:
        stage_objects = []
        by_index = {
            int(item.get("index", position)): item
            for position, item in enumerate(inputs["pipeline_data"].get("objects") or [])
            if isinstance(item, dict)
        }
        for index in mask_indices:
            item = by_index.get(index, {})
            stage_objects.append(
                {
                    "edit_index": index,
                    "object_index": index,
                    "image": f"{index}.png",
                    "category": str(item.get("category") or f"object_{index:03d}"),
                    "bbox": item.get("bbox"),
                }
            )
        write_json(
            case_dir / "stage3_inputs.json",
            {"scene_id": args.case_name, "objects": stage_objects},
        )
        materialized["stage3_inputs"] = "generated"

    if inputs["pipeline_manifest"] is not None:
        materialized["pipeline_manifest"] = materialize_file(
            inputs["pipeline_manifest"], case_dir / "pipeline_manifest.json", args.copy_mode
        )

    integration_manifest = {
        "schema": "fysicsmagic.optional_post_refine_input.v1",
        "status": "prepared",
        "case_name": args.case_name,
        "case_dir": str(case_dir),
        "run_dir": str(run_dir),
        "source": {
            "pipeline_manifest": str(inputs["pipeline_manifest"]) if inputs["pipeline_manifest"] else None,
            "scene_glb": str(inputs["scene_glb"]),
            "image": str(inputs["image"]),
            "mask_dir": str(inputs["mask_dir"]),
            "camera_json": str(inputs["camera_json"]),
            "stage_json": str(inputs["stage_json"]) if inputs["stage_json"] else None,
        },
        "materialized": materialized,
        "object_count": len(mask_indices),
        "mask_indices": mask_indices,
        "camera": camera_summary,
        "glb": glb_summary,
        "accuracy_scope": (
            "operational_only_preview_camera"
            if camera_summary["source"] == "generated_preview"
            else "input_view_camera_alignment"
        ),
    }
    manifest_path = case_dir / "integration_manifest.json"
    write_json(manifest_path, integration_manifest)
    print(json.dumps({"status": "prepared", "integration_manifest": str(manifest_path)}, ensure_ascii=False))
    return 0


def finalize(args: argparse.Namespace) -> int:
    integration_path = require_file(args.integration_manifest, "integration manifest")
    integration = read_json(integration_path)
    run_dir = require_dir(args.run_dir, "post-refine run directory")
    refined_glb = require_file(args.refined_glb, "refined GLB")
    export_report = require_file(args.export_report, "refined GLB export report")
    metrics_path = require_file(run_dir / "metrics_summary.json", "metrics summary")
    optimization_path = require_file(run_dir / "object_pose_optimization.json", "pose optimization report")
    import_report_path = require_file(run_dir / "import_report.json", "GLB import report")
    separation_report_path = require_file(run_dir / "separation_report.json", "separation report")

    metrics = read_json(metrics_path)
    optimization = read_json(optimization_path)
    import_report = read_json(import_report_path)
    expected_count = int(integration["object_count"])
    imported_count = int(import_report.get("object_count") or 0)
    if imported_count != expected_count:
        raise ValueError(
            f"imported mesh count does not match masks: expected={expected_count}, imported={imported_count}"
        )

    objects = [item for item in optimization.get("objects") or [] if isinstance(item, dict)]
    translation_limit = float((optimization.get("optimization") or {}).get("max_translation") or 0.0)
    scale_delta_limit = float((optimization.get("optimization") or {}).get("max_scale_delta") or 0.0)
    translation_bound_hits = 0
    scale_bound_hits = 0
    for item in objects:
        translation = [abs(float(value)) for value in item.get("delta_translation") or []]
        if translation_limit > 0 and any(value >= translation_limit - 1.0e-6 for value in translation):
            translation_bound_hits += 1
        scale = float(item.get("scale") or 1.0)
        if scale_delta_limit > 0 and abs(scale - 1.0) >= scale_delta_limit - 1.0e-6:
            scale_bound_hits += 1

    before = metrics.get("before") or {}
    after = metrics.get("after") or {}
    before_pen = before.get("penetration") or {}
    after_pen = after.get("penetration") or {}
    summary = {
        "schema": "fysicsmagic.optional_post_refine_summary.v1",
        "status": "ok",
        "case_name": integration["case_name"],
        "accuracy_scope": integration.get("accuracy_scope"),
        "object_count": expected_count,
        "camera": integration.get("camera"),
        "outputs": {
            "refined_glb": str(refined_glb),
            "run_dir": str(run_dir),
            "integration_manifest": str(integration_path),
            "metrics_summary": str(metrics_path),
            "pose_optimization": str(optimization_path),
            "separation_report": str(separation_report_path),
            "export_report": str(export_report),
            "before_rgb": str(run_dir / "metrics_before" / "rgb.png"),
            "after_rgb": str(run_dir / "metrics_after" / "rgb.png"),
        },
        "metrics": {
            "before_miou": before.get("mIoU"),
            "after_miou": after.get("mIoU"),
            "miou_delta": (
                float(after["mIoU"]) - float(before["mIoU"])
                if before.get("mIoU") is not None and after.get("mIoU") is not None
                else None
            ),
            "before_r_pen": before_pen.get("r_pen"),
            "after_r_pen": after_pen.get("r_pen"),
            "before_penetrating_triangle_pairs": before_pen.get("inter_object_penetrating_triangle_pairs"),
            "after_penetrating_triangle_pairs": after_pen.get("inter_object_penetrating_triangle_pairs"),
        },
        "optimization": {
            "accepted_objects": sum(item.get("accepted") is True for item in objects),
            "rejected_objects": sum(item.get("accepted") is False for item in objects),
            "translation_bound_hits": translation_bound_hits,
            "scale_bound_hits": scale_bound_hits,
        },
        "interpretation_limits": [
            "The after metric combines pose refinement, global ground translation, and convex separation.",
            "r_pen is an unbounded PAT3D-style normalized crossing count, not a percentage.",
            "The refined GLB remains a visual asset unless a separate USD/URDF physics export is run.",
        ],
    }
    write_json(args.output.expanduser().resolve(), summary)
    print(json.dumps({"status": "ok", "summary": str(args.output.expanduser().resolve())}, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and summarize Fysiverse visual post refinement.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prep = subparsers.add_parser("prepare")
    prep.add_argument("--case-name", required=True)
    prep.add_argument("--case-dir", required=True, type=Path)
    prep.add_argument("--run-dir", required=True, type=Path)
    prep.add_argument("--pipeline-manifest", type=Path)
    prep.add_argument("--scene-glb", type=Path)
    prep.add_argument("--image", type=Path)
    prep.add_argument("--mask-dir", type=Path)
    prep.add_argument("--camera-json", required=True, type=Path)
    prep.add_argument("--stage-json", type=Path)
    prep.add_argument("--copy-mode", choices=("hardlink", "copy"), default="hardlink")
    prep.add_argument("--allow-preview-camera", action="store_true")
    prep.add_argument("--overwrite", action="store_true")
    prep.set_defaults(func=prepare)

    fin = subparsers.add_parser("finalize")
    fin.add_argument("--integration-manifest", required=True, type=Path)
    fin.add_argument("--run-dir", required=True, type=Path)
    fin.add_argument("--refined-glb", required=True, type=Path)
    fin.add_argument("--export-report", required=True, type=Path)
    fin.add_argument("--output", required=True, type=Path)
    fin.set_defaults(func=finalize)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
