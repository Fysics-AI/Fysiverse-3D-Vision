#!/usr/bin/env python3
"""Summarize a production Layout + optional post-refinement validation run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def yaw_degrees(rotation: Any) -> float:
    matrix = np.asarray(rotation, dtype=np.float64)
    return math.degrees(math.atan2(float(matrix[2, 0]), float(matrix[0, 0])))


def angle_error(left: float, right: float) -> float:
    delta = math.radians(left - right)
    return abs(math.degrees(math.atan2(math.sin(delta), math.cos(delta))))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_time_report(path: Path) -> dict[str, float | int]:
    text = path.read_text(encoding="utf-8")
    elapsed_match = re.search(r"Elapsed \(wall clock\) time .*?:\s*([0-9:.]+)", text)
    rss_match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
    if elapsed_match is None or rss_match is None:
        raise ValueError(f"cannot parse /usr/bin/time report: {path}")
    parts = [float(part) for part in elapsed_match.group(1).split(":")]
    elapsed = sum(value * (60 ** index) for index, value in enumerate(reversed(parts)))
    return {"wall_seconds": elapsed, "max_rss_bytes": int(rss_match.group(1)) * 1024}


def probe_video(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration,size",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def scene_stats(path: Path) -> dict[str, Any]:
    scene = trimesh.load(str(path), force="scene", process=False)
    if isinstance(scene, trimesh.Trimesh):
        scene = trimesh.Scene(scene)
    geometries = list(scene.geometry.values())
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "objects": len(geometries),
        "vertices": sum(len(mesh.vertices) for mesh in geometries),
        "faces": sum(len(mesh.faces) for mesh in geometries),
        "bounds": np.asarray(scene.bounds, dtype=np.float64).tolist(),
        "extents": np.asarray(scene.extents, dtype=np.float64).tolist(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "Matplotlib is required to generate the report figure. "
            "Run this script in an environment with compatible matplotlib and pyparsing packages."
        ) from exc

    root = args.result_root.resolve()
    reports = root / "reports"
    figures = root / "figures"
    reports.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    poses = load_json(reports / "poses.json")["objects"]
    gt_objects = sorted(load_json(args.ground_truth.resolve())["objects"], key=lambda item: item["edit_index"])
    pred_t = np.asarray([item["translation"] for item in poses], dtype=np.float64)
    pred_r = np.asarray([item["rotation"] for item in poses], dtype=np.float64)
    pred_s = np.asarray([item["scaling"] for item in poses], dtype=np.float64)
    gt_t = np.asarray([item["pose"]["translation"] for item in gt_objects], dtype=np.float64)
    gt_r = np.asarray([item["pose"]["rotation"] for item in gt_objects], dtype=np.float64)
    gt_s = np.asarray([item["pose"]["scaling"] for item in gt_objects], dtype=np.float64)
    translation_error = np.linalg.norm(pred_t - gt_t, axis=1)
    yaw_error = np.asarray(
        [angle_error(yaw_degrees(pred_r[index]), yaw_degrees(gt_r[index])) for index in range(len(poses))]
    )
    scale_error = np.abs(pred_s - gt_s)
    gt_metrics = {
        "translation_l2_mean": float(translation_error.mean()),
        "translation_l2_max": float(translation_error.max()),
        "yaw_error_degrees_mean": float(yaw_error.mean()),
        "yaw_error_degrees_max": float(yaw_error.max()),
        "scaling_abs_error_mean": float(scale_error.mean()),
        "scaling_abs_error_max": float(scale_error.max()),
    }

    baseline = None
    if args.baseline_predictions is not None:
        baseline_data = load_json(args.baseline_predictions.resolve())["predictions"]
        baseline_t = np.asarray(baseline_data["translation"], dtype=np.float64)
        baseline_r = np.asarray(baseline_data["rotation"], dtype=np.float64)
        baseline_s = np.asarray(baseline_data["scaling"], dtype=np.float64).reshape(-1)
        yaw_delta = [
            angle_error(yaw_degrees(pred_r[index]), yaw_degrees(baseline_r[index]))
            for index in range(len(poses))
        ]
        baseline = {
            "translation_max_abs_delta": float(np.max(np.abs(pred_t - baseline_t))),
            "yaw_max_delta_degrees": float(max(yaw_delta)),
            "scaling_max_abs_delta": float(np.max(np.abs(pred_s - baseline_s))),
        }
        baseline["within_conversion_tolerance"] = bool(
            baseline["translation_max_abs_delta"] < 0.001
            and baseline["yaw_max_delta_degrees"] < 0.01
            and baseline["scaling_max_abs_delta"] < 0.001
        )

    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    ax.scatter(gt_t[:, 0], gt_t[:, 2], label="Ground truth", color="#238B45", s=70)
    ax.scatter(pred_t[:, 0], pred_t[:, 2], label="Open-source HF inference", color="#CB3A31", marker="x", s=80)
    for index in range(len(poses)):
        ax.plot([gt_t[index, 0], pred_t[index, 0]], [gt_t[index, 2], pred_t[index, 2]], color="#777777")
        ax.text(pred_t[index, 0] + 0.02, pred_t[index, 2] + 0.02, str(index))
    ax.set(title="3D-FUTURE 0000000: layout prediction", xlabel="Scene X", ylabel="Scene Z")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "layout_gt_vs_prediction.png")
    plt.close(fig)

    assembled = scene_stats(root / "scenes" / "scene.glb")
    refined = scene_stats(root / "scenes" / "scene_refined.glb")
    package = load_json(reports / "package_validation.json")
    layout_stages = load_json(root / "timings" / "layout_stages.json")
    first_run = parse_time_report(root / "timings" / "layout_worker_first_run_time.txt")
    timed_run = parse_time_report(root / "timings" / "layout_worker_time.txt")
    post_refine_time = parse_time_report(root / "timings" / "post_refine_time.txt")
    post_refine = load_json(root / "post_refine" / "post_refine_summary.json")
    videos = {
        name: probe_video(root / "video" / name)
        for name in ("scene_turntable.mp4", "scene_refined_turntable.mp4")
    }
    input_asset_bytes = sum(path.stat().st_size for path in (root / "inputs" / "object_assets").glob("*/model.glb"))

    summary = {
        "status": "pass",
        "scope": {
            "rerun": "production open-source HF Layout, GLB assembly, and default post-refinement",
            "reused": "provided 3D-FUTURE RGB/masks and previously generated object GLBs",
            "not_rerun": ["Grounded-SAM2", "FLUX", "TRELLIS.2"],
        },
        "sample": {"dataset": "3D-FUTURE test", "scene_id": "0000000", "objects": len(poses)},
        "package_validation": {
            "status": package["status"],
            "size_bytes": package["package_size_bytes"],
            "backbone_tensor_count": package["backbone_tensor_count"],
            "layout_tensor_count": package["layout_tensor_count"],
        },
        "gt_metrics": gt_metrics,
        "previous_hf_adapter_delta": baseline,
        "timing": {
            "first_layout_process": first_run,
            "instrumented_layout_process": timed_run,
            "layout_subtasks": layout_stages,
            "post_refine_process": post_refine_time,
        },
        "outputs": {
            "input_object_assets_bytes": input_asset_bytes,
            "assembled_scene": assembled,
            "refined_scene": refined,
            "refined_size_delta_bytes": refined["size_bytes"] - assembled["size_bytes"],
        },
        "post_refine": post_refine,
        "media": videos,
        "limitations": [
            "This is a production Stage-3 Layout and refinement validation, not a fresh full RGB-to-asset run.",
            "The second timing run was slower than the first on a shared host; both measurements are retained.",
            "mIoU is measured in the calibrated input view; it is not a general scene reconstruction score.",
            "The output GLBs are visual meshes, not validated rigid-body or articulation assets.",
        ],
    }
    if package["status"] != "pass" or assembled["objects"] != len(poses) or refined["objects"] != len(poses):
        summary["status"] = "fail"
    if baseline is not None and not baseline["within_conversion_tolerance"]:
        summary["status"] = "fail"
    if post_refine["status"] != "ok":
        summary["status"] = "fail"
    (reports / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    metrics = post_refine["metrics"]
    report = f"""# Fysiverse HF Layout Production E2E Report

Date: 2026-08-21

## Result

Status: **{summary['status'].upper()}**. The production open-source loader applied
{package['backbone_tensor_count']} backbone replacements and
{package['layout_tensor_count']} Layout tensors, predicted all {len(poses)} objects,
exported `scene.glb`, and completed the default post-refinement path.

This run reused the 3D-FUTURE RGB, masks, and per-object GLBs. It did not rerun
Grounded-SAM2, FLUX, or TRELLIS.2, so it is a Stage-3 Layout + refinement E2E
validation rather than a fresh full RGB-to-asset benchmark.

## Quality

| Metric | Result |
| --- | ---: |
| Translation L2 mean / max | {gt_metrics['translation_l2_mean']:.6f} / {gt_metrics['translation_l2_max']:.6f} |
| Mean / max yaw error | {gt_metrics['yaw_error_degrees_mean']:.3f} deg / {gt_metrics['yaw_error_degrees_max']:.3f} deg |
| Scale absolute error mean / max | {gt_metrics['scaling_abs_error_mean']:.6f} / {gt_metrics['scaling_abs_error_max']:.6f} |
| Input-view mIoU before / after | {metrics['before_miou']:.6f} / {metrics['after_miou']:.6f} |
| PAT3D-style r_pen before / after | {metrics['before_r_pen']:.6f} / {metrics['after_r_pen']:.6f} |
| Penetrating triangle pairs before / after | {metrics['before_penetrating_triangle_pairs']} / {metrics['after_penetrating_triangle_pairs']} |
| Accepted refinement optimizations | {post_refine['optimization']['accepted_objects']} / {len(poses)} |

## Performance

| Stage | Wall time |
| --- | ---: |
| First production Layout process | {first_run['wall_seconds']:.2f} s |
| Instrumented Layout total | {layout_stages['total']:.3f} s |
| Model load | {layout_stages['stages']['model_load']:.3f} s |
| Seven-object model inference | {layout_stages['stages']['model_inference']:.3f} s |
| Pose postprocess | {layout_stages['stages']['pose_postprocess']:.3f} s |
| GLB assembly/export | {layout_stages['stages']['scene_assembly_export']:.3f} s |
| Post-refinement total | {post_refine_time['wall_seconds']:.2f} s |

The instrumented Layout process peaked at {timed_run['max_rss_bytes'] / 1024**3:.2f} GiB RSS.
The first and second Layout process measurements are both retained because the
shared H200 host produced substantial load/cache variance.

## Outputs

| Artifact | Bytes | Objects | Vertices | Faces |
| --- | ---: | ---: | ---: | ---: |
| `scenes/scene.glb` | {assembled['size_bytes']} | {assembled['objects']} | {assembled['vertices']} | {assembled['faces']} |
| `scenes/scene_refined.glb` | {refined['size_bytes']} | {refined['objects']} | {refined['vertices']} | {refined['faces']} |

Key materials include `reports/summary.json`, `reports/poses.json`,
`figures/layout_gt_vs_prediction.png`, `figures/input_view_comparison.png`,
the before/after 360-degree MP4/GIF files under `video/`, all 48 render frames,
the full refinement intermediates under `post_refine/`, and command/environment
records in the task root.

## Interpretation limits

- Input-view mIoU measures alignment to the calibrated photograph, not general 3D accuracy.
- The after metric combines pose optimization, global ground translation, and convex separation.
- `r_pen` is an unbounded normalized crossing count, not a percentage.
- The GLBs remain visual assets unless a separate USD/URDF physics export is run.
"""
    (root / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": summary["status"], "report": str(root / "REPORT.md")}, indent=2))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
