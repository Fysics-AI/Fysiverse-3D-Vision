"""Predict per-object layout from a manifest and assemble the scene GLB."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from fysiverse.assembly import export_scene
from fysiverse.layout_runtime.loader import load_layout_model
from fysiverse.layout_runtime.postprocess import postprocess_predictions
from fysiverse.manifest import load_manifest, resolve_artifact


def run(args: argparse.Namespace) -> dict[str, object]:
    started_at = time.perf_counter()
    stages: dict[str, float] = {}
    stage_started_at = time.perf_counter()
    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    image = Path(manifest["image"])
    if not image.is_absolute():
        image = resolve_artifact(manifest_path, manifest["image"])
    if not image.is_file():
        raise FileNotFoundError(f"scene image not found: {image}")
    masks = [resolve_artifact(manifest_path, item["mask"]) for item in manifest["objects"]]
    for mask in masks:
        if not mask.is_file():
            raise FileNotFoundError(f"object mask not found: {mask}")
    stages["input_validation"] = time.perf_counter() - stage_started_at

    stage_started_at = time.perf_counter()
    model, config = load_layout_model(
        g2vlm_source_root=args.g2vlm_source_root,
        base_model_dir=args.base_model,
        layout_model_dir=args.layout_model,
        sam3d_source_root=args.sam3d_source_root,
        sam3d_pipeline_config=args.sam3d_config,
        device=args.device,
    )
    stages["model_load"] = time.perf_counter() - stage_started_at
    stage_started_at = time.perf_counter()
    predictions = model([str(image)] * len(masks), [str(mask) for mask in masks])
    stages["model_inference"] = time.perf_counter() - stage_started_at
    stage_started_at = time.perf_counter()
    decoded = postprocess_predictions(predictions, config)
    poses = []
    for item, pose in zip(manifest["objects"], decoded):
        poses.append(
            {
                "id": str(item["id"]),
                "category": str(item.get("category") or "object"),
                "mask": item["mask"],
                "asset": item["asset"],
                **pose,
            }
        )
    stages["pose_postprocess"] = time.perf_counter() - stage_started_at
    stage_started_at = time.perf_counter()
    result = export_scene(
        manifest_path,
        poses,
        normalize=config.normalize_scene and not args.no_scene_normalization,
        normalization_margin=config.normalization_margin,
    )
    stages["scene_assembly_export"] = time.perf_counter() - stage_started_at
    if args.timings_output is not None:
        timings_path = args.timings_output.resolve()
        timings_path.parent.mkdir(parents=True, exist_ok=True)
        timings_path.write_text(
            json.dumps(
                {
                    "unit": "seconds",
                    "device": args.device,
                    "objects": len(masks),
                    "stages": {name: round(value, 3) for name, value in stages.items()},
                    "total": round(time.perf_counter() - started_at, 3),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=root / "models" / "g2vlm")
    parser.add_argument(
        "--layout-model",
        type=Path,
        default=Path(
            os.environ.get(
                "FYSIVERSE_LAYOUT_MODEL", str(root / "models" / "layout")
            )
        ),
    )
    parser.add_argument("--g2vlm-source-root", type=Path, default=root / "third_party" / "src" / "G2VLM")
    parser.add_argument("--sam3d-source-root", type=Path, default=root / "third_party" / "src" / "sam-3d-objects")
    parser.add_argument("--sam3d-config", type=Path, default=root / "configs" / "sam3d_layout.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--timings-output", type=Path)
    parser.add_argument("--no-scene-normalization", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args)
    print(json.dumps({"status": "ok", "objects": len(result["objects"]), "stage": "layout"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
