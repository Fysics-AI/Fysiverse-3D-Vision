"""TRELLIS.2 image-to-GLB worker."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image

from fysiverse.manifest import load_manifest, relative_artifact, resolve_artifact, save_manifest


def _local_pipeline_config(args: argparse.Namespace, temporary_root: Path) -> Path:
    model_root = args.model.resolve()
    payload = json.loads((model_root / "pipeline.json").read_text(encoding="utf-8"))
    pipeline_args = payload["args"]
    for name, model_name in list(pipeline_args["models"].items()):
        if model_name == "microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16":
            source = args.sparse_structure_model.resolve() / "ckpts" / "ss_dec_conv3d_16l8_fp16"
            relative = Path("external") / "ss_dec_conv3d_16l8_fp16"
        else:
            source = model_root / str(model_name)
            relative = Path(str(model_name))
        for suffix in (".json", ".safetensors"):
            source_file = Path(str(source) + suffix)
            if not source_file.is_file():
                raise FileNotFoundError(f"TRELLIS.2 {name} file not found: {source_file}")
            destination = Path(str(temporary_root / relative) + suffix)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to(source_file)
        pipeline_args["models"][name] = relative.as_posix()
    pipeline_args["image_cond_model"]["args"]["model_name"] = str(
        args.image_encoder_model.resolve()
    )
    pipeline_args["rembg_model"]["args"]["model_name"] = str(args.rembg_model.resolve())
    config_path = temporary_root / "pipeline.json"
    config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return config_path


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_root = args.source_root.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"TRELLIS.2 source directory not found: {source_root}")
    if not args.model.resolve().is_dir():
        raise FileNotFoundError(f"TRELLIS.2 model directory not found: {args.model.resolve()}")
    for directory, label in (
        (args.sparse_structure_model.resolve(), "TRELLIS-image-large"),
        (args.image_encoder_model.resolve(), "DINOv3"),
        (args.rembg_model.resolve(), "RMBG-2.0"),
    ):
        if not directory.is_dir():
            raise FileNotFoundError(f"{label} model directory not found: {directory}")
    sys.path.insert(0, str(source_root))
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import o_voxel
    import torch
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("TRELLIS.2 inference requires a CUDA device")
    torch.set_grad_enabled(False)
    with tempfile.TemporaryDirectory(prefix="fysiverse-trellis2-") as temp_dir:
        runtime_root = Path(temp_dir)
        _local_pipeline_config(args, runtime_root)
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(str(runtime_root))
    pipeline.cuda()

    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    for item in manifest["objects"]:
        output = manifest_path.parent / "object_assets" / str(item["id"]) / "model.glb"
        if output.is_file() and not args.overwrite:
            item["asset"] = relative_artifact(manifest_path, output)
            continue
        completed_image = item.get("completed_image")
        if not completed_image:
            raise ValueError(f"object {item['id']} has no completed_image; run the FLUX stage first")
        with Image.open(resolve_artifact(manifest_path, completed_image)) as source:
            image = source.convert("RGB")
        mesh = pipeline.run(image)[0]
        mesh.simplify(args.simplify_max_faces)
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=args.decimation_target,
            texture_size=args.texture_size,
            remesh=True,
            remesh_band=1,
            remesh_project=0,
            verbose=args.verbose,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        glb.export(str(output), extension_webp=True)
        item["asset"] = relative_artifact(manifest_path, output)
        save_manifest(manifest_path, manifest)
        del image, mesh, glb
        gc.collect()
        torch.cuda.empty_cache()

    manifest["stages"]["trellis2"] = "complete"
    save_manifest(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=project_root / "third_party" / "src" / "TRELLIS.2")
    parser.add_argument("--model", type=Path, default=project_root / "models" / "trellis2" / "TRELLIS.2-4B")
    parser.add_argument(
        "--sparse-structure-model",
        type=Path,
        default=project_root / "models" / "trellis_image_large",
    )
    parser.add_argument(
        "--image-encoder-model",
        type=Path,
        default=project_root / "models" / "dinov3_vitl16",
    )
    parser.add_argument(
        "--rembg-model", type=Path, default=project_root / "models" / "rmbg2"
    )
    parser.add_argument("--simplify-max-faces", type=int, default=300000)
    parser.add_argument("--decimation-target", type=int, default=150000)
    parser.add_argument("--texture-size", type=int, default=1024)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = run(args)
    print(json.dumps({"status": "ok", "objects": len(manifest["objects"]), "stage": "trellis2"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
