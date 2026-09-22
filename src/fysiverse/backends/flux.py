"""FLUX.2 object-completion worker."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from fysiverse.manifest import load_manifest, relative_artifact, resolve_artifact, save_manifest


DEFAULT_PROMPT = (
    "Face the object toward the camera. Preserve its material, texture, pattern, reflectivity, "
    "and colors. Complete occluded or missing parts. Show one complete object with a clear "
    "silhouette on a clean white background, without unrelated objects or obstructions."
)


def prepare_condition(path: Path, *, padding: int = 2, min_side: int = 128, multiple: int = 16) -> Image.Image:
    with Image.open(path) as source:
        rgba = source.convert("RGBA")
    alpha = np.asarray(rgba.getchannel("A"), dtype=np.uint8)
    ys, xs = np.where(alpha > 10)
    if not xs.size:
        raise ValueError(f"cutout has no visible foreground: {path}")
    left = max(int(xs.min()) - padding, 0)
    top = max(int(ys.min()) - padding, 0)
    right = min(int(xs.max()) + padding + 1, rgba.width)
    bottom = min(int(ys.max()) + padding + 1, rgba.height)
    cropped = rgba.crop((left, top, right, bottom))
    background = Image.new("RGBA", cropped.size, (255, 255, 255, 255))
    condition = Image.alpha_composite(background, cropped).convert("RGB")

    if min(condition.size) < min_side:
        scale = min_side / float(min(condition.size))
        condition = condition.resize(
            (int(math.ceil(condition.width * scale)), int(math.ceil(condition.height * scale))),
            Image.Resampling.LANCZOS,
        )
    width = int(math.ceil(condition.width / multiple) * multiple)
    height = int(math.ceil(condition.height / multiple) * multiple)
    if (width, height) != condition.size:
        canvas = Image.new("RGB", (width, height), "white")
        canvas.paste(condition, ((width - condition.width) // 2, (height - condition.height) // 2))
        condition = canvas
    return condition


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from diffusers import Flux2KleinPipeline

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot see a CUDA device")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    pipeline = Flux2KleinPipeline.from_pretrained(str(args.model.resolve()), torch_dtype=dtype)
    if args.cpu_offload:
        pipeline.enable_model_cpu_offload()
    else:
        pipeline.to(torch.device(args.device))

    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    completed_dir = manifest_path.parent / "work" / "completed"
    completed_dir.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(manifest["objects"]):
        output = completed_dir / f"{item['id']}.png"
        if output.is_file() and not args.overwrite:
            item["completed_image"] = relative_artifact(manifest_path, output)
            continue
        condition = prepare_condition(resolve_artifact(manifest_path, item["cutout"]))
        category = str(item.get("category") or "object")
        prompt = args.prompt or DEFAULT_PROMPT
        prompt = f"This is a {category}. {prompt}"
        generator = torch.Generator(device=args.device).manual_seed(args.seed + index)
        result = pipeline(
            condition,
            prompt,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.steps,
            generator=generator,
        ).images[0]
        result.save(output)
        item["completed_image"] = relative_artifact(manifest_path, output)
        item["completion_seed"] = args.seed + index
        save_manifest(manifest_path, manifest)

    manifest["stages"]["flux"] = "complete"
    save_manifest(manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=project_root / "models" / "flux" / "FLUX.2-klein-9B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = run(args)
    print(json.dumps({"status": "ok", "objects": len(manifest["objects"]), "stage": "flux"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
