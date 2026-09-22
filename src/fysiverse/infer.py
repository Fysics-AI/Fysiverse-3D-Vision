"""Run the third-edition image-to-scene pipeline across five Conda environments."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _is_conda_prefix(environment: str) -> bool:
    expanded = os.path.expanduser(environment)
    return (
        Path(expanded).is_absolute()
        or environment.startswith(".")
        or os.sep in environment
        or (os.altsep is not None and os.altsep in environment)
    )


def _conda_selector(environment: str) -> list[str]:
    if _is_conda_prefix(environment):
        return ["-p", str(Path(environment).expanduser().resolve())]
    return ["-n", environment]


def _conda_python(conda_bin: str, environment: str, *arguments: str) -> list[str]:
    return [
        conda_bin,
        "run",
        "--no-capture-output",
        *_conda_selector(environment),
        "python",
        *arguments,
    ]


def _write_sam3d_runtime_config(
    template: Path,
    destination: Path,
    *,
    model_root: Path,
    source_root: Path,
) -> None:
    payload = yaml.safe_load(template.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"SAM3D config must contain a mapping: {template}")
    dinov2_root = model_root / "dinov2"
    payload["dinov2_source_root"] = str(source_root / "dinov2")
    payload["dinov2_torch_hub_dir"] = str(dinov2_root / "torch_hub")
    payload["dinov2_weights_path"] = str(
        dinov2_root / "torch_hub" / "checkpoints" / "dinov2_vitl14_reg4_pretrain.pth"
    )
    payload["ss_generator_config_path"] = str(
        model_root / "sam3d" / "checkpoints" / "ss_generator.yaml"
    )
    payload["ss_generator_ckpt_path"] = str(
        model_root / "sam3d" / "checkpoints" / "ss_generator.ckpt"
    )
    try:
        payload["depth_model"]["model"]["pretrained_model_name_or_path"] = str(
            model_root / "moge" / "model.pt"
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(f"SAM3D config has no depth_model.model mapping: {template}") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


def _run_stage(name: str, command: Sequence[str], log_path: Path, env: dict[str, str]) -> float:
    started_at = time.perf_counter()
    rendered = " ".join(str(part) for part in command)
    with log_path.open("a", encoding="utf-8") as log:
        header = f"\n[{name}] {rendered}\n"
        log.write(header)
        log.flush()
        print(header.rstrip(), flush=True)
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        return_code = process.wait()
    if return_code:
        raise RuntimeError(f"{name} failed with exit code {return_code}; see {log_path}")
    return time.perf_counter() - started_at


def _write_timings(path: Path, stages: dict[str, float | None], started_at: float) -> None:
    payload = {
        "unit": "seconds",
        "stages": {
            name: None if elapsed is None else round(elapsed, 3)
            for name, elapsed in stages.items()
        },
        "total": round(time.perf_counter() - started_at, 3),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    root = PROJECT_ROOT
    image = args.image.resolve()
    if not image.is_file():
        raise FileNotFoundError(f"input image not found: {image}")
    if args.mask is not None and not args.mask.resolve().is_file():
        raise FileNotFoundError(f"input mask not found: {args.mask.resolve()}")
    if shutil.which(args.conda_bin) is None:
        raise FileNotFoundError(f"Conda executable not found: {args.conda_bin}")
    model_root = args.model_root.expanduser().resolve()
    source_root = args.source_root.expanduser().resolve()
    layout_model = (
        args.layout_model.expanduser().resolve()
        if args.layout_model is not None
        else model_root / "layout"
    )
    if not layout_model.is_dir():
        raise FileNotFoundError(f"Layout model directory not found: {layout_model}")
    if not args.skip_refine:
        if args.camera is None:
            raise ValueError(
                "post-refinement is enabled by default and requires a calibrated input-view camera JSON; "
                "provide --camera path/to/camera.json or explicitly pass --skip-refine"
            )
        if not args.camera.resolve().is_file():
            raise FileNotFoundError(f"camera JSON not found: {args.camera.resolve()}")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sam3d_runtime_config = output / "work" / "sam3d_layout.runtime.yaml"
    _write_sam3d_runtime_config(
        root / "configs" / "sam3d_layout.yaml",
        sam3d_runtime_config,
        model_root=model_root,
        source_root=source_root,
    )
    log_path = output / "run.log"
    timings_path = output / "timings.json"
    run_started_at = time.perf_counter()
    timings: dict[str, float | None] = {
        "preflight": None,
        "mask": None,
        "prepare": None,
        "flux": None,
        "trellis2": None,
        "layout": None,
        "post_refine": None,
    }

    def run_stage(
        name: str,
        command: Sequence[str],
        source_paths: Sequence[Path] = (),
    ) -> None:
        stage_started_at = time.perf_counter()
        stage_env = runtime_env.copy()
        if source_paths:
            inherited = stage_env.get("PYTHONPATH")
            entries = [str(path) for path in source_paths]
            if inherited:
                entries.append(inherited)
            stage_env["PYTHONPATH"] = os.pathsep.join(entries)
        try:
            timings[name] = _run_stage(name, command, log_path, stage_env)
        except Exception:
            timings[name] = time.perf_counter() - stage_started_at
            _write_timings(timings_path, timings, run_started_at)
            raise
        _write_timings(timings_path, timings, run_started_at)

    runtime_env = os.environ.copy()
    runtime_env["CUDA_VISIBLE_DEVICES"] = args.gpu
    runtime_env.setdefault("PYTHONUNBUFFERED", "1")
    runtime_env.setdefault("HF_HUB_OFFLINE", "1")
    runtime_env.setdefault("TRANSFORMERS_OFFLINE", "1")
    project_source = str(root / "src")
    inherited_pythonpath = runtime_env.get("PYTHONPATH")
    runtime_env["PYTHONPATH"] = (
        project_source
        if not inherited_pythonpath
        else project_source + os.pathsep + inherited_pythonpath
    )

    run_config = {
        "project_root": str(root),
        "model_root": str(model_root),
        "source_root": str(source_root),
        "inputs": {
            "image": str(image),
            "mask": None if args.mask is None else str(args.mask.resolve()),
            "camera": None if args.camera is None else str(args.camera.resolve()),
        },
        "models": {
            "layout": str(layout_model),
            "g2vlm": str(model_root / "g2vlm"),
            "grounded_sam2": str(model_root / "grounded_sam2"),
            "bert_base_uncased": str(model_root / "bert_base_uncased"),
            "flux": str(model_root / "flux" / "FLUX.2-klein-9B"),
            "trellis2": str(model_root / "trellis2" / "TRELLIS.2-4B"),
            "trellis_image_large": str(model_root / "trellis_image_large"),
            "dinov3_vitl16": str(model_root / "dinov3_vitl16"),
            "rmbg2": str(model_root / "rmbg2"),
            "sam3d": str(model_root / "sam3d"),
            "moge": str(model_root / "moge"),
            "dinov2": str(model_root / "dinov2"),
        },
        "sources": {
            "grounded_sam2": str(source_root / "Grounded-SAM-2"),
            "trellis2": str(source_root / "TRELLIS.2"),
            "g2vlm": str(source_root / "G2VLM"),
            "sam3d": str(source_root / "sam-3d-objects"),
            "moge": str(source_root / "MoGe"),
            "dinov2": str(source_root / "dinov2"),
            "diffusers": str(source_root / "diffusers"),
        },
        "sources_root": str(source_root),
        "sam3d_runtime_config": str(sam3d_runtime_config),
        "environments": {
            "layout": args.layout_env,
            "mask": args.mask_env,
            "flux": args.flux_env,
            "trellis2": args.trellis_env,
            "refine": args.refine_env,
        },
        "executables": {"conda": args.conda_bin, "blender": args.blender},
        "options": {
            "gpu": args.gpu,
            "seed": args.seed,
            "dedupe_iou": args.dedupe_iou,
            "dedupe_containment": args.dedupe_containment,
            "skip_preflight": args.skip_preflight,
            "skip_refine": args.skip_refine,
            "scene_normalization": not args.no_scene_normalization,
        },
    }
    (output / "run_config.json").write_text(
        json.dumps(run_config, indent=2) + "\n", encoding="utf-8"
    )

    mask = args.mask.resolve() if args.mask is not None else None

    if not args.skip_preflight:
        preflight = [
            sys.executable,
            str(root / "scripts" / "preflight.py"),
            "--conda-bin",
            args.conda_bin,
            "--output-dir",
            str(output),
            "--layout-model",
            str(layout_model),
            "--model-root",
            str(model_root),
            "--source-root",
            str(source_root),
            "--layout-env",
            args.layout_env,
            "--mask-env",
            args.mask_env,
            "--flux-env",
            args.flux_env,
            "--trellis-env",
            args.trellis_env,
            "--refine-env",
            args.refine_env,
        ]
        if mask is not None:
            preflight.append("--provided-mask")
        if args.skip_refine:
            preflight.append("--skip-refine")
        if args.blender:
            preflight.extend(["--blender", args.blender])
        run_stage("preflight", preflight)

    metadata: Path | None = None
    instance_masks: Path | None = None
    if mask is None:
        auto_mask_dir = output / "work" / "auto_mask"
        mask_command = _conda_python(
            args.conda_bin,
            args.mask_env,
            "-m",
            "fysiverse.backends.grounded_sam2",
            "--image",
            str(image),
            "--output",
            str(auto_mask_dir),
            "--source-root",
            str(source_root / "Grounded-SAM-2"),
            "--text-prompt",
            args.text_prompt,
            "--box-threshold",
            str(args.box_threshold),
            "--text-threshold",
            str(args.text_threshold),
            "--min-mask-area",
            str(args.min_mask_area),
            "--dedupe-iou",
            str(args.dedupe_iou),
            "--dedupe-containment",
            str(args.dedupe_containment),
            "--sam2-checkpoint",
            str(model_root / "grounded_sam2" / "checkpoints" / "sam2.1_hiera_large.pt"),
            "--gdino-checkpoint",
            str(model_root / "grounded_sam2" / "gdino_checkpoints" / "groundingdino_swint_ogc.pth"),
            "--text-encoder",
            str(model_root / "bert_base_uncased"),
        )
        run_stage("mask", mask_command, [source_root / "Grounded-SAM-2"])
        mask = auto_mask_dir / "mask.png"
        metadata = auto_mask_dir / "mask_metadata.json"
        instance_masks = auto_mask_dir / "masks"

    prepare_command = _conda_python(
        args.conda_bin,
        args.layout_env,
        "-m",
        "fysiverse.prepare",
        "--image",
        str(image),
        "--mask",
        str(mask),
        "--output",
        str(output),
        "--mask-backend",
        "grounded_sam2" if metadata else "provided",
        "--mask-mode",
        args.mask_mode,
        "--min-mask-area",
        str(args.min_mask_area),
    )
    if metadata is not None and instance_masks is not None:
        prepare_command.extend(["--metadata", str(metadata), "--instance-masks-dir", str(instance_masks)])
    run_stage("prepare", prepare_command)
    manifest = output / "manifest.json"

    flux_command = _conda_python(
        args.conda_bin,
        args.flux_env,
        "-m",
        "fysiverse.backends.flux",
        "--manifest",
        str(manifest),
        "--model",
        str(model_root / "flux" / "FLUX.2-klein-9B"),
        "--device",
        "cuda:0",
        "--seed",
        str(args.seed),
    )
    run_stage("flux", flux_command, [source_root / "diffusers" / "src"])

    trellis_command = _conda_python(
        args.conda_bin,
        args.trellis_env,
        "-m",
        "fysiverse.backends.trellis2",
        "--manifest",
        str(manifest),
        "--source-root",
        str(source_root / "TRELLIS.2"),
        "--model",
        str(model_root / "trellis2" / "TRELLIS.2-4B"),
        "--sparse-structure-model",
        str(model_root / "trellis_image_large"),
        "--image-encoder-model",
        str(model_root / "dinov3_vitl16"),
        "--rembg-model",
        str(model_root / "rmbg2"),
    )
    run_stage(
        "trellis2",
        trellis_command,
        [source_root / "TRELLIS.2", source_root / "utils3d-trellis"],
    )

    layout_command = _conda_python(
        args.conda_bin,
        args.layout_env,
        "-m",
        "fysiverse.layout_worker",
        "--manifest",
        str(manifest),
        "--device",
        "cuda:0",
        "--layout-model",
        str(layout_model),
        "--base-model",
        str(model_root / "g2vlm"),
        "--g2vlm-source-root",
        str(source_root / "G2VLM"),
        "--sam3d-source-root",
        str(source_root / "sam-3d-objects"),
        "--sam3d-config",
        str(sam3d_runtime_config),
        "--timings-output",
        str(output / "layout_timings.json"),
    )
    if args.no_scene_normalization:
        layout_command.append("--no-scene-normalization")
    run_stage(
        "layout",
        layout_command,
        [
            source_root / "G2VLM",
            source_root / "sam-3d-objects",
            source_root / "MoGe",
            source_root / "utils3d-moge",
        ],
    )

    scene_path = output / "scene.glb"
    if not args.skip_refine:
        camera = args.camera.resolve()
        refined_path = output / "scene_refined.glb"
        refine_root = output / "post_refine"
        refine_command = [
            "bash",
            str(root / "scripts" / "post_refine" / "run.sh"),
            "--scene-glb",
            str(scene_path),
            "--image",
            str(image),
            "--mask-dir",
            str(output / "masks"),
            "--camera-json",
            str(camera),
            "--case-name",
            output.name,
            "--output-root",
            str(refine_root),
            "--final-glb",
            str(refined_path),
            "--overwrite",
        ]
        runtime_env["FYSIVERSE_REFINE_ENV"] = args.refine_env
        runtime_env["POST_REFINE_CONDA_BIN"] = args.conda_bin
        if args.blender:
            runtime_env["POST_REFINE_BLENDER"] = args.blender
        run_stage("post_refine", refine_command)
        manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
        manifest_data.setdefault("outputs", {})["scene_refined"] = "scene_refined.glb"
        manifest_data["outputs"]["timings"] = "timings.json"
        manifest_data["outputs"]["layout_timings"] = "layout_timings.json"
        manifest_data.setdefault("stages", {})["post_refine"] = "complete"
        manifest.write_text(json.dumps(manifest_data, indent=2) + "\n", encoding="utf-8")
        _write_timings(timings_path, timings, run_started_at)
        return refined_path
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_data.setdefault("stages", {})["post_refine"] = "skipped"
    manifest_data.setdefault("outputs", {})["timings"] = "timings.json"
    manifest_data["outputs"]["layout_timings"] = "layout_timings.json"
    manifest.write_text(json.dumps(manifest_data, indent=2) + "\n", encoding="utf-8")
    _write_timings(timings_path, timings, run_started_at)
    return scene_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mask", type=Path, help="Optional instance mask; omitted means automatic Grounded-SAM2")
    parser.add_argument(
        "--camera",
        type=Path,
        help="Calibrated input-view camera.json required by the default post-refinement stage.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", default="0", help="CUDA device index exposed to each stage")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--text-prompt", default="sofa. bed. chair. ceiling light. table. cabinet. shelf. desk. stool.")
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--dedupe-iou", type=float, default=0.9)
    parser.add_argument("--dedupe-containment", type=float, default=0.95)
    parser.add_argument("--min-mask-area", type=int, default=100)
    parser.add_argument("--mask-mode", choices=("auto", "color", "connected"), default="auto")
    parser.add_argument("--conda-bin", default=os.environ.get("CONDA_BIN", "conda"))
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path(os.environ.get("FYSIVERSE_MODEL_ROOT", str(PROJECT_ROOT / "models"))),
        help="Root containing model directories downloaded by scripts/download_models.py",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(
            os.environ.get(
                "FYSIVERSE_SOURCE_ROOT",
                str(PROJECT_ROOT / "third_party" / "src"),
            )
        ),
        help="Root containing source checkouts fetched by scripts/fetch_sources.py",
    )
    parser.add_argument("--mask-env", default=os.environ.get("FYSIVERSE_MASK_ENV", "fysiverse-mask"))
    parser.add_argument("--flux-env", default=os.environ.get("FYSIVERSE_FLUX_ENV", "fysiverse-flux"))
    parser.add_argument("--trellis-env", default=os.environ.get("FYSIVERSE_TRELLIS_ENV", "fysiverse-trellis2"))
    parser.add_argument("--refine-env", default=os.environ.get("FYSIVERSE_REFINE_ENV", "fysiverse-refine"))
    parser.add_argument(
        "--blender",
        default=os.environ.get("POST_REFINE_BLENDER"),
        help="Blender executable for refinement",
    )
    parser.add_argument("--layout-env", default=os.environ.get("FYSIVERSE_LAYOUT_ENV", "fysiverse-layout"))
    parser.add_argument(
        "--layout-model",
        type=Path,
        default=(
            Path(os.environ["FYSIVERSE_LAYOUT_MODEL"])
            if os.environ.get("FYSIVERSE_LAYOUT_MODEL")
            else None
        ),
        help="Layout model directory; defaults to <model-root>/layout",
    )
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument(
        "--skip-refine",
        "--skip_refine",
        "--no-post-refine",
        action="store_true",
        help="Keep scene.glb only and skip the default post-refinement stage.",
    )
    parser.add_argument("--no-scene-normalization", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    scene = run(args)
    print(json.dumps({"status": "ok", "scene": str(scene)}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
