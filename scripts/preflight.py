#!/usr/bin/env python3
"""Check model directories and optional conda environments before inference."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ROOT = ROOT / "models"
DEFAULT_SOURCE_ROOT = ROOT / "third_party" / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from download_models import check_model as check_downloaded_model


ENV_IMPORTS = {
    "mask": ("torch", "sam2", "groundingdino", "fysiverse"),
    "flux": ("torch", "diffusers", "fysiverse"),
    "trellis": (
        "torch",
        "trellis2",
        "o_voxel",
        "cumesh",
        "flex_gemm",
        "flash_attn",
        "nvdiffrast",
        "nvdiffrec_render",
        "utils3d",
        "fysiverse",
    ),
    "layout": (
        "torch",
        "fysiverse",
        "sam3d_objects",
        "moge",
        "pytorch3d",
        "kaolin",
        "gsplat",
        "flash_attn",
        "lightning",
        "loguru",
        "spconv",
        "utils3d",
    ),
    "refine": ("torch", "nvdiffrast", "cv2", "imageio", "PIL", "fysiverse"),
}
SOURCE_CONFIG = ROOT / "configs" / "sources.json"
AUTO_MASK_MODELS = frozenset({"grounded_sam2", "bert_base_uncased"})
ENV_SOURCE_RELATIVE_PATHS = {
    "mask": ("Grounded-SAM-2",),
    "layout": (
        "G2VLM",
        "sam-3d-objects",
        "MoGe",
    ),
    "trellis": ("TRELLIS.2", "utils3d-trellis"),
    "flux": ("diffusers/src",),
}
ENV_SOURCE_PATHS = {
    role: tuple(DEFAULT_SOURCE_ROOT / relative for relative in relative_paths)
    for role, relative_paths in ENV_SOURCE_RELATIVE_PATHS.items()
}
ENV_RUNTIME_PROBES = {
    "mask": (
        "import importlib; "
        "from grounding_dino.groundingdino.util.inference import load_image, load_model, predict; "
        "from sam2.build_sam import build_sam2; "
        "from sam2.sam2_image_predictor import SAM2ImagePredictor; "
        "importlib.import_module('sam2._C'); "
        "importlib.import_module('grounding_dino.groundingdino._C')"
    ),
    "flux": (
        "from diffusers import Flux2KleinPipeline; "
        "from fysiverse.backends.flux import prepare_condition, run"
    ),
    "trellis": (
        "from transformers import DINOv3ViTModel; "
        "from fysiverse.backends.trellis2 import run"
    ),
    "layout": (
        "from fysiverse.layout_runtime.layout_network import _import_sam3d_runtime; "
        f"_import_sam3d_runtime({str(ROOT / 'third_party' / 'src' / 'sam-3d-objects')!r}); "
        "import moge; "
        "from sam3d_objects.data.utils import expand_as_right, tree_tensor_map; "
        "from sam3d_objects.data.dataset.tdfy.preprocessor import PreProcessor; "
        "from sam3d_objects.model.backbone.dit.embedder.embedder_fuser import EmbedderFuser; "
        "from sam3d_objects.model.backbone.dit.embedder.dino import Dino; "
        "from sam3d_objects.model.backbone.dit.embedder.pointmap import PointPatchEmbed; "
        "from sam3d_objects.pipeline.depth_models.moge import MoGe"
    ),
    "refine": (
        "import runpy; "
        f"runpy.run_path({str(ROOT / 'scripts' / 'post_refine' / 'post_refine_adapter.py')!r}, "
        "run_name='fysiverse_refine_adapter_probe'); "
        f"runpy.run_path({str(ROOT / 'scripts' / 'post_refine' / 'vendor' / 'fysiverse_3d' / 'scripts' / 'optimize_object_poses_nvdiffrast.py')!r}, "
        "run_name='fysiverse_refine_runtime_probe')"
    ),
}


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


def _configured_path(configured: str, prefix: tuple[str, ...], root: Path) -> Path:
    path = Path(configured)
    if path.is_absolute():
        return path
    if path.parts[: len(prefix)] == prefix:
        return root.joinpath(*path.parts[len(prefix) :])
    return ROOT / path


def _environment_source_paths(role: str, source_root: Path) -> list[Path]:
    return [
        source_root / relative
        for relative in ENV_SOURCE_RELATIVE_PATHS.get(role, ())
    ]


def _runtime_probe(role: str, source_root: Path) -> str:
    probe = ENV_RUNTIME_PROBES.get(role, "")
    if role == "layout":
        probe = probe.replace(
            str(DEFAULT_SOURCE_ROOT / "sam-3d-objects"),
            str(source_root / "sam-3d-objects"),
        )
    return probe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-only", action="store_true")
    parser.add_argument("--allow-missing-models", action="store_true")
    parser.add_argument("--skip-conda", action="store_true")
    parser.add_argument("--skip-imports", action="store_true")
    parser.add_argument("--allow-no-gpu", action="store_true")
    parser.add_argument(
        "--provided-mask",
        action="store_true",
        help="Skip Grounded-SAM2 model/source/environment checks because the caller supplies a mask.",
    )
    parser.add_argument("--conda-bin", default=os.environ.get("CONDA_BIN", "conda"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "result")
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path(os.environ.get("FYSIVERSE_MODEL_ROOT", str(DEFAULT_MODEL_ROOT))),
        help="Root containing model directories downloaded by scripts/download_models.py",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(os.environ.get("FYSIVERSE_SOURCE_ROOT", str(DEFAULT_SOURCE_ROOT))),
        help="Root containing source checkouts fetched by scripts/fetch_sources.py",
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
    parser.add_argument("--mask-env", default=os.environ.get("FYSIVERSE_MASK_ENV", "fysiverse-mask"))
    parser.add_argument("--flux-env", default=os.environ.get("FYSIVERSE_FLUX_ENV", "fysiverse-flux"))
    parser.add_argument("--trellis-env", default=os.environ.get("FYSIVERSE_TRELLIS_ENV", "fysiverse-trellis2"))
    parser.add_argument("--refine-env", default=os.environ.get("FYSIVERSE_REFINE_ENV", "fysiverse-refine"))
    parser.add_argument("--blender", default=os.environ.get("POST_REFINE_BLENDER", "blender"))
    parser.add_argument("--min-blender", default="4.5")
    parser.add_argument("--skip-refine", action="store_true", help="Skip refinement environment and Blender checks.")
    return parser.parse_args()


def check_models(
    allow_missing: bool,
    *,
    provided_mask: bool = False,
    layout_model: Path | None = None,
    model_root: Path | None = None,
) -> int:
    resolved_model_root = (
        ROOT / "models" if model_root is None else model_root.expanduser().resolve()
    )
    config_path = ROOT / "configs" / "models.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    failures = 0
    for name, spec in config["models"].items():
        if provided_mask and name in AUTO_MASK_MODELS:
            print(f"[models] {name}: skipped (using provided mask)")
            continue
        resolved_spec = dict(spec)
        if name == "layout" and layout_model is not None:
            model_dir = layout_model.expanduser().resolve()
        else:
            model_dir = _configured_path(
                str(spec["local_dir"]), ("models",), resolved_model_root
            )
        resolved_spec["local_dir"] = str(model_dir)
        if not spec.get("required", True) and not model_dir.exists():
            print(f"[models] {name}: optional and not installed")
            continue
        missing = check_downloaded_model(name, resolved_spec, ROOT)
        if missing:
            failures += 1
            print(f"[models] {name}: {len(missing)} validation issue(s)", file=sys.stderr)
            for path in missing:
                print(f"  {path}", file=sys.stderr)
        else:
            print(f"[models] {name}: ready ({model_dir})")
    if allow_missing:
        return 0
    return failures


def check_layout_model(
    model_dir: Path | None = None,
    *,
    model_root: Path | None = None,
) -> int:
    resolved_model_root = (
        ROOT / "models" if model_root is None else model_root.expanduser().resolve()
    )
    model_dir = (
        resolved_model_root / "layout"
        if model_dir is None
        else model_dir.expanduser().resolve()
    )
    if not (model_dir / "config.json").is_file() or not (model_dir / "layout.safetensors").is_file():
        return 0
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "validate_layout_model.py"), "--model-dir", str(model_dir)],
        check=False,
    )
    return result.returncode


def check_conda(conda_bin: str, expected: set[str]) -> int:
    if shutil.which(conda_bin) is None:
        print(f"[conda] not found: {conda_bin}", file=sys.stderr)
        return 1
    prefix_targets = {
        selector: Path(selector).expanduser().resolve()
        for selector in expected
        if _is_conda_prefix(selector)
    }
    missing_prefixes = [
        selector
        for selector, prefix in prefix_targets.items()
        if not (prefix / "conda-meta").is_dir()
    ]
    name_targets = expected - set(prefix_targets)
    if not name_targets:
        if missing_prefixes:
            print(
                "[conda] missing environment prefixes: "
                + ", ".join(sorted(missing_prefixes)),
                file=sys.stderr,
            )
            return 1
        print(f"[conda] all {len(expected)} inference environment prefixes are present")
        return 0
    result = subprocess.run([conda_bin, "env", "list"], check=False, capture_output=True, text=True)
    if result.returncode:
        print(result.stderr.strip(), file=sys.stderr)
        return result.returncode
    names = {
        line.split()[0]
        for line in result.stdout.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    missing = sorted(name_targets - names) + sorted(missing_prefixes)
    if missing:
        print("[conda] missing environments: " + ", ".join(missing), file=sys.stderr)
        return 1
    print(f"[conda] all {len(expected)} inference environments are present")
    return 0


def check_env_imports(
    conda_bin: str,
    environments: dict[str, str],
    *,
    source_root: Path | None = None,
) -> int:
    resolved_source_root = (
        ROOT / "third_party" / "src"
        if source_root is None
        else source_root.expanduser().resolve()
    )
    failures = 0
    probe_env = os.environ.copy()
    probe_env.pop("PYTHONPATH", None)
    for role, env_name in environments.items():
        modules = ENV_IMPORTS[role]
        source_paths = [str(ROOT / "src")]
        source_paths.extend(
            str(path) for path in _environment_source_paths(role, resolved_source_root)
        )
        runtime_probe = _runtime_probe(role, resolved_source_root)
        probe = (
            "import importlib.util,json,os,sys; "
            "os.environ.setdefault('LIDRA_SKIP_INIT','true'); "
            f"sys.path[:0]={source_paths!r}; "
            f"mods={list(modules)!r}; "
            "missing=[m for m in mods if importlib.util.find_spec(m) is None]; "
            f"exec({runtime_probe!r}) if not missing else None; "
            "print(json.dumps(missing))"
        )
        result = subprocess.run(
            [conda_bin, "run", *_conda_selector(env_name), "python", "-c", probe],
            check=False,
            capture_output=True,
            text=True,
            env=probe_env,
        )
        if result.returncode:
            failures += 1
            detail = result.stderr.strip() or result.stdout.strip() or "interpreter probe failed"
            print(f"[imports] {role} ({env_name}): {detail}", file=sys.stderr)
            continue
        try:
            missing = json.loads(result.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            failures += 1
            print(f"[imports] {role} ({env_name}): invalid probe output", file=sys.stderr)
            continue
        if missing:
            failures += 1
            print(
                f"[imports] {role} ({env_name}): missing {', '.join(missing)}",
                file=sys.stderr,
            )
        else:
            print(f"[imports] {role} ({env_name}): ready")
    return failures


def check_env_dependencies(conda_bin: str, environments: dict[str, str]) -> int:
    failures = 0
    dependency_env = os.environ.copy()
    dependency_env.pop("PYTHONPATH", None)
    for role, env_name in environments.items():
        result = subprocess.run(
            [conda_bin, "run", *_conda_selector(env_name), "python", "-m", "pip", "check"],
            check=False,
            capture_output=True,
            text=True,
            env=dependency_env,
        )
        if result.returncode:
            failures += 1
            detail = result.stdout.strip() or result.stderr.strip() or "pip check failed"
            print(f"[dependencies] {role} ({env_name}): {detail}", file=sys.stderr)
        else:
            print(f"[dependencies] {role} ({env_name}): consistent")
    return failures


def check_output_dir(output_dir: Path) -> int:
    candidate = output_dir.resolve()
    existing = candidate
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    if not existing.is_dir() or not os.access(existing, os.W_OK):
        print(
            f"[output] not writable: {candidate} (nearest existing parent: {existing})",
            file=sys.stderr,
        )
        return 1
    print(f"[output] writable parent: {existing}")
    return 0


def check_sources(
    *,
    provided_mask: bool = False,
    source_root: Path | None = None,
) -> int:
    resolved_source_root = (
        ROOT / "third_party" / "src"
        if source_root is None
        else source_root.expanduser().resolve()
    )
    if shutil.which("git") is None:
        print("[source] git was not found", file=sys.stderr)
        return 1
    try:
        specs = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))["sources"]
    except (OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"[source] cannot read {SOURCE_CONFIG}: {exc}", file=sys.stderr)
        return 1
    failures = 0
    for name, spec in specs.items():
        if provided_mask and name == "grounded_sam2":
            print("[source] Grounded-SAM2: skipped (using provided mask)")
            continue
        destination = _configured_path(
            str(spec["local_dir"]), ("third_party", "src"), resolved_source_root
        )
        required = destination / str(spec["required_file"])
        if not (destination / ".git").is_dir() or not required.is_file():
            failures += 1
            print(f"[source] {name}: missing or incomplete ({destination})", file=sys.stderr)
            continue
        result = subprocess.run(
            ["git", "-C", str(destination), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        current = result.stdout.strip()
        expected = str(spec["revision"])
        if result.returncode or current != expected:
            failures += 1
            print(
                f"[source] {name}: revision mismatch (expected {expected}, got {current or 'unknown'})",
                file=sys.stderr,
            )
            continue
        origin = subprocess.run(
            ["git", "-C", str(destination), "remote", "get-url", "origin"],
            check=False,
            capture_output=True,
            text=True,
        )
        expected_origin = str(spec["repo_url"])
        if origin.returncode or origin.stdout.strip() != expected_origin:
            failures += 1
            print(
                f"[source] {name}: origin mismatch "
                f"(expected {expected_origin}, got {origin.stdout.strip() or 'missing'})",
                file=sys.stderr,
            )
            continue
        if spec.get("recursive"):
            submodules = subprocess.run(
                ["git", "-C", str(destination), "submodule", "status", "--recursive"],
                check=False,
                capture_output=True,
                text=True,
            )
            invalid = [
                line
                for line in submodules.stdout.splitlines()
                if line.startswith(("-", "+", "U"))
            ]
            if submodules.returncode or invalid:
                failures += 1
                print(f"[source] {name}: submodules are not at pinned revisions", file=sys.stderr)
                continue
        print(f"[source] {name}: ready ({current})")
    if failures:
        print("[source] run: bash scripts/fetch_sources.sh", file=sys.stderr)
    return failures


def check_gpu(allow_no_gpu: bool) -> int:
    gpu = shutil.which("nvidia-smi")
    if gpu:
        result = subprocess.run(
            [gpu, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            print(f"[gpu] available: {result.stdout.strip()}")
            return 0
        if allow_no_gpu:
            print("[gpu] nvidia-smi cannot query a GPU; accepted by --allow-no-gpu", file=sys.stderr)
            return 0
        print("[gpu] nvidia-smi cannot query a usable GPU", file=sys.stderr)
        return 1
    if allow_no_gpu:
        print("[gpu] not found; accepted by --allow-no-gpu", file=sys.stderr)
        return 0
    print("[gpu] nvidia-smi not found; full inference requires a supported NVIDIA GPU", file=sys.stderr)
    return 1


def check_blender(executable: str, minimum: str) -> int:
    path = shutil.which(executable) or (executable if Path(executable).is_file() else None)
    if path is None:
        print(f"[blender] not found: {executable}; install Blender >= {minimum} or set POST_REFINE_BLENDER", file=sys.stderr)
        return 1
    result = subprocess.run([path, "--version"], check=False, capture_output=True, text=True)
    match = re.search(r"Blender\s+(\d+(?:\.\d+)+)", result.stdout + "\n" + result.stderr)
    if result.returncode != 0 or match is None:
        print(f"[blender] could not validate executable: {path}", file=sys.stderr)
        return 1
    detected = tuple(int(part) for part in match.group(1).split("."))
    required = tuple(int(part) for part in minimum.split("."))
    print(f"[blender] {path}: {match.group(1)} (required: >= {minimum})")
    if detected < required:
        print(f"[blender] version {match.group(1)} is too old for refinement", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    args = parse_args()
    model_root = args.model_root.expanduser().resolve()
    source_root = args.source_root.expanduser().resolve()
    layout_model = (
        args.layout_model.expanduser().resolve()
        if args.layout_model is not None
        else model_root / "layout"
    )
    failures = check_models(
        args.allow_missing_models,
        provided_mask=args.provided_mask,
        layout_model=layout_model,
        model_root=model_root,
    )
    failures += check_layout_model(layout_model, model_root=model_root)
    if args.models_only:
        return 1 if failures else 0

    failures += check_sources(
        provided_mask=args.provided_mask,
        source_root=source_root,
    )

    environments = {
        "layout": args.layout_env,
        "flux": args.flux_env,
        "trellis": args.trellis_env,
    }
    if not args.skip_refine:
        environments["refine"] = args.refine_env
    if not args.provided_mask:
        environments["mask"] = args.mask_env
    if not args.skip_conda:
        conda_failures = check_conda(args.conda_bin, set(environments.values()))
        failures += conda_failures
        if not conda_failures:
            failures += check_env_dependencies(args.conda_bin, environments)
            if not args.skip_imports:
                failures += check_env_imports(
                    args.conda_bin,
                    environments,
                    source_root=source_root,
                )
    elif not args.skip_imports:
        print("[imports] skipped because --skip-conda was specified", file=sys.stderr)

    failures += check_gpu(args.allow_no_gpu)
    if not args.skip_refine:
        failures += check_blender(args.blender, args.min_blender)
    failures += check_output_dir(args.output_dir)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
