#!/usr/bin/env python3
"""Check host prerequisites before creating inference environments."""

from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-free-gb", type=float, default=30.0)
    parser.add_argument("--min-vram-gb", type=float, default=24.0)
    parser.add_argument("--min-cuda", default="12.4", help="Minimum CUDA capability reported by the NVIDIA driver.")
    parser.add_argument("--allow-no-gpu", action="store_true")
    parser.add_argument("--path", type=Path, default=Path.cwd())
    parser.add_argument("--conda-bin", default="conda", help="Conda/Mamba executable to check.")
    parser.add_argument("--blender", default=None, help="Blender executable (default: POST_REFINE_BLENDER or blender).")
    parser.add_argument("--min-blender", default="4.5", help="Minimum Blender version required by refinement.")
    parser.add_argument("--skip-blender", action="store_true", help="Skip Blender validation for non-refinement setup.")
    parser.add_argument(
        "--require-nvcc",
        action="store_true",
        help="Require a local CUDA toolkit compiler for source/CUDA extension builds.",
    )
    return parser.parse_args()


def check_command(name: str, failures: list[str]) -> None:
    if shutil.which(name) is None:
        failures.append(f"required command not found: {name}")


def version_tuple(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in value.split("."))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid version: {value}") from exc


def check_gpu(min_vram_gb: float, min_cuda: str, allow_no_gpu: bool, failures: list[str]) -> None:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        if not allow_no_gpu:
            failures.append("nvidia-smi was not found; use --allow-no-gpu only for dependency setup")
        return

    query = subprocess.run(
        [nvidia_smi, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
        check=False,
        capture_output=True,
        text=True,
    )
    if query.returncode != 0 or not query.stdout.strip():
        if allow_no_gpu:
            print("[gpu] nvidia-smi could not query a GPU; accepted by --allow-no-gpu", file=sys.stderr)
            return
        failures.append("nvidia-smi could not query a usable GPU")
        return

    max_vram_mib = 0.0
    for line in query.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 2:
            continue
        try:
            max_vram_mib = max(max_vram_mib, float(fields[1]))
        except ValueError:
            continue
    print(f"[gpu] detected:\n{query.stdout.strip()}")
    if max_vram_mib < min_vram_gb * 1024:
        failures.append(
            f"GPU VRAM is below the recommended {min_vram_gb:.0f} GiB "
            f"(largest detected GPU: {max_vram_mib / 1024:.1f} GiB)"
        )

    summary = subprocess.run([nvidia_smi], check=False, capture_output=True, text=True)
    match = re.search(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)?)", summary.stdout)
    if match:
        detected_cuda = match.group(1)
        print(f"[gpu] driver CUDA capability: {detected_cuda} (required: >= {min_cuda})")
        if version_tuple(detected_cuda) < version_tuple(min_cuda):
            failures.append(
                f"NVIDIA driver CUDA capability {detected_cuda} is below the required {min_cuda}; "
                "the host owner must update the NVIDIA driver or choose compatible runtime builds"
            )
    else:
        failures.append("nvidia-smi did not report the NVIDIA driver CUDA capability")


def check_cuda_toolkit(min_cuda: str, failures: list[str]) -> None:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        failures.append(
            "nvcc was not found; install a CUDA toolkit compatible with the configured "
            "PyTorch builds before compiling inference extensions"
        )
        return
    result = subprocess.run([nvcc, "--version"], check=False, capture_output=True, text=True)
    match = re.search(r"release\s+([0-9]+(?:\.[0-9]+)?)", result.stdout + "\n" + result.stderr)
    if result.returncode != 0 or match is None:
        failures.append(f"could not determine the CUDA toolkit version from: {nvcc}")
        return
    detected = match.group(1)
    print(f"[cuda-toolkit] {nvcc}: {detected} (required: >= {min_cuda})")
    if version_tuple(detected) < version_tuple(min_cuda):
        failures.append(
            f"CUDA toolkit {detected} is below the required {min_cuda}; "
            "the host owner must install a compatible toolkit"
        )


def check_blender(executable: str, minimum: str, failures: list[str]) -> None:
    path = shutil.which(executable) or (executable if Path(executable).is_file() else None)
    if path is None:
        failures.append(
            f"Blender was not found ({executable}); install Blender >= {minimum} and set POST_REFINE_BLENDER"
        )
        return
    result = subprocess.run([path, "--version"], check=False, capture_output=True, text=True)
    output = (result.stdout + "\n" + result.stderr).strip()
    match = re.search(r"Blender\s+(\d+(?:\.\d+)+)", output)
    if result.returncode != 0 or match is None:
        failures.append(f"could not validate Blender executable: {path}")
        return
    detected = version_tuple(match.group(1))
    print(f"[blender] {path}: {match.group(1)} (required: >= {minimum})")
    if detected < version_tuple(minimum):
        failures.append(f"Blender {match.group(1)} is too old; refinement requires >= {minimum}")


def main() -> int:
    args = parse_args()
    failures: list[str] = []
    if platform.system() != "Linux":
        failures.append(f"unsupported operating system: {platform.system()} (Linux is required)")
    if sys.maxsize <= 2**32:
        failures.append("a 64-bit Python/runtime is required")

    check_command(args.conda_bin, failures)
    check_command("git", failures)
    check_command("gcc", failures)
    check_command("g++", failures)
    check_command("make", failures)
    check_gpu(args.min_vram_gb, args.min_cuda, args.allow_no_gpu, failures)
    if args.require_nvcc and not args.allow_no_gpu:
        check_cuda_toolkit(args.min_cuda, failures)
    if not args.skip_blender:
        check_blender(args.blender or os.environ.get("POST_REFINE_BLENDER", "blender"), args.min_blender, failures)

    usage = shutil.disk_usage(args.path.resolve())
    free_gb = usage.free / (1024**3)
    print(f"[disk] {args.path.resolve()} free={free_gb:.1f} GiB, required={args.min_free_gb:.1f} GiB")
    if free_gb < args.min_free_gb:
        failures.append(
            f"insufficient free disk space: {free_gb:.1f} GiB available, "
            f"at least {args.min_free_gb:.1f} GiB required"
        )

    if failures:
        print("[system] prerequisite check failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print("[system] prerequisite check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
