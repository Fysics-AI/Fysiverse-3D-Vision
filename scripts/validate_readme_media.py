#!/usr/bin/env python3
"""Validate the lightweight README inputs and formal before/after GIFs.

The public examples use full-resolution 3D-FUTURE input images and 640 px
turntables.  The manifest keeps those dimensions and the GIF timing contract
explicit so a preview cannot silently be replaced by a resized or grayscale
asset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageStat


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_media(manifest_path: Path, relative_path: str) -> Path:
    path = (manifest_path.parent / relative_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def validate_file(path: Path, record: dict[str, Any], size_key: str, hash_key: str) -> None:
    expected_size = int(record[size_key])
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(f"{path}: expected {expected_size} bytes, got {actual_size}")
    expected_hash = str(record[hash_key])
    actual_hash = sha256(path)
    if actual_hash != expected_hash:
        raise ValueError(f"{path}: SHA256 mismatch: {actual_hash}")


def frame_chroma(frame: Image.Image) -> float:
    red, green, blue = frame.split()
    return max(
        ImageStat.Stat(ImageChops.difference(red, green)).mean[0],
        ImageStat.Stat(ImageChops.difference(green, blue)).mean[0],
        ImageStat.Stat(ImageChops.difference(red, blue)).mean[0],
    )


def _allowed_durations(renderer: dict[str, Any]) -> set[int]:
    """Return the permitted per-frame durations, supporting old manifests."""
    values = renderer.get("frame_durations_ms")
    if values is None:
        values = renderer.get("frame_duration_ms")
    if isinstance(values, (int, float, str)):
        values = [values]
    if not isinstance(values, list) or not values:
        raise ValueError("renderer.frame_durations_ms must be a non-empty list")
    return {int(value) for value in values}


def _gif_size(renderer: dict[str, Any]) -> tuple[int, int]:
    """Read the GIF canvas size, retaining compatibility with v2 manifests."""
    values = renderer.get("gif_canvas_size", renderer.get("canvas_size"))
    if not isinstance(values, list) or len(values) != 2:
        raise ValueError("renderer.gif_canvas_size must contain width and height")
    return tuple(int(value) for value in values)


def validate_readme_references(manifest_path: Path, samples: list[dict[str, Any]]) -> None:
    """Ensure the root README embeds every public sample file."""
    repository_root = Path(__file__).resolve().parents[1]
    readme_path = repository_root / "README.md"
    text = readme_path.read_text(encoding="utf-8")
    missing: list[str] = []
    for sample in samples:
        for key in ("input", "before", "refined"):
            record = sample[key]
            relative = record.get("path", record.get("gif"))
            filename = Path(str(relative)).name
            reference = f"docs/assets/examples/{filename}"
            if reference not in text:
                missing.append(reference)
    if missing:
        raise ValueError(f"README.md is missing public media references: {', '.join(missing)}")


def validate_gif(path: Path, record: dict[str, Any], renderer: dict[str, Any]) -> None:
    validate_file(path, record, "gif_bytes", "gif_sha256")
    expected_frames = int(renderer["frames"])
    allowed_durations = _allowed_durations(renderer)
    expected_total = renderer.get("duration_ms")
    expected_size = _gif_size(renderer)

    with Image.open(path) as animation:
        if animation.n_frames != expected_frames:
            raise ValueError(f"{path}: expected {expected_frames} frames, got {animation.n_frames}")
        previous: Image.Image | None = None
        moving_pairs = 0
        max_chroma = 0.0
        durations: list[int] = []
        for index in range(animation.n_frames):
            animation.seek(index)
            duration = int(animation.info.get("duration", 0))
            durations.append(duration)
            if duration not in allowed_durations:
                raise ValueError(
                    f"{path}: frame {index} duration {duration} ms is not in "
                    f"{sorted(allowed_durations)}"
                )
            frame = animation.convert("RGB").copy()
            if frame.size != expected_size:
                raise ValueError(f"{path}: frame {index} expected {expected_size}, got {frame.size}")
            max_chroma = max(max_chroma, frame_chroma(frame))
            if previous is not None and ImageChops.difference(previous, frame).getbbox() is not None:
                moving_pairs += 1
            previous = frame

    if expected_total is not None and sum(durations) != int(expected_total):
        raise ValueError(
            f"{path}: expected total duration {int(expected_total)} ms, "
            f"got {sum(durations)} ms"
        )
    if moving_pairs != expected_frames - 1:
        raise ValueError(f"{path}: only {moving_pairs}/{expected_frames - 1} adjacent pairs move")
    if max_chroma < 1.0:
        raise ValueError(f"{path}: frames appear grayscale (maximum channel difference {max_chroma:.3f})")


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "docs" / "assets" / "examples" / "manifest.json",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest_path = args.manifest.resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    renderer = payload["renderer"]
    samples = payload["samples"]
    expected_ids = ("0000000", "0000008", "0000035")
    actual_ids = tuple(str(sample.get("id")) for sample in samples)
    if actual_ids != expected_ids:
        raise ValueError(f"expected README samples {expected_ids}, got {actual_ids}")
    input_size = renderer.get("input_size")
    if not isinstance(input_size, list) or len(input_size) != 2:
        raise ValueError("renderer.input_size must contain width and height")
    expected_input_size = tuple(int(value) for value in input_size)
    validate_readme_references(manifest_path, samples)

    for sample in samples:
        input_record = sample["input"]
        input_path = resolve_media(manifest_path, input_record["path"])
        validate_file(input_path, input_record, "bytes", "sha256")
        with Image.open(input_path) as image:
            if image.size != expected_input_size:
                raise ValueError(f"{input_path}: unexpected input preview size {image.size}")
            if image.mode != "RGB":
                raise ValueError(f"{input_path}: expected RGB input, got {image.mode}")
        for variant in ("before", "refined"):
            record = sample[variant]
            gif_path = resolve_media(manifest_path, record["gif"])
            validate_gif(gif_path, record, renderer)
        print(f"{sample['id']}: input + formal before/refined media PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
