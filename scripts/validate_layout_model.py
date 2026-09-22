#!/usr/bin/env python3
"""Validate the public Hugging Face G2VLM + Layout inference package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MODEL_TYPE = "fysiverse-g2vlm-layout"
BASE_REPO_ID = "InternRobotics/G2VLM-2B-MoT"
BASE_REVISION = "4e75aa3b47695d543fc9cede09bb9ab4754149a9"
BACKBONE_TENSOR_COUNT = 1116
LAYOUT_TENSOR_COUNT = 146
REQUIRED_LAYOUT_PREFIXES = ("condition2llm.", "layout_decoder.", "layout_head.")
FORBIDDEN_KEY_PARTS = (
    "optimizer",
    "epoch",
    "step",
    "loss_history",
    "training",
    "source_checkpoint",
)


def find_absolute_path(value: Any, location: str = "config") -> str | None:
    if isinstance(value, dict):
        for key, item in value.items():
            found = find_absolute_path(item, f"{location}.{key}")
            if found:
                return found
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found = find_absolute_path(item, f"{location}[{index}]")
            if found:
                return found
    elif isinstance(value, str) and Path(value).is_absolute():
        return location
    return None


def find_forbidden_key(value: Any, location: str = "config") -> str | None:
    if isinstance(value, dict):
        for key, item in value.items():
            key_location = f"{location}.{key}"
            if any(part in str(key).lower() for part in FORBIDDEN_KEY_PARTS):
                return key_location
            found = find_forbidden_key(item, key_location)
            if found:
                return found
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found = find_forbidden_key(item, f"{location}[{index}]")
            if found:
                return found
    return None


def _mapping(config: dict[str, Any], key: str, errors: list[str]) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict):
        errors.append(f"{key} must be an object")
        return {}
    return value


def validate_config(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if config.get("model_type") != MODEL_TYPE:
        errors.append(f"model_type must be {MODEL_TYPE!r}")
    if config.get("format_version") != 1:
        errors.append("format_version must be 1")

    base_model = _mapping(config, "base_model", errors)
    backbone = _mapping(config, "backbone", errors)
    layout = _mapping(config, "layout", errors)
    inference = _mapping(config, "inference", errors)

    expected_base = {
        "repo_id": BASE_REPO_ID,
        "revision": BASE_REVISION,
        "weights": "model.safetensors",
    }
    for key, expected in expected_base.items():
        if base_model.get(key) != expected:
            errors.append(f"base_model.{key} must be {expected!r}")

    if backbone.get("type") != "replacement_state_dict":
        errors.append("backbone.type must be 'replacement_state_dict'")
    if backbone.get("index") != "backbone.safetensors.index.json":
        errors.append("backbone.index must be 'backbone.safetensors.index.json'")
    if backbone.get("tensor_count") != BACKBONE_TENSOR_COUNT:
        errors.append(f"backbone.tensor_count must be {BACKBONE_TENSOR_COUNT}")

    if layout.get("weights") != "layout.safetensors":
        errors.append("layout.weights must be 'layout.safetensors'")
    if layout.get("tensor_count") != LAYOUT_TENSOR_COUNT:
        errors.append(f"layout.tensor_count must be {LAYOUT_TENSOR_COUNT}")
    if layout.get("sections") != ["condition2llm", "layout_decoder", "layout_head"]:
        errors.append("layout.sections must list condition2llm, layout_decoder, and layout_head")

    expected_inference = {
        "loss_name": "raw-mse",
        "rotation_dim": 9,
        "layout_heads": ["translation", "rotation", "scaling"],
        "checkpoint_up_axis": "z",
        "default_scene_up_axis": "y",
        "default_rotation_postprocess": "corner-yaw-only",
    }
    for key, expected in expected_inference.items():
        if inference.get(key) != expected:
            errors.append(f"inference.{key} must be {expected!r}")

    absolute = find_absolute_path(config)
    if absolute:
        errors.append(f"absolute path is not allowed at {absolute}")
    forbidden = find_forbidden_key(config)
    if forbidden:
        errors.append(f"private or unsupported field is not allowed at {forbidden}")
    return errors


def _safe_tensor_header(path: Path) -> tuple[set[str], dict[str, str], str | None]:
    try:
        from safetensors import safe_open
    except ImportError:
        return set(), {}, "safetensors is required to inspect model package headers"
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            return set(handle.keys()), handle.metadata() or {}, None
    except Exception as exc:
        return set(), {}, f"cannot inspect {path.name}: {exc}"


def _validate_metadata(path: Path, metadata: dict[str, str]) -> list[str]:
    errors = []
    for key, value in metadata.items():
        if any(part in key.lower() for part in FORBIDDEN_KEY_PARTS):
            errors.append(f"{path.name} contains forbidden metadata key: {key}")
            break
        if Path(str(value)).is_absolute():
            errors.append(f"{path.name} contains an absolute metadata path in: {key}")
            break
    return errors


def validate_model(model_dir: Path) -> list[str]:
    config_path = model_dir / "config.json"
    errors: list[str] = []
    if not config_path.is_file():
        return [f"missing config: {config_path}"]

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read config: {exc}"]
    if not isinstance(config, dict):
        return ["config.json must contain a JSON object"]
    errors.extend(validate_config(config))

    backbone = config.get("backbone") if isinstance(config.get("backbone"), dict) else {}
    layout = config.get("layout") if isinstance(config.get("layout"), dict) else {}
    index_name = str(backbone.get("index") or "backbone.safetensors.index.json")
    layout_name = str(layout.get("weights") or "layout.safetensors")
    index_path = model_dir / index_name
    layout_path = model_dir / layout_name
    if not index_path.is_file():
        errors.append(f"missing backbone index: {index_path}")
    if not layout_path.is_file():
        errors.append(f"missing layout weights: {layout_path}")
    if errors and (not index_path.is_file() or not layout_path.is_file()):
        return errors

    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"cannot read backbone index: {exc}")
        return errors
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        errors.append("backbone index weight_map must be a non-empty object")
        return errors
    normalized_map = {str(key): str(value) for key, value in weight_map.items()}
    if len(normalized_map) != BACKBONE_TENSOR_COUNT:
        errors.append(
            f"backbone index must contain {BACKBONE_TENSOR_COUNT} tensors, found {len(normalized_map)}"
        )
    index_metadata = index.get("metadata")
    if isinstance(index_metadata, dict) and index_metadata.get("tensor_count") != len(normalized_map):
        errors.append("backbone index metadata.tensor_count does not match weight_map")

    indexed_shards = sorted(set(normalized_map.values()))
    if not indexed_shards:
        errors.append("backbone index references no shards")
    for shard_name in indexed_shards:
        shard_relative = Path(shard_name)
        if shard_relative.is_absolute() or ".." in shard_relative.parts or shard_relative.name != shard_name:
            errors.append(f"invalid backbone shard path: {shard_name}")
            continue
        shard_path = model_dir / shard_name
        if not shard_path.is_file():
            errors.append(f"missing backbone shard: {shard_path}")
            continue
        actual_keys, metadata, header_error = _safe_tensor_header(shard_path)
        if header_error:
            errors.append(header_error)
            continue
        expected_keys = {key for key, value in normalized_map.items() if value == shard_name}
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            unexpected = sorted(actual_keys - expected_keys)
            if missing:
                errors.append(f"{shard_name} is missing indexed tensor: {missing[0]}")
            if unexpected:
                errors.append(f"{shard_name} contains unindexed tensor: {unexpected[0]}")
        errors.extend(_validate_metadata(shard_path, metadata))

    layout_keys, layout_metadata, header_error = _safe_tensor_header(layout_path)
    if header_error:
        errors.append(header_error)
        return errors
    if len(layout_keys) != LAYOUT_TENSOR_COUNT:
        errors.append(
            f"layout weights must contain {LAYOUT_TENSOR_COUNT} tensors, found {len(layout_keys)}"
        )
    for prefix in REQUIRED_LAYOUT_PREFIXES:
        if not any(key.startswith(prefix) for key in layout_keys):
            errors.append(f"layout weights contain no keys with required prefix {prefix}")
    unexpected_layout = sorted(
        key for key in layout_keys if not key.startswith(REQUIRED_LAYOUT_PREFIXES)
    )
    if unexpected_layout:
        errors.append(f"layout weights contain unexpected tensor key: {unexpected_layout[0]}")
    errors.extend(_validate_metadata(layout_path, layout_metadata))
    return errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models" / "layout")
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional machine-readable validation report path.",
    )
    return parser


def build_report(model_dir: Path, errors: list[str]) -> dict[str, Any]:
    files = []
    if model_dir.is_dir():
        for path in sorted(item for item in model_dir.iterdir() if item.is_file()):
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            files.append(
                {
                    "name": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": digest.hexdigest(),
                }
            )
    return {
        "status": "pass" if not errors else "fail",
        "model_type": MODEL_TYPE,
        "model_dir": str(model_dir),
        "base_model": {"repo_id": BASE_REPO_ID, "revision": BASE_REVISION},
        "backbone_tensor_count": BACKBONE_TENSOR_COUNT,
        "layout_tensor_count": LAYOUT_TENSOR_COUNT,
        "package_size_bytes": sum(item["size_bytes"] for item in files),
        "files": files,
        "errors": errors,
    }


def main() -> int:
    args = build_parser().parse_args()
    model_dir = args.model_dir.resolve()
    errors = validate_model(model_dir)
    if args.json_output is not None:
        output_path = args.json_output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(build_report(model_dir, errors), indent=2) + "\n",
            encoding="utf-8",
        )
    if errors:
        print("[layout-model] validation failed:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print(f"[layout-model] ready: {model_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
