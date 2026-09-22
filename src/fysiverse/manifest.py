"""Helpers for reading and atomically updating a run manifest."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def load_manifest(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != "1.0" or not isinstance(data.get("objects"), list):
        raise ValueError(f"unsupported or invalid manifest: {path}")
    return data


def resolve_artifact(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (manifest_path.resolve().parent / path).resolve()


def relative_artifact(manifest_path: Path, path: Path) -> str:
    return os.path.relpath(path.resolve(), manifest_path.resolve().parent)


def save_manifest(path: Path, data: dict[str, Any]) -> None:
    path = path.resolve()
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
