"""Validated configuration for the Hugging Face layout model package."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


LOSS_RAW_MSE = "raw_mse"
LAYOUT_HEAD_TRANSLATION = "translation"
LAYOUT_HEAD_ROTATION = "rotation"
LAYOUT_HEAD_SCALING = "scaling"
SUPPORTED_LAYOUT_HEADS = (
    LAYOUT_HEAD_TRANSLATION,
    LAYOUT_HEAD_ROTATION,
    LAYOUT_HEAD_SCALING,
)


def validate_layout_loss_name(loss_name: str) -> str:
    if loss_name != LOSS_RAW_MSE:
        raise ValueError(f"Public inference supports only {LOSS_RAW_MSE!r}, got {loss_name!r}")
    return loss_name


def normalize_layout_heads(layout_heads: Sequence[str] | str | None) -> tuple[str, ...]:
    if layout_heads is None:
        return SUPPORTED_LAYOUT_HEADS
    if isinstance(layout_heads, str):
        values = tuple(part.strip() for part in layout_heads.split(",") if part.strip())
    else:
        values = tuple(str(part) for part in layout_heads)
    if values != SUPPORTED_LAYOUT_HEADS:
        raise ValueError(f"layout_heads must be {list(SUPPORTED_LAYOUT_HEADS)!r}, got {list(values)!r}")
    return values


def resolve_layout_rotation_dim(rotation_dim: int | None, *, loss_name: str) -> int:
    validate_layout_loss_name(loss_name)
    resolved = 9 if rotation_dim is None else int(rotation_dim)
    if resolved != 9:
        raise ValueError(f"Public inference requires a 9D rotation matrix head, got {resolved}")
    return resolved


@dataclass(frozen=True)
class LayoutRuntimeConfig:
    model_type: str
    base_model_repo_id: str
    base_model_revision: str
    base_model_weights: str
    backbone_index: str
    backbone_tensor_count: int
    weights: str
    layout_tensor_count: int
    layout_heads: tuple[str, ...]
    checkpoint_up_axis: str
    output_up_axis: str
    rotation_postprocess: str
    scene_units: str
    normalize_scene: bool
    normalization_margin: float
    prompt: str
    control_padding_ratio: float

    @classmethod
    def from_file(cls, path: Path) -> "LayoutRuntimeConfig":
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("model_type") != "fysiverse-g2vlm-layout":
            raise ValueError("layout config model_type must be 'fysiverse-g2vlm-layout'")
        if data.get("format_version") != 1:
            raise ValueError("layout config format_version must be 1")

        base_model = _require_mapping(data.get("base_model"), "base_model")
        backbone = _require_mapping(data.get("backbone"), "backbone")
        layout = _require_mapping(data.get("layout"), "layout")
        inference = _require_mapping(data.get("inference"), "inference")
        if backbone.get("type") != "replacement_state_dict":
            raise ValueError("layout config backbone.type must be 'replacement_state_dict'")

        base_model_weights = _validate_relative_file(base_model.get("weights"), "base_model.weights")
        backbone_index = _validate_relative_file(backbone.get("index"), "backbone.index")
        weights = _validate_relative_file(layout.get("weights"), "layout.weights")
        backbone_tensor_count = _validate_positive_int(
            backbone.get("tensor_count"), "backbone.tensor_count"
        )
        layout_tensor_count = _validate_positive_int(
            layout.get("tensor_count"), "layout.tensor_count"
        )
        checkpoint_up_axis = _validate_up_axis(
            inference.get("checkpoint_up_axis"), "inference.checkpoint_up_axis"
        )
        output_up_axis = _validate_up_axis(
            inference.get("default_scene_up_axis", "y"), "inference.default_scene_up_axis"
        )
        rotation_postprocess = str(inference.get("default_rotation_postprocess", ""))
        if rotation_postprocess != "corner-yaw-only":
            raise ValueError("layout config default_rotation_postprocess must be 'corner-yaw-only'")
        if inference.get("loss_name") != "raw-mse":
            raise ValueError("layout config inference.loss_name must be 'raw-mse'")
        if int(inference.get("rotation_dim", 0)) != 9:
            raise ValueError("layout config inference.rotation_dim must be 9")
        normalization = inference.get("scene_normalization", {})
        margin = float(normalization.get("margin", 0.02))
        if not 0 <= margin < 1:
            raise ValueError("scene normalization margin must be in [0, 1)")
        return cls(
            model_type=str(data["model_type"]),
            base_model_repo_id=str(base_model.get("repo_id", "")),
            base_model_revision=str(base_model.get("revision", "")),
            base_model_weights=base_model_weights,
            backbone_index=backbone_index,
            backbone_tensor_count=backbone_tensor_count,
            weights=weights,
            layout_tensor_count=layout_tensor_count,
            layout_heads=normalize_layout_heads(inference.get("layout_heads")),
            checkpoint_up_axis=checkpoint_up_axis,
            output_up_axis=output_up_axis,
            rotation_postprocess=rotation_postprocess,
            scene_units="normalized_scene_units",
            normalize_scene=bool(normalization.get("enabled", True)),
            normalization_margin=margin,
            prompt=str(inference.get("default_prompt", "Predict the masked object's layout in the scene.")),
            control_padding_ratio=float(inference.get("control_padding_ratio", 0.15)),
        )


def _validate_up_axis(value: object, name: str) -> str:
    axis = str(value)
    if axis not in {"y", "z"}:
        raise ValueError(f"layout config {name} must be 'y' or 'z', got {axis!r}")
    return axis


def _require_mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"layout config {name} must be an object")
    return value


def _validate_relative_file(value: object, name: str) -> str:
    text = str(value or "")
    path = Path(text)
    if not text or path.is_absolute() or ".." in path.parts or path.name != text:
        raise ValueError(f"layout config {name} must be a package-root filename")
    return text


def _validate_positive_int(value: object, name: str) -> int:
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"layout config {name} must be a positive integer") from exc
    if resolved <= 0:
        raise ValueError(f"layout config {name} must be a positive integer")
    return resolved
