"""Convert raw public layout-head outputs to scene T/R/S records."""

from __future__ import annotations

from typing import Any

import numpy as np

from .runtime_config import LayoutRuntimeConfig


def axis_conversion_matrix(source: str, target: str) -> np.ndarray:
    if source == target:
        return np.eye(3, dtype=np.float64)
    if (source, target) == ("z", "y"):
        return np.asarray(((1, 0, 0), (0, 0, 1), (0, -1, 0)), dtype=np.float64)
    if (source, target) == ("y", "z"):
        return np.asarray(((1, 0, 0), (0, 0, -1), (0, 1, 0)), dtype=np.float64)
    raise ValueError(f"Unsupported up-axis conversion: {source!r} to {target!r}")


def corner_fit_yaw(raw_rotation: np.ndarray, up_axis: str) -> np.ndarray:
    if up_axis == "y":
        sine = raw_rotation[..., 2, 0] - raw_rotation[..., 0, 2]
        cosine = raw_rotation[..., 0, 0] + raw_rotation[..., 2, 2]
    elif up_axis == "z":
        sine = raw_rotation[..., 1, 0] - raw_rotation[..., 0, 1]
        cosine = raw_rotation[..., 0, 0] + raw_rotation[..., 1, 1]
    else:
        raise ValueError(f"Unsupported up axis: {up_axis!r}")
    return np.arctan2(sine, cosine)


def yaw_rotation(yaw: np.ndarray, up_axis: str) -> np.ndarray:
    rotation = np.zeros((*yaw.shape, 3, 3), dtype=np.float64)
    cosine, sine = np.cos(yaw), np.sin(yaw)
    if up_axis == "y":
        rotation[..., 0, 0] = cosine
        rotation[..., 0, 2] = -sine
        rotation[..., 1, 1] = 1
        rotation[..., 2, 0] = sine
        rotation[..., 2, 2] = cosine
    elif up_axis == "z":
        rotation[..., 0, 0] = cosine
        rotation[..., 0, 1] = -sine
        rotation[..., 1, 0] = sine
        rotation[..., 1, 1] = cosine
        rotation[..., 2, 2] = 1
    else:
        raise ValueError(f"Unsupported up axis: {up_axis!r}")
    return rotation


def postprocess_predictions(predictions: dict[str, Any], config: LayoutRuntimeConfig) -> list[dict[str, Any]]:
    arrays = {
        key: np.asarray(value.detach().float().cpu(), dtype=np.float64)
        for key, value in predictions.items()
    }
    required = {"translation", "rotation", "scaling"}
    if set(arrays) != required:
        raise ValueError(f"Layout output keys must be {sorted(required)}, got {sorted(arrays)}")
    translation, raw_rotation, scaling = arrays["translation"], arrays["rotation"], arrays["scaling"]
    if translation.ndim != 2 or translation.shape[1:] != (3,):
        raise ValueError(f"translation output must have shape (N, 3), got {translation.shape}")
    if raw_rotation.shape != (translation.shape[0], 3, 3):
        raise ValueError(f"rotation output must have shape (N, 3, 3), got {raw_rotation.shape}")
    if scaling.shape != (translation.shape[0], 1):
        raise ValueError(f"scaling output must have shape (N, 1), got {scaling.shape}")
    if not all(np.all(np.isfinite(value)) for value in arrays.values()):
        raise ValueError("Layout model returned a non-finite value")
    if np.any(scaling <= 0):
        raise ValueError("Layout model returned a non-positive scale")

    yaw = corner_fit_yaw(raw_rotation, config.checkpoint_up_axis)
    rotation = yaw_rotation(yaw, config.checkpoint_up_axis)
    conversion = axis_conversion_matrix(config.checkpoint_up_axis, config.output_up_axis)
    translation = translation @ conversion.T
    rotation = conversion @ rotation @ conversion.T
    return [
        {
            "translation": translation[index].tolist(),
            "rotation": rotation[index].tolist(),
            "scaling": float(scaling[index, 0]),
        }
        for index in range(translation.shape[0])
    ]
