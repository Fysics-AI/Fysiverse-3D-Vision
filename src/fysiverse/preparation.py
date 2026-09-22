"""Normalize a scene image and instance mask for the public inference pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def palette(index: int) -> tuple[int, int, int]:
    hue = (index * 137.508) % 360.0
    c = 255.0
    x = c * (1.0 - abs((hue / 60.0) % 2.0 - 1.0))
    if hue < 60:
        rgb = (c, x, 0)
    elif hue < 120:
        rgb = (x, c, 0)
    elif hue < 180:
        rgb = (0, c, x)
    elif hue < 240:
        rgb = (0, x, c)
    elif hue < 300:
        rgb = (x, 0, c)
    else:
        rgb = (c, 0, x)
    return tuple(int(round(value)) for value in rgb)


def connected_components(binary: np.ndarray, min_area: int, connectivity: int = 8) -> list[dict[str, Any]]:
    if binary.ndim != 2:
        raise ValueError(f"binary mask must have 2 dimensions, got {binary.shape}")
    try:
        import cv2

        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary.astype(np.uint8), connectivity=connectivity
        )
        components = []
        for label in range(1, count):
            x, y, width, height, area = (int(value) for value in stats[label])
            if area >= min_area:
                components.append(
                    {"mask": labels == label, "bbox": [x, y, width, height], "area": area}
                )
        return components
    except ImportError:
        from scipy import ndimage

        structure = np.ones((3, 3), dtype=np.uint8) if connectivity == 8 else None
        labels, count = ndimage.label(binary, structure=structure)
        components = []
        for label in range(1, int(count) + 1):
            ys, xs = np.where(labels == label)
            if xs.size < min_area:
                continue
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            components.append(
                {
                    "mask": labels == label,
                    "bbox": [x0, y0, x1 - x0, y1 - y0],
                    "area": int(xs.size),
                }
            )
        return components


def load_components(
    mask_path: Path,
    *,
    mode: str = "auto",
    threshold: int = 10,
    min_area: int = 100,
    connectivity: int = 8,
    max_instance_colors: int = 256,
) -> tuple[tuple[int, int], list[dict[str, Any]]]:
    if mode not in {"auto", "color", "connected"}:
        raise ValueError(f"unsupported mask mode: {mode}")
    with Image.open(mask_path) as source:
        rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
        alpha = np.asarray(source.getchannel("A"), dtype=np.uint8) if "A" in source.getbands() else None

    foreground = alpha > threshold if alpha is not None and int(alpha.max()) > 0 else rgb.max(axis=-1) > threshold
    pixels = rgb[foreground]
    colored = bool(pixels.size) and not bool(np.all(pixels[:, 0] == pixels[:, 1]) and np.all(pixels[:, 1] == pixels[:, 2]))
    colors = np.unique(pixels, axis=0) if colored else np.empty((0, 3), dtype=np.uint8)
    use_colors = mode == "color" or (mode == "auto" and colored and len(colors) <= max_instance_colors)

    components: list[dict[str, Any]] = []
    if use_colors:
        if len(colors) > max_instance_colors:
            raise ValueError(
                f"mask has {len(colors)} foreground colors; expected at most {max_instance_colors}. "
                "Use --mask-mode connected for an antialiased or photographic mask."
            )
        for color in colors:
            color_mask = foreground & np.all(rgb == color, axis=-1)
            for component in connected_components(color_mask, min_area, connectivity):
                component["source_color"] = [int(value) for value in color]
                components.append(component)
    else:
        components = connected_components(foreground, min_area, connectivity)

    components.sort(key=lambda item: (item["bbox"][1], item["bbox"][0], -item["area"]))
    return (rgb.shape[1], rgb.shape[0]), components


def _metadata_by_color(metadata_path: Path | None) -> dict[tuple[int, int, int], dict[str, Any]]:
    if metadata_path is None or not metadata_path.is_file():
        return {}
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    result = {}
    for item in data.get("objects", []):
        color = item.get("color")
        if isinstance(color, list) and len(color) == 3:
            result[tuple(int(value) for value in color)] = item
    return result


def _metadata_by_id(metadata_path: Path | None) -> dict[str, dict[str, Any]]:
    if metadata_path is None or not metadata_path.is_file():
        return {}
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {str(item["id"]): item for item in data.get("objects", []) if "id" in item}


def load_individual_components(
    masks_dir: Path,
    *,
    threshold: int,
    min_area: int,
) -> tuple[tuple[int, int], list[dict[str, Any]]]:
    paths = sorted(
        masks_dir.glob("*.png"),
        key=lambda path: (0, int(path.stem)) if path.stem.isdigit() else (1, path.name),
    )
    if not paths:
        raise FileNotFoundError(f"no PNG instance masks found in {masks_dir}")
    components = []
    expected_size: tuple[int, int] | None = None
    for path in paths:
        with Image.open(path) as source:
            mask = np.asarray(source.convert("L"), dtype=np.uint8) > threshold
            size = source.size
        if expected_size is None:
            expected_size = size
        elif size != expected_size:
            raise ValueError(f"instance mask size mismatch: {path} is {size}, expected {expected_size}")
        ys, xs = np.where(mask)
        if xs.size < min_area:
            continue
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        components.append(
            {
                "mask": mask,
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "area": int(xs.size),
                "source_id": path.stem,
            }
        )
    assert expected_size is not None
    return expected_size, components


def prepare_inputs(
    image_path: Path,
    mask_path: Path,
    output_dir: Path,
    *,
    mask_backend: str = "provided",
    metadata_path: Path | None = None,
    instance_masks_dir: Path | None = None,
    mask_mode: str = "auto",
    threshold: int = 10,
    min_area: int = 100,
    connectivity: int = 8,
) -> dict[str, Any]:
    image_path = image_path.resolve()
    mask_path = mask_path.resolve()
    output_dir = output_dir.resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    if not mask_path.is_file():
        raise FileNotFoundError(mask_path)

    with Image.open(image_path) as source:
        scene = source.convert("RGB")
    scene_rgb = np.asarray(scene, dtype=np.uint8)
    if instance_masks_dir is not None:
        mask_size, components = load_individual_components(
            instance_masks_dir.resolve(), threshold=threshold, min_area=min_area
        )
    else:
        mask_size, components = load_components(
            mask_path,
            mode=mask_mode,
            threshold=threshold,
            min_area=min_area,
            connectivity=connectivity,
        )
    if mask_size != scene.size:
        raise ValueError(f"mask/image size mismatch: mask={mask_size}, image={scene.size}")
    if not components:
        raise RuntimeError(f"no mask instance with at least {min_area} pixels was found in {mask_path}")

    masks_dir = output_dir / "masks"
    cutouts_dir = output_dir / "work" / "cutouts"
    assets_dir = output_dir / "object_assets"
    for directory in (masks_dir, cutouts_dir, assets_dir):
        directory.mkdir(parents=True, exist_ok=True)

    metadata = _metadata_by_color(metadata_path)
    metadata_ids = _metadata_by_id(metadata_path)
    normalized_mask = np.zeros_like(scene_rgb, dtype=np.uint8)
    objects = []
    for index, component in enumerate(components):
        object_id = str(index)
        binary = component["mask"]
        color = palette(index)
        normalized_mask[binary] = np.asarray(color, dtype=np.uint8)

        mask_output = masks_dir / f"{object_id}.png"
        Image.fromarray(np.where(binary, 255, 0).astype(np.uint8), mode="L").save(mask_output)

        rgba = np.zeros((*binary.shape, 4), dtype=np.uint8)
        rgba[..., :3] = np.where(binary[..., None], scene_rgb, 0)
        rgba[..., 3] = np.where(binary, 255, 0).astype(np.uint8)
        cutout_output = cutouts_dir / f"{object_id}.png"
        Image.fromarray(rgba, mode="RGBA").save(cutout_output)

        source_color = tuple(component.get("source_color", []))
        source_metadata = metadata_ids.get(str(component.get("source_id", "")), metadata.get(source_color, {}))
        objects.append(
            {
                "id": object_id,
                "category": str(source_metadata.get("class_name", "object")),
                "score": source_metadata.get("score"),
                "mask": f"masks/{object_id}.png",
                "cutout": f"work/cutouts/{object_id}.png",
                "asset": f"object_assets/{object_id}/model.glb",
                "bbox": component["bbox"],
                "area": component["area"],
                "color": list(color),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(normalized_mask, mode="RGB").save(output_dir / "mask.png")
    manifest = {
        "version": "1.0",
        "image": str(image_path),
        "source_mask": str(mask_path),
        "mask_backend": mask_backend,
        "objects": objects,
        "outputs": {"mask": "mask.png", "poses": "poses.json", "scene": "scene.glb"},
        "coordinate_system": {
            "units": "normalized_scene_units",
            "up_axis": "y",
            "rotation": "3x3 row-major matrix",
            "scaling": "dimensionless uniform scale",
        },
        "stages": {"prepare": "complete", "flux": "pending", "trellis2": "pending", "layout": "pending"},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
