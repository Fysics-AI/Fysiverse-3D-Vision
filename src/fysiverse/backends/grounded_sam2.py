"""Grounding DINO plus SAM 2 worker for automatic instance masks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from fysiverse.preparation import palette


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = int(np.logical_and(first, second).sum())
    if not intersection:
        return 0.0
    return intersection / float(max(int(np.logical_or(first, second).sum()), 1))


def mask_containment(first: np.ndarray, second: np.ndarray) -> float:
    intersection = int(np.logical_and(first, second).sum())
    if not intersection:
        return 0.0
    smaller_area = min(int(first.sum()), int(second.sum()))
    return intersection / float(max(smaller_area, 1))


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_root = args.source_root.resolve()
    _require_file(args.image.resolve(), "input image")
    _require_file(args.sam2_checkpoint.resolve(), "SAM 2 checkpoint")
    _require_file(args.gdino_checkpoint.resolve(), "Grounding DINO checkpoint")
    _require_file(args.gdino_config.resolve(), "Grounding DINO config")
    text_encoder = args.text_encoder.resolve()
    _require_file(text_encoder / "config.json", "BERT config")
    _require_file(text_encoder / "model.safetensors", "BERT weights")
    if not source_root.is_dir():
        raise FileNotFoundError(f"Grounded-SAM2 source directory not found: {source_root}")
    sys.path.insert(0, str(source_root))

    import torch
    from grounding_dino.groundingdino.models import build_model as build_grounding_model
    from grounding_dino.groundingdino.util.inference import load_image, predict
    from grounding_dino.groundingdino.util.misc import clean_state_dict
    from grounding_dino.groundingdino.util.slconfig import SLConfig
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot see a CUDA device")

    sam2_model = build_sam2(args.sam2_config, str(args.sam2_checkpoint), device=args.device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    grounding_args = SLConfig.fromfile(str(args.gdino_config))
    grounding_args.device = args.device
    grounding_args.text_encoder_type = str(text_encoder)
    grounding_model = build_grounding_model(grounding_args)
    grounding_checkpoint = torch.load(str(args.gdino_checkpoint), map_location="cpu")
    grounding_model.load_state_dict(
        clean_state_dict(grounding_checkpoint["model"]), strict=False
    )
    grounding_model.eval()

    image_source, transformed = load_image(str(args.image))
    sam2_predictor.set_image(image_source)
    boxes, confidences, labels = predict(
        model=grounding_model,
        image=transformed,
        caption=args.text_prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device=args.device,
    )
    if len(boxes) == 0:
        raise RuntimeError("Grounding DINO found no objects; change the prompt or lower the thresholds")

    height, width = image_source.shape[:2]
    boxes = boxes * torch.tensor([width, height, width, height])
    cx, cy, box_width, box_height = boxes.unbind(dim=1)
    input_boxes = torch.stack(
        [cx - box_width / 2, cy - box_height / 2, cx + box_width / 2, cy + box_height / 2], dim=1
    ).numpy()
    masks, _, _ = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )
    if masks.ndim == 2:
        masks = masks[None, ...]
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    output_dir = args.output.resolve()
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)
    accepted_masks: list[np.ndarray] = []
    objects: list[dict[str, Any]] = []
    candidates = sorted(
        zip(masks, labels, confidences.tolist(), input_boxes.tolist()),
        key=lambda item: float(item[2]),
        reverse=True,
    )
    for mask, label, confidence, box in candidates:
        binary = np.asarray(mask, dtype=bool)
        area = int(binary.sum())
        if area < args.min_mask_area:
            continue
        if any(
            mask_iou(binary, previous) >= args.dedupe_iou
            or mask_containment(binary, previous) >= args.dedupe_containment
            for previous in accepted_masks
        ):
            continue
        index = len(objects)
        color = palette(index)
        Image.fromarray(np.where(binary, 255, 0).astype(np.uint8), mode="L").save(masks_dir / f"{index}.png")
        unassigned = binary & np.all(color_mask == 0, axis=-1)
        color_mask[unassigned] = np.asarray(color, dtype=np.uint8)
        accepted_masks.append(binary)
        objects.append(
            {
                "id": str(index),
                "class_name": str(label),
                "score": float(confidence),
                "bbox": [float(value) for value in box],
                "area": area,
                "mask": f"masks/{index}.png",
                "color": list(color),
            }
        )

    if not objects:
        raise RuntimeError("Grounded-SAM2 produced no mask above the minimum area")
    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(color_mask, mode="RGB").save(output_dir / "mask.png")
    metadata = {
        "version": "1.0",
        "backend": "grounded_sam2",
        "image": str(args.image.resolve()),
        "text_prompt": args.text_prompt,
        "objects": objects,
    }
    (output_dir / "mask_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[3]
    source_root = project_root / "third_party" / "src" / "Grounded-SAM-2"
    model_root = project_root / "models" / "grounded_sam2"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=source_root)
    parser.add_argument("--text-prompt", default="sofa. bed. chair. ceiling light. table. cabinet. shelf. desk. stool.")
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--min-mask-area", type=int, default=100)
    parser.add_argument("--dedupe-iou", type=float, default=0.9)
    parser.add_argument(
        "--dedupe-containment",
        type=float,
        default=0.95,
        help="Reject a lower-confidence mask when this fraction of the smaller mask is contained in an accepted mask",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--text-encoder",
        type=Path,
        default=project_root / "models" / "bert_base_uncased",
        help="Local bert-base-uncased directory used without network access",
    )
    parser.add_argument("--sam2-checkpoint", type=Path, default=model_root / "checkpoints" / "sam2.1_hiera_large.pt")
    parser.add_argument("--sam2-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument(
        "--gdino-config",
        type=Path,
        default=source_root / "grounding_dino" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py",
    )
    parser.add_argument(
        "--gdino-checkpoint",
        type=Path,
        default=model_root / "gdino_checkpoints" / "groundingdino_swint_ogc.pth",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args)
    print(json.dumps({"status": "ok", "objects": len(result["objects"]), "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
