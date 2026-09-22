"""Command-line entry for public input and mask normalization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .preparation import prepare_inputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mask-backend", default="provided")
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--instance-masks-dir", type=Path)
    parser.add_argument("--mask-mode", choices=("auto", "color", "connected"), default="auto")
    parser.add_argument("--mask-threshold", type=int, default=10)
    parser.add_argument("--min-mask-area", type=int, default=100)
    parser.add_argument("--connectivity", type=int, choices=(4, 8), default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = prepare_inputs(
        args.image,
        args.mask,
        args.output,
        mask_backend=args.mask_backend,
        metadata_path=args.metadata,
        instance_masks_dir=args.instance_masks_dir,
        mask_mode=args.mask_mode,
        threshold=args.mask_threshold,
        min_area=args.min_mask_area,
        connectivity=args.connectivity,
    )
    print(json.dumps({"status": "ok", "objects": len(manifest["objects"]), "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
