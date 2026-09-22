#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write the external dataset ground and separation plan.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--margin", type=float, default=0.01)
    parser.add_argument("--iters", type=int, default=32)
    parser.add_argument("--convex-hull-max-vertices", type=int, default=8000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = {
        "schema": "fysiverse_external_dataset_separation_plan.v1",
        "stage": {"key": "external_dataset_separated"},
        "transform": {
            "matrix_4x4": [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            "target_up": [0, 0, 1],
            "ground_y": 0.0,
        },
        "snap_bbox_to_ground": True,
        "bbox_snap_mode": "global_support_aware",
        "ground_y": 0.0,
        "support_adjust": False,
        "support_require_scene_graph": False,
        "separate_overlaps": True,
        "bbox_overlap_margin": float(args.margin),
        "bbox_overlap_iters": int(args.iters),
        "bbox_min_overlap_volume_ratio": 0.0,
        "bbox_overlap_eps": 1e-8,
        "overlap_collision_method": "convex_hull_sat",
        "convex_hull_max_vertices": int(args.convex_hull_max_vertices),
        "convex_decomposition_method": "single_convex_hull",
        "convex_collision_eps": 1e-6,
        "convex_min_horizontal_axis": 0.25,
        "support_gap": 0.001,
        "support_xy_overlap_ratio": 0.15,
        "support_max_gap_ratio": 0.03,
        "support_max_penetration_ratio": 0.01,
        "support_min_lower_area_ratio": 0.25,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(json.dumps(plan, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

