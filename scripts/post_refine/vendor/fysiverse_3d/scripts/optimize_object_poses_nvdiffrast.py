#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import multiprocessing as mp
from pathlib import Path
from typing import Any

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize each object's position, yaw, and uniform scale with nvdiffrast silhouette alignment."
    )
    parser.add_argument("--mesh", required=True, type=Path, help="NPZ from blender_export_nvdiffrast_mesh.py")
    parser.add_argument("--target-mask", required=True, type=Path, help="2D label mask; pixel value equals mask_id.")
    parser.add_argument("--final-manifest", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--preview-dir", type=Path, default=None)
    parser.add_argument("--input-image", type=Path, default=None)
    parser.add_argument("--camera-optimization", type=Path, default=None)
    parser.add_argument("--camera-report", type=Path, default=None)
    parser.add_argument("--stage-report", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=0, help="<=0 means one worker per object.")
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--position-steps", type=int, default=120)
    parser.add_argument("--yaw-steps", type=int, default=80)
    parser.add_argument("--scale-steps", type=int, default=60)
    parser.add_argument("--position-lr", type=float, default=0.01)
    parser.add_argument("--yaw-lr", type=float, default=0.02)
    parser.add_argument("--scale-lr", type=float, default=0.01)
    parser.add_argument("--dt-weight", type=float, default=0.25)
    parser.add_argument("--center-weight", type=float, default=0.15, help="Differentiable silhouette center-of-mass alignment weight.")
    parser.add_argument("--area-weight", type=float, default=0.05, help="Differentiable silhouette area-ratio alignment weight.")
    parser.add_argument("--translation-prior-weight", type=float, default=0.005)
    parser.add_argument("--yaw-prior-weight", type=float, default=0.002)
    parser.add_argument("--scale-prior-weight", type=float, default=0.002)
    parser.add_argument("--translation-mode", choices=["camera_plane", "world_xy", "world_xyz"], default="camera_plane", help="Parameterization for position optimization. camera_plane avoids depth/scale ambiguity.")
    parser.add_argument("--translation-initial-grid", type=int, default=3, help="Evaluate an NxN camera-plane/world-XY translation seed grid before gradient position refinement; <=1 disables it.")
    parser.add_argument("--yaw-initial-samples", type=int, default=5, help="Evaluate this many yaw seeds in [-max_yaw, max_yaw] before gradient yaw refinement.")
    parser.add_argument("--scale-initial-samples", type=int, default=5, help="Evaluate this many scale seeds in [1-max_delta, 1+max_delta] before gradient scale refinement.")
    parser.add_argument("--phase-accept-iou-drop", type=float, default=0.01, help="Reject a phase update if its IoU drops more than this from the previous accepted phase.")
    parser.add_argument("--max-translation", type=float, default=0.35)
    parser.add_argument("--max-yaw-deg", type=float, default=35.0)
    parser.add_argument("--max-scale-delta", type=float, default=0.25, help="Uniform scale is clamped to 1 +/- this value.")
    parser.add_argument("--converge-min-delta", type=float, default=1e-5)
    parser.add_argument("--converge-patience", type=int, default=12)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--no-antialias", action="store_true")
    parser.add_argument("--accept-worse-iou", action="store_true")
    parser.add_argument("--object-name", action="append", default=[], help="Optional object name filter; may be repeated.")
    return parser.parse_args()


def read_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def matrix3(values: Any, fallback: np.ndarray | None = None) -> np.ndarray:
    if values is None:
        if fallback is None:
            raise ValueError("missing 3x3 matrix")
        return fallback.copy()
    return np.asarray(values, dtype=np.float64).reshape(3, 3)


def matrix4(values: Any, fallback: np.ndarray | None = None) -> np.ndarray:
    if values is None:
        if fallback is None:
            raise ValueError("missing 4x4 matrix")
        return fallback.copy()
    return np.asarray(values, dtype=np.float64).reshape(4, 4)


def normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError(f"cannot normalize vector {vec}")
    return vec / norm


def raw_camera_to_blender_matrix(report: dict[str, Any]) -> np.ndarray:
    moge = report.get("moge") or {}
    scene_to_blender = matrix3(moge.get("scene_to_blender_matrix"), np.eye(3, dtype=np.float64))
    cv_to_scene = matrix3(
        moge.get("opencv_camera_to_pytorch3d_scene_matrix"),
        np.diag([-1.0, -1.0, 1.0]).astype(np.float64),
    )
    right_scene = cv_to_scene @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
    up_scene = cv_to_scene @ np.array([0.0, -1.0, 0.0], dtype=np.float64)
    back_scene = cv_to_scene @ np.array([0.0, 0.0, -1.0], dtype=np.float64)
    right = normalize(scene_to_blender @ right_scene)
    up = normalize(scene_to_blender @ up_scene)
    back = normalize(scene_to_blender @ back_scene)
    out = np.eye(4, dtype=np.float64)
    out[:3, 0] = right
    out[:3, 1] = up
    out[:3, 2] = back
    return out


def stage_global_translation(stage_report: dict[str, Any] | None) -> np.ndarray:
    if not stage_report:
        return np.zeros(3, dtype=np.float64)
    bbox = ((stage_report.get("transform") or {}).get("bbox_snap_objects"))
    if isinstance(bbox, dict) and bbox.get("global_translation") is not None:
        return np.asarray(bbox["global_translation"], dtype=np.float64).reshape(3)
    return np.zeros(3, dtype=np.float64)


def initial_camera_to_world(camera_report: dict[str, Any], stage_report: dict[str, Any] | None) -> np.ndarray:
    raw_c2w = raw_camera_to_blender_matrix(camera_report)
    upright = matrix4((camera_report.get("transform") or {}).get("matrix_4x4"), np.eye(4, dtype=np.float64))
    extra = np.eye(4, dtype=np.float64)
    extra[:3, 3] = stage_global_translation(stage_report)
    return extra @ upright @ raw_c2w


def normalized_intrinsics(report: dict[str, Any]) -> tuple[float, float, float, float]:
    intrinsics = (report.get("moge") or {}).get("intrinsics")
    if intrinsics is None:
        raise ValueError("camera report does not contain moge.intrinsics")
    mat = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
    return float(mat[0, 0]), float(mat[1, 1]), float(mat[0, 2]), float(mat[1, 2])


def camera_from_inputs(args: argparse.Namespace, target_mask: Path) -> tuple[np.ndarray, tuple[float, float, float, float], tuple[int, int], dict[str, Any]]:
    cam_opt = read_json(args.camera_optimization)
    if cam_opt is not None:
        c2w = matrix4(cam_opt["optimized_camera_to_world"])
        intr = cam_opt["intrinsics_normalized"]
        fx, fy, cx, cy = float(intr[0][0]), float(intr[1][1]), float(intr[0][2]), float(intr[1][2])
        source_size = cam_opt.get("source_image_size") or cam_opt.get("image_size")
        if source_size is None:
            raise ValueError("camera optimization report does not contain source_image_size")
        return c2w, (fx, fy, cx, cy), (int(source_size[0]), int(source_size[1])), {
            "source": "camera_optimization",
            "camera_optimization": str(args.camera_optimization),
        }

    camera_report = read_json(args.camera_report)
    if camera_report is None:
        raise ValueError("Either --camera-optimization or --camera-report is required")
    stage_report = read_json(args.stage_report)
    c2w = initial_camera_to_world(camera_report, stage_report)
    intr = normalized_intrinsics(camera_report)
    sample = ((camera_report.get("moge") or {}).get("sample_info") or {})
    size = sample.get("image_size")
    if isinstance(size, list) and len(size) >= 2:
        source_size = (int(size[0]), int(size[1]))
    else:
        mask = cv2.imread(str(target_mask), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(target_mask)
        source_size = (int(mask.shape[1]), int(mask.shape[0]))
    return c2w, intr, source_size, {
        "source": "camera_report",
        "camera_report": str(args.camera_report),
        "stage_report": str(args.stage_report) if args.stage_report else None,
    }


def target_resolution(width: int, height: int, max_side: int) -> tuple[int, int]:
    if max_side <= 0:
        out_w, out_h = width, height
    else:
        scale = min(1.0, float(max_side) / float(max(width, height)))
        out_w = max(2, int(round(width * scale)))
        out_h = max(2, int(round(height * scale)))
    if out_w % 2:
        out_w += 1
    if out_h % 2:
        out_h += 1
    return out_w, out_h


def load_label_mask(path: Path, width: int, height: int) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(path)
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask


def load_preview_image(path: Path | None, width: int, height: int) -> np.ndarray | None:
    if path is None or not path.is_file():
        return None
    image = imageio.imread(path)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.shape[2] == 4:
        image = image[..., :3]
    return cv2.resize(image.astype(np.uint8), (width, height), interpolation=cv2.INTER_AREA)


def project_to_clip(
    vertices_world: torch.Tensor,
    camera_to_world: torch.Tensor,
    intrinsics: tuple[float, float, float, float],
    near: float,
    far: float,
) -> torch.Tensor:
    fx, fy, cx, cy = intrinsics
    rot = camera_to_world[:3, :3]
    trans = camera_to_world[:3, 3]
    local = (vertices_world - trans[None, :]) @ rot
    cv = local * torch.tensor([1.0, -1.0, -1.0], dtype=vertices_world.dtype, device=vertices_world.device)[None, :]
    z = cv[:, 2].clamp_min(max(float(near) * 0.25, 1e-5))
    u = float(fx) * cv[:, 0] / z + float(cx)
    v = float(fy) * cv[:, 1] / z + float(cy)
    x_ndc = 2.0 * u - 1.0
    y_ndc = 2.0 * v - 1.0
    z_ndc = 2.0 * (z - float(near)) / max(float(far - near), 1e-6) - 1.0
    return torch.stack([x_ndc * z, y_ndc * z, z_ndc * z, z], dim=1)


def render_silhouette(
    ctx: Any,
    vertices_world: torch.Tensor,
    faces: torch.Tensor,
    camera_to_world: torch.Tensor,
    intrinsics: tuple[float, float, float, float],
    near: float,
    far: float,
    height: int,
    width: int,
    antialias: bool,
) -> torch.Tensor:
    clip = project_to_clip(vertices_world, camera_to_world, intrinsics, near, far).contiguous()
    rast, _ = dr.rasterize(ctx, clip[None, :, :], faces, resolution=[height, width])
    ones = torch.ones((1, vertices_world.shape[0], 1), dtype=vertices_world.dtype, device=vertices_world.device)
    sil, _ = dr.interpolate(ones, rast, faces)
    if antialias:
        sil = dr.antialias(sil.contiguous(), rast, clip[None, :, :], faces)
    return sil[0, :, :, 0].clamp(0.0, 1.0)


def compute_near_far(vertices: torch.Tensor, camera_to_world: torch.Tensor) -> tuple[float, float]:
    rot = camera_to_world[:3, :3]
    trans = camera_to_world[:3, 3]
    local = (vertices - trans[None, :]) @ rot
    cv_z = (-local[:, 2]).detach().cpu().numpy()
    positive = cv_z[np.isfinite(cv_z) & (cv_z > 1e-5)]
    if positive.size == 0:
        return 0.01, 100.0
    near = max(0.001, float(np.percentile(positive, 0.5)) * 0.25)
    far = max(near + 1.0, float(np.percentile(positive, 99.5)) * 4.0)
    return near, far


def mask_metrics(rendered: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    r = rendered > 0.5
    t = target > 0.5
    inter = int(np.logical_and(r, t).sum())
    union = int(np.logical_or(r, t).sum())
    iou = float(inter / union) if union else 0.0

    def bbox(mask: np.ndarray) -> list[int] | None:
        yy, xx = np.where(mask)
        if len(xx) == 0:
            return None
        return [int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())]

    rb = bbox(r)
    tb = bbox(t)
    center_error = None
    if rb is not None and tb is not None:
        rc = np.array([(rb[0] + rb[2]) * 0.5, (rb[1] + rb[3]) * 0.5], dtype=np.float64)
        tc = np.array([(tb[0] + tb[2]) * 0.5, (tb[1] + tb[3]) * 0.5], dtype=np.float64)
        center_error = float(np.linalg.norm(rc - tc))
    return {
        "iou": iou,
        "intersection_pixels": inter,
        "union_pixels": union,
        "render_bbox_xyxy": rb,
        "target_bbox_xyxy": tb,
        "bbox_center_error_pixels": center_error,
    }


def yaw_matrix_torch(yaw: torch.Tensor, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    c = torch.cos(yaw)
    s = torch.sin(yaw)
    zero = torch.zeros((), dtype=dtype, device=device)
    one = torch.ones((), dtype=dtype, device=device)
    return torch.stack(
        [
            torch.stack([c, -s, zero]),
            torch.stack([s, c, zero]),
            torch.stack([zero, zero, one]),
        ]
    )


def transform_vertices(
    vertices: torch.Tensor,
    pivot: torch.Tensor,
    translation: torch.Tensor,
    yaw: torch.Tensor,
    scale_delta: torch.Tensor,
) -> torch.Tensor:
    rot = yaw_matrix_torch(yaw, dtype=vertices.dtype, device=vertices.device)
    scale = 1.0 + scale_delta
    return ((vertices - pivot[None, :]) * scale) @ rot.T + pivot[None, :] + translation[None, :]


def translation_basis_from_camera(camera_to_world: torch.Tensor, mode: str) -> torch.Tensor:
    dtype = camera_to_world.dtype
    device = camera_to_world.device
    if mode == "camera_plane":
        right = F.normalize(camera_to_world[:3, 0], dim=0)
        up = F.normalize(camera_to_world[:3, 1], dim=0)
        return torch.stack([right, up], dim=1)
    if mode == "world_xy":
        return torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=dtype, device=device)
    if mode == "world_xyz":
        return torch.eye(3, dtype=dtype, device=device)
    raise ValueError(mode)


def translation_to_param(translation: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    if basis.shape[1] == 3:
        return translation.detach().clone()
    try:
        return torch.linalg.lstsq(basis, translation.detach()).solution
    except Exception:
        return basis.T @ translation.detach()


def translation_from_param(param: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return basis @ param


def mask_moment_context(target: torch.Tensor) -> dict[str, torch.Tensor]:
    height, width = target.shape
    xs = torch.linspace(-1.0, 1.0, width, dtype=target.dtype, device=target.device)
    ys = torch.linspace(-1.0, 1.0, height, dtype=target.dtype, device=target.device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    target_mass = target.sum().clamp_min(1e-6)
    target_center = torch.stack([(target * grid_x).sum() / target_mass, (target * grid_y).sum() / target_mass])
    target_area = target.mean().clamp_min(1e-6)
    return {
        "grid_x": grid_x,
        "grid_y": grid_y,
        "target_center": target_center.detach(),
        "target_area": target_area.detach(),
    }


def silhouette_alignment_loss(
    sil: torch.Tensor,
    target: torch.Tensor,
    outside_dt: torch.Tensor,
    moments: dict[str, torch.Tensor],
    *,
    dt_weight: float,
    center_weight: float,
    area_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    l1 = F.l1_loss(sil, target)
    dt_loss = torch.mean(sil * outside_dt)
    mass = sil.sum().clamp_min(1e-6)
    center = torch.stack([(sil * moments["grid_x"]).sum() / mass, (sil * moments["grid_y"]).sum() / mass])
    center_loss = torch.mean((center - moments["target_center"]) ** 2)
    area = sil.mean().clamp_min(1e-6)
    area_loss = torch.log(area / moments["target_area"]).pow(2)
    loss = l1 + float(dt_weight) * dt_loss + float(center_weight) * center_loss + float(area_weight) * area_loss
    return loss, {
        "l1": l1,
        "dt": dt_loss,
        "center": center_loss,
        "area": area_loss,
        "silhouette_area": area,
        "target_area": moments["target_area"],
        "center_x": center[0],
        "center_y": center[1],
    }


def render_pose_loss(
    *,
    ctx: Any,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    pivot: torch.Tensor,
    translation: torch.Tensor,
    yaw: torch.Tensor,
    scale_delta: torch.Tensor,
    camera_to_world: torch.Tensor,
    intrinsics: tuple[float, float, float, float],
    near: float,
    far: float,
    target: torch.Tensor,
    outside_dt: torch.Tensor,
    moments: dict[str, torch.Tensor],
    height: int,
    width: int,
    antialias: bool,
    dt_weight: float,
    center_weight: float,
    area_weight: float,
    translation_prior_weight: float,
    yaw_prior_weight: float,
    scale_prior_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    sil = render_silhouette(
        ctx,
        transform_vertices(vertices, pivot, translation, yaw, scale_delta),
        faces,
        camera_to_world,
        intrinsics,
        near,
        far,
        height,
        width,
        antialias,
    )
    align_loss, components = silhouette_alignment_loss(
        sil,
        target,
        outside_dt,
        moments,
        dt_weight=dt_weight,
        center_weight=center_weight,
        area_weight=area_weight,
    )
    t_prior = float(translation_prior_weight) * torch.sum(translation * translation)
    y_prior = float(yaw_prior_weight) * yaw * yaw
    s_prior = float(scale_prior_weight) * scale_delta * scale_delta
    loss = align_loss + t_prior + y_prior + s_prior
    components.update({"translation_prior": t_prior, "yaw_prior": y_prior, "scale_prior": s_prior})
    return loss, sil, components


def choose_scalar_seed(
    *,
    name: str,
    candidates: torch.Tensor,
    ctx: Any,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    pivot: torch.Tensor,
    translation: torch.Tensor,
    yaw: torch.Tensor,
    scale_delta: torch.Tensor,
    camera_to_world: torch.Tensor,
    intrinsics: tuple[float, float, float, float],
    near: float,
    far: float,
    target: torch.Tensor,
    outside_dt: torch.Tensor,
    moments: dict[str, torch.Tensor],
    height: int,
    width: int,
    antialias: bool,
    dt_weight: float,
    center_weight: float,
    area_weight: float,
    translation_prior_weight: float,
    yaw_prior_weight: float,
    scale_prior_weight: float,
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    best_value = None
    best_loss = float("inf")
    history: list[dict[str, float]] = []
    with torch.no_grad():
        for candidate in candidates:
            cur_yaw = candidate.reshape(()) if name == "yaw" else yaw
            cur_scale = candidate.reshape(()) if name == "scale" else scale_delta
            loss, sil, components = render_pose_loss(
                ctx=ctx,
                vertices=vertices,
                faces=faces,
                pivot=pivot,
                translation=translation,
                yaw=cur_yaw,
                scale_delta=cur_scale,
                camera_to_world=camera_to_world,
                intrinsics=intrinsics,
                near=near,
                far=far,
                target=target,
                outside_dt=outside_dt,
                moments=moments,
                height=height,
                width=width,
                antialias=antialias,
                dt_weight=dt_weight,
                center_weight=center_weight,
                area_weight=area_weight,
                translation_prior_weight=translation_prior_weight,
                yaw_prior_weight=yaw_prior_weight,
                scale_prior_weight=scale_prior_weight,
            )
            value = float(loss.detach().cpu())
            history.append(
                {
                    "value": float(candidate.detach().cpu()),
                    "loss": value,
                    "l1": float(components["l1"].detach().cpu()),
                    "dt": float(components["dt"].detach().cpu()),
                    "center": float(components["center"].detach().cpu()),
                    "area": float(components["area"].detach().cpu()),
                    "silhouette_area": float(components["silhouette_area"].detach().cpu()),
                }
            )
            if value < best_loss:
                best_loss = value
                best_value = candidate.detach().clone().reshape(())
    if best_value is None:
        best_value = yaw.detach().clone().reshape(()) if name == "yaw" else scale_delta.detach().clone().reshape(())
    return best_value, history


def choose_translation_seed(
    *,
    ctx: Any,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    pivot: torch.Tensor,
    translation_basis: torch.Tensor,
    max_translation: float,
    grid_size: int,
    yaw: torch.Tensor,
    scale_delta: torch.Tensor,
    camera_to_world: torch.Tensor,
    intrinsics: tuple[float, float, float, float],
    near: float,
    far: float,
    target: torch.Tensor,
    outside_dt: torch.Tensor,
    moments: dict[str, torch.Tensor],
    height: int,
    width: int,
    antialias: bool,
    dt_weight: float,
    center_weight: float,
    area_weight: float,
    translation_prior_weight: float,
    yaw_prior_weight: float,
    scale_prior_weight: float,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    dims = int(translation_basis.shape[1])
    if dims <= 0:
        return torch.zeros(3, dtype=vertices.dtype, device=vertices.device), []
    count = max(1, int(grid_size))
    radius = max(0.0, float(max_translation))
    if count <= 1 or radius <= 0.0:
        return torch.zeros(3, dtype=vertices.dtype, device=vertices.device), []

    values = torch.linspace(-radius, radius, count, dtype=vertices.dtype, device=vertices.device)
    zero = torch.zeros((), dtype=vertices.dtype, device=vertices.device)
    if not torch.any(torch.isclose(values, zero, atol=1e-7)):
        values = torch.sort(torch.cat([values, zero.reshape(1)]))[0]
    meshes = torch.meshgrid(*([values] * dims), indexing="ij")
    candidates = torch.stack([axis.reshape(-1) for axis in meshes], dim=1)
    order = torch.argsort(torch.linalg.norm(candidates, dim=1))
    candidates = candidates[order]

    best_translation: torch.Tensor | None = None
    best_loss = float("inf")
    history: list[dict[str, Any]] = []
    with torch.no_grad():
        for candidate in candidates:
            cur_translation = translation_from_param(candidate, translation_basis)
            loss, sil, components = render_pose_loss(
                ctx=ctx,
                vertices=vertices,
                faces=faces,
                pivot=pivot,
                translation=cur_translation,
                yaw=yaw,
                scale_delta=scale_delta,
                camera_to_world=camera_to_world,
                intrinsics=intrinsics,
                near=near,
                far=far,
                target=target,
                outside_dt=outside_dt,
                moments=moments,
                height=height,
                width=width,
                antialias=antialias,
                dt_weight=dt_weight,
                center_weight=center_weight,
                area_weight=area_weight,
                translation_prior_weight=translation_prior_weight,
                yaw_prior_weight=yaw_prior_weight,
                scale_prior_weight=scale_prior_weight,
            )
            rendered = sil > 0.5
            target_bool = target > 0.5
            inter = torch.logical_and(rendered, target_bool).sum()
            union = torch.logical_or(rendered, target_bool).sum().clamp_min(1)
            value = float(loss.detach().cpu())
            history.append(
                {
                    "param": [float(v) for v in candidate.detach().cpu().tolist()],
                    "translation": [float(v) for v in cur_translation.detach().cpu().tolist()],
                    "loss": value,
                    "iou": float((inter.float() / union.float()).detach().cpu()),
                    "l1": float(components["l1"].detach().cpu()),
                    "dt": float(components["dt"].detach().cpu()),
                    "center": float(components["center"].detach().cpu()),
                    "area": float(components["area"].detach().cpu()),
                    "silhouette_area": float(components["silhouette_area"].detach().cpu()),
                }
            )
            if value < best_loss:
                best_loss = value
                best_translation = cur_translation.detach().clone()

    if best_translation is None:
        best_translation = torch.zeros(3, dtype=vertices.dtype, device=vertices.device)
    return best_translation, history


def optimize_phase(
    *,
    phase: str,
    ctx: Any,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    pivot: torch.Tensor,
    camera_to_world: torch.Tensor,
    intrinsics: tuple[float, float, float, float],
    near: float,
    far: float,
    target: torch.Tensor,
    outside_dt: torch.Tensor,
    moments: dict[str, torch.Tensor],
    height: int,
    width: int,
    antialias: bool,
    translation: torch.Tensor,
    yaw: torch.Tensor,
    scale_delta: torch.Tensor,
    steps: int,
    lr: float,
    dt_weight: float,
    center_weight: float,
    area_weight: float,
    translation_prior_weight: float,
    yaw_prior_weight: float,
    scale_prior_weight: float,
    max_translation: float,
    max_yaw: float,
    max_scale_delta: float,
    translation_mode: str,
    translation_basis: torch.Tensor,
    min_delta: float,
    patience: int,
    log_every: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, float]]]:
    if phase == "position":
        opt_param = torch.nn.Parameter(translation_to_param(translation, translation_basis))
        fixed_yaw = yaw.detach().clone()
        fixed_scale = scale_delta.detach().clone()
        params = [opt_param]
    elif phase == "yaw":
        opt_param = torch.nn.Parameter(yaw.detach().clone().reshape(()))
        fixed_translation = translation.detach().clone()
        fixed_scale = scale_delta.detach().clone()
        params = [opt_param]
    elif phase == "scale":
        opt_param = torch.nn.Parameter(scale_delta.detach().clone().reshape(()))
        fixed_translation = translation.detach().clone()
        fixed_yaw = yaw.detach().clone()
        params = [opt_param]
    else:
        raise ValueError(phase)

    optimizer = torch.optim.Adam(params, lr=float(lr))
    best_loss = float("inf")
    best_translation = translation.detach().clone()
    best_yaw = yaw.detach().clone()
    best_scale_delta = scale_delta.detach().clone()
    best_sil: torch.Tensor | None = None
    stale = 0
    history: list[dict[str, float]] = []

    for step in range(1, int(steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        if phase == "position":
            cur_translation = translation_from_param(opt_param, translation_basis)
            cur_yaw = fixed_yaw
            cur_scale = fixed_scale
        elif phase == "yaw":
            cur_translation = fixed_translation
            cur_yaw = opt_param
            cur_scale = fixed_scale
        else:
            cur_translation = fixed_translation
            cur_yaw = fixed_yaw
            cur_scale = opt_param
        loss, sil, components = render_pose_loss(
            ctx=ctx,
            vertices=vertices,
            faces=faces,
            pivot=pivot,
            translation=cur_translation,
            yaw=cur_yaw,
            scale_delta=cur_scale,
            camera_to_world=camera_to_world,
            intrinsics=intrinsics,
            near=near,
            far=far,
            target=target,
            outside_dt=outside_dt,
            moments=moments,
            height=height,
            width=width,
            antialias=antialias,
            dt_weight=dt_weight,
            center_weight=center_weight,
            area_weight=area_weight,
            translation_prior_weight=translation_prior_weight,
            yaw_prior_weight=yaw_prior_weight,
            scale_prior_weight=scale_prior_weight,
        )
        loss.backward()
        value = float(loss.detach().cpu())
        improved = value < best_loss - float(min_delta)
        if improved:
            best_loss = value
            if phase == "position":
                best_translation = cur_translation.detach().clone()
                best_yaw = fixed_yaw.detach().clone()
                best_scale_delta = fixed_scale.detach().clone()
            elif phase == "yaw":
                best_translation = fixed_translation.detach().clone()
                best_yaw = opt_param.detach().clone().reshape(())
                best_scale_delta = fixed_scale.detach().clone()
            else:
                best_translation = fixed_translation.detach().clone()
                best_yaw = fixed_yaw.detach().clone()
                best_scale_delta = opt_param.detach().clone().reshape(())
            best_sil = sil.detach().clone()
            stale = 0
        else:
            stale += 1

        optimizer.step()
        with torch.no_grad():
            if phase == "position":
                opt_param.clamp_(min=-float(max_translation), max=float(max_translation))
            elif phase == "yaw":
                opt_param.clamp_(min=-float(max_yaw), max=float(max_yaw))
            else:
                opt_param.clamp_(min=-float(max_scale_delta), max=float(max_scale_delta))

        if step == 1 or step == int(steps) or (int(log_every) > 0 and step % int(log_every) == 0):
            history.append(
                {
                    "step": float(step),
                    "loss": value,
                    "l1": float(components["l1"].detach().cpu()),
                    "dt": float(components["dt"].detach().cpu()),
                    "center": float(components["center"].detach().cpu()),
                    "area": float(components["area"].detach().cpu()),
                    "silhouette_area": float(components["silhouette_area"].detach().cpu()),
                    "target_area": float(components["target_area"].detach().cpu()),
                    "translation_norm": float(torch.linalg.norm((cur_translation if phase == "position" else fixed_translation).detach()).cpu()),
                    "yaw_degrees": float((opt_param if phase == "yaw" else fixed_yaw).detach().cpu()) * 180.0 / math.pi,
                    "scale": float((1.0 + (opt_param if phase == "scale" else fixed_scale)).detach().cpu()),
                }
            )
        if int(patience) > 0 and stale >= int(patience):
            break

    if best_sil is None:
        best_sil = render_silhouette(
            ctx, transform_vertices(vertices, pivot, best_translation, best_yaw, best_scale_delta), faces, camera_to_world,
            intrinsics, near, far, height, width, antialias
        ).detach()
    return best_translation.detach(), best_yaw.detach().reshape(()), best_scale_delta.detach().reshape(()), best_sil.detach(), history


def save_preview(path: Path, target: np.ndarray, initial: np.ndarray, optimized: np.ndarray, image: np.ndarray | None) -> None:
    h, w = target.shape
    base = image if image is not None else np.full((h, w, 3), 36, dtype=np.uint8)

    def overlay(render: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
        out = base.copy().astype(np.float32)
        target_rgb = np.zeros_like(out)
        target_rgb[..., 1] = 255
        render_rgb = np.zeros_like(out)
        render_rgb[..., 0] = color[0]
        render_rgb[..., 1] = color[1]
        render_rgb[..., 2] = color[2]
        out = out * 0.55 + target_rgb * (target[..., None] * 0.22) + render_rgb * (render[..., None] * 0.35)
        return np.clip(out, 0, 255).astype(np.uint8)

    target_panel = np.repeat((target * 255.0).astype(np.uint8)[..., None], 3, axis=2)
    initial_panel = overlay(initial, (255, 180, 0))
    optimized_panel = overlay(optimized, (0, 255, 255))
    diff = np.zeros((h, w, 3), dtype=np.uint8)
    diff[..., 1] = (target * 255).astype(np.uint8)
    diff[..., 0] = (optimized * 255).astype(np.uint8)
    diff[..., 2] = (initial * 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, np.concatenate([target_panel, initial_panel, optimized_panel, diff], axis=1))


def object_delta_matrix(pivot: np.ndarray, translation: np.ndarray, yaw: float, scale_delta: float) -> np.ndarray:
    c = math.cos(float(yaw))
    s = math.sin(float(yaw))
    rot = np.eye(4, dtype=np.float64)
    rot[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    scale = np.eye(4, dtype=np.float64)
    scale[:3, :3] *= max(1e-6, 1.0 + float(scale_delta))
    t_pos = np.eye(4, dtype=np.float64)
    t_pos[:3, 3] = pivot + translation
    t_neg = np.eye(4, dtype=np.float64)
    t_neg[:3, 3] = -pivot
    return t_pos @ rot @ scale @ t_neg


def as_json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return as_json(value.tolist())
    if isinstance(value, torch.Tensor):
        return as_json(value.detach().cpu().numpy())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): as_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_json(v) for v in value]
    return value


def optimize_one_object(job: dict[str, Any]) -> dict[str, Any]:
    device = torch.device(str(job["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    mesh = np.load(job["mesh"], allow_pickle=False)
    vertices_np = np.asarray(mesh["vertices"], dtype=np.float32)
    faces_np = np.asarray(mesh["faces"], dtype=np.int32)
    metadata = json.loads(str(mesh["metadata"])) if "metadata" in mesh else {}
    object_meta = job["object_meta"]
    mask_id = int(job["mask_id"])
    name = str(job["name"])

    label = load_label_mask(Path(job["target_mask"]), int(job["render_width"]), int(job["render_height"]))
    target_np = (label == mask_id).astype(np.float32)
    if int(target_np.sum()) <= 0:
        return {
            "name": name,
            "mask_id": mask_id,
            "status": "skipped_empty_target_mask",
            "delta_translation": [0.0, 0.0, 0.0],
            "delta_yaw_radians": 0.0,
            "delta_scale": 0.0,
            "scale": 1.0,
            "delta_transform_world": np.eye(4, dtype=np.float64),
        }

    v0 = int(object_meta["vertices_start"])
    vc = int(object_meta["vertices_count"])
    f0 = int(object_meta["faces_start"])
    fc = int(object_meta["faces_count"])
    if vc <= 0 or fc <= 0:
        return {
            "name": name,
            "mask_id": mask_id,
            "status": "skipped_empty_mesh",
            "delta_translation": [0.0, 0.0, 0.0],
            "delta_yaw_radians": 0.0,
            "delta_scale": 0.0,
            "scale": 1.0,
            "delta_transform_world": np.eye(4, dtype=np.float64),
        }

    vertices = torch.from_numpy(vertices_np[v0 : v0 + vc]).to(device=device, dtype=torch.float32)
    faces = torch.from_numpy(faces_np[f0 : f0 + fc] - v0).to(device=device, dtype=torch.int32).contiguous()
    c2w = torch.tensor(np.asarray(job["camera_to_world"], dtype=np.float32), dtype=torch.float32, device=device)
    intrinsics = tuple(float(v) for v in job["intrinsics"])
    near, far = compute_near_far(vertices, c2w)
    target = torch.from_numpy(target_np).to(device=device, dtype=torch.float32)
    outside_dist = cv2.distanceTransform((target_np < 0.5).astype(np.uint8), cv2.DIST_L2, 3)
    if float(outside_dist.max()) > 0:
        outside_dist = outside_dist / float(outside_dist.max())
    outside_dt = torch.from_numpy(outside_dist.astype(np.float32)).to(device)
    moments = mask_moment_context(target)
    pivot_np = (np.asarray(object_meta["bbox_min"], dtype=np.float64) + np.asarray(object_meta["bbox_max"], dtype=np.float64)) * 0.5
    pivot = torch.tensor(pivot_np, dtype=torch.float32, device=device)
    translation0 = torch.zeros(3, dtype=torch.float32, device=device)
    yaw0 = torch.zeros((), dtype=torch.float32, device=device)
    scale0 = torch.zeros((), dtype=torch.float32, device=device)
    translation_basis = translation_basis_from_camera(c2w, str(job["translation_mode"]))
    ctx = dr.RasterizeCudaContext() if device.type == "cuda" else dr.RasterizeGLContext()
    antialias = not bool(job["no_antialias"])

    with torch.no_grad():
        initial_sil = render_silhouette(
            ctx, vertices, faces, c2w, intrinsics, near, far, int(job["render_height"]), int(job["render_width"]), antialias
        )
        initial_np = initial_sil.detach().cpu().numpy()
        initial_metrics = mask_metrics(initial_np, target_np)

    translation_seed_history: list[dict[str, Any]] = []
    translation_seed = translation0
    if int(job["translation_initial_grid"]) > 1:
        translation_seed, translation_seed_history = choose_translation_seed(
            ctx=ctx,
            vertices=vertices,
            faces=faces,
            pivot=pivot,
            translation_basis=translation_basis,
            max_translation=float(job["max_translation"]),
            grid_size=int(job["translation_initial_grid"]),
            yaw=yaw0,
            scale_delta=scale0,
            camera_to_world=c2w,
            intrinsics=intrinsics,
            near=near,
            far=far,
            target=target,
            outside_dt=outside_dt,
            moments=moments,
            height=int(job["render_height"]),
            width=int(job["render_width"]),
            antialias=antialias,
            dt_weight=float(job["dt_weight"]),
            center_weight=float(job["center_weight"]),
            area_weight=float(job["area_weight"]),
            translation_prior_weight=float(job["translation_prior_weight"]),
            yaw_prior_weight=0.0,
            scale_prior_weight=0.0,
        )

    best_translation, _, best_scale_delta, position_sil, position_history = optimize_phase(
        phase="position",
        ctx=ctx,
        vertices=vertices,
        faces=faces,
        pivot=pivot,
        camera_to_world=c2w,
        intrinsics=intrinsics,
        near=near,
        far=far,
        target=target,
        outside_dt=outside_dt,
        moments=moments,
        height=int(job["render_height"]),
        width=int(job["render_width"]),
        antialias=antialias,
        translation=translation_seed,
        yaw=yaw0,
        scale_delta=scale0,
        steps=int(job["position_steps"]),
        lr=float(job["position_lr"]),
        dt_weight=float(job["dt_weight"]),
        center_weight=float(job["center_weight"]),
        area_weight=float(job["area_weight"]),
        translation_prior_weight=float(job["translation_prior_weight"]),
        yaw_prior_weight=0.0,
        scale_prior_weight=0.0,
        max_translation=float(job["max_translation"]),
        max_yaw=float(job["max_yaw"]),
        max_scale_delta=float(job["max_scale_delta"]),
        translation_mode=str(job["translation_mode"]),
        translation_basis=translation_basis,
        min_delta=float(job["converge_min_delta"]),
        patience=int(job["converge_patience"]),
        log_every=int(job["log_every"]),
    )
    position_np = position_sil.detach().cpu().numpy()
    position_metrics = mask_metrics(position_np, target_np)
    phase_acceptance: list[dict[str, Any]] = []
    iou_drop = float(job["phase_accept_iou_drop"])
    if position_metrics["iou"] + iou_drop < initial_metrics["iou"]:
        phase_acceptance.append({"phase": "position", "accepted": False, "reason": "iou_drop_guard", "from_iou": initial_metrics["iou"], "to_iou": position_metrics["iou"]})
        best_translation = translation0.detach().clone()
        best_yaw = yaw0.detach().clone()
        best_scale_delta = scale0.detach().clone()
        position_sil = initial_sil.detach().clone()
        position_np = initial_np
        position_metrics = initial_metrics
    else:
        phase_acceptance.append({"phase": "position", "accepted": True, "from_iou": initial_metrics["iou"], "to_iou": position_metrics["iou"]})

    yaw_seed_history: list[dict[str, float]] = []
    yaw_seed = torch.zeros((), dtype=torch.float32, device=device)
    if int(job["yaw_initial_samples"]) > 1 and float(job["max_yaw"]) > 0.0:
        candidates = torch.linspace(-float(job["max_yaw"]), float(job["max_yaw"]), int(job["yaw_initial_samples"]), dtype=torch.float32, device=device)
        if not torch.any(torch.isclose(candidates, torch.zeros((), dtype=torch.float32, device=device), atol=1e-7)):
            candidates = torch.sort(torch.cat([candidates, torch.zeros(1, dtype=torch.float32, device=device)]))[0]
        yaw_seed, yaw_seed_history = choose_scalar_seed(
            name="yaw",
            candidates=candidates,
            ctx=ctx,
            vertices=vertices,
            faces=faces,
            pivot=pivot,
            translation=best_translation,
            yaw=yaw0,
            scale_delta=best_scale_delta,
            camera_to_world=c2w,
            intrinsics=intrinsics,
            near=near,
            far=far,
            target=target,
            outside_dt=outside_dt,
            moments=moments,
            height=int(job["render_height"]),
            width=int(job["render_width"]),
            antialias=antialias,
            dt_weight=float(job["dt_weight"]),
            center_weight=float(job["center_weight"]),
            area_weight=float(job["area_weight"]),
            translation_prior_weight=float(job["translation_prior_weight"]),
            yaw_prior_weight=float(job["yaw_prior_weight"]),
            scale_prior_weight=0.0,
        )
    best_translation, best_yaw, best_scale_delta, yaw_sil, yaw_history = optimize_phase(
        phase="yaw",
        ctx=ctx,
        vertices=vertices,
        faces=faces,
        pivot=pivot,
        camera_to_world=c2w,
        intrinsics=intrinsics,
        near=near,
        far=far,
        target=target,
        outside_dt=outside_dt,
        moments=moments,
        height=int(job["render_height"]),
        width=int(job["render_width"]),
        antialias=antialias,
        translation=best_translation,
        yaw=yaw_seed,
        scale_delta=best_scale_delta,
        steps=int(job["yaw_steps"]),
        lr=float(job["yaw_lr"]),
        dt_weight=float(job["dt_weight"]),
        center_weight=float(job["center_weight"]),
        area_weight=float(job["area_weight"]),
        translation_prior_weight=float(job["translation_prior_weight"]),
        yaw_prior_weight=float(job["yaw_prior_weight"]),
        scale_prior_weight=0.0,
        max_translation=float(job["max_translation"]),
        max_yaw=float(job["max_yaw"]),
        max_scale_delta=float(job["max_scale_delta"]),
        translation_mode=str(job["translation_mode"]),
        translation_basis=translation_basis,
        min_delta=float(job["converge_min_delta"]),
        patience=int(job["converge_patience"]),
        log_every=int(job["log_every"]),
    )
    yaw_np = yaw_sil.detach().cpu().numpy()
    yaw_metrics = mask_metrics(yaw_np, target_np)
    if yaw_metrics["iou"] + iou_drop < position_metrics["iou"]:
        phase_acceptance.append({"phase": "yaw", "accepted": False, "reason": "iou_drop_guard", "from_iou": position_metrics["iou"], "to_iou": yaw_metrics["iou"]})
        best_yaw = torch.zeros((), dtype=torch.float32, device=device)
        yaw_sil = position_sil.detach().clone()
        yaw_np = position_np
        yaw_metrics = position_metrics
    else:
        phase_acceptance.append({"phase": "yaw", "accepted": True, "from_iou": position_metrics["iou"], "to_iou": yaw_metrics["iou"], "seed_degrees": float(yaw_seed.detach().cpu()) * 180.0 / math.pi})

    scale_seed_history: list[dict[str, float]] = []
    scale_seed = best_scale_delta.detach().clone().reshape(())
    if int(job["scale_initial_samples"]) > 1 and float(job["max_scale_delta"]) > 0.0:
        candidates = torch.linspace(-float(job["max_scale_delta"]), float(job["max_scale_delta"]), int(job["scale_initial_samples"]), dtype=torch.float32, device=device)
        if not torch.any(torch.isclose(candidates, torch.zeros((), dtype=torch.float32, device=device), atol=1e-7)):
            candidates = torch.sort(torch.cat([candidates, torch.zeros(1, dtype=torch.float32, device=device)]))[0]
        scale_seed, scale_seed_history = choose_scalar_seed(
            name="scale",
            candidates=candidates,
            ctx=ctx,
            vertices=vertices,
            faces=faces,
            pivot=pivot,
            translation=best_translation,
            yaw=best_yaw,
            scale_delta=best_scale_delta,
            camera_to_world=c2w,
            intrinsics=intrinsics,
            near=near,
            far=far,
            target=target,
            outside_dt=outside_dt,
            moments=moments,
            height=int(job["render_height"]),
            width=int(job["render_width"]),
            antialias=antialias,
            dt_weight=float(job["dt_weight"]),
            center_weight=float(job["center_weight"]),
            area_weight=float(job["area_weight"]),
            translation_prior_weight=float(job["translation_prior_weight"]),
            yaw_prior_weight=float(job["yaw_prior_weight"]),
            scale_prior_weight=float(job["scale_prior_weight"]),
        )
    best_translation, best_yaw, best_scale_delta, optimized_sil, scale_history = optimize_phase(
        phase="scale",
        ctx=ctx,
        vertices=vertices,
        faces=faces,
        pivot=pivot,
        camera_to_world=c2w,
        intrinsics=intrinsics,
        near=near,
        far=far,
        target=target,
        outside_dt=outside_dt,
        moments=moments,
        height=int(job["render_height"]),
        width=int(job["render_width"]),
        antialias=antialias,
        translation=best_translation,
        yaw=best_yaw,
        scale_delta=scale_seed,
        steps=int(job["scale_steps"]),
        lr=float(job["scale_lr"]),
        dt_weight=float(job["dt_weight"]),
        center_weight=float(job["center_weight"]),
        area_weight=float(job["area_weight"]),
        translation_prior_weight=float(job["translation_prior_weight"]),
        yaw_prior_weight=float(job["yaw_prior_weight"]),
        scale_prior_weight=float(job["scale_prior_weight"]),
        max_translation=float(job["max_translation"]),
        max_yaw=float(job["max_yaw"]),
        max_scale_delta=float(job["max_scale_delta"]),
        translation_mode=str(job["translation_mode"]),
        translation_basis=translation_basis,
        min_delta=float(job["converge_min_delta"]),
        patience=int(job["converge_patience"]),
        log_every=int(job["log_every"]),
    )

    optimized_np = optimized_sil.detach().cpu().numpy()
    optimized_metrics = mask_metrics(optimized_np, target_np)
    if optimized_metrics["iou"] + iou_drop < yaw_metrics["iou"]:
        phase_acceptance.append({"phase": "scale", "accepted": False, "reason": "iou_drop_guard", "from_iou": yaw_metrics["iou"], "to_iou": optimized_metrics["iou"]})
        best_scale_delta = torch.zeros((), dtype=torch.float32, device=device)
        optimized_sil = yaw_sil.detach().clone()
        optimized_np = yaw_np
        optimized_metrics = yaw_metrics
    else:
        phase_acceptance.append({"phase": "scale", "accepted": True, "from_iou": yaw_metrics["iou"], "to_iou": optimized_metrics["iou"], "seed_scale": float((1.0 + scale_seed).detach().cpu())})
    translation_np = best_translation.detach().cpu().numpy().astype(np.float64)
    yaw_value = float(best_yaw.detach().cpu())
    scale_delta_value = float(best_scale_delta.detach().cpu())
    accepted = True
    status = "ok"
    if (not bool(job["accept_worse_iou"])) and optimized_metrics["iou"] < initial_metrics["iou"]:
        accepted = False
        status = "kept_initial_due_to_iou_guard"
        phase_acceptance.append({"phase": "final", "accepted": False, "reason": "final_iou_guard", "from_iou": initial_metrics["iou"], "to_iou": optimized_metrics["iou"]})
        translation_np = np.zeros(3, dtype=np.float64)
        yaw_value = 0.0
        scale_delta_value = 0.0
        optimized_np = initial_np
        optimized_metrics = initial_metrics
    else:
        phase_acceptance.append({"phase": "final", "accepted": True, "from_iou": initial_metrics["iou"], "to_iou": optimized_metrics["iou"]})

    delta = object_delta_matrix(pivot_np, translation_np, yaw_value, scale_delta_value)
    preview_path = None
    if job.get("preview_dir"):
        preview_path = str(Path(job["preview_dir"]) / f"{name}_pose_optimization.png")
        image = load_preview_image(Path(job["input_image"]) if job.get("input_image") else None, int(job["render_width"]), int(job["render_height"]))
        save_preview(Path(preview_path), target_np, initial_np, optimized_np, image)

    return {
        "name": name,
        "mask_id": mask_id,
        "status": status,
        "accepted": bool(accepted),
        "semantic_label": job.get("semantic_label"),
        "mesh_object": object_meta,
        "pivot_world": pivot_np,
        "delta_translation": translation_np,
        "delta_yaw_radians": yaw_value,
        "delta_yaw_degrees": yaw_value * 180.0 / math.pi,
        "delta_scale": scale_delta_value,
        "scale": 1.0 + scale_delta_value,
        "delta_transform_world": delta,
        "preview": preview_path,
        "metrics": {
            "initial": initial_metrics,
            "after_position": position_metrics,
            "after_yaw": yaw_metrics,
            "optimized": optimized_metrics,
            "iou_delta": float(optimized_metrics["iou"] - initial_metrics["iou"]),
        },
        "optimization": {
            "near": float(near),
            "far": float(far),
            "position_history": position_history,
            "translation_seed_history": translation_seed_history,
            "yaw_seed_history": yaw_seed_history,
            "yaw_history": yaw_history,
            "scale_seed_history": scale_seed_history,
            "scale_history": scale_history,
            "phase_acceptance": phase_acceptance,
            "translation_mode": str(job["translation_mode"]),
            "translation_initial_grid": int(job["translation_initial_grid"]),
            "center_weight": float(job["center_weight"]),
            "area_weight": float(job["area_weight"]),
            "yaw_initial_samples": int(job["yaw_initial_samples"]),
            "scale_initial_samples": int(job["scale_initial_samples"]),
            "notes": [
                "Position is optimized first.",
                "Position starts from a coarse bounded translation seed grid before gradient refinement.",
                "Position defaults to camera-plane translation to avoid depth/scale ambiguity in a single view.",
                "Alignment score combines mask L1, outside-target distance, differentiable center alignment, and silhouette area-ratio terms.",
                "Yaw and scale evaluate multiple scalar seeds before gradient refinement.",
                "Each phase is rejected independently if it drops IoU too much from the previous accepted phase.",
                "Yaw is optimized second with translation fixed.",
                "Uniform scale is optimized third with translation and yaw fixed.",
                "Yaw is rotation around Blender/SAPIEN gravity axis Z.",
            ],
        },
        "mesh_metadata_schema": metadata.get("schema"),
    }


def build_jobs(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    final_manifest = read_json(args.final_manifest)
    assert final_manifest is not None
    c2w, intrinsics, source_size, camera_meta = camera_from_inputs(args, args.target_mask)
    render_width, render_height = target_resolution(int(source_size[0]), int(source_size[1]), int(args.max_side))
    mesh = np.load(args.mesh, allow_pickle=False)
    metadata = json.loads(str(mesh["metadata"])) if "metadata" in mesh else {}
    mesh_objects = {str(item.get("name")): item for item in metadata.get("objects") or []}
    filters = set(args.object_name or [])

    jobs = []
    for obj in final_manifest.get("objects") or []:
        name = obj.get("final_3d_object_name")
        mask_id = obj.get("mask_id")
        if not name or mask_id is None:
            continue
        if filters and str(name) not in filters:
            continue
        mesh_obj = mesh_objects.get(str(name))
        if mesh_obj is None:
            continue
        jobs.append(
            {
                "mesh": str(args.mesh),
                "target_mask": str(args.target_mask),
                "input_image": str(args.input_image) if args.input_image else None,
                "preview_dir": str(args.preview_dir) if args.preview_dir else None,
                "device": args.device,
                "render_width": int(render_width),
                "render_height": int(render_height),
                "camera_to_world": c2w,
                "intrinsics": intrinsics,
                "name": str(name),
                "mask_id": int(mask_id),
                "semantic_label": obj.get("semantic_label"),
                "object_meta": mesh_obj,
                "position_steps": int(args.position_steps),
                "yaw_steps": int(args.yaw_steps),
                "scale_steps": int(args.scale_steps),
                "position_lr": float(args.position_lr),
                "yaw_lr": float(args.yaw_lr),
                "scale_lr": float(args.scale_lr),
                "dt_weight": float(args.dt_weight),
                "center_weight": float(args.center_weight),
                "area_weight": float(args.area_weight),
                "translation_prior_weight": float(args.translation_prior_weight),
                "yaw_prior_weight": float(args.yaw_prior_weight),
                "scale_prior_weight": float(args.scale_prior_weight),
                "translation_mode": str(args.translation_mode),
                "translation_initial_grid": int(args.translation_initial_grid),
                "yaw_initial_samples": int(args.yaw_initial_samples),
                "scale_initial_samples": int(args.scale_initial_samples),
                "phase_accept_iou_drop": float(args.phase_accept_iou_drop),
                "max_translation": float(args.max_translation),
                "max_yaw": math.radians(float(args.max_yaw_deg)),
                "max_scale_delta": float(args.max_scale_delta),
                "converge_min_delta": float(args.converge_min_delta),
                "converge_patience": int(args.converge_patience),
                "log_every": int(args.log_every),
                "no_antialias": bool(args.no_antialias),
                "accept_worse_iou": bool(args.accept_worse_iou),
            }
        )
    meta = {
        "camera": camera_meta,
        "source_image_size": [int(source_size[0]), int(source_size[1])],
        "render_resolution": [int(render_width), int(render_height)],
        "intrinsics_normalized": [[intrinsics[0], 0.0, intrinsics[2]], [0.0, intrinsics[1], intrinsics[3]], [0.0, 0.0, 1.0]],
        "camera_to_world": c2w,
        "mesh_metadata": {
            "schema": metadata.get("schema"),
            "input": metadata.get("input"),
            "num_vertices": metadata.get("num_vertices"),
            "num_faces": metadata.get("num_faces"),
            "object_count": len(metadata.get("objects") or []),
        },
    }
    return jobs, meta


def main() -> int:
    args = parse_args()
    jobs, meta = build_jobs(args)
    if not jobs:
        raise RuntimeError("No optimizable objects found in manifest/mesh")
    if args.preview_dir:
        args.preview_dir.mkdir(parents=True, exist_ok=True)
    max_workers = len(jobs) if int(args.workers) <= 0 else min(int(args.workers), len(jobs))
    print(f"Optimizing {len(jobs)} objects with {max_workers} worker(s)", flush=True)

    results: list[dict[str, Any]] = []
    if max_workers == 1:
        for job in jobs:
            print(f"[object] {job['name']} mask_id={job['mask_id']}", flush=True)
            results.append(optimize_one_object(job))
    else:
        ctx = mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as pool:
            future_to_name = {pool.submit(optimize_one_object, job): job["name"] for job in jobs}
            for future in concurrent.futures.as_completed(future_to_name):
                name = future_to_name[future]
                result = future.result()
                print(f"[object] {name} -> {result.get('status')} iou={((result.get('metrics') or {}).get('optimized') or {}).get('iou')}", flush=True)
                results.append(result)

    order = {job["name"]: idx for idx, job in enumerate(jobs)}
    results.sort(key=lambda item: order.get(str(item.get("name")), 10**9))
    output = {
        "schema": "fysiverse_nvdiffrast_object_pose_optimization.v1",
        "status": "ok",
        "mesh": str(args.mesh),
        "target_mask": str(args.target_mask),
        "final_manifest": str(args.final_manifest),
        "input_image": str(args.input_image) if args.input_image else None,
        "preview_dir": str(args.preview_dir) if args.preview_dir else None,
        "parallel": {
            "workers": int(max_workers),
            "objects": len(jobs),
        },
        "optimization": {
            "position_steps": int(args.position_steps),
            "yaw_steps": int(args.yaw_steps),
            "scale_steps": int(args.scale_steps),
            "position_lr": float(args.position_lr),
            "yaw_lr": float(args.yaw_lr),
            "scale_lr": float(args.scale_lr),
            "dt_weight": float(args.dt_weight),
            "center_weight": float(args.center_weight),
            "area_weight": float(args.area_weight),
            "translation_prior_weight": float(args.translation_prior_weight),
            "yaw_prior_weight": float(args.yaw_prior_weight),
            "scale_prior_weight": float(args.scale_prior_weight),
            "translation_mode": str(args.translation_mode),
            "translation_initial_grid": int(args.translation_initial_grid),
            "yaw_initial_samples": int(args.yaw_initial_samples),
            "scale_initial_samples": int(args.scale_initial_samples),
            "phase_accept_iou_drop": float(args.phase_accept_iou_drop),
            "max_translation": float(args.max_translation),
            "max_yaw_degrees": float(args.max_yaw_deg),
            "max_scale_delta": float(args.max_scale_delta),
            "converge_min_delta": float(args.converge_min_delta),
            "converge_patience": int(args.converge_patience),
        },
        **meta,
        "objects": results,
        "coordinate_notes": {
            "translation": "World-space XYZ translation in Blender/SAPIEN Z-up coordinates.",
            "yaw": "Rotation around Blender/SAPIEN gravity axis +Z, applied after position optimization.",
            "scale": "Uniform scale around the exported world-space object bbox center, optimized after translation and yaw.",
            "delta_transform_world": "World-space transform T(pivot + translation) @ Rz(yaw) @ S(scale) @ T(-pivot).",
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(as_json(output), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote object pose optimization report: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
