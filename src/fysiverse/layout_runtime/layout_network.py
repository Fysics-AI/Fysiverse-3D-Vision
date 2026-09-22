# Layout inference implementation for Fysiverse-3D-Vision.
# The G2VLM and Pi3 dependencies retain their upstream licenses.
"""Inference-only G2VLM object-conditioned layout network."""

from copy import deepcopy
from contextlib import contextmanager
import json
import os
import sys
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn

from .g2vlm_inference import G2VLM
from .runtime_config import (
    LAYOUT_HEAD_ROTATION,
    LAYOUT_HEAD_SCALING,
    LAYOUT_HEAD_TRANSLATION,
    LOSS_RAW_MSE,
    normalize_layout_heads,
    resolve_layout_rotation_dim,
    validate_layout_loss_name,
)
from modeling.g2vlm.qwen2vl import NaiveCache
from modeling.pi3.models.dinov2.layers import Mlp
from modeling.pi3.models.layers.attention import FlashAttentionRope, FlashCrossAttentionRope
from modeling.pi3.models.layers.block import CrossBlockRope
from modeling.pi3.models.layers.camera_head import Pi3CameraHead, ResConvBlock


ImageSource = Union[str, Image.Image]
ConditionRegion = Tuple[float, float, float]
FILTERED_CONDITION_MODALITIES = frozenset({"image", "mask", "pointmap", "rgb_image_mask"})
DEFAULT_SAM3D_REPO_ROOT = ""
DEFAULT_SAM3D_PIPELINE_CONFIG = ""


def _sample_source_to_log_value(source: ImageSource) -> str:
    if isinstance(source, str):
        return source
    filename = getattr(source, "filename", None)
    if filename:
        return str(filename)
    return "<in-memory-image>"


def _is_skippable_condition_error(exc: Exception) -> bool:
    message = str(exc)
    if isinstance(exc, ValueError):
        return any(
            fragment in message
            for fragment in (
                "Mask contains no foreground pixels",
                "Mask contains no visible foreground pixels",
                "Mask becomes empty after resizing to pointmap resolution",
                "Mask selects no pointmap values after resizing",
                "Mask selects no finite pointmap values",
                "No valid points found in mask",
            )
        )
    if isinstance(exc, RuntimeError):
        return "input.numel() == 0" in message and "max()" in message
    return False


def _log_skipped_condition_sample(scene_image: ImageSource, mask_image: ImageSource, exc: Exception) -> None:
    rank_raw = os.environ.get("RANK")
    local_rank_raw = os.environ.get("LOCAL_RANK")
    payload: Dict[str, object] = {
        "event": "layout_sample_skipped",
        "scene_image_path": _sample_source_to_log_value(scene_image),
        "mask_image_path": _sample_source_to_log_value(mask_image),
        "error_type": exc.__class__.__name__,
        "error": str(exc),
    }
    if rank_raw is not None:
        payload["rank"] = int(rank_raw)
    if local_rank_raw is not None:
        payload["local_rank"] = int(local_rank_raw)
    print(json.dumps(payload, ensure_ascii=True), flush=True)


def _copy_linear_weight(dst: nn.Linear, src: nn.Linear) -> None:
    dst.weight.data.copy_(src.weight.data)
    if dst.bias is not None and src.bias is not None:
        dst.bias.data.copy_(src.bias.data)


def _copy_module_if_possible(dst: nn.Module, src: nn.Module) -> None:
    if isinstance(dst, nn.Identity) or isinstance(src, nn.Identity):
        return
    dst.load_state_dict(src.state_dict())


def _copy_cross_attention_from_self_attention(cross_attn: nn.Module, self_attn: nn.Module) -> None:
    q_weight, k_weight, v_weight = self_attn.qkv.weight.data.chunk(3, dim=0)
    cross_attn.q_proj.weight.data.copy_(q_weight)
    cross_attn.k_proj.weight.data.copy_(k_weight)
    cross_attn.v_proj.weight.data.copy_(v_weight)

    if self_attn.qkv.bias is not None:
        q_bias, k_bias, v_bias = self_attn.qkv.bias.data.chunk(3, dim=0)
        cross_attn.q_proj.bias.data.copy_(q_bias)
        cross_attn.k_proj.bias.data.copy_(k_bias)
        cross_attn.v_proj.bias.data.copy_(v_bias)

    _copy_module_if_possible(cross_attn.q_norm, self_attn.q_norm)
    _copy_module_if_possible(cross_attn.k_norm, self_attn.k_norm)
    _copy_linear_weight(cross_attn.proj, self_attn.proj)


def _ensure_rgb_image(image: ImageSource) -> Image.Image:
    if isinstance(image, Image.Image):
        pil_image = image
    else:
        pil_image = Image.open(image)

    if pil_image.mode == "RGBA":
        background = Image.new("RGBA", pil_image.size, (255, 255, 255, 255))
        pil_image = Image.alpha_composite(background, pil_image)

    return pil_image.convert("RGB")


def _resolve_path(path: str, workspace_dir: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(workspace_dir, path)


def _dtype_from_name(dtype_name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[dtype_name]


@contextmanager
def _offline_dinov2_hub(source_root: str, hub_dir: str):
    source_path = os.path.abspath(source_root)
    cache_path = os.path.abspath(hub_dir)
    if not os.path.isfile(os.path.join(source_path, "hubconf.py")):
        raise FileNotFoundError(f"DINOv2 source checkout is incomplete: {source_path}")
    os.makedirs(cache_path, exist_ok=True)
    original_load = torch.hub.load
    original_hub_dir = torch.hub.get_dir()

    def load_local(repo_or_dir, model, *args, **kwargs):
        normalized = str(repo_or_dir).rstrip("/")
        if normalized in {
            "facebookresearch/dinov2",
            "facebookresearch/dinov2:main",
            "https://github.com/facebookresearch/dinov2",
            "https://github.com/facebookresearch/dinov2.git",
        }:
            repo_or_dir = source_path
            kwargs["source"] = "local"
        return original_load(repo_or_dir, model, *args, **kwargs)

    torch.hub.set_dir(cache_path)
    torch.hub.load = load_local
    try:
        yield
    finally:
        torch.hub.load = original_load
        torch.hub.set_dir(original_hub_dir)


def _import_sam3d_runtime(repo_root: str) -> Dict[str, Any]:
    sam3d_root = Path(repo_root).resolve()
    runtime_source_roots = (sam3d_root, sam3d_root.parent / "MoGe")
    for source_root in reversed(runtime_source_roots):
        source_path = str(source_root)
        if source_path not in sys.path:
            sys.path.insert(0, source_path)
    # Layout imports only SAM3D's preprocessing and condition-encoder modules.
    # The upstream package initializer supports this flag to avoid unrelated
    # application-wide initialization before those modules are imported.
    os.environ.setdefault("LIDRA_SKIP_INIT", "true")

    try:
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        from pytorch3d.renderer import look_at_view_transform
        from pytorch3d.transforms import Transform3d

        from sam3d_objects.data.dataset.tdfy.img_and_mask_transforms import get_mask
        from sam3d_objects.data.dataset.tdfy.transforms_3d import DecomposedTransform
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "SAM3D condition encoder dependencies are unavailable in the current Python "
            f"environment. Missing module: {exc.name}. Install a new Layout environment "
            "with scripts/bootstrap_envs.sh --only layout."
        ) from exc

    # These two helpers are intentionally kept local. Importing their upstream
    # modules initializes SAM3D's complete mesh-generation pipeline, including
    # renderers and postprocessing packages that Layout condition encoding does
    # not execute.
    def filter_and_remove_prefix_state_dict_fn(prefix: str):
        prefix_length = len(prefix)

        def filter_state_dict(state_dict):
            return {
                key[prefix_length:]: value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }

        return filter_state_dict

    def camera_to_pytorch3d_camera(device: str | torch.device = "cpu"):
        rotation, translation = look_at_view_transform(
            eye=np.array([[0, 0, -1]]),
            at=np.array([[0, 0, 0]]),
            up=np.array([[0, -1, 0]]),
            device=device,
        )
        return DecomposedTransform(
            rotation=rotation,
            translation=translation,
            scale=torch.tensor(1.0, dtype=rotation.dtype, device=device),
        )

    return {
        "OmegaConf": OmegaConf,
        "Transform3d": Transform3d,
        "camera_to_pytorch3d_camera": camera_to_pytorch3d_camera,
        "filter_and_remove_prefix_state_dict_fn": filter_and_remove_prefix_state_dict_fn,
        "get_mask": get_mask,
        "instantiate": instantiate,
    }


def _extract_alpha_mask(mask: ImageSource, size: Optional[Tuple[int, int]] = None) -> Image.Image:
    if isinstance(mask, Image.Image):
        mask_image = mask
    else:
        mask_image = Image.open(mask)

    if size is not None and mask_image.size != size:
        mask_image = mask_image.resize(size, Image.Resampling.NEAREST)

    if "A" in mask_image.getbands():
        alpha = mask_image.getchannel("A")
        if np.asarray(alpha).max() > 0:
            return alpha

    if mask_image.mode == "L":
        gray = mask_image
    else:
        rgb = np.asarray(mask_image.convert("RGB"))
        binary = (rgb.max(axis=-1) > 0).astype(np.uint8) * 255
        gray = Image.fromarray(binary, mode="L")

    if np.asarray(gray).max() == 0:
        raise ValueError("Mask contains no visible foreground pixels.")
    return gray


def _resize_for_dino(image: Image.Image, target_width: int, patch_size: int) -> torch.Tensor:
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {image.size}")

    target_height = max(patch_size, round(height * (target_width / width) / patch_size) * patch_size)
    resized = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    return torchvision.transforms.functional.to_tensor(resized)


class SAM3DConditionEncoder(nn.Module):
    def __init__(
        self,
        repo_root: str,
        pipeline_config_path: str,
        device: torch.device,
    ):
        super().__init__()
        runtime = _import_sam3d_runtime(repo_root)
        pipeline_config_path = os.path.abspath(pipeline_config_path)
        workspace_dir = os.path.dirname(pipeline_config_path)

        pipeline_config = runtime["OmegaConf"].load(pipeline_config_path)
        dinov2_source_root = _resolve_path(str(pipeline_config["dinov2_source_root"]), workspace_dir)
        dinov2_torch_hub_dir = _resolve_path(str(pipeline_config["dinov2_torch_hub_dir"]), workspace_dir)
        dinov2_weights_path = _resolve_path(str(pipeline_config["dinov2_weights_path"]), workspace_dir)
        if not os.path.isfile(dinov2_weights_path):
            raise FileNotFoundError(f"DINOv2 offline weights not found: {dinov2_weights_path}")
        ss_generator_config_path = _resolve_path(str(pipeline_config["ss_generator_config_path"]), workspace_dir)
        ss_generator_ckpt_path = _resolve_path(str(pipeline_config["ss_generator_ckpt_path"]), workspace_dir)

        self.sam3d_dtype = _dtype_from_name(str(pipeline_config.get("dtype", "float16")))
        self.crop_box_size_factor = 1.0
        self.crop_padding_factor = 0.0
        ss_preprocessor_config = pipeline_config.get("ss_preprocessor", {})
        for transform_config in ss_preprocessor_config.get("img_mask_pointmap_joint_transform", []):
            target_name = str(transform_config.get("_target_", ""))
            if target_name.endswith("crop_around_mask_with_padding"):
                self.crop_box_size_factor = float(transform_config.get("box_size_factor", 1.0))
                self.crop_padding_factor = float(transform_config.get("padding_factor", 0.0))
                break
        self.ss_preprocessor = runtime["instantiate"](pipeline_config["ss_preprocessor"])
        depth_weights = pipeline_config.get("depth_model", {}).get("model", {}).get(
            "pretrained_model_name_or_path"
        )
        if depth_weights:
            pipeline_config["depth_model"]["model"]["pretrained_model_name_or_path"] = _resolve_path(
                str(depth_weights), workspace_dir
            )
        with _offline_dinov2_hub(dinov2_source_root, dinov2_torch_hub_dir):
            self.depth_model = runtime["instantiate"](pipeline_config["depth_model"], device=str(device))
            condition_config = runtime["OmegaConf"].load(ss_generator_config_path)["module"]["condition_embedder"]["backbone"]
            self.condition_encoder = runtime["instantiate"](condition_config)
        if hasattr(self.depth_model, "model"):
            self.depth_model.model.requires_grad_(False)
            self.depth_model.model.eval()

        artifact = torch.load(ss_generator_ckpt_path, map_location="cpu", weights_only=True)
        state_dict = artifact["state_dict"]
        state_dict = runtime["filter_and_remove_prefix_state_dict_fn"]("_base_models.condition_embedder.")(state_dict)
        self.condition_encoder.load_state_dict(state_dict, strict=True)
        self.condition_encoder.requires_grad_(False)
        self.condition_encoder.eval()
        self.condition_encoder.to(device)

        self._Transform3d = runtime["Transform3d"]
        self._camera_to_pytorch3d_camera = runtime["camera_to_pytorch3d_camera"]
        self._get_mask = runtime["get_mask"]
        self._restrict_to_object_modalities()

    @property
    def device(self) -> torch.device:
        return next(self.condition_encoder.parameters()).device

    @property
    def output_dim(self) -> int:
        return int(self.condition_encoder.embed_dims)

    def _restrict_to_object_modalities(self) -> None:
        filtered_embedder_list = []
        for condition_embedder, kwargs_info in getattr(self.condition_encoder, "embedder_list", []):
            filtered_kwargs_info = [
                (kwarg_name, pos_group)
                for kwarg_name, pos_group in kwargs_info
                if kwarg_name in FILTERED_CONDITION_MODALITIES
            ]
            if filtered_kwargs_info:
                filtered_embedder_list.append((condition_embedder, filtered_kwargs_info))

        if not filtered_embedder_list:
            raise ValueError("No object-level condition modalities remain after filtering the SAM3D encoder.")

        self.condition_encoder.embedder_list = filtered_embedder_list
        self.condition_encoder.drop_modalities_weight = [
            (modalities, weight)
            for modalities, weight in getattr(self.condition_encoder, "drop_modalities_weight", [])
            if all(modality in FILTERED_CONDITION_MODALITIES for modality in modalities)
        ]
        if getattr(self.condition_encoder, "force_drop_modalities", None):
            self.condition_encoder.force_drop_modalities = [
                modality
                for modality in self.condition_encoder.force_drop_modalities
                if modality in FILTERED_CONDITION_MODALITIES
            ]

    def _autocast_context(self):
        return torch.amp.autocast(
            device_type=self.device.type if self.device.type in {"cuda", "cpu"} else "cpu",
            enabled=self.device.type == "cuda",
            dtype=self.sam3d_dtype,
        )

    def _image_to_float(self, image: np.ndarray) -> np.ndarray:
        return image.astype(np.float32) / 255.0

    def _build_rgba_image(self, scene_image: ImageSource, mask_image: ImageSource) -> np.ndarray:
        rgb_image = np.asarray(_ensure_rgb_image(scene_image), dtype=np.uint8)
        alpha_mask = np.asarray(
            _extract_alpha_mask(mask_image, size=(rgb_image.shape[1], rgb_image.shape[0])),
            dtype=np.uint8,
        )
        if alpha_mask.max() <= 0:
            raise ValueError(f"Mask contains no foreground pixels: {mask_image}")
        return np.concatenate([rgb_image, alpha_mask[..., None]], axis=-1)

    @torch.no_grad()
    def _compute_pointmap(self, rgba_image: np.ndarray) -> torch.Tensor:
        loaded_image = torch.from_numpy(self._image_to_float(rgba_image))
        loaded_rgb = loaded_image.permute(2, 0, 1).contiguous()[:3]
        with self._autocast_context():
            output = self.depth_model(loaded_rgb)

        pointmaps = output["pointmaps"]
        camera_convention_transform = (
            self._Transform3d()
            .rotate(self._camera_to_pytorch3d_camera(device=self.device).rotation)
            .to(self.device)
        )
        points_tensor = camera_convention_transform.transform_points(pointmaps)
        return points_tensor.permute(2, 0, 1).contiguous().detach().cpu()

    def _preprocess_condition_inputs(self, rgba_image: np.ndarray, pointmap: torch.Tensor) -> Dict[str, torch.Tensor]:
        rgba_tensor = torch.from_numpy(self._image_to_float(rgba_image))
        rgba_tensor = rgba_tensor.permute(2, 0, 1).contiguous()
        rgb_image = rgba_tensor[:3]
        rgb_image_mask = self._get_mask(rgba_tensor, None, "ALPHA_CHANNEL")
        self._validate_condition_mask_pointmap(rgb_image_mask, pointmap)
        preprocessor_output = self.ss_preprocessor._process_image_mask_pointmap_mess(
            rgb_image,
            rgb_image_mask,
            pointmap,
        )

        item: Dict[str, torch.Tensor] = {
            "mask": preprocessor_output["mask"][None].to(self.device),
            "image": preprocessor_output["image"][None].to(self.device),
            "rgb_image_mask": preprocessor_output["rgb_image_mask"][None].to(self.device),
        }

        for key in (
            "pointmap",
            "pointmap_scale",
            "pointmap_shift",
        ):
            if key in preprocessor_output:
                item[key] = preprocessor_output[key][None].to(self.device)
        return item

    def _validate_condition_mask_pointmap(self, mask: torch.Tensor, pointmap: torch.Tensor) -> None:
        pointmap_size = (pointmap.shape[1], pointmap.shape[2])
        mask_resized = torchvision.transforms.functional.resize(
            mask,
            pointmap_size,
            interpolation=torchvision.transforms.InterpolationMode.NEAREST,
        ).squeeze(0)
        mask_bool = mask_resized.reshape(-1) > 0.5
        if not bool(mask_bool.any().item()):
            raise ValueError("Mask becomes empty after resizing to pointmap resolution.")

        mask_points = pointmap.reshape(pointmap.shape[0], -1)[:, mask_bool]
        if mask_points.numel() == 0:
            raise ValueError("Mask selects no pointmap values after resizing.")
        if not bool(mask_points.isfinite().any().item()):
            raise ValueError("Mask selects no finite pointmap values.")

    def _build_full_scene_region(self, image_height: int, image_width: int) -> ConditionRegion:
        square_size = float(max(image_height, image_width))
        offset_x = -0.5 * (square_size - float(image_width))
        offset_y = -0.5 * (square_size - float(image_height))
        return (offset_x, offset_y, square_size)

    def _build_cropped_object_region(self, alpha_mask: np.ndarray) -> ConditionRegion:
        ys, xs = np.nonzero(alpha_mask > 0)
        if len(xs) == 0 or len(ys) == 0:
            raise ValueError("Mask contains no visible foreground pixels after scene alignment.")

        min_x = float(xs.min())
        min_y = float(ys.min())
        max_x = float(xs.max())
        max_y = float(ys.max())
        center_x = 0.5 * (min_x + max_x)
        center_y = 0.5 * (min_y + max_y)
        bbox_w = max_x - min_x
        bbox_h = max_y - min_y
        square_size = float(int(max(bbox_w, bbox_h, 2.0) * self.crop_box_size_factor))
        half_extent = float(int(square_size) // 2)
        x0 = float(int(center_x - half_extent))
        y0 = float(int(center_y - half_extent))
        x1 = float(int(center_x + half_extent))
        y1 = float(int(center_y + half_extent))
        region_size = max(x1 - x0, y1 - y0, 1.0)

        if self.crop_padding_factor > 0:
            extend_size = float(int(region_size * self.crop_padding_factor))
            x0 -= extend_size
            y0 -= extend_size
            region_size += 2.0 * extend_size

        return (x0, y0, region_size)

    def _scene_coords_to_patch_positions(
        self,
        x_coords: torch.Tensor,
        y_coords: torch.Tensor,
        *,
        image_height: int,
        image_width: int,
        scene_patch_shape: Tuple[int, int],
    ) -> torch.Tensor:
        patch_h, patch_w = scene_patch_shape
        max_x = max(float(image_width) - 1e-6, 0.0)
        max_y = max(float(image_height) - 1e-6, 0.0)
        x_coords = x_coords.clamp(0.0, max_x)
        y_coords = y_coords.clamp(0.0, max_y)

        x_pos = torch.floor(x_coords * (float(patch_w) / max(float(image_width), 1.0))).to(dtype=torch.long)
        y_pos = torch.floor(y_coords * (float(patch_h) / max(float(image_height), 1.0))).to(dtype=torch.long)
        x_pos = x_pos.clamp_(0, patch_w - 1)
        y_pos = y_pos.clamp_(0, patch_h - 1)
        return torch.stack((y_pos, x_pos), dim=-1)

    def _positions_for_region_grid(
        self,
        grid_height: int,
        grid_width: int,
        *,
        region: ConditionRegion,
        image_height: int,
        image_width: int,
        scene_patch_shape: Tuple[int, int],
        include_cls: bool,
        cls_anchor: Tuple[float, float],
        device: torch.device,
    ) -> torch.Tensor:
        region_x0, region_y0, region_size = region
        grid = torch.cartesian_prod(
            torch.arange(grid_height, device=device, dtype=torch.float32),
            torch.arange(grid_width, device=device, dtype=torch.float32),
        )
        y_coords = region_y0 + (grid[:, 0] + 0.5) * (region_size / float(grid_height))
        x_coords = region_x0 + (grid[:, 1] + 0.5) * (region_size / float(grid_width))
        positions = self._scene_coords_to_patch_positions(
            x_coords,
            y_coords,
            image_height=image_height,
            image_width=image_width,
            scene_patch_shape=scene_patch_shape,
        )

        if include_cls:
            cls_x = torch.tensor([cls_anchor[0]], device=device, dtype=torch.float32)
            cls_y = torch.tensor([cls_anchor[1]], device=device, dtype=torch.float32)
            cls_position = self._scene_coords_to_patch_positions(
                cls_x,
                cls_y,
                image_height=image_height,
                image_width=image_width,
                scene_patch_shape=scene_patch_shape,
            )
            positions = torch.cat((cls_position, positions), dim=0)

        return positions.unsqueeze(0)

    def _positions_for_condition_embedder(
        self,
        embedder: nn.Module,
        *,
        region: ConditionRegion,
        image_height: int,
        image_width: int,
        scene_patch_shape: Tuple[int, int],
        cls_anchor: Tuple[float, float],
        device: torch.device,
    ) -> torch.Tensor:
        if hasattr(embedder, "backbone") and hasattr(embedder.backbone, "patch_embed"):
            patch_size = embedder.backbone.patch_embed.patch_size
            if isinstance(patch_size, tuple):
                patch_size = patch_size[0]
            grid_size = int(embedder.input_size) // int(patch_size)
            return self._positions_for_region_grid(
                grid_size,
                grid_size,
                region=region,
                image_height=image_height,
                image_width=image_width,
                scene_patch_shape=scene_patch_shape,
                include_cls=True,
                cls_anchor=cls_anchor,
                device=device,
            )

        if hasattr(embedder, "input_size") and hasattr(embedder, "patch_size"):
            grid_size = int(embedder.input_size) // int(embedder.patch_size)
            return self._positions_for_region_grid(
                grid_size,
                grid_size,
                region=region,
                image_height=image_height,
                image_width=image_width,
                scene_patch_shape=scene_patch_shape,
                include_cls=False,
                cls_anchor=cls_anchor,
                device=device,
            )

        raise TypeError(f"Unsupported SAM3D condition embedder type: {type(embedder)!r}")

    def _build_condition_positions(
        self,
        rgba_image: np.ndarray,
        scene_patch_shape: Tuple[int, int],
        device: torch.device,
    ) -> torch.Tensor:
        if getattr(self.condition_encoder, "compression_projection_multiplier", 0) > 0:
            raise ValueError("Compressed SAM3D condition embedders are not supported for scene-aligned positions.")

        image_height, image_width = rgba_image.shape[:2]
        alpha_mask = rgba_image[..., 3]
        full_region = self._build_full_scene_region(image_height, image_width)
        cropped_region = self._build_cropped_object_region(alpha_mask)
        cls_anchor = (
            cropped_region[0] + 0.5 * cropped_region[2],
            cropped_region[1] + 0.5 * cropped_region[2],
        )

        position_chunks: List[torch.Tensor] = []
        for condition_embedder, kwargs_info in getattr(self.condition_encoder, "embedder_list", []):
            for kwarg_name, pos_group in kwargs_info:
                region = full_region if kwarg_name == "rgb_image_mask" or pos_group == "full" else cropped_region
                position_chunks.append(
                    self._positions_for_condition_embedder(
                        condition_embedder,
                        region=region,
                        image_height=image_height,
                        image_width=image_width,
                        scene_patch_shape=scene_patch_shape,
                        cls_anchor=cls_anchor,
                        device=device,
                    )
                )

        if not position_chunks:
            raise ValueError("SAM3D condition encoder produced no positional chunks.")
        return torch.cat(position_chunks, dim=1)

    @torch.no_grad()
    def forward(
        self,
        scene_image: ImageSource,
        mask_image: ImageSource,
        scene_patch_shape: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rgba_image = self._build_rgba_image(scene_image, mask_image)
        pointmap = self._compute_pointmap(rgba_image)
        condition_inputs = self._preprocess_condition_inputs(rgba_image, pointmap)
        with self._autocast_context():
            condition_tokens = self.condition_encoder(**condition_inputs)
        condition_tokens = condition_tokens.detach()
        condition_positions = self._build_condition_positions(rgba_image, scene_patch_shape, condition_tokens.device)
        if condition_positions.shape[1] != condition_tokens.shape[1]:
            raise ValueError(
                "Condition position/token length mismatch: "
                f"{condition_positions.shape[1]} vs {condition_tokens.shape[1]}"
            )
        return condition_tokens, condition_positions


class Pi3HybridLayoutDecoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dec_embed_dim: int = 512,
        depth: int = 5,
        dec_num_heads: int = 8,
        mlp_ratio: float = 4.0,
        rope=None,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                CrossBlockRope(
                    dim=dec_embed_dim,
                    num_heads=dec_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    act_layer=nn.GELU,
                    ffn_layer=Mlp,
                    init_values=None,
                    qk_norm=False,
                    attn_class=FlashAttentionRope,
                    cross_attn_class=FlashCrossAttentionRope,
                    rope=rope,
                )
                for _ in range(depth)
            ]
        )

        self.linear_out = nn.Linear(dec_embed_dim, out_dim)

    def init_from_camera_decoder(self, camera_decoder: nn.Module) -> None:
        _copy_linear_weight(self.linear_out, camera_decoder.linear_out)
        for idx, dst_block in enumerate(self.blocks):
            src_block = camera_decoder.blocks[idx]
            _copy_module_if_possible(dst_block.ls1, src_block.ls1)
            _copy_module_if_possible(dst_block.ls2, src_block.ls2)
            _copy_module_if_possible(dst_block.ls_y, src_block.ls1)
            _copy_module_if_possible(dst_block.norm1, src_block.norm1)
            _copy_module_if_possible(dst_block.norm2, src_block.norm2)
            _copy_module_if_possible(dst_block.norm3, src_block.norm2)
            _copy_module_if_possible(dst_block.norm_y, src_block.norm2)
            dst_block.attn.load_state_dict(src_block.attn.state_dict())
            _copy_cross_attention_from_self_attention(dst_block.cross_attn, src_block.attn)
            dst_block.mlp.load_state_dict(src_block.mlp.state_dict())

    def forward(
        self,
        hidden: torch.Tensor,
        context: torch.Tensor,
        xpos: Optional[torch.Tensor] = None,
        ypos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for block in self.blocks:
            hidden = block(hidden, context, xpos=xpos, ypos=ypos)
        return self.linear_out(hidden)


class Pi3LayoutHead(nn.Module):
    def __init__(
        self,
        dim: int = 512,
        loss_name: str = LOSS_RAW_MSE,
        rotation_dim: int | None = None,
        layout_heads: Sequence[str] | str | None = None,
    ):
        super().__init__()
        self.loss_name = validate_layout_loss_name(loss_name)
        self.rotation_dim = resolve_layout_rotation_dim(rotation_dim, loss_name=self.loss_name)
        self.layout_heads = normalize_layout_heads(layout_heads)
        self.output_dim = dim
        self.res_conv = nn.ModuleList([ResConvBlock(dim, dim) for _ in range(2)])
        self.more_mlps = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )
        self.fc_translation = nn.Linear(dim, 3) if LAYOUT_HEAD_TRANSLATION in self.layout_heads else None
        self.fc_rotation = nn.Linear(dim, self.rotation_dim) if LAYOUT_HEAD_ROTATION in self.layout_heads else None
        self.fc_scaling = nn.Linear(dim, 1) if LAYOUT_HEAD_SCALING in self.layout_heads else None

    def init_from_camera_head(self, camera_head: Pi3CameraHead) -> None:
        self.res_conv.load_state_dict(camera_head.res_conv.state_dict())
        self.more_mlps.load_state_dict(camera_head.more_mlps.state_dict())
        if self.fc_translation is not None:
            _copy_linear_weight(self.fc_translation, camera_head.fc_t)
        if self.fc_rotation is not None:
            if self.rotation_dim == 9:
                _copy_linear_weight(self.fc_rotation, camera_head.fc_rot)
            elif self.rotation_dim in (10, 12):
                self.fc_rotation.weight.data[:9].copy_(camera_head.fc_rot.weight.data)
                self.fc_rotation.bias.data[:9].copy_(camera_head.fc_rot.bias.data)
            elif self.rotation_dim == 2:
                # Legacy norm_2D_raw checkpoints predict [sin(yaw), cos(yaw)].
                self.fc_rotation.weight.data[0].copy_(camera_head.fc_rot.weight.data[6])
                self.fc_rotation.weight.data[1].copy_(camera_head.fc_rot.weight.data[0])
                self.fc_rotation.bias.data[0].copy_(camera_head.fc_rot.bias.data[6])
                self.fc_rotation.bias.data[1].copy_(camera_head.fc_rot.bias.data[0])
        # rotation_dim=1/3 has no direct counterpart in the base camera head,
        # so fc_rotation keeps its default initialization.

        if self.fc_scaling is not None:
            self.fc_scaling.weight.data.copy_(camera_head.fc_t.weight.data.mean(dim=0, keepdim=True))
            mean_bias = camera_head.fc_t.bias.data.mean().view(1)
            self.fc_scaling.bias.data.copy_(mean_bias)

    def forward(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        for block in self.res_conv:
            feat = block(feat)

        # Object-centric decoding does not preserve a dense scene grid, so pool across query tokens directly.
        feat = feat.mean(dim=1)
        batch_size = feat.shape[0]
        feat = self.more_mlps(feat)

        with torch.amp.autocast(
            device_type=feat.device.type if feat.device.type in {"cuda", "cpu"} else "cpu",
            enabled=False,
        ):
            feat = feat.float()
            outputs: Dict[str, torch.Tensor] = {}
            if self.fc_translation is not None:
                outputs["translation"] = self.fc_translation(feat)
            if self.fc_rotation is not None:
                rotation = self.fc_rotation(feat)
                if self.rotation_dim == 9:
                    rotation = rotation.reshape(batch_size, 3, 3)
                outputs["rotation"] = rotation
            if self.fc_scaling is not None:
                outputs["scaling"] = self.fc_scaling(feat)

        return outputs


class G2VLMLayout(nn.Module):
    def __init__(
        self,
        base_model: G2VLM,
        tokenizer,
        new_token_ids: Dict[str, int],
        loss_name: str = LOSS_RAW_MSE,
        rotation_dim: int | None = None,
        layout_heads: Sequence[str] | str | None = None,
        prompt: str = "Predict the masked object's layout in the scene.",
        control_padding_ratio: float = 0.15,
        sam3d_repo_root: str = DEFAULT_SAM3D_REPO_ROOT,
        sam3d_pipeline_config: str = DEFAULT_SAM3D_PIPELINE_CONFIG,
    ):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.new_token_ids = new_token_ids
        self.loss_name = validate_layout_loss_name(loss_name)
        self.rotation_dim = resolve_layout_rotation_dim(rotation_dim, loss_name=self.loss_name)
        self.layout_heads = normalize_layout_heads(layout_heads)
        self.prompt = prompt
        self.control_padding_ratio = control_padding_ratio
        self.sam3d_repo_root = sam3d_repo_root
        self.sam3d_pipeline_config = sam3d_pipeline_config

        if self.base_model.use_registers:
            raise NotImplementedError("G2VLMLayout currently expects use_registers=False.")

        self.patch_size = 16 if self.base_model.use_dinov3 else 14
        self.target_width = 512 if self.base_model.use_dinov3 else 518
        base_device = next(self.base_model.parameters()).device

        self.layout_decoder = Pi3HybridLayoutDecoder(
            in_dim=self.base_model.hidden_size,
            dec_embed_dim=self.base_model.hidden_size,
            dec_num_heads=16,
            out_dim=512,
            rope=self.base_model.pi3rope,
        )
        self.layout_head = Pi3LayoutHead(
            dim=512,
            loss_name=self.loss_name,
            rotation_dim=self.rotation_dim,
            layout_heads=self.layout_heads,
        )
        self.layout_decoder.init_from_camera_decoder(self.base_model.camera_decoder)
        self.layout_head.init_from_camera_head(self.base_model.camera_head)
        self.condition2llm = deepcopy(self.base_model.dino2llm)
        self.sam3d_condition_encoder = SAM3DConditionEncoder(
            repo_root=self.sam3d_repo_root,
            pipeline_config_path=self.sam3d_pipeline_config,
            device=base_device,
        )
        if self.sam3d_condition_encoder.output_dim != self.base_model.dino2llm.in_features:
            raise ValueError(
                "SAM3D condition encoder output dim does not match G2VLM dino2llm input dim: "
                f"{self.sam3d_condition_encoder.output_dim} vs {self.base_model.dino2llm.in_features}"
            )
        self.requires_grad_(False)
        self.base_model.eval()
        self.sam3d_condition_encoder.eval()

    def load_layout_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        modules = {
            "condition2llm.": self.condition2llm,
            "layout_decoder.": self.layout_decoder,
            "layout_head.": self.layout_head,
        }
        unexpected = [key for key in state_dict if not key.startswith(tuple(modules))]
        if unexpected:
            raise ValueError(f"Unexpected layout tensor key: {unexpected[0]}")
        for prefix, module in modules.items():
            section = {key[len(prefix):]: value for key, value in state_dict.items() if key.startswith(prefix)}
            if not section:
                raise ValueError(f"Layout weights contain no tensors with prefix {prefix}")
            module.load_state_dict(section, strict=True)

    @property
    def device(self) -> torch.device:
        return next(self.layout_decoder.parameters()).device

    def _move_tensor_dict(self, tensor_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        output = {}
        for key, value in tensor_dict.items():
            if torch.is_tensor(value):
                output[key] = value.to(self.device)
            else:
                output[key] = value
        return output

    def _autocast_context(self):
        return torch.amp.autocast(
            device_type=self.device.type if self.device.type in {"cuda", "cpu"} else "cpu",
            enabled=self.device.type == "cuda",
            dtype=torch.bfloat16,
        )

    def _load_image_tensor(self, image: ImageSource) -> torch.Tensor:
        pil_image = _ensure_rgb_image(image)
        return _resize_for_dino(pil_image, self.target_width, self.patch_size)

    @torch.no_grad()
    def _encode_scene_hidden(
        self,
        scene_image: ImageSource,
        prompt: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        scene_tensor = self._load_image_tensor(scene_image).unsqueeze(0)
        prompt = prompt or self.prompt
        past_key_values = NaiveCache(self.base_model.config.llm_config.num_hidden_layers)
        curr_kvlens = [0]
        curr_rope = [0]

        prompt_inputs, curr_kvlens, curr_rope = self.base_model.prepare_prompts_addbos(
            curr_kvlens=curr_kvlens,
            curr_rope=curr_rope,
            prompts=[prompt],
            tokenizer=self.tokenizer,
            new_token_ids=self.new_token_ids,
        )
        prompt_inputs = self._move_tensor_dict(prompt_inputs)
        with self._autocast_context():
            past_key_values = self.base_model.forward_cache_update_text(past_key_values, **prompt_inputs)

        dino_inputs, _, _ = self.base_model.prepare_dino_images_none(
            curr_kvlens=curr_kvlens,
            curr_rope=curr_rope,
            images=scene_tensor,
            transforms=None,
            new_token_ids=self.new_token_ids,
        )
        dino_inputs = self._move_tensor_dict(dino_inputs)
        with self._autocast_context():
            _, last_hidden_state = self.base_model.forward_cache_update_dino(past_key_values, **dino_inputs)

        hidden = last_hidden_state[dino_inputs["packed_dino_token_indexes"]].reshape(1, -1, self.base_model.hidden_size)
        _, _, height, width = dino_inputs["packed_dino_images"].shape
        patch_h, patch_w = height // self.patch_size, width // self.patch_size
        pos = self.base_model.position_getter(1, patch_h, patch_w, hidden.device).reshape(1, -1, 2)
        return hidden, pos, patch_h, patch_w

    def _encode_object_query_hidden(
        self,
        scene_image: ImageSource,
        mask_image: ImageSource,
        scene_patch_shape: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        object_hidden, object_pos = self.sam3d_condition_encoder(
            scene_image,
            mask_image,
            scene_patch_shape,
        )
        projector_dtype = self.condition2llm.weight.dtype
        with self._autocast_context():
            object_hidden = self.condition2llm(
                object_hidden.reshape(-1, object_hidden.shape[-1]).to(dtype=projector_dtype)
            ).reshape(object_hidden.shape[0], object_hidden.shape[1], -1)
        object_pos = object_pos.to(object_hidden.device)
        return object_hidden, object_pos

    def _predict_single(
        self,
        scene_image: ImageSource,
        mask_image: ImageSource,
        prompt: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        scene_hidden, scene_pos, patch_h, patch_w = self._encode_scene_hidden(scene_image, prompt=prompt)
        object_hidden, object_pos = self._encode_object_query_hidden(scene_image, mask_image, (patch_h, patch_w))

        decoder_dtype = self.layout_decoder.linear_out.weight.dtype
        scene_hidden = scene_hidden.to(dtype=decoder_dtype)
        object_hidden = object_hidden.to(dtype=decoder_dtype)

        layout_hidden = self.layout_decoder(object_hidden, scene_hidden, xpos=object_pos, ypos=scene_pos)
        return self.layout_head(layout_hidden)

    def forward(
        self,
        scene_images: Sequence[ImageSource],
        mask_images: Sequence[ImageSource],
        prompt: Optional[str] = None,
        *,
        skip_invalid_samples: bool = False,
        return_valid_indices: bool = False,
    ) -> Union[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], List[int]]]:
        if isinstance(scene_images, (str, Image.Image)):
            scene_images = [scene_images]
        if isinstance(mask_images, (str, Image.Image)):
            mask_images = [mask_images]

        if len(scene_images) != len(mask_images):
            raise ValueError("scene_images and mask_images must have the same length.")

        outputs: List[Dict[str, torch.Tensor]] = []
        valid_indices: List[int] = []
        for index, (scene, mask) in enumerate(zip(scene_images, mask_images)):
            try:
                outputs.append(self._predict_single(scene, mask, prompt=prompt))
                valid_indices.append(index)
            except Exception as exc:
                if not skip_invalid_samples or not _is_skippable_condition_error(exc):
                    raise
                _log_skipped_condition_sample(scene, mask, exc)

        if not outputs:
            if return_valid_indices:
                return {}, valid_indices
            return {}

        merged: Dict[str, torch.Tensor] = {}
        for key in outputs[0]:
            merged[key] = torch.cat([item[key] for item in outputs], dim=0)
        if return_valid_indices:
            return merged, valid_indices
        return merged
