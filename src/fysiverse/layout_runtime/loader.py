"""Load the G2VLM base and public inference-only layout weights."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def _resolve_package_file(model_dir: Path, name: str, label: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or relative.name != name:
        raise ValueError(f"{label} must be a package-root filename: {name!r}")
    path = model_dir / relative
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _resolve_backbone_shards(
    model_dir: Path, index_name: str
) -> tuple[dict[str, str], dict[str, Path]]:
    index_path = _resolve_package_file(model_dir, index_name, "backbone index")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"Invalid or empty weight_map in {index_path}")
    normalized_map = {str(key): str(value) for key, value in weight_map.items()}
    shards = {
        shard_name: _resolve_package_file(model_dir, shard_name, "backbone shard")
        for shard_name in sorted(set(normalized_map.values()))
    }
    return normalized_map, shards


def load_layout_model(
    *,
    g2vlm_source_root: Path,
    base_model_dir: Path,
    layout_model_dir: Path,
    sam3d_source_root: Path,
    sam3d_pipeline_config: Path,
    device: str,
) -> tuple[Any, Any]:
    source_root = g2vlm_source_root.resolve()
    for path, label in (
        (source_root, "G2VLM source directory"),
        (base_model_dir.resolve(), "G2VLM base model directory"),
        (layout_model_dir.resolve(), "layout model directory"),
        (sam3d_source_root.resolve(), "SAM3D source directory"),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"{label} not found: {path}")
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

    import torch
    from data.data_utils import add_special_tokens
    from modeling.g2vlm.dinov2_model import Dinov2WithRegistersConfig, Dinov2WithRegistersModel
    from modeling.g2vlm.qwen2vl import Qwen2VLConfig, Qwen2VLForCausalLM
    from modeling.qwen2 import Qwen2Tokenizer
    from modeling.qwen2vl.configuration_qwen2_vl import Qwen2VLVisionConfig
    from modeling.qwen2vl.modeling_qwen2_vl import Qwen2VisionTransformerPretrainedModel
    from safetensors.torch import load_file

    from .g2vlm_inference import G2VLM, G2VLMConfig
    from .layout_network import G2VLMLayout
    from .runtime_config import LayoutRuntimeConfig

    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for layout inference but PyTorch cannot see a CUDA device")

    base_dir = base_model_dir.resolve()
    layout_dir = layout_model_dir.resolve()
    config = LayoutRuntimeConfig.from_file(layout_dir / "config.json")
    required_base_files = (
        config.base_model_weights,
        "text_config.json",
        "vit_config.json",
        "dino_config.json",
        "vocab.json",
        "merges.txt",
    )
    for name in required_base_files:
        if not (base_dir / name).is_file():
            raise FileNotFoundError(f"G2VLM base model file not found: {base_dir / name}")

    llm_config = Qwen2VLConfig.from_json_file(str(base_dir / "text_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2VLMoTDecoderLayer"
    vit_config = Qwen2VLVisionConfig.from_json_file(str(base_dir / "vit_config.json"))
    vit_config.patch_size = 14
    dino_config = Dinov2WithRegistersConfig.from_json_file(str(base_dir / "dino_config.json"))
    g2vlm_config = G2VLMConfig(
        visual_und=True,
        visual_recon=True,
        llm_config=llm_config,
        vit_config=vit_config,
        dino_config=dino_config,
        vit_max_num_patch_per_side=36,
    )
    base_model = G2VLM(
        Qwen2VLForCausalLM(llm_config),
        Qwen2VisionTransformerPretrainedModel(vit_config),
        Dinov2WithRegistersModel(dino_config),
        g2vlm_config,
    )
    tokenizer = Qwen2Tokenizer.from_pretrained(str(base_dir))
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    base_state = load_file(str(base_dir / config.base_model_weights), device="cpu")
    replacement_map, replacement_shards = _resolve_backbone_shards(
        layout_dir, config.backbone_index
    )
    if len(replacement_map) != config.backbone_tensor_count:
        raise ValueError(
            "Backbone index tensor count does not match config: "
            f"expected {config.backbone_tensor_count}, found {len(replacement_map)}"
        )
    replacement_count = 0
    loaded_replacement_keys: set[str] = set()
    for shard_name, shard_path in replacement_shards.items():
        replacements = load_file(str(shard_path), device="cpu")
        shard_keys = set(replacements)
        expected_keys = {
            key for key, indexed_shard in replacement_map.items() if indexed_shard == shard_name
        }
        if shard_keys != expected_keys:
            missing = sorted(expected_keys - shard_keys)
            unexpected = sorted(shard_keys - expected_keys)
            detail = missing[0] if missing else unexpected[0]
            kind = "missing" if missing else "unexpected"
            raise ValueError(f"Backbone shard {shard_name} has {kind} tensor key: {detail}")
        duplicate_keys = loaded_replacement_keys & shard_keys
        if duplicate_keys:
            raise ValueError(
                f"Backbone replacement tensor occurs in multiple shards: {sorted(duplicate_keys)[0]}"
            )
        base_state.update(replacements)
        replacement_count += len(replacements)
        loaded_replacement_keys.update(shard_keys)
        del replacements
    if replacement_count != config.backbone_tensor_count:
        raise ValueError(
            "Loaded backbone tensor count does not match config: "
            f"expected {config.backbone_tensor_count}, found {replacement_count}"
        )
    print(
        json.dumps(
            {"event": "layout_backbone_replacements", "tensor_count": replacement_count},
            ensure_ascii=True,
        )
    )
    incompatible = base_model.load_state_dict(base_state, strict=False)
    del base_state
    if incompatible.unexpected_keys:
        raise ValueError(f"Unexpected G2VLM base tensor key: {incompatible.unexpected_keys[0]}")
    if incompatible.missing_keys:
        print(
            json.dumps(
                {"event": "g2vlm_missing_base_keys", "count": len(incompatible.missing_keys)},
                ensure_ascii=True,
            )
        )
    base_model.to(resolved_device)
    base_model.eval()

    model = G2VLMLayout(
        base_model=base_model,
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
        loss_name="raw_mse",
        rotation_dim=9,
        layout_heads=config.layout_heads,
        prompt=config.prompt,
        control_padding_ratio=config.control_padding_ratio,
        sam3d_repo_root=str(sam3d_source_root.resolve()),
        sam3d_pipeline_config=str(sam3d_pipeline_config.resolve()),
    ).to(resolved_device)
    weights_path = _resolve_package_file(layout_dir, config.weights, "layout weights")
    layout_state = load_file(str(weights_path), device="cpu")
    if len(layout_state) != config.layout_tensor_count:
        raise ValueError(
            "Loaded layout tensor count does not match config: "
            f"expected {config.layout_tensor_count}, found {len(layout_state)}"
        )
    model.load_layout_state_dict(layout_state)
    del layout_state
    model.eval()
    return model, config
