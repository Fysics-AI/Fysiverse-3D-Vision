# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Inference-only G2VLM base used by the Fysiverse layout runtime.

Derived from InternRobotics/G2VLM. Only the runtime forward paths required by
the public layout pipeline are included here.
"""

from typing import Optional

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
import torchvision

from data.data_utils import (
    get_flattened_position_ids_extrapolate, 
    get_flattened_position_ids_interpolate,
    get_rope_index_image_3D,
    get_rope_index_image_3D_dino,
    patchify, 
)
from modeling.g2vlm.qwen2vl import NaiveCache

from modeling.pi3.models.layers.transformer_head import Pi3TransformerDecoder, Pi3LinearPts3d, Pi3ContextTransformerDecoder
from modeling.pi3.models.layers.camera_head import Pi3CameraHead
from modeling.pi3.models.layers.pos_embed import RoPE2D, PositionGetter

from data.transforms_vggt import load_and_preprocess_images, load_and_resize14


_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


def slice_expand_and_flatten(token_tensor, B, S):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined


class G2VLMConfig(PretrainedConfig):
    def __init__(
        self,
        visual_und=True,
        visual_recon=True,
        use_dinov3=False,
        llm_config=None,
        vit_config=None,
        dino_config=None,
        latent_patch_size=2,
        max_latent_size=32,
        vit_max_num_patch_per_side=70,
        dino_max_num_patch_per_side=37,
        interpolate_pos=False,
        use_registers=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.visual_und = visual_und
        self.visual_recon = visual_recon
        self.llm_config = llm_config
        self.vit_config = vit_config
        self.dino_config = dino_config
        self.latent_patch_size = latent_patch_size
        self.max_latent_size = max_latent_size
        self.vit_max_num_patch_per_side = vit_max_num_patch_per_side
        self.dino_max_num_patch_per_side = dino_max_num_patch_per_side
        self.interpolate_pos = interpolate_pos
        self.use_registers = use_registers
        self.use_dinov3 = use_dinov3


class G2VLM(PreTrainedModel):
    config_class = G2VLMConfig
    base_model_prefix = 'g2vlm'

    def __init__(self, language_model, vit_model, dino_model, config: G2VLMConfig):
        super().__init__(config)    
        self.language_model = language_model
        self.hidden_size = config.llm_config.hidden_size
        self.use_moe = "Mo" in config.llm_config.layer_module
        self.num_heads = config.llm_config.num_attention_heads

        self.conf_head = None
        self.global_point_head = None
        self.camera_head = None
        self.point_head = None
        self.use_dinov3 = config.use_dinov3

        
        if config.visual_recon:
            self.dino_model = dino_model
            self.dino_patch_size = config.dino_config.patch_size #14 
            self.dino_max_num_patch_per_side = config.dino_max_num_patch_per_side
            self.dino_hidden_size = config.dino_config.hidden_size
            self.embed_dim = self.hidden_size  
            self.resnet_normalize = torchvision.transforms.Normalize(mean=_RESNET_MEAN, std=_RESNET_STD)
            self.dino2llm = nn.Linear(self.dino_hidden_size, self.hidden_size) 
            self.use_registers = config.use_registers
            if self.use_registers:
                self.register_token = nn.Parameter(torch.randn(1, 2, 4, self.hidden_size))

            if RoPE2D is None: raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
            freq = float('rope100'[len('rope'):])
            self.pi3rope = RoPE2D(freq=freq)
            self.position_getter = PositionGetter()

            if self.use_registers:
                num_register_tokens = 5
                self.patch_start_idx = num_register_tokens
                self.register_token = nn.Parameter(torch.randn(1, 1, num_register_tokens, self.hidden_size))
            else: 
                self.patch_start_idx = 0 
            self.point_decoder = Pi3TransformerDecoder(
                in_dim=self.hidden_size,   #2*self.dec_embed_dim, 
                dec_embed_dim=self.hidden_size, #1024,
                dec_num_heads=16,
                out_dim=1024,
                rope=self.pi3rope,
            )
            if self.use_dinov3:
                self.point_head = Pi3LinearPts3d(patch_size=16, dec_embed_dim=1024, output_dim=3)
            else:
                self.point_head = Pi3LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3)
            # ----------------------
            #  Camera Pose Decoder
            # ----------------------

            self.camera_decoder = Pi3TransformerDecoder(
                in_dim=self.hidden_size,
                dec_embed_dim=self.hidden_size,
                dec_num_heads=16,                
                out_dim=512,
                rope=self.pi3rope,
                use_checkpoint=False
            )
            self.camera_head = Pi3CameraHead(dim=512)
            # ----------------------
            #  Global Points Decoder
            # ----------------------
            use_global_points = True  
            self.use_global_points = use_global_points

            if use_global_points:
                self.global_points_decoder = Pi3ContextTransformerDecoder(
                    in_dim=self.hidden_size,
                    dec_embed_dim=self.hidden_size,
                    dec_num_heads=16,
                    out_dim=1024,
                    rope=self.pi3rope,
                )
                if self.use_dinov3:
                    self.global_point_head = Pi3LinearPts3d(patch_size=16, dec_embed_dim=1024, output_dim=3)
                else:
                    self.global_point_head = Pi3LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3)
            else:
                self.global_point_head = None
            self.conf_head = None 

    
  
        if config.visual_und:
            self.vit_model = vit_model
            self.vit_patch_size = config.vit_config.patch_size
            self.vit_max_num_patch_per_side = 32 
            self.vit_hidden_size = config.vit_config.hidden_size
            self.use_registers = config.use_registers
       
        if config.interpolate_pos:
            self.get_flattened_position_ids = get_flattened_position_ids_interpolate
        else:
            self.get_flattened_position_ids = get_flattened_position_ids_extrapolate

        self.config = config
        self._init_weights()

    def _init_weights(self):
        if self.config.visual_recon:
            nn.init.constant_(self.dino2llm.weight, 0)
            nn.init.constant_(self.dino2llm.bias, 0)   
        if self.use_registers:
            nn.init.normal_(self.register_token, std=1e-6)

    def prepare_prompts_addbos(self, curr_kvlens, curr_rope, prompts, tokenizer, new_token_ids):
        packed_text_ids = list()
        packed_text_position_ids = list()
        text_token_lens = list()
        packed_text_indexes = list()
        packed_key_value_indexes = list()

        curr = 0
        newlens, new_rope = list(), list()
        for prompt, curr_kvlen, curr_position_id in zip(prompts, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            text_ids = tokenizer.encode(prompt)
            text_ids = [new_token_ids['bos_token_id']] + text_ids 
            text_token_lens.append(len(text_ids))
            packed_text_ids.extend(text_ids)
            packed_text_position_ids.extend(range(curr_position_id, curr_position_id + len(text_ids)))
            packed_text_indexes.extend(range(curr, curr + len(text_ids)))
            newlens.append(curr_kvlen + len(text_ids))
            new_rope.append(curr_position_id + len(text_ids))
            curr += len(text_ids)
        

        generation_input = {
            "text_token_lens": torch.tensor(text_token_lens, dtype=torch.int),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_position_ids": torch.tensor(packed_text_position_ids, dtype=torch.long).expand(3, -1),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    def prepare_prompts_addeos(self, curr_kvlens, curr_rope, prompts, tokenizer, new_token_ids):
        packed_text_ids = list()
        packed_text_position_ids = list()
        text_token_lens = list()
        packed_text_indexes = list()
        packed_key_value_indexes = list()

        curr = 0
        newlens, new_rope = list(), list()
        for prompt, curr_kvlen, curr_position_id in zip(prompts, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            text_ids = tokenizer.encode(prompt)
            assistant_ids = tokenizer.encode('assistant\n')
            text_ids = text_ids + [new_token_ids['eos_token_id']] + [new_token_ids['bos_token_id']] + assistant_ids
            text_token_lens.append(len(text_ids))
            packed_text_ids.extend(text_ids)
            packed_text_position_ids.extend(range(curr_position_id, curr_position_id + len(text_ids)))
            packed_text_indexes.extend(range(curr, curr + len(text_ids)))
            newlens.append(curr_kvlen + len(text_ids))
            new_rope.append(curr_position_id + len(text_ids))
            curr += len(text_ids)
        

        generation_input = {
            "text_token_lens": torch.tensor(text_token_lens, dtype=torch.int),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_position_ids": torch.tensor(packed_text_position_ids, dtype=torch.long).expand(3, -1),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope
    
    def prepare_prompts_pure_text(self, curr_kvlens, curr_rope, prompts, tokenizer, new_token_ids):
        packed_text_ids = list()
        packed_text_position_ids = list()
        text_token_lens = list()
        packed_text_indexes = list()
        packed_key_value_indexes = list()

        curr = 0
        newlens, new_rope = list(), list()
        for prompt, curr_kvlen, curr_position_id in zip(prompts, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            text_ids = tokenizer.encode(prompt)
            text_token_lens.append(len(text_ids))
            packed_text_ids.extend(text_ids)
            packed_text_position_ids.extend(range(curr_position_id, curr_position_id + len(text_ids)))
            packed_text_indexes.extend(range(curr, curr + len(text_ids)))
            newlens.append(curr_kvlen + len(text_ids))
            new_rope.append(curr_position_id + len(text_ids))
            curr += len(text_ids)
        

        generation_input = {
            "text_token_lens": torch.tensor(text_token_lens, dtype=torch.int),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_position_ids": torch.tensor(packed_text_position_ids, dtype=torch.long).expand(3, -1),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope
    
    def prepare_prompts(self, curr_kvlens, curr_rope, prompts, tokenizer, new_token_ids):
        packed_text_ids = list()
        packed_text_position_ids = list()
        text_token_lens = list()
        packed_text_indexes = list()
        packed_key_value_indexes = list()

        curr = 0
        newlens, new_rope = list(), list()
        for prompt, curr_kvlen, curr_position_id in zip(prompts, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            text_ids = tokenizer.encode(prompt)
            text_ids = [new_token_ids['bos_token_id']] + text_ids + [new_token_ids['eos_token_id']]
            text_token_lens.append(len(text_ids))
            packed_text_ids.extend(text_ids)
            packed_text_position_ids.extend(range(curr_position_id, curr_position_id + len(text_ids)))
            packed_text_indexes.extend(range(curr, curr + len(text_ids)))
            newlens.append(curr_kvlen + len(text_ids))
            new_rope.append(curr_position_id + len(text_ids))
            curr += len(text_ids)
        

        generation_input = {
            "text_token_lens": torch.tensor(text_token_lens, dtype=torch.int),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_position_ids": torch.tensor(packed_text_position_ids, dtype=torch.long).expand(3, -1),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    @torch.no_grad
    def forward_cache_update_text(
        self,
        past_key_values: NaiveCache,
        packed_text_ids: torch.IntTensor,
        packed_text_position_ids: torch.LongTensor,
        text_token_lens: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
    ):  
 
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {"mode": "und"}

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_text_embedding,
            query_lens=text_token_lens,
            packed_query_position_ids=packed_text_position_ids,
            packed_query_indexes=packed_text_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            update_past_key_values=True,
            is_causal=True,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values

    def prepare_vit_images(self, curr_kvlens, curr_rope, images, transforms, new_token_ids):
        packed_vit_token_indexes = list()
        vit_token_seqlens, packed_vit_tokens, packed_vit_position_ids = list(), list(), list()
        packed_vit_images = list()
        packed_image_grid_thw = list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        _curr = curr = 0
        newlens, new_rope = list(), list()
        for image, curr_kvlen, curr_position_id in zip(images, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1
         
            pos_tensor = torch.full((1,), curr_position_id, dtype=torch.long)
            packed_position_ids.extend([pos_tensor.expand(3, 1)])
            curr_position_id += 1

            image_tensor,  image_grid_thw = transforms([image])
            packed_image_grid_thw.append(image_grid_thw[0])
            num_img_tokens = image_tensor.shape[0] // 4 
            packed_vit_images.append(image_tensor)
            
            vit_token_seqlens.append(num_img_tokens)
            packed_vit_token_indexes.extend(range(_curr, _curr + num_img_tokens))
            packed_indexes.extend(range(curr, curr + num_img_tokens))
            curr += num_img_tokens
            _curr += num_img_tokens


            postions_ids_from_vit_for_rope, rope_deltas = get_rope_index_image_3D(
                image_grid_thw[0],
                curr_position_id,
                device=image_tensor.device
            )

            packed_position_ids.extend([postions_ids_from_vit_for_rope])
            curr_position_id += rope_deltas + 1

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            pos_tensor = torch.full((1,), curr_position_id, dtype=torch.long)
            packed_position_ids.extend([pos_tensor.expand(3, 1)])
            curr_position_id += 1


            packed_seqlens.append(num_img_tokens + 2)
            newlens.append(curr_kvlen + num_img_tokens + 2)
            new_rope.append(curr_position_id)

        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "vit_token_seqlens": torch.tensor(vit_token_seqlens, dtype=torch.int),
            "packed_image_grid_thw": torch.stack(packed_image_grid_thw, dim=0),
            "packed_vit_images": torch.stack(packed_vit_images, dim=0),
            "packed_vit_token_indexes": torch.tensor(packed_vit_token_indexes, dtype=torch.long),
            "packed_position_ids": torch.cat(packed_position_ids, dim=1),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    @torch.no_grad
    def forward_cache_update_vit(
        self,
        past_key_values: NaiveCache,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_vit_images: torch.Tensor,
        packed_image_grid_thw:  torch.IntTensor,
        packed_vit_token_indexes: torch.LongTensor,
        vit_token_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_vit_tokens: Optional[torch.Tensor]=None,
        packed_vit_position_ids: Optional[torch.LongTensor] = None,
    ):  

        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0))
        cu_seqlens = cu_seqlens.to(torch.int32)
        max_seqlen = torch.max(vit_token_seqlens).item()

        image_embeds = self.vit_model(packed_vit_images, grid_thw=packed_image_grid_thw)
        packed_vit_token_embed = image_embeds



        if packed_vit_token_embed.dtype != packed_sequence.dtype:
            packed_vit_token_embed = packed_vit_token_embed.to(packed_sequence.dtype)
        packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {"mode": "und"}

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            update_past_key_values=True,
            is_causal=False,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values
    
    def prepare_dino_images_pi3 (self, curr_kvlens, curr_rope, images, transforms, new_token_ids):
        packed_dino_token_indexes = list()
        dino_token_seqlens, packed_dino_tokens, packed_dino_position_ids = list(), list(), list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        _curr = curr = 0
        newlens, new_rope = list(), list()
    
        vggt_fixed_resolution = 518 # hardcode 
        img_load_resolution = 1024

        images = load_and_resize14(images,vggt_fixed_resolution)

        curr_kvlen = curr_kvlens[0]
        curr_position_id = curr_rope[0]
        
        packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
        curr += curr_kvlen
        for image in images:
            
            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            image_tensor = image
            height, width = image_tensor.shape[1:]
            grid_t = 1  
            grid_h, grid_w = height // 14, width // 14

            # add 3d pos for <|startofimage|> token
            pos_tensor = torch.full((1,), curr_position_id, dtype=torch.long)
            packed_position_ids.extend([pos_tensor.expand(3, 1)])
            curr_position_id += 1

            dino_tokens = patchify(image_tensor, self.dino_patch_size)
            packed_dino_tokens.append(dino_tokens)
            num_img_tokens = dino_tokens.shape[0]
            dino_token_seqlens.append(num_img_tokens)
  
            packed_dino_token_indexes.extend(range(_curr, _curr + num_img_tokens))
            packed_indexes.extend(range(curr, curr + num_img_tokens))
            curr += num_img_tokens
            _curr += num_img_tokens

            ###3d rope embedding for QKV attention: 
            dino_image_thw = torch.tensor([grid_t, grid_h, grid_w], dtype=torch.long) 
            postions_ids_from_dino_for_rope, rope_deltas = get_rope_index_image_3D_dino(
                dino_image_thw,
                curr_position_id,
                device=image_tensor.device
            )
            packed_position_ids.extend([postions_ids_from_dino_for_rope])
            curr_position_id += rope_deltas + 1

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            pos_tensor = torch.full((1,), curr_position_id, dtype=torch.long)
            packed_position_ids.extend([pos_tensor.expand(3, 1)])
            curr_position_id += 1

            packed_seqlens.append(num_img_tokens + 2)
            newlens.append(curr_kvlen + num_img_tokens + 2)
            curr_kvlen += num_img_tokens + 2

            new_rope.append(curr_position_id)

        newlens = [newlens[-1]]
        new_rope = [new_rope[-1]]
        packed_seqlens = [sum(packed_seqlens)]


        assert len(images.shape) == 4
        assert images.shape[1] == 3
        original_images = images.clone()
        images = torchvision.transforms.Normalize(mean=_RESNET_MEAN, std=_RESNET_STD)(images) 

        generation_input = {
            "packed_dino_images": images, 
            'original_images': original_images,
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "dino_token_seqlens": torch.tensor(dino_token_seqlens, dtype=torch.int),
            "packed_dino_token_indexes": torch.tensor(packed_dino_token_indexes, dtype=torch.long),
            "packed_position_ids": torch.cat(packed_position_ids, dim=1),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }
    
        return generation_input, newlens, new_rope
    def prepare_dino_images_none (self, curr_kvlens, curr_rope, images, transforms, new_token_ids):
        packed_dino_token_indexes = list()
        dino_token_seqlens, packed_dino_tokens, packed_dino_position_ids = list(), list(), list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        _curr = curr = 0
        newlens, new_rope = list(), list()
    
        vggt_fixed_resolution = 518 # hardcode 
        img_load_resolution = 1024

        curr_kvlen = curr_kvlens[0]
        curr_position_id = curr_rope[0]
        
        packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
        curr += curr_kvlen
        for image in images:
            
            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            image_tensor = image
            height, width = image_tensor.shape[1:]
            grid_t = 1  
            grid_h, grid_w = height // 14, width // 14

            # add 3d pos for <|startofimage|> token
            pos_tensor = torch.full((1,), curr_position_id, dtype=torch.long)
            packed_position_ids.extend([pos_tensor.expand(3, 1)])
            curr_position_id += 1

            dino_tokens = patchify(image_tensor, self.dino_patch_size)
            packed_dino_tokens.append(dino_tokens)
            num_img_tokens = dino_tokens.shape[0]
            dino_token_seqlens.append(num_img_tokens)
  
            packed_dino_token_indexes.extend(range(_curr, _curr + num_img_tokens))
            packed_indexes.extend(range(curr, curr + num_img_tokens))
            curr += num_img_tokens
            _curr += num_img_tokens

            ###3d rope embedding for QKV attention: 
            dino_image_thw = torch.tensor([grid_t, grid_h, grid_w], dtype=torch.long) 
            postions_ids_from_dino_for_rope, rope_deltas = get_rope_index_image_3D_dino(
                dino_image_thw,
                curr_position_id,
                device=image_tensor.device
            )
            packed_position_ids.extend([postions_ids_from_dino_for_rope])
            curr_position_id += rope_deltas + 1

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            pos_tensor = torch.full((1,), curr_position_id, dtype=torch.long)
            packed_position_ids.extend([pos_tensor.expand(3, 1)])
            curr_position_id += 1

            packed_seqlens.append(num_img_tokens + 2)
            newlens.append(curr_kvlen + num_img_tokens + 2)
            curr_kvlen += num_img_tokens + 2

            new_rope.append(curr_position_id)

        newlens = [newlens[-1]]
        new_rope = [new_rope[-1]]
        packed_seqlens = [sum(packed_seqlens)]


        assert len(images.shape) == 4
        assert images.shape[1] == 3
        original_images = images.clone()
        images = torchvision.transforms.Normalize(mean=_RESNET_MEAN, std=_RESNET_STD)(images) 

        generation_input = {
            "packed_dino_images": images, 
            'original_images': original_images,
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "dino_token_seqlens": torch.tensor(dino_token_seqlens, dtype=torch.int),
            "packed_dino_token_indexes": torch.tensor(packed_dino_token_indexes, dtype=torch.long),
            "packed_position_ids": torch.cat(packed_position_ids, dim=1),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }
    
        return generation_input, newlens, new_rope

    @torch.no_grad
    def forward_cache_update_dino(
        self,
        past_key_values: NaiveCache,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_dino_token_indexes: torch.LongTensor,
        dino_token_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_dino_images: torch.Tensor, 
        original_images: torch.Tensor, 
    ):
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding
        
        cu_seqlens = torch.nn.functional.pad(torch.cumsum(dino_token_seqlens, dim=0), (1, 0))
        cu_seqlens = cu_seqlens.to(torch.int32)
        max_seqlen = torch.max(dino_token_seqlens).item()
 
        packed_dino_token_embed = self.dino_model(
            packed_pixel_values=packed_dino_images, 
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        B, P, D = packed_dino_token_embed.size() #
        packed_dino_token_embed = packed_dino_token_embed.reshape(B*P, D)

        packed_dino_token_embed = self.dino2llm(packed_dino_token_embed)

        BS, C_in, H, W = packed_dino_images.shape
        S = BS ### constant for now 
        B = BS // S 
        assert B==1

        if packed_dino_token_embed.dtype != packed_sequence.dtype:
            packed_dino_token_embed = packed_dino_token_embed.to(packed_sequence.dtype)
        packed_sequence[packed_dino_token_indexes] = packed_dino_token_embed

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {
                "mode": "geo",
                "packed_geo_token_indexes": packed_dino_token_indexes, 
                "packed_text_indexes": packed_text_indexes
            } 

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            update_past_key_values=True,
            output_hidden_states=False, 
            is_causal=False,
            **extra_inputs,
        )

     
        past_key_values = output.past_key_values
        last_hidden_state = output.packed_query_sequence


        return past_key_values, last_hidden_state

