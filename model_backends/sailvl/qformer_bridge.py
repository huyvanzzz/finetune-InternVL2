from __future__ import annotations

import json
import os
from typing import Dict

import torch
from torch import nn
from types import MethodType
from safetensors.torch import load_file, save_file

from qformer_bridge import (
    _clear_qformer_text,
    _encode_qformer_texts,
    _load_qformer_from_source,
    _set_qformer_text,
    get_qformer_config,
)


BRIDGE_WEIGHTS_NAME = "qformer_bridge.safetensors"
BRIDGE_CONFIG_NAME = "qformer_bridge_config.json"


def _infer_runtime_device_and_dtype(model):
    for module_name in ("language_model", "vision_model"):
        module = getattr(model, module_name, None)
        if module is None:
            continue
        try:
            first_param = next(module.parameters())
        except StopIteration:
            continue
        return first_param.device, first_param.dtype

    try:
        first_param = next(model.parameters())
        return first_param.device, first_param.dtype
    except StopIteration:
        return torch.device("cpu"), torch.float32


def _ensure_bridge_device(model, reference: torch.Tensor):
    device = reference.device
    reference_dtype = reference.dtype

    qformer = getattr(model, "qformer", None)
    if qformer is not None:
        qformer.to(device=device, dtype=reference_dtype)

    for module_name in ("qformer_input_proj", "qformer_to_mlp1_proj"):
        module = getattr(model, module_name, None)
        if module is not None:
            module.to(device=device, dtype=torch.float32)

    model.qformer_query_tokens.data = model.qformer_query_tokens.data.to(device=device, dtype=reference_dtype)


def align_sail_qformer_bridge_runtime(model):
    if not getattr(model, "qformer_enabled", False):
        return None

    device, reference_dtype = _infer_runtime_device_and_dtype(model)
    bridge_reference = torch.empty(1, device=device, dtype=reference_dtype)
    _ensure_bridge_device(model, bridge_reference)
    return device


def _extract_vit_tokens_sail(model, pixel_values):
    if model.select_layer == -1:
        vit_embeds = model.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=False,
            return_dict=True,
        ).last_hidden_state
    else:
        vit_embeds = model.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        ).hidden_states[model.select_layer]

    h = w = int(vit_embeds.shape[1] ** 0.5)
    vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
    vit_embeds = model.pixel_shuffle(vit_embeds, scale_factor=model.downsample_ratio)
    vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
    return vit_embeds


def _extract_feature_with_qformer(self, pixel_values):
    vit_embeds = _extract_vit_tokens_sail(self, pixel_values)
    _ensure_bridge_device(self, vit_embeds)

    proj_in_dtype = next(self.qformer_input_proj.parameters()).dtype
    encoder_hidden_states = self.qformer_input_proj(vit_embeds.to(proj_in_dtype))
    encoder_attention_mask = torch.ones(
        encoder_hidden_states.size()[:-1],
        dtype=torch.long,
        device=encoder_hidden_states.device,
    )

    qformer_input_ids = getattr(self, "_qformer_input_ids", None)
    qformer_attention_mask = getattr(self, "_qformer_attention_mask", None)
    if qformer_input_ids is None or qformer_attention_mask is None:
        qformer_input_ids, qformer_attention_mask = self.encode_qformer_texts(
            ["" for _ in range(pixel_values.shape[0])],
            device=encoder_hidden_states.device,
        )
    else:
        qformer_input_ids = qformer_input_ids.to(encoder_hidden_states.device)
        qformer_attention_mask = qformer_attention_mask.to(encoder_hidden_states.device)

    if qformer_input_ids.shape[0] != encoder_hidden_states.shape[0]:
        raise ValueError(
            "Q-Former text batch must match image/tile batch: "
            f"{qformer_input_ids.shape[0]} vs {encoder_hidden_states.shape[0]}"
        )

    query_tokens = self.qformer_query_tokens.expand(encoder_hidden_states.shape[0], -1, -1)
    query_attention_mask = torch.ones(
        query_tokens.size()[:-1],
        dtype=torch.long,
        device=encoder_hidden_states.device,
    )
    qformer_attention_mask = torch.cat([query_attention_mask, qformer_attention_mask], dim=1)
    qformer_dtype = next(self.qformer.parameters()).dtype
    encoder_hidden_states = encoder_hidden_states.to(qformer_dtype)
    query_tokens = query_tokens.to(qformer_dtype)
    query_outputs = self.qformer(
        input_ids=qformer_input_ids,
        attention_mask=qformer_attention_mask,
        query_embeds=query_tokens,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        return_dict=True,
    )
    query_output = query_outputs[0][:, : query_tokens.size(1), :]
    proj_out_dtype = next(self.qformer_to_mlp1_proj.parameters()).dtype
    mlp1_inputs = self.qformer_to_mlp1_proj(query_output.to(proj_out_dtype))
    mlp1_dtype = next(self.mlp1.parameters()).dtype
    mlp1_inputs = mlp1_inputs.to(mlp1_dtype)
    return self.mlp1(mlp1_inputs)


def attach_sail_qformer_bridge(model, config: Dict, logger=None):
    q_cfg = get_qformer_config(config)
    if not q_cfg["enabled"]:
        model.qformer_enabled = False
        return model

    qformer, query_tokens, qformer_tokenizer, blip_config = _load_qformer_from_source(
        q_cfg["source_model"],
        q_cfg["cache_dir"],
    )

    vit_hidden_size = model.config.vision_config.hidden_size
    llm_hidden_size = model.config.llm_config.hidden_size
    downsample_ratio = model.config.downsample_ratio
    pixel_shuffle_dim = vit_hidden_size * int(1 / downsample_ratio) ** 2
    qformer_encoder_dim = blip_config.qformer_config.encoder_hidden_size
    qformer_hidden_size = blip_config.qformer_config.hidden_size
    num_query_tokens = int(q_cfg["num_query_tokens"])

    if query_tokens.shape[1] < num_query_tokens:
        raise ValueError(
            f"Requested {num_query_tokens} query tokens, checkpoint has {query_tokens.shape[1]}"
        )
    query_tokens = nn.Parameter(query_tokens[:, :num_query_tokens, :].clone())

    model.qformer_enabled = True
    model.qformer_source_model = q_cfg["source_model"]
    model.qformer_max_text_length = int(q_cfg["max_text_length"])
    model.qformer_tokenizer = qformer_tokenizer
    model.qformer = qformer
    model.qformer_query_tokens = query_tokens
    model.qformer_input_proj = nn.Sequential(
        nn.LayerNorm(pixel_shuffle_dim),
        nn.Linear(pixel_shuffle_dim, qformer_encoder_dim),
    ).to(dtype=torch.float32)
    model.qformer_to_mlp1_proj = nn.Sequential(
        nn.LayerNorm(qformer_hidden_size),
        nn.Linear(qformer_hidden_size, pixel_shuffle_dim),
    ).to(dtype=torch.float32)
    model.num_image_token = num_query_tokens

    if q_cfg["freeze_qformer"]:
        model.qformer.requires_grad_(False)
        model.qformer_query_tokens.requires_grad_(False)
    if q_cfg["freeze_mlp1"]:
        model.mlp1.requires_grad_(False)

    model.extract_feature = MethodType(_extract_feature_with_qformer, model)
    model.encode_qformer_texts = MethodType(_encode_qformer_texts, model)
    model.set_qformer_text = MethodType(_set_qformer_text, model)
    model.clear_qformer_text = MethodType(_clear_qformer_text, model)

    if logger:
        logger.info(
            "Attached SAIL prompt-aware Q-Former bridge: "
            f"pixel_shuffle_dim={pixel_shuffle_dim}, "
            f"qformer_encoder_dim={qformer_encoder_dim}, "
            f"qformer_hidden_size={qformer_hidden_size}, "
            f"num_query_tokens={num_query_tokens}, "
            f"llm_hidden_size={llm_hidden_size}"
        )
    return model


def save_sail_qformer_bridge(model, output_dir: str):
    if not getattr(model, "qformer_enabled", False):
        return
    os.makedirs(output_dir, exist_ok=True)
    state = {}
    for module_name in ("qformer_input_proj", "qformer_to_mlp1_proj"):
        module = getattr(model, module_name, None)
        if module is None:
            continue
        for key, value in module.state_dict().items():
            state[f"{module_name}.{key}"] = value.detach().cpu()
    save_file(state, os.path.join(output_dir, BRIDGE_WEIGHTS_NAME))


def load_sail_qformer_bridge(model, checkpoint_dir: str, strict: bool = True):
    weights_path = os.path.join(checkpoint_dir, BRIDGE_WEIGHTS_NAME)
    if not os.path.exists(weights_path):
        if strict:
            raise FileNotFoundError(f"SAIL bridge weights not found: {weights_path}")
        return False
    state = load_file(weights_path, device="cpu")
    for module_name in ("qformer_input_proj", "qformer_to_mlp1_proj"):
        module = getattr(model, module_name, None)
        if module is None:
            continue
        module_state = {
            key[len(module_name) + 1 :]: value
            for key, value in state.items()
            if key.startswith(module_name + ".")
        }
        module.load_state_dict(module_state, strict=True)
    return True


def write_sail_bridge_metadata(output_dir: str, extra: Dict | None = None):
    metadata = {
        "bridge_backend": "sailvl",
        "bridge_mode": "prompt_aware_preproj_mlp1",
    }
    if extra:
        metadata.update(extra)
    with open(os.path.join(output_dir, BRIDGE_CONFIG_NAME), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
