from __future__ import annotations

import json
import os
from typing import Dict, Optional

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

from model.conversation import get_conv_template
from qformer_bridge import qformer_enabled

from .preprocess import preprocess_sail_image
from .qformer_bridge import (
    BRIDGE_CONFIG_NAME,
    attach_sail_qformer_bridge,
    load_sail_qformer_bridge,
    save_sail_qformer_bridge,
    write_sail_bridge_metadata,
)


BACKEND_NAME = "sailvl"
IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
SYSTEM_MESSAGE = "You are a navigation assistant for visually impaired users."


def maybe_pad(inner_lists, padding_value):
    tensor_list = [torch.tensor(inner_list, dtype=torch.long) for inner_list in inner_lists]
    return pad_sequence(tensor_list, batch_first=True, padding_value=padding_value)


class SailCollateFn:
    def __init__(self, tokenizer, model, config) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.config = config

    def __call__(self, batch):
        batch = [sample for sample in batch if sample is not None]
        label_ids_batch = []
        input_ids_batch = []
        attention_mask_batch = []
        pixel_values_batch = []
        qformer_texts = []
        samples_batch = []

        if not batch:
            return None, None, None, None, None, []

        for sample in batch:
            question = sample["question"]
            answer = sample["answer"]
            image = sample["image"][0] if isinstance(sample.get("image"), list) else sample["image"]
            pixel_values = preprocess_sail_image(image, self.config)
            samples_batch.append(sample)

            template = _get_runtime_template(self.model)
            template.system_message = getattr(self.model, "system_message", SYSTEM_MESSAGE)
            eos_token_id = self.tokenizer.convert_tokens_to_ids(template.sep)
            eot_token_id = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")

            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            num_patches = pixel_values.shape[0]
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.model.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace("<image>", image_tokens, 1)

            input_ids = self.tokenizer.encode(query, add_special_tokens=False)
            answer_ids = self.tokenizer.encode(answer, add_special_tokens=False)
            label_ids = [-100] * len(input_ids) + answer_ids + [eos_token_id]
            input_ids = input_ids + answer_ids + [eos_token_id]
            attention_mask = [1] * len(input_ids)

            label_ids_batch.append(label_ids)
            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
            pixel_values_batch.append(pixel_values)
            if getattr(self.model, "qformer_enabled", False):
                qformer_texts.extend([sample.get("qformer_text", question.replace("<image>", "").strip())] * num_patches)

        input_ids_tensor = maybe_pad(input_ids_batch, eot_token_id)
        label_ids_tensor = maybe_pad(label_ids_batch, -100)
        attention_mask_tensor = maybe_pad(attention_mask_batch, 0)
        pixel_values_tensor = torch.cat(pixel_values_batch)
        qformer_inputs = None
        if getattr(self.model, "qformer_enabled", False):
            qformer_inputs = self.model.encode_qformer_texts(qformer_texts)
        return input_ids_tensor, label_ids_tensor, attention_mask_tensor, pixel_values_tensor, qformer_inputs, samples_batch


def _get_runtime_template(model):
    conv_template = getattr(model, "conv_template", None)
    if conv_template is not None and hasattr(conv_template, "copy"):
        return conv_template.copy()
    return get_conv_template(model.template)


def load_model_and_tokenizer(config: Dict, checkpoint_dir: Optional[str] = None):
    model_name_or_path = config["model"]["name"]
    quant_cfg = config["model"]["quantization"]
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=quant_cfg["enabled"],
        bnb_4bit_compute_dtype=torch.bfloat16 if quant_cfg["compute_dtype"] == "bfloat16" else torch.float16,
        bnb_4bit_use_double_quant=quant_cfg["double_quant"],
        bnb_4bit_quant_type=quant_cfg["type"],
    )
    model = AutoModel.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=config["model"]["trust_remote_code"],
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, use_fast=False)
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.system_message = SYSTEM_MESSAGE
    return model, tokenizer


def build_train_collate_fn(tokenizer, model, config):
    return SailCollateFn(tokenizer, model, config)


def build_eval_collate_fn(tokenizer, model, config):
    return SailCollateFn(tokenizer, model, config)


def _build_image_flags(pixel_values_batch: torch.Tensor):
    return torch.ones((pixel_values_batch.shape[0], 1), dtype=torch.long)


def forward_train_batch(model, batch, config):
    input_ids_batch, label_ids_batch, _, pixel_values_batch, qformer_inputs, _ = batch
    input_ids_batch = input_ids_batch.cuda()
    label_ids_batch = label_ids_batch.cuda()
    pixel_values_batch = pixel_values_batch.to(torch.bfloat16).cuda()
    image_flags_batch = _build_image_flags(pixel_values_batch).cuda()
    if getattr(model, "qformer_enabled", False) and qformer_inputs is not None:
        model.set_qformer_text(qformer_inputs[0].cuda(), qformer_inputs[1].cuda())
    outputs = model(
        input_ids=input_ids_batch,
        pixel_values=pixel_values_batch,
        labels=label_ids_batch,
        image_flags=image_flags_batch,
        return_dict=True,
    )
    if getattr(model, "qformer_enabled", False):
        model.clear_qformer_text()
    return outputs


def forward_eval_batch(model, batch, config):
    return forward_train_batch(model, batch, config)


def generate_response(model, tokenizer, sample, generation_config, config):
    pixel_values = preprocess_sail_image(sample["image"][0] if isinstance(sample["image"], list) else sample["image"], config)
    pixel_values = pixel_values.to(torch.bfloat16).cuda()
    question = str(sample["question"])
    if getattr(model, "qformer_enabled", False):
        qformer_text = sample.get("qformer_text", question.replace("<image>", "").strip())
        q_ids, q_mask = model.encode_qformer_texts([qformer_text] * pixel_values.shape[0], device=pixel_values.device)
        model.set_qformer_text(q_ids, q_mask)
    response = model.chat(tokenizer, pixel_values, question, generation_config)
    if getattr(model, "qformer_enabled", False):
        model.clear_qformer_text()
    return response


def attach_qformer_if_enabled(model, config, logger=None):
    if not qformer_enabled(config):
        model.qformer_enabled = False
        return model
    return attach_sail_qformer_bridge(model, config, logger=logger)


def save_backend_artifacts(model, tokenizer, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model.language_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    if getattr(model, "qformer_enabled", False):
        save_sail_qformer_bridge(model, output_dir)
        write_sail_bridge_metadata(output_dir)


def load_backend_artifacts(model, checkpoint_dir, config):
    if getattr(model, "language_model", None) is not None:
        # LoRA weights are loaded externally via PEFT.
        pass
    if qformer_enabled(config):
        config_path = os.path.join(checkpoint_dir, BRIDGE_CONFIG_NAME)
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                bridge_cfg = json.load(f)
            if bridge_cfg.get("bridge_backend") != "sailvl":
                raise ValueError("Expected sailvl bridge backend in qformer bridge config.")
        load_sail_qformer_bridge(model, checkpoint_dir, strict=True)
