from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from types import MethodType
from typing import Dict, Optional

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from transformers.modeling_outputs import CausalLMOutputWithPast
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


def _log_info(msg, *args):
    logger = logging.getLogger("MyLogger")
    if logger.handlers:
        logger.info(msg, *args)
        return
    rendered = msg % args if args else msg
    print(rendered)


def maybe_pad(inner_lists, padding_value):
    tensor_list = [torch.tensor(inner_list, dtype=torch.long) for inner_list in inner_lists]
    return pad_sequence(tensor_list, batch_first=True, padding_value=padding_value)


class CloneOutputEmbeddingWrapper(nn.Module):
    def __init__(self, embedding_module: nn.Module):
        super().__init__()
        self.embedding_module = embedding_module

    def forward(self, *args, **kwargs):
        return self.embedding_module(*args, **kwargs).clone()


@contextmanager
def _safe_distributed_rank():
    dist = getattr(torch, "distributed", None)
    get_rank = getattr(dist, "get_rank", None)
    is_available = getattr(dist, "is_available", None)
    is_initialized = getattr(dist, "is_initialized", None)

    if not callable(get_rank) or not callable(is_available) or not callable(is_initialized):
        yield
        return

    if not is_available() or not is_initialized():
        original_get_rank = dist.get_rank
        dist.get_rank = lambda: 0
        try:
            yield
        finally:
            dist.get_rank = original_get_rank
        return

    yield


def wrap_input_embeddings_for_safe_scatter(model):
    language_model = getattr(model, "language_model", None)
    if language_model is None:
        return

    get_embeddings = getattr(language_model, "get_input_embeddings", None)
    if not callable(get_embeddings):
        return

    embedding_module = get_embeddings()
    if isinstance(embedding_module, CloneOutputEmbeddingWrapper):
        return

    wrapped = CloneOutputEmbeddingWrapper(embedding_module)

    def _get_input_embeddings():
        return wrapped

    language_model.get_input_embeddings = _get_input_embeddings


def patch_sail_forward_runtime(model):
    if getattr(model, "_safe_forward_patched", False):
        return

    def _forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if image_flags is None:
            raise ValueError("image_flags is required for SailVL forward.")

        image_flags_local = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        vit_embeds = self.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags_local == 1]
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        flat_input_embeds = input_embeds.reshape(B * N, C).clone()

        dist = getattr(torch, "distributed", None)
        rank_zero = True
        if (
            dist is not None
            and callable(getattr(dist, "is_available", None))
            and callable(getattr(dist, "is_initialized", None))
            and dist.is_available()
            and dist.is_initialized()
            and callable(getattr(dist, "get_rank", None))
        ):
            rank_zero = dist.get_rank() == 0
        if rank_zero:
            print(
                f"dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}"
            )

        flat_input_ids = input_ids.reshape(B * N)
        selected = flat_input_ids == self.img_context_token_id
        flat_vit_embeds = vit_embeds.reshape(-1, C)
        n_token = int(selected.sum().item())
        if n_token > 0:
            flat_input_embeds[selected] = flat_vit_embeds[:n_token].to(flat_input_embeds.device)

        input_embeds = flat_input_embeds.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=getattr(outputs, "past_key_values", None),
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
        )

    model.forward = MethodType(_forward, model)
    model._safe_forward_patched = True


class SailCollateFn:
    def __init__(self, tokenizer, model, config) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.config = config
        self.log_token_stats = False
        self.token_log_remaining = 0
        self.log_prompt_samples = False
        self.prompt_log_remaining = 0

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
            if self.log_token_stats and self.token_log_remaining != 0:
                total_image_tokens_in_sample = num_patches * self.model.num_image_token
                total_sequence_length = len(input_ids) + len(answer_ids) + 1
                _log_info(
                    "[INFO] Image token stats | frames=%s | tiles_per_frame=%s | query_tokens_per_tile=%s | total_image_tokens=%s",
                    1,
                    [num_patches],
                    self.model.num_image_token,
                    total_image_tokens_in_sample,
                )
                _log_info(
                    "[INFO] Text tokens - input: %s, answer: %s, total: %s",
                    len(input_ids),
                    len(answer_ids),
                    total_sequence_length,
                )
                if self.token_log_remaining > 0:
                    self.token_log_remaining -= 1
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
        if self.log_prompt_samples and self.prompt_log_remaining != 0:
            for sample in samples_batch:
                if sample.get("task_type") != "alter":
                    continue
                _log_info(
                    "[PROMPT SAMPLE] questionId=%s | frame_path=%s | task_type=%s | selected_prompt_id=%s\n"
                    "selected_prompt_text:\n%s\n"
                    "question:\n%s\n"
                    "qformer_text:\n%s\n"
                    "answer:\n%s",
                    sample.get("questionId"),
                    sample.get("frame_path"),
                    sample.get("task_type"),
                    sample.get("selected_prompt_id"),
                    sample.get("selected_prompt_text"),
                    sample.get("question"),
                    sample.get("qformer_text"),
                    sample.get("answer"),
                )
            if self.prompt_log_remaining > 0:
                self.prompt_log_remaining -= 1
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
    wrap_input_embeddings_for_safe_scatter(model)
    patch_sail_forward_runtime(model)
    return model, tokenizer


def build_train_collate_fn(tokenizer, model, config):
    return SailCollateFn(tokenizer, model, config)


def build_eval_collate_fn(tokenizer, model, config):
    return SailCollateFn(tokenizer, model, config)


def _build_image_flags(pixel_values_batch: torch.Tensor):
    return torch.ones((pixel_values_batch.shape[0], 1), dtype=torch.long)


def forward_train_batch(model, batch, config):
    patch_sail_forward_runtime(model)
    input_ids_batch, label_ids_batch, _, pixel_values_batch, qformer_inputs, _ = batch
    wrap_input_embeddings_for_safe_scatter(model)
    input_ids_batch = input_ids_batch.cuda()
    label_ids_batch = label_ids_batch.cuda()
    pixel_values_batch = pixel_values_batch.to(torch.bfloat16).cuda()
    image_flags_batch = _build_image_flags(pixel_values_batch).cuda()
    if getattr(model, "qformer_enabled", False) and qformer_inputs is not None:
        model.set_qformer_text(qformer_inputs[0].cuda(), qformer_inputs[1].cuda())
    with _safe_distributed_rank():
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
    patch_sail_forward_runtime(model)
    wrap_input_embeddings_for_safe_scatter(model)
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
