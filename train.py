import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import contextlib
import datetime
import io
import json
import pickle
import re
import random
import sys
from collections import defaultdict
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from datasets import load_dataset
from huggingface_hub import snapshot_download
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig, get_cosine_schedule_with_warmup

from logutil import get_logger, init_logger
from optimizer_state_utils import (
    count_optimizer_state_tensors_on_cpu,
    enforce_safe_optimizer_param_groups,
    export_sanitized_optimizer_state_dict,
    move_optimizer_state_to_param_device,
    sanitize_optimizer_state_dict,
)
from pretrain_checkpoint_verify import verify_loaded_pretrain_modules


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)

def get_config_path_from_argv(default="internvl_config.yaml"):
    for idx, arg in enumerate(sys.argv):
        if arg == "--config" and idx + 1 < len(sys.argv):
            return sys.argv[idx + 1]
        if arg.startswith("--config="):
            return arg.split("=", 1)[1]
    return default


CONFIG_PATH = get_config_path_from_argv()
logger = None


class SilentLogger:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None


class _LineFilterBuffer(io.TextIOBase):
    def __init__(self, wrapped, blocked_substrings):
        self.wrapped = wrapped
        self.blocked_substrings = tuple(blocked_substrings)
        self._pending = ""

    def write(self, s):
        self._pending += s
        while "\n" in self._pending:
            line, self._pending = self._pending.split("\n", 1)
            if not any(token in line for token in self.blocked_substrings):
                self.wrapped.write(line + "\n")
        return len(s)

    def flush(self):
        if self._pending and not any(token in self._pending for token in self.blocked_substrings):
            self.wrapped.write(self._pending)
        self._pending = ""
        self.wrapped.flush()


@contextlib.contextmanager
def suppress_runtime_noise():
    blocked = (
        "dynamic ViT batch size:",
        "`use_cache=True` is incompatible with gradient checkpointing.",
    )
    stdout_filter = _LineFilterBuffer(sys.stdout, blocked)
    stderr_filter = _LineFilterBuffer(sys.stderr, blocked)
    with contextlib.redirect_stdout(stdout_filter), contextlib.redirect_stderr(stderr_filter):
        yield


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_output_dir(config: Dict) -> str:
    base_out_dir = config["training"]["output_dir"]
    return f'{base_out_dir}/{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}/'


def broadcast_output_dir(output_dir: str | None, accelerator: Accelerator | None) -> str:
    if accelerator is None or accelerator.num_processes == 1:
        assert output_dir is not None
        return output_dir
    payload = [output_dir]
    dist.broadcast_object_list(payload, src=0)
    assert payload[0] is not None
    return payload[0]


def build_dataloader_kwargs(config: Dict) -> Dict:
    hardware_cfg = config.get("hardware", {})
    num_workers = int(hardware_cfg.get("num_workers", 0))
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": bool(hardware_cfg.get("pin_memory", False)),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(hardware_cfg.get("persistent_workers", False))
        if "prefetch_factor" in hardware_cfg:
            kwargs["prefetch_factor"] = int(hardware_cfg["prefetch_factor"])
    return kwargs


def verify_flash_attention_runtime(model) -> Dict:
    modules = []
    for name, module in model.named_modules():
        if hasattr(module, "use_flash_attn"):
            requested = bool(getattr(getattr(module, "config", None), "use_flash_attn", False))
            enabled = bool(getattr(module, "use_flash_attn", False))
            modules.append(
                {
                    "module": name,
                    "requested": requested,
                    "enabled": enabled,
                    "status": "flash" if enabled else "fallback",
                    "fallback_reason": None if enabled else "use_flash_attn is false at runtime",
                }
            )
    return {
        "supported_count": len(modules),
        "flash_enabled_count": sum(1 for item in modules if item["enabled"]),
        "fallback_count": sum(1 for item in modules if not item["enabled"]),
        "modules": modules,
    }


def log_flash_attention_runtime(model, logger):
    report = verify_flash_attention_runtime(model)
    logger.info(
        "FlashAttention runtime | supported=%s | flash=%s | fallback=%s",
        report["supported_count"],
        report["flash_enabled_count"],
        report["fallback_count"],
    )
    for item in report["modules"][:20]:
        logger.info(
            "FlashAttention module | name=%s | requested=%s | status=%s | reason=%s",
            item["module"],
            item["requested"],
            item["status"],
            item["fallback_reason"],
        )
    return report

from wad_dataset import build_dataset
from wad_dataset import WADDatasetForInternVL
from model.conversation import get_conv_template
from preprocessing import get_response_format
from qformer_bridge import (
    align_qformer_bridge_runtime,
    attach_qformer_bridge,
    load_qformer_bridge,
    qformer_enabled,
    read_qformer_bridge_metadata,
    save_qformer_bridge,
    trainable_parameter_summary,
)
from trajectory_branch import (
    load_trajectory_branch,
    read_trajectory_branch_metadata,
    save_trajectory_branch,
    trajectory_enabled,
)
from trajectory_trainability import apply_mode_gated_trajectory_trainability
from trajectory_branch import build_trajectory_source_from_config
from scripts.pairs_output import write_prediction_pairs


IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
SYSTEM_MESSAGE = "You are a navigation assistant for visually impaired users."


def parse_args():
    parser = argparse.ArgumentParser(description="Train InternVL with optional checkpoint resume.")
    parser.add_argument("--config", type=str, default=CONFIG_PATH, help="Path to YAML config file.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint dir to resume from. Omit this flag to train from the default LoRA/base setup.",
    )
    parser.add_argument(
        "--pretrain_checkpoint",
        type=str,
        default=None,
        help="Optional pretrain checkpoint dir or HF repo id. Loads only qformer_bridge + trajectory_branch, then initializes fresh LoRA.",
    )
    parser.add_argument("--start_epoch", type=int, default=None, help="Zero-based epoch index to resume from.")
    parser.add_argument("--start_step", type=int, default=None, help="Batch step inside the resume epoch.")
    return parser.parse_args()


def infer_resume_position(checkpoint_dir):
    name = os.path.basename(os.path.normpath(checkpoint_dir))
    step_match = re.fullmatch(r"epoch_(\d+)_step_(\d+)", name)
    if step_match:
        epoch_num = int(step_match.group(1))
        step = int(step_match.group(2))
        return max(epoch_num - 1, 0), step

    epoch_match = re.fullmatch(r"epoch_(\d+)", name)
    if epoch_match:
        epoch_num = int(epoch_match.group(1))
        return epoch_num, 0

    return None, None


def resolve_checkpoint_path(checkpoint):
    if not checkpoint:
        return None
    if os.path.exists(checkpoint):
        logger.info(f"Using local checkpoint path: {checkpoint}")
        return checkpoint
    logger.info(f"Checkpoint is not a local path. Downloading from Hugging Face: {checkpoint}")
    return snapshot_download(
        repo_id=checkpoint,
        allow_patterns=[
            "adapter_config.json",
            "adapter_model.safetensors",
            "adapter_model.bin",
            "qformer_bridge.safetensors",
            "qformer_bridge_config.json",
            "trajectory_branch.safetensors",
            "trajectory_branch_config.json",
            "optimizer.pt",
            "scheduler.pt",
            "tokenizer*",
            "special_tokens_map.json",
            "added_tokens.json",
        ],
    )


def resolve_resume_config(args, config):
    checkpoint = resolve_checkpoint_path(args.checkpoint)
    start_epoch = args.start_epoch
    start_step = args.start_step

    inferred_epoch, inferred_step = infer_resume_position(checkpoint) if checkpoint else (None, None)
    if start_epoch is None:
        start_epoch = inferred_epoch if inferred_epoch is not None else 0
    if start_step is None:
        start_step = inferred_step if inferred_step is not None else 0

    return checkpoint, int(start_epoch or 0), int(start_step or 0)


def build_fresh_lora_model(language_model, config, logger):
    lora_cfg = config["model"]["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["dropout"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
    )
    logger.info("Initializing a fresh LoRA adapter from config.")
    return get_peft_model(language_model, peft_config)


def validate_pretrain_checkpoint_for_finetune(model, checkpoint_dir, logger):
    bridge_metadata = read_qformer_bridge_metadata(checkpoint_dir)
    traj_metadata = read_trajectory_branch_metadata(checkpoint_dir)

    if not os.path.exists(os.path.join(checkpoint_dir, "qformer_bridge.safetensors")):
        raise FileNotFoundError(f"Pretrain checkpoint is missing qformer_bridge.safetensors: {checkpoint_dir}")
    if not os.path.exists(os.path.join(checkpoint_dir, "trajectory_branch.safetensors")):
        raise FileNotFoundError(f"Pretrain checkpoint is missing trajectory_branch.safetensors: {checkpoint_dir}")

    stages = {m.get("stage") for m in (bridge_metadata, traj_metadata) if m}
    if stages:
        if stages != {"pretrain"}:
            raise ValueError(f"Pretrain checkpoint stage metadata mismatch: {sorted(stages)}")
    else:
        logger.warning("Pretrain checkpoint has no explicit stage metadata; treating it as a legacy pretrain-compatible checkpoint.")

    checkpoint_mode = traj_metadata.get("fusion_mode")
    current_mode = getattr(model, "trajectory_fusion_mode", None)
    if checkpoint_mode and current_mode and checkpoint_mode != current_mode:
        raise ValueError(
            f"Pretrain checkpoint fusion mode mismatch: checkpoint={checkpoint_mode}, current={current_mode}"
        )

    return bridge_metadata, traj_metadata


def log_pretrain_checkpoint_verification(model, checkpoint_dir, logger):
    verification = verify_loaded_pretrain_modules(model, checkpoint_dir)
    for section_name in ("bridge", "trajectory"):
        section = verification[section_name]
        logger.info(
            "Pretrain %s verification | all_matched=%s | checked=%s",
            section_name,
            section["all_matched"],
            len(section["checked_keys"]),
        )
        for item in section["checked_keys"]:
            logger.info(
                "Pretrain %s key check | key=%s | matched=%s | model_norm=%.6f | checkpoint_norm=%.6f",
                section_name,
                item["key"],
                item["matched"],
                item["model_norm"],
                item["checkpoint_norm"],
            )
    return verification


def compute_sequence_loss(logits, labels, loss_mode: str, label_smoothing: float):
    shifted_logits = logits[..., :-1, :].contiguous()
    shifted_labels = labels[..., 1:].contiguous().to(shifted_logits.device)
    flat_logits = shifted_logits.view(-1, shifted_logits.size(-1))
    flat_labels = shifted_labels.view(-1)
    if loss_mode == "label_smoothing":
        return F.cross_entropy(
            flat_logits,
            flat_labels,
            ignore_index=-100,
            label_smoothing=label_smoothing,
        )
    if loss_mode == "cross_entropy":
        return F.cross_entropy(flat_logits, flat_labels, ignore_index=-100)
    raise ValueError(f"Unsupported loss_mode: {loss_mode}")


def build_optimizer_param_groups(model, *, lora_lr: float, bridge_lr: float, trajectory_lr: float):
    groups = {
        "trajectory": {"params": [], "param_names": [], "lr": trajectory_lr, "name": "trajectory"},
        "bridge": {"params": [], "param_names": [], "lr": bridge_lr, "name": "bridge"},
        "lora_rest": {"params": [], "param_names": [], "lr": lora_lr, "name": "lora_rest"},
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(part in name for part in ("trajectory_backbone", "trajectory_cls_head", "trajectory_token_projector")):
            group = groups["trajectory"]
        elif any(part in name for part in ("qformer_input_proj", "qformer_to_mlp1_proj")):
            group = groups["bridge"]
        else:
            group = groups["lora_rest"]
        group["params"].append(param)
        group["param_names"].append(name)

    grouped_ids = [id(param) for group in groups.values() for param in group["params"]]
    if len(grouped_ids) != len(set(grouped_ids)):
        raise ValueError("Duplicate trainable parameter detected across optimizer groups.")
    trainable_ids = {id(param) for param in model.parameters() if param.requires_grad}
    if set(grouped_ids) != trainable_ids:
        raise ValueError("Optimizer groups do not cover exactly all trainable parameters.")
    return [group for group in groups.values() if group["params"]]


def maybe_pad(inner_lists, padding_value):
    tensor_list = [torch.tensor(inner_list, dtype=torch.long) for inner_list in inner_lists]
    return pad_sequence(tensor_list, batch_first=True, padding_value=padding_value)


class CollaterFn:
    def __init__(self, tokenizer, model) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.log_token_stats = False
        self.token_log_remaining = 0
        self.alter_only = False

    def __call__(self, batch):
        label_ids_batch = []
        input_ids_batch = []
        attention_mask_batch = []
        pixel_values_batch = []
        qformer_texts = []
        trajectory_label_ids_batch = []
        trajectory_direction_ids_batch = []
        trajectory_numeric_feats_batch = []
        trajectory_object_mask_batch = []
        samples_batch = []

        for sample in batch:
            question = sample["question"]
            answer = sample["answer"]
            pixel_values = sample["pixel_values"]
            samples_batch.append(sample)

            template = get_conv_template(self.model.template)
            template.system_message = self.model.system_message
            eos_token_id = self.tokenizer.convert_tokens_to_ids(template.sep)
            eot_token_id = self.tokenizer.convert_tokens_to_ids("<|endoftext|>")

            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            num_patches_list = [pv.shape[0] for pv in pixel_values]
            total_tiles = sum(num_patches_list)
            for num_patches in num_patches_list:
                image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.model.num_image_token * num_patches + IMG_END_TOKEN
                query = query.replace("<image>", image_tokens, 1)

            input_ids = self.tokenizer.encode(query, add_special_tokens=False)
            answer_ids = self.tokenizer.encode(answer, add_special_tokens=False)
            if self.log_token_stats and self.token_log_remaining != 0:
                total_image_tokens_in_sample = total_tiles * self.model.num_image_token
                total_sequence_length = len(input_ids) + len(answer_ids) + 1
                logger.info(
                    "[INFO][ALTER_ONLY=%s] Image token stats | frames=%s | tiles_per_frame=%s | query_tokens_per_tile=%s | total_image_tokens=%s",
                    self.alter_only,
                    len(pixel_values),
                    num_patches_list,
                    self.model.num_image_token,
                    total_image_tokens_in_sample,
                )
                logger.info(
                    "[INFO][ALTER_ONLY=%s] Text tokens - input: %s, answer: %s, total: %s",
                    self.alter_only,
                    len(input_ids),
                    len(answer_ids),
                    total_sequence_length,
                )
                if self.token_log_remaining > 0:
                    self.token_log_remaining -= 1

            label_ids = [-100] * len(input_ids) + answer_ids + [eos_token_id]
            input_ids = input_ids + answer_ids + [eos_token_id]
            attention_mask = [1] * len(input_ids)
            assert len(input_ids) == len(attention_mask) == len(label_ids)

            label_ids_batch.append(label_ids)
            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
            pixel_values_batch.append(torch.cat(pixel_values, dim=0))
            if getattr(self.model, "qformer_enabled", False):
                qformer_text = sample.get("qformer_text", question.replace("<image>", "").strip())
                qformer_texts.extend([qformer_text] * total_tiles)
            if getattr(self.model, "trajectory_enabled", False):
                label_ids = sample.get("trajectory_label_ids")
                direction_ids = sample.get("trajectory_direction_ids")
                numeric_feats = sample.get("trajectory_numeric_feats")
                object_mask = sample.get("trajectory_object_mask")
                if label_ids is None or direction_ids is None or numeric_feats is None or object_mask is None:
                    raise ValueError("Trajectory-enabled sample is missing trajectory fields.")
                for _ in range(total_tiles):
                    trajectory_label_ids_batch.append(torch.as_tensor(label_ids, dtype=torch.long))
                    trajectory_direction_ids_batch.append(torch.as_tensor(direction_ids, dtype=torch.long))
                    trajectory_numeric_feats_batch.append(torch.as_tensor(numeric_feats, dtype=torch.float32))
                    trajectory_object_mask_batch.append(torch.as_tensor(object_mask, dtype=torch.long))

        input_ids_tensor = maybe_pad(input_ids_batch, eot_token_id)
        label_ids_tensor = maybe_pad(label_ids_batch, -100)
        attention_mask_tensor = maybe_pad(attention_mask_batch, 0)
        pixel_values_tensor = torch.cat(pixel_values_batch)
        qformer_inputs = None
        trajectory_inputs = None
        if getattr(self.model, "qformer_enabled", False):
            qformer_inputs = self.model.encode_qformer_texts(qformer_texts)
        if getattr(self.model, "trajectory_enabled", False):
            trajectory_inputs = (
                torch.stack(trajectory_label_ids_batch, dim=0),
                torch.stack(trajectory_direction_ids_batch, dim=0),
                torch.stack(trajectory_numeric_feats_batch, dim=0),
                torch.stack(trajectory_object_mask_batch, dim=0),
            )
        return (
            input_ids_tensor,
            label_ids_tensor,
            attention_mask_tensor,
            pixel_values_tensor,
            qformer_inputs,
            trajectory_inputs,
            samples_batch,
        )


def test_model(model, tokenizer, val_loader_with_shuffle, shuffle=False):
    model.eval()
    with torch.no_grad():
        total_test_batches = 0
        for batch in tqdm(val_loader_with_shuffle):
            _, _, _, _, _, _, samples = batch
            for sample in samples:
                pixel_values = torch.cat(sample["pixel_values"], dim=0).to(torch.bfloat16).cuda()
                generation_config = dict(max_new_tokens=512, do_sample=False)
                question = f"{sample['question']}"
                if getattr(model, "qformer_enabled", False):
                    q_ids, q_mask = model.encode_qformer_texts(
                        [sample.get("qformer_text", question.replace("<image>", "").strip())] * pixel_values.shape[0],
                        device=pixel_values.device,
                    )
                    model.set_qformer_text(q_ids, q_mask)
                if getattr(model, "trajectory_enabled", False):
                    model.set_trajectory_inputs(
                        sample["trajectory_label_ids"].unsqueeze(0).repeat(pixel_values.shape[0], 1).cuda(),
                        sample["trajectory_direction_ids"].unsqueeze(0).repeat(pixel_values.shape[0], 1).cuda(),
                        sample["trajectory_numeric_feats"].unsqueeze(0).repeat(pixel_values.shape[0], 1, 1).cuda(),
                        sample["trajectory_object_mask"].unsqueeze(0).repeat(pixel_values.shape[0], 1).cuda(),
                    )
                response = model.chat(tokenizer, pixel_values, question, generation_config)
                if getattr(model, "qformer_enabled", False):
                    model.clear_qformer_text()
                if getattr(model, "trajectory_enabled", False):
                    model.clear_trajectory_inputs()
                question_token_count = len(tokenizer.encode(question, add_special_tokens=False))
                response_token_count = len(tokenizer.encode(response, add_special_tokens=False))
                ground_truth_token_count = len(tokenizer.encode(sample["answer"], add_special_tokens=False))
                logger.info(
                    f"\nToken stats | question: {question_token_count} | "
                    f"response: {response_token_count} | ground_truth: {ground_truth_token_count}"
                )
                logger.info(f'\nUser: {question}\nAssistant: {response}\nGround truth:{sample["answer"]}\n\n')
            total_test_batches += 1
            if total_test_batches == 2:
                break


def prepare_test_auxiliary_data(config):
    index_file = "./wad_dataset/frame_index.pkl"
    if not os.path.exists(index_file):
        raise FileNotFoundError(f"Frame index not found at {index_file}.")
    with open(index_file, "rb") as f:
        frame_index = pickle.load(f)

    bbox_file = "all_bboxes_1.jsonl"
    if os.path.exists(bbox_file):
        bbox_dataset = load_dataset("json", data_files=bbox_file, split="train")
    else:
        bbox_dataset = load_dataset(config["data"]["name"], data_files="all_bboxes_1.jsonl", split="train")

    bbox_by_folder = defaultdict(lambda: defaultdict(list))
    for bbox_entry in bbox_dataset:
        folder_id = bbox_entry["folder_id"]
        frame_id = bbox_entry["frame_id"]
        bbox_by_folder[folder_id][frame_id].append(
            {
                "label": bbox_entry["label"],
                "confidence": bbox_entry["probs"],
                "bbox": bbox_entry["boxs"],
                "relative_position": bbox_entry.get("relative_position", "unknown"),
                "distance_zone": bbox_entry.get("distance_zone", "unknown"),
                "coming_to_user": bbox_entry.get("coming_to_user", False),
                "speed": bbox_entry.get("speed", 0.0),
                "danger_score": bbox_entry.get("danger_score", 0.0),
            }
        )
    trajectory_source = build_trajectory_source_from_config(config)
    return frame_index, bbox_by_folder, trajectory_source


def build_test_alter_loader(config):
    response_format = get_response_format(config)
    frame_index, bbox_by_folder, trajectory_source = prepare_test_auxiliary_data(config)
    dataset_dict = load_dataset(
        config["data"]["name"],
        data_files={"test": "test_alter.json"},
    )
    test_dataset = WADDatasetForInternVL(
        metadata_dataset=dataset_dict,
        frame_index=frame_index,
        bbox_by_folder=bbox_by_folder,
        trajectory_source=trajectory_source,
        split="test",
        response_format=response_format,
    )
    return DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda batch: batch)


def run_model_chat_for_eval(model, tokenizer, pixel_values, question, generation_config):
    num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
    template = get_conv_template(model.template)
    template.system_message = model.system_message
    eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)
    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()

    for num_patches in num_patches_list:
        image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * model.num_image_token * num_patches + IMG_END_TOKEN
        query = query.replace("<image>", image_tokens, 1)

    model_inputs = tokenizer(query, return_tensors="pt")
    embedding_device = model.language_model.get_input_embeddings().weight.device
    input_ids = model_inputs["input_ids"].to(embedding_device)
    attention_mask = model_inputs["attention_mask"].to(embedding_device)
    generation_config = dict(generation_config)
    generation_config["eos_token_id"] = eos_token_id
    generation_config["pad_token_id"] = eos_token_id
    with suppress_runtime_noise():
        generation_output = model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config,
        )
    response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
    return response.split(template.sep)[0].strip()


def run_epoch_test_infer(model, tokenizer, config, output_dir, epoch, device):
    from scripts.metrics import VLMMetrics

    test_loader = build_test_alter_loader(config)
    response_format = get_response_format(config)
    metric_target_field = "raw_text" if response_format == "direct_text" else "instruction"
    epoch_dir = os.path.join(output_dir, f"epoch_{epoch}")
    output_file = os.path.join(epoch_dir, "eval_test_alter.json")
    checkpoint_label = epoch_dir

    predictions = []
    references = []
    detailed_results = []
    evaluator = VLMMetrics()

    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            sample = batch[0]
            pixel_values = torch.cat([torch.as_tensor(p) for p in sample["pixel_values"]], dim=0)
            pixel_values = pixel_values.to(torch.bfloat16 if device.type == "cuda" else torch.float32).to(device)
            question = str(sample["question"])
            ground_truth = str(sample["answer"])
            generation_config = dict(
                max_new_tokens=512,
                num_beams=3,
                do_sample=False,
                repetition_penalty=1.3,
                early_stopping=True,
            )
            if getattr(model, "qformer_enabled", False):
                qformer_text = sample.get("qformer_text", question.replace("<image>", "").strip())
                q_ids, q_mask = model.encode_qformer_texts(
                    [qformer_text] * pixel_values.shape[0],
                    device=pixel_values.device,
                )
                model.set_qformer_text(q_ids, q_mask)
            if getattr(model, "trajectory_enabled", False):
                model.set_trajectory_inputs(
                    sample["trajectory_label_ids"].unsqueeze(0).repeat(pixel_values.shape[0], 1).to(device),
                    sample["trajectory_direction_ids"].unsqueeze(0).repeat(pixel_values.shape[0], 1).to(device),
                    sample["trajectory_numeric_feats"].unsqueeze(0).repeat(pixel_values.shape[0], 1, 1).to(device),
                    sample["trajectory_object_mask"].unsqueeze(0).repeat(pixel_values.shape[0], 1).to(device),
                )
            response = run_model_chat_for_eval(model, tokenizer, pixel_values, question, generation_config)
            if getattr(model, "qformer_enabled", False):
                model.clear_qformer_text()
            if getattr(model, "trajectory_enabled", False):
                model.clear_trajectory_inputs()

            predictions.append(response)
            references.append(ground_truth)
            detailed_results.append(
                {
                    "id": idx,
                    "question": question,
                    "prediction": response,
                    "ground_truth": ground_truth,
                }
            )

    metrics = evaluator.compute(predictions, references, target_field=metric_target_field)
    final_output = {
        "checkpoint": checkpoint_label,
        "split": "test_alter",
        "metrics": metrics,
        "samples": detailed_results,
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)
    pairs_path = write_prediction_pairs(output_file, checkpoint_label, "test_alter", detailed_results)
    logger.info("Epoch %s test_alter metrics | %s", epoch, json.dumps(metrics, ensure_ascii=False))
    logger.info("Epoch %s test_alter JSON saved at: %s", epoch, output_file)
    logger.info("Pairs JSON saved at: %s", pairs_path)
    model.train()
    if getattr(model, "qformer_enabled", False):
        model.qformer.eval()
        model.mlp1.eval()
    return {"metrics": metrics, "output_file": output_file, "pairs_file": pairs_path}


def eval_model(model, val_loader, step, epoch, epochs, loss_mode: str, label_smoothing: float, device, accelerator=None):
    model.eval()
    with torch.no_grad():
        total_eval_loss = 0.0
        total_eval_batchs = 0
        eval_desc = f"Eval @ step {step} | epoch {epoch + 1}/{epochs}"
        for batch in val_loader:
            input_ids_batch, label_ids_batch, attention_mask_batch, pixel_values_batch, qformer_inputs, trajectory_inputs, _ = batch
            input_ids_batch = input_ids_batch.to(device)
            label_ids_batch = label_ids_batch.to(device)
            attention_mask_batch = attention_mask_batch.to(device)
            pixel_values_batch = pixel_values_batch.to(torch.bfloat16 if device.type == "cuda" else torch.float32).to(device)
            image_flags_batch = torch.ones((pixel_values_batch.shape[0], 1), dtype=torch.long, device=device)
            if getattr(model, "qformer_enabled", False) and qformer_inputs is not None:
                model.set_qformer_text(qformer_inputs[0].to(device), qformer_inputs[1].to(device))
            if getattr(model, "trajectory_enabled", False) and trajectory_inputs is not None:
                model.set_trajectory_inputs(
                    trajectory_inputs[0].to(device),
                    trajectory_inputs[1].to(device),
                    trajectory_inputs[2].to(device),
                    trajectory_inputs[3].to(device),
                )

            with suppress_runtime_noise():
                outputs = model(
                    input_ids=input_ids_batch,
                    pixel_values=pixel_values_batch,
                    labels=label_ids_batch,
                    image_flags=image_flags_batch,
                    return_dict=True,
                )
            if getattr(model, "qformer_enabled", False):
                model.clear_qformer_text()
            if getattr(model, "trajectory_enabled", False):
                model.clear_trajectory_inputs()
            loss = compute_sequence_loss(
                logits=outputs.logits,
                labels=label_ids_batch,
                loss_mode=loss_mode,
                label_smoothing=label_smoothing,
            )
            if accelerator is not None:
                gathered_loss = accelerator.gather(loss.detach().reshape(1))
                total_eval_loss += float(gathered_loss.mean().item())
            else:
                total_eval_loss += loss.item()
            total_eval_batchs += 1
            if total_eval_batchs == 200:
                break
        avg_eval_loss = total_eval_loss / total_eval_batchs if total_eval_batchs > 0 else float("nan")
        if accelerator is None or accelerator.is_main_process:
            logger.info(f"Validation loss after {step} batches training in epoch {epoch + 1}/{epochs}: {avg_eval_loss:.4f}")
    model.train()
    if getattr(model, "qformer_enabled", False):
        model.qformer.eval()
        model.mlp1.eval()
    return avg_eval_loss


def train_model(
    model,
    tokenizer,
    train_loader,
    val_loader,
    val_loader_with_shuffle,
    config,
    output_dir,
    accelerator=None,
    resume_dir=None,
    start_epoch=0,
    start_step=0,
):
    epochs = config["training"]["num_epochs"]
    lr = float(config["training"]["learning_rate"])
    accum_steps = config["training"]["gradient_accumulation_steps"]
    weight_decay = float(config["training"]["weight_decay"])
    warmup_steps = config["training"]["warmup_steps"]
    max_grad_norm = float(config["training"]["max_grad_norm"])
    eval_steps = config["training"].get("eval_steps")
    log_token_stats = bool(config["training"].get("log_token_stats", False))
    train_log_interval = int(config["training"].get("train_log_interval", 100))
    loss_mode = str(config["training"].get("loss_mode", "cross_entropy"))
    label_smoothing = float(config["training"].get("label_smoothing", 0.0))
    lora_lr = float(config["training"].get("lora_learning_rate", lr))
    bridge_lr = float(config["training"].get("bridge_learning_rate", config["training"].get("proj_learning_rate", lr)))
    trajectory_lr = float(config["training"].get("trajectory_learning_rate", config["training"].get("proj_learning_rate", lr)))
    metrics_path = os.path.join(output_dir, "metrics.json")
    is_main_process = accelerator is None or accelerator.is_main_process
    device = accelerator.device if accelerator is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def save_metrics(metrics: dict):
        import json
        with open(metrics_path, "w", encoding="utf-8") as _f:
            json.dump(metrics, _f, indent=2, ensure_ascii=False)

    metrics = {"train_loss": [], "val_loss": [], "epoch_summary": []}

    if is_main_process:
        logger.info(f"Total params: {sum(p.numel() for p in model.parameters())}")
        logger.info(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        logger.info(
            "Training config: loss_mode=%s, label_smoothing=%.3f, accum_steps=%s, weight_decay=%s, "
            "lora_lr=%s, bridge_lr=%s, trajectory_lr=%s",
            loss_mode,
            label_smoothing,
            accum_steps,
            weight_decay,
            lora_lr,
            bridge_lr,
            trajectory_lr,
        )

    optimizer_groups = build_optimizer_param_groups(
        model,
        lora_lr=lora_lr,
        bridge_lr=bridge_lr,
        trajectory_lr=trajectory_lr,
    )
    if is_main_process:
        logger.info(
            "Param groups | %s",
            " | ".join(
                f"{group['name']}: {sum(p.numel() for p in group['params']):,} params @ lr={group['lr']}"
                for group in optimizer_groups
            ),
        )

    optimizer = AdamW(
        optimizer_groups,
        weight_decay=weight_decay,
        foreach=False,
    )
    enforce_safe_optimizer_param_groups(optimizer)
    save_steps = config["training"].get("save_steps")
    train_collate_fn = train_loader.collate_fn

    total_training_steps = (len(train_loader) * epochs) // accum_steps
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )
    if accelerator is not None:
        model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
            model,
            optimizer,
            train_loader,
            val_loader,
            lr_scheduler,
        )
    unwrapped_model = accelerator.unwrap_model(model) if accelerator is not None else model

    if resume_dir and os.path.exists(resume_dir):
        if is_main_process:
            logger.info(f"Resuming training from {resume_dir} | Epoch: {start_epoch+1}, Step: {start_step}")
        if getattr(unwrapped_model, "qformer_enabled", False):
            load_qformer_bridge(unwrapped_model, resume_dir, strict=True)
            align_qformer_bridge_runtime(unwrapped_model)
            if is_main_process:
                logger.info("Loaded Q-Former bridge states successfully!")
        if getattr(unwrapped_model, "trajectory_enabled", False):
            load_trajectory_branch(unwrapped_model, resume_dir, strict=True)
            align_qformer_bridge_runtime(unwrapped_model)
            if is_main_process:
                logger.info("Loaded trajectory branch states successfully!")

        opt_path = os.path.join(resume_dir, "optimizer.pt")
        sch_path = os.path.join(resume_dir, "scheduler.pt")

        if os.path.exists(opt_path) and os.path.exists(sch_path):
            optimizer_state_dict, converted, overridden_groups = sanitize_optimizer_state_dict(
                torch.load(opt_path, map_location="cpu")
            )
            optimizer.load_state_dict(optimizer_state_dict)
            overridden_groups_after_load = enforce_safe_optimizer_param_groups(optimizer)
            moved = move_optimizer_state_to_param_device(optimizer)
            lr_scheduler.load_state_dict(torch.load(sch_path))
            if converted:
                logger.info("Sanitized %s optimizer state tensors to float32 after load.", converted)
            if overridden_groups:
                logger.info("Overrode foreach/fused flags in %s optimizer param_groups before load.", overridden_groups)
            if overridden_groups_after_load:
                logger.info("Re-applied safe foreach/fused flags to %s optimizer param_groups after load.", overridden_groups_after_load)
            if moved:
                logger.info("Moved %s optimizer state tensors to parameter devices after load.", moved)
            remaining_cpu_tensors = count_optimizer_state_tensors_on_cpu(optimizer)
            if is_main_process:
                logger.info("Optimizer state CPU tensor count after load: %s", remaining_cpu_tensors)
                logger.info("Loaded Optimizer and Scheduler states successfully!")
        else:
            if is_main_process:
                logger.warning("No Optimizer/Scheduler states found in checkpoint. Starting with fresh states.")

    for epoch in range(start_epoch, epochs):
        model.train()
        if getattr(unwrapped_model, "qformer_enabled", False):
            unwrapped_model.qformer.eval()
            unwrapped_model.mlp1.eval()
        optimizer.zero_grad()

        accumulated_loss_for_log = 0.0
        set_seed(42)
        if is_main_process:
            logger.info("Epoch seed fixed | epoch=%s | seed=42", epoch + 1)
        batch_iterator = iter(train_loader)
        train_collate_fn.log_token_stats = False

        if epoch == start_epoch and start_step > 0:
            if is_main_process:
                logger.info(f" Skipping {start_step} batches to resume state...")
            for _ in tqdm(range(start_step), desc="Skipping to resume point", leave=False, disable=not is_main_process):
                next(batch_iterator)
            i = start_step
        else:
            i = 0
            
        train_collate_fn.log_token_stats = log_token_stats
        for batch in batch_iterator:
            i += 1
            input_ids_batch, label_ids_batch, attention_mask_batch, pixel_values_batch, qformer_inputs, trajectory_inputs, _ = batch

            input_ids_batch = input_ids_batch.to(device)
            label_ids_batch = label_ids_batch.to(device)
            attention_mask_batch = attention_mask_batch.to(device)
            pixel_values_batch = pixel_values_batch.to(torch.bfloat16 if device.type == "cuda" else torch.float32).to(device)
            image_flags_batch = torch.ones((pixel_values_batch.shape[0], 1), dtype=torch.long, device=device)
            if getattr(unwrapped_model, "qformer_enabled", False) and qformer_inputs is not None:
                unwrapped_model.set_qformer_text(qformer_inputs[0].to(device), qformer_inputs[1].to(device))
            if getattr(unwrapped_model, "trajectory_enabled", False) and trajectory_inputs is not None:
                unwrapped_model.set_trajectory_inputs(
                    trajectory_inputs[0].to(device),
                    trajectory_inputs[1].to(device),
                    trajectory_inputs[2].to(device),
                    trajectory_inputs[3].to(device),
                )

            context = accelerator.accumulate(model) if accelerator is not None else torch.enable_grad()
            with context:
                with suppress_runtime_noise():
                    outputs = model(
                        input_ids=input_ids_batch,
                        pixel_values=pixel_values_batch,
                        labels=label_ids_batch,
                        image_flags=image_flags_batch,
                        return_dict=True,
                    )
                if getattr(unwrapped_model, "qformer_enabled", False):
                    unwrapped_model.clear_qformer_text()
                if getattr(unwrapped_model, "trajectory_enabled", False):
                    unwrapped_model.clear_trajectory_inputs()

                raw_loss = compute_sequence_loss(
                    logits=outputs.logits,
                    labels=label_ids_batch,
                    loss_mode=loss_mode,
                    label_smoothing=label_smoothing,
                )
                loss = raw_loss / accum_steps
                if accelerator is not None:
                    accelerator.backward(loss)
                else:
                    loss.backward()

                gathered_loss = accelerator.gather(raw_loss.detach().reshape(1)) if accelerator is not None else raw_loss.detach().reshape(1)
                mean_loss = float(gathered_loss.mean().item())
                accumulated_loss_for_log += mean_loss

                should_step = accelerator.sync_gradients if accelerator is not None else (i % accum_steps == 0)
                if should_step:
                    if accelerator is not None:
                        accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                    avg_loss = accumulated_loss_for_log / accum_steps
                    global_step = i // accum_steps
                    if is_main_process:
                        metrics["train_loss"].append({"step": global_step, "epoch": epoch + 1, "loss": round(avg_loss, 6)})
                        if global_step % train_log_interval == 0:
                            current_lr = lr_scheduler.get_last_lr()[0] if lr_scheduler is not None else lora_lr
                            logger.info(
                                "Train progress | epoch=%s/%s | batch=%s/%s | opt_step=%s | avg_loss=%.4f | lr=%.6g",
                                epoch + 1,
                                epochs,
                                i,
                                len(train_loader),
                                global_step,
                                avg_loss,
                                current_lr,
                            )
                            save_metrics(metrics)
                    accumulated_loss_for_log = 0.0

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

            if eval_steps and i % eval_steps == 0:
                if is_main_process:
                    logger.info(f"Running evaluation at step {i}...")
                val_loss = eval_model(model, val_loader, i, epoch, epochs, loss_mode, label_smoothing, device, accelerator=accelerator)
                if val_loss is not None and is_main_process:
                    metrics["val_loss"].append({"step": i, "epoch": epoch + 1, "loss": round(val_loss, 6)})
                    save_metrics(metrics)

            if save_steps and i % save_steps == 0 and is_main_process:
                if accelerator is not None:
                    accelerator.wait_for_everyone()
                step_save_dir = f"{output_dir}/epoch_{epoch+1}_step_{i}/"
                os.makedirs(step_save_dir, exist_ok=True)
                logger.info(f"Saving model, tokenizer, opt, scheduler at step {i} to {step_save_dir}")

                unwrapped_model.language_model.save_pretrained(step_save_dir)
                save_qformer_bridge(unwrapped_model, step_save_dir)
                save_trajectory_branch(unwrapped_model, step_save_dir)
                tokenizer.save_pretrained(step_save_dir)
                optimizer_state_dict, converted, overridden_groups = export_sanitized_optimizer_state_dict(optimizer)
                if converted:
                    logger.info("Sanitized %s optimizer state tensors to float32 before save.", converted)
                if overridden_groups:
                    logger.info("Normalized foreach/fused flags in %s optimizer param_groups before save.", overridden_groups)
                torch.save(optimizer_state_dict, os.path.join(step_save_dir, "optimizer.pt"))
                torch.save(lr_scheduler.state_dict(), os.path.join(step_save_dir, "scheduler.pt"))

        if accelerator is not None:
            accelerator.wait_for_everyone()

        if is_main_process:
            epoch_save_dir = f"{output_dir}/epoch_{epoch+1}/"
            os.makedirs(epoch_save_dir, exist_ok=True)
            logger.info(f"Saving model and tokenizer for epoch {epoch+1} to {epoch_save_dir}")
            unwrapped_model.language_model.save_pretrained(epoch_save_dir)
            save_qformer_bridge(unwrapped_model, epoch_save_dir)
            save_trajectory_branch(unwrapped_model, epoch_save_dir)
            tokenizer.save_pretrained(epoch_save_dir)
            optimizer_state_dict, converted, overridden_groups = export_sanitized_optimizer_state_dict(optimizer)
            if converted:
                logger.info("Sanitized %s optimizer state tensors to float32 before save.", converted)
            if overridden_groups:
                logger.info("Normalized foreach/fused flags in %s optimizer param_groups before save.", overridden_groups)
            torch.save(optimizer_state_dict, os.path.join(epoch_save_dir, "optimizer.pt"))
            torch.save(lr_scheduler.state_dict(), os.path.join(epoch_save_dir, "scheduler.pt"))

            epoch_train = [e["loss"] for e in metrics["train_loss"] if e["epoch"] == epoch + 1]
            avg_epoch_loss = sum(epoch_train) / len(epoch_train) if epoch_train else float("nan")
            metrics["epoch_summary"].append({"epoch": epoch + 1, "avg_train_loss": round(avg_epoch_loss, 6)})
            logger.info(f"Epoch {epoch+1} summary | avg_train_loss={avg_epoch_loss:.4f}")
            test_summary = run_epoch_test_infer(
                model=unwrapped_model,
                tokenizer=tokenizer,
                config=config,
                output_dir=output_dir,
                epoch=epoch + 1,
                device=device,
            )
            metrics["epoch_summary"][-1]["test_alter_metrics"] = {
                key: round(float(value), 6) for key, value in test_summary["metrics"].items()
            }
            metrics["epoch_summary"][-1]["test_alter_output_file"] = test_summary["output_file"]
            metrics["epoch_summary"][-1]["test_alter_pairs_file"] = test_summary["pairs_file"]
            save_metrics(metrics)
        if accelerator is not None:
            accelerator.wait_for_everyone()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    use_accelerate = bool(config["training"].get("use_accelerate", False))
    accum_steps = int(config["training"]["gradient_accumulation_steps"])
    accelerator = Accelerator(gradient_accumulation_steps=accum_steps) if use_accelerate else None
    is_main_process = accelerator is None or accelerator.is_main_process
    world_size = accelerator.num_processes if accelerator is not None else 1
    local_rank = accelerator.local_process_index if accelerator is not None else 0
    distributed = world_size > 1

    output_dir = build_output_dir(config) if is_main_process else None
    output_dir = broadcast_output_dir(output_dir, accelerator)
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        init_logger(output_dir)
        logger = get_logger()
    else:
        logger = SilentLogger()

    if args.checkpoint and args.pretrain_checkpoint:
        raise ValueError("Use either --checkpoint for finetune resume or --pretrain_checkpoint for pretrain preload, not both.")
    resume_dir, start_epoch, start_step = resolve_resume_config(args, config)
    pretrain_checkpoint_dir = resolve_checkpoint_path(args.pretrain_checkpoint) if args.pretrain_checkpoint else None
    model_name_or_path = config["model"]["name"]
    batch_size = config["training"]["batch_size"]
    quant_enabled = bool(config["model"]["quantization"]["enabled"])
    quantization_config = None
    if quant_enabled:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if config["model"]["quantization"]["compute_dtype"] == "bfloat16" else torch.float16,
            bnb_4bit_use_double_quant=config["model"]["quantization"]["double_quant"],
            bnb_4bit_quant_type=config["model"]["quantization"]["type"],
        )

    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "trust_remote_code": config["model"]["trust_remote_code"],
    }
    if "attn_implementation" in config["model"]:
        model_kwargs["attn_implementation"] = config["model"]["attn_implementation"]
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        if distributed and torch.cuda.is_available():
            model_kwargs["device_map"] = {"": local_rank}

    if is_main_process:
        logger.info(
            "Distributed runtime | distributed=%s | world_size=%s | local_rank=%s | quantization_enabled=%s | bf16=%s",
            distributed,
            world_size,
            local_rank,
            quant_enabled,
            bool(config["training"].get("bf16", False)),
        )
        logger.info(
            "Concat best-shot runtime | trajectory_mode=%s | alter_only=%s | seed=%s | lora_r=%s | batch_size=%s | accum_steps=%s | global_batch=%s",
            config["trajectory"]["fusion_mode"],
            bool(config["data"].get("alter_only", False)),
            config["data"].get("seed", 42),
            config["model"]["lora"]["r"],
            batch_size,
            accum_steps,
            int(batch_size) * int(accum_steps) * world_size,
        )
        logger.info(
            "Attention runtime config | attn_implementation=%s",
            config["model"].get("attn_implementation", "default"),
        )

    logger.info("Loading model %s | quantization_enabled=%s", model_name_or_path, quant_enabled)
    model = AutoModel.from_pretrained(model_name_or_path, **model_kwargs)
    if is_main_process:
        log_flash_attention_runtime(model, logger)

    model.config.use_cache = False
    if config["training"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, use_fast=False)
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.system_message = SYSTEM_MESSAGE

    if config["model"]["vision"]["freeze_encoder"]:
        model.vision_model.requires_grad_(False)
    if qformer_enabled(config):
        attach_qformer_bridge(model, config, logger=logger)
        align_qformer_bridge_runtime(model)

    logger.info("Applying LoRA...")
    if quant_enabled:
        model.language_model = prepare_model_for_kbit_training(model.language_model)

    if hasattr(model.language_model, "get_input_embeddings"):
        model.language_model.get_input_embeddings().to(torch.bfloat16)

    if resume_dir:
        logger.info(f"Loading LoRA adapter from checkpoint: {resume_dir}")
        model.language_model = PeftModel.from_pretrained(
            model.language_model,
            resume_dir,
            is_trainable=True,
        )
    else:
        model.language_model = build_fresh_lora_model(model.language_model, config, logger)
        if pretrain_checkpoint_dir:
            logger.info(f"Preloading qformer bridge + trajectory branch from pretrain checkpoint: {pretrain_checkpoint_dir}")
            validate_pretrain_checkpoint_for_finetune(model, pretrain_checkpoint_dir, logger)
            load_qformer_bridge(model, pretrain_checkpoint_dir, strict=True)
            align_qformer_bridge_runtime(model)
            if getattr(model, "trajectory_enabled", False):
                load_trajectory_branch(model, pretrain_checkpoint_dir, strict=True)
                align_qformer_bridge_runtime(model)
            log_pretrain_checkpoint_verification(model, pretrain_checkpoint_dir, logger)

    apply_mode_gated_trajectory_trainability(model, logger)

    if is_main_process:
        model.language_model.print_trainable_parameters()
    model.train()

    logger.info("Building dataset...")
    train_dataset, val_dataset = build_dataset(config)

    collate_fn_wrapper = CollaterFn(tokenizer, model)
    collate_fn_wrapper.log_token_stats = bool(config["training"].get("log_token_stats", False))
    collate_fn_wrapper.token_log_remaining = int(config["training"].get("token_log_batches", 0))
    collate_fn_wrapper.alter_only = bool(config["data"].get("alter_only", False))

    if is_main_process:
        logger.info(
            "Runtime check | qformer_enabled=%s | trajectory_enabled=%s | trajectory_mode=%s | trajectory_source=%s | num_image_token=%s | qformer_tokens=%s | log_token_stats=%s | token_log_batches=%s | alter_only=%s | loss_mode=%s | label_smoothing=%.3f",
            getattr(model, "qformer_enabled", False),
            getattr(model, "trajectory_enabled", False),
            getattr(model, "trajectory_fusion_mode", "disabled"),
            getattr(model, "trajectory_source_file", "n/a"),
            getattr(model, "num_image_token", "unknown"),
            getattr(model, "qformer_num_query_tokens", getattr(model, "num_image_token", "unknown")),
            collate_fn_wrapper.log_token_stats,
            collate_fn_wrapper.token_log_remaining,
            collate_fn_wrapper.alter_only,
            config["training"].get("loss_mode", "cross_entropy"),
            float(config["training"].get("label_smoothing", 0.0)),
        )

    dataloader_kwargs = build_dataloader_kwargs(config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn_wrapper,
        shuffle=True,
        **dataloader_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn_wrapper,
        shuffle=False,
        **dataloader_kwargs,
    )

    val_loader_with_shuffle = DataLoader(
        val_dataset,
        batch_size=1,
        collate_fn=collate_fn_wrapper,
        shuffle=True,
        **dataloader_kwargs,
    )

    logger.info("STARTING TRAINING...")
    train_model(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        val_loader_with_shuffle=val_loader_with_shuffle,
        config=config,
        output_dir=output_dir,
        accelerator=accelerator,
        resume_dir=resume_dir,
        start_epoch=start_epoch,
        start_step=start_step,
    )
