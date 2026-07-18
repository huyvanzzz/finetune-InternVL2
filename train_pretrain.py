import argparse
import copy
import datetime
import importlib.metadata
import json
import os
import random
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from math import floor
from typing import Dict

import numpy as np
import torch
import yaml
from accelerate import Accelerator
from huggingface_hub import snapshot_download
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig, get_cosine_schedule_with_warmup

from logutil import get_logger, init_logger
from model.conversation import get_conv_template
from optimizer_state_utils import enforce_safe_optimizer_param_groups
from pretrain_dataset import DEFAULT_FRAME_INDEX_PATH, build_pretrain_datasets
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
    build_trajectory_source_from_config,
    load_trajectory_branch,
    read_trajectory_branch_metadata,
    save_trajectory_branch,
    trajectory_enabled,
)


IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
SYSTEM_MESSAGE = "You are a navigation assistant for visually impaired users."
EARLY_STOPPING_STATE_FILENAME = "early_stopping_state.json"


class SilentLogger:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None


@dataclass
class EarlyStoppingState:
    patience: int
    min_delta: float
    best_val_loss: float | None = None
    best_epoch: int | None = None
    num_bad_epochs: int = 0

    def update(self, epoch: int, val_loss: float):
        improved = self.best_val_loss is None or val_loss < (self.best_val_loss - self.min_delta)
        if improved:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            self.num_bad_epochs = 0
            return True, False

        self.num_bad_epochs += 1
        should_stop = self.num_bad_epochs >= self.patience
        return False, should_stop


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _package_versions() -> Dict[str, str]:
    packages = ("torch", "transformers", "accelerate", "bitsandbytes", "flash-attn")
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _build_run_metadata(config: Dict, global_optimizer_step: int = 0) -> Dict:
    return {
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "global_optimizer_step": int(global_optimizer_step),
        "git_commit": _git_commit(),
        "package_versions": _package_versions(),
        "config_snapshot": copy.deepcopy(config),
    }


def _write_run_metadata(output_dir: str, metadata: Dict):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "pretrain_run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def build_dataloader_kwargs(config: Dict) -> Dict:
    hardware_cfg = config.get("hardware", {})
    num_workers = int(hardware_cfg.get("num_workers", hardware_cfg.get("num_workers_per_process", 0)))
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": bool(hardware_cfg.get("pin_memory", False)),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(hardware_cfg.get("persistent_workers", False))
        if "prefetch_factor" in hardware_cfg:
            kwargs["prefetch_factor"] = int(hardware_cfg["prefetch_factor"])
    return kwargs


def inspect_optimizer_param_groups(model, optimizer) -> Dict:
    trainable = {id(p): name for name, p in model.named_parameters() if p.requires_grad}
    seen = {}
    duplicate_names = []
    for group_idx, group in enumerate(optimizer.param_groups):
        for param in group["params"]:
            param_id = id(param)
            if param_id in seen and param_id in trainable:
                duplicate_names.append(trainable[param_id])
            seen[param_id] = group_idx

    missing_names = [name for param_id, name in trainable.items() if param_id not in seen]
    return {
        "missing_param_names": sorted(missing_names),
        "duplicate_param_names": sorted(set(duplicate_names)),
        "trainable_param_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def log_optimizer_param_group_health(model, optimizer, logger):
    report = inspect_optimizer_param_groups(model, optimizer)
    logger.info(
        "Optimizer health | trainable_params=%s | missing=%s | duplicates=%s",
        report["trainable_param_count"],
        len(report["missing_param_names"]),
        len(report["duplicate_param_names"]),
    )
    if report["missing_param_names"]:
        logger.warning("Optimizer missing trainable params: %s", report["missing_param_names"][:20])
    if report["duplicate_param_names"]:
        logger.warning("Optimizer duplicate trainable params: %s", report["duplicate_param_names"][:20])
    return report


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


def reduce_token_weighted_loss(loss_sum: torch.Tensor, token_count: torch.Tensor, accelerator=None) -> torch.Tensor:
    loss_sum = loss_sum.to(dtype=torch.float32)
    token_count = token_count.to(dtype=torch.float32)
    if accelerator is not None:
        loss_sum = accelerator.reduce(loss_sum, reduction="sum")
        token_count = accelerator.reduce(token_count, reduction="sum")
    return loss_sum / token_count.clamp_min(1.0)


def count_valid_target_tokens(batch) -> torch.Tensor:
    labels = batch[1]
    return (labels != -100).sum().to(dtype=torch.float32)


def log_gradient_health(model, logger, max_names: int = 20):
    branch_sums = {
        "trajectory": 0.0,
        "bridge": 0.0,
        "other": 0.0,
    }
    grad_none = []
    bad_grad = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            grad_none.append(name)
            continue
        grad = param.grad.detach()
        if not torch.isfinite(grad).all():
            bad_grad.append(name)
            continue
        norm = float(grad.float().norm().item())
        if "trajectory_" in name:
            branch_sums["trajectory"] += norm
        elif "qformer_input_proj" in name or "qformer_to_mlp1_proj" in name:
            branch_sums["bridge"] += norm
        else:
            branch_sums["other"] += norm
    logger.info("Gradient health | norms=%s | grad_none=%s | bad_grad=%s", branch_sums, len(grad_none), len(bad_grad))
    if grad_none:
        logger.warning("Trainable params with grad=None: %s", grad_none[:max_names])
    if bad_grad:
        logger.warning("Trainable params with NaN/Inf gradients: %s", bad_grad[:max_names])


class PretrainCollaterFn:
    def __init__(self, tokenizer, model) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.log_token_stats = False
        self.token_log_remaining = 0

    @staticmethod
    def _safe_logger():
        try:
            return get_logger()
        except AssertionError:
            return SilentLogger()

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
                logger = self._safe_logger()
                logger.info(
                    "[INFO] Pretrain image token stats | frames=%s | tiles_per_frame=%s | query_tokens_per_tile=%s | total_image_tokens=%s",
                    len(pixel_values),
                    num_patches_list,
                    self.model.num_image_token,
                    total_image_tokens_in_sample,
                )
                logger.info(
                    "[INFO] Pretrain text tokens | input=%s | answer=%s | total=%s | total_input_tokens=%s",
                    len(input_ids),
                    len(answer_ids),
                    total_sequence_length,
                    len(input_ids),
                )
                if self.token_log_remaining > 0:
                    self.token_log_remaining -= 1

            label_ids = [-100] * len(input_ids) + answer_ids + [eos_token_id]
            input_ids = input_ids + answer_ids + [eos_token_id]
            attention_mask = [1] * len(input_ids)

            label_ids_batch.append(label_ids)
            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
            pixel_values_batch.append(torch.cat(pixel_values, dim=0))

            if getattr(self.model, "qformer_enabled", False):
                qformer_text = sample.get("qformer_text", question.replace("<image>", "").strip())
                qformer_texts.extend([qformer_text] * total_tiles)

            if getattr(self.model, "trajectory_enabled", False):
                for _ in range(total_tiles):
                    trajectory_label_ids_batch.append(torch.as_tensor(sample["trajectory_label_ids"], dtype=torch.long))
                    trajectory_direction_ids_batch.append(torch.as_tensor(sample["trajectory_direction_ids"], dtype=torch.long))
                    trajectory_numeric_feats_batch.append(torch.as_tensor(sample["trajectory_numeric_feats"], dtype=torch.float32))
                    trajectory_object_mask_batch.append(torch.as_tensor(sample["trajectory_object_mask"], dtype=torch.long))

        input_ids_tensor = _maybe_pad(input_ids_batch, eot_token_id)
        label_ids_tensor = _maybe_pad(label_ids_batch, -100)
        attention_mask_tensor = _maybe_pad(attention_mask_batch, 0)
        pixel_values_tensor = torch.cat(pixel_values_batch)

        qformer_inputs = None
        if getattr(self.model, "qformer_enabled", False):
            qformer_inputs = self.model.encode_qformer_texts(qformer_texts)

        trajectory_inputs = None
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


def _maybe_pad(inner_lists, padding_value):
    tensor_list = [torch.tensor(inner_list, dtype=torch.long) for inner_list in inner_lists]
    return pad_sequence(tensor_list, batch_first=True, padding_value=padding_value)


def forward_pretrain_batch(model, batch, device: torch.device):
    input_ids_batch, label_ids_batch, attention_mask_batch, pixel_values_batch, qformer_inputs, trajectory_inputs, _ = batch
    input_ids_batch = input_ids_batch.to(device)
    label_ids_batch = label_ids_batch.to(device)
    attention_mask_batch = attention_mask_batch.to(device)
    pixel_values_batch = pixel_values_batch.to(
        torch.bfloat16 if device.type == "cuda" else torch.float32
    ).to(device)
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

    try:
        outputs = model(
            input_ids=input_ids_batch,
            pixel_values=pixel_values_batch,
            labels=label_ids_batch,
            image_flags=image_flags_batch,
            return_dict=True,
        )
    finally:
        if getattr(model, "qformer_enabled", False):
            model.clear_qformer_text()
        if getattr(model, "trajectory_enabled", False):
            model.clear_trajectory_inputs()
    return outputs.loss


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain InternVL-QFormer trajectory branch with generation loss.")
    parser.add_argument("--config", type=str, required=True, help="Path to pretrain YAML config.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional pretrain checkpoint dir or HF repo id to resume pretrain.",
    )
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Alias for --checkpoint when resuming pretrain.",
    )
    parser.add_argument("--start_epoch", type=int, default=None, help="Zero-based epoch index to resume from.")
    parser.add_argument("--start_step", type=int, default=None, help="Batch step inside the resume epoch.")
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_checkpoint_path(checkpoint: str | None, logger=None):
    if not checkpoint:
        return None
    if os.path.exists(checkpoint):
        if logger:
            logger.info(f"Using local pretrain checkpoint path: {checkpoint}")
        return checkpoint
    if logger:
        logger.info(f"Pretrain checkpoint is not a local path. Downloading from Hugging Face: {checkpoint}")
    return snapshot_download(
        repo_id=checkpoint,
        allow_patterns=[
            "qformer_bridge.safetensors",
            "qformer_bridge_config.json",
            "trajectory_branch.safetensors",
            "trajectory_branch_config.json",
            "optimizer.pt",
            "scheduler.pt",
            "training_state.json",
            EARLY_STOPPING_STATE_FILENAME,
            "tokenizer*",
            "special_tokens_map.json",
            "added_tokens.json",
        ],
    )


def infer_resume_position(checkpoint_dir):
    state = read_training_state(checkpoint_dir)
    if state is not None:
        return int(state.get("next_epoch", 0)), int(state.get("next_step", 0))

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


def read_training_state(checkpoint_dir):
    state_path = os.path.join(checkpoint_dir, "training_state.json")
    if not os.path.exists(state_path):
        return None
    with open(state_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_resume_config(args, logger):
    requested_checkpoint = args.resume_checkpoint or args.checkpoint
    checkpoint = resolve_checkpoint_path(requested_checkpoint, logger=logger)
    start_epoch = args.start_epoch
    start_step = args.start_step
    inferred_epoch, inferred_step = infer_resume_position(checkpoint) if checkpoint else (None, None)
    if checkpoint and logger:
        logger.info("Resolved pretrain resume checkpoint path: %s", checkpoint)
        logger.info(
            "Resume state discovery | training_state.json=%s | inferred_epoch=%s | inferred_step=%s",
            os.path.exists(os.path.join(checkpoint, "training_state.json")),
            inferred_epoch if inferred_epoch is not None else "missing",
            inferred_step if inferred_step is not None else "missing",
        )
    if start_epoch is None:
        start_epoch = inferred_epoch if inferred_epoch is not None else 0
    if start_step is None:
        start_step = inferred_step if inferred_step is not None else 0
    if checkpoint and logger:
        logger.info("Final pretrain resume position | start_epoch=%s | start_step=%s", start_epoch, start_step)
    return checkpoint, int(start_epoch or 0), int(start_step or 0)


def build_output_dir(config: Dict) -> str:
    base_out_dir = config["training"]["output_dir"]
    mode = str(config["trajectory"]["fusion_mode"])
    movement_flag = "movement_on" if bool(config["pretrain"]["movement_enabled"]) else "movement_off"
    run_name = f"stage_pretrain__{mode}__{movement_flag}__{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    output_dir = os.path.join(base_out_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def build_model_and_tokenizer(config: Dict, logger):
    model_name_or_path = config["model"]["name"]
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=config["model"]["quantization"]["enabled"],
        bnb_4bit_compute_dtype=torch.bfloat16 if config["model"]["quantization"]["compute_dtype"] == "bfloat16" else torch.float16,
        bnb_4bit_use_double_quant=config["model"]["quantization"]["double_quant"],
        bnb_4bit_quant_type=config["model"]["quantization"]["type"],
    )

    logger.info("Loading model %s in pretrain mode...", model_name_or_path)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device_map = None
    if config["model"]["quantization"]["enabled"] and world_size > 1 and torch.cuda.is_available():
        device_map = {"": local_rank}
        logger.info("Using explicit LOCAL_RANK device_map for 4-bit distributed training: %s", device_map)
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "quantization_config": quantization_config,
        "low_cpu_mem_usage": True,
        "trust_remote_code": config["model"]["trust_remote_code"],
    }
    if device_map is not None:
        model_kwargs["device_map"] = device_map
    if "attn_implementation" in config["model"]:
        model_kwargs["attn_implementation"] = config["model"]["attn_implementation"]
    model = AutoModel.from_pretrained(
        model_name_or_path,
        **model_kwargs,
    )
    model.config.use_cache = False
    if config["training"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, use_fast=False)
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.system_message = SYSTEM_MESSAGE

    if config["model"]["vision"]["freeze_encoder"]:
        model.vision_model.requires_grad_(False)
    if not qformer_enabled(config):
        raise ValueError("Pretrain core only supports InternVL-QFormer with qformer.enabled=true.")

    attach_qformer_bridge(model, config, logger=logger)
    align_qformer_bridge_runtime(model)

    if not trajectory_enabled(config):
        raise ValueError("Pretrain core requires trajectory.enabled=true.")

    model.pretrain_stage = "pretrain"
    model.training_stage = "pretrain"
    model.pretrain_data_source = config["pretrain"]["question_train_file"]
    model.question_format_version = "v1_qa_question_answer"
    model.pretrain_movement_enabled = bool(config["pretrain"]["movement_enabled"])

    model.language_model.requires_grad_(False)
    _freeze_modules_for_pretrain(model)
    logger.info(
        "Mode-gated trajectory heads | mode=%s | cls_head_trainable=%s | token_projector_trainable=%s",
        getattr(model, "trajectory_fusion_mode", "unknown"),
        _active_trajectory_heads(model)["trajectory_cls_head"],
        _active_trajectory_heads(model)["trajectory_token_projector"],
    )
    model.train()
    return model, tokenizer


def _freeze_modules_for_pretrain(model):
    if hasattr(model, "qformer"):
        model.qformer.requires_grad_(False)
    if hasattr(model, "qformer_query_tokens"):
        model.qformer_query_tokens.requires_grad_(False)
    if hasattr(model, "mlp1"):
        model.mlp1.requires_grad_(False)
    if hasattr(model, "vision_model"):
        model.vision_model.requires_grad_(False)

    for module_name in (
        "qformer_input_proj",
        "qformer_to_mlp1_proj",
        "trajectory_backbone",
    ):
        module = getattr(model, module_name, None)
        if module is not None:
            module.requires_grad_(True)

    fusion_mode = str(getattr(model, "trajectory_fusion_mode", "cls_add"))
    cls_head = getattr(model, "trajectory_cls_head", None)
    token_projector = getattr(model, "trajectory_token_projector", None)

    if cls_head is not None:
        cls_head.requires_grad_(fusion_mode in {"cls_add", "dual"})
    if token_projector is not None:
        token_projector.requires_grad_(fusion_mode in {"concat", "dual"})


def _active_trajectory_heads(model) -> Dict[str, bool]:
    cls_head = getattr(model, "trajectory_cls_head", None)
    token_projector = getattr(model, "trajectory_token_projector", None)
    return {
        "trajectory_cls_head": bool(cls_head is not None and any(p.requires_grad for p in cls_head.parameters())),
        "trajectory_token_projector": bool(
            token_projector is not None and any(p.requires_grad for p in token_projector.parameters())
        ),
    }


def build_dataloaders(config: Dict, model, tokenizer):
    pretrain_cfg = config["pretrain"]
    train_dataset, val_dataset, test_dataset, split_stats = build_pretrain_datasets(
        question_train_file=pretrain_cfg["question_train_file"],
        frame_index_path=pretrain_cfg.get("frame_index_file", DEFAULT_FRAME_INDEX_PATH),
        trajectory_source=build_trajectory_source_from_config(config),
        val_split_ratio=float(pretrain_cfg["val_split_ratio"]),
        val_split_seed=int(pretrain_cfg["val_split_seed"]),
        movement_enabled=bool(pretrain_cfg["movement_enabled"]),
        train_split_ratio=float(pretrain_cfg.get("train_split_ratio", 0.8)),
        test_split_ratio=float(pretrain_cfg.get("test_split_ratio", 0.1)),
        question_val_file=pretrain_cfg.get("question_val_file"),
        question_test_file=pretrain_cfg.get("question_test_file"),
    )

    collate_fn = PretrainCollaterFn(tokenizer, model)
    collate_fn.log_token_stats = bool(config["training"].get("log_token_stats", False))
    collate_fn.token_log_remaining = int(config["training"].get("token_log_batches", 0))

    batch_size = int(config["training"]["batch_size"])
    dataloader_kwargs = build_dataloader_kwargs(config)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        **dataloader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        **dataloader_kwargs,
    )
    return train_dataset, val_dataset, test_dataset, split_stats, train_loader, val_loader


def resolve_warmup_steps(config: Dict, total_training_steps: int) -> int:
    training_cfg = config["training"]
    if "warmup_ratio" in training_cfg:
        warmup_ratio = float(training_cfg.get("warmup_ratio", 0.0))
        warmup_min_steps = int(training_cfg.get("warmup_min_steps", 0))
        warmup_max_steps = int(training_cfg.get("warmup_max_steps", total_training_steps))
        computed = floor(total_training_steps * warmup_ratio)
        return max(warmup_min_steps, min(warmup_max_steps, computed))
    return int(training_cfg.get("warmup_steps", 0))


def _build_optimizer(model, config: Dict, logger):
    base_lr = float(config["training"]["learning_rate"])
    trajectory_lr = float(config["training"].get("trajectory_learning_rate", config["training"].get("proj_learning_rate", base_lr)))
    bridge_lr = float(config["training"].get("bridge_learning_rate", base_lr))
    weight_decay = float(config["training"]["weight_decay"])

    bridge_param_names = {
        "qformer_input_proj",
        "qformer_to_mlp1_proj",
    }
    trajectory_param_names = {
        "trajectory_backbone",
        "trajectory_cls_head",
        "trajectory_token_projector",
    }

    bridge_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and any(pn in n for pn in bridge_param_names)
    ]
    trajectory_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad and any(pn in n for pn in trajectory_param_names)
    ]
    other_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad
        and not any(pn in n for pn in bridge_param_names)
        and not any(pn in n for pn in trajectory_param_names)
    ]
    logger.info(
        "Pretrain param groups | trajectory=%s params @ lr=%s | bridge=%s params @ lr=%s | other=%s params @ lr=%s",
        sum(p.numel() for p in trajectory_params),
        trajectory_lr,
        sum(p.numel() for p in bridge_params),
        bridge_lr,
        sum(p.numel() for p in other_params),
        base_lr,
    )

    param_groups = []
    if trajectory_params:
        param_groups.append({"params": trajectory_params, "lr": trajectory_lr})
    if bridge_params:
        param_groups.append({"params": bridge_params, "lr": bridge_lr})
    if other_params:
        param_groups.append({"params": other_params, "lr": base_lr})

    optimizer = AdamW(param_groups, weight_decay=weight_decay, foreach=False)
    enforce_safe_optimizer_param_groups(optimizer)
    return optimizer


def _save_pretrain_checkpoint(model, tokenizer, output_dir: str, optimizer=None, lr_scheduler=None, run_metadata=None):
    os.makedirs(output_dir, exist_ok=True)
    save_qformer_bridge(model, output_dir)
    save_trajectory_branch(model, output_dir)
    tokenizer.save_pretrained(output_dir)
    if optimizer is not None:
        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
    if lr_scheduler is not None:
        torch.save(lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
    if run_metadata is not None:
        _write_run_metadata(output_dir, run_metadata)


def _write_training_state(output_dir: str, next_epoch: int, next_step: int = 0, global_optimizer_step: int = 0):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "training_state.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "next_epoch": int(next_epoch),
                "next_step": int(next_step),
                "global_optimizer_step": int(global_optimizer_step),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )


def _write_early_stopping_state(output_dir: str, early_stopping: EarlyStoppingState):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, EARLY_STOPPING_STATE_FILENAME), "w", encoding="utf-8") as f:
        json.dump(
            {
                "patience": int(early_stopping.patience),
                "min_delta": float(early_stopping.min_delta),
                "best_val_loss": None if early_stopping.best_val_loss is None else float(early_stopping.best_val_loss),
                "best_epoch": None if early_stopping.best_epoch is None else int(early_stopping.best_epoch),
                "num_bad_epochs": int(early_stopping.num_bad_epochs),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )


def _load_early_stopping_state(checkpoint_dir: str) -> EarlyStoppingState | None:
    state_path = os.path.join(checkpoint_dir, EARLY_STOPPING_STATE_FILENAME)
    if not os.path.exists(state_path):
        return None
    with open(state_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return EarlyStoppingState(
        patience=int(payload["patience"]),
        min_delta=float(payload["min_delta"]),
        best_val_loss=None if payload.get("best_val_loss") is None else float(payload["best_val_loss"]),
        best_epoch=None if payload.get("best_epoch") is None else int(payload["best_epoch"]),
        num_bad_epochs=int(payload.get("num_bad_epochs", 0)),
    )


def _log_error_stats(logger, prefix: str, stats: Dict):
    logger.info(
        "%s sample errors | count=%s | rate=%.4f | requested=%s",
        prefix,
        stats["sample_error_count"],
        stats["sample_error_rate"],
        stats["requested_count"],
    )
    for example in stats["error_examples"]:
        logger.info("%s error example | %s", prefix, json.dumps(example, ensure_ascii=False))


def _print_debug_samples(logger, dataset, prefix: str, limit: int = 2):
    if len(dataset) == 0:
        return
    for idx in range(min(limit, len(dataset))):
        snapshot = dataset.get_debug_snapshot(idx)
        logger.info(
            "[DEBUG] %s sample %s | frame_path=%s | frame_id=%s",
            prefix,
            idx + 1,
            snapshot["frame_path"],
            snapshot["frame_id"],
        )
        logger.info("[DEBUG] question: %s", snapshot["question"])
        logger.info("[DEBUG] qformer_text: %s", snapshot["qformer_text"])
        logger.info("[DEBUG] answer: %s", snapshot["answer"])
        logger.info(
            "[DEBUG] trajectory | label_ids=%s | direction_ids=%s | object_mask=%s | numeric_feats=%s",
            snapshot["trajectory_label_ids"],
            snapshot["trajectory_direction_ids"],
            snapshot["trajectory_object_mask"],
            snapshot["trajectory_numeric_feats"],
        )


def eval_pretrain(model, val_loader, epoch: int, epochs: int, device: torch.device, logger, accelerator=None):
    model.eval()
    total_loss_sum = torch.tensor(0.0, device=device)
    total_valid_tokens = torch.tensor(0.0, device=device)
    val_loader.dataset.reset_error_stats()
    with torch.no_grad():
        iterator = tqdm(
            val_loader,
            desc=f"Pretrain eval {epoch + 1}/{epochs}",
            leave=False,
            disable=accelerator is not None and not accelerator.is_main_process,
        )
        for batch in iterator:
            loss = forward_pretrain_batch(model, batch, device=device)
            valid_tokens = count_valid_target_tokens(batch).to(device)
            total_loss_sum = total_loss_sum + (loss.detach() * valid_tokens)
            total_valid_tokens = total_valid_tokens + valid_tokens
    avg_eval_loss_tensor = reduce_token_weighted_loss(total_loss_sum, total_valid_tokens, accelerator=accelerator)
    avg_eval_loss = float(avg_eval_loss_tensor.detach().cpu().item())
    if accelerator is None or accelerator.is_main_process:
        _log_error_stats(logger, "VAL", val_loader.dataset.consume_error_stats())
    model.train()
    if getattr(model, "qformer_enabled", False):
        model.qformer.eval()
        model.mlp1.eval()
    return avg_eval_loss


def train_pretrain(model, tokenizer, train_loader, val_loader, config: Dict, output_dir: str, logger):
    epochs = int(config["training"]["num_epochs"])
    accum_steps = int(config["training"]["gradient_accumulation_steps"])
    max_grad_norm = float(config["training"]["max_grad_norm"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = _build_optimizer(model, config, logger)
    total_training_steps = max((len(train_loader) * epochs) // max(accum_steps, 1), 1)
    warmup_steps = resolve_warmup_steps(config, total_training_steps)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    logger.info("Trainable parameter summary:")
    for row in trainable_parameter_summary(model):
        logger.info(row)
    log_optimizer_param_group_health(model, optimizer, logger)

    return optimizer, lr_scheduler


def validate_pretrain_resume_checkpoint(model, checkpoint_dir, logger):
    bridge_metadata = read_qformer_bridge_metadata(checkpoint_dir)
    traj_metadata = read_trajectory_branch_metadata(checkpoint_dir)
    if not os.path.exists(os.path.join(checkpoint_dir, "qformer_bridge.safetensors")):
        raise FileNotFoundError(f"Pretrain resume checkpoint is missing qformer_bridge.safetensors: {checkpoint_dir}")
    if not os.path.exists(os.path.join(checkpoint_dir, "trajectory_branch.safetensors")):
        raise FileNotFoundError(f"Pretrain resume checkpoint is missing trajectory_branch.safetensors: {checkpoint_dir}")

    stages = {m.get("stage") for m in (bridge_metadata, traj_metadata) if m}
    if stages and stages != {"pretrain"}:
        raise ValueError(f"Resume checkpoint stage metadata mismatch: {sorted(stages)}")
    if not stages:
        logger.warning("Pretrain resume checkpoint has no explicit stage metadata; treating it as a legacy compatible checkpoint.")

    checkpoint_mode = traj_metadata.get("fusion_mode")
    current_mode = getattr(model, "trajectory_fusion_mode", None)
    if checkpoint_mode and current_mode and checkpoint_mode != current_mode:
        raise ValueError(
            f"Pretrain resume fusion mode mismatch: checkpoint={checkpoint_mode}, current={current_mode}"
        )


def run_pretrain_training(model, tokenizer, train_loader, val_loader, config: Dict, output_dir: str, logger, resume_dir=None, start_epoch=0, start_step=0):
    epochs = int(config["training"]["num_epochs"])
    accum_steps = int(config["training"]["gradient_accumulation_steps"])
    max_grad_norm = float(config["training"]["max_grad_norm"])
    metrics_path = os.path.join(output_dir, "metrics.json")
    use_accelerate = bool(config["training"].get("use_accelerate", False))
    accelerator = Accelerator(gradient_accumulation_steps=accum_steps) if use_accelerate else None
    is_main_process = accelerator is None or accelerator.is_main_process
    device = accelerator.device if accelerator is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patience = int(config["training"].get("early_stopping_patience", 0))
    min_delta = float(config["training"].get("early_stopping_min_delta", 0.0))
    burn_in_epochs = int(config["training"].get("early_stopping_burn_in_epochs", 0))
    restore_best_checkpoint = bool(config["training"].get("restore_best_checkpoint", True))
    best_dir = os.path.join(output_dir, "best")
    last_dir = os.path.join(output_dir, "last")
    early_stopping = EarlyStoppingState(patience=patience, min_delta=min_delta) if patience > 0 else None
    metrics = {"train_loss": [], "val_loss": [], "epoch_summary": []}
    if is_main_process:
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        logger.info(
            "Pretrain distributed runtime | accelerate=%s | world_size=%s | rank=%s | device=%s | accum_steps=%s",
            use_accelerate,
            accelerator.num_processes if accelerator is not None else 1,
            accelerator.process_index if accelerator is not None else 0,
            device,
            accum_steps,
        )
        logger.info(
            "Pretrain config runtime | mode=%s | batch_size=%s | global_batch=%s | grad_ckpt=%s | bf16=%s | quant=%s",
            config.get("trajectory", {}).get("fusion_mode", "unknown"),
            config["training"].get("batch_size", "unknown"),
            int(config["training"].get("batch_size", 1)) * accum_steps * (accelerator.num_processes if accelerator is not None else 1),
            config["training"].get("gradient_checkpointing", False),
            config["training"].get("bf16", False),
            config.get("model", {}).get("quantization", {}).get("enabled", False),
        )
        log_flash_attention_runtime(model, logger)
        _write_run_metadata(output_dir, _build_run_metadata(config, global_optimizer_step=0))

    optimizer, lr_scheduler = train_pretrain(model, tokenizer, train_loader, val_loader, config, output_dir, logger)
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
            logger.info(f"Resuming pretrain from {resume_dir} | Epoch: {start_epoch+1}, Step: {start_step}")
        validate_pretrain_resume_checkpoint(unwrapped_model, resume_dir, logger)
        load_qformer_bridge(unwrapped_model, resume_dir, strict=True)
        align_qformer_bridge_runtime(unwrapped_model)
        load_trajectory_branch(unwrapped_model, resume_dir, strict=True)
        align_qformer_bridge_runtime(unwrapped_model)

        opt_path = os.path.join(resume_dir, "optimizer.pt")
        sch_path = os.path.join(resume_dir, "scheduler.pt")
        if os.path.exists(opt_path) and os.path.exists(sch_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))
            lr_scheduler.load_state_dict(torch.load(sch_path, map_location="cpu"))
        elif is_main_process:
            logger.warning("No optimizer/scheduler states found in pretrain checkpoint. Starting with fresh optimizer states.")

        if early_stopping is not None:
            restored_early_stopping = _load_early_stopping_state(resume_dir)
            if restored_early_stopping is None:
                if is_main_process:
                    logger.warning(
                        "No %s found in pretrain checkpoint. Early stopping state will restart from scratch.",
                        EARLY_STOPPING_STATE_FILENAME,
                    )
            else:
                early_stopping.best_val_loss = restored_early_stopping.best_val_loss
                early_stopping.best_epoch = restored_early_stopping.best_epoch
                early_stopping.num_bad_epochs = restored_early_stopping.num_bad_epochs
                if is_main_process:
                    logger.info(
                        "Restored early stopping state | best_epoch=%s | best_val_loss=%s | bad_epochs=%s/%s",
                        early_stopping.best_epoch,
                        f"{early_stopping.best_val_loss:.4f}" if early_stopping.best_val_loss is not None else "None",
                        early_stopping.num_bad_epochs,
                        early_stopping.patience,
                    )

    global_optimizer_step = 0
    if resume_dir and os.path.exists(resume_dir):
        restored_training_state = read_training_state(resume_dir)
        if restored_training_state is not None:
            global_optimizer_step = int(restored_training_state.get("global_optimizer_step", 0))
            if is_main_process:
                logger.info("Restored global_optimizer_step=%s", global_optimizer_step)
    profile_steps = int(config["training"].get("profile_steps", 0))
    gradient_debug_steps = int(config["training"].get("gradient_debug_steps", 1))

    for epoch in range(start_epoch, epochs):
        model.train()
        if getattr(unwrapped_model, "qformer_enabled", False):
            unwrapped_model.qformer.eval()
            unwrapped_model.mlp1.eval()
        optimizer.zero_grad()
        train_loader.dataset.reset_error_stats()
        accumulated_loss_for_log = 0.0
        global_steps_this_epoch = 0

        batch_iterator = iter(train_loader)
        if epoch == start_epoch and start_step > 0:
            if is_main_process:
                logger.info(f"Skipping {start_step} batches to resume pretrain state...")
            for _ in tqdm(
                range(start_step),
                desc="Skipping to pretrain resume point",
                leave=False,
                disable=not is_main_process,
            ):
                next(batch_iterator)
            initial_step = start_step
        else:
            initial_step = 0

        progress_bar = tqdm(
            batch_iterator,
            desc=f"Pretrain epoch {epoch + 1}/{epochs}",
            total=len(train_loader),
            initial=initial_step,
            disable=not is_main_process,
        )
        for step, batch in enumerate(progress_bar, start=initial_step + 1):
            step_start = time.perf_counter()
            if accelerator is not None:
                with accelerator.accumulate(model):
                    loss = forward_pretrain_batch(model, batch, device=device)
                    accelerator.backward(loss)
                    accumulated_loss_for_log += float(loss.detach().cpu().item())
                    if accelerator.sync_gradients:
                        if torch.cuda.is_available() and global_optimizer_step < profile_steps:
                            torch.cuda.synchronize()
                        accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                        if is_main_process and global_optimizer_step < gradient_debug_steps:
                            log_gradient_health(unwrapped_model, logger)
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                        global_optimizer_step += 1
                        global_steps_this_epoch += 1
                        avg_loss = accumulated_loss_for_log / accum_steps
                        if is_main_process:
                            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
                            metrics["train_loss"].append(
                                {"epoch": epoch + 1, "step": global_steps_this_epoch, "loss": round(avg_loss, 6)}
                            )
                            if global_optimizer_step <= profile_steps:
                                if torch.cuda.is_available():
                                    torch.cuda.synchronize()
                                logger.info(
                                    "Profile step | global_step=%s | elapsed_sec=%.4f",
                                    global_optimizer_step,
                                    time.perf_counter() - step_start,
                                )
                        accumulated_loss_for_log = 0.0
            else:
                loss = forward_pretrain_batch(model, batch, device=device)
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")
                (loss / accum_steps).backward()
                accumulated_loss_for_log += float(loss.item())
                if step % accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    if global_optimizer_step < gradient_debug_steps:
                        log_gradient_health(model, logger)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    global_optimizer_step += 1
                    global_steps_this_epoch += 1
                    avg_loss = accumulated_loss_for_log / accum_steps
                    metrics["train_loss"].append({"epoch": epoch + 1, "step": global_steps_this_epoch, "loss": round(avg_loss, 6)})
                    accumulated_loss_for_log = 0.0

        if accelerator is not None:
            accelerator.wait_for_everyone()
        train_stats = train_loader.dataset.consume_error_stats()
        if is_main_process:
            _log_error_stats(logger, "TRAIN", train_stats)

        val_loss = eval_pretrain(model, val_loader, epoch, epochs, device, logger, accelerator=accelerator)
        if is_main_process:
            metrics["val_loss"].append({"epoch": epoch + 1, "loss": round(val_loss, 6)})

            epoch_losses = [item["loss"] for item in metrics["train_loss"] if item["epoch"] == epoch + 1]
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else float("nan")
            metrics["epoch_summary"].append(
                {"epoch": epoch + 1, "avg_train_loss": round(avg_epoch_loss, 6), "val_loss": round(val_loss, 6)}
            )
            logger.info(
                "Pretrain epoch %s summary | avg_train_loss=%.4f | val_loss=%.4f",
                epoch + 1,
                avg_epoch_loss,
                val_loss,
            )
        improved = False
        should_stop = False
        if early_stopping is not None:
            if epoch + 1 <= burn_in_epochs:
                improved = early_stopping.best_val_loss is None
                if improved:
                    early_stopping.best_val_loss = val_loss
                    early_stopping.best_epoch = epoch + 1
                    early_stopping.num_bad_epochs = 0
            else:
                improved, should_stop = early_stopping.update(epoch=epoch + 1, val_loss=val_loss)
            if is_main_process:
                logger.info(
                    "Early stopping status | improved=%s | best_epoch=%s | best_val_loss=%.4f | bad_epochs=%s/%s | burn_in_epochs=%s",
                    improved,
                    early_stopping.best_epoch,
                    early_stopping.best_val_loss if early_stopping.best_val_loss is not None else float("nan"),
                    early_stopping.num_bad_epochs,
                    early_stopping.patience,
                    burn_in_epochs,
                )

        if is_main_process:
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)

            save_model = accelerator.unwrap_model(model) if accelerator is not None else model
            if os.path.exists(last_dir):
                shutil.rmtree(last_dir)
            _save_pretrain_checkpoint(
                save_model,
                tokenizer,
                last_dir,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                run_metadata=_build_run_metadata(config, global_optimizer_step=global_optimizer_step),
            )
            _write_training_state(last_dir, next_epoch=epoch + 1, next_step=0, global_optimizer_step=global_optimizer_step)
            if early_stopping is not None:
                _write_early_stopping_state(last_dir, early_stopping)
            logger.info("Saved latest pretrain checkpoint to %s", last_dir)
            if improved:
                if os.path.exists(best_dir):
                    shutil.rmtree(best_dir)
                _save_pretrain_checkpoint(
                    save_model,
                    tokenizer,
                    best_dir,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    run_metadata=_build_run_metadata(config, global_optimizer_step=global_optimizer_step),
                )
                _write_training_state(best_dir, next_epoch=epoch + 1, next_step=0, global_optimizer_step=global_optimizer_step)
                if early_stopping is not None:
                    _write_early_stopping_state(best_dir, early_stopping)
                logger.info("Updated best pretrain checkpoint at %s", best_dir)
        if accelerator is not None:
            accelerator.wait_for_everyone()
        if should_stop:
            if is_main_process:
                logger.info("Early stopping triggered at epoch %s", epoch + 1)
            break

    if restore_best_checkpoint and os.path.exists(best_dir):
        if is_main_process:
            logger.info("Restoring best checkpoint from %s", best_dir)
        final_model = accelerator.unwrap_model(model) if accelerator is not None else model
        load_qformer_bridge(final_model, best_dir, strict=True)
        align_qformer_bridge_runtime(final_model)
        load_trajectory_branch(final_model, best_dir, strict=True)
        align_qformer_bridge_runtime(final_model)
    if accelerator is not None:
        accelerator.wait_for_everyone()
    return metrics


def main():
    set_seed(42)
    args = parse_args()
    config = load_config(args.config)
    output_dir = build_output_dir(config)
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        init_logger(output_dir)
        logger = get_logger()
    else:
        logger = SilentLogger()
    resume_dir, start_epoch, start_step = resolve_resume_config(args, logger)

    model, tokenizer = build_model_and_tokenizer(config, logger)
    train_dataset, val_dataset, test_dataset, split_stats, train_loader, val_loader = build_dataloaders(config, model, tokenizer)
    if rank == 0:
        logger.info("Pretrain split stats: %s", json.dumps(split_stats, ensure_ascii=False))
        _print_debug_samples(logger, train_dataset, "train")
        _print_debug_samples(logger, val_dataset, "val")

    logger.info(
        "Pretrain runtime check | qformer_enabled=%s | trajectory_enabled=%s | trajectory_mode=%s | num_image_token=%s | train_rows=%s | val_rows=%s | test_rows=%s",
        getattr(model, "qformer_enabled", False),
        getattr(model, "trajectory_enabled", False),
        getattr(model, "trajectory_fusion_mode", "disabled"),
        getattr(model, "num_image_token", "unknown"),
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
    )

    run_pretrain_training(
        model,
        tokenizer,
        train_loader,
        val_loader,
        config,
        output_dir,
        logger,
        resume_dir=resume_dir,
        start_epoch=start_epoch,
        start_step=start_step,
    )


if __name__ == "__main__":
    main()
