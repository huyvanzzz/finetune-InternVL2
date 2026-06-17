import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import datetime
import re
import random

import numpy as np
import torch
import yaml
from huggingface_hub import snapshot_download
from peft import PeftModel, prepare_model_for_kbit_training
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig, get_cosine_schedule_with_warmup

from logutil import get_logger, init_logger


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)

with open("internvl_config.yaml", "r") as f:
    config = yaml.safe_load(f)

base_out_dir = config["training"]["output_dir"]
output_dir = f'{base_out_dir}/{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}/'
os.makedirs(output_dir, exist_ok=True)
init_logger(output_dir)
logger = get_logger()

from wad_dataset import build_dataset
from model.conversation import get_conv_template
from qformer_bridge import (
    attach_qformer_bridge,
    load_qformer_bridge,
    qformer_enabled,
    save_qformer_bridge,
    trainable_parameter_summary,
)


IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
SYSTEM_MESSAGE = "You are a navigation assistant for visually impaired users."
DEFAULT_LORA_CHECKPOINT = "huyvanzzz/internvl2.5_config1"


def parse_args():
    parser = argparse.ArgumentParser(description="Train InternVL with optional checkpoint resume.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint dir to resume from. Omit this flag to train from the default LoRA/base setup.",
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


def maybe_pad(inner_lists, padding_value):
    tensor_list = [torch.tensor(inner_list, dtype=torch.long) for inner_list in inner_lists]
    return pad_sequence(tensor_list, batch_first=True, padding_value=padding_value)


class CollaterFn:
    def __init__(self, tokenizer, model) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.log_token_stats = False

    def __call__(self, batch):
        label_ids_batch = []
        input_ids_batch = []
        attention_mask_batch = []
        pixel_values_batch = []
        qformer_texts = []
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
            if self.log_token_stats:
                total_image_tokens_in_sample = total_tiles * self.model.num_image_token
                total_sequence_length = len(input_ids) + len(answer_ids) + 1
                logger.info(
                    "[INFO] Image token stats | frames=%s | tiles_per_frame=%s | query_tokens_per_tile=%s | total_image_tokens=%s",
                    len(pixel_values),
                    num_patches_list,
                    self.model.num_image_token,
                    total_image_tokens_in_sample,
                )
                logger.info(
                    "[INFO] Text tokens - input: %s, answer: %s, total: %s",
                    len(input_ids),
                    len(answer_ids),
                    total_sequence_length,
                )

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

        input_ids_tensor = maybe_pad(input_ids_batch, eot_token_id)
        label_ids_tensor = maybe_pad(label_ids_batch, -100)
        attention_mask_tensor = maybe_pad(attention_mask_batch, 0)
        pixel_values_tensor = torch.cat(pixel_values_batch)
        qformer_inputs = None
        if getattr(self.model, "qformer_enabled", False):
            qformer_inputs = self.model.encode_qformer_texts(qformer_texts)
        return input_ids_tensor, label_ids_tensor, attention_mask_tensor, pixel_values_tensor, qformer_inputs, samples_batch


def test_model(model, tokenizer, val_loader_with_shuffle, shuffle=False):
    model.eval()
    with torch.no_grad():
        total_test_batches = 0
        for batch in tqdm(val_loader_with_shuffle):
            _, _, _, _, _, samples = batch
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
                response = model.chat(tokenizer, pixel_values, question, generation_config)
                if getattr(model, "qformer_enabled", False):
                    model.clear_qformer_text()
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


def eval_model(model, val_loader, step, epoch, epochs):
    model.eval()
    with torch.no_grad():
        total_eval_loss = 0
        total_eval_batchs = 0
        for batch in tqdm(val_loader, desc=f"Validation after {step} batches training in epoch {epoch + 1}/{epochs}"):
            input_ids_batch, label_ids_batch, attention_mask_batch, pixel_values_batch, qformer_inputs, _ = batch
            input_ids_batch = input_ids_batch.cuda()
            label_ids_batch = label_ids_batch.cuda()
            attention_mask_batch = attention_mask_batch.cuda()
            pixel_values_batch = pixel_values_batch.to(torch.bfloat16).cuda()
            image_flags_batch = torch.ones((pixel_values_batch.shape[0], 1), dtype=torch.long).cuda()
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
            loss = outputs.loss
            total_eval_loss += loss.item()
            total_eval_batchs += 1
            if total_eval_batchs == 200:
                break
        logger.info(f"Validation loss after {step} batches training in epoch {epoch + 1}/{epochs}: {total_eval_loss / total_eval_batchs:.4f}")


def train_model(model, tokenizer, train_loader, val_loader, val_loader_with_shuffle, config, output_dir, resume_dir=None, start_epoch=0, start_step=0):
    epochs = config["training"]["num_epochs"]
    lr = float(config["training"]["learning_rate"])
    accum_steps = config["training"]["gradient_accumulation_steps"]
    weight_decay = float(config["training"]["weight_decay"])
    warmup_steps = config["training"]["warmup_steps"]
    max_grad_norm = float(config["training"]["max_grad_norm"])

    logger.info(f"Total params: {sum(p.numel() for p in model.parameters())}")
    logger.info(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    logger.info(f"Training config: LR={lr}, Accum_steps={accum_steps}, Weight_decay={weight_decay}")

    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, weight_decay=weight_decay)
    save_steps = config["training"].get("save_steps")

    total_training_steps = (len(train_loader) * epochs) // accum_steps
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    if resume_dir and os.path.exists(resume_dir):
        logger.info(f"Resuming training from {resume_dir} | Epoch: {start_epoch+1}, Step: {start_step}")
        if getattr(model, "qformer_enabled", False):
            load_qformer_bridge(model, resume_dir, strict=True)
            logger.info("Loaded Q-Former bridge states successfully!")

        opt_path = os.path.join(resume_dir, "optimizer.pt")
        sch_path = os.path.join(resume_dir, "scheduler.pt")

        if os.path.exists(opt_path) and os.path.exists(sch_path):
            optimizer.load_state_dict(torch.load(opt_path))
            lr_scheduler.load_state_dict(torch.load(sch_path))
            logger.info("Loaded Optimizer and Scheduler states successfully!")
        else:
            logger.warning("No Optimizer/Scheduler states found in checkpoint. Starting with fresh states.")

    for epoch in range(start_epoch, epochs):
        model.train()
        if getattr(model, "qformer_enabled", False):
            model.qformer.eval()
            model.mlp1.eval()
        optimizer.zero_grad()

        accumulated_loss_for_log = 0.0
        set_seed(42 + epoch)
        batch_iterator = iter(train_loader)
        train_loader.collate_fn.log_token_stats = False

        if epoch == start_epoch and start_step > 0:
            logger.info(f" Skipping {start_step} batches to resume state...")
            for _ in tqdm(range(start_step), desc="Skipping to resume point", leave=False):
                next(batch_iterator)
            i = start_step
        else:
            i = 0
            
        train_loader.collate_fn.log_token_stats = False
        progress_bar = tqdm(batch_iterator, desc=f"Training Epoch {epoch + 1}/{epochs}", total=len(train_loader), initial=i)
        for batch in progress_bar:
            i += 1
            input_ids_batch, label_ids_batch, attention_mask_batch, pixel_values_batch, qformer_inputs, _ = batch

            input_ids_batch = input_ids_batch.cuda()
            label_ids_batch = label_ids_batch.cuda()
            attention_mask_batch = attention_mask_batch.cuda()
            pixel_values_batch = pixel_values_batch.to(torch.bfloat16).cuda()
            image_flags_batch = torch.ones((pixel_values_batch.shape[0], 1), dtype=torch.long).cuda()
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

            loss = outputs.loss / accum_steps
            progress_bar.set_postfix(loss=f"{outputs.loss.item():.4f}")
            loss.backward()

            accumulated_loss_for_log += outputs.loss.item()

            if i % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                avg_loss = accumulated_loss_for_log / accum_steps
                if (i // accum_steps) % 50 == 0:
                    logger.info(f"Step {i//accum_steps} | Avg Loss: {avg_loss:.4f}")
                accumulated_loss_for_log = 0.0

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if save_steps and i % save_steps == 0:
                step_save_dir = f"{output_dir}/epoch_{epoch+1}_step_{i}/"
                os.makedirs(step_save_dir, exist_ok=True)
                logger.info(f"Saving model, tokenizer, opt, scheduler at step {i} to {step_save_dir}")

                model.language_model.save_pretrained(step_save_dir)
                save_qformer_bridge(model, step_save_dir)
                tokenizer.save_pretrained(step_save_dir)
                torch.save(optimizer.state_dict(), os.path.join(step_save_dir, "optimizer.pt"))
                torch.save(lr_scheduler.state_dict(), os.path.join(step_save_dir, "scheduler.pt"))

        epoch_save_dir = f"{output_dir}/epoch_{epoch+1}/"
        os.makedirs(epoch_save_dir, exist_ok=True)
        logger.info(f"Saving model and tokenizer for epoch {epoch+1} to {epoch_save_dir}")
        model.language_model.save_pretrained(epoch_save_dir)
        save_qformer_bridge(model, epoch_save_dir)
        tokenizer.save_pretrained(epoch_save_dir)
        torch.save(optimizer.state_dict(), os.path.join(epoch_save_dir, "optimizer.pt"))
        torch.save(lr_scheduler.state_dict(), os.path.join(epoch_save_dir, "scheduler.pt"))


if __name__ == "__main__":
    args = parse_args()
    resume_dir, start_epoch, start_step = resolve_resume_config(args, config)
    model_name_or_path = config["model"]["name"]
    batch_size = config["training"]["batch_size"]

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=config["model"]["quantization"]["enabled"],
        bnb_4bit_compute_dtype=torch.bfloat16 if config["model"]["quantization"]["compute_dtype"] == "bfloat16" else torch.float16,
        bnb_4bit_use_double_quant=config["model"]["quantization"]["double_quant"],
        bnb_4bit_quant_type=config["model"]["quantization"]["type"],
    )

    logger.info(f"Loading model {model_name_or_path} in 4-bit...")
    model = AutoModel.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=config["model"]["trust_remote_code"],
    )

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

    logger.info("Applying LoRA...")
    model.language_model = prepare_model_for_kbit_training(model.language_model)

    if hasattr(model.language_model, "get_input_embeddings"):
        model.language_model.get_input_embeddings().to(torch.bfloat16)

    lora_checkpoint = resume_dir or DEFAULT_LORA_CHECKPOINT
    logger.info(f"Loading LoRA adapter from: {lora_checkpoint}")
    model.language_model = PeftModel.from_pretrained(
        model.language_model,
        lora_checkpoint,
        is_trainable=True,
    )

    model.language_model.print_trainable_parameters()
    model.train()

    logger.info("Building dataset...")
    train_dataset, val_dataset = build_dataset(config)

    collate_fn_wrapper = CollaterFn(tokenizer, model)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, collate_fn=collate_fn_wrapper, shuffle=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, collate_fn=collate_fn_wrapper, shuffle=False
    )

    val_loader_with_shuffle = DataLoader(
        val_dataset, batch_size=1, collate_fn=collate_fn_wrapper, shuffle=True
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
        resume_dir=resume_dir,
        start_epoch=start_epoch,
        start_step=start_step,
    )
