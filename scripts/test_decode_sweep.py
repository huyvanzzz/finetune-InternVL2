import argparse
import json
import os
import pickle
import sys
from collections import defaultdict
from typing import Dict, List

import evaluate
import torch
import yaml
from datasets import load_dataset
from huggingface_hub import snapshot_download
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

sys.path.append(".")

from model.conversation import get_conv_template
from preprocessing import get_response_format
from qformer_bridge import attach_qformer_bridge, load_qformer_bridge, qformer_enabled
from trajectory_branch import build_trajectory_source_from_config, load_trajectory_branch
from wad_dataset import WADDatasetForInternVL


IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
SYSTEM_MESSAGE = "You are a navigation assistant for visually impaired users."


DECODE_PRESETS = [
    {
        "name": "greedy_rp10",
        "generation_config": {
            "max_new_tokens": 512,
            "num_beams": 1,
            "do_sample": False,
            "repetition_penalty": 1.0,
        },
    },
    {
        "name": "beam5_rp12",
        "generation_config": {
            "max_new_tokens": 512,
            "num_beams": 5,
            "do_sample": False,
            "repetition_penalty": 1.2,
            "early_stopping": True,
        },
    },
    {
        "name": "beam3_rp10_len08",
        "generation_config": {
            "max_new_tokens": 512,
            "num_beams": 3,
            "do_sample": False,
            "repetition_penalty": 1.0,
            "length_penalty": 0.8,
            "early_stopping": True,
        },
    },
    {
        "name": "beam5_rp10_len08",
        "generation_config": {
            "max_new_tokens": 512,
            "num_beams": 5,
            "do_sample": False,
            "repetition_penalty": 1.0,
            "length_penalty": 0.8,
            "early_stopping": True,
        },
    },
]


def replace_image_placeholders(query: str, num_patches_list, num_image_token: int) -> str:
    if query.count("<image>") != len(num_patches_list):
        raise ValueError(
            f"Image placeholder count mismatch: placeholders={query.count('<image>')} "
            f"frames={len(num_patches_list)} tiles_per_frame={list(num_patches_list)}"
        )
    for num_patches in num_patches_list:
        image_tokens = "<img>" + IMG_CONTEXT_TOKEN * num_image_token * int(num_patches) + "</img>"
        query = query.replace("<image>", image_tokens, 1)
    if "<image>" in query:
        raise ValueError("Unreplaced <image> placeholder remains after image token replacement.")
    return query


def clean_text(text: str) -> str:
    text = str(text).strip()
    if "<answer>" in text:
        text = text.split("<answer>")[-1]
    if "</answer>" in text:
        text = text.split("</answer>")[0]
    return text.strip()


def extract_field(text: str, key: str) -> str:
    if key in (None, "", "raw_text"):
        return clean_text(text)
    try:
        data = json.loads(clean_text(text))
        return str(data.get(key, "")).strip()
    except json.JSONDecodeError:
        return ""


def compute_rouge_only(rouge_metric, predictions: List[str], references: List[str], target_field: str) -> Dict[str, float]:
    pred_texts = [extract_field(pred, target_field) for pred in predictions]
    ref_texts = [extract_field(ref, target_field) for ref in references]
    scores = rouge_metric.compute(predictions=pred_texts, references=ref_texts, use_stemmer=True)
    return {
        "ROUGE-1": scores["rouge1"] * 100,
        "ROUGE-2": scores["rouge2"] * 100,
        "ROUGE-L": scores["rougeL"] * 100,
    }


def compute_sample_rouge(rouge_metric, prediction: str, reference: str, target_field: str) -> Dict[str, float]:
    return compute_rouge_only(rouge_metric, [prediction], [reference], target_field)


def align_language_model_devices(model):
    target_device = torch.device("cuda:0")
    input_embeddings = model.language_model.get_input_embeddings()
    input_embeddings.to(device=target_device)
    output_embeddings = model.language_model.get_output_embeddings()
    if output_embeddings is not None:
        output_embeddings.to(device=target_device)
    embedding_device = next(input_embeddings.parameters()).device
    print(f"[DEVICE CHECK] input_embeddings device: {embedding_device}", flush=True)


def run_model_batch_chat(model, tokenizer, batch, generation_config, device):
    if not batch:
        return []

    pixel_values_chunks = []
    questions = []
    batch_num_patches_lists = []
    qformer_texts = []
    trajectory_label_ids = []
    trajectory_direction_ids = []
    trajectory_numeric_feats = []
    trajectory_object_mask = []
    pixel_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    for sample in batch:
        frame_num_patches = [int(torch.as_tensor(p).shape[0]) for p in sample["pixel_values"]]
        pixel_values = torch.cat([torch.as_tensor(p) for p in sample["pixel_values"]], dim=0)
        pixel_values = pixel_values.to(dtype=pixel_dtype, device=device)
        pixel_values_chunks.append(pixel_values)
        num_patches = sum(frame_num_patches)
        batch_num_patches_lists.append(frame_num_patches)
        question = str(sample["question"])
        questions.append(question)

        if getattr(model, "qformer_enabled", False):
            qformer_text = sample.get("qformer_text", question.replace("<image>", "").strip())
            qformer_texts.extend([qformer_text] * num_patches)

        if getattr(model, "trajectory_enabled", False):
            trajectory_label_ids.append(sample["trajectory_label_ids"].unsqueeze(0).repeat(num_patches, 1))
            trajectory_direction_ids.append(sample["trajectory_direction_ids"].unsqueeze(0).repeat(num_patches, 1))
            trajectory_numeric_feats.append(sample["trajectory_numeric_feats"].unsqueeze(0).repeat(num_patches, 1, 1))
            trajectory_object_mask.append(sample["trajectory_object_mask"].unsqueeze(0).repeat(num_patches, 1))

    pixel_values_batch = torch.cat(pixel_values_chunks, dim=0)

    if getattr(model, "qformer_enabled", False):
        q_ids, q_mask = model.encode_qformer_texts(qformer_texts, device=device)
        model.set_qformer_text(q_ids, q_mask)
    if getattr(model, "trajectory_enabled", False):
        model.set_trajectory_inputs(
            torch.cat(trajectory_label_ids, dim=0).to(device),
            torch.cat(trajectory_direction_ids, dim=0).to(device),
            torch.cat(trajectory_numeric_feats, dim=0).to(device),
            torch.cat(trajectory_object_mask, dim=0).to(device),
        )

    queries = []
    template = None
    for question, frame_num_patches in zip(questions, batch_num_patches_lists):
        template = get_conv_template(model.template)
        template.system_message = model.system_message
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        query = replace_image_placeholders(query, frame_num_patches, model.num_image_token)
        queries.append(query)

    tokenizer.padding_side = "left"
    model_inputs = tokenizer(queries, return_tensors="pt", padding=True)
    embedding_device = model.language_model.get_input_embeddings().weight.device
    input_ids = model_inputs["input_ids"].to(embedding_device)
    attention_mask = model_inputs["attention_mask"].to(embedding_device)
    eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)
    generation_config = dict(generation_config)
    generation_config["eos_token_id"] = eos_token_id
    generation_config["pad_token_id"] = eos_token_id
    generation_output = model.generate(
        pixel_values=pixel_values_batch,
        input_ids=input_ids,
        attention_mask=attention_mask,
        **generation_config,
    )
    responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
    responses = [response.split(template.sep)[0].strip() for response in responses]

    if getattr(model, "qformer_enabled", False):
        model.clear_qformer_text()
    if getattr(model, "trajectory_enabled", False):
        model.clear_trajectory_inputs()

    return responses


class TestCollaterFn:
    def __call__(self, batch):
        return batch


def resolve_checkpoint_path(checkpoint):
    if not checkpoint:
        return None
    if os.path.exists(checkpoint):
        print(f"Using local checkpoint path: {checkpoint}")
        return checkpoint
    print(f"Checkpoint is not a local path. Downloading from Hugging Face: {checkpoint}")
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
            "tokenizer*",
            "special_tokens_map.json",
            "added_tokens.json",
        ],
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run four non-official decode presets for 1-frame concat analysis.")
    parser.add_argument("--config", type=str, default="internvl_config_traj_concat_bestshot_bf16_2gpu.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="test_alter", choices=["test_QA", "test_alter", "val"])
    parser.add_argument("--output_dir", type=str, default="results/decode_sweep")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--print_samples", type=int, default=0)
    return parser.parse_args()


def prepare_auxiliary_data(config):
    print("--- Loading auxiliary data ---")
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


def load_model_and_tokenizer(config, checkpoint):
    model_name_or_path = config["model"]["name"]
    quantization_enabled = bool(config["model"]["quantization"]["enabled"])
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": {"": 0},
        "low_cpu_mem_usage": True,
        "trust_remote_code": config["model"]["trust_remote_code"],
    }
    if "attn_implementation" in config["model"]:
        model_kwargs["attn_implementation"] = config["model"]["attn_implementation"]
    if quantization_enabled:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=(
                torch.bfloat16
                if config["model"]["quantization"]["compute_dtype"] == "bfloat16"
                else torch.float16
            ),
            bnb_4bit_use_double_quant=config["model"]["quantization"]["double_quant"],
            bnb_4bit_quant_type=config["model"]["quantization"]["type"],
        )

    model = AutoModel.from_pretrained(model_name_or_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, use_fast=False)
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.system_message = SYSTEM_MESSAGE
    if qformer_enabled(config):
        attach_qformer_bridge(model, config)
    model.eval()

    if checkpoint:
        print(f"Loading checkpoint: {checkpoint}")
        if qformer_enabled(config):
            load_qformer_bridge(model, checkpoint, strict=True)
            print("Q-Former bridge loaded.")
        if getattr(model, "trajectory_enabled", False):
            load_trajectory_branch(model, checkpoint, strict=True)
            print("Trajectory branch loaded.")
        model.language_model = PeftModel.from_pretrained(
            model.language_model,
            checkpoint,
            is_trainable=False,
            device_map={"": 0},
        )
        print("LoRA adapter loaded.")
    else:
        print("No checkpoint provided. Evaluating base model.")

    if not quantization_enabled:
        model = model.cuda()
    align_language_model_devices(model)
    return model, tokenizer


def build_loader(config, split, batch_size, tokenizer, model, response_format):
    frame_index, bbox_by_folder, trajectory_source = prepare_auxiliary_data(config)
    if split == "test_alter":
        data_file = "test_alter.json"
    elif split == "test_QA":
        data_file = "test_QA.json"
    else:
        data_file = "train.json"

    print(f"Loading metadata from {data_file}...")
    dataset_dict = load_dataset(config["data"]["name"], data_files={"test": data_file})
    test_dataset = WADDatasetForInternVL(
        metadata_dataset=dataset_dict,
        frame_index=frame_index,
        bbox_by_folder=bbox_by_folder,
        trajectory_source=trajectory_source,
        split="test",
        response_format=response_format,
        num_frames=int(config["data"].get("num_frames", 1)),
        frame_indices=config["data"].get("frame_indices", [4, 6, 8]),
    )
    print(f"Loaded split={split} | samples={len(test_dataset)} | batch_size={batch_size}")
    return DataLoader(test_dataset, batch_size=batch_size, collate_fn=TestCollaterFn(), shuffle=False)


def run_decode_preset(model, tokenizer, loader, preset, response_format, split, checkpoint, output_dir, print_samples):
    target_field = "raw_text" if response_format == "direct_text" else "instruction"
    rouge_metric = evaluate.load("rouge")
    predictions = []
    references = []
    detailed_results = []
    sample_counter = 0
    generation_config = preset["generation_config"]

    print(f"\n=== Running decode preset: {preset['name']} ===")
    print(json.dumps(generation_config, ensure_ascii=False))
    inference_device = next(model.language_model.get_input_embeddings().parameters()).device
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Testing {preset['name']}"):
            responses = run_model_batch_chat(model, tokenizer, batch, generation_config, inference_device)
            for sample, response in zip(batch, responses):
                question = str(sample["question"])
                ground_truth = str(sample["answer"])
                predictions.append(response)
                references.append(ground_truth)
                sample_metrics = compute_sample_rouge(rouge_metric, response, ground_truth, target_field)
                if sample_counter < print_samples:
                    print(f"\n--- Sample {sample_counter + 1} | {preset['name']} ---")
                    print(f"Pred: {response}")
                    print(f"GT:   {ground_truth}")
                    print(f"ROUGE-L: {sample_metrics['ROUGE-L']:.2f}")
                detailed_results.append(
                    {
                        "id": sample_counter,
                        "question": question,
                        "prediction": response,
                        "ground_truth": ground_truth,
                        "sample_metrics": sample_metrics,
                    }
                )
                sample_counter += 1

    metrics = compute_rouge_only(rouge_metric, predictions, references, target_field)
    print(f"\nResults for {preset['name']}:")
    print(f"  ROUGE-1: {metrics['ROUGE-1']:.2f}")
    print(f"  ROUGE-2: {metrics['ROUGE-2']:.2f}")
    print(f"  ROUGE-L: {metrics['ROUGE-L']:.2f}")

    output = {
        "checkpoint": checkpoint if checkpoint else "Base Model",
        "split": split,
        "decode_preset": preset["name"],
        "generation_config": generation_config,
        "metric_target_field": target_field,
        "metrics": metrics,
        "samples": detailed_results,
    }
    output_path = os.path.join(output_dir, f"{preset['name']}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)
    print(f"Saved: {output_path}")
    return {
        "decode_preset": preset["name"],
        "generation_config": generation_config,
        "metrics": metrics,
        "output_file": output_path,
    }


def main():
    args = parse_args()
    args.checkpoint = resolve_checkpoint_path(args.checkpoint)
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    num_frames = int(config["data"].get("num_frames", 1))
    if num_frames != 1:
        raise ValueError(
            f"Decode sweep v1 is only for 1-frame config. Got num_frames={num_frames} from {args.config}."
        )

    response_format = get_response_format(config)
    os.makedirs(args.output_dir, exist_ok=True)
    model, tokenizer = load_model_and_tokenizer(config, args.checkpoint)
    loader = build_loader(config, args.split, args.batch_size, tokenizer, model, response_format)

    summary = {
        "checkpoint": args.checkpoint if args.checkpoint else "Base Model",
        "split": args.split,
        "config": args.config,
        "batch_size": args.batch_size,
        "num_presets": len(DECODE_PRESETS),
        "results": [],
    }
    for preset in DECODE_PRESETS:
        summary["results"].append(
            run_decode_preset(
                model=model,
                tokenizer=tokenizer,
                loader=loader,
                preset=preset,
                response_format=response_format,
                split=args.split,
                checkpoint=args.checkpoint,
                output_dir=args.output_dir,
                print_samples=args.print_samples,
            )
        )

    summary_path = os.path.join(args.output_dir, "decode_sweep_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
