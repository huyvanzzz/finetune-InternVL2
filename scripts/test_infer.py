import os
import yaml
import json
import torch
import argparse
import pickle
from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import DataLoader
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset

# Import class Metrics từ file metrics.py
from .metrics import VLMMetrics

# Import các thành phần data từ project của bạn
from wad_dataset import WADDatasetForInternVL

IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'

class TestCollaterFn:
    def __init__(self, tokenizer, model) -> None:
        self.tokenizer = tokenizer
        self.model = model
    
    def __call__(self, batch):
        return batch

def parse_args():
    parser = argparse.ArgumentParser(description="Test InternVL VLM Model")
    parser.add_argument("--config", type=str, default="internvl_config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Hugging Face Repo ID")
    parser.add_argument("--split", type=str, default="test_QA", choices=["test_QA", "test_alter", "val"])
    parser.add_argument("--output_file", type=str, default="results/eval_results.json")
    parser.add_argument("--print_samples", type=int, default=5)
    return parser.parse_args()

def prepare_auxiliary_data(config):
    """Hàm phụ trợ load frame index và bbox cho tập test"""
    print("--- Loading Auxiliary Data for Testing ---")
    
    index_file = "./wad_dataset/frame_index.pkl"
    if os.path.exists(index_file):
        with open(index_file, 'rb') as f:
            frame_index = pickle.load(f)
    else:
        raise FileNotFoundError(f"Frame index not found at {index_file}.")

    bbox_file = "all_bboxes.jsonl"
    if os.path.exists(bbox_file):
        bbox_dataset = load_dataset("json", data_files=bbox_file, split="train")
    else:
        bbox_dataset = load_dataset(config['data']['name'], data_files="all_bboxes.jsonl", split="train")

    bbox_by_folder = defaultdict(lambda: defaultdict(list))
    for bbox_entry in bbox_dataset:
        folder_id = bbox_entry['folder_id']
        frame_id = bbox_entry['frame_id']
        bbox_by_folder[folder_id][frame_id].append({
            'label': bbox_entry['label'],
            'confidence': bbox_entry['probs'],
            'bbox': bbox_entry['boxs']
        })
    return frame_index, bbox_by_folder

def main():
    args = parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # 1. Load Base Model & Tokenizer
    model_name_or_path = config['model']['name']
    print(f"Loading Base Model: {model_name_or_path}...")
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=config['model']['quantization']['enabled'],
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModel.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, use_fast=False)
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.eval()

    # 2. Load Checkpoint LoRA
    if args.checkpoint:
        print(f"Loading LoRA weights from: {args.checkpoint}")
        model.language_model = PeftModel.from_pretrained(
            model.language_model, 
            args.checkpoint, 
            is_trainable=False,
            token=True
        )
        print("✓ LoRA Adapter loaded successfully.")
    else:
        print("No checkpoint provided. Evaluating Zero-shot (Base Model).")

    # ==========================================
    # 3. CHUẨN BỊ TẬP TEST CHÍNH XÁC THEO ARGUMENTS
    # ==========================================
    print(f"Building dataset for split: {args.split}...")
    
    frame_index, bbox_by_folder = prepare_auxiliary_data(config)
    
    if args.split == "test_alter":
        data_file = "test_alter.json" 
    elif args.split == "test_QA":
        data_file = "test_QA.json"
    else:
        data_file = "val.json" # Fallback nếu gọi split khác
        
    print(f"Loading metadata from {data_file}...")
    dataset_dict = load_dataset(
        "json", # Dùng "json" nếu load file local trực tiếp
        data_files={"test": data_file}
    )

    image_size = tuple(config['model']['vision']['image_size']) if 'image_size' in config['model']['vision'] else (448, 448)

    test_dataset = WADDatasetForInternVL(
        metadata_dataset=dataset_dict,
        frame_index=frame_index,
        bbox_by_folder=bbox_by_folder,
        processor=None, # Tùy vào WADDatasetForInternVL của bạn có yêu cầu processor không, InternVL thường build chung trong model
        tokenizer=tokenizer,
        split='test',
        num_frames=config['data'].get('num_frames', 1),
        image_size=image_size
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1, 
        collate_fn=TestCollaterFn(tokenizer, model), 
        shuffle=False
    )

    # 4. Evaluation Loop
    predictions, references, detailed_results = [], [], []

    print("\n" + "="*50)
    print(f" BẮT ĐẦU CHẠY ĐÁNH GIÁ TRÊN TẬP: {args.split} (Tổng: {len(test_dataset)} samples)")
    print("="*50)

    evaluator = VLMMetrics()

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Testing")):
            sample = batch[0]
            pixel_values = sample['pixel_values'].to(torch.bfloat16).cuda()
            question = str(sample['question'])
            ground_truth = str(sample['answer'])
            
            generation_config = dict(max_new_tokens=512, do_sample=False)
            
            response = model.chat(tokenizer, pixel_values, question, generation_config)
            
            predictions.append(response)
            references.append(ground_truth)
            
            if i < args.print_samples:
                print(f"\n--- Sample {i+1} ---")
                print(f"Q: {question}")
                print(f"Pred: {response}")
                print(f"GT:   {ground_truth}")
            
            detailed_results.append({
                "id": i,
                "question": question,
                "prediction": response,
                "ground_truth": ground_truth
            })

    # 5. Compute Metrics
    print("\nComputing Metrics (ROUGE, TF-IDF) on 'instruction' field...")
    metrics = evaluator.compute(predictions, references, target_field="instruction")

    print("\n" + "="*50)
    print("🏆 KẾT QUẢ ĐÁNH GIÁ:")
    for k, v in metrics.items():
        print(f"  - {k}: {v:.2f}")
    print("="*50)

    # 6. Save File
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    final_output = {
        "checkpoint": args.checkpoint if args.checkpoint else "Base Model",
        "split": args.split,
        "metrics": metrics,
        "samples": detailed_results
    }
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=4)
    print(f"\n✓ Đã lưu chi tiết kết quả tại: {args.output_file}")

if __name__ == "__main__":
    main()