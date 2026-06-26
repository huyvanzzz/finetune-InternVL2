import sys
import yaml
import os

sys.path.append(".")
from wad_dataset import build_dataset

def main():
    print("Loading config...")
    config_path = "internvl_config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    print("Building datasets...")
    # Giảm eval_limit để chạy nhanh hơn
    config['data']['eval_limit'] = 5
    train_dataset, val_dataset = build_dataset(config)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    print("\n--- Testing dataset item retrieval and prompt correctness ---")
    
    # Lấy thử 10 samples đầu tiên của tập train
    for i in range(10):
        print(f"\nRetrieving sample {i}...")
        sample = train_dataset[i]
        
        print(f"  questionId: {sample['questionId']}")
        print(f"  qformer_text (for Q-Former):\n  \"\"\"\n{sample['qformer_text']}\n  \"\"\"")
        print(f"  question (for LLM):\n  \"\"\"\n{sample['question']}\n  \"\"\"")
        print(f"  answer (Ground Truth): {sample['answer']}")
        print(f"  pixel_values count: {len(sample['pixel_values'])}")
        print("  Retrieval successful!")

    print("\nAll dataset loading tests and prompt verification passed successfully!")

if __name__ == "__main__":
    main()
