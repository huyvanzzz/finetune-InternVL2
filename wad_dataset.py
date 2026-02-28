import tarfile
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError
import io
from typing import List, Dict
import random
from sklearn.model_selection import train_test_split

from preprocessing import POLMData, map_metadata_to_ground_truth
# BẮT BUỘC: Import hàm xử lý ảnh từ file data.py của tác giả zhangfaen
from data import process_image 

class WADDatasetForInternVL(Dataset):
    def __init__(
        self,
        metadata_dataset,
        frame_index: dict,
        bbox_by_folder: dict,
        split: str = 'train',
    ):
        self.metadata = metadata_dataset[split]
        self.frame_index = frame_index
        self.bbox_by_folder = bbox_by_folder
        self.split = split

    def __len__(self):
        return len(self.metadata)

    # ... (Giữ nguyên các hàm _load_frames, _load_bboxes, _select_frames_safe như cũ của bạn) ...
    def _load_frames(self, frame_path: str, frame_ids: List[int]) -> List[Image.Image]:
        shard_to_frames = {}
        for frame_id in frame_ids:
            if frame_id not in self.frame_index[frame_path]:
                raise ValueError(f"Frame {frame_id} not in index")
            frame_info = self.frame_index[frame_path][frame_id]
            shard_path = frame_info['shard']
            if shard_path not in shard_to_frames:
                shard_to_frames[shard_path] = []
            shard_to_frames[shard_path].append((frame_id, frame_info['tar_path']))
        
        frames_dict = {}
        for shard_path, frame_list in shard_to_frames.items():
            with tarfile.open(shard_path, 'r') as tar:
                for frame_id, tar_path in frame_list:
                    member = tar.getmember(tar_path)
                    file_obj = tar.extractfile(member)
                    img = Image.open(io.BytesIO(file_obj.read())).convert('RGB')
                    frames_dict[frame_id] = img
        return [frames_dict[fid] for fid in frame_ids]

    def _load_bboxes(self, frame_path: str, frame_ids: List[int]) -> List[POLMData]:
        polm_list = []
        if frame_path not in self.bbox_by_folder:
            return polm_list
        for frame_id in frame_ids:
            if frame_id in self.bbox_by_folder[frame_path]:
                bboxes = self.bbox_by_folder[frame_path][frame_id]
                for bbox in bboxes:
                    polm = POLMData(object_type=bbox['label'], bbox=bbox['bbox'], confidence=bbox['confidence'])
                    polm_list.append(polm)
        polm_list = [p for p in polm_list if p.confidence >= 0.6]
        polm_list.sort(key=lambda x: x.confidence, reverse=True)
        return polm_list[:50]

    def _select_frames_safe(self, frame_path: str, num_frames: int = 1) -> List[int]:
        available_frames = sorted(self.frame_index[frame_path].keys())
        return [available_frames[-1]]

    # ĐÂY LÀ PHẦN THAY ĐỔI QUAN TRỌNG NHẤT
    def __getitem__(self, idx):
        try:
            sample = self.metadata[idx]
            frame_path = sample['frame_path']
            
            # 1. Load Ảnh
            frame_ids = self._select_frames_safe(frame_path, num_frames=1)
            frames = self._load_frames(frame_path, frame_ids)
            image = frames[0] # Lấy 1 ảnh PIL
            
            # 2. Xử lý ảnh bằng hàm của tác giả zhangfaen (Tự cắt tile, tự tạo tensor)
            pixel_values = process_image(image) 
            
            # 3. Tạo Text Prompt
            polm_list = self._load_bboxes(frame_path, frame_ids)
            polm_text = "\n".join([f"- {polm.to_text()}" for polm in polm_list])
            
            # Bắt buộc nối thêm <image>\n vào đầu để CollaterFn của tác giả nhận diện
            question = f"<image>\nYou are a navigation assistant for blind people.\nDetected objects:\n{polm_text}\nAnalyze: location, weather, traffic, scene → then give instruction."
            if sample.get('QA') and sample['QA'].get('Q'):
                question += f"\n\nQuestion: {sample['QA']['Q']}"
                
            # 4. Tạo Answer
            ground_truth_dict = map_metadata_to_ground_truth(sample)
            answer = ground_truth_dict.to_json()

            # 5. TRẢ VỀ DICT THEO ĐÚNG FORMAT CỦA TÁC GIẢ ZHANGFAEN
            return {
                'question': question, 
                'answer': answer, 
                'pixel_values': pixel_values, 
                'questionId': str(idx),
                'image': image
            }

        except Exception as e:
            new_idx = random.randint(0, len(self) - 1)
            return self.__getitem__(new_idx)

def build_dataset(config: Dict):
    """Build train/eval datasets from config"""
    
    from datasets import load_dataset
    from collections import defaultdict
    import pickle
    import os
    
    # Load metadata
    print("Loading metadata...")
    metadata = load_dataset(
        config['data']['name'],
        data_files={
            "train": "train.json",
            "test": "test_alter.json"
        }
    )
    
    # Load bboxes
    print("Loading bboxes...")
    bbox_dataset = load_dataset(
        config['data']['name'],
        data_files="all_bboxes.jsonl",
        split="train"
    )
    
    bbox_by_folder = defaultdict(lambda: defaultdict(list))
    for bbox_entry in bbox_dataset:
        folder_id = bbox_entry['folder_id']
        frame_id = bbox_entry['frame_id']
        
        bbox_by_folder[folder_id][frame_id].append({
            'label': bbox_entry['label'],
            'confidence': bbox_entry['probs'],
            'bbox': bbox_entry['boxs']
        })
    
    # Load frame index
    print("Loading frame index...")
    index_file = "./wad_dataset/frame_index.pkl"
    
    if os.path.exists(index_file):
        with open(index_file, 'rb') as f:
            frame_index = pickle.load(f)
    else:
        raise FileNotFoundError(f"Frame index not found at {index_file}. Run build_frame_index.py first.")
    
    architecture = config['model']['architecture']
    
    # Determine image size based on architecture
    if architecture == 'qwen':
        image_size = None  # Dynamic resolution
        print(f"✓ Using dynamic resolution for Qwen")
    elif architecture == 'internvl':
        image_size = (448, 448)  # Fixed tile size
        print(f"✓ Using fixed tile size {image_size} for InternVL")
    else:
        image_size = tuple(config['model']['vision']['image_size'])
        print(f"✓ Using image size {image_size} for {architecture}")
    
    # Create datasets
    train_dataset = WADDatasetForInternVL(
        metadata_dataset=metadata,
        frame_index=frame_index,
        bbox_by_folder=bbox_by_folder,
        split='train',
    )
    
    # Train/val split
    train_size = config['data']['train_split']
    indices = list(range(len(train_dataset)))
    
    train_indices, val_indices = train_test_split(
        indices,
        train_size=train_size,
        random_state=config['data']['seed']
    )
    
    from torch.utils.data import Subset
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)
    
    print(f"✓ Train: {len(train_subset)}, Val: {len(val_subset)}")
    
    # Limit eval dataset size
    eval_limit = config['data'].get('eval_limit', 200)
    
    if len(val_subset) > eval_limit:
        print(f"  Limiting eval dataset: {len(val_subset)} → {eval_limit} samples")
        val_subset = Subset(val_subset, list(range(eval_limit)))

    return train_subset, val_subset