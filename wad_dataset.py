import io
import os
import pickle
import random
import tarfile
from collections import defaultdict
from typing import Dict, List

import torch
from PIL import Image
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset

from data import process_image
from preprocessing import format_ground_truth, get_response_format
from trajectory_branch import build_trajectory_source_from_config


class WADDatasetForInternVL(Dataset):
    def __init__(
        self,
        metadata_dataset,
        frame_index: dict,
        bbox_by_folder: dict,
        trajectory_source=None,
        split: str = "train",
        response_format: str = "structured_json",
    ):
        self.metadata = metadata_dataset[split]
        self.frame_index = frame_index
        self.bbox_by_folder = bbox_by_folder
        self.trajectory_source = trajectory_source
        self.split = split
        self.response_format = response_format

    def __len__(self):
        return len(self.metadata)

    def _load_frames(self, frame_path: str, frame_ids: List[int]) -> List[Image.Image]:
        shard_to_frames = {}
        for frame_id in frame_ids:
            if frame_id not in self.frame_index[frame_path]:
                raise ValueError(f"Frame {frame_id} not in index")
            frame_info = self.frame_index[frame_path][frame_id]
            shard_path = frame_info["shard"]
            shard_to_frames.setdefault(shard_path, []).append((frame_id, frame_info["tar_path"]))

        frames_dict = {}
        for shard_path, frame_list in shard_to_frames.items():
            with tarfile.open(shard_path, "r") as tar:
                for frame_id, tar_path in frame_list:
                    member = tar.getmember(tar_path)
                    file_obj = tar.extractfile(member)
                    img = Image.open(io.BytesIO(file_obj.read())).convert("RGB")
                    frames_dict[frame_id] = img
        return [frames_dict[fid] for fid in frame_ids]

    def _select_frames_safe(self, frame_path: str) -> List[int]:
        available_frames = sorted(self.frame_index[frame_path].keys())
        target_indices = [4, 6, 8]

        selected_frames = []
        for idx in target_indices:
            if idx < len(available_frames):
                selected_frames.append(available_frames[idx])
            else:
                selected_frames.append(available_frames[-1])
        return selected_frames

    def _build_text_content(self, sample: Dict) -> str:
        if self.response_format == "direct_text":
            text_content = "Describe the scene for a visually impaired user based on the final frame."
        else:
            text_content = """

Analyze: location, weather, traffic, scene → then give instruction.

Follow Chain-of-Thought reasoning:
1. Perception: Extract "location", "weather", and "traffic".
2. Comprehension: Synthesize details into the "scene".
3. Decision: Formulate the final "instruction"."""

        has_question = sample.get("QA") and sample["QA"].get("Q")
        if has_question:
            if self.response_format == "direct_text":
                text_content += (
                    "\nFocus on obstacles, nearby people or vehicles, free walking space, direction, and safety."
                    f"\nQuestion: {sample['QA']['Q']}"
                )
            else:
                text_content += f"\n\nQuestion: {sample['QA']['Q']}"
            if self.response_format == "direct_text":
                text_content += "\nAnswer the question directly in natural language."
            else:
                text_content += """\n\nFormat response:
<answer>{"location": "...", "weather": "...", "traffic": "...", "scene": "<concise visual summary, max 2 sentences>", "instruction": "<your answer to the question>"}</answer>"""
        else:
            if self.response_format == "direct_text":
                text_content += (
                    "\nFocus on immediate obstacles, safe direction, and what action the user should take."
                    "\nProvide only the final spoken guidance in natural language."
                )
            else:
                text_content += """\n\nFormat response:
<answer>{"location": "...", "weather": "...", "traffic": "...", "scene": "<concise visual summary, max 2 sentences>", "instruction": "<actionable alert and guidance>"}</answer>"""
        return text_content

    def _build_question(self, text_content: str) -> str:
        return f"<image>\n{text_content}"

    def get_debug_snapshot(self, idx: int) -> Dict:
        sample = self.metadata[idx]
        frame_path = sample["frame_path"]
        frame_ids = self._select_frames_safe(frame_path)
        last_frame_id = frame_ids[-1]
        text_content = self._build_text_content(sample)
        answer = format_ground_truth(sample, self.response_format)

        snapshot = {
            "frame_path": frame_path,
            "last_frame_id": last_frame_id,
            "question": self._build_question(text_content),
            "qformer_text": text_content.strip(),
            "answer": answer,
            "has_trajectory": False,
        }
        if self.trajectory_source is not None:
            trajectory = self.trajectory_source.encode(frame_path, last_frame_id)
            snapshot["has_trajectory"] = bool(trajectory["trajectory_object_mask"].sum().item() > 0)
            snapshot["trajectory_label_ids"] = trajectory["trajectory_label_ids"].tolist()
            snapshot["trajectory_direction_ids"] = trajectory["trajectory_direction_ids"].tolist()
            snapshot["trajectory_object_mask"] = trajectory["trajectory_object_mask"].tolist()
            snapshot["trajectory_numeric_feats"] = trajectory["trajectory_numeric_feats"].tolist()
        return snapshot

    def __getitem__(self, idx):
        try:
            sample = self.metadata[idx]
            frame_path = sample["frame_path"]

            frame_ids = self._select_frames_safe(frame_path)
            last_frame_id = frame_ids[-1]
            frames = self._load_frames(frame_path, [last_frame_id])
            pixel_values = [process_image(img) for img in frames]

            text_content = self._build_text_content(sample)
            question = self._build_question(text_content)
            answer = format_ground_truth(sample, self.response_format)
            trajectory_fields = (
                self.trajectory_source.encode(frame_path, last_frame_id)
                if self.trajectory_source is not None
                else {}
            )

            return {
                "question": question,
                "answer": answer,
                "qformer_text": text_content.strip(),
                "pixel_values": pixel_values,
                "questionId": str(idx),
                "image": frames,
                **trajectory_fields,
            }
        except Exception:
            new_idx = random.randint(0, len(self) - 1)
            return self.__getitem__(new_idx)


def _trajectory_match_stats(dataset: WADDatasetForInternVL, subset: Subset):
    exact_matches = 0
    empty_objects = 0
    for idx in subset.indices:
        sample = dataset.metadata[idx]
        frame_path = sample["frame_path"]
        frame_ids = dataset._select_frames_safe(frame_path)
        last_frame_id = frame_ids[-1]
        if dataset.trajectory_source.has_record(frame_path, last_frame_id):
            exact_matches += 1
        else:
            empty_objects += 1
    return exact_matches, empty_objects


def _print_debug_samples(name: str, dataset: WADDatasetForInternVL, subset: Subset, limit: int = 2):
    if len(subset.indices) == 0:
        return
    print(f"[DEBUG] {name} prompt/trajectory samples:")
    for local_i, idx in enumerate(subset.indices[:limit], start=1):
        snapshot = dataset.get_debug_snapshot(idx)
        print(f"[DEBUG] {name} sample {local_i} | frame_path={snapshot['frame_path']} | last_frame_id={snapshot['last_frame_id']}")
        print(f"[DEBUG] question: {snapshot['question']}")
        print(f"[DEBUG] qformer_text: {snapshot['qformer_text']}")
        print(f"[DEBUG] answer: {snapshot['answer']}")
        if "trajectory_label_ids" in snapshot:
            print(
                f"[DEBUG] trajectory | has_trajectory={snapshot['has_trajectory']} | "
                f"label_ids={snapshot['trajectory_label_ids']} | "
                f"direction_ids={snapshot['trajectory_direction_ids']} | "
                f"object_mask={snapshot['trajectory_object_mask']} | "
                f"numeric_feats={snapshot['trajectory_numeric_feats']}"
            )


def build_dataset(config: Dict):
    response_format = get_response_format(config)
    trajectory_source = build_trajectory_source_from_config(config)

    print("Loading metadata...")
    metadata = load_dataset(
        config["data"]["name"],
        data_files={
            "train": "train.json",
            "test": "test_alter.json",
        },
    )

    print("Loading bboxes...")
    bbox_dataset = load_dataset(
        config["data"]["name"],
        data_files="all_bboxes_1.jsonl",
        split="train",
    )

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

    print("Loading frame index...")
    index_file = "./wad_dataset/frame_index.pkl"
    if os.path.exists(index_file):
        with open(index_file, "rb") as f:
            frame_index = pickle.load(f)
    else:
        raise FileNotFoundError(f"Frame index not found at {index_file}. Run build_frame_index.py first.")

    architecture = config["model"]["architecture"]
    if architecture == "qwen":
        image_size = None
        print("✓ Using dynamic resolution for Qwen")
    elif architecture == "internvl":
        image_size = (448, 448)
        print(f"✓ Using fixed tile size {image_size} for InternVL")
    else:
        image_size = tuple(config["model"]["vision"]["image_size"])
        print(f"✓ Using image size {image_size} for {architecture}")

    train_dataset = WADDatasetForInternVL(
        metadata_dataset=metadata,
        frame_index=frame_index,
        bbox_by_folder=bbox_by_folder,
        trajectory_source=trajectory_source,
        split="train",
        response_format=response_format,
    )

    train_size = config["data"]["train_split"]
    indices = list(range(len(train_dataset)))
    train_indices, val_indices = train_test_split(
        indices,
        train_size=train_size,
        random_state=config["data"]["seed"],
    )

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)

    print(f"✓ Train: {len(train_subset)}, Val: {len(val_subset)}")
    if trajectory_source is not None:
        train_exact, train_empty = _trajectory_match_stats(train_dataset, train_subset)
        val_exact, val_empty = _trajectory_match_stats(train_dataset, val_subset)
        print(
            "[DEBUG] Trajectory join stats | "
            f"source={trajectory_source.source_file} | "
            f"train(exact={train_exact}, empty={train_empty}) | "
            f"val(exact={val_exact}, empty={val_empty})"
        )
        _print_debug_samples("train", train_dataset, train_subset, limit=2)
        _print_debug_samples("val", train_dataset, val_subset, limit=2)

    eval_limit = config["data"].get("eval_limit", 200)
    if len(val_subset) > eval_limit:
        print(f"  Limiting eval dataset: {len(val_subset)} → {eval_limit} samples")
        val_subset = Subset(val_subset, list(range(eval_limit)))

    return train_subset, val_subset
