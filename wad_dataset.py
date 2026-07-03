import io
import os
import pickle
import random
import tarfile
from collections import Counter, defaultdict
from typing import Dict, List

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from data import process_image
from preprocessing import format_ground_truth, get_response_format


ALTER_DIRECT_TEXT_LEGACY_PROMPT = (
    "Describe the scene for a visually impaired user based on this image.\n"
    "Focus on immediate obstacles, safe direction, and what action the user should take.\n"
    "Provide only the final spoken guidance in natural language."
)

ALTER_DIRECT_TEXT_TEMPLATES = {
    "T1": (
        "Provide brief navigation guidance for a visually impaired user based on this image.\n"
        "Focus on immediate obstacles, the safe direction, and the next safe action.\n"
        "Output only the final guidance in natural language."
    ),
    "T2": (
        "Look at this image and guide a visually impaired user.\n"
        "Mention nearby hazards, where it is safe to move, and what to do next.\n"
        "Return only the final spoken guidance."
    ),
    "T3": (
        "Give spoken navigation guidance for a visually impaired user from this image.\n"
        "Focus on close obstacles, the safest direction, and the next action.\n"
        "Output only the final guidance sentence."
    ),
    "T4": (
        "From this image, give concise navigation guidance for a visually impaired user.\n"
        "Mention nearby obstacles, the safe way forward, and what the user should do now.\n"
        "Output only the final spoken guidance in natural language."
    ),
}


def get_sample_task_type(sample: Dict) -> str:
    qa = sample.get("QA")
    if qa and isinstance(qa, dict) and qa.get("Q"):
        return "qa"
    return "alter"


def summarize_task_types(samples) -> Dict[str, int]:
    counts = Counter(get_sample_task_type(sample) for sample in samples)
    return {"qa": counts.get("qa", 0), "alter": counts.get("alter", 0)}


def summarize_task_types_from_indices(metadata, indices) -> Dict[str, int]:
    counts = Counter(get_sample_task_type(metadata[idx]) for idx in indices)
    return {"qa": counts.get("qa", 0), "alter": counts.get("alter", 0)}


def get_allowed_task_types(task_filter: str):
    normalized_filter = str(task_filter or "all").strip().lower()
    if normalized_filter == "all":
        return {"qa", "alter"}
    if normalized_filter == "alter_only":
        return {"alter"}
    if normalized_filter == "qa_only":
        return {"qa"}
    raise ValueError(f"Unsupported task filter: {task_filter}")


def filter_samples_by_task_filter(samples, task_filter: str):
    allowed_task_types = get_allowed_task_types(task_filter)
    return [sample for sample in samples if get_sample_task_type(sample) in allowed_task_types]


def should_stratify_task_split(task_labels) -> bool:
    return len(set(task_labels)) > 1


def should_use_stratified_split(config: Dict, task_labels) -> bool:
    data_cfg = config.get("data", {})
    if not data_cfg.get("stratify_split", True):
        return False
    return should_stratify_task_split(task_labels)


def build_balanced_sample_weights(task_types, task_target_weights: Dict[str, float]):
    counts = Counter(task_types)
    weights = []
    for task_type in task_types:
        target_weight = float(task_target_weights.get(task_type, 0.0))
        task_count = counts.get(task_type, 0)
        if task_count <= 0 or target_weight <= 0:
            weights.append(0.0)
            continue
        weights.append(target_weight / task_count)
    return weights


def resolve_eval_limit(eval_limit):
    if eval_limit is None:
        return None
    eval_limit = int(eval_limit)
    if eval_limit <= 0:
        return None
    return eval_limit


def get_direct_text_alter_prompt_templates() -> Dict[str, str]:
    return dict(ALTER_DIRECT_TEXT_TEMPLATES)


def get_direct_text_alter_prompt_text(prompt_mode: str, split: str, prompt_id: str | None = None):
    normalized_mode = str(prompt_mode or "fixed_legacy").strip().lower()
    normalized_split = str(split or "train").strip().lower()
    if normalized_mode == "fixed_legacy":
        return "legacy", ALTER_DIRECT_TEXT_LEGACY_PROMPT
    if normalized_mode == "fixed_v1":
        return "T1", ALTER_DIRECT_TEXT_TEMPLATES["T1"]
    if normalized_mode == "balanced_v1":
        if normalized_split in {"val", "test"}:
            return "T1", ALTER_DIRECT_TEXT_TEMPLATES["T1"]
        selected_prompt_id = prompt_id or "T1"
        if selected_prompt_id not in ALTER_DIRECT_TEXT_TEMPLATES:
            raise ValueError(f"Unknown prompt id for balanced_v1: {selected_prompt_id}")
        return selected_prompt_id, ALTER_DIRECT_TEXT_TEMPLATES[selected_prompt_id]
    raise ValueError(f"Unsupported direct_text alter prompt mode: {prompt_mode}")


class WADDatasetForInternVL(Dataset):
    def __init__(
        self,
        metadata_dataset,
        frame_index: dict,
        bbox_by_folder: dict,
        split: str = "train",
        response_format: str = "structured_json",
        direct_text_alter_prompt_mode: str = "fixed_legacy",
        seed: int = 42,
    ):
        self.metadata = metadata_dataset[split]
        self.frame_index = frame_index
        self.bbox_by_folder = bbox_by_folder
        self.split = split
        self.response_format = response_format
        self.direct_text_alter_prompt_mode = direct_text_alter_prompt_mode
        self.seed = int(seed)
        self.prompt_templates = get_direct_text_alter_prompt_templates()
        self.prompt_assignment = {}
        self.task_types = [get_sample_task_type(sample) for sample in self.metadata]
        if self.response_format == "direct_text" and self.direct_text_alter_prompt_mode == "balanced_v1":
            self.set_epoch(0)

    def __len__(self):
        return len(self.metadata)

    def set_epoch(self, epoch: int):
        if self.response_format != "direct_text" or self.direct_text_alter_prompt_mode != "balanced_v1" or self.split != "train":
            self.prompt_assignment = {}
            return

        alter_indices = [idx for idx, task_type in enumerate(self.task_types) if task_type == "alter"]
        prompt_ids = list(self.prompt_templates.keys())
        if not alter_indices:
            self.prompt_assignment = {}
            return

        assignments = []
        base_count = len(alter_indices) // len(prompt_ids)
        remainder = len(alter_indices) % len(prompt_ids)
        for prompt_offset, prompt_id in enumerate(prompt_ids):
            prompt_count = base_count + (1 if prompt_offset < remainder else 0)
            assignments.extend([prompt_id] * prompt_count)

        rng = random.Random(self.seed + int(epoch))
        rng.shuffle(assignments)
        self.prompt_assignment = {
            sample_idx: prompt_id
            for sample_idx, prompt_id in zip(alter_indices, assignments)
        }

    def _load_frames(self, frame_path: str, frame_ids: List[int]) -> List[Image.Image]:
        shard_to_frames = defaultdict(list)
        for frame_id in frame_ids:
            if frame_id not in self.frame_index[frame_path]:
                raise ValueError(f"Frame {frame_id} not in index")
            frame_info = self.frame_index[frame_path][frame_id]
            shard_to_frames[frame_info["shard"]].append((frame_id, frame_info["tar_path"]))

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

    def _get_selected_direct_text_prompt(self, idx: int, sample: Dict):
        task_type = get_sample_task_type(sample)
        if task_type == "qa":
            qa_prompt = (
            "Describe the scene for a visually impaired user based on this frame.\n"
            "Focus on obstacles, nearby people or vehicles, free walking space, direction, and safety.\n"
            f"Question: {sample['QA']['Q']}"
            )
            return "qa_default", qa_prompt

        prompt_id = None
        if self.direct_text_alter_prompt_mode == "balanced_v1" and self.split == "train":
            prompt_id = self.prompt_assignment.get(idx, "T1")
        return get_direct_text_alter_prompt_text(
            self.direct_text_alter_prompt_mode,
            self.split,
            prompt_id=prompt_id,
        )

    def __getitem__(self, idx):
        try:
            sample = self.metadata[idx]
            frame_path = sample["frame_path"]

            frame_ids = self._select_frames_safe(frame_path)
            last_frame_id = frame_ids[-1]
            frames = self._load_frames(frame_path, [last_frame_id])
            pixel_values = [process_image(img) for img in frames]

            task_type = get_sample_task_type(sample)
            has_question = task_type == "qa"
            if self.response_format == "direct_text":
                selected_prompt_id, text_content = self._get_selected_direct_text_prompt(idx, sample)
            else:
                selected_prompt_id = "structured_json"
                text_content = """

Analyze: location, weather, traffic, scene → then give instruction.

Follow Chain-of-Thought reasoning:
1. Perception: Extract "location", "weather", and "traffic".
2. Comprehension: Synthesize details into the "scene".
3. Decision: Formulate the final "instruction"."""

                if has_question:
                    text_content += f"\n\nQuestion: {sample['QA']['Q']}"
                    text_content += """\n\nFormat response:
<answer>{"location": "...", "weather": "...", "traffic": "...", "scene": "<concise visual summary, max 2 sentences>", "instruction": "<your answer to the question>"}</answer>"""
                else:
                    text_content += """\n\nFormat response:
<answer>{"location": "...", "weather": "...", "traffic": "...", "scene": "<concise visual summary, max 2 sentences>", "instruction": "<actionable alert and guidance>"}</answer>"""

            question = f"<image>\n{text_content}"
            answer = format_ground_truth(sample, self.response_format)

            return {
                "question": question,
                "answer": answer,
                "qformer_text": text_content.strip(),
                "pixel_values": pixel_values,
                "questionId": str(idx),
                "image": frames,
                "task_type": task_type,
                "selected_prompt_id": selected_prompt_id,
                "selected_prompt_text": text_content.strip(),
                "frame_path": frame_path,
            }

        except Exception as e:
            frame_path = None
            try:
                frame_path = self.metadata[idx].get("frame_path")
            except Exception:
                pass

            if self.split in ("test", "val"):
                print(
                    f"[DATA ERROR][{self.split}] idx={idx} | frame_path={frame_path} | "
                    f"error={type(e).__name__}: {e}"
                )
                return None

            new_idx = random.randint(0, len(self) - 1)
            return self.__getitem__(new_idx)


def build_dataset(config: Dict):
    """Build train/eval datasets from config"""

    from datasets import load_dataset

    response_format = get_response_format(config)
    prompt_mode = config["data"].get("direct_text_alter_prompt_mode", "fixed_legacy")
    print(
        f"[DATA SUMMARY] response_format={response_format} | prompt_mode={prompt_mode} | "
        f"train_task_filter={config['data'].get('train_task_filter', 'all')} | "
        f"val_task_filter={config['data'].get('val_task_filter', config['data'].get('train_task_filter', 'all'))} | "
        f"stratify_split={config['data'].get('stratify_split', True)}"
    )

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
        print("✓ Using dynamic resolution for Qwen")
    elif architecture == "internvl":
        print("✓ Using fixed tile size (448, 448) for InternVL")
    elif architecture == "sailvl":
        image_size = config["model"]["vision"]["force_image_size"]
        max_dynamic_patch = config["model"]["vision"].get("max_dynamic_patch")
        print(
            f"✓ Using native dynamic resolution for SAIL "
            f"(force_image_size={image_size}, max_dynamic_patch={max_dynamic_patch})"
        )
    else:
        image_size = tuple(config["model"]["vision"]["image_size"])
        print(f"✓ Using image size {image_size} for {architecture}")

    raw_train_samples = list(metadata["train"])
    raw_stats = summarize_task_types(raw_train_samples)
    print(
        f"  Task distribution (full train.json) | "
        f"QA={raw_stats['qa']} | alter={raw_stats['alter']}"
    )

    train_task_filter = config["data"].get("train_task_filter", "all")
    val_task_filter = config["data"].get("val_task_filter", train_task_filter)
    train_allowed = get_allowed_task_types(train_task_filter)
    val_allowed = get_allowed_task_types(val_task_filter)
    combined_allowed = train_allowed | val_allowed
    filtered_train_samples = [
        sample for sample in raw_train_samples if get_sample_task_type(sample) in combined_allowed
    ]
    filtered_stats = summarize_task_types(filtered_train_samples)
    print(
        f"  After task filter union | train={train_task_filter} | val={val_task_filter} | "
        f"QA={filtered_stats['qa']} | alter={filtered_stats['alter']}"
    )
    if train_task_filter == "all" and val_task_filter == "all":
        print("  Mixed setup confirmed: both QA and alter are eligible for train/val.")
    if not filtered_train_samples:
        raise ValueError(
            "Task filtering removed all training samples. "
            f"train_task_filter={train_task_filter}, val_task_filter={val_task_filter}"
        )

    indices = list(range(len(filtered_train_samples)))
    task_labels = [get_sample_task_type(filtered_train_samples[idx]) for idx in indices]
    split_kwargs = {
        "train_size": config["data"]["train_split"],
        "random_state": config["data"]["seed"],
    }
    if should_use_stratified_split(config, task_labels):
        split_kwargs["stratify"] = task_labels
    else:
        print("  Using plain random split without task stratification.")

    train_indices, val_indices = train_test_split(indices, **split_kwargs)
    train_samples = [filtered_train_samples[idx] for idx in train_indices if task_labels[idx] in train_allowed]
    val_samples = [filtered_train_samples[idx] for idx in val_indices if task_labels[idx] in val_allowed]
    if not train_samples:
        raise ValueError(
            "No training samples remain after applying train_task_filter. "
            f"train_task_filter={train_task_filter}"
        )
    if not val_samples:
        raise ValueError(
            "No validation samples remain after applying val_task_filter. "
            f"val_task_filter={val_task_filter}"
        )

    train_limit = resolve_eval_limit(config["data"].get("train_limit"))
    if train_limit is not None and len(train_samples) > train_limit:
        print(f"  Limiting train dataset: {len(train_samples)} → {train_limit} samples")
        train_samples = train_samples[:train_limit]

    eval_limit = resolve_eval_limit(config["data"].get("eval_limit"))
    if eval_limit is not None and len(val_samples) > eval_limit:
        print(f"  Limiting eval dataset: {len(val_samples)} → {eval_limit} samples")
        val_samples = val_samples[:eval_limit]
        limited_stats = summarize_task_types(val_samples)
        print(f"  Limited val stats | QA={limited_stats['qa']} | alter={limited_stats['alter']}")

    train_dataset = WADDatasetForInternVL(
        metadata_dataset={"train": train_samples},
        frame_index=frame_index,
        bbox_by_folder=bbox_by_folder,
        split="train",
        response_format=response_format,
        direct_text_alter_prompt_mode=prompt_mode,
        seed=config["data"]["seed"],
    )
    train_dataset.task_types = [get_sample_task_type(sample) for sample in train_samples]

    val_dataset = WADDatasetForInternVL(
        metadata_dataset={"val": val_samples},
        frame_index=frame_index,
        bbox_by_folder=bbox_by_folder,
        split="val",
        response_format=response_format,
        direct_text_alter_prompt_mode=prompt_mode,
        seed=config["data"]["seed"],
    )
    val_dataset.task_types = [get_sample_task_type(sample) for sample in val_samples]

    print(f"✓ Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    train_stats = summarize_task_types(train_samples)
    val_stats = summarize_task_types(val_samples)
    print(
        f"  Final split stats | "
        f"train(QA={train_stats['qa']}, alter={train_stats['alter']}) | "
        f"val(QA={val_stats['qa']}, alter={val_stats['alter']})"
    )

    return train_dataset, val_dataset
