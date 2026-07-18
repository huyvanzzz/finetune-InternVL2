import io
import json
import os
import pickle
import random
import tarfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from data import process_image


REQUIRED_QUESTION_FIELDS = ("frame_path", "frame_id", "question", "gt")
DEFAULT_FRAME_INDEX_PATH = "./wad_dataset/frame_index.pkl"


@dataclass
class PretrainSplit:
    train_rows: List[Dict]
    val_rows: List[Dict]
    test_rows: List[Dict]
    stats: Dict


class PretrainSampleError(RuntimeError):
    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


@dataclass
class PretrainErrorStats:
    requested_count: int = 0
    sample_error_count: int = 0
    error_examples: Optional[List[Dict]] = None

    def __post_init__(self):
        if self.error_examples is None:
            self.error_examples = []

    @property
    def sample_error_rate(self) -> float:
        if self.requested_count <= 0:
            return 0.0
        return float(self.sample_error_count) / float(self.requested_count)

    def to_dict(self) -> Dict:
        return {
            "requested_count": self.requested_count,
            "sample_error_count": self.sample_error_count,
            "sample_error_rate": self.sample_error_rate,
            "error_examples": list(self.error_examples),
        }


def load_question_train_rows(path: os.PathLike | str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Row {line_no} of {path} must be a JSON object.")
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows found in question train file: {path}")
    return rows


def split_question_rows_grouped(rows: Sequence[Dict], val_ratio: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in [0.0, 1.0), got {val_ratio}")

    grouped: Dict[str, List[Dict]] = {}
    for row in rows:
        frame_path = str(row["frame_path"])
        grouped.setdefault(frame_path, []).append(row)

    frame_paths = sorted(grouped.keys())
    if val_ratio <= 0.0 or len(frame_paths) <= 1:
        return list(rows), []

    train_frame_paths, val_frame_paths = train_test_split(
        frame_paths,
        test_size=val_ratio,
        random_state=seed,
    )
    train_set = set(train_frame_paths)
    val_set = set(val_frame_paths)

    train_rows = [row for row in rows if str(row["frame_path"]) in train_set]
    val_rows = [row for row in rows if str(row["frame_path"]) in val_set]
    return train_rows, val_rows


def _question_id_counts(rows: Sequence[Dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        key = str(row.get("question_id", ""))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _split_stats(rows: Sequence[Dict]) -> Dict:
    return {
        "row_count": len(rows),
        "frame_count": len({str(row["frame_path"]) for row in rows}),
        "question_id_counts": _question_id_counts(rows),
    }


def split_question_rows_grouped_three_way(
    rows: Sequence[Dict],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> PretrainSplit:
    if not rows:
        raise ValueError("Cannot split an empty question row list.")
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"train/val/test ratios must sum to 1.0, got {ratio_sum}")
    if min(train_ratio, val_ratio, test_ratio) < 0.0:
        raise ValueError("train/val/test ratios must be non-negative.")

    grouped: Dict[str, List[Dict]] = {}
    for row in rows:
        frame_path = str(row["frame_path"])
        grouped.setdefault(frame_path, []).append(row)

    frame_paths = sorted(grouped.keys())
    rng = random.Random(seed)
    rng.shuffle(frame_paths)
    total_frames = len(frame_paths)
    test_count = int(round(total_frames * test_ratio))
    val_count = int(round(total_frames * val_ratio))
    if total_frames >= 3:
        if test_ratio > 0.0:
            test_count = max(1, test_count)
        if val_ratio > 0.0:
            val_count = max(1, val_count)
    if test_count + val_count >= total_frames:
        overflow = test_count + val_count - total_frames + 1
        val_count = max(0, val_count - overflow)

    test_paths = set(frame_paths[:test_count])
    val_paths = set(frame_paths[test_count : test_count + val_count])
    train_paths = set(frame_paths[test_count + val_count :])
    train_rows = [row for row in rows if str(row["frame_path"]) in train_paths]
    val_rows = [row for row in rows if str(row["frame_path"]) in val_paths]
    test_rows = [row for row in rows if str(row["frame_path"]) in test_paths]

    return PretrainSplit(
        train_rows=train_rows,
        val_rows=val_rows,
        test_rows=test_rows,
        stats={
            "train": _split_stats(train_rows),
            "val": _split_stats(val_rows),
            "test": _split_stats(test_rows),
        },
    )


def load_frame_index(frame_index_path: os.PathLike | str = DEFAULT_FRAME_INDEX_PATH) -> Dict:
    if not os.path.exists(frame_index_path):
        raise FileNotFoundError(
            f"Frame index not found at {frame_index_path}. Run build_frame_index.py first."
        )
    with open(frame_index_path, "rb") as f:
        return pickle.load(f)


class PretrainQADataset(Dataset):
    def __init__(
        self,
        rows: Sequence[Dict],
        frame_index_path: os.PathLike | str,
        trajectory_source,
        split_name: str,
        movement_enabled: bool,
    ):
        self.rows = list(rows)
        self.frame_index = load_frame_index(frame_index_path)
        self.trajectory_source = trajectory_source
        self.split_name = split_name
        self.movement_enabled = bool(movement_enabled)
        self._error_stats = PretrainErrorStats()

    def __len__(self):
        return len(self.rows)

    def reset_error_stats(self):
        self._error_stats = PretrainErrorStats()

    def peek_error_stats(self) -> Dict:
        return self._error_stats.to_dict()

    def consume_error_stats(self) -> Dict:
        snapshot = self.peek_error_stats()
        self.reset_error_stats()
        return snapshot

    def _record_error(self, row: Optional[Dict], reason: str, exc: Exception):
        self._error_stats.sample_error_count += 1
        if len(self._error_stats.error_examples) < 5:
            payload = {
                "split": self.split_name,
                "reason": reason,
                "message": str(exc),
            }
            if row is not None:
                payload.update(
                    {
                        "frame_path": row.get("frame_path"),
                        "frame_id": row.get("frame_id"),
                        "question_id": row.get("question_id"),
                    }
                )
            self._error_stats.error_examples.append(payload)

    def _validate_row(self, row: Dict):
        missing = [field for field in REQUIRED_QUESTION_FIELDS if field not in row]
        if missing:
            raise PretrainSampleError("parse_row_error", f"Question row is missing required fields: {missing}")

    def _load_frame(self, frame_path: str, frame_id: int):
        if frame_path not in self.frame_index:
            raise PretrainSampleError("missing_frame_path", f"Frame path not found in frame index: {frame_path}")
        if frame_id not in self.frame_index[frame_path]:
            raise PretrainSampleError(
                "missing_frame_id",
                f"Frame id {frame_id} not found in frame index for {frame_path}",
            )

        frame_info = self.frame_index[frame_path][frame_id]
        shard_path = frame_info["shard"]
        tar_path = frame_info["tar_path"]
        try:
            with tarfile.open(shard_path, "r") as tar:
                member = tar.getmember(tar_path)
                file_obj = tar.extractfile(member)
                if file_obj is None:
                    raise FileNotFoundError(f"Tar member not readable: {tar_path}")
                return Image.open(io.BytesIO(file_obj.read())).convert("RGB")
        except PretrainSampleError:
            raise
        except Exception as exc:
            raise PretrainSampleError("image_load_error", str(exc)) from exc

    def _prepare_pixel_values(self, image_or_tensor):
        if torch.is_tensor(image_or_tensor):
            if image_or_tensor.ndim == 3:
                return image_or_tensor.unsqueeze(0)
            return image_or_tensor
        return process_image(image_or_tensor)

    def _build_sample(self, row: Dict) -> Dict:
        self._validate_row(row)
        frame_path = str(row["frame_path"])
        frame_id = int(row["frame_id"])
        question_text = str(row["question"]).strip()
        answer_text = str(row["gt"]).strip()
        question_id = str(row.get("question_id", ""))

        try:
            image = self._load_frame(frame_path, frame_id)
        except PretrainSampleError:
            raise
        except Exception as exc:
            raise PretrainSampleError("image_load_error", str(exc)) from exc

        pixel_values = self._prepare_pixel_values(image)

        try:
            if hasattr(self.trajectory_source, "has_record") and not self.trajectory_source.has_record(frame_path, frame_id):
                raise PretrainSampleError(
                    "missing_trajectory_record",
                    f"Trajectory record not found for {(frame_path, frame_id)}",
                )
            trajectory_fields = self.trajectory_source.encode(frame_path, frame_id)
        except Exception as exc:
            if isinstance(exc, PretrainSampleError):
                raise
            raise PretrainSampleError("trajectory_join_error", str(exc)) from exc

        if not self.movement_enabled:
            trajectory_fields = dict(trajectory_fields)
            numeric_feats = torch.as_tensor(trajectory_fields["trajectory_numeric_feats"], dtype=torch.float32).clone()
            numeric_feats[:, 4] = 0.0
            numeric_feats[:, 5] = 0.0
            trajectory_fields["trajectory_numeric_feats"] = numeric_feats

        qformer_text = f"Question: {question_text}"
        return {
            "question": f"<image>\n{qformer_text}",
            "answer": f"Answer: {answer_text}",
            "qformer_text": qformer_text,
            "pixel_values": [pixel_values],
            "frame_path": frame_path,
            "frame_id": frame_id,
            "question_id": question_id,
            **trajectory_fields,
        }

    def get_debug_snapshot(self, idx: int) -> Dict:
        sample = self._build_sample(self.rows[idx])
        return {
            "frame_path": sample["frame_path"],
            "frame_id": sample["frame_id"],
            "question": sample["question"],
            "qformer_text": sample["qformer_text"],
            "answer": sample["answer"],
            "trajectory_label_ids": sample["trajectory_label_ids"].tolist(),
            "trajectory_direction_ids": sample["trajectory_direction_ids"].tolist(),
            "trajectory_object_mask": sample["trajectory_object_mask"].tolist(),
            "trajectory_numeric_feats": sample["trajectory_numeric_feats"].tolist(),
        }

    def __getitem__(self, idx: int):
        self._error_stats.requested_count += 1
        candidate_indices = [idx]
        if len(self.rows) > 1:
            others = [i for i in range(len(self.rows)) if i != idx]
            candidate_indices.extend(random.sample(others, len(others)))

        last_exc = None
        for candidate_idx in candidate_indices:
            row = self.rows[candidate_idx]
            try:
                return self._build_sample(row)
            except PretrainSampleError as exc:
                self._record_error(row, exc.reason, exc)
                last_exc = exc
            except Exception as exc:
                self._record_error(row, "unknown_sample_error", exc)
                last_exc = exc

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Pretrain dataset could not produce a sample and did not capture an exception.")


def build_pretrain_datasets(
    question_train_file: os.PathLike | str,
    frame_index_path: os.PathLike | str,
    trajectory_source,
    val_split_ratio: float,
    val_split_seed: int,
    movement_enabled: bool,
    train_split_ratio: float = 0.8,
    test_split_ratio: float = 0.1,
    question_val_file: os.PathLike | str | None = None,
    question_test_file: os.PathLike | str | None = None,
):
    train_rows = load_question_train_rows(question_train_file)
    if question_val_file is not None and question_test_file is not None:
        val_rows = load_question_train_rows(question_val_file)
        test_rows = load_question_train_rows(question_test_file)
        split = PretrainSplit(
            train_rows=train_rows,
            val_rows=val_rows,
            test_rows=test_rows,
            stats={
                "train": _split_stats(train_rows),
                "val": _split_stats(val_rows),
                "test": _split_stats(test_rows),
            },
        )
        verify_pretrain_split_no_frame_leak(split)
    else:
        split = split_question_rows_grouped_three_way(
            train_rows,
            train_ratio=train_split_ratio,
            val_ratio=val_split_ratio,
            test_ratio=test_split_ratio,
            seed=val_split_seed,
        )
    train_dataset = PretrainQADataset(
        rows=split.train_rows,
        frame_index_path=frame_index_path,
        trajectory_source=trajectory_source,
        split_name="train",
        movement_enabled=movement_enabled,
    )
    val_dataset = PretrainQADataset(
        rows=split.val_rows,
        frame_index_path=frame_index_path,
        trajectory_source=trajectory_source,
        split_name="val",
        movement_enabled=movement_enabled,
    )
    test_dataset = PretrainQADataset(
        rows=split.test_rows,
        frame_index_path=frame_index_path,
        trajectory_source=trajectory_source,
        split_name="test",
        movement_enabled=movement_enabled,
    )
    return train_dataset, val_dataset, test_dataset, split.stats


def verify_pretrain_split_no_frame_leak(split: PretrainSplit):
    frame_sets = {
        "train": {str(row["frame_path"]) for row in split.train_rows},
        "val": {str(row["frame_path"]) for row in split.val_rows},
        "test": {str(row["frame_path"]) for row in split.test_rows},
    }
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = frame_sets[left] & frame_sets[right]
        if overlap:
            raise ValueError(f"frame_path leakage between {left} and {right}: {sorted(overlap)[:5]}")
    for name, rows in (("train", split.train_rows), ("val", split.val_rows), ("test", split.test_rows)):
        if not rows:
            raise ValueError(f"Pretrain split {name} is empty.")
