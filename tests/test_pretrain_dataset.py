import json
import pickle
from pathlib import Path

import pytest
import torch

from pretrain_dataset import (
    PretrainQADataset,
    PretrainSampleError,
    build_pretrain_datasets,
    load_question_train_rows,
    split_question_rows_grouped,
    split_question_rows_grouped_three_way,
)


def _write_jsonl_with_bom(path: Path, rows):
    with path.open("w", encoding="utf-8-sig") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_frame_index(path: Path, frame_paths):
    index = {}
    for frame_path in frame_paths:
        index[frame_path] = {
            8: {
                "shard": "dummy.tar",
                "tar_path": f"{frame_path}/0008.jpg",
            }
        }
    with path.open("wb") as f:
        pickle.dump(index, f)


class _FakeTrajectorySource:
    def __init__(self, fail_keys=None):
        self.fail_keys = set(fail_keys or [])

    def has_record(self, frame_path, frame_id):
        return (frame_path, frame_id) not in self.fail_keys

    def encode(self, frame_path, frame_id):
        if (frame_path, frame_id) in self.fail_keys:
            raise ValueError(f"trajectory missing for {(frame_path, frame_id)}")
        return {
            "trajectory_label_ids": torch.tensor([1, 2, 0, 0, 0, 0], dtype=torch.long),
            "trajectory_direction_ids": torch.tensor([3, 4, 0, 0, 0, 0], dtype=torch.long),
            "trajectory_numeric_feats": torch.ones(6, 6, dtype=torch.float32),
            "trajectory_object_mask": torch.tensor([1, 1, 0, 0, 0, 0], dtype=torch.long),
        }


def test_load_question_train_rows_supports_utf8_bom(tmp_path):
    rows = [
        {
            "frame_path": "sample_a.frame",
            "frame_id": 8,
            "question_id": "q1",
            "question": "What is closest?",
            "gt": "a chair",
        }
    ]
    path = tmp_path / "question_train.jsonl"
    _write_jsonl_with_bom(path, rows)

    loaded = load_question_train_rows(path)

    assert loaded == rows


def test_split_question_rows_grouped_keeps_same_frame_path_together():
    rows = [
        {"frame_path": "frame_a", "frame_id": 8, "question_id": "q1", "question": "q", "gt": "a"},
        {"frame_path": "frame_a", "frame_id": 8, "question_id": "q2", "question": "q", "gt": "a"},
        {"frame_path": "frame_b", "frame_id": 8, "question_id": "q3", "question": "q", "gt": "a"},
        {"frame_path": "frame_c", "frame_id": 8, "question_id": "q4", "question": "q", "gt": "a"},
    ]

    train_rows, val_rows = split_question_rows_grouped(rows, val_ratio=0.34, seed=42)

    train_paths = {row["frame_path"] for row in train_rows}
    val_paths = {row["frame_path"] for row in val_rows}
    assert train_paths.isdisjoint(val_paths)
    assert train_paths | val_paths == {"frame_a", "frame_b", "frame_c"}


def test_split_question_rows_grouped_three_way_uses_frame_path_groups_and_reports_stats():
    rows = []
    for idx in range(10):
        frame_path = f"frame_{idx:02d}"
        for q_idx in range(2):
            rows.append(
                {
                    "frame_path": frame_path,
                    "frame_id": 8,
                    "question_id": f"q{q_idx}",
                    "question": "q",
                    "gt": "a",
                }
            )

    split = split_question_rows_grouped_three_way(
        rows,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42,
    )

    train_paths = {row["frame_path"] for row in split.train_rows}
    val_paths = {row["frame_path"] for row in split.val_rows}
    test_paths = {row["frame_path"] for row in split.test_rows}
    assert len(train_paths) == 8
    assert len(val_paths) == 1
    assert len(test_paths) == 1
    assert train_paths.isdisjoint(val_paths)
    assert train_paths.isdisjoint(test_paths)
    assert val_paths.isdisjoint(test_paths)
    assert split.stats["train"]["frame_count"] == 8
    assert split.stats["val"]["row_count"] == 2
    assert split.stats["test"]["question_id_counts"] == {"q0": 1, "q1": 1}


def test_pretrain_dataset_builds_expected_sample(tmp_path, monkeypatch):
    frame_index_path = tmp_path / "frame_index.pkl"
    _write_frame_index(frame_index_path, ["sample_a.frame"])

    dataset = PretrainQADataset(
        rows=[
            {
                "frame_path": "sample_a.frame",
                "frame_id": 8,
                "question_id": "q1",
                "question": "What is closest?",
                "gt": "a chair",
            }
        ],
        frame_index_path=frame_index_path,
        trajectory_source=_FakeTrajectorySource(),
        split_name="train",
        movement_enabled=True,
    )

    monkeypatch.setattr(
        dataset,
        "_load_frame",
        lambda frame_path, frame_id: torch.zeros(3, 4, 4),
    )

    sample = dataset[0]

    assert sample["question"] == "<image>\nQuestion: What is closest?"
    assert sample["answer"] == "Answer: a chair"
    assert sample["qformer_text"] == "Question: What is closest?"
    assert sample["trajectory_numeric_feats"].shape == (6, 6)
    assert sample["frame_path"] == "sample_a.frame"
    assert sample["frame_id"] == 8
    assert sample["question_id"] == "q1"


def test_pretrain_dataset_without_movement_zeroes_last_two_features(tmp_path, monkeypatch):
    frame_index_path = tmp_path / "frame_index.pkl"
    _write_frame_index(frame_index_path, ["sample_a.frame"])

    dataset = PretrainQADataset(
        rows=[
            {
                "frame_path": "sample_a.frame",
                "frame_id": 8,
                "question_id": "q1",
                "question": "What is closest?",
                "gt": "a chair",
            }
        ],
        frame_index_path=frame_index_path,
        trajectory_source=_FakeTrajectorySource(),
        split_name="train",
        movement_enabled=False,
    )
    monkeypatch.setattr(dataset, "_load_frame", lambda frame_path, frame_id: torch.zeros(3, 4, 4))

    sample = dataset[0]

    assert sample["trajectory_numeric_feats"][:, 4].tolist() == pytest.approx([0.0] * 6)
    assert sample["trajectory_numeric_feats"][:, 5].tolist() == pytest.approx([0.0] * 6)


def test_pretrain_dataset_tracks_random_fallback_errors(tmp_path, monkeypatch):
    frame_index_path = tmp_path / "frame_index.pkl"
    _write_frame_index(frame_index_path, ["broken.frame", "good.frame"])

    dataset = PretrainQADataset(
        rows=[
            {
                "frame_path": "broken.frame",
                "frame_id": 8,
                "question_id": "q1",
                "question": "Broken?",
                "gt": "no",
            },
            {
                "frame_path": "good.frame",
                "frame_id": 8,
                "question_id": "q2",
                "question": "Good?",
                "gt": "yes",
            },
        ],
        frame_index_path=frame_index_path,
        trajectory_source=_FakeTrajectorySource(fail_keys={("broken.frame", 8)}),
        split_name="train",
        movement_enabled=True,
    )

    def _fake_load_frame(frame_path, frame_id):
        if frame_path == "broken.frame":
            raise FileNotFoundError("missing image")
        return torch.zeros(3, 4, 4)

    monkeypatch.setattr(dataset, "_load_frame", _fake_load_frame)

    sample = dataset[0]
    stats = dataset.peek_error_stats()

    assert sample["frame_path"] == "good.frame"
    assert stats["sample_error_count"] == 1
    assert stats["sample_error_rate"] == pytest.approx(1.0)
    assert stats["error_examples"][0]["reason"] in {"image_load_error", "missing_trajectory_record"}


def test_pretrain_dataset_marks_missing_trajectory_record_explicitly(tmp_path, monkeypatch):
    frame_index_path = tmp_path / "frame_index.pkl"
    _write_frame_index(frame_index_path, ["missing.frame"])

    dataset = PretrainQADataset(
        rows=[
            {
                "frame_path": "missing.frame",
                "frame_id": 8,
                "question_id": "q1",
                "question": "Missing?",
                "gt": "no",
            }
        ],
        frame_index_path=frame_index_path,
        trajectory_source=_FakeTrajectorySource(fail_keys={("missing.frame", 8)}),
        split_name="train",
        movement_enabled=True,
    )

    monkeypatch.setattr(dataset, "_load_frame", lambda frame_path, frame_id: torch.zeros(3, 4, 4))

    with pytest.raises(PretrainSampleError, match="Trajectory record not found"):
        dataset[0]

    stats = dataset.peek_error_stats()
    assert stats["sample_error_count"] == 1
    assert stats["error_examples"][0]["reason"] == "missing_trajectory_record"


def test_build_pretrain_datasets_returns_disjoint_grouped_splits(tmp_path):
    rows = [
        {"frame_path": "frame_a", "frame_id": 8, "question_id": "q1", "question": "q", "gt": "a"},
        {"frame_path": "frame_a", "frame_id": 8, "question_id": "q2", "question": "q", "gt": "a"},
        {"frame_path": "frame_b", "frame_id": 8, "question_id": "q3", "question": "q", "gt": "a"},
        {"frame_path": "frame_c", "frame_id": 8, "question_id": "q4", "question": "q", "gt": "a"},
    ]
    question_path = tmp_path / "question_train.jsonl"
    frame_index_path = tmp_path / "frame_index.pkl"
    _write_jsonl_with_bom(question_path, rows)
    _write_frame_index(frame_index_path, ["frame_a", "frame_b", "frame_c"])

    train_dataset, val_dataset, test_dataset, split_stats = build_pretrain_datasets(
        question_train_file=question_path,
        frame_index_path=frame_index_path,
        trajectory_source=_FakeTrajectorySource(),
        val_split_ratio=0.1,
        val_split_seed=42,
        movement_enabled=True,
    )

    train_paths = {row["frame_path"] for row in train_dataset.rows}
    val_paths = {row["frame_path"] for row in val_dataset.rows}
    assert train_paths.isdisjoint(val_paths)
    assert split_stats["train"]["row_count"] == len(train_dataset.rows)
    assert test_dataset.split_name == "test"


def test_build_pretrain_datasets_reads_fixed_split_files_and_rejects_frame_leak(tmp_path):
    train_rows = [
        {"frame_path": "frame_train", "frame_id": 8, "question_id": "q1", "question": "q", "gt": "a"},
    ]
    val_rows = [
        {"frame_path": "frame_val", "frame_id": 8, "question_id": "q1", "question": "q", "gt": "a"},
    ]
    test_rows = [
        {"frame_path": "frame_test", "frame_id": 8, "question_id": "q1", "question": "q", "gt": "a"},
    ]
    train_path = tmp_path / "train.jsonl"
    val_path = tmp_path / "val.jsonl"
    test_path = tmp_path / "test.jsonl"
    frame_index_path = tmp_path / "frame_index.pkl"
    _write_jsonl_with_bom(train_path, train_rows)
    _write_jsonl_with_bom(val_path, val_rows)
    _write_jsonl_with_bom(test_path, test_rows)
    _write_frame_index(frame_index_path, ["frame_train", "frame_val", "frame_test"])

    train_dataset, val_dataset, test_dataset, stats = build_pretrain_datasets(
        question_train_file=train_path,
        question_val_file=val_path,
        question_test_file=test_path,
        frame_index_path=frame_index_path,
        trajectory_source=_FakeTrajectorySource(),
        val_split_ratio=0.1,
        val_split_seed=42,
        movement_enabled=True,
    )

    assert train_dataset.rows == train_rows
    assert val_dataset.rows == val_rows
    assert test_dataset.rows == test_rows
    assert stats["test"]["frame_count"] == 1

    _write_jsonl_with_bom(test_path, [{"frame_path": "frame_val", "frame_id": 8, "question_id": "q1", "question": "q", "gt": "a"}])
    with pytest.raises(ValueError, match="frame_path leakage"):
        build_pretrain_datasets(
            question_train_file=train_path,
            question_val_file=val_path,
            question_test_file=test_path,
            frame_index_path=frame_index_path,
            trajectory_source=_FakeTrajectorySource(),
            val_split_ratio=0.1,
            val_split_seed=42,
            movement_enabled=True,
        )
