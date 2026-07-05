import io
import logging
import importlib.machinery
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
import yaml
from PIL import Image

import wad_dataset
if "peft" not in sys.modules:
    peft_stub = ModuleType("peft")
    peft_stub.__spec__ = importlib.machinery.ModuleSpec("peft", loader=None)
    peft_stub.LoraConfig = object
    peft_stub.PeftModel = object
    peft_stub.get_peft_model = lambda *args, **kwargs: None
    peft_stub.prepare_model_for_kbit_training = lambda model, *args, **kwargs: model
    sys.modules["peft"] = peft_stub
from train import CollaterFn
from wad_dataset import WADDatasetForInternVL, build_dataset


ALTER_779_PROMPT = (
    "Describe the scene for a visually impaired user based on the final frame.\n"
    "Focus on immediate obstacles, safe direction, and what action the user should take.\n"
    "Provide only the final spoken guidance in natural language."
)

QA_779_PROMPT = (
    "Describe the scene for a visually impaired user based on the final frame.\n"
    "Focus on obstacles, nearby people or vehicles, free walking space, direction, and safety.\n"
    "Question: current road condition\n"
    "Answer the question directly in natural language."
)

ALTER_T1_PROMPT = (
    "Provide brief navigation guidance for a visually impaired user based on this image.\n"
    "Focus on immediate obstacles, the safe direction, and the next safe action.\n"
    "Output only the final guidance in natural language."
)


def make_sample(*, frame_path="video.frame", alter="move forward", qa=None):
    return {
        "video": "clip.mp4",
        "frame_path": frame_path,
        "weather_condition": "Sunny",
        "area_type": "Pedestrian Path",
        "danger_level": "High",
        "traffic_flow_rating": "High",
        "summary": "scene summary",
        "alter": alter,
        "QA": qa,
    }


def make_dataset(
    monkeypatch,
    *,
    split="train",
    response_format="direct_text",
    prompt_mode="fixed_legacy",
    qa_prompt_mode="current_v1",
    non_train_error_policy="skip",
    samples=None,
):
    samples = samples or [make_sample()]
    dataset = WADDatasetForInternVL(
        metadata_dataset={split: samples},
        frame_index={sample["frame_path"]: {0: {"shard": "dummy.tar", "tar_path": "f.jpg"}} for sample in samples},
        bbox_by_folder={},
        split=split,
        response_format=response_format,
        direct_text_alter_prompt_mode=prompt_mode,
        direct_text_qa_prompt_mode=qa_prompt_mode,
        non_train_error_policy=non_train_error_policy,
        seed=42,
    )
    monkeypatch.setattr(dataset, "_select_frames_safe", lambda frame_path: [0])
    monkeypatch.setattr(dataset, "_load_frames", lambda frame_path, frame_ids: [Image.new("RGB", (2, 2))])
    monkeypatch.setattr(wad_dataset, "process_image", lambda img: torch.zeros((1, 3, 2, 2)))
    return dataset


def test_fixed_779_mode_keeps_original_alter_prompt(monkeypatch):
    dataset = make_dataset(monkeypatch, prompt_mode="fixed_779")

    sample = dataset[0]

    assert sample["selected_prompt_id"] == "legacy_779"
    assert sample["selected_prompt_text"] == ALTER_779_PROMPT
    assert sample["question"] == f"<image>\n{ALTER_779_PROMPT}"
    assert sample["qformer_text"] == ALTER_779_PROMPT


def test_fixed_v1_mode_uses_t1_for_alter(monkeypatch):
    dataset = make_dataset(monkeypatch, prompt_mode="fixed_v1")

    sample = dataset[0]

    assert sample["selected_prompt_id"] == "T1"
    assert sample["selected_prompt_text"] == ALTER_T1_PROMPT
    assert sample["question"] == f"<image>\n{ALTER_T1_PROMPT}"
    assert sample["qformer_text"] == ALTER_T1_PROMPT


def test_balanced_v1_train_assignment_is_balanced_and_reproducible(monkeypatch):
    samples = [make_sample(frame_path=f"video_{idx}.frame", alter=f"alter {idx}") for idx in range(9)]
    dataset = make_dataset(monkeypatch, prompt_mode="balanced_v1", samples=samples)

    dataset.set_epoch(0)
    first_assignment = dict(dataset.prompt_assignment)
    counts = {}
    for prompt_id in first_assignment.values():
        counts[prompt_id] = counts.get(prompt_id, 0) + 1

    assert set(first_assignment.values()) == {"T1", "T2", "T3", "T4"}
    assert max(counts.values()) - min(counts.values()) <= 1

    dataset.set_epoch(0)
    assert dataset.prompt_assignment == first_assignment

    dataset.set_epoch(1)
    assert dataset.prompt_assignment != first_assignment


def test_balanced_v1_val_and_test_use_fixed_t1(monkeypatch):
    val_dataset = make_dataset(monkeypatch, split="val", prompt_mode="balanced_v1")
    test_dataset = make_dataset(monkeypatch, split="test", prompt_mode="balanced_v1")

    val_sample = val_dataset[0]
    test_sample = test_dataset[0]

    assert val_sample["selected_prompt_id"] == "T1"
    assert val_sample["selected_prompt_text"] == ALTER_T1_PROMPT
    assert test_sample["selected_prompt_id"] == "T1"
    assert test_sample["selected_prompt_text"] == ALTER_T1_PROMPT


def test_qa_direct_text_prompt_stays_unchanged(monkeypatch):
    qa_dataset = make_dataset(
        monkeypatch,
        samples=[make_sample(qa={"Q": "current road condition", "A": "clear"})],
    )

    sample = qa_dataset[0]

    assert sample["task_type"] == "qa"
    assert sample["selected_prompt_id"] == "qa_default"
    assert sample["qformer_text"] == (
        "Describe the scene for a visually impaired user based on this frame.\n"
        "Focus on obstacles, nearby people or vehicles, free walking space, direction, and safety.\n"
        "Question: current road condition"
    )


def test_qa_legacy_779_mode_restores_old_prompt_suffix(monkeypatch):
    qa_dataset = make_dataset(
        monkeypatch,
        samples=[make_sample(qa={"Q": "current road condition", "A": "clear"})],
        qa_prompt_mode="legacy_779",
    )

    sample = qa_dataset[0]

    assert sample["selected_prompt_id"] == "qa_legacy_779"
    assert sample["qformer_text"] == QA_779_PROMPT
    assert sample["question"] == f"<image>\n{QA_779_PROMPT}"


def test_structured_json_path_is_unchanged(monkeypatch):
    dataset = make_dataset(monkeypatch, response_format="structured_json")

    sample = dataset[0]

    assert "Analyze: location, weather, traffic, scene" in sample["question"]
    assert sample["task_type"] == "alter"
    assert sample["selected_prompt_id"] == "structured_json"


def test_alter_direct_text_sample_exposes_prompt_debug_fields(monkeypatch):
    dataset = make_dataset(monkeypatch, prompt_mode="balanced_v1")
    dataset.set_epoch(0)

    sample = dataset[0]

    assert sample["task_type"] == "alter"
    assert sample["selected_prompt_id"] in {"T1", "T2", "T3", "T4"}
    assert sample["selected_prompt_text"]
    assert sample["frame_path"] == "video.frame"


def test_val_skip_policy_returns_none_on_data_error(monkeypatch):
    dataset = make_dataset(monkeypatch, split="val", non_train_error_policy="skip")
    monkeypatch.setattr(dataset, "_load_frames", lambda frame_path, frame_ids: (_ for _ in ()).throw(ValueError("boom")))

    sample = dataset[0]

    assert sample is None


def test_val_resample_policy_recovers_from_data_error(monkeypatch):
    samples = [
        make_sample(frame_path="bad.frame", alter="bad sample"),
        make_sample(frame_path="good.frame", alter="good sample"),
    ]
    dataset = make_dataset(
        monkeypatch,
        split="val",
        samples=samples,
        non_train_error_policy="resample",
    )
    calls = {"count": 0}

    def flaky_load(frame_path, frame_ids):
        calls["count"] += 1
        if frame_path == "bad.frame":
            raise ValueError("boom")
        return [Image.new("RGB", (2, 2))]

    monkeypatch.setattr(dataset, "_load_frames", flaky_load)
    monkeypatch.setattr(wad_dataset.random, "randint", lambda a, b: 1)

    sample = dataset[0]

    assert calls["count"] >= 2
    assert sample is not None
    assert sample["frame_path"] == "good.frame"


def test_build_dataset_returns_separate_train_and_val_datasets(monkeypatch):
    raw_samples = [
        make_sample(frame_path=f"frame_{idx}.frame", alter=f"alter {idx}")
        for idx in range(10)
    ]

    def fake_load_dataset(name, data_files=None, split=None):
        if split == "train":
            return []
        if isinstance(data_files, dict):
            return {"train": raw_samples}
        return []

    def fake_exists(path):
        return str(path).endswith("frame_index.pkl")

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    monkeypatch.setattr(wad_dataset.os.path, "exists", fake_exists, raising=False)
    monkeypatch.setattr(wad_dataset.pickle, "load", lambda f: {sample["frame_path"]: {0: {"shard": "dummy", "tar_path": "f.jpg"}} for sample in raw_samples})
    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: io.BytesIO(b"dummy"))

    config = yaml.safe_load(
        """
model:
  architecture: internvl
  vision:
    image_size: [448, 448]
data:
  name: dummy
  train_split: 0.8
  seed: 42
  train_task_filter: alter_only
  val_task_filter: alter_only
  eval_limit: 0
  response_format: direct_text
  direct_text_alter_prompt_mode: balanced_v1
"""
    )

    train_dataset, val_dataset = build_dataset(config)

    assert train_dataset.split == "train"
    assert val_dataset.split == "val"
    assert train_dataset.response_format == "direct_text"
    assert val_dataset.response_format == "direct_text"


class DummyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))

    def convert_tokens_to_ids(self, token):
        return 1


class DummyTemplate:
    sep = "<sep>"
    roles = ("user", "assistant")
    system_message = ""

    def __init__(self):
        self.messages = []

    def append_message(self, role, content):
        self.messages.append((role, content))

    def get_prompt(self):
        return "\n".join(content or "" for _, content in self.messages)


class DummyModel:
    template = "dummy"
    system_message = "system"
    num_image_token = 32
    qformer_enabled = True

    def encode_qformer_texts(self, texts):
        return (torch.zeros((len(texts), 1), dtype=torch.long), torch.ones((len(texts), 1), dtype=torch.long))


def test_collate_logs_prompt_samples_for_train_only(monkeypatch):
    monkeypatch.setattr("train.get_conv_template", lambda _: DummyTemplate())
    captured_logs = []
    monkeypatch.setattr("train.logger.info", lambda msg, *args: captured_logs.append(msg % args if args else msg))
    collate = CollaterFn(DummyTokenizer(), DummyModel())
    collate.log_prompt_samples = True
    collate.prompt_log_remaining = 1

    batch = [
        {
            "question": f"<image>\n{ALTER_T1_PROMPT}",
            "answer": "move forward",
            "pixel_values": [torch.zeros((1, 3, 2, 2))],
            "qformer_text": ALTER_T1_PROMPT,
            "questionId": "0",
            "task_type": "alter",
            "selected_prompt_id": "T1",
            "selected_prompt_text": ALTER_T1_PROMPT,
            "frame_path": "video.frame",
        }
    ]

    collate(batch)

    joined_logs = "\n".join(captured_logs)
    assert "[PROMPT SAMPLE]" in joined_logs
    assert "selected_prompt_id=T1" in joined_logs
    assert "frame_path=video.frame" in joined_logs


def test_collate_logs_token_stats_with_limited_budget(monkeypatch):
    monkeypatch.setattr("train.get_conv_template", lambda _: DummyTemplate())
    captured_logs = []
    monkeypatch.setattr("train.logger.info", lambda msg, *args: captured_logs.append(msg % args if args else msg))
    collate = CollaterFn(DummyTokenizer(), DummyModel())
    collate.log_token_stats = True
    collate.token_log_remaining = 1

    batch = [
        {
            "question": f"<image>\n{ALTER_T1_PROMPT}",
            "answer": "move forward",
            "pixel_values": [torch.zeros((1, 3, 2, 2))],
            "qformer_text": ALTER_T1_PROMPT,
            "questionId": "0",
            "task_type": "alter",
            "selected_prompt_id": "T1",
            "selected_prompt_text": ALTER_T1_PROMPT,
            "frame_path": "video.frame",
        }
    ]

    collate(batch)

    joined_logs = "\n".join(captured_logs)
    assert "[INFO] Image token stats" in joined_logs
    assert "[INFO] Text tokens - input:" in joined_logs
    assert collate.token_log_remaining == 0
