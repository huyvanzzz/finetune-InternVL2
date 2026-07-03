import io
import importlib.machinery
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

import wad_dataset
if "peft" not in sys.modules:
    peft_stub = ModuleType("peft")
    peft_stub.__spec__ = importlib.machinery.ModuleSpec("peft", loader=None)
    peft_stub.LoraConfig = object
    peft_stub.PeftModel = object
    peft_stub.get_peft_model = lambda *args, **kwargs: None
    peft_stub.prepare_model_for_kbit_training = lambda model, *args, **kwargs: model
    sys.modules["peft"] = peft_stub
from train import build_train_sampler
from wad_dataset import build_dataset


def make_sample(*, frame_path, qa=None, alter=None):
    sample = {
        "video": "clip.mp4",
        "frame_path": frame_path,
        "weather_condition": "Sunny",
        "area_type": "Pedestrian Path",
        "danger_level": "High",
        "traffic_flow_rating": "High",
        "summary": "scene summary",
    }
    if qa is not None:
        sample["QA"] = qa
    if alter is not None:
        sample["alter"] = alter
    return sample


def mixed_raw_samples():
    return [
        make_sample(frame_path="qa_0.frame", qa={"Q": "where is the obstacle", "A": "left side"}, alter="unused"),
        make_sample(frame_path="alter_0.frame", alter="move forward carefully"),
        make_sample(frame_path="qa_1.frame", qa={"Q": "current road condition", "A": "clear"}, alter="unused"),
        make_sample(frame_path="alter_1.frame", alter="slow down and keep left"),
        make_sample(frame_path="alter_2.frame", alter="turn slightly right"),
        make_sample(frame_path="qa_2.frame", qa={"Q": "which direction is safe", "A": "11 o'clock"}, alter="unused"),
    ]


def test_mixed_configs_exist_and_disable_alter_only_features():
    expected = {
        "internvl_config_mixed.yaml": ("internvl", True),
        "internvl_config_no_qformer_mixed.yaml": ("internvl", False),
        "sailvl_config_mixed.yaml": ("sailvl", True),
        "sailvl_config_no_qformer_mixed.yaml": ("sailvl", False),
    }

    for config_name, (architecture, qformer_enabled) in expected.items():
        path = Path(config_name)
        assert path.exists(), f"Missing mixed config: {config_name}"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert config["model"]["architecture"] == architecture
        assert config["model"]["qformer"]["enabled"] is qformer_enabled
        assert config["data"]["train_task_filter"] == "all"
        assert config["data"]["val_task_filter"] == "all"
        assert config["data"]["direct_text_alter_prompt_mode"] == "fixed_legacy"
        assert config["data"]["stratify_split"] is False
        assert config["training"]["task_balancing"]["enabled"] is False


def test_mixed_configs_use_separate_output_dirs_from_alter_only_configs():
    alter_only = yaml.safe_load(Path("internvl_config.yaml").read_text(encoding="utf-8"))
    mixed = yaml.safe_load(Path("internvl_config_mixed.yaml").read_text(encoding="utf-8"))

    assert alter_only["training"]["output_dir"] != mixed["training"]["output_dir"]


def test_build_dataset_mixed_config_uses_plain_split_without_stratify(monkeypatch):
    raw_samples = mixed_raw_samples()
    captured_kwargs = {}

    def fake_load_dataset(name, data_files=None, split=None):
        if split == "train":
            return []
        if isinstance(data_files, dict):
            return {"train": raw_samples}
        return []

    def fake_train_test_split(indices, **kwargs):
        captured_kwargs.update(kwargs)
        return indices[:4], indices[4:]

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    monkeypatch.setattr(wad_dataset, "train_test_split", fake_train_test_split)
    monkeypatch.setattr(wad_dataset.os.path, "exists", lambda path: str(path).endswith("frame_index.pkl"), raising=False)
    monkeypatch.setattr(
        wad_dataset.pickle,
        "load",
        lambda f: {sample["frame_path"]: {0: {"shard": "dummy", "tar_path": "f.jpg"}} for sample in raw_samples},
    )
    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: io.BytesIO(b"dummy"))

    config = yaml.safe_load(Path("internvl_config_mixed.yaml").read_text(encoding="utf-8"))
    config["data"]["train_limit"] = 0
    config["data"]["eval_limit"] = 0

    train_dataset, val_dataset = build_dataset(config)

    assert "stratify" not in captured_kwargs
    assert train_dataset.split == "train"
    assert val_dataset.split == "val"
    assert {"qa", "alter"} <= set(train_dataset.task_types)
    assert {"qa", "alter"} & set(val_dataset.task_types)


def test_mixed_legacy_prompts_match_f8692d7_contract(monkeypatch):
    samples = mixed_raw_samples()
    dataset = wad_dataset.WADDatasetForInternVL(
        metadata_dataset={"train": samples},
        frame_index={sample["frame_path"]: {0: {"shard": "dummy.tar", "tar_path": "f.jpg"}} for sample in samples},
        bbox_by_folder={},
        split="train",
        response_format="direct_text",
        direct_text_alter_prompt_mode="fixed_legacy",
        seed=42,
    )
    monkeypatch.setattr(dataset, "_select_frames_safe", lambda frame_path: [0])
    monkeypatch.setattr(dataset, "_load_frames", lambda frame_path, frame_ids: [])

    qa_sample = dataset[0]
    alter_sample = dataset[1]

    assert qa_sample["task_type"] == "qa"
    assert qa_sample["selected_prompt_id"] == "qa_default"
    assert qa_sample["qformer_text"] == (
        "Based on this image, answer the following question for a visually impaired user directly in natural language.\n"
        "Question: where is the obstacle"
    )

    assert alter_sample["task_type"] == "alter"
    assert alter_sample["selected_prompt_id"] == "legacy"
    assert alter_sample["qformer_text"] == wad_dataset.ALTER_DIRECT_TEXT_LEGACY_PROMPT


def test_mixed_mode_sampler_is_disabled_and_uses_plain_shuffle():
    train_dataset = type(
        "DummyDataset",
        (),
        {"task_types": ["qa", "alter", "qa", "alter"]},
    )()
    config = yaml.safe_load(Path("internvl_config_mixed.yaml").read_text(encoding="utf-8"))

    sampler = build_train_sampler(train_dataset, config)

    assert sampler is None
