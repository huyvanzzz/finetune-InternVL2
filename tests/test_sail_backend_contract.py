import importlib.machinery
import sys
from pathlib import Path
from types import ModuleType

import torch
import yaml
from PIL import Image

import wad_dataset
from wad_dataset import WADDatasetForInternVL


if "peft" not in sys.modules:
    peft_stub = ModuleType("peft")
    peft_stub.__spec__ = importlib.machinery.ModuleSpec("peft", loader=None)
    peft_stub.LoraConfig = object
    peft_stub.PeftModel = object
    peft_stub.get_peft_model = lambda *args, **kwargs: None
    peft_stub.prepare_model_for_kbit_training = lambda model, *args, **kwargs: model
    sys.modules["peft"] = peft_stub


def make_alter_sample(*, frame_path="video.frame", alter="move forward"):
    return {
        "video": "clip.mp4",
        "frame_path": frame_path,
        "weather_condition": "Sunny",
        "area_type": "Pedestrian Path",
        "danger_level": "High",
        "traffic_flow_rating": "High",
        "summary": "scene summary",
        "alter": alter,
    }


def make_dataset(monkeypatch, *, split="train", samples=None):
    samples = samples or [make_alter_sample()]
    dataset = WADDatasetForInternVL(
        metadata_dataset={split: samples},
        frame_index={sample["frame_path"]: {0: {"shard": "dummy.tar", "tar_path": "f.jpg"}} for sample in samples},
        bbox_by_folder={},
        split=split,
        response_format="direct_text",
        direct_text_alter_prompt_mode="balanced_v1",
        seed=42,
    )
    monkeypatch.setattr(dataset, "_select_frames_safe", lambda frame_path: [0])
    monkeypatch.setattr(dataset, "_load_frames", lambda frame_path, frame_ids: [Image.new("RGB", (2, 2))])
    monkeypatch.setattr(wad_dataset, "process_image", lambda img: torch.zeros((1, 3, 2, 2)))
    return dataset


def test_dataset_sample_contains_both_image_and_pixel_values_for_backend_compatibility(monkeypatch):
    dataset = make_dataset(monkeypatch)
    sample = dataset[0]

    assert "image" in sample
    assert "pixel_values" in sample
    assert isinstance(sample["image"], list)
    assert len(sample["image"]) == 1
    assert sample["pixel_values"][0].shape == (1, 3, 2, 2)


def test_sail_config_files_exist_and_use_sail_architecture():
    for config_path, qformer_enabled in (
        ("sailvl_config.yaml", True),
        ("sailvl_config_no_qformer.yaml", False),
    ):
        path = Path(config_path)
        assert path.exists(), f"Missing config file: {config_path}"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert config["model"]["architecture"] == "sailvl"
        assert config["data"]["response_format"] == "direct_text"
        assert config["data"]["train_task_filter"] == "alter_only"
        assert config["data"]["val_task_filter"] == "alter_only"
        assert config["data"]["direct_text_alter_prompt_mode"] == "balanced_v1"
        assert config["model"]["qformer"]["enabled"] is qformer_enabled


def test_backend_registry_resolves_internvl_and_sailvl():
    from model_backends.base import get_backend

    internvl_backend = get_backend("internvl")
    sailvl_backend = get_backend("sailvl")

    assert internvl_backend.name == "internvl"
    assert sailvl_backend.name == "sailvl"


def test_sail_backend_exposes_required_contract():
    from model_backends.base import get_backend

    sailvl_backend = get_backend("sailvl")

    for attr in (
        "load_model_and_tokenizer",
        "build_train_collate_fn",
        "build_eval_collate_fn",
        "forward_train_batch",
        "forward_eval_batch",
        "generate_response",
        "attach_qformer_if_enabled",
        "save_backend_artifacts",
        "load_backend_artifacts",
    ):
        assert hasattr(sailvl_backend, attr), f"Missing backend API: {attr}"


def test_sail_qformer_bridge_contract_uses_shared_artifact_names():
    from model_backends.sailvl import qformer_bridge as sail_bridge

    assert sail_bridge.BRIDGE_WEIGHTS_NAME == "qformer_bridge.safetensors"
    assert sail_bridge.BRIDGE_CONFIG_NAME == "qformer_bridge_config.json"


def test_sail_preprocess_defaults_follow_native_contract():
    from model_backends.sailvl.preprocess import get_sail_preprocess_config

    config = {
        "model": {
            "vision": {
                "force_image_size": 448,
                "use_thumbnail": True,
                "min_dynamic_patch": 1,
                "max_dynamic_patch": 12,
            }
        }
    }

    preprocess_cfg = get_sail_preprocess_config(config)

    assert preprocess_cfg["image_size"] == 448
    assert preprocess_cfg["use_thumbnail"] is True
    assert preprocess_cfg["min_dynamic_patch"] == 1
    assert preprocess_cfg["max_dynamic_patch"] == 12


def test_sail_configs_use_native_dynamic_patch_budget():
    for config_path in ("sailvl_config.yaml", "sailvl_config_no_qformer.yaml"):
        config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        assert config["model"]["vision"]["max_dynamic_patch"] == 12


def test_sail_no_qformer_config_does_not_define_projection_learning_rate():
    config = yaml.safe_load(Path("sailvl_config_no_qformer.yaml").read_text(encoding="utf-8"))
    assert "proj_learning_rate" not in config["training"]


def test_sail_lora_targets_follow_qwen2_naming():
    from model_backends.sailvl.lora import DEFAULT_SAIL_LORA_TARGET_MODULES

    assert DEFAULT_SAIL_LORA_TARGET_MODULES == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]


def test_sail_configs_do_not_require_internvl_image_size_key():
    for config_path in ("sailvl_config.yaml", "sailvl_config_no_qformer.yaml"):
        config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        assert "image_size" not in config["model"]["vision"]
        assert config["model"]["vision"]["force_image_size"] == 448


def test_sail_mixed_configs_enable_verify_friendly_logging_defaults():
    for config_path in ("sailvl_config_mixed.yaml", "sailvl_config_no_qformer_mixed.yaml"):
        config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        assert config["training"]["log_run_summary"] is True
        assert config["training"]["log_data_summary"] is True
        assert config["training"]["log_token_stats"] is True
        assert config["training"]["log_prompt_samples"] is True
        assert config["training"]["prompt_log_batches"] == 2
        assert config["evaluation"]["print_samples"] == 3


def test_sail_collate_supports_verify_logs_with_limited_budget(monkeypatch):
    from model_backends.sailvl.runtime import SailCollateFn

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
        conv_template = None

        def encode_qformer_texts(self, texts):
            return (torch.zeros((len(texts), 1), dtype=torch.long), torch.ones((len(texts), 1), dtype=torch.long))

    monkeypatch.setattr("model_backends.sailvl.runtime.get_conv_template", lambda _: DummyTemplate())
    monkeypatch.setattr("model_backends.sailvl.runtime.preprocess_sail_image", lambda image, config: torch.zeros((3, 3, 2, 2)))
    captured_logs = []
    monkeypatch.setattr("model_backends.sailvl.runtime._log_info", lambda msg, *args: captured_logs.append(msg % args if args else msg))

    collate = SailCollateFn(
        DummyTokenizer(),
        DummyModel(),
        {"model": {"vision": {"force_image_size": 448, "use_thumbnail": True, "min_dynamic_patch": 1, "max_dynamic_patch": 12}}},
    )
    collate.log_token_stats = True
    collate.token_log_remaining = 1
    collate.log_prompt_samples = True
    collate.prompt_log_remaining = 1

    sample = {
        "question": "<image>\nDescribe the scene for a visually impaired user based on this image.",
        "answer": "move forward",
        "image": [Image.new("RGB", (2, 2))],
        "qformer_text": "Describe the scene for a visually impaired user based on this image.",
        "questionId": "0",
        "task_type": "alter",
        "selected_prompt_id": "legacy",
        "selected_prompt_text": "Describe the scene for a visually impaired user based on this image.",
        "frame_path": "video.frame",
    }

    collate([sample])

    joined_logs = "\n".join(captured_logs)
    assert "[INFO] Image token stats" in joined_logs
    assert "[PROMPT SAMPLE]" in joined_logs
    assert collate.token_log_remaining == 0
    assert collate.prompt_log_remaining == 0
