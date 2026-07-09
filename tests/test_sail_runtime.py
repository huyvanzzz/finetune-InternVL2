import torch
import json
from pathlib import Path


class DummyTemplate:
    def __init__(self):
        self.roles = ("user", "assistant")
        self.sep = "<sep>"
        self.system_message = ""
        self.messages = []

    def append_message(self, role, message):
        self.messages.append((role, message))

    def get_prompt(self):
        user_message = self.messages[0][1]
        return f"{self.system_message}\n{user_message}\nassistant:"


class DummyTokenizer:
    def convert_tokens_to_ids(self, token):
        return 0

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))


class DummyModel:
    def __init__(self):
        self.template = "sailvl-chat"
        self.system_message = "system"
        self.num_image_token = 32
        self.qformer_enabled = False


def test_sail_train_collate_preprocesses_from_raw_images(monkeypatch):
    from model_backends.sailvl.runtime import build_train_collate_fn

    monkeypatch.setattr("model_backends.sailvl.runtime.get_conv_template", lambda _: DummyTemplate())
    monkeypatch.setattr(
        "model_backends.sailvl.runtime.preprocess_sail_image",
        lambda image, config: torch.zeros((3, 3, 2, 2)),
    )

    collate = build_train_collate_fn(
        DummyTokenizer(),
        DummyModel(),
        {"model": {"vision": {"force_image_size": 448}}},
    )

    batch = [
        {
            "question": "<image>\nDescribe the scene",
            "answer": "move forward carefully",
            "qformer_text": "Describe the scene",
            "image": [object()],
            "task_type": "alter",
            "selected_prompt_id": "legacy_779",
        }
    ]

    _, _, _, pixel_values, qformer_inputs, samples = collate(batch)

    assert tuple(pixel_values.shape) == (3, 3, 2, 2)
    assert qformer_inputs is None
    assert len(samples) == 1


def test_sail_train_collate_encodes_qformer_text_when_enabled(monkeypatch):
    from model_backends.sailvl.runtime import build_train_collate_fn

    class QFormerDummyModel(DummyModel):
        def __init__(self):
            super().__init__()
            self.qformer_enabled = True
            self.last_qformer_texts = None

        def encode_qformer_texts(self, texts):
            self.last_qformer_texts = list(texts)
            return torch.tensor([[1, 2, 3]]), torch.tensor([[1, 1, 1]])

    model = QFormerDummyModel()
    monkeypatch.setattr("model_backends.sailvl.runtime.get_conv_template", lambda _: DummyTemplate())
    monkeypatch.setattr(
        "model_backends.sailvl.runtime.preprocess_sail_image",
        lambda image, config: torch.zeros((2, 3, 2, 2)),
    )

    collate = build_train_collate_fn(
        DummyTokenizer(),
        model,
        {"model": {"vision": {"force_image_size": 448}}},
    )

    batch = [
        {
            "question": "<image>\nDescribe the scene",
            "answer": "move forward carefully",
            "qformer_text": "Describe the scene",
            "image": [object()],
            "task_type": "alter",
            "selected_prompt_id": "legacy_779",
        }
    ]

    _, _, _, pixel_values, qformer_inputs, _ = collate(batch)

    assert tuple(pixel_values.shape) == (2, 3, 2, 2)
    assert qformer_inputs is not None
    assert model.last_qformer_texts == ["Describe the scene", "Describe the scene"]


def test_attach_qformer_if_enabled_delegates_to_sail_bridge(monkeypatch):
    from model_backends.sailvl.runtime import attach_qformer_if_enabled

    called = {}

    def fake_attach(model, config, logger=None):
        called["model"] = model
        called["config"] = config
        return "attached"

    monkeypatch.setattr("model_backends.sailvl.runtime.attach_sail_qformer_bridge", fake_attach)

    result = attach_qformer_if_enabled(object(), {"model": {"qformer": {"enabled": True}}})

    assert result == "attached"
    assert "config" in called


def test_save_backend_artifacts_writes_sail_bridge_metadata(tmp_path, monkeypatch):
    from model_backends.sailvl.runtime import save_backend_artifacts

    class DummyLanguageModel:
        def save_pretrained(self, output_dir):
            (tmp_path / "adapter_model.bin").write_text("ok", encoding="utf-8")

    class DummyTokenizerSave:
        def save_pretrained(self, output_dir):
            (tmp_path / "tokenizer.json").write_text("ok", encoding="utf-8")

    class DummySaveModel:
        def __init__(self):
            self.language_model = DummyLanguageModel()
            self.qformer_enabled = True
            self.peft_base_model_name_or_path = "Qwen/Qwen2.5-1.5B-Instruct"

    def fake_save_bridge(model, output_dir):
        (tmp_path / "qformer_bridge.safetensors").write_text("ok", encoding="utf-8")

    monkeypatch.setattr("model_backends.sailvl.runtime.save_sail_qformer_bridge", fake_save_bridge)
    monkeypatch.setattr("model_backends.sailvl.runtime.sanitize_peft_checkpoint_metadata", lambda *args, **kwargs: None)

    save_backend_artifacts(DummySaveModel(), DummyTokenizerSave(), str(tmp_path))

    metadata = json.loads((tmp_path / "qformer_bridge_config.json").read_text(encoding="utf-8"))
    assert metadata["bridge_backend"] == "sailvl"


def test_load_model_and_tokenizer_builds_sail_config_without_autoconfig(tmp_path, monkeypatch):
    from model_backends.sailvl.runtime import load_model_and_tokenizer

    config_json = {
        "architectures": ["SailVLModel"],
        "model_type": "internvl_chat",
        "llm_config": {
            "architectures": ["Qwen2ForCausalLM"],
            "model_type": "qwen2",
        },
        "vision_config": {},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config_json), encoding="utf-8")

    class DummySailConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    created = {}

    def fake_get_class_from_dynamic_module(class_ref, pretrained_model_name_or_path, **kwargs):
        created["class_ref"] = class_ref
        created["repo"] = pretrained_model_name_or_path
        return DummySailConfig

    def fake_hf_hub_download(repo_id, filename):
        created["download_repo"] = repo_id
        created["download_filename"] = filename
        return str(config_path)

    class DummyAutoModel:
        def __init__(self):
            self.img_context_token_id = None
            self.system_message = None

    model_calls = {}

    def fake_model_from_pretrained(model_name_or_path, **kwargs):
        model_calls["model_name_or_path"] = model_name_or_path
        model_calls["kwargs"] = kwargs
        return DummyAutoModel()

    class DummyTokenizerLoad:
        def convert_tokens_to_ids(self, token):
            return 123

    tokenizer_calls = {}

    def fake_tokenizer_from_pretrained(model_name_or_path, **kwargs):
        tokenizer_calls["model_name_or_path"] = model_name_or_path
        tokenizer_calls["kwargs"] = kwargs
        return DummyTokenizerLoad()

    class DummyBitsAndBytesConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr("model_backends.sailvl.runtime.hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr("model_backends.sailvl.runtime.get_class_from_dynamic_module", fake_get_class_from_dynamic_module)
    monkeypatch.setattr("model_backends.sailvl.runtime.AutoModel.from_pretrained", fake_model_from_pretrained)
    monkeypatch.setattr("model_backends.sailvl.runtime.AutoTokenizer.from_pretrained", fake_tokenizer_from_pretrained)
    monkeypatch.setattr("model_backends.sailvl.runtime.BitsAndBytesConfig", DummyBitsAndBytesConfig)
    monkeypatch.setattr("model_backends.sailvl.runtime.wrap_input_embeddings_for_safe_scatter", lambda model: None)
    monkeypatch.setattr("model_backends.sailvl.runtime.patch_sail_forward_runtime", lambda model: None)

    model, tokenizer = load_model_and_tokenizer(
        {
            "model": {
                "name": "BytedanceDouyinContent/SAIL-VL-1d5-2B",
                "trust_remote_code": True,
                "quantization": {
                    "enabled": True,
                    "compute_dtype": "bfloat16",
                    "double_quant": True,
                    "type": "nf4",
                },
            }
        }
    )

    assert created["download_repo"] == "BytedanceDouyinContent/SAIL-VL-1d5-2B"
    assert created["class_ref"] == "configuration_sailvl.SailVLConfig"
    assert isinstance(model_calls["kwargs"]["config"], DummySailConfig)
    assert model_calls["kwargs"]["config"].kwargs["llm_config"]["architectures"] == ["Qwen2ForCausalLM"]
    assert tokenizer_calls["model_name_or_path"] == "BytedanceDouyinContent/SAIL-VL-1d5-2B"
    assert model.img_context_token_id == 123
    assert tokenizer is not None
