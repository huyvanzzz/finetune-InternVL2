import importlib.machinery
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


if "peft" not in sys.modules:
    peft_stub = ModuleType("peft")
    peft_stub.__spec__ = importlib.machinery.ModuleSpec("peft", loader=None)
    peft_stub.LoraConfig = object
    peft_stub.PeftModel = object
    peft_stub.get_peft_model = lambda *args, **kwargs: None
    peft_stub.prepare_model_for_kbit_training = lambda model, *args, **kwargs: model
    sys.modules["peft"] = peft_stub


class DummyTokenizer:
    def __init__(self):
        self.saved_to = None

    def encode(self, text, add_special_tokens=False):
        return list(range(max(len(text.split()), 1)))

    def convert_tokens_to_ids(self, token):
        return 1

    def save_pretrained(self, output_dir):
        self.saved_to = output_dir


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


class DummyLanguageModel:
    def __init__(self):
        self.saved_to = None

    def save_pretrained(self, output_dir):
        self.saved_to = output_dir

    def get_input_embeddings(self):
        return torch.nn.Embedding(32, 16)

    def get_output_embeddings(self):
        return torch.nn.Embedding(32, 16)


class DummyModel:
    def __init__(self):
        self.template = "dummy"
        self.system_message = "system"
        self.num_image_token = 32
        self.qformer_enabled = False
        self.language_model = DummyLanguageModel()

    def generate(self, **kwargs):
        return torch.tensor([[1, 2, 3]])


def test_sail_train_collate_preprocesses_from_raw_images(monkeypatch):
    from model_backends.sailvl.runtime import build_train_collate_fn

    dummy_model = DummyModel()
    dummy_tokenizer = DummyTokenizer()
    monkeypatch.setattr("model_backends.sailvl.runtime.get_conv_template", lambda _: DummyTemplate())
    monkeypatch.setattr(
        "model_backends.sailvl.runtime.preprocess_sail_image",
        lambda image, config: torch.zeros((2, 3, 2, 2)),
    )

    collate = build_train_collate_fn(dummy_tokenizer, dummy_model, {"model": {"vision": {}}})
    batch = [
        {
            "question": "<image>\nDescribe the scene.",
            "answer": "move forward",
            "qformer_text": "Describe the scene.",
            "image": [object()],
            "task_type": "alter",
            "selected_prompt_id": "T1",
            "selected_prompt_text": "Describe the scene.",
            "frame_path": "video.frame",
            "questionId": "1",
        }
    ]

    _, _, _, pixel_values, qformer_inputs, samples = collate(batch)

    assert pixel_values.shape == (2, 3, 2, 2)
    assert qformer_inputs is None
    assert samples[0]["questionId"] == "1"


def test_sail_attach_qformer_if_enabled_calls_bridge(monkeypatch):
    from model_backends.sailvl.runtime import attach_qformer_if_enabled

    model = DummyModel()
    called = {}

    monkeypatch.setattr(
        "model_backends.sailvl.runtime.attach_sail_qformer_bridge",
        lambda model, config, logger=None: called.setdefault("attached", True) or model,
    )

    config = {"model": {"qformer": {"enabled": True}}}
    attach_qformer_if_enabled(model, config)

    assert called["attached"] is True


def test_sail_save_backend_artifacts_saves_bridge_with_backend_marker(tmp_path, monkeypatch):
    from model_backends.sailvl.runtime import save_backend_artifacts

    model = DummyModel()
    tokenizer = DummyTokenizer()
    model.qformer_enabled = True

    monkeypatch.setattr(
        "model_backends.sailvl.runtime.save_sail_qformer_bridge",
        lambda model, output_dir: (tmp_path / "qformer_bridge.safetensors").write_bytes(b"bridge"),
    )

    save_backend_artifacts(model, tokenizer, str(tmp_path))

    assert model.language_model.saved_to == str(tmp_path)
    assert tokenizer.saved_to == str(tmp_path)
    metadata = json.loads((tmp_path / "qformer_bridge_config.json").read_text(encoding="utf-8"))
    assert metadata["bridge_backend"] == "sailvl"


def test_sail_load_backend_artifacts_rejects_foreign_bridge_backend(tmp_path):
    from model_backends.sailvl.runtime import load_backend_artifacts

    (tmp_path / "qformer_bridge_config.json").write_text(
        json.dumps({"bridge_backend": "internvl"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Expected sailvl bridge backend"):
        load_backend_artifacts(DummyModel(), str(tmp_path), {"model": {"qformer": {"enabled": True}}})


def test_sail_forward_train_batch_tolerates_uninitialized_distributed(monkeypatch):
    from model_backends.sailvl.runtime import forward_train_batch

    class DummyDistributed:
        def __init__(self):
            self.calls = []

        def is_available(self):
            self.calls.append("is_available")
            return True

        def is_initialized(self):
            self.calls.append("is_initialized")
            return False

        def get_rank(self):
            raise AssertionError("get_rank should not be called when distributed is not initialized")

    class DummyTrainModel:
        def __init__(self):
            self.qformer_enabled = False
            self.calls = 0

        def __call__(self, **kwargs):
            self.calls += 1
            import torch.distributed as dist

            rank = dist.get_rank()
            return {"rank": rank, "pixel_values_shape": tuple(kwargs["pixel_values"].shape)}

    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self, raising=False)
    dummy_dist = DummyDistributed()
    monkeypatch.setattr(torch, "distributed", dummy_dist)

    batch = (
        torch.zeros((1, 4), dtype=torch.long),
        torch.zeros((1, 4), dtype=torch.long),
        None,
        torch.zeros((2, 3, 2, 2), dtype=torch.float32),
        None,
        [],
    )

    outputs = forward_train_batch(DummyTrainModel(), batch, config={})

    assert outputs["rank"] == 0
    assert outputs["pixel_values_shape"] == (2, 3, 2, 2)
    assert dummy_dist.calls == ["is_available", "is_initialized"]


def test_wrap_input_embeddings_for_safe_scatter_returns_non_leaf_outputs():
    from model_backends.sailvl.runtime import wrap_input_embeddings_for_safe_scatter

    class LanguageModel:
        def __init__(self):
            self.embedding = torch.nn.Embedding(8, 4)

        def get_input_embeddings(self):
            return self.embedding

    model = DummyModel()
    model.language_model = LanguageModel()

    wrap_input_embeddings_for_safe_scatter(model)

    embeddings = model.language_model.get_input_embeddings()
    outputs = embeddings(torch.tensor([[1, 2]], dtype=torch.long))

    assert outputs.requires_grad is True
    assert outputs.is_leaf is False


def test_wrap_input_embeddings_for_safe_scatter_is_idempotent():
    from model_backends.sailvl.runtime import wrap_input_embeddings_for_safe_scatter

    class LanguageModel:
        def __init__(self):
            self.embedding = torch.nn.Embedding(8, 4)

        def get_input_embeddings(self):
            return self.embedding

    model = DummyModel()
    model.language_model = LanguageModel()

    wrap_input_embeddings_for_safe_scatter(model)
    first = model.language_model.get_input_embeddings()
    wrap_input_embeddings_for_safe_scatter(model)
    second = model.language_model.get_input_embeddings()

    assert first is second


def test_sail_forward_train_batch_rewraps_current_language_model(monkeypatch):
    from model_backends.sailvl.runtime import forward_train_batch

    class TrainLanguageModel:
        def __init__(self):
            self.embedding = torch.nn.Embedding(8, 4)

        def get_input_embeddings(self):
            return self.embedding

    class DummyDistributedReady:
        def is_available(self):
            return True

        def is_initialized(self):
            return True

        def get_rank(self):
            return 0

    class DummyTrainModel:
        def __init__(self):
            self.qformer_enabled = False
            self.language_model = TrainLanguageModel()

        def __call__(self, **kwargs):
            embeds = self.language_model.get_input_embeddings()(kwargs["input_ids"])
            return {"is_leaf": embeds.is_leaf, "requires_grad": embeds.requires_grad}

    monkeypatch.setattr(torch.Tensor, "cuda", lambda self: self, raising=False)
    monkeypatch.setattr(torch, "distributed", DummyDistributedReady())

    batch = (
        torch.tensor([[1, 2]], dtype=torch.long),
        torch.tensor([[1, 2]], dtype=torch.long),
        None,
        torch.zeros((1, 3, 2, 2), dtype=torch.float32),
        None,
        [],
    )

    outputs = forward_train_batch(DummyTrainModel(), batch, config={})

    assert outputs["requires_grad"] is True
    assert outputs["is_leaf"] is False


def test_patch_sail_forward_runtime_avoids_inplace_leaf_failure(monkeypatch):
    from model_backends.sailvl.runtime import patch_sail_forward_runtime

    class TinyLanguageModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(16, 4)
            self.config = SimpleNamespace(vocab_size=16)

        def get_input_embeddings(self):
            return self.embedding

        def forward(self, inputs_embeds=None, **kwargs):
            logits = torch.randn(
                inputs_embeds.shape[0],
                inputs_embeds.shape[1],
                self.config.vocab_size,
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
                requires_grad=True,
            )
            return SimpleNamespace(
                logits=logits,
                past_key_values=None,
                hidden_states=None,
                attentions=None,
            )

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = TinyLanguageModel()
            self.img_context_token_id = 7
            self.config = SimpleNamespace(use_return_dict=True)

        def extract_feature(self, pixel_values):
            return torch.ones((pixel_values.shape[0], 1, 4), device=pixel_values.device, dtype=pixel_values.dtype)

    monkeypatch.setattr(torch, "distributed", SimpleNamespace(
        is_available=lambda: True,
        is_initialized=lambda: False,
        get_rank=lambda: 0,
    ))

    model = TinyModel()
    patch_sail_forward_runtime(model)

    outputs = model(
        pixel_values=torch.zeros((1, 3, 2, 2), dtype=torch.float32),
        input_ids=torch.tensor([[7]], dtype=torch.long),
        attention_mask=torch.tensor([[1]], dtype=torch.long),
        image_flags=torch.tensor([[1]], dtype=torch.long),
        labels=torch.tensor([[7]], dtype=torch.long),
        return_dict=True,
    )

    assert outputs.logits.shape == (1, 1, 16)
