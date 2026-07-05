from types import SimpleNamespace

import torch


class DummyVisionOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class DummyVisionModel(torch.nn.Module):
    def forward(self, pixel_values=None, output_hidden_states=False, return_dict=True):
        batch_size = pixel_values.shape[0]
        hidden = torch.zeros((batch_size, 256, 16), dtype=pixel_values.dtype, device=pixel_values.device)
        return DummyVisionOutput(hidden)


class DummyQFormer(torch.nn.Module):
    def __init__(self, hidden_size=8):
        super().__init__()
        self.proj = torch.nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        query_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        return_dict=True,
    ):
        batch = query_embeds.shape[0]
        num_queries = query_embeds.shape[1]
        hidden = query_embeds.shape[2]
        output = torch.zeros((batch, num_queries, hidden), device=query_embeds.device, dtype=query_embeds.dtype)
        return (output,)


class DummyTokenizer:
    def __call__(self, texts, padding=True, truncation=True, max_length=128, return_tensors="pt"):
        batch = len(texts)
        return {
            "input_ids": torch.ones((batch, 4), dtype=torch.long),
            "attention_mask": torch.ones((batch, 4), dtype=torch.long),
        }


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            vision_config=SimpleNamespace(hidden_size=16),
            llm_config=SimpleNamespace(hidden_size=12),
            downsample_ratio=0.5,
        )
        self.template = "sailvl-chat"
        self.system_message = "system"
        self.select_layer = -1
        self.downsample_ratio = 0.5
        self.vision_model = DummyVisionModel()
        self.mlp1 = torch.nn.Sequential(
            torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 12),
            torch.nn.GELU(),
            torch.nn.Linear(12, 12),
        )

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        x = x.reshape(n, w, int(h * scale_factor), int(c / scale_factor))
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.reshape(n, int(h * scale_factor), int(w * scale_factor), int(c / (scale_factor * scale_factor)))
        x = x.permute(0, 2, 1, 3).contiguous()
        return x


def test_attach_sail_qformer_bridge_sets_expected_runtime_contract(monkeypatch):
    from model_backends.sailvl.qformer_bridge import attach_sail_qformer_bridge

    dummy_qformer = DummyQFormer(hidden_size=8)
    query_tokens = torch.nn.Parameter(torch.zeros((1, 32, 8)))
    dummy_tokenizer = DummyTokenizer()
    dummy_blip_config = SimpleNamespace(
        qformer_config=SimpleNamespace(hidden_size=8, encoder_hidden_size=64),
    )

    monkeypatch.setattr(
        "model_backends.sailvl.qformer_bridge._load_qformer_from_source",
        lambda source_model, cache_dir: (dummy_qformer, query_tokens, dummy_tokenizer, dummy_blip_config),
    )

    model = DummyModel()
    config = {
        "model": {
            "qformer": {
                "enabled": True,
                "source_model": "Salesforce/instructblip-flan-t5-xl",
                "cache_dir": "./qformer_cache",
                "num_query_tokens": 32,
                "freeze_qformer": True,
                "freeze_mlp1": True,
                "prompt_aware": True,
                "max_text_length": 128,
            }
        }
    }

    attach_sail_qformer_bridge(model, config)

    assert model.qformer_enabled is True
    assert model.num_image_token == 32
    assert hasattr(model, "qformer_input_proj")
    assert hasattr(model, "qformer_to_mlp1_proj")
    assert hasattr(model, "encode_qformer_texts")
    assert hasattr(model, "set_qformer_text")
    assert hasattr(model, "clear_qformer_text")
    assert model.qformer_query_tokens.requires_grad is False
    assert next(model.qformer.parameters()).requires_grad is False
    assert next(model.mlp1.parameters()).requires_grad is False
    assert next(model.qformer_input_proj.parameters()).requires_grad is True
    assert next(model.qformer_to_mlp1_proj.parameters()).requires_grad is True


def test_sail_qformer_bridge_extract_feature_runs_end_to_end(monkeypatch):
    from model_backends.sailvl.qformer_bridge import attach_sail_qformer_bridge

    dummy_qformer = DummyQFormer(hidden_size=8)
    query_tokens = torch.nn.Parameter(torch.zeros((1, 32, 8)))
    dummy_tokenizer = DummyTokenizer()
    dummy_blip_config = SimpleNamespace(
        qformer_config=SimpleNamespace(hidden_size=8, encoder_hidden_size=64),
    )

    monkeypatch.setattr(
        "model_backends.sailvl.qformer_bridge._load_qformer_from_source",
        lambda source_model, cache_dir: (dummy_qformer, query_tokens, dummy_tokenizer, dummy_blip_config),
    )

    model = DummyModel()
    config = {
        "model": {
            "qformer": {
                "enabled": True,
                "source_model": "Salesforce/instructblip-flan-t5-xl",
                "cache_dir": "./qformer_cache",
                "num_query_tokens": 32,
                "freeze_qformer": True,
                "freeze_mlp1": True,
                "prompt_aware": True,
                "max_text_length": 128,
            }
        }
    }

    attach_sail_qformer_bridge(model, config)
    input_ids, attention_mask = model.encode_qformer_texts(["describe scene", "describe scene"])
    model.set_qformer_text(input_ids, attention_mask)
    pixel_values = torch.zeros((2, 3, 4, 4), dtype=torch.float32)
    features = model.extract_feature(pixel_values)
    model.clear_qformer_text()

    assert features.shape == (2, 32, 12)


def test_align_sail_qformer_bridge_runtime_uses_language_model_device(monkeypatch):
    from model_backends.sailvl.qformer_bridge import align_sail_qformer_bridge_runtime, attach_sail_qformer_bridge

    dummy_qformer = DummyQFormer(hidden_size=8)
    query_tokens = torch.nn.Parameter(torch.zeros((1, 32, 8)))
    dummy_tokenizer = DummyTokenizer()
    dummy_blip_config = SimpleNamespace(
        qformer_config=SimpleNamespace(hidden_size=8, encoder_hidden_size=64),
    )

    monkeypatch.setattr(
        "model_backends.sailvl.qformer_bridge._load_qformer_from_source",
        lambda source_model, cache_dir: (dummy_qformer, query_tokens, dummy_tokenizer, dummy_blip_config),
    )

    model = DummyModel()
    model.language_model = torch.nn.Linear(12, 12)
    config = {
        "model": {
            "qformer": {
                "enabled": True,
                "source_model": "Salesforce/instructblip-flan-t5-xl",
                "cache_dir": "./qformer_cache",
                "num_query_tokens": 32,
                "freeze_qformer": True,
                "freeze_mlp1": True,
                "prompt_aware": True,
                "max_text_length": 128,
            }
        }
    }

    attach_sail_qformer_bridge(model, config)
    device = align_sail_qformer_bridge_runtime(model)

    assert str(device) == "cpu"
    assert next(model.qformer_input_proj.parameters()).device.type == "cpu"
    assert next(model.qformer_to_mlp1_proj.parameters()).device.type == "cpu"
