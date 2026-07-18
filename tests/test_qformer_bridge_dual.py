from types import SimpleNamespace

import torch

import qformer_bridge


class _DummyQFormer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(
        self,
        input_ids,
        attention_mask,
        query_embeds,
        encoder_hidden_states,
        encoder_attention_mask,
        return_dict=True,
    ):
        return (query_embeds + encoder_hidden_states[:, : query_embeds.shape[1], :],)


class _IdentityModule(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(hidden_size))

    def forward(self, x):
        return self.proj(x)


class _DummyModel(torch.nn.Module):
    def __init__(self, fusion_mode: str):
        super().__init__()
        hidden_size = 4
        self.select_layer = -1
        self.vision_model = torch.nn.Linear(1, 1)
        self.qformer_enabled = True
        self.trajectory_enabled = True
        self.trajectory_fusion_mode = fusion_mode
        self.qformer = _DummyQFormer()
        self.qformer_input_proj = _IdentityModule(hidden_size)
        self.qformer_to_mlp1_proj = _IdentityModule(hidden_size)
        self.mlp1 = _IdentityModule(hidden_size)
        self.qformer_query_tokens = torch.nn.Parameter(torch.zeros(1, 32, hidden_size))
        self._qformer_input_ids = torch.zeros(1, 1, dtype=torch.long)
        self._qformer_attention_mask = torch.ones(1, 1, dtype=torch.long)
        self.language_model = SimpleNamespace(
            get_input_embeddings=lambda: SimpleNamespace(weight=torch.zeros(hidden_size, hidden_size, dtype=torch.bfloat16))
        )
        self.config = SimpleNamespace(llm_config=SimpleNamespace(hidden_size=hidden_size))
        self.trajectory_cls_head = lambda traj_tokens, object_mask: torch.full((1, 1, hidden_size), 2.0)
        self.trajectory_token_projector = lambda traj_tokens: torch.full((1, 6, hidden_size), 3.0)

    def encode_qformer_texts(self, texts, device=None):
        input_ids = torch.zeros(len(texts), 1, dtype=torch.long, device=device)
        attention_mask = torch.ones(len(texts), 1, dtype=torch.long, device=device)
        return input_ids, attention_mask


def test_extract_feature_with_dual_applies_add_and_concat(monkeypatch):
    model = _DummyModel("dual")

    monkeypatch.setattr(
        qformer_bridge,
        "_extract_vit_tokens",
        lambda self, pixel_values: torch.ones(pixel_values.shape[0], 32, 4, dtype=pixel_values.dtype, device=pixel_values.device),
    )
    monkeypatch.setattr(qformer_bridge, "_ensure_bridge_device", lambda self, reference: None)
    monkeypatch.setattr(
        qformer_bridge,
        "build_trajectory_tokens_base",
        lambda model, batch_size, device: (
            torch.zeros(batch_size, 6, 128, device=device, dtype=torch.float32),
            torch.ones(batch_size, 6, device=device, dtype=torch.long),
        ),
    )

    pixel_values = torch.ones(1, 3, 448, 448, dtype=torch.float32)
    visual_embeds = qformer_bridge._extract_feature_with_qformer(model, pixel_values)

    assert visual_embeds.shape == (1, 38, 4)
    assert visual_embeds.dtype == torch.bfloat16
    assert torch.allclose(visual_embeds[:, :32, :].float(), torch.full((1, 32, 4), 3.0))
    assert torch.allclose(visual_embeds[:, 32:, :].float(), torch.full((1, 6, 4), 3.0))


def test_extract_feature_with_qformer_matches_llm_embedding_dtype(monkeypatch):
    model = _DummyModel("cls_add")

    monkeypatch.setattr(
        qformer_bridge,
        "_extract_vit_tokens",
        lambda self, pixel_values: torch.ones(pixel_values.shape[0], 32, 4, dtype=torch.float32, device=pixel_values.device),
    )
    monkeypatch.setattr(qformer_bridge, "_ensure_bridge_device", lambda self, reference: None)
    monkeypatch.setattr(
        qformer_bridge,
        "build_trajectory_features",
        lambda model, batch_size, device: torch.zeros(batch_size, 1, 4, device=device, dtype=torch.float32),
    )

    pixel_values = torch.ones(1, 3, 448, 448, dtype=torch.float32)
    visual_embeds = qformer_bridge._extract_feature_with_qformer(model, pixel_values)

    assert visual_embeds.dtype == torch.bfloat16


def test_extract_feature_with_cls_add_records_trajectory_debug(monkeypatch):
    model = _DummyModel("cls_add")
    model._last_trajectory_debug = {"backbone_stage_debug": {"tokens_output_abs_mean": 0.5}}

    monkeypatch.setattr(
        qformer_bridge,
        "_extract_vit_tokens",
        lambda self, pixel_values: torch.ones(pixel_values.shape[0], 32, 4, dtype=torch.float32, device=pixel_values.device),
    )
    monkeypatch.setattr(qformer_bridge, "_ensure_bridge_device", lambda self, reference: None)
    monkeypatch.setattr(
        qformer_bridge,
        "build_trajectory_features",
        lambda model, batch_size, device: torch.zeros(batch_size, 1, 4, device=device, dtype=torch.float32, requires_grad=True),
    )

    pixel_values = torch.ones(1, 3, 448, 448, dtype=torch.float32)
    qformer_bridge._extract_feature_with_qformer(model, pixel_values)

    debug = model._last_trajectory_debug
    assert debug["backbone_stage_debug"]["tokens_output_abs_mean"] == 0.5
    assert debug["fusion_mode"] == "cls_add"
    assert debug["traj_path_active"] is True
    assert debug["traj_cls_requires_grad"] is True
    assert debug["mlp1_inputs_requires_grad_before_add"] is True
    assert debug["mlp1_inputs_requires_grad_after_add"] is True
    assert "traj_cls" in model._last_trajectory_debug_tensors
    assert "mlp1_inputs_before_add" in model._last_trajectory_debug_tensors
    assert "mlp1_inputs_after_add" in model._last_trajectory_debug_tensors
