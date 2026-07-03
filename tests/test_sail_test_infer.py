import importlib.machinery
import io
import sys
import builtins
from types import ModuleType, SimpleNamespace

import pytest
import torch


if "peft" not in sys.modules:
    peft_stub = ModuleType("peft")
    peft_stub.__spec__ = importlib.machinery.ModuleSpec("peft", loader=None)
    peft_stub.PeftModel = SimpleNamespace(from_pretrained=lambda model, *args, **kwargs: model)
    sys.modules["peft"] = peft_stub

if "evaluate" not in sys.modules:
    evaluate_stub = ModuleType("evaluate")
    evaluate_stub.__spec__ = importlib.machinery.ModuleSpec("evaluate", loader=None)
    evaluate_stub.load = lambda *args, **kwargs: None
    sys.modules["evaluate"] = evaluate_stub


class StopAfterCheckpointLoad(RuntimeError):
    pass


class DummyEmbeddings(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros((8, 4)))


class DummyLanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_embeddings = DummyEmbeddings()
        self.output_embeddings = DummyEmbeddings()

    def get_input_embeddings(self):
        return self.input_embeddings

    def get_output_embeddings(self):
        return self.output_embeddings

    def eval(self):
        return self


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = DummyLanguageModel()
        self.qformer_enabled = True
        self.qformer = torch.nn.Linear(4, 4)
        self.template = "sailvl-chat"
        self.system_message = "system"
        self.img_context_token_id = 1

    def eval(self):
        return self


def test_sail_test_infer_loads_backend_artifacts_instead_of_internvl_bridge(monkeypatch):
    import scripts.test_infer as test_infer

    calls = {
        "backend_load_backend_artifacts": 0,
        "legacy_qformer_bridge_load": 0,
    }

    dummy_backend = SimpleNamespace(
        name="sailvl",
        load_model_and_tokenizer=lambda config, checkpoint_dir=None: (DummyModel(), object()),
        attach_qformer_if_enabled=lambda model, config: model,
        load_backend_artifacts=lambda model, checkpoint_dir, config: calls.__setitem__(
            "backend_load_backend_artifacts",
            calls["backend_load_backend_artifacts"] + 1,
        ),
    )

    monkeypatch.setattr(
        test_infer,
        "parse_args",
        lambda: SimpleNamespace(
            config="dummy.yaml",
            checkpoint="dummy-checkpoint",
            split="test_alter",
            output_file="results/eval_results.json",
            print_samples=1,
        ),
    )
    monkeypatch.setattr(test_infer, "resolve_checkpoint_path", lambda checkpoint: checkpoint)
    monkeypatch.setattr(builtins, "open", lambda *args, **kwargs: io.StringIO("dummy: true"))
    monkeypatch.setattr(
        test_infer.yaml,
        "safe_load",
        lambda stream: {
            "model": {
                "architecture": "sailvl",
                "name": "BytedanceDouyinContent/SAIL-VL-1d5-2B",
                "trust_remote_code": True,
                "quantization": {
                    "enabled": True,
                    "compute_dtype": "bfloat16",
                    "double_quant": True,
                    "type": "nf4",
                },
                "qformer": {"enabled": True},
            },
            "training": {"batch_size": 1},
            "evaluation": {"batch_size": 1},
            "data": {
                "name": "dummy",
                "direct_text_alter_prompt_mode": "balanced_v1",
            },
        },
    )
    monkeypatch.setattr(test_infer, "get_response_format", lambda config: "direct_text")
    monkeypatch.setattr(test_infer, "get_backend", lambda architecture: dummy_backend)
    monkeypatch.setattr(test_infer, "qformer_enabled", lambda config: True)
    monkeypatch.setattr(test_infer, "BitsAndBytesConfig", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(
        test_infer,
        "load_qformer_bridge",
        lambda *args, **kwargs: calls.__setitem__(
            "legacy_qformer_bridge_load",
            calls["legacy_qformer_bridge_load"] + 1,
        ),
    )
    monkeypatch.setattr(
        test_infer,
        "PeftModel",
        SimpleNamespace(from_pretrained=lambda model, *args, **kwargs: model),
    )
    monkeypatch.setattr(test_infer, "align_language_model_devices", lambda model: None)
    monkeypatch.setattr(test_infer, "log_eval_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        test_infer,
        "prepare_auxiliary_data",
        lambda config: (_ for _ in ()).throw(StopAfterCheckpointLoad()),
    )

    with pytest.raises(StopAfterCheckpointLoad):
        test_infer.main()

    assert calls["backend_load_backend_artifacts"] == 1
    assert calls["legacy_qformer_bridge_load"] == 0
