import io
import importlib.machinery
import sys
from types import ModuleType
from types import SimpleNamespace

import yaml

if "peft" not in sys.modules:
    peft_stub = ModuleType("peft")
    peft_stub.__spec__ = importlib.machinery.ModuleSpec("peft", loader=None)
    peft_stub.PeftModel = object
    sys.modules["peft"] = peft_stub

if "evaluate" not in sys.modules:
    evaluate_stub = ModuleType("evaluate")
    evaluate_stub.__spec__ = importlib.machinery.ModuleSpec("evaluate", loader=None)
    evaluate_stub.load = lambda *args, **kwargs: None
    sys.modules["evaluate"] = evaluate_stub

from scripts import test_infer


class DummyModel:
    def __init__(self):
        self.template = "dummy"
        self.system_message = ""
        self.qformer_enabled = False

    def eval(self):
        return self

    def cuda(self):
        return self


class DummyTokenizer:
    def convert_tokens_to_ids(self, token):
        return 0


class EmptyDataset:
    def __len__(self):
        return 0


def test_test_infer_passes_legacy_dataset_modes(monkeypatch, tmp_path):
    captured = {}
    output_file = tmp_path / "results.json"
    config_text = yaml.safe_dump(
        {
            "model": {
                "architecture": "internvl",
                "name": "dummy/model",
                "trust_remote_code": True,
                "quantization": {
                    "enabled": False,
                    "compute_dtype": "float16",
                    "double_quant": False,
                    "type": "nf4",
                },
            },
            "training": {"batch_size": 1},
            "evaluation": {"batch_size": 1},
            "data": {
                "name": "dummy",
                "response_format": "direct_text",
                "direct_text_alter_prompt_mode": "fixed_779",
                "direct_text_qa_prompt_mode": "legacy_779",
                "non_train_error_policy": "resample",
                "seed": 42,
            },
        }
    )

    monkeypatch.setattr(
        test_infer,
        "parse_args",
        lambda: SimpleNamespace(
            config="dummy.yaml",
            checkpoint=None,
            split="test_alter",
            output_file=str(output_file),
            print_samples=0,
        ),
    )
    monkeypatch.setattr(test_infer, "resolve_checkpoint_path", lambda checkpoint: checkpoint)
    monkeypatch.setattr(test_infer, "prepare_auxiliary_data", lambda config: ({}, {}))
    monkeypatch.setattr(test_infer, "load_dataset", lambda *args, **kwargs: {"test": []})
    monkeypatch.setattr(test_infer, "get_response_format", lambda config: "direct_text")
    monkeypatch.setattr(test_infer, "qformer_enabled", lambda config: False)
    monkeypatch.setattr(test_infer, "align_language_model_devices", lambda model: None)
    monkeypatch.setattr(test_infer, "log_eval_state", lambda model, stage: None)
    monkeypatch.setattr(test_infer, "log_runtime_prompt_state", lambda model, stage: None)
    monkeypatch.setattr(test_infer, "BitsAndBytesConfig", lambda **kwargs: kwargs)
    monkeypatch.setattr(
        test_infer.AutoModel,
        "from_pretrained",
        lambda *args, **kwargs: DummyModel(),
    )
    monkeypatch.setattr(
        test_infer.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: DummyTokenizer(),
    )
    monkeypatch.setattr(test_infer, "DataLoader", lambda *args, **kwargs: [])
    monkeypatch.setattr(test_infer, "VLMMetrics", lambda: SimpleNamespace(compute=lambda *args, **kwargs: {}))
    monkeypatch.setattr(test_infer.torch.nn.Module, "cuda", lambda self: self, raising=False)

    def fake_dataset_ctor(**kwargs):
        captured.update(kwargs)
        return EmptyDataset()

    monkeypatch.setattr(test_infer, "WADDatasetForInternVL", fake_dataset_ctor)

    real_open = open

    def fake_open(path, mode="r", *args, **kwargs):
        if path == "dummy.yaml":
            return io.StringIO(config_text)
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", fake_open)

    test_infer.main()

    assert captured["direct_text_alter_prompt_mode"] == "fixed_779"
    assert captured["direct_text_qa_prompt_mode"] == "legacy_779"
    assert captured["non_train_error_policy"] == "resample"
