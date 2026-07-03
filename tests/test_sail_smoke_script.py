import builtins
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT_PATH = Path("scripts/smoke_sail_backend.py")


def load_module():
    spec = importlib.util.spec_from_file_location("smoke_sail_backend", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DummyModel:
    def __init__(self, qformer_enabled=True, num_image_token=32):
        self.qformer_enabled = qformer_enabled
        self.num_image_token = num_image_token
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self


def test_smoke_script_uses_sail_backend_and_loads_checkpoint(monkeypatch, capsys):
    module = load_module()
    calls = {
        "load_model": 0,
        "attach_qformer": 0,
        "load_backend_artifacts": 0,
        "generate_response": 0,
    }

    dummy_model = DummyModel(qformer_enabled=True, num_image_token=32)
    dummy_backend = SimpleNamespace(
        name="sailvl",
        load_model_and_tokenizer=lambda config, checkpoint_dir=None: (
            calls.__setitem__("load_model", calls["load_model"] + 1) or dummy_model,
            object(),
        ),
        attach_qformer_if_enabled=lambda model, config, logger=None: calls.__setitem__(
            "attach_qformer",
            calls["attach_qformer"] + 1,
        )
        or model,
        load_backend_artifacts=lambda model, checkpoint_dir, config: calls.__setitem__(
            "load_backend_artifacts",
            calls["load_backend_artifacts"] + 1,
        ),
        generate_response=lambda model, tokenizer, sample, generation_config, config: calls.__setitem__(
            "generate_response",
            calls["generate_response"] + 1,
        )
        or "move forward carefully",
    )

    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: SimpleNamespace(config="dummy.yaml", checkpoint="dummy-ckpt"),
    )
    monkeypatch.setattr(module, "resolve_checkpoint_path", lambda checkpoint: checkpoint)
    monkeypatch.setattr(module, "get_backend", lambda architecture: dummy_backend)
    monkeypatch.setattr(
        module.yaml,
        "safe_load",
        lambda stream: {
            "model": {
                "architecture": "sailvl",
                "name": "BytedanceDouyinContent/SAIL-VL-1d5-2B",
                "qformer": {"enabled": True},
                "vision": {
                    "force_image_size": 448,
                    "use_thumbnail": True,
                    "min_dynamic_patch": 1,
                    "max_dynamic_patch": 12,
                },
            }
        },
    )
    monkeypatch.setattr(module, "preprocess_sail_image", lambda image, config: [0, 1, 2])
    monkeypatch.setattr(builtins, "open", lambda *args, **kwargs: io.StringIO("dummy: true"))

    module.main()
    out = capsys.readouterr().out

    assert calls["load_model"] == 1
    assert calls["attach_qformer"] == 1
    assert calls["load_backend_artifacts"] == 1
    assert calls["generate_response"] == 1
    assert "tiles: 3" in out
    assert "image tokens: 96" in out
    assert "qformer enabled: True" in out
    assert "response:" in out


def test_smoke_script_requires_sailvl_backend(monkeypatch):
    module = load_module()
    monkeypatch.setattr(
        module,
        "parse_args",
        lambda: SimpleNamespace(config="dummy.yaml", checkpoint=None),
    )
    monkeypatch.setattr(
        module.yaml,
        "safe_load",
        lambda stream: {"model": {"architecture": "internvl"}},
    )
    monkeypatch.setattr(builtins, "open", lambda *args, **kwargs: io.StringIO("dummy: true"))

    with pytest.raises(SystemExit, match="SAIL smoke test only supports sailvl configs"):
        module.main()
