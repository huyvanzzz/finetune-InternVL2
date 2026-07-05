import importlib.machinery
import sys
from types import ModuleType

import yaml

if "peft" not in sys.modules:
    peft_stub = ModuleType("peft")
    peft_stub.__spec__ = importlib.machinery.ModuleSpec("peft", loader=None)
    peft_stub.LoraConfig = object
    peft_stub.PeftModel = object
    peft_stub.get_peft_model = lambda *args, **kwargs: None
    peft_stub.prepare_model_for_kbit_training = lambda model, *args, **kwargs: model
    sys.modules["peft"] = peft_stub

from train import should_restore_runtime_state


def test_should_restore_runtime_state_defaults_to_new_behavior():
    assert should_restore_runtime_state({"training": {}}) is True


def test_should_restore_runtime_state_can_switch_back_to_legacy_skip_only():
    assert should_restore_runtime_state({"training": {"resume_runtime_mode": "legacy_skip_only"}}) is False


def test_mixed_internvl_config_uses_old_runtime_modes():
    with open("internvl_config_mixed.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert config["data"]["direct_text_alter_prompt_mode"] == "fixed_779"
    assert config["data"]["direct_text_qa_prompt_mode"] == "legacy_779"
    assert config["data"]["non_train_error_policy"] == "resample"
    assert config["training"]["resume_runtime_mode"] == "legacy_skip_only"


def test_mixed_sail_config_uses_old_runtime_modes():
    with open("sailvl_config_mixed.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert config["data"]["direct_text_alter_prompt_mode"] == "fixed_779"
    assert config["data"]["direct_text_qa_prompt_mode"] == "legacy_779"
    assert config["data"]["non_train_error_policy"] == "resample"
    assert config["training"]["resume_runtime_mode"] == "legacy_skip_only"
