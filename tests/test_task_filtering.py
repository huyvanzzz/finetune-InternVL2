import json
from pathlib import Path

import pytest
import yaml

from wad_dataset import filter_samples_by_task_filter, should_stratify_task_split


def test_filter_samples_by_task_filter_supports_all_alter_only_and_qa_only():
    samples = [
        {"QA": {"Q": "where", "A": "left"}},
        {"alter": "move forward"},
        {"alter": "slow down"},
    ]

    assert filter_samples_by_task_filter(samples, "all") == samples
    assert filter_samples_by_task_filter(samples, "alter_only") == samples[1:]
    assert filter_samples_by_task_filter(samples, "qa_only") == samples[:1]


def test_filter_samples_by_task_filter_rejects_unknown_value():
    with pytest.raises(ValueError, match="Unsupported task filter"):
        filter_samples_by_task_filter([], "bad_filter")


def test_should_stratify_task_split_is_disabled_for_single_task_only():
    assert should_stratify_task_split(["alter", "alter", "qa"]) is True
    assert should_stratify_task_split(["alter", "alter"]) is False
    assert should_stratify_task_split([]) is False


def test_qformer_config_defaults_train_and_val_to_alter_only():
    with open("internvl_config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert config["data"]["train_task_filter"] == "alter_only"
    assert config["data"]["val_task_filter"] == "alter_only"


def test_no_qformer_config_exists_and_defaults_to_alter_only():
    with open("internvl_config_no_qformer.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert config["model"]["qformer"]["enabled"] is False
    assert config["data"]["train_task_filter"] == "alter_only"
    assert config["data"]["val_task_filter"] == "alter_only"


def test_run_no_qformer_notebook_uses_committed_config_without_generating_it():
    notebook = json.loads(Path("run_no_qformer.ipynb").read_text(encoding="utf-8"))
    code_cells = [
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    ]
    notebook_source = "\n".join(code_cells)

    assert 'CONFIG_PATH = "internvl_config_no_qformer.yaml"' in notebook_source
    assert 'with open("internvl_config.yaml", "r", encoding="utf-8") as f:' not in notebook_source
    assert 'yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)' not in notebook_source
