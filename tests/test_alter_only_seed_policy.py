import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _notebook_text(name: str) -> str:
    notebook = json.loads((ROOT / name).read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )


def test_epoch_seed_is_fixed():
    source = (ROOT / "train.py").read_text(encoding="utf-8")
    assert "set_seed(42 + epoch)" not in source
    assert "set_seed(42)" in source
    assert "Epoch seed fixed" in source


def test_primary_configs_are_alter_only():
    for config_name in [
        "internvl_config.yaml",
        "internvl_config_no_qformer.yaml",
        "sailvl_config.yaml",
        "sailvl_config_no_qformer.yaml",
    ]:
        cfg = yaml.safe_load((ROOT / config_name).read_text(encoding="utf-8"))
        assert cfg["data"]["train_task_filter"] == "alter_only"
        assert cfg["data"]["val_task_filter"] == "alter_only"


def test_primary_notebooks_use_alter_only_configs():
    expected = {
        "run_qformer.ipynb": "internvl_config.yaml",
        "run_no_qformer.ipynb": "internvl_config_no_qformer.yaml",
        "run_sail_qformer.ipynb": "sailvl_config.yaml",
        "run_sail_no_qformer.ipynb": "sailvl_config_no_qformer.yaml",
    }
    for notebook_name, config_name in expected.items():
        text = _notebook_text(notebook_name)
        assert f'CONFIG_PATH = "{config_name}"' in text
        assert "_mixed.yaml" not in text
        assert '"--split", "test_alter"' in text
