import json
from pathlib import Path

import yaml

from wad_dataset import filter_alter_only_rows, has_nonempty_qa, summarize_qa_rows


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


def test_alter_only_helpers_treat_only_nonempty_qa_q_as_qa():
    rows = [
        {"id": "alter"},
        {"id": "empty_qa", "QA": {"Q": "   "}},
        {"id": "qa", "QA": {"Q": "What is visible?"}},
    ]
    assert has_nonempty_qa(rows[0]) is False
    assert has_nonempty_qa(rows[1]) is False
    assert has_nonempty_qa(rows[2]) is True
    assert summarize_qa_rows(rows) == {"total": 3, "with_qa": 1, "without_qa": 2}
    assert [row["id"] for row in filter_alter_only_rows(rows)] == ["alter", "empty_qa"]


def test_default_config_enables_alter_only_debug():
    cfg = yaml.safe_load((ROOT / "internvl_config.yaml").read_text(encoding="utf-8"))
    assert cfg["data"]["alter_only"] is True
    assert cfg["training"]["debug_dataset_stats"] is True
    assert cfg["training"]["debug_dataset_samples"] == 2


def test_primary_notebooks_infer_on_test_alter():
    for notebook_name in ["run_qformer.ipynb", "run_no_qformer.ipynb"]:
        text = _notebook_text(notebook_name)
        assert '"--split", "test_alter"' in text
        assert "test_QA" not in text


def test_no_qformer_notebook_writes_alter_only_config():
    text = _notebook_text("run_no_qformer.ipynb")
    assert 'base_cfg.setdefault("data", {})["alter_only"] = True' in text
