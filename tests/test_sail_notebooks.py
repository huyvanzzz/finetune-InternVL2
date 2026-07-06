import json
from pathlib import Path


def _load_notebook(path_str):
    path = Path(path_str)
    assert path.exists(), f"Missing notebook: {path_str}"
    return json.loads(path.read_text(encoding="utf-8"))


def _cell_source(nb, idx):
    return "".join(nb["cells"][idx]["source"])


def _all_sources(nb):
    return ["".join(cell["source"]) for cell in nb["cells"]]


def test_run_sail_qformer_notebook_targets_raw_779_branch_and_test_alter():
    nb = _load_notebook("run_sail_qformer.ipynb")
    assert 'BRANCH = "feature/sail-on-raw-779"' in _cell_source(nb, 0)
    assert 'CONFIG_PATH = "sailvl_config_mixed.yaml"' in _cell_source(nb, 0)
    sources = _all_sources(nb)
    assert any('--split", "test_alter"' in src for src in sources)
    assert any('prepare_qformer.py' in src for src in sources)
    assert any('smoke_qformer_bridge.py' in src for src in sources)


def test_run_sail_no_qformer_notebook_targets_raw_779_branch_and_test_alter():
    nb = _load_notebook("run_sail_no_qformer.ipynb")
    assert 'BRANCH = "feature/sail-on-raw-779"' in _cell_source(nb, 0)
    assert 'CONFIG_PATH = "sailvl_config_no_qformer_mixed.yaml"' in _cell_source(nb, 0)
    assert any('--split", "test_alter"' in src for src in _all_sources(nb))
