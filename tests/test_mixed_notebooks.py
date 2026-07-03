import json
from pathlib import Path


def load_notebook_source(path: str) -> str:
    notebook = json.loads(Path(path).read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


def test_all_runtime_notebooks_default_to_mixed_configs():
    expected = {
        "run_qformer.ipynb": 'CONFIG_PATH = "internvl_config_mixed.yaml"',
        "run_no_qformer.ipynb": 'CONFIG_PATH = "internvl_config_no_qformer_mixed.yaml"',
        "run_sail_qformer.ipynb": 'CONFIG_PATH = "sailvl_config_mixed.yaml"',
        "run_sail_no_qformer.ipynb": 'CONFIG_PATH = "sailvl_config_no_qformer_mixed.yaml"',
    }

    for notebook_name, config_line in expected.items():
        source = load_notebook_source(notebook_name)
        assert config_line in source


def test_all_runtime_notebooks_pull_feature_sailvl_core_branch():
    for notebook_name in (
        "run_qformer.ipynb",
        "run_no_qformer.ipynb",
        "run_sail_qformer.ipynb",
        "run_sail_no_qformer.ipynb",
    ):
        source = load_notebook_source(notebook_name)
        assert 'BRANCH = "feature/sailvl-core"' in source


def test_notebooks_use_config_path_for_prepare_train_and_eval():
    for notebook_name in (
        "run_qformer.ipynb",
        "run_no_qformer.ipynb",
        "run_sail_qformer.ipynb",
        "run_sail_no_qformer.ipynb",
    ):
        source = load_notebook_source(notebook_name)
        assert '--config", CONFIG_PATH' in source
        assert '--split", "test_alter"' in source


def test_notebooks_discover_checkpoints_from_yaml_output_dir_not_hardcoded_legacy_paths():
    for notebook_name in (
        "run_qformer.ipynb",
        "run_no_qformer.ipynb",
        "run_sail_qformer.ipynb",
        "run_sail_no_qformer.ipynb",
    ):
        source = load_notebook_source(notebook_name)
        assert 'OUTPUT_DIR = cfg["training"]["output_dir"]' in source
        assert 'Path(OUTPUT_DIR)' in source
        assert 'outputs/internvl3_2b' not in source
        assert 'outputs/internvl3_2b_no_qformer' not in source
        assert 'outputs/sailvl_1d5_2b' not in source
        assert 'outputs/sailvl_1d5_2b_no_qformer' not in source


def test_run_qformer_train_cell_is_active_not_commented_out():
    source = load_notebook_source("run_qformer.ipynb")
    assert 'cmd = ["python", "train.py", "--config", CONFIG_PATH]' in source
    assert '# cmd = ["python", "train.py"]' not in source

