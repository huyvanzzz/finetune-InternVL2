import json
from pathlib import Path


TARGET_COMMIT = "779cc7b2284c8fa480ef1d5cc91a89c5f21ee862"
TRAJ_BRANCH = "feature/trajectory-qformer-restore-779"


def _read_cells(path: str):
    notebook = json.loads(Path(path).read_text(encoding="utf-8"))
    cell0 = "".join(notebook["cells"][0]["source"])
    cell1 = "".join(notebook["cells"][1]["source"])
    return cell0, cell1


def test_run_qformer_notebook_uses_trajectory_branch_and_config_selector():
    notebook = json.loads(Path("run_qformer.ipynb").read_text(encoding="utf-8"))
    cell0 = "".join(notebook["cells"][0]["source"])
    cell1 = "".join(notebook["cells"][1]["source"])
    train_cell = "".join(notebook["cells"][7]["source"])
    infer_cell = "".join(notebook["cells"][8]["source"])

    assert f'TARGET_BRANCH = "{TRAJ_BRANCH}"' in cell0
    assert 'CONFIG_PATH = "internvl_config_traj_cls.yaml"' in cell0
    assert 'subprocess.run(["git", "fetch", "origin", TARGET_BRANCH], check=True)' in cell1
    assert 'subprocess.run(["git", "checkout", "-B", TARGET_BRANCH, f"origin/{TARGET_BRANCH}"], check=True)' in cell1
    assert 'cmd = ["python", "train.py", "--config", CONFIG_PATH]' in train_cell
    assert '"--config", CONFIG_PATH' in infer_cell
    assert '"--split", "test_alter"' in infer_cell
    assert Path("internvl_config_traj_cls.yaml").exists()
    assert Path("internvl_config_traj_concat.yaml").exists()


def test_run_no_qformer_notebook_pins_779_commit():
    cell0, cell1 = _read_cells("run_no_qformer.ipynb")

    assert 'TARGET_BRANCH = "restore-779cc7b"' in cell0
    assert 'CONFIG_PATH = "internvl_config_no_qformer.yaml"' in cell0
    assert 'subprocess.run(["git", "fetch", "origin", TARGET_BRANCH], check=True)' in cell1
    assert 'subprocess.run(["git", "checkout", "-B", TARGET_BRANCH, f"origin/{TARGET_BRANCH}"], check=True)' in cell1
