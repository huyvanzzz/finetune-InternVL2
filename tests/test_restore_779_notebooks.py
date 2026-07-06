import json
from pathlib import Path


TARGET_COMMIT = "779cc7b2284c8fa480ef1d5cc91a89c5f21ee862"


def _read_cells(path: str):
    notebook = json.loads(Path(path).read_text(encoding="utf-8"))
    cell0 = "".join(notebook["cells"][0]["source"])
    cell1 = "".join(notebook["cells"][1]["source"])
    return cell0, cell1


def test_run_qformer_notebook_pins_779_commit():
    cell0, cell1 = _read_cells("run_qformer.ipynb")

    assert f'TARGET_COMMIT = "{TARGET_COMMIT}"' in cell0
    assert 'subprocess.run(["git", "checkout", TARGET_COMMIT], check=True)' in cell1
    assert 'origin/{BRANCH}' not in cell1


def test_run_no_qformer_notebook_pins_779_commit():
    cell0, cell1 = _read_cells("run_no_qformer.ipynb")

    assert f'TARGET_COMMIT = "{TARGET_COMMIT}"' in cell0
    assert 'subprocess.run(["git", "checkout", TARGET_COMMIT], check=True)' in cell1
    assert 'origin/{BRANCH}' not in cell1
