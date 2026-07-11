from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def test_train_checkpoint_download_patterns_include_trajectory_artifacts():
    content = _read("train.py")
    assert '"trajectory_branch.safetensors"' in content
    assert '"trajectory_branch_config.json"' in content


def test_infer_checkpoint_download_patterns_include_trajectory_artifacts():
    content = _read("scripts/test_infer.py")
    assert '"trajectory_branch.safetensors"' in content
    assert '"trajectory_branch_config.json"' in content
