from pathlib import Path

import yaml

import wad_dataset


ROOT = Path(__file__).resolve().parents[1]


def test_alter_only_helpers_filter_out_rows_with_nonempty_qa():
    rows = [
        {"frame_path": "alter_a", "answer": "alter only"},
        {"frame_path": "qa_b", "QA": {"Q": "What is ahead?"}, "answer": "qa"},
        {"frame_path": "empty_c", "QA": {"Q": ""}, "answer": "empty qa"},
        {"frame_path": "spaces_d", "QA": {"Q": "   "}, "answer": "spaces qa"},
    ]

    assert wad_dataset.has_nonempty_qa(rows[0]) is False
    assert wad_dataset.has_nonempty_qa(rows[1]) is True
    assert wad_dataset.has_nonempty_qa(rows[2]) is False
    assert wad_dataset.has_nonempty_qa(rows[3]) is False

    assert wad_dataset.summarize_qa_rows(rows) == {
        "total": 4,
        "with_qa": 1,
        "without_qa": 3,
    }
    assert [row["frame_path"] for row in wad_dataset.filter_alter_only_rows(rows)] == [
        "alter_a",
        "empty_c",
        "spaces_d",
    ]


def test_trajectory_finetune_configs_enable_alter_only_debug():
    for config_name in (
        "internvl_config_traj_cls.yaml",
        "internvl_config_traj_concat.yaml",
        "internvl_config_traj_dual.yaml",
    ):
        cfg = yaml.safe_load((ROOT / config_name).read_text(encoding="utf-8"))

        assert cfg["data"]["alter_only"] is True
        assert cfg["training"]["debug_dataset_stats"] is True
        assert cfg["training"]["debug_dataset_samples"] == 2


def test_finetune_epoch_seed_is_fixed():
    source = (ROOT / "train.py").read_text(encoding="utf-8")

    assert "set_seed(42 + epoch)" not in source
    assert "set_seed(42)" in source
    assert "Epoch seed fixed" in source


def test_trajectory_finetune_configs_use_upscaled_architecture():
    for config_name in (
        "internvl_config_traj_cls.yaml",
        "internvl_config_traj_concat.yaml",
        "internvl_config_traj_dual.yaml",
    ):
        cfg = yaml.safe_load((ROOT / config_name).read_text(encoding="utf-8"))
        traj_cfg = cfg["trajectory"]

        assert traj_cfg["d_traj"] == 384
        assert traj_cfg["num_layers"] == 4
        assert traj_cfg["ffn_dim"] == 768
