import json
from pathlib import Path

import pytest
import torch

from qformer_bridge import BRIDGE_CONFIG_NAME, save_qformer_bridge
from trajectory_branch import TRAJECTORY_BRANCH_CONFIG_NAME, save_trajectory_branch


class _TinyBridge(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qformer_enabled = True
        self.qformer_source_model = "dummy-source"
        self.qformer_num_query_tokens = 32
        self.num_image_token = 32
        self.qformer_input_proj = torch.nn.Linear(4, 4)
        self.qformer_to_mlp1_proj = torch.nn.Linear(4, 4)
        self.pretrain_stage = "pretrain"
        self.pretrain_data_source = "./json/question_train.jsonl"
        self.question_format_version = "v1_qa_question_answer"
        self.pretrain_movement_enabled = True


class _TinyTrajectory(torch.nn.Module):
    def __init__(self, fusion_mode: str):
        super().__init__()
        from trajectory_branch import TrajectoryBackbone, TrajectoryCLSHead, TrajectoryConcatHead

        self.trajectory_enabled = True
        self.trajectory_fusion_mode = fusion_mode
        self.trajectory_source_file = "json/results_botsort_top6_sorted.jsonl"
        self.trajectory_num_objects = 6
        self.trajectory_qformer_token_count = 32
        self.num_image_token = 38 if fusion_mode in {"concat", "dual"} else 32
        self.trajectory_backbone = TrajectoryBackbone(vocab_size=5, direction_vocab_size=5)
        self.trajectory_cls_head = TrajectoryCLSHead(output_dim=1024)
        self.trajectory_token_projector = TrajectoryConcatHead(output_dim=896)
        self.pretrain_stage = "pretrain"
        self.pretrain_data_source = "./json/question_train.jsonl"
        self.question_format_version = "v1_qa_question_answer"
        self.pretrain_movement_enabled = False


def test_pretrain_bridge_metadata_contains_handoff_fields(tmp_path):
    model = _TinyBridge()
    save_qformer_bridge(model, str(tmp_path))

    metadata = json.loads((tmp_path / BRIDGE_CONFIG_NAME).read_text(encoding="utf-8"))

    assert metadata["stage"] == "pretrain"
    assert metadata["movement_enabled"] is True
    assert metadata["pretrain_data_source"] == "./json/question_train.jsonl"
    assert metadata["question_format_version"] == "v1_qa_question_answer"


def test_pretrain_trajectory_metadata_contains_handoff_fields(tmp_path):
    model = _TinyTrajectory("dual")
    save_trajectory_branch(model, str(tmp_path))

    metadata = json.loads((tmp_path / TRAJECTORY_BRANCH_CONFIG_NAME).read_text(encoding="utf-8"))

    assert metadata["stage"] == "pretrain"
    assert metadata["movement_enabled"] is False
    assert metadata["pretrain_data_source"] == "./json/question_train.jsonl"
    assert metadata["question_format_version"] == "v1_qa_question_answer"


def test_train_py_defines_separate_pretrain_checkpoint_interface():
    content = Path("train.py").read_text(encoding="utf-8")

    assert '--pretrain_checkpoint' in content
    assert 'if args.checkpoint and args.pretrain_checkpoint' in content
    assert 'Loading LoRA adapter from checkpoint' in content


def test_train_pretrain_supports_resume_checkpoint_surface():
    content = Path("train_pretrain.py").read_text(encoding="utf-8")

    assert '--checkpoint' in content or '--resume_checkpoint' in content
    assert 'optimizer.pt' in content
    assert 'scheduler.pt' in content
    assert 'Resuming pretrain' in content or 'resume pretrain' in content.lower()


def test_run_pretrain_notebook_exists_and_targets_pretrain_branch():
    notebook = json.loads(Path("run_pretrain_qformer.ipynb").read_text(encoding="utf-8"))
    cell0 = "".join(notebook["cells"][0]["source"])
    train_cell = "\n".join("".join(c.get("source", [])) for c in notebook["cells"] if c.get("cell_type") == "code")

    assert 'TARGET_BRANCH = "feature/trajectory-pretrain-qformer"' in cell0
    assert 'CONFIG_PATH = "internvl_pretrain_config_traj_cls.yaml"' in cell0
    assert 'train_pretrain.py' in train_cell
