import json
import sys
import types
from pathlib import Path

import pytest
import torch

from pretrain_checkpoint_verify import verify_loaded_pretrain_modules
from qformer_bridge import BRIDGE_CONFIG_NAME, save_qformer_bridge
from trajectory_branch import TRAJECTORY_BRANCH_CONFIG_NAME, save_trajectory_branch
from trajectory_trainability import apply_mode_gated_trajectory_trainability


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
    assert '"last"' in content or "'last'" in content
    assert 'training_state.json' in content
    assert 'early_stopping_state.json' in content
    assert 'best_val_loss' in content


def test_hf_resume_allow_patterns_include_state_files():
    content = Path("train_pretrain.py").read_text(encoding="utf-8")

    assert 'training_state.json' in content
    assert 'early_stopping_state.json' in content


def test_run_pretrain_notebook_exists_and_targets_pretrain_branch():
    notebook = json.loads(Path("run_pretrain_qformer.ipynb").read_text(encoding="utf-8"))
    cell0 = "".join(notebook["cells"][0]["source"])
    train_cell = "\n".join("".join(c.get("source", [])) for c in notebook["cells"] if c.get("cell_type") == "code")

    assert 'TARGET_BRANCH = "feature/trajectory-pretrain-qformer"' in cell0
    assert 'CONFIG_PATH = "internvl_pretrain_config_traj_cls.yaml"' in cell0
    assert 'train_pretrain.py' in train_cell


class _TinyCombined(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qformer_enabled = True
        self.qformer_source_model = "dummy-source"
        self.qformer_num_query_tokens = 32
        self.trajectory_enabled = True
        self.qformer_input_proj = torch.nn.Linear(4, 4)
        self.qformer_to_mlp1_proj = torch.nn.Linear(4, 4)
        self.trajectory_fusion_mode = "dual"
        self.trajectory_source_file = "json/results_botsort_top6_sorted.jsonl"
        self.trajectory_num_objects = 6
        self.trajectory_qformer_token_count = 32
        self.num_image_token = 38
        self.trajectory_backbone = torch.nn.Linear(4, 4)
        self.trajectory_cls_head = torch.nn.Linear(4, 4)
        self.trajectory_token_projector = torch.nn.Linear(4, 4)


def _projector_param_names(model):
    proj_param_names = {
        "qformer_input_proj",
        "qformer_to_mlp1_proj",
        "trajectory_backbone",
        "trajectory_cls_head",
        "trajectory_token_projector",
    }
    return [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and any(group_name in name for group_name in proj_param_names)
    ]


@pytest.mark.parametrize(
    ("fusion_mode", "expected_present", "expected_absent"),
    [
        ("cls_add", ["trajectory_backbone", "trajectory_cls_head"], ["trajectory_token_projector"]),
        ("concat", ["trajectory_backbone", "trajectory_token_projector"], ["trajectory_cls_head"]),
        ("dual", ["trajectory_backbone", "trajectory_cls_head", "trajectory_token_projector"], []),
    ],
)
def test_mode_gated_trajectory_trainability_controls_optimizer_membership(
    fusion_mode,
    expected_present,
    expected_absent,
):
    model = _TinyCombined()
    model.trajectory_fusion_mode = fusion_mode

    states = apply_mode_gated_trajectory_trainability(model)
    optimizer_like_names = _projector_param_names(model)

    for module_name in expected_present:
        assert states[module_name] is True
        assert any(module_name in name for name in optimizer_like_names)
    for module_name in expected_absent:
        assert states[module_name] is False
        assert not any(module_name in name for name in optimizer_like_names)


def test_finetune_optimizer_groups_split_lora_bridge_and_trajectory_lrs(monkeypatch):
    import importlib

    fake_peft = types.SimpleNamespace(
        LoraConfig=object,
        PeftModel=types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: None),
        get_peft_model=lambda model, *_args, **_kwargs: model,
        prepare_model_for_kbit_training=lambda model, **_kwargs: model,
    )
    monkeypatch.setitem(sys.modules, "peft", fake_peft)
    sys.modules.pop("train", None)
    train = importlib.import_module("train")
    model = _TinyCombined()
    model.trajectory_fusion_mode = "cls_add"
    model.language_model = torch.nn.Module()
    model.language_model.lora_A = torch.nn.Linear(4, 4, bias=False)
    model.language_model.lora_B = torch.nn.Linear(4, 4, bias=False)

    apply_mode_gated_trajectory_trainability(model)
    groups = train.build_optimizer_param_groups(
        model,
        lora_lr=5e-5,
        bridge_lr=5e-4,
        trajectory_lr=5e-4,
    )

    names_by_group = {group["name"]: set(group["param_names"]) for group in groups}
    lrs = {group["name"]: group["lr"] for group in groups}
    grouped_param_ids = [id(param) for group in groups for param in group["params"]]

    assert lrs["trajectory"] == pytest.approx(5e-4)
    assert lrs["bridge"] == pytest.approx(5e-4)
    assert lrs["lora_rest"] == pytest.approx(5e-5)
    assert any("trajectory_backbone" in name for name in names_by_group["trajectory"])
    assert any("trajectory_cls_head" in name for name in names_by_group["trajectory"])
    assert not any("trajectory_token_projector" in name for name in names_by_group["trajectory"])
    assert any("qformer_input_proj" in name for name in names_by_group["bridge"])
    assert any("qformer_to_mlp1_proj" in name for name in names_by_group["bridge"])
    assert any("lora_A" in name for name in names_by_group["lora_rest"])
    assert len(grouped_param_ids) == len(set(grouped_param_ids))


def test_verify_loaded_pretrain_modules_detects_exact_tensor_match(tmp_path):
    model = _TinyCombined()
    save_qformer_bridge(model, str(tmp_path))
    save_trajectory_branch(model, str(tmp_path))

    result = verify_loaded_pretrain_modules(model, str(tmp_path))

    assert result["bridge"]["all_matched"] is True
    assert result["trajectory"]["all_matched"] is True
    assert result["bridge"]["checked_keys"]
    assert result["trajectory"]["checked_keys"]


def test_verify_loaded_pretrain_modules_detects_mismatch(tmp_path):
    model = _TinyCombined()
    save_qformer_bridge(model, str(tmp_path))
    save_trajectory_branch(model, str(tmp_path))
    with torch.no_grad():
        model.qformer_input_proj.weight.add_(1.0)

    result = verify_loaded_pretrain_modules(model, str(tmp_path))

    assert result["bridge"]["all_matched"] is False
    assert any(not item["matched"] for item in result["bridge"]["checked_keys"])
