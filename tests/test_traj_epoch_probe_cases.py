import copy
import importlib
import importlib.machinery
import json
import sys
import types
from pathlib import Path

import pytest
import torch
import yaml

import wad_dataset


ROOT = Path(__file__).resolve().parents[1]


def _load_train_module():
    train_path = ROOT / "train.py"
    old_argv = sys.argv[:]
    old_modules = {}
    fake_peft = types.ModuleType("peft")
    fake_peft.__spec__ = importlib.machinery.ModuleSpec("peft", loader=None)
    fake_peft.LoraConfig = object
    fake_peft.PeftModel = object
    fake_peft.get_peft_model = lambda model, config: model
    fake_peft.prepare_model_for_kbit_training = lambda model: model
    for name, module in {"peft": fake_peft}.items():
        old_modules[name] = sys.modules.get(name)
        sys.modules[name] = module
    sys.argv = ["train.py", "--config", str(ROOT / "internvl_config_traj_cls.yaml")]
    try:
        spec = importlib.util.spec_from_file_location("train_under_test", train_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.argv = old_argv
        for name, module in old_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_alter_only_row_helpers_filter_out_qa_and_report_counts():
    rows = [
        {"frame_path": "a", "answer": "alter only sample"},
        {"frame_path": "b", "QA": {"Q": "What is ahead?"}, "answer": "qa sample"},
        {"frame_path": "c", "QA": {"Q": ""}, "answer": "empty qa should still count as non-qa"},
    ]

    stats = wad_dataset.summarize_qa_rows(rows)
    filtered = wad_dataset.filter_alter_only_rows(rows)

    assert stats == {"total": 3, "with_qa": 1, "without_qa": 2}
    assert [row["frame_path"] for row in filtered] == ["a", "c"]


def test_label_smoothed_loss_matches_cross_entropy_when_smoothing_zero():
    train = _load_train_module()
    logits = torch.tensor([[[3.0, 0.5, -1.0], [0.1, 0.2, 1.7]]], dtype=torch.float32)
    labels = torch.tensor([[0, 2]], dtype=torch.long)

    ce = torch.nn.functional.cross_entropy(
        logits[..., :-1, :].contiguous().view(-1, logits.shape[-1]),
        labels[..., 1:].contiguous().view(-1),
        ignore_index=-100,
    )
    smoothed = train.compute_sequence_loss(
        logits=logits,
        labels=labels,
        loss_mode="label_smoothing",
        label_smoothing=0.0,
    )

    assert torch.allclose(smoothed, ce, atol=1e-6)


def test_cross_entropy_loss_uses_causal_shift_like_model_forward():
    train = _load_train_module()
    logits = torch.tensor(
        [[[4.0, 0.1, -2.0], [0.2, 3.5, -1.0], [0.1, -0.3, 2.8]]],
        dtype=torch.float32,
    )
    labels = torch.tensor([[-100, 1, 2]], dtype=torch.long)

    expected = torch.nn.functional.cross_entropy(
        logits[..., :-1, :].contiguous().view(-1, logits.shape[-1]),
        labels[..., 1:].contiguous().view(-1),
        ignore_index=-100,
    )
    actual = train.compute_sequence_loss(
        logits=logits,
        labels=labels,
        loss_mode="cross_entropy",
        label_smoothing=0.0,
    )

    assert torch.allclose(actual, expected, atol=1e-6)


def test_label_smoothed_loss_is_finite_and_differs_from_ce_for_positive_smoothing():
    train = _load_train_module()
    logits = torch.tensor([[[5.0, -1.0, -2.0], [0.1, 0.2, 2.5]]], dtype=torch.float32)
    labels = torch.tensor([[0, 2]], dtype=torch.long)

    ce = train.compute_sequence_loss(
        logits=logits,
        labels=labels,
        loss_mode="cross_entropy",
        label_smoothing=0.0,
    )
    smoothed = train.compute_sequence_loss(
        logits=logits,
        labels=labels,
        loss_mode="label_smoothing",
        label_smoothing=0.10,
    )

    assert torch.isfinite(smoothed)
    assert not torch.allclose(smoothed, ce)


class _TinyTrainableModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qformer_input_proj = torch.nn.Linear(4, 4)
        self.qformer_to_mlp1_proj = torch.nn.Linear(4, 4)
        self.trajectory_backbone = torch.nn.Linear(4, 4)
        self.trajectory_cls_head = torch.nn.Linear(4, 4)
        self.trajectory_token_projector = torch.nn.Linear(4, 4)
        self.language_model = torch.nn.Module()
        self.language_model.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4)])
        self.language_model.layers[0].lora_A = torch.nn.Linear(4, 4, bias=False)
        self.language_model.layers[0].lora_B = torch.nn.Linear(4, 4, bias=False)
        self.frozen = torch.nn.Linear(4, 4)
        for p in self.frozen.parameters():
            p.requires_grad = False


def test_optimizer_groups_split_lora_bridge_and_trajectory_with_distinct_lrs():
    train = _load_train_module()
    model = _TinyTrainableModel()

    grouped = train.build_optimizer_param_groups(
        model,
        lora_lr=5e-5,
        bridge_lr=5e-4,
        trajectory_lr=5e-4,
    )

    assert [group["name"] for group in grouped] == ["trajectory", "bridge", "lora"]
    assert grouped[0]["lr"] == pytest.approx(5e-4)
    assert grouped[1]["lr"] == pytest.approx(5e-4)
    assert grouped[2]["lr"] == pytest.approx(5e-5)

    trajectory_ids = {id(p) for p in grouped[0]["params"]}
    bridge_ids = {id(p) for p in grouped[1]["params"]}
    lora_ids = {id(p) for p in grouped[2]["params"]}
    assert trajectory_ids
    assert bridge_ids
    assert lora_ids
    assert trajectory_ids.isdisjoint(bridge_ids)
    assert trajectory_ids.isdisjoint(lora_ids)
    assert bridge_ids.isdisjoint(lora_ids)


@pytest.mark.parametrize(
    ("mode", "expected_trainable", "expected_frozen"),
    [
        ("cls_add", {"trajectory_backbone", "trajectory_cls_head"}, {"trajectory_token_projector"}),
        ("concat", {"trajectory_backbone", "trajectory_token_projector"}, {"trajectory_cls_head"}),
        ("dual", {"trajectory_backbone", "trajectory_cls_head", "trajectory_token_projector"}, set()),
    ],
)
def test_mode_gated_trajectory_trainability_controls_optimizer_membership(mode, expected_trainable, expected_frozen):
    train = _load_train_module()
    model = _TinyTrainableModel()
    model.trajectory_enabled = True
    model.trajectory_fusion_mode = mode

    train.apply_mode_gated_trajectory_trainability(model)
    grouped = train.build_optimizer_param_groups(
        model,
        lora_lr=5e-5,
        bridge_lr=5e-4,
        trajectory_lr=5e-4,
    )

    trajectory_param_ids = {id(param) for param in grouped[0]["params"]}
    for module_name in expected_trainable:
        module_params = list(getattr(model, module_name).parameters())
        assert all(param.requires_grad for param in module_params)
        assert any(id(param) in trajectory_param_ids for param in module_params)
    for module_name in expected_frozen:
        module_params = list(getattr(model, module_name).parameters())
        assert all(not param.requires_grad for param in module_params)
        assert all(id(param) not in trajectory_param_ids for param in module_params)


@pytest.mark.parametrize(
    ("config_name", "mode"),
    [
        ("internvl_config_traj_cls.yaml", "cls_add"),
        ("internvl_config_traj_concat.yaml", "concat"),
        ("internvl_config_traj_dual.yaml", "dual"),
    ],
)
def test_baseline_configs_enable_alter_only_and_large_trajectory(config_name, mode):
    cfg = yaml.safe_load((ROOT / config_name).read_text(encoding="utf-8"))

    assert cfg["trajectory"]["fusion_mode"] == mode
    assert cfg["trajectory"]["d_traj"] == 384
    assert cfg["trajectory"]["num_layers"] == 4
    assert cfg["trajectory"]["ffn_dim"] == 768
    assert cfg["data"]["alter_only"] is True


def test_case1_config_targets_label_smoothing_only():
    cfg = yaml.safe_load((ROOT / "internvl_config_traj_cls_case1_label_smoothing.yaml").read_text(encoding="utf-8"))

    assert cfg["trajectory"]["fusion_mode"] == "cls_add"
    assert cfg["data"]["alter_only"] is True
    assert cfg["training"]["loss_mode"] == "label_smoothing"
    assert cfg["training"]["label_smoothing"] == pytest.approx(0.10)
    assert cfg["training"]["lora_learning_rate"] == pytest.approx(2e-4)
    assert cfg["training"]["bridge_learning_rate"] == pytest.approx(5e-4)
    assert cfg["training"]["trajectory_learning_rate"] == pytest.approx(5e-4)


def test_case2_config_targets_low_lora_lr_only():
    cfg = yaml.safe_load((ROOT / "internvl_config_traj_cls_case2_low_lora_lr.yaml").read_text(encoding="utf-8"))

    assert cfg["trajectory"]["fusion_mode"] == "cls_add"
    assert cfg["data"]["alter_only"] is True
    assert cfg["training"]["loss_mode"] == "cross_entropy"
    assert cfg["training"]["lora_learning_rate"] == pytest.approx(5e-5)
    assert cfg["training"]["bridge_learning_rate"] == pytest.approx(5e-4)
    assert cfg["training"]["trajectory_learning_rate"] == pytest.approx(5e-4)


@pytest.mark.parametrize(
    ("notebook_name", "config_name"),
    [
        ("run_qformer_cls_case1_label_smoothing.ipynb", "internvl_config_traj_cls_case1_label_smoothing.yaml"),
        ("run_qformer_cls_case2_low_lora_lr.ipynb", "internvl_config_traj_cls_case2_low_lora_lr.yaml"),
    ],
)
def test_new_cls_case_notebooks_point_to_expected_branch_and_config(notebook_name, config_name):
    notebook = json.loads((ROOT / notebook_name).read_text(encoding="utf-8"))
    cell0 = "".join(notebook["cells"][0]["source"])
    cell1 = "".join(notebook["cells"][1]["source"])
    train_cell = "".join(notebook["cells"][7]["source"])
    infer_cell = "".join(notebook["cells"][8]["source"])

    assert 'TARGET_BRANCH = "feature/trajectory-qformer-restore-779"' in cell0
    assert f'CONFIG_PATH = "{config_name}"' in cell0
    assert 'subprocess.run(["git", "fetch", "origin", TARGET_BRANCH], check=True)' in cell1
    assert 'subprocess.run(["git", "checkout", "-B", TARGET_BRANCH, f"origin/{TARGET_BRANCH}"], check=True)' in cell1
    assert 'cmd = ["python", "train.py", "--config", CONFIG_PATH]' in train_cell
    assert '"--config", CONFIG_PATH' in infer_cell
