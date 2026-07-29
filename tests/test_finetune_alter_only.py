import importlib
import json
import sys
import types
from pathlib import Path

import pytest
import torch
import yaml

import wad_dataset


ROOT = Path(__file__).resolve().parents[1]


def _load_train_module(monkeypatch):
    fake_peft = types.SimpleNamespace(
        LoraConfig=object,
        PeftModel=types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: None),
        get_peft_model=lambda model, *_args, **_kwargs: model,
        prepare_model_for_kbit_training=lambda model, **_kwargs: model,
    )
    monkeypatch.setitem(sys.modules, "peft", fake_peft)
    sys.modules.pop("train", None)
    return importlib.import_module("train")


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
        assert traj_cfg["dropout"] == 0.10


def test_sequence_loss_supports_cross_entropy_and_label_smoothing(monkeypatch):
    train = _load_train_module(monkeypatch)
    logits = torch.tensor([[[5.0, -1.0, -2.0], [0.1, 0.2, 2.5]]], dtype=torch.float32)
    labels = torch.tensor([[0, 2]], dtype=torch.long)

    ce = train.compute_sequence_loss(
        logits=logits,
        labels=labels,
        loss_mode="cross_entropy",
        label_smoothing=0.0,
    )
    expected_ce = torch.nn.functional.cross_entropy(
        logits[..., :-1, :].contiguous().view(-1, logits.shape[-1]),
        labels[..., 1:].contiguous().view(-1),
        ignore_index=-100,
    )
    smoothed_zero = train.compute_sequence_loss(
        logits=logits,
        labels=labels,
        loss_mode="label_smoothing",
        label_smoothing=0.0,
    )
    smoothed = train.compute_sequence_loss(
        logits=logits,
        labels=labels,
        loss_mode="label_smoothing",
        label_smoothing=0.10,
    )

    assert torch.allclose(ce, expected_ce, atol=1e-6)
    assert torch.allclose(smoothed_zero, ce, atol=1e-6)
    assert torch.isfinite(smoothed)
    assert not torch.allclose(smoothed, ce)


def test_cls_case3_config_combines_label_smoothing_low_lora_and_pretrain_ready_architecture():
    cfg = yaml.safe_load((ROOT / "internvl_config_traj_cls_case3_label_smoothing_low_lora.yaml").read_text(encoding="utf-8"))

    assert cfg["trajectory"]["fusion_mode"] == "cls_add"
    assert cfg["trajectory"]["d_traj"] == 384
    assert cfg["trajectory"]["num_layers"] == 4
    assert cfg["trajectory"]["ffn_dim"] == 768
    assert cfg["trajectory"]["dropout"] == 0.10
    assert cfg["data"]["alter_only"] is True
    assert cfg["training"]["loss_mode"] == "label_smoothing"
    assert cfg["training"]["label_smoothing"] == pytest.approx(0.10)
    assert cfg["training"]["lora_learning_rate"] == pytest.approx(5e-5)
    assert cfg["training"]["bridge_learning_rate"] == pytest.approx(5e-4)
    assert cfg["training"]["trajectory_learning_rate"] == pytest.approx(5e-4)


def test_cls_case3_notebook_keeps_pretrain_checkpoint_surface():
    notebook = json.loads((ROOT / "run_qformer_cls_case3_label_smoothing_low_lora.ipynb").read_text(encoding="utf-8"))
    cell0 = "".join(notebook["cells"][0]["source"])
    train_cell = "".join(notebook["cells"][7]["source"])
    infer_cell = "".join(notebook["cells"][8]["source"])

    assert 'TARGET_BRANCH = "feature/trajectory-pretrain-qformer"' in cell0
    assert 'CONFIG_PATH = "internvl_config_traj_cls_case3_label_smoothing_low_lora.yaml"' in cell0
    assert 'TRAIN_CHECKPOINT = ""' in train_cell
    assert 'PRETRAIN_CHECKPOINT = ""' in train_cell
    assert 'cmd += ["--pretrain_checkpoint", PRETRAIN_CHECKPOINT]' in train_cell
    assert '"--split", "test_alter"' in infer_cell


def test_concat_bestshot_bf16_2gpu_config_disables_4bit_and_enables_accelerate():
    cfg = yaml.safe_load((ROOT / "internvl_config_traj_concat_bestshot_bf16_2gpu.yaml").read_text(encoding="utf-8"))

    assert cfg["trajectory"]["fusion_mode"] == "concat"
    assert cfg["trajectory"]["d_traj"] == 384
    assert cfg["trajectory"]["num_layers"] == 4
    assert cfg["trajectory"]["ffn_dim"] == 768
    assert cfg["trajectory"]["dropout"] == pytest.approx(0.10)
    assert cfg["data"]["alter_only"] is True
    assert cfg["data"]["response_format"] == "direct_text"
    assert cfg["model"]["lora"]["r"] == 32
    assert cfg["model"]["quantization"]["enabled"] is False
    assert cfg["model"]["attn_implementation"] == "flash_attention_2"
    assert cfg["training"]["bf16"] is True
    assert cfg["training"]["fp16"] is False
    assert cfg["training"]["use_accelerate"] is True
    assert cfg["training"]["batch_size"] == 2
    assert cfg["training"]["val_batch_size"] == 8
    assert cfg["training"]["gradient_accumulation_steps"] == 8
    assert cfg["evaluation"]["batch_size"] == 8
    assert cfg["hardware"]["num_workers"] == 4
    assert cfg["hardware"]["pin_memory"] is True
    assert cfg["hardware"]["persistent_workers"] is True
    assert cfg["hardware"]["prefetch_factor"] == 2


def test_concat_bestshot_bf16_2gpu_notebook_keeps_pretrain_checkpoint_and_uses_accelerate():
    notebook = json.loads((ROOT / "run_qformer_concat_bestshot_bf16_2gpu.ipynb").read_text(encoding="utf-8"))
    cell0 = "".join(notebook["cells"][0]["source"])
    train_cell = "".join(notebook["cells"][7]["source"])
    infer_cell = "".join(notebook["cells"][8]["source"])

    assert 'TARGET_BRANCH = "feature/trajectory-pretrain-qformer-concat-bestshot-bf16"' in cell0
    assert 'CONFIG_PATH = "internvl_config_traj_concat_bestshot_bf16_2gpu.yaml"' in cell0
    assert 'TRAIN_CHECKPOINT = ""' in train_cell
    assert 'PRETRAIN_CHECKPOINT = ""' in train_cell
    assert 'cmd += ["--pretrain_checkpoint", PRETRAIN_CHECKPOINT]' in train_cell
    assert 'accelerate", "launch", "--num_processes", "2"' in train_cell
    assert '"--split", "test_alter"' in infer_cell
    assert 'EVAL_ALL_EPOCHS = True' in infer_cell
    assert "glob('epoch_*')" in infer_cell
    assert "Pairs JSON:" in infer_cell


def test_train_source_contains_distributed_runtime_hooks_for_bestshot_concat():
    source = (ROOT / "train.py").read_text(encoding="utf-8")

    assert "from accelerate import Accelerator" in source
    assert 'os.environ.get("LOCAL_RANK"' in source
    assert 'config["training"].get("use_accelerate", False)' in source
    assert "accelerator.prepare(" in source
    assert "accelerator.is_main_process" in source
    assert 'if "attn_implementation" in config["model"]' in source
    assert "log_flash_attention_runtime" in source


def test_train_source_keeps_manual_test_helpers_but_does_not_auto_test_during_train():
    source = (ROOT / "train.py").read_text(encoding="utf-8")

    assert "def run_epoch_test_infer(" in source
    assert "build_test_alter_loader" in source
    assert 'data_files={"test": "test_alter.json"}' in source
    assert "write_prediction_pairs" in source
    assert "Epoch %s test_alter metrics" in source
    assert "Pairs JSON saved at:" in source
    assert "test_summary = run_epoch_test_infer(" not in source
    assert '"test_alter_metrics"' not in source
    assert '"test_alter_output_file"' not in source
    assert '"test_alter_pairs_file"' not in source


def test_train_source_uses_interval_logging_and_suppresses_runtime_noise():
    source = (ROOT / "train.py").read_text(encoding="utf-8")

    assert "def suppress_runtime_noise()" in source
    assert "dynamic ViT batch size:" in source
    assert 'train_log_interval = int(config["training"].get("train_log_interval", 100))' in source
    assert "Train progress | epoch=%s/%s | batch=%s/%s | opt_step=%s | avg_loss=%.4f | lr=%.6g" in source


def test_train_source_supports_separate_val_and_test_batch_sizes_with_accelerate_gather():
    source = (ROOT / "train.py").read_text(encoding="utf-8")

    assert 'val_batch_size = int(config["training"].get("val_batch_size", batch_size))' in source
    assert 'test_batch_size = int(config.get("evaluation", {}).get("batch_size", 1))' in source
    assert "dist.gather_object(" in source
    assert "build_test_alter_loader(config, batch_size=test_batch_size)" in source


def test_train_source_prepares_epoch_test_runtime_once_and_reuses_it():
    source = (ROOT / "train.py").read_text(encoding="utf-8")

    assert "def prepare_epoch_test_runtime(" in source
    assert "epoch_test_runtime=None" in source
    assert "test_loader = epoch_test_runtime[\"test_loader\"]" in source
    assert "evaluator = epoch_test_runtime[\"evaluator\"]" in source


def test_train_and_test_infer_sources_use_batched_generation_helpers():
    train_source = (ROOT / "train.py").read_text(encoding="utf-8")
    infer_source = (ROOT / "scripts" / "test_infer.py").read_text(encoding="utf-8")

    assert "def run_model_batch_chat_for_eval(" in train_source
    assert "responses = run_model_batch_chat_for_eval(" in train_source
    assert "for sample in batch:" not in train_source.split("def run_epoch_test_infer(", 1)[1].split("def eval_model(", 1)[0]

    assert "def run_model_batch_chat(" in infer_source
    assert "responses = run_model_batch_chat(" in infer_source
    assert "for sample in batch:" not in infer_source.split("with torch.no_grad():", 1)[1].split("# 5. Compute Metrics", 1)[0]


def test_test_infer_uses_configurable_batch_size_and_not_hardcoded_one():
    source = (ROOT / "scripts" / "test_infer.py").read_text(encoding="utf-8")

    assert "from tqdm import tqdm" in source
    assert 'parser.add_argument("--batch_size"' in source
    assert 'test_batch_size = int(args.batch_size or config.get("evaluation", {}).get("batch_size", 1))' in source
    assert "batch_size=test_batch_size" in source
    assert 'for batch in tqdm(test_loader, desc="Testing"):' in source


def test_wad_dataset_train_builder_does_not_load_test_alter_with_train_schema():
    source = (ROOT / "wad_dataset.py").read_text(encoding="utf-8")

    assert 'data_files={"train": "train.json"}' in source or '"train": "train.json"' in source
