import json
import logging
from io import StringIO
from types import SimpleNamespace

import torch

import train_pretrain
from train_pretrain import (
    EarlyStoppingState,
    PretrainCollaterFn,
    _active_trajectory_heads,
    _build_optimizer,
    _freeze_modules_for_pretrain,
    _build_run_metadata,
    _write_training_state,
    build_dataloader_kwargs,
    compute_generation_loss,
    forward_pretrain_batch,
    get_cuda_memory_stats,
    infer_resume_position,
    inspect_optimizer_param_groups,
    _DynamicVitDebugFilter,
    log_trainable_parameter_summary,
    log_trajectory_dropout_summary,
    log_trajectory_path_gradients,
    log_runtime_batch_debug,
    reduce_token_weighted_loss,
    resolve_warmup_steps,
    summarize_runtime_batch,
    suppress_loaded_model_debug_logs,
    verify_flash_attention_runtime,
)


class _DummyModel(torch.nn.Module):
    def __init__(self, fusion_mode: str = "cls_add"):
        super().__init__()
        self.template = "internlm2-chat"
        self.system_message = "You are a navigation assistant."
        self.num_image_token = 32
        self.qformer_enabled = True
        self.trajectory_enabled = True
        self.trajectory_fusion_mode = fusion_mode
        self.qformer_calls = []
        self.trajectory_calls = []
        self.qformer = torch.nn.Linear(4, 4)
        self.mlp1 = torch.nn.Linear(4, 4)
        self.qformer_input_proj = torch.nn.Linear(4, 4)
        self.qformer_to_mlp1_proj = torch.nn.Linear(4, 4)
        self.trajectory_backbone = torch.nn.Linear(4, 4)
        self.trajectory_cls_head = torch.nn.Linear(4, 4)
        self.trajectory_token_projector = torch.nn.Linear(4, 4)
        self.other_trainable = torch.nn.Linear(4, 4)

    def encode_qformer_texts(self, texts, device=None):
        self.qformer_calls.append(list(texts))
        return (
            torch.ones(len(texts), 5, dtype=torch.long),
            torch.ones(len(texts), 5, dtype=torch.long),
        )

    def set_qformer_text(self, ids, mask):
        self.last_qformer = (ids, mask)

    def clear_qformer_text(self):
        self.last_qformer = None

    def set_trajectory_inputs(self, label_ids, direction_ids, numeric_feats, object_mask):
        self.trajectory_calls.append((label_ids, direction_ids, numeric_feats, object_mask))

    def clear_trajectory_inputs(self):
        self.last_trajectory = None

    def __call__(self, input_ids, pixel_values, labels, image_flags, return_dict=True):
        loss = self.other_trainable.weight.sum() * 0 + pixel_values.float().mean() + 1.0
        return SimpleNamespace(loss=loss)


class _DummyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [len(text), max(len(text) - 1, 1)]

    def convert_tokens_to_ids(self, token):
        mapping = {"<|endoftext|>": 7}
        return mapping.get(token, 3)


def _make_sample(question_text: str):
    return {
        "question": f"<image>\nQuestion: {question_text}",
        "answer": "Answer: yes",
        "qformer_text": f"Question: {question_text}",
        "pixel_values": [torch.zeros(1, 3, 4, 4)],
        "trajectory_label_ids": torch.tensor([1, 2, 0, 0, 0, 0], dtype=torch.long),
        "trajectory_direction_ids": torch.tensor([3, 4, 0, 0, 0, 0], dtype=torch.long),
        "trajectory_numeric_feats": torch.ones(6, 6, dtype=torch.float32),
        "trajectory_object_mask": torch.tensor([1, 1, 0, 0, 0, 0], dtype=torch.long),
    }


def test_pretrain_collater_uses_question_for_qformer_text():
    model = _DummyModel()
    tokenizer = _DummyTokenizer()
    collater = PretrainCollaterFn(tokenizer, model)

    batch = collater([_make_sample("What is closest?")])
    _, _, _, _, qformer_inputs, trajectory_inputs, samples = batch

    assert model.qformer_calls == [["Question: What is closest?"]]
    assert qformer_inputs[0].shape[0] == 1
    assert trajectory_inputs[0].shape == (1, 6)
    assert samples[0]["answer"] == "Answer: yes"


def test_forward_pretrain_batch_runs_and_clears_runtime_state():
    model = _DummyModel()
    batch = (
        torch.ones(1, 4, dtype=torch.long),
        torch.ones(1, 4, dtype=torch.long),
        torch.ones(1, 4, dtype=torch.long),
        torch.zeros(1, 3, 4, 4),
        (torch.ones(1, 5, dtype=torch.long), torch.ones(1, 5, dtype=torch.long)),
        (
            torch.ones(1, 6, dtype=torch.long),
            torch.ones(1, 6, dtype=torch.long),
            torch.ones(1, 6, 6, dtype=torch.float32),
            torch.ones(1, 6, dtype=torch.long),
        ),
        [],
    )

    loss = forward_pretrain_batch(model, batch, device=torch.device("cpu"))

    assert torch.isfinite(loss)
    assert len(model.trajectory_calls) == 1


def test_forward_pretrain_batch_sets_aux_inputs_on_inner_runtime_model():
    inner = _DummyModel()

    class _WrappedModel:
        def __init__(self, module):
            self.module = module
            self.qformer_enabled = True
            self.trajectory_enabled = True
            self.wrapper_qformer_calls = 0
            self.wrapper_trajectory_calls = 0

        def set_qformer_text(self, ids, mask):
            self.wrapper_qformer_calls += 1

        def clear_qformer_text(self):
            return None

        def set_trajectory_inputs(self, label_ids, direction_ids, numeric_feats, object_mask):
            self.wrapper_trajectory_calls += 1

        def clear_trajectory_inputs(self):
            return None

        def __call__(self, *args, **kwargs):
            return self.module(*args, **kwargs)

    wrapped = _WrappedModel(inner)
    batch = (
        torch.ones(1, 4, dtype=torch.long),
        torch.ones(1, 4, dtype=torch.long),
        torch.ones(1, 4, dtype=torch.long),
        torch.zeros(1, 3, 4, 4),
        (torch.ones(1, 5, dtype=torch.long), torch.ones(1, 5, dtype=torch.long)),
        (
            torch.ones(1, 6, dtype=torch.long),
            torch.ones(1, 6, dtype=torch.long),
            torch.ones(1, 6, 6, dtype=torch.float32),
            torch.ones(1, 6, dtype=torch.long),
        ),
        [],
    )

    loss = forward_pretrain_batch(wrapped, batch, device=torch.device("cpu"))

    assert torch.isfinite(loss)
    assert len(inner.trajectory_calls) == 1
    assert wrapped.wrapper_trajectory_calls == 0


def test_compute_generation_loss_uses_label_smoothing_when_enabled():
    logits = torch.tensor(
        [
            [
                [4.0, 0.0, 0.0],
                [0.0, 4.0, 0.0],
                [0.0, 0.0, 4.0],
            ]
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([[0, 1, -100]], dtype=torch.long)
    outputs = SimpleNamespace(loss=torch.tensor(0.0), logits=logits)

    smoothed = compute_generation_loss(outputs, labels, loss_mode="label_smoothing", label_smoothing=0.10)
    plain = compute_generation_loss(outputs, labels, loss_mode="label_smoothing", label_smoothing=0.0)

    assert smoothed > plain
    assert torch.isfinite(smoothed)


def test_pretrain_collater_logs_total_input_tokens(monkeypatch):
    model = _DummyModel()
    tokenizer = _DummyTokenizer()
    collater = PretrainCollaterFn(tokenizer, model)
    collater.log_token_stats = True
    collater.token_log_remaining = 1

    messages = []

    class _FakeLogger:
        def info(self, msg, *args):
            messages.append(msg % args if args else msg)

    monkeypatch.setattr(train_pretrain, "get_logger", lambda: _FakeLogger())

    collater([_make_sample("What is closest?")])

    assert any("total_input_tokens" in message for message in messages)


def test_pretrain_collater_skips_token_logging_when_logger_is_uninitialized(monkeypatch):
    model = _DummyModel()
    tokenizer = _DummyTokenizer()
    collater = PretrainCollaterFn(tokenizer, model)
    collater.log_token_stats = True
    collater.token_log_remaining = 1

    monkeypatch.setattr(
        train_pretrain,
        "get_logger",
        lambda: (_ for _ in ()).throw(AssertionError("Logger is not initialized.")),
    )

    batch = collater([_make_sample("What is closest?")])

    assert batch[0].shape[0] == 1
    assert collater.token_log_remaining == 0


def test_log_runtime_batch_debug_includes_trajectory_path_debug(monkeypatch):
    model = _DummyModel()
    model._last_trajectory_debug = {"fusion_mode": "cls_add", "traj_cls_requires_grad": True}
    batch = (
        torch.ones(1, 4, dtype=torch.long),
        torch.ones(1, 4, dtype=torch.long),
        torch.ones(1, 4, dtype=torch.long),
        torch.zeros(1, 3, 4, 4),
        (torch.ones(1, 5, dtype=torch.long), torch.ones(1, 5, dtype=torch.long)),
        (
            torch.ones(1, 6, dtype=torch.long),
            torch.ones(1, 6, dtype=torch.long),
            torch.ones(1, 6, 6, dtype=torch.float32),
            torch.ones(1, 6, dtype=torch.long),
        ),
        [],
    )
    messages = []

    class _FakeLogger:
        def info(self, msg, *args):
            messages.append(msg % args if args else msg)

    monkeypatch.setattr(train_pretrain, "get_cuda_memory_stats", lambda device: {"allocated_mb": 0.0})

    log_runtime_batch_debug(batch, model, torch.device("cpu"), _FakeLogger(), global_step=1, phase="pre_forward")

    assert any("Trajectory path debug" in message for message in messages)


def test_log_trajectory_path_gradients_reports_tensor_gradients():
    messages = []

    class _FakeLogger:
        def info(self, msg, *args):
            messages.append(msg % args if args else msg)

    model = _DummyModel()
    source = torch.tensor([[1.0, -2.0]], requires_grad=True)
    loss = (source ** 2).sum()
    loss.backward()
    model._last_trajectory_debug = {"fusion_mode": "cls_add"}
    model._last_trajectory_debug_tensors = {"traj_cls": source}

    log_trajectory_path_gradients(model, _FakeLogger())

    assert any("Trajectory grad path" in message for message in messages)
    assert any("grad_norm" in message for message in messages)


def test_resolve_warmup_steps_prefers_ratio_policy():
    config = {
        "training": {
            "warmup_ratio": 0.1,
            "warmup_min_steps": 20,
            "warmup_max_steps": 100,
        }
    }

    assert resolve_warmup_steps(config, total_training_steps=50) == 20
    assert resolve_warmup_steps(config, total_training_steps=1000) == 100


def test_build_optimizer_separates_trajectory_and_bridge_groups():
    model = _DummyModel()
    config = {
        "training": {
            "learning_rate": 1e-5,
            "trajectory_learning_rate": 1e-4,
            "bridge_learning_rate": 3e-5,
            "weight_decay": 0.01,
        }
    }

    class _FakeLogger:
        def info(self, *args, **kwargs):
            return None

    optimizer = _build_optimizer(model, config, _FakeLogger())

    lrs = sorted(group["lr"] for group in optimizer.param_groups)
    assert lrs == [1e-05, 3e-05, 1e-04]


def test_log_trainable_parameter_summary_is_compact_by_default():
    model = _DummyModel()
    messages = []

    class _FakeLogger:
        def info(self, msg, *args):
            messages.append(msg % args if args else msg)

    log_trainable_parameter_summary(model, _FakeLogger(), include_names=False)

    joined = "\n".join(messages)
    assert "TOTAL_TRAINABLE" in joined
    assert "trajectory_backbone.weight" not in joined


def test_log_trainable_parameter_summary_can_dump_names():
    model = _DummyModel()
    messages = []

    class _FakeLogger:
        def info(self, msg, *args):
            messages.append(msg % args if args else msg)

    log_trainable_parameter_summary(model, _FakeLogger(), include_names=True)

    joined = "\n".join(messages)
    assert "trajectory_backbone.weight" in joined
    assert "TOTAL_TRAINABLE" in joined


def test_log_trajectory_dropout_summary_reports_config_and_modules():
    model = SimpleNamespace(
        trajectory_backbone=torch.nn.Sequential(torch.nn.Dropout(0.05), torch.nn.Linear(2, 2)),
        trajectory_cls_head=torch.nn.Dropout(0.1),
        trajectory_token_projector=torch.nn.Identity(),
    )
    messages = []

    class _FakeLogger:
        def info(self, msg, *args):
            messages.append(msg % args if args else msg)

    log_trajectory_dropout_summary(model, {"trajectory": {"dropout": 0.05}}, _FakeLogger())

    joined = "\n".join(messages)
    assert "Trajectory dropout config" in joined
    assert "trajectory.dropout=0.0500" in joined
    assert "trajectory_backbone:dropout_modules=1" in joined
    assert "trajectory_cls_head:dropout_modules=1" in joined
    assert "trajectory_token_projector:dropout_modules=0" in joined


def test_freeze_modules_for_pretrain_cls_add_only_enables_cls_head():
    model = _DummyModel("cls_add")

    _freeze_modules_for_pretrain(model)

    heads = _active_trajectory_heads(model)
    assert heads == {
        "trajectory_cls_head": True,
        "trajectory_token_projector": False,
    }


def test_freeze_modules_for_pretrain_concat_only_enables_token_projector():
    model = _DummyModel("concat")

    _freeze_modules_for_pretrain(model)

    heads = _active_trajectory_heads(model)
    assert heads == {
        "trajectory_cls_head": False,
        "trajectory_token_projector": True,
    }


def test_freeze_modules_for_pretrain_dual_enables_both_heads():
    model = _DummyModel("dual")

    _freeze_modules_for_pretrain(model)

    heads = _active_trajectory_heads(model)
    assert heads == {
        "trajectory_cls_head": True,
        "trajectory_token_projector": True,
    }


def test_inspect_optimizer_param_groups_detects_missing_trainable_params():
    model = _DummyModel()
    config = {
        "training": {
            "learning_rate": 1e-5,
            "trajectory_learning_rate": 1e-4,
            "bridge_learning_rate": 3e-5,
            "weight_decay": 0.01,
        }
    }

    class _FakeLogger:
        def info(self, *args, **kwargs):
            return None

    optimizer = _build_optimizer(model, config, _FakeLogger())
    removed = optimizer.param_groups[0]["params"].pop()

    report = inspect_optimizer_param_groups(model, optimizer)

    assert removed.requires_grad
    assert report["missing_param_names"]
    assert report["duplicate_param_names"] == []


def test_build_dataloader_kwargs_enables_server_worker_options():
    cfg = {
        "hardware": {
            "num_workers": 4,
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 3,
        }
    }

    kwargs = build_dataloader_kwargs(cfg)

    assert kwargs == {
        "num_workers": 4,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 3,
    }


def test_build_dataloader_kwargs_omits_worker_only_options_when_workers_zero():
    cfg = {
        "hardware": {
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": True,
            "prefetch_factor": 3,
        }
    }

    kwargs = build_dataloader_kwargs(cfg)

    assert kwargs == {"num_workers": 0, "pin_memory": False}


def test_reduce_token_weighted_loss_uses_token_counts():
    loss_sum = torch.tensor(12.0)
    token_count = torch.tensor(6.0)

    assert reduce_token_weighted_loss(loss_sum, token_count, accelerator=None).item() == 2.0


def test_get_cuda_memory_stats_returns_zeroes_on_cpu():
    stats = get_cuda_memory_stats(torch.device("cpu"))

    assert stats == {
        "allocated_mb": 0.0,
        "reserved_mb": 0.0,
        "max_allocated_mb": 0.0,
        "max_reserved_mb": 0.0,
    }


def test_summarize_runtime_batch_reports_tiles_and_tokens():
    model = _DummyModel()
    tokenizer = _DummyTokenizer()
    collater = PretrainCollaterFn(tokenizer, model)
    batch = collater([_make_sample("What is closest?")])

    summary = summarize_runtime_batch(batch, model)

    assert summary["samples"] == 1
    assert summary["frames"] == 1
    assert summary["tiles_per_frame"] == [1]
    assert summary["total_tiles"] == 1
    assert summary["query_tokens_per_tile"] == 32
    assert summary["total_image_tokens"] == 32
    assert summary["max_input_tokens"] > 0
    assert summary["max_target_tokens"] > 0


def test_log_runtime_batch_debug_includes_memory_and_token_summary():
    model = _DummyModel()
    tokenizer = _DummyTokenizer()
    collater = PretrainCollaterFn(tokenizer, model)
    batch = collater([_make_sample("What is closest?")])
    messages = []

    class _FakeLogger:
        def info(self, msg, *args):
            messages.append(msg % args if args else msg)

    log_runtime_batch_debug(batch, model, torch.device("cpu"), _FakeLogger(), global_step=7, phase="pre_forward")

    joined = "\n".join(messages)
    assert "Runtime batch debug" in joined
    assert "global_step=7" in joined
    assert "total_image_tokens=32" in joined
    assert "cuda_mem={'allocated_mb': 0.0" in joined


def test_build_run_metadata_contains_reproducibility_fields(monkeypatch):
    monkeypatch.setattr(train_pretrain, "_git_commit", lambda: "abc123")
    monkeypatch.setattr(train_pretrain, "_package_versions", lambda: {"torch": "x", "transformers": "y"})

    metadata = _build_run_metadata(
        config={"training": {"batch_size": 4}},
        global_optimizer_step=9,
    )

    assert metadata["global_optimizer_step"] == 9
    assert metadata["git_commit"] == "abc123"
    assert metadata["package_versions"] == {"torch": "x", "transformers": "y"}
    assert metadata["config_snapshot"]["training"]["batch_size"] == 4


def test_verify_flash_attention_runtime_reports_flash_and_fallback_modules():
    class _Attn(torch.nn.Module):
        def __init__(self, use_flash):
            super().__init__()
            self.use_flash_attn = use_flash
            self.config = SimpleNamespace(use_flash_attn=True)

    model = torch.nn.Module()
    model.flash = _Attn(True)
    model.fallback = _Attn(False)

    report = verify_flash_attention_runtime(model)

    assert report["supported_count"] == 2
    assert report["flash_enabled_count"] == 1
    assert any(item["status"] == "flash" for item in report["modules"])
    assert any(item["status"] == "fallback" for item in report["modules"])


def test_suppress_loaded_model_debug_logs_sets_model_module_to_warning():
    logger_name = "fake_remote_modeling"
    logging.getLogger(logger_name).setLevel(logging.DEBUG)
    messages = []

    class _RemoteClass:
        __module__ = logger_name

    model = _RemoteClass()

    class _FakeLogger:
        def info(self, msg, *args):
            messages.append(msg % args if args else msg)

    suppress_loaded_model_debug_logs(model, _FakeLogger())

    assert logging.getLogger(logger_name).level == logging.WARNING
    assert any("InternVL model code path" in message for message in messages)


def test_dynamic_vit_debug_filter_drops_entire_forbidden_write():
    stream = StringIO()
    filtered = _DynamicVitDebugFilter(stream)

    filtered.write("Pretrain epoch 1 loss=3.0]dynamic ViT batch size: 72, images per sample: 3.0")
    filtered.write("\nnext line")

    assert stream.getvalue() == "next line"


def test_early_stopping_state_tracks_best_and_stops_after_patience():
    state = EarlyStoppingState(patience=2, min_delta=0.005)

    improved, should_stop = state.update(epoch=1, val_loss=1.0)
    assert improved is True
    assert should_stop is False
    assert state.best_epoch == 1

    improved, should_stop = state.update(epoch=2, val_loss=0.998)
    assert improved is False
    assert should_stop is False

    improved, should_stop = state.update(epoch=3, val_loss=0.999)
    assert improved is False
    assert should_stop is True


def test_run_pretrain_training_tracks_metrics_without_nameerror(tmp_path, monkeypatch):
    model = _DummyModel()

    class _DummyDataset:
        def reset_error_stats(self):
            return None

        def consume_error_stats(self):
            return {
                "sample_error_count": 0,
                "sample_error_rate": 0.0,
                "requested_count": 1,
                "error_examples": [],
            }

        def __len__(self):
            return 1

    class _DummyLoader:
        def __init__(self):
            self.dataset = _DummyDataset()

        def __len__(self):
            return 1

        def __iter__(self):
            yield (
                torch.ones(1, 4, dtype=torch.long),
                torch.ones(1, 4, dtype=torch.long),
                torch.ones(1, 4, dtype=torch.long),
                torch.zeros(1, 3, 4, 4),
                (torch.ones(1, 5, dtype=torch.long), torch.ones(1, 5, dtype=torch.long)),
                (
                    torch.ones(1, 6, dtype=torch.long),
                    torch.ones(1, 6, dtype=torch.long),
                    torch.ones(1, 6, 6, dtype=torch.float32),
                    torch.ones(1, 6, dtype=torch.long),
                ),
                [],
            )

    train_loader = _DummyLoader()
    val_loader = _DummyLoader()

    class _FakeScheduler:
        def step(self):
            return None

        def state_dict(self):
            return {}

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    monkeypatch.setattr(
        train_pretrain,
        "train_pretrain",
        lambda *args, **kwargs: (optimizer, _FakeScheduler()),
    )
    monkeypatch.setattr(train_pretrain, "eval_pretrain", lambda *args, **kwargs: 0.5)
    monkeypatch.setattr(train_pretrain, "_save_pretrain_checkpoint", lambda *args, **kwargs: None)

    class _FakeLogger:
        def info(self, *args, **kwargs):
            return None

        def warning(self, *args, **kwargs):
            return None

    config = {
        "training": {
            "num_epochs": 1,
            "gradient_accumulation_steps": 1,
            "max_grad_norm": 1.0,
            "early_stopping_patience": 12,
            "early_stopping_min_delta": 0.005,
            "restore_best_checkpoint": False,
        }
    }

    train_pretrain.run_pretrain_training(
        model,
        tokenizer=_DummyTokenizer(),
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        output_dir=str(tmp_path),
        logger=_FakeLogger(),
    )

    metrics = (tmp_path / "metrics.json").read_text(encoding="utf-8")
    assert '"val_loss"' in metrics


def test_infer_resume_position_reads_last_training_state(tmp_path):
    last_dir = tmp_path / "last"
    last_dir.mkdir()
    (last_dir / "training_state.json").write_text(
        '{"next_epoch": 3, "next_step": 0}',
        encoding="utf-8",
    )

    assert infer_resume_position(str(last_dir)) == (3, 0)


def test_early_stopping_state_round_trip_json(tmp_path):
    state = EarlyStoppingState(
        patience=8,
        min_delta=0.005,
        best_val_loss=0.4321,
        best_epoch=7,
        num_bad_epochs=3,
    )

    train_pretrain._write_early_stopping_state(str(tmp_path), state)
    restored = train_pretrain._load_early_stopping_state(str(tmp_path))

    assert restored is not None
    assert restored.patience == 8
    assert restored.min_delta == 0.005
    assert restored.best_val_loss == 0.4321
    assert restored.best_epoch == 7
    assert restored.num_bad_epochs == 3


def test_load_early_stopping_state_returns_none_when_missing(tmp_path):
    assert train_pretrain._load_early_stopping_state(str(tmp_path)) is None


def test_write_training_state_persists_epoch_and_step(tmp_path):
    _write_training_state(str(tmp_path), next_epoch=5, next_step=12, global_optimizer_step=34)
    payload = json.loads((tmp_path / "training_state.json").read_text(encoding="utf-8"))

    assert payload == {"next_epoch": 5, "next_step": 12, "global_optimizer_step": 34}
