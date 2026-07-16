from types import SimpleNamespace

import torch

import train_pretrain
from train_pretrain import (
    EarlyStoppingState,
    PretrainCollaterFn,
    _build_optimizer,
    forward_pretrain_batch,
    infer_resume_position,
    resolve_warmup_steps,
)


class _DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.template = "internlm2-chat"
        self.system_message = "You are a navigation assistant."
        self.num_image_token = 32
        self.qformer_enabled = True
        self.trajectory_enabled = True
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
