from types import SimpleNamespace

import torch

from train_pretrain import PretrainCollaterFn, forward_pretrain_batch


class _DummyModel:
    def __init__(self):
        self.template = "internlm2-chat"
        self.system_message = "You are a navigation assistant."
        self.num_image_token = 32
        self.qformer_enabled = True
        self.trajectory_enabled = True
        self.qformer_calls = []
        self.trajectory_calls = []

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
        loss = (input_ids.float().mean() * 0) + pixel_values.float().mean() + 1.0
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
