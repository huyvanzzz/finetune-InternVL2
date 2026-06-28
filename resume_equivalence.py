import copy
import hashlib
import math
import random
from dataclasses import dataclass

import numpy as np
import torch

from resume_state import capture_full_runtime_state, restore_full_runtime_state


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class TinyDropoutModel(torch.nn.Module):
    def __init__(self, dropout_p: float = 0.5):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.dropout = torch.nn.Dropout(dropout_p)
        self.head = torch.nn.Linear(4, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.linear(features)
        hidden = self.dropout(hidden)
        return self.head(hidden)


@dataclass
class ToyCheckpoint:
    completed_steps: int
    epoch_seed: int
    dropout_p: float
    model_state: dict
    optimizer_state: dict
    scheduler_state: dict
    runtime_state: dict


def make_dataset(num_samples: int):
    dataset = []
    for sample_id in range(num_samples):
        features = torch.tensor(
            [
                float(sample_id),
                float(sample_id + 1),
                float(sample_id % 3),
                1.0,
            ],
            dtype=torch.float32,
        )
        target = torch.tensor([float((sample_id % 5) / 10.0)], dtype=torch.float32)
        dataset.append((sample_id, features, target))
    return dataset


def build_sample_order(num_samples: int, epoch_seed: int):
    set_all_seeds(epoch_seed)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=torch.ones(num_samples, dtype=torch.double),
        num_samples=num_samples,
        replacement=False,
    )
    return list(sampler)


def _build_components(dropout_p: float):
    set_all_seeds(0)
    model = TinyDropoutModel(dropout_p=dropout_p)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    criterion = torch.nn.MSELoss()
    return model, optimizer, scheduler, criterion


def _weight_checksum(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for tensor in model.state_dict().values():
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _scheduler_snapshot(scheduler):
    return {
        "last_epoch": scheduler.last_epoch,
        "_step_count": scheduler._step_count,
        "_last_lr": [round(x, 12) for x in scheduler.get_last_lr()],
    }


def _clone_state_dict(state_dict: dict) -> dict:
    cloned = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            cloned[key] = value.clone()
        elif isinstance(value, dict):
            cloned[key] = _clone_state_dict(value)
        elif isinstance(value, list):
            cloned[key] = [item.clone() if torch.is_tensor(item) else copy.deepcopy(item) for item in value]
        else:
            cloned[key] = copy.deepcopy(value)
    return cloned


def run_reference_epoch(dataset, epoch_seed: int, dropout_p: float, checkpoint_after_steps: int | None = None):
    model, optimizer, scheduler, criterion = _build_components(dropout_p)
    sample_order = build_sample_order(len(dataset), epoch_seed)
    records = []
    checkpoint = None

    for step_index, sample_index in enumerate(sample_order):
        sample_id, features, target = dataset[sample_index]
        optimizer.zero_grad()
        output = model(features)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()

        records.append(
            {
                "step": step_index,
                "sample_id": sample_id,
                "loss": round(float(loss.item()), 8),
                "weight_checksum": _weight_checksum(model),
                "optimizer_step": step_index + 1,
                "scheduler": _scheduler_snapshot(scheduler),
            }
        )

        if checkpoint_after_steps is not None and (step_index + 1) == checkpoint_after_steps:
            checkpoint = ToyCheckpoint(
                completed_steps=checkpoint_after_steps,
                epoch_seed=epoch_seed,
                dropout_p=dropout_p,
                model_state=_clone_state_dict(model.state_dict()),
                optimizer_state=_clone_state_dict(optimizer.state_dict()),
                scheduler_state=copy.deepcopy(scheduler.state_dict()),
                runtime_state=capture_full_runtime_state(include_cuda=False),
            )

    return {
        "records": records,
        "sample_order": [record["sample_id"] for record in records],
        "checkpoint": checkpoint,
    }


def run_resumed_epoch(dataset, checkpoint: ToyCheckpoint, restore_rng_state: bool):
    model, optimizer, scheduler, criterion = _build_components(checkpoint.dropout_p)
    model.load_state_dict(checkpoint.model_state)
    optimizer.load_state_dict(checkpoint.optimizer_state)
    scheduler.load_state_dict(checkpoint.scheduler_state)

    sample_order = build_sample_order(len(dataset), checkpoint.epoch_seed)
    if restore_rng_state:
        restore_full_runtime_state(checkpoint.runtime_state)

    records = []
    for step_index in range(checkpoint.completed_steps, len(sample_order)):
        sample_id, features, target = dataset[sample_order[step_index]]
        optimizer.zero_grad()
        output = model(features)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()

        records.append(
            {
                "step": step_index,
                "sample_id": sample_id,
                "loss": round(float(loss.item()), 8),
                "weight_checksum": _weight_checksum(model),
                "optimizer_step": step_index + 1,
                "scheduler": _scheduler_snapshot(scheduler),
            }
        )

    return {
        "records": records,
        "sample_order": [record["sample_id"] for record in records],
    }


def run_epoch_from_checkpoint(dataset, checkpoint: ToyCheckpoint, epoch_seed: int):
    model, optimizer, scheduler, criterion = _build_components(checkpoint.dropout_p)
    model.load_state_dict(checkpoint.model_state)
    optimizer.load_state_dict(checkpoint.optimizer_state)
    scheduler.load_state_dict(checkpoint.scheduler_state)

    sample_order = build_sample_order(len(dataset), epoch_seed)
    records = []
    for step_index, sample_index in enumerate(sample_order):
        sample_id, features, target = dataset[sample_index]
        optimizer.zero_grad()
        output = model(features)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()

        records.append(
            {
                "step": step_index,
                "sample_id": sample_id,
                "loss": round(float(loss.item()), 8),
                "weight_checksum": _weight_checksum(model),
                "optimizer_step": step_index + checkpoint.completed_steps + 1,
                "scheduler": _scheduler_snapshot(scheduler),
            }
        )

    return {
        "records": records,
        "sample_order": [record["sample_id"] for record in records],
    }


def build_verdict(reference_records, candidate_records):
    def _values_equal(left, right):
        if isinstance(left, float) and isinstance(right, float):
            if math.isnan(left) and math.isnan(right):
                return True
        return left == right

    def _sequence_equal(left_items, right_items, key):
        if len(left_items) != len(right_items):
            return False
        return all(_values_equal(left[key], right[key]) for left, right in zip(left_items, right_items))

    sample_order_equal = [r["sample_id"] for r in reference_records] == [r["sample_id"] for r in candidate_records]
    losses_equal = _sequence_equal(reference_records, candidate_records, "loss")
    weights_equal = _sequence_equal(reference_records, candidate_records, "weight_checksum")
    scheduler_equal = _sequence_equal(reference_records, candidate_records, "scheduler")

    if sample_order_equal and losses_equal and weights_equal and scheduler_equal:
        verdict = "fully_equivalent"
    elif sample_order_equal and not (losses_equal and weights_equal and scheduler_equal):
        verdict = "diverges_due_to_rng"
    else:
        verdict = "diverges_due_to_other_state"

    return {
        "sample_order_equal": sample_order_equal,
        "losses_equal": losses_equal,
        "weight_checksums_equal": weights_equal,
        "scheduler_state_equal": scheduler_equal,
        "verdict": verdict,
    }
