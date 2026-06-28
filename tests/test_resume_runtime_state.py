import io
import random

import numpy as np
import torch

from resume_state import (
    capture_full_runtime_state,
    capture_python_rng_state,
    capture_torch_rng_state,
    restore_full_runtime_state,
    restore_python_rng_state,
    restore_torch_rng_state,
)


def _draw_random_triplet():
    return {
        "python": random.random(),
        "numpy": float(np.random.rand()),
        "torch": float(torch.rand(1).item()),
    }


def test_python_numpy_torch_rng_restore_replays_same_random_stream():
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    full_state = capture_full_runtime_state(include_cuda=False)
    first_draw = _draw_random_triplet()
    second_draw = _draw_random_triplet()

    restore_full_runtime_state(full_state)
    replay_first_draw = _draw_random_triplet()
    replay_second_draw = _draw_random_triplet()

    assert replay_first_draw == first_draw
    assert replay_second_draw == second_draw


def test_seed_reset_is_not_equivalent_to_rng_restore_after_consumption():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    _draw_random_triplet()
    consumed_state = capture_full_runtime_state(include_cuda=False)
    post_consumption_draw = _draw_random_triplet()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    seed_reset_draw = _draw_random_triplet()

    restore_full_runtime_state(consumed_state)
    restored_draw = _draw_random_triplet()

    assert seed_reset_draw != post_consumption_draw
    assert restored_draw == post_consumption_draw


def test_runtime_state_roundtrip_is_torch_save_compatible():
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)

    state = capture_full_runtime_state(include_cuda=False)

    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    restored_state = torch.load(buffer, weights_only=True)

    _draw_random_triplet()
    restore_full_runtime_state(restored_state)
    first = _draw_random_triplet()
    restore_full_runtime_state(restored_state)
    second = _draw_random_triplet()

    assert first == second


def test_capture_runtime_state_cpu_only_does_not_require_cuda():
    python_state = capture_python_rng_state()
    torch_state = capture_torch_rng_state(cuda=False)
    full_state = capture_full_runtime_state(include_cuda=False)

    assert "python_random_state" in python_state
    assert "numpy_random_state" in python_state
    assert "torch_cpu_rng_state" in torch_state
    assert "torch_cuda_rng_state" not in torch_state
    assert "python" in full_state
    assert "torch" in full_state
    assert "cuda_included" in full_state["torch"]

    restore_python_rng_state(python_state)
    restore_torch_rng_state(torch_state)
