import random

import numpy as np
import torch


def _pack_numpy_random_state(state):
    algo, keys, pos, has_gauss, cached_gaussian = state
    return {
        "algorithm": algo,
        "keys": keys.tolist(),
        "pos": int(pos),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def _unpack_numpy_random_state(state_dict):
    return (
        state_dict["algorithm"],
        np.array(state_dict["keys"], dtype=np.uint32),
        state_dict["pos"],
        state_dict["has_gauss"],
        state_dict["cached_gaussian"],
    )


def capture_python_rng_state() -> dict:
    return {
        "python_random_state": random.getstate(),
        "numpy_random_state": _pack_numpy_random_state(np.random.get_state()),
    }


def restore_python_rng_state(state: dict) -> None:
    random.setstate(state["python_random_state"])
    np.random.set_state(_unpack_numpy_random_state(state["numpy_random_state"]))


def capture_torch_rng_state(cuda: bool = True) -> dict:
    state = {
        "torch_cpu_rng_state": torch.get_rng_state(),
        "cuda_included": bool(cuda and torch.cuda.is_available()),
    }
    if state["cuda_included"]:
        state["torch_cuda_rng_state"] = torch.cuda.get_rng_state_all()
    return state


def restore_torch_rng_state(state: dict) -> None:
    torch.set_rng_state(state["torch_cpu_rng_state"])
    cuda_state = state.get("torch_cuda_rng_state")
    if state.get("cuda_included") and cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def capture_full_runtime_state(include_cuda: bool) -> dict:
    return {
        "python": capture_python_rng_state(),
        "torch": capture_torch_rng_state(cuda=include_cuda),
    }


def restore_full_runtime_state(state: dict) -> None:
    restore_python_rng_state(state["python"])
    restore_torch_rng_state(state["torch"])
