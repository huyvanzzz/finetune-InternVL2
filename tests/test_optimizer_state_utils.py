import torch
from torch import nn

from optimizer_state_utils import (
    export_sanitized_optimizer_state_dict,
    move_optimizer_state_to_param_device,
    sanitize_optimizer_state_dict,
)


def _make_optimizer_with_state():
    param = nn.Parameter(torch.ones(2, 2, dtype=torch.bfloat16))
    optimizer = torch.optim.AdamW([param], lr=1e-3, foreach=False, fused=False)
    (param.sum()).backward()
    optimizer.step()
    optimizer.zero_grad()
    return optimizer


def test_raw_state_dict_mutation_would_change_live_optimizer_state():
    optimizer = _make_optimizer_with_state()
    live_state = next(iter(optimizer.state.values()))
    raw_state_dict = optimizer.state_dict()
    raw_state = next(iter(raw_state_dict["state"].values()))

    assert live_state["exp_avg"].dtype == torch.bfloat16
    assert raw_state["exp_avg"].data_ptr() == live_state["exp_avg"].data_ptr()

    raw_state["exp_avg"] = raw_state["exp_avg"].float()

    assert live_state["exp_avg"].dtype == torch.float32


def test_export_sanitized_optimizer_state_dict_does_not_mutate_live_optimizer_state():
    optimizer = _make_optimizer_with_state()
    live_state = next(iter(optimizer.state.values()))

    exported_state_dict, converted, overridden_groups = export_sanitized_optimizer_state_dict(optimizer)
    exported_state = next(iter(exported_state_dict["state"].values()))

    assert converted >= 2
    assert overridden_groups >= 0
    assert exported_state_dict["param_groups"][0]["foreach"] is False
    assert exported_state_dict["param_groups"][0].get("fused", False) is False
    assert exported_state["exp_avg"].dtype == torch.float32
    assert exported_state["exp_avg_sq"].dtype == torch.float32
    assert live_state["exp_avg"].dtype == torch.bfloat16
    assert live_state["exp_avg_sq"].dtype == torch.bfloat16


def test_sanitized_state_dict_still_loads_into_optimizer():
    source_optimizer = _make_optimizer_with_state()
    exported_state_dict, _, _ = export_sanitized_optimizer_state_dict(source_optimizer)

    target_param = nn.Parameter(torch.ones(2, 2, dtype=torch.bfloat16))
    target_optimizer = torch.optim.AdamW([target_param], lr=1e-3, foreach=False, fused=False)
    (target_param.sum()).backward()
    target_optimizer.step()
    target_optimizer.zero_grad()

    sanitized_state_dict, _, _ = sanitize_optimizer_state_dict(exported_state_dict)
    target_optimizer.load_state_dict(sanitized_state_dict)
    moved = move_optimizer_state_to_param_device(target_optimizer)
    loaded_state = next(iter(target_optimizer.state.values()))

    assert moved == 0
    assert loaded_state["exp_avg"].dtype == target_param.dtype
    assert loaded_state["exp_avg_sq"].dtype == target_param.dtype
