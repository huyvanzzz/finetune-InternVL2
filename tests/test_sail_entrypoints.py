from __future__ import annotations

import importlib


def test_backend_dispatch_module_exists():
    dispatch = importlib.import_module("backend_dispatch")
    assert hasattr(dispatch, "get_backend_for_config")


def test_backend_selector_returns_none_for_internvl():
    dispatch = importlib.import_module("backend_dispatch")
    backend = dispatch.get_backend_for_config({"model": {"architecture": "internvl"}})
    assert backend is None


def test_backend_selector_loads_sail_backend():
    dispatch = importlib.import_module("backend_dispatch")
    backend = dispatch.get_backend_for_config({"model": {"architecture": "sailvl"}})
    assert backend is not None
    assert backend.name == "sailvl"
