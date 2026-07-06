from __future__ import annotations

from model_backends import get_backend


def get_backend_for_config(config):
    architecture = str(config.get("model", {}).get("architecture", "internvl")).strip().lower()
    if architecture in {"", "internvl"}:
        return None
    return get_backend(architecture)
