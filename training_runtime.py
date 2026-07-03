from __future__ import annotations

from typing import Optional


def enable_gradient_checkpointing(model, enabled: bool, logger=None) -> Optional[str]:
    if not enabled:
        return None

    candidates = [
        ("model", model),
        ("language_model", getattr(model, "language_model", None)),
    ]

    for name, module in candidates:
        if module is None:
            continue

        enable_fn = getattr(module, "gradient_checkpointing_enable", None)
        if not callable(enable_fn):
            continue

        try:
            enable_fn()
            if logger is not None:
                logger.info(f"Enabled gradient checkpointing on {name}.")
            return name
        except ValueError as exc:
            if "does not support gradient checkpointing" not in str(exc):
                raise

    if logger is not None:
        logger.warning(
            "Gradient checkpointing was requested but no compatible module was found. Continuing without it."
        )
    return None
