from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Callable


@dataclass
class BackendSpec:
    name: str
    load_model_and_tokenizer: Callable
    build_train_collate_fn: Callable
    build_eval_collate_fn: Callable
    forward_train_batch: Callable
    forward_eval_batch: Callable
    generate_response: Callable
    attach_qformer_if_enabled: Callable
    prepare_model_for_training: Callable
    save_backend_artifacts: Callable
    load_backend_artifacts: Callable


def _load_backend_module(architecture: str):
    normalized = str(architecture or "").strip().lower()
    if normalized == "sailvl":
        return import_module("model_backends.sailvl.runtime")
    raise ValueError(f"Unsupported backend architecture: {architecture}")


def get_backend(architecture: str) -> BackendSpec:
    module = _load_backend_module(architecture)
    return BackendSpec(
        name=module.BACKEND_NAME,
        load_model_and_tokenizer=module.load_model_and_tokenizer,
        build_train_collate_fn=module.build_train_collate_fn,
        build_eval_collate_fn=module.build_eval_collate_fn,
        forward_train_batch=module.forward_train_batch,
        forward_eval_batch=module.forward_eval_batch,
        generate_response=module.generate_response,
        attach_qformer_if_enabled=module.attach_qformer_if_enabled,
        prepare_model_for_training=getattr(module, "prepare_model_for_training", lambda model, logger=None: None),
        save_backend_artifacts=module.save_backend_artifacts,
        load_backend_artifacts=module.load_backend_artifacts,
    )
