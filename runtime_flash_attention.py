from typing import Dict, Iterable


FLASH_ATTENTION_2_IMPLEMENTATION = "flash_attention_2"


def flash_attention_requested(config: Dict, default: bool = True) -> bool:
    model_cfg = config.get("model", {})
    flash_cfg = model_cfg.get("flash_attention", {})
    if isinstance(flash_cfg, dict) and "enabled" in flash_cfg:
        return bool(flash_cfg["enabled"])
    return bool(default)


def enable_flash_attention_for_config(model_config, requested: bool = True) -> bool:
    if not requested:
        return False

    changed = False
    for config_obj in _iter_config_objects(model_config):
        if hasattr(config_obj, "use_flash_attn"):
            setattr(config_obj, "use_flash_attn", True)
            changed = True
        if hasattr(config_obj, "attn_implementation"):
            setattr(config_obj, "attn_implementation", FLASH_ATTENTION_2_IMPLEMENTATION)
            changed = True
        if hasattr(config_obj, "_attn_implementation"):
            setattr(config_obj, "_attn_implementation", FLASH_ATTENTION_2_IMPLEMENTATION)
            changed = True

    return changed


def flash_attention_from_pretrained_kwargs(config: Dict) -> Dict[str, str]:
    if not flash_attention_requested(config) or not flash_attention_available():
        return {}
    return {"attn_implementation": FLASH_ATTENTION_2_IMPLEMENTATION}


def flash_attention_available() -> bool:
    try:
        import flash_attn  # noqa: F401
    except Exception:
        return False
    return True


def _iter_config_objects(config_obj) -> Iterable[object]:
    seen = set()
    stack = [config_obj]
    child_attrs = ("vision_config", "llm_config", "text_config", "language_config")
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        for attr in child_attrs:
            stack.append(getattr(current, attr, None))


def _iter_attention_candidates(model) -> Iterable[object]:
    seen = set()
    roots = [
        model,
        getattr(model, "config", None),
        getattr(model, "vision_model", None),
        getattr(model, "language_model", None),
        getattr(model, "llm", None),
        getattr(model, "model", None),
    ]
    for root in roots:
        if root is None or id(root) in seen:
            continue
        seen.add(id(root))
        yield root
        modules = getattr(root, "modules", None)
        if callable(modules):
            for module in modules():
                if id(module) in seen:
                    continue
                seen.add(id(module))
                yield module


def _flash_attention_candidate_active(candidate) -> tuple[bool, bool]:
    if hasattr(candidate, "use_flash_attn"):
        return True, bool(getattr(candidate, "use_flash_attn"))

    for attr in ("attn_implementation", "_attn_implementation"):
        if hasattr(candidate, attr):
            implementation = str(getattr(candidate, attr) or "").lower()
            return True, implementation == FLASH_ATTENTION_2_IMPLEMENTATION or "flash" in implementation

    class_name = type(candidate).__name__.lower()
    if "flash" in class_name and "attn" in class_name:
        return True, True

    return False, False


def collect_flash_attention_status(model, requested: bool) -> Dict[str, bool]:
    active_layers = []
    for candidate in _iter_attention_candidates(model):
        seen, active = _flash_attention_candidate_active(candidate)
        if seen:
            active_layers.append(active)

    available = flash_attention_available()
    active = any(active_layers)
    inactive_reason = None
    if not requested:
        inactive_reason = "not_requested"
    elif not available:
        inactive_reason = "flash_attn_not_installed"
    elif not active_layers:
        inactive_reason = "no_flash_attention_markers_detected"
    elif not active:
        inactive_reason = "flash_attention_markers_detected_but_inactive"

    return {
        "flash_attention_requested": bool(requested),
        "flash_attention_available": available,
        "flash_attention_active": active,
        "flash_attention_layer_count": len(active_layers),
        "flash_attention_active_layer_count": sum(1 for active in active_layers if active),
        "flash_attention_inactive_reason": inactive_reason,
    }
