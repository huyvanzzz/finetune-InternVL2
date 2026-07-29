from typing import Dict


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
    if hasattr(model_config, "use_flash_attn"):
        setattr(model_config, "use_flash_attn", True)
        changed = True

    vision_config = getattr(model_config, "vision_config", None)
    if vision_config is not None and hasattr(vision_config, "use_flash_attn"):
        setattr(vision_config, "use_flash_attn", True)
        changed = True

    return changed


def flash_attention_available() -> bool:
    try:
        import flash_attn  # noqa: F401
    except Exception:
        return False
    return True


def collect_flash_attention_status(model, requested: bool) -> Dict[str, bool]:
    vision_model = getattr(model, "vision_model", None)
    active_layers = []
    if vision_model is not None:
        for module in vision_model.modules():
            if hasattr(module, "use_flash_attn"):
                active_layers.append(bool(getattr(module, "use_flash_attn")))

    return {
        "flash_attention_requested": bool(requested),
        "flash_attention_available": flash_attention_available(),
        "flash_attention_active": any(active_layers),
    }
