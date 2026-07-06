import yaml
from pathlib import Path


def test_sail_no_qformer_config_exists_and_uses_sail_architecture():
    path = Path("sailvl_config_no_qformer_mixed.yaml")
    assert path.exists(), "Missing SAIL no-qformer mixed config"

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["model"]["architecture"] == "sailvl"
    assert config["model"]["qformer"]["enabled"] is False
    assert config["data"]["response_format"] == "direct_text"


def test_sail_qformer_config_exists_and_enables_qformer():
    path = Path("sailvl_config_mixed.yaml")
    assert path.exists(), "Missing SAIL qformer mixed config"

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["model"]["architecture"] == "sailvl"
    assert config["model"]["qformer"]["enabled"] is True
    assert config["model"]["qformer"]["num_query_tokens"] == 32
    assert config["data"]["response_format"] == "direct_text"


def test_backend_registry_resolves_sail_backend():
    from model_backends import get_backend

    backend = get_backend("sailvl")
    assert backend.name == "sailvl"
    for attr in (
        "load_model_and_tokenizer",
        "build_train_collate_fn",
        "build_eval_collate_fn",
        "forward_train_batch",
        "forward_eval_batch",
        "generate_response",
        "attach_qformer_if_enabled",
        "save_backend_artifacts",
        "load_backend_artifacts",
    ):
        assert hasattr(backend, attr), f"Missing backend API: {attr}"
