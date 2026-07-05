import json
from pathlib import Path


def test_sanitize_peft_checkpoint_metadata_rewrites_adapter_config_and_readme(tmp_path):
    from checkpoint_metadata import sanitize_peft_checkpoint_metadata

    (tmp_path / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": "/tmp/huggingface_cache/Qwen2.5-1.5B-Instruct",
                "r": 16,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text(
        "---\n"
        "base_model: /tmp/huggingface_cache/Qwen2.5-1.5B-Instruct\n"
        "library_name: peft\n"
        "pipeline_tag: text-generation\n"
        "tags:\n"
        "  - base_model:adapter:/tmp/huggingface_cache/Qwen2.5-1.5B-Instruct\n"
        "  - lora\n"
        "  - transformers\n"
        "---\n"
        "\n"
        "# Adapter\n",
        encoding="utf-8",
    )

    sanitize_peft_checkpoint_metadata(str(tmp_path), "Qwen/Qwen2.5-1.5B-Instruct")

    adapter_cfg = json.loads((tmp_path / "adapter_config.json").read_text(encoding="utf-8"))
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")

    assert adapter_cfg["base_model_name_or_path"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert "base_model: Qwen/Qwen2.5-1.5B-Instruct" in readme
    assert "base_model:adapter:Qwen/Qwen2.5-1.5B-Instruct" in readme
    assert "/tmp/huggingface_cache/Qwen2.5-1.5B-Instruct" not in readme


def test_sail_configs_pin_clean_base_model_repo_id():
    for config_name in (
        "sailvl_config.yaml",
        "sailvl_config_no_qformer.yaml",
        "sailvl_config_mixed.yaml",
        "sailvl_config_no_qformer_mixed.yaml",
    ):
        config_text = Path(config_name).read_text(encoding="utf-8")
        assert 'base_model_name_or_path: "Qwen/Qwen2.5-1.5B-Instruct"' in config_text
