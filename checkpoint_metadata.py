from __future__ import annotations

import json
from pathlib import Path


def _log(logger, message, *args):
    if logger is not None:
        logger.info(message, *args)


def rewrite_readme_base_model(readme_text: str, base_model_name_or_path: str) -> str:
    if not readme_text.startswith("---\n"):
        return readme_text

    parts = readme_text.split("---\n", 2)
    if len(parts) < 3:
        return readme_text

    frontmatter = parts[1]
    body = parts[2]
    updated_lines = []
    saw_base_model = False

    for line in frontmatter.splitlines():
        stripped = line.strip()
        if stripped.startswith("base_model:"):
            updated_lines.append(f"base_model: {base_model_name_or_path}")
            saw_base_model = True
            continue
        if stripped.startswith("- base_model:adapter:"):
            indent = line[: len(line) - len(line.lstrip())]
            updated_lines.append(f"{indent}- base_model:adapter:{base_model_name_or_path}")
            continue
        updated_lines.append(line)

    if not saw_base_model:
        updated_lines.insert(0, f"base_model: {base_model_name_or_path}")

    return "---\n" + "\n".join(updated_lines) + "\n---\n" + body


def sanitize_peft_checkpoint_metadata(output_dir: str, base_model_name_or_path: str, logger=None) -> None:
    if not base_model_name_or_path:
        return

    output_path = Path(output_dir)
    adapter_config_path = output_path / "adapter_config.json"
    if adapter_config_path.exists():
        adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        current = adapter_config.get("base_model_name_or_path")
        if current != base_model_name_or_path:
            adapter_config["base_model_name_or_path"] = base_model_name_or_path
            adapter_config_path.write_text(
                json.dumps(adapter_config, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            _log(logger, "Normalized adapter_config.json base_model_name_or_path: %s -> %s", current, base_model_name_or_path)

    readme_path = output_path / "README.md"
    if readme_path.exists():
        original = readme_path.read_text(encoding="utf-8")
        rewritten = rewrite_readme_base_model(original, base_model_name_or_path)
        if rewritten != original:
            readme_path.write_text(rewritten, encoding="utf-8")
            _log(logger, "Normalized README.md base_model metadata to repo id: %s", base_model_name_or_path)
