from pathlib import Path


def test_gptscore_env_example_exists_with_required_keys():
    content = Path("gptscore/.env.example").read_text(encoding="utf-8")

    assert "OPENAI_API_KEY=" in content
    assert "OPENROUTER_API_KEY=" in content
