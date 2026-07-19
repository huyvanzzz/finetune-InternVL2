import json
from pathlib import Path

import requests

from gptscore.constants import (
    DEFAULT_PROMPT_PROFILE,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_OPENROUTER_MODEL,
)
from gptscore.env_utils import get_env_value, load_runtime_env


def resolve_provider_config(provider, model=None):
    if provider == "openrouter":
        return {
            "provider": "openrouter",
            "model": model or DEFAULT_OPENROUTER_MODEL,
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_env": "OPENROUTER_API_KEY",
        }

    return {
        "provider": "openai",
        "model": model or DEFAULT_OPENAI_MODEL,
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
    }


PROMPT_PROFILE_TO_FILE = {
    "baseline_142": (
        "gptscore_alter_system_prompt.txt",
        "gptscore-alter-v4f-baseline-142-no-gates-direction-anchor-and-route-structure-tuned",
    ),
    "variant_action_looser": (
        "gptscore_alter_system_prompt_variant_action_looser.txt",
        "gptscore-alter-v4f-variant-action-looser-no-gates-direction-anchor-and-route-structure-tuned",
    ),
}


def load_prompt_assets(package_dir, prompt_profile=DEFAULT_PROMPT_PROFILE):
    prompts_dir = Path(package_dir) / "prompts"
    schemas_dir = Path(package_dir) / "schemas"
    prompt_filename, prompt_version = PROMPT_PROFILE_TO_FILE[prompt_profile]
    system_prompt = (prompts_dir / prompt_filename).read_text(encoding="utf-8")
    user_prompt_template = (
        prompts_dir / "gptscore_alter_user_prompt.txt"
    ).read_text(encoding="utf-8")
    json_schema = json.loads(
        (schemas_dir / "gptscore_alter_judge_schema.json").read_text(encoding="utf-8")
    )
    return system_prompt, user_prompt_template, json_schema, prompt_version


def render_user_prompt(template, ground_truth, generation):
    return (
        template.replace("{{GROUND_TRUTH}}", ground_truth).replace(
            "{{GENERATION}}", generation
        )
    )


def build_chat_completion_payload(model, system_prompt, user_prompt, json_schema):
    return {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": json_schema,
        },
    }


def make_provider_judge_callable(
    provider,
    model,
    repo_root,
    max_retries=2,
    timeout=90,
    prompt_profile=DEFAULT_PROMPT_PROFILE,
):
    loaded_env = load_runtime_env(repo_root)
    provider_config = resolve_provider_config(provider, model)
    api_key = get_env_value(provider_config["api_key_env"], loaded_env)
    if not api_key:
        raise RuntimeError(
            f"Missing API key: {provider_config['api_key_env']}. Create .env from gptscore/.env.example or export the variable."
        )

    system_prompt, user_template, json_schema, prompt_version = load_prompt_assets(
        Path(__file__).parent,
        prompt_profile=prompt_profile,
    )
    url = provider_config["base_url"].rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if provider == "openrouter":
        headers["HTTP-Referer"] = get_env_value("OPENROUTER_HTTP_REFERER", loaded_env) or "http://localhost"
        headers["X-Title"] = get_env_value("OPENROUTER_X_TITLE", loaded_env) or "restore-779-gptscore"

    def call_once(ground_truth, generation):
        user_prompt = render_user_prompt(user_template, ground_truth, generation)
        payload = build_chat_completion_payload(
            provider_config["model"], system_prompt, user_prompt, json_schema
        )
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        raw_text = response.text
        if response.status_code >= 400:
            raise RuntimeError(f"HTTP {response.status_code}: {raw_text[:500]}")

        body = response.json()
        choice = body["choices"][0]["message"]
        if "refusal" in choice and choice["refusal"]:
            raise RuntimeError(f"Model refusal: {choice['refusal']}")
        content = choice.get("content")
        if not isinstance(content, str):
            raise ValueError("Model content is not a plain JSON string")
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSON content: {exc}") from exc
        return parsed, raw_text

    def judge_callable(ground_truth, generation):
        attempts = 0
        last_error = None
        while attempts <= max_retries:
            try:
                return call_once(ground_truth, generation)
            except RuntimeError as exc:
                attempts += 1
                last_error = exc
                if attempts > max_retries:
                    raise
            except ValueError as exc:
                attempts += 1
                last_error = exc
                if attempts > max_retries:
                    raise
        raise last_error

    return judge_callable, provider_config["model"], prompt_version
