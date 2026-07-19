from pathlib import Path

from gptscore.providers import load_prompt_assets


def test_load_prompt_assets_uses_action_looser_profile_by_default():
    package_dir = Path("gptscore")

    system_prompt, user_template, json_schema, prompt_version = load_prompt_assets(
        package_dir
    )

    assert "Do not return overall_score." in system_prompt
    assert "unsafe_action" not in system_prompt
    assert "{{GROUND_TRUTH}}" in user_template
    assert json_schema["name"] == "gptscore_alter_judge"
    assert "gate" not in json_schema["schema"]["properties"]
    assert prompt_version == "gptscore-alter-v4f-variant-action-looser-no-gates-direction-anchor-and-route-structure-tuned"


def test_load_prompt_assets_can_select_baseline_profile_with_two_gates():
    package_dir = Path("gptscore")

    system_prompt, _user_template, json_schema, prompt_version = load_prompt_assets(
        package_dir,
        prompt_profile="baseline_142",
    )

    assert "General warnings such as pay attention to safety do not by themselves create an action demand." in system_prompt
    assert "Do not return overall_score." in system_prompt
    assert "unsafe_action" not in system_prompt
    assert "gate" not in json_schema["schema"]["properties"]
    assert prompt_version == "gptscore-alter-v4f-baseline-142-no-gates-direction-anchor-and-route-structure-tuned"


def test_load_prompt_assets_can_select_action_looser_profile_explicitly():
    package_dir = Path("gptscore")

    system_prompt, _user_template, _json_schema, prompt_version = load_prompt_assets(
        package_dir,
        prompt_profile="variant_action_looser",
    )

    assert "When the generation still gives a broadly safe warning or avoidance cue, do not use Fail unless it clearly contradicts or defeats the required action." in system_prompt
    assert "Milder issues such as partial action loss, weaker caution, or a less specific but still broadly safe next-step cue should usually be Weak rather than Fail." in system_prompt
    assert "unsafe_action" not in system_prompt
    assert prompt_version == "gptscore-alter-v4f-variant-action-looser-no-gates-direction-anchor-and-route-structure-tuned"
