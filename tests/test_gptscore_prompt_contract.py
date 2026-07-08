from pathlib import Path


def test_runtime_prompt_clarifies_non_critical_direction_drift_and_action_boundary():
    prompt_path = (
        Path(__file__).resolve().parents[1]
        / "gptscore"
        / "prompts"
        / "gptscore_alter_system_prompt.txt"
    )
    prompt = prompt_path.read_text(encoding="utf-8")

    assert "one adjacent clock step may still be non-critical" in prompt
    assert "Direction errors must be judged through the criteria" in prompt
    assert "unsafe_direction_reversal" not in prompt
    assert "Do not mark this criterion applicable merely because the ground truth mentions an object, obstacle, or scene structure." in prompt
    assert "General warnings such as pay attention to safety do not by themselves create an action demand." in prompt
    assert "A warning or path description counts only when it clearly implies or permits a next-step guidance expectation." in prompt
    assert "Hazard-only or location-only descriptions are usually not enough by themselves." in prompt
