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
    assert "this criterion is still applicable even when the generation omits direction words" in prompt
    assert "one-step nearby clock drift should usually be Weak rather than Fail" in prompt
    assert "Treat ahead, in front, straight, straight ahead, turn left, and turn right as real direction anchors" in prompt
    assert "Treat route phrases such as current road, main road, return to the route, return to the main route, crossroad, intersection, fork, and upcoming turning as direction-related guidance" in prompt
    assert "Route-structure guidance such as go straight, turn left, turn right, return to the main route" in prompt
    assert "do not mark this criterion as not applicable" in prompt
    assert "this criterion is still applicable and must not be N/A" in prompt
    assert "Compare generic forward cues against specific clock directions by semantic compatibility" in prompt
    assert "Do not return overall_score." in prompt
    assert "unsafe_action" not in prompt
    assert "Do not mark this criterion applicable merely because the ground truth mentions an object, obstacle, or scene structure." in prompt
    assert "General warnings such as pay attention to safety do not by themselves create an action demand." in prompt
    assert "A warning or path description counts only when it clearly implies or permits a next-step guidance expectation." in prompt
    assert "Hazard-only or location-only descriptions are usually not enough by themselves." in prompt
    assert "If the generation preserves a broadly similar caution scenario but gets the object identity, subtype, or secondary scene detail wrong, prefer Weak rather than Fail." in prompt
    assert "If both ground truth and generation still describe a nearby obstacle or caution scenario of the same broad type, prefer Weak unless the wrong object or path-state would clearly change what the user should do." in prompt
    assert "If the ground truth hazard is generic or broadly obstacle-like and the generation also describes a nearby obstacle that still calls for avoidance or caution, prefer Weak rather than Fail" in prompt
    assert "Use Fail more readily for route-critical mismatches such as clear versus blocked path, stair versus flat path" in prompt
    assert "Treat hazard-localizing phrases such as obstacle ahead, pedestrian ahead, vehicle ahead" in prompt
    assert "Treat phrases such as pedestrian ahead, vehicle ahead, obstacle ahead, in front, or straight ahead as valid forward direction anchors" in prompt
    assert "Route-structure information such as crossroad, fork, intersection, road narrowing, route deviation, return-to-route, or crossing availability is part of the main scene meaning" in prompt
    assert "If the ground truth includes both an obstacle and route-structure guidance, and the generation keeps only the obstacle while dropping the route-structure constraint" in prompt
    assert "If the generation keeps only a generic caution cue but loses a route-critical continue, turn, return, or crossing decision from the ground truth" in prompt
    assert "If the ground truth includes a route-continuation or route-choice instruction such as keep going straight, continue on the current road, return to the route, or choose a branch at a fork or intersection" in prompt
    assert "Strong hazard warnings that clearly imply a needed response" in prompt
    assert "If the ground truth warns about an immediate obstacle or moving risk" in prompt
    assert "If the ground truth says pay attention or pay attention to safety in a context with immediate obstacle, moving traffic, route deviation, or crossing decision, treat that as a light action demand" in prompt
    assert "Mark has_direction_anchor=true when the ground truth includes o'clock direction, ahead, in front, front, straight, straight ahead, turn left, turn right, current road, main road, return to the route, return to the main route, fork, crossroad, intersection, upcoming turning" in prompt
    assert "Mark has_action_demand=true when the ground truth includes an explicit action, route-continuation instruction, a strong warning with an expected response, or a clear-path instruction" in prompt
    assert "Do not output has_direction_anchor=false for cues like ahead, in front, straight, turn left, turn right, current road, main road, crossroad, intersection, fork, or turning" in prompt
    assert "Do not output has_action_demand=false for cues like keep still, give way, keep walking slowly, keep moving forward, return to the route, or pay attention to safety in an immediate-risk context." in prompt
    assert "If the ground truth uses a generic forward cue and the generation uses a specific clock direction, keep has_direction_anchor=true and score Direction Fidelity instead of using N/A." in prompt
    assert "The placeholder booleans shown below are only structural examples" in prompt
    assert "Judge this criterion independently from semantic correctness." in prompt
    assert '\"has_direction_anchor\": false' in prompt
