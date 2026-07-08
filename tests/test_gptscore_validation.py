from gptscore.validation import validate_input_pair, validate_judge_output


def _valid_judge_output():
    return {
        "gate": {
            "polarity_reversal": False,
            "unsafe_action": False,
        },
        "signals_in_gt": {
            "has_direction_anchor": True,
            "has_action_demand": True,
            "has_hazard_or_path_state": True,
        },
        "criteria": {
            "safety_correctness": {
                "applicable": True,
                "label": "Acceptable",
                "rationale": "Main safety meaning is preserved.",
            },
            "hazard_path_state_fidelity": {
                "applicable": True,
                "label": "Acceptable",
                "rationale": "Main hazard is preserved.",
            },
            "direction_fidelity": {
                "applicable": True,
                "label": "Strong",
                "rationale": "Direction matches clearly.",
            },
            "action_usefulness": {
                "applicable": True,
                "label": "Acceptable",
                "rationale": "Action is reasonable.",
            },
            "spoken_guidance_quality": {
                "applicable": True,
                "label": "Strong",
                "rationale": "Concise and usable.",
            },
        },
        "overall_rationale": "The generation is mostly correct and safe.",
    }


def test_validate_input_pair_flags_missing_generation():
    errors = validate_input_pair({"id": 1, "ground_truth": "gt only"})

    assert any("generation" in error for error in errors)


def test_validate_judge_output_accepts_valid_contract():
    errors = validate_judge_output(_valid_judge_output())

    assert errors == []


def test_validate_judge_output_rejects_invalid_label():
    payload = _valid_judge_output()
    payload["criteria"]["direction_fidelity"]["label"] = "Good"

    errors = validate_judge_output(payload)

    assert any("direction_fidelity" in error and "label" in error for error in errors)


def test_validate_judge_output_rejects_non_applicable_with_non_null_label():
    payload = _valid_judge_output()
    payload["criteria"]["direction_fidelity"]["applicable"] = False
    payload["criteria"]["direction_fidelity"]["label"] = "Weak"

    errors = validate_judge_output(payload)

    assert any("direction_fidelity" in error and "null" in error for error in errors)
