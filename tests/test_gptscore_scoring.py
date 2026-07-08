from gptscore.scoring import score_judge_output, score_judged_document


def _judge_output(label="Acceptable", unsafe_action=False, polarity_reversal=False):
    return {
        "gate": {
            "polarity_reversal": polarity_reversal,
            "unsafe_action": unsafe_action,
        },
        "signals_in_gt": {
            "has_direction_anchor": True,
            "has_action_demand": True,
            "has_hazard_or_path_state": True,
        },
        "criteria": {
            "safety_correctness": {"applicable": True, "label": label, "rationale": "ok"},
            "hazard_path_state_fidelity": {"applicable": True, "label": label, "rationale": "ok"},
            "direction_fidelity": {"applicable": True, "label": label, "rationale": "ok"},
            "action_usefulness": {"applicable": True, "label": label, "rationale": "ok"},
            "spoken_guidance_quality": {"applicable": True, "label": label, "rationale": "ok"},
        },
        "overall_rationale": "ok",
    }


def test_score_judge_output_maps_labels_and_computes_mean():
    scored = score_judge_output(_judge_output(label="Acceptable"))

    assert scored["mean_before_gate"] == 2.0
    assert scored["applied_gate_cap"] == "none"
    assert scored["overall_score"] == 2.0


def test_score_judge_output_caps_score_for_unsafe_action():
    scored = score_judge_output(_judge_output(label="Strong", unsafe_action=True))

    assert scored["mean_before_gate"] == 3.0
    assert scored["overall_score"] == 0.0
    assert scored["applied_gate_cap"] == "score=0.0_by_unsafe_action"


def test_score_judged_document_skips_non_scored_items_and_reports_mean():
    judged = {
        "items": [
            {
                "id": 0,
                "judge_status": "scored",
                "ground_truth": "gt0",
                "generation": "gen0",
                "judge_output": _judge_output(label="Acceptable"),
            },
            {
                "id": 1,
                "judge_status": "parse_failed",
                "ground_truth": "gt1",
                "generation": "gen1",
                "judge_output": None,
            },
        ]
    }

    result = score_judged_document(judged)

    assert result["summary"]["total_items"] == 2
    assert result["summary"]["scored_items"] == 1
    assert result["summary"]["skipped_items"] == 1
    assert result["summary"]["mean_over_valid_samples"] == 2.0
