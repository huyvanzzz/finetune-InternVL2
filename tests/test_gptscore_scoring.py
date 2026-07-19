from gptscore.scoring import score_judge_output, score_judged_document


def _judge_output(label="Acceptable", unsafe_action=False, polarity_reversal=False):
    return {
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

    assert scored["overall_score"] == 2.0


def test_score_judge_output_uses_mean_over_applicable_criteria_only():
    payload = _judge_output(label="Strong")
    payload["criteria"]["direction_fidelity"]["applicable"] = False
    payload["criteria"]["direction_fidelity"]["label"] = None

    scored = score_judge_output(payload)

    assert scored["overall_score"] == 3.0


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
