from gptscore.judge_runner import judge_pairs_document


def test_judge_pairs_document_records_scored_and_transport_error_items():
    doc = {
        "checkpoint": "dummy",
        "split": "test_alter",
        "pairs": [
            {"id": 0, "ground_truth": "gt0", "generation": "gen0"},
            {"id": 1, "ground_truth": "gt1", "generation": "gen1"},
        ],
    }

    valid_output = {
        "signals_in_gt": {
            "has_direction_anchor": False,
            "has_action_demand": True,
            "has_hazard_or_path_state": True,
        },
        "criteria": {
            "safety_correctness": {"applicable": True, "label": "Acceptable", "rationale": "ok"},
            "hazard_path_state_fidelity": {"applicable": True, "label": "Acceptable", "rationale": "ok"},
            "direction_fidelity": {"applicable": False, "label": None, "rationale": "no direction"},
            "action_usefulness": {"applicable": True, "label": "Acceptable", "rationale": "ok"},
            "spoken_guidance_quality": {"applicable": True, "label": "Acceptable", "rationale": "ok"},
        },
        "overall_rationale": "ok",
    }

    calls = {"count": 0}

    def fake_judge(_ground_truth, _generation):
        calls["count"] += 1
        if calls["count"] == 1:
            return valid_output, '{"mock": "raw"}'
        raise RuntimeError("network down")

    judged = judge_pairs_document(
        doc,
        provider="openai",
        model="gpt-4o",
        judge_callable=fake_judge,
        prompt_version="v1",
        schema_version="v1",
        limit=None,
        offset=0,
        sample_mode="head",
        sample_seed=0,
    )

    assert judged["provider"] == "openai"
    assert judged["model"] == "gpt-4o"
    assert judged["items"][0]["judge_status"] == "scored"
    assert judged["items"][1]["judge_status"] == "transport_error"


def test_judge_pairs_document_can_limit_random_subset():
    doc = {
        "checkpoint": "dummy",
        "split": "test_alter",
        "pairs": [
            {"id": i, "ground_truth": f"gt{i}", "generation": f"gen{i}"}
            for i in range(8)
        ],
    }

    def fake_judge(_ground_truth, _generation):
        return {
            "signals_in_gt": {
                "has_direction_anchor": False,
                "has_action_demand": False,
                "has_hazard_or_path_state": False,
            },
            "criteria": {
                "safety_correctness": {"applicable": True, "label": "Acceptable", "rationale": "ok"},
                "hazard_path_state_fidelity": {"applicable": False, "label": None, "rationale": "na"},
                "direction_fidelity": {"applicable": False, "label": None, "rationale": "na"},
                "action_usefulness": {"applicable": False, "label": None, "rationale": "na"},
                "spoken_guidance_quality": {"applicable": True, "label": "Acceptable", "rationale": "ok"},
            },
            "overall_rationale": "ok",
        }, '{"mock": "raw"}'

    judged = judge_pairs_document(
        doc,
        provider="openrouter",
        model="openai/gpt-4o-mini",
        judge_callable=fake_judge,
        prompt_version="v1",
        schema_version="v1",
        limit=3,
        offset=0,
        sample_mode="random",
        sample_seed=11,
    )

    assert len(judged["items"]) == 3


def test_judge_pairs_document_emits_preview_for_first_selected_sample():
    doc = {
        "checkpoint": "dummy",
        "split": "test_alter",
        "pairs": [
            {"id": 7, "ground_truth": "gt7", "generation": "gen7"},
            {"id": 8, "ground_truth": "gt8", "generation": "gen8"},
        ],
    }

    messages = []

    def fake_emit(message):
        messages.append(message)

    def fake_judge(_ground_truth, _generation):
        return {
            "signals_in_gt": {
                "has_direction_anchor": False,
                "has_action_demand": False,
                "has_hazard_or_path_state": False,
            },
            "criteria": {
                "safety_correctness": {"applicable": True, "label": "Acceptable", "rationale": "ok"},
                "hazard_path_state_fidelity": {"applicable": False, "label": None, "rationale": "na"},
                "direction_fidelity": {"applicable": False, "label": None, "rationale": "na"},
                "action_usefulness": {"applicable": False, "label": None, "rationale": "na"},
                "spoken_guidance_quality": {"applicable": True, "label": "Acceptable", "rationale": "ok"},
            },
            "overall_rationale": "ok",
        }, '{"mock": "raw"}'

    judge_pairs_document(
        doc,
        provider="openai",
        model="gpt-4o",
        judge_callable=fake_judge,
        prompt_version="v1",
        schema_version="v1",
        preview_count=1,
        offset=0,
        emit=fake_emit,
    )

    preview_blob = "\n".join(messages)
    assert "Previewing 1 selected sample(s)" in preview_blob
    assert "id=7" in preview_blob
    assert "ground_truth: gt7" in preview_blob
    assert "generation: gen7" in preview_blob


def test_judge_pairs_document_uses_progress_factory_when_requested():
    doc = {
        "checkpoint": "dummy",
        "split": "test_alter",
        "pairs": [
            {"id": i, "ground_truth": f"gt{i}", "generation": f"gen{i}"}
            for i in range(3)
        ],
    }

    progress_calls = []

    def fake_progress(iterable, total=None, desc=None):
        progress_calls.append({"total": total, "desc": desc})
        return iterable

    def fake_judge(_ground_truth, _generation):
        return {
            "signals_in_gt": {
                "has_direction_anchor": False,
                "has_action_demand": False,
                "has_hazard_or_path_state": False,
            },
            "criteria": {
                "safety_correctness": {"applicable": True, "label": "Acceptable", "rationale": "ok"},
                "hazard_path_state_fidelity": {"applicable": False, "label": None, "rationale": "na"},
                "direction_fidelity": {"applicable": False, "label": None, "rationale": "na"},
                "action_usefulness": {"applicable": False, "label": None, "rationale": "na"},
                "spoken_guidance_quality": {"applicable": True, "label": "Acceptable", "rationale": "ok"},
            },
            "overall_rationale": "ok",
        }, '{"mock": "raw"}'

    judge_pairs_document(
        doc,
        provider="openrouter",
        model="openai/gpt-4o-mini",
        judge_callable=fake_judge,
        prompt_version="v1",
        schema_version="v1",
        offset=0,
        show_progress=True,
        progress_factory=fake_progress,
    )

    assert progress_calls == [{"total": 3, "desc": "Judging pairs"}]


def test_judge_pairs_document_appends_to_existing_output_and_skips_existing_ids(tmp_path):
    doc = {
        "checkpoint": "dummy",
        "split": "test_alter",
        "pairs": [
            {"id": 0, "ground_truth": "gt0", "generation": "gen0"},
            {"id": 1, "ground_truth": "gt1", "generation": "gen1"},
            {"id": 2, "ground_truth": "gt2", "generation": "gen2"},
        ],
    }
    output_path = tmp_path / "judged.json"
    existing = {
        "items": [
            {
                "id": 0,
                "ground_truth": "gt0",
                "generation": "gen0",
                "judge_status": "scored",
                "raw_response": "{}",
                "judge_output": {
                    "signals_in_gt": {
                        "has_direction_anchor": False,
                        "has_action_demand": False,
                        "has_hazard_or_path_state": False,
                    },
                    "criteria": {
                        "safety_correctness": {"applicable": True, "label": "Acceptable", "rationale": "ok"},
                        "hazard_path_state_fidelity": {"applicable": False, "label": None, "rationale": "na"},
                        "direction_fidelity": {"applicable": False, "label": None, "rationale": "na"},
                        "action_usefulness": {"applicable": False, "label": None, "rationale": "na"},
                        "spoken_guidance_quality": {"applicable": True, "label": "Acceptable", "rationale": "ok"},
                    },
                    "overall_rationale": "ok",
                },
                "error_type": None,
                "error_message": None,
            }
        ]
    }

    calls = []

    def fake_judge(ground_truth, generation):
        calls.append((ground_truth, generation))
        return {
            "signals_in_gt": {
                "has_direction_anchor": False,
                "has_action_demand": False,
                "has_hazard_or_path_state": False,
            },
            "criteria": {
                "safety_correctness": {"applicable": True, "label": "Acceptable", "rationale": "ok"},
                "hazard_path_state_fidelity": {"applicable": False, "label": None, "rationale": "na"},
                "direction_fidelity": {"applicable": False, "label": None, "rationale": "na"},
                "action_usefulness": {"applicable": False, "label": None, "rationale": "na"},
                "spoken_guidance_quality": {"applicable": True, "label": "Acceptable", "rationale": "ok"},
            },
            "overall_rationale": "ok",
        }, '{"mock":"raw"}'

    judged = judge_pairs_document(
        doc,
        provider="openai",
        model="gpt-4o",
        judge_callable=fake_judge,
        prompt_version="v1",
        schema_version="v1",
        limit=2,
        offset=1,
        sample_mode="head",
        sample_seed=0,
        existing_judged=existing,
        output_path=output_path,
        flush_every=1,
        show_progress=False,
    )

    assert [item["id"] for item in judged["items"]] == [0, 1, 2]
    assert calls == [("gt1", "gen1"), ("gt2", "gen2")]
    assert output_path.exists()
