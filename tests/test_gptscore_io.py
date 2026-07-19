import json

from gptscore.io_utils import load_existing_json, load_pairs_file, select_pair_items
from gptscore.run_judge import default_output_path as judge_default_output_path
from gptscore.score_results import default_output_path as score_default_output_path


def test_load_pairs_file_reads_existing_pairs_shape(tmp_path):
    path = tmp_path / "pairs.json"
    path.write_text(
        json.dumps(
            {
                "checkpoint": "dummy",
                "split": "test_alter",
                "pairs": [
                    {"id": 0, "ground_truth": "gt0", "generation": "gen0"},
                    {"id": 1, "ground_truth": "gt1", "generation": "gen1"},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    loaded = load_pairs_file(path)

    assert loaded["checkpoint"] == "dummy"
    assert loaded["split"] == "test_alter"
    assert [item["id"] for item in loaded["pairs"]] == [0, 1]


def test_select_pair_items_head_limit_keeps_first_n():
    doc = {
        "pairs": [
            {"id": 0, "ground_truth": "gt0", "generation": "gen0"},
            {"id": 1, "ground_truth": "gt1", "generation": "gen1"},
            {"id": 2, "ground_truth": "gt2", "generation": "gen2"},
        ]
    }

    selected = select_pair_items(doc, limit=2, sample_mode="head", sample_seed=123)

    assert [item["id"] for item in selected] == [0, 1]


def test_select_pair_items_random_limit_is_reproducible():
    doc = {
        "pairs": [
            {"id": i, "ground_truth": f"gt{i}", "generation": f"gen{i}"}
            for i in range(6)
        ]
    }

    selected_a = select_pair_items(doc, limit=3, sample_mode="random", sample_seed=7)
    selected_b = select_pair_items(doc, limit=3, sample_mode="random", sample_seed=7)

    assert [item["id"] for item in selected_a] == [item["id"] for item in selected_b]
    assert len(selected_a) == 3


def test_select_pair_items_head_with_offset_starts_from_requested_position():
    doc = {
        "pairs": [
            {"id": i, "ground_truth": f"gt{i}", "generation": f"gen{i}"}
            for i in range(6)
        ]
    }

    selected = select_pair_items(doc, limit=2, offset=3, sample_mode="head", sample_seed=0)

    assert [item["id"] for item in selected] == [3, 4]


def test_load_existing_json_returns_none_when_missing(tmp_path):
    assert load_existing_json(tmp_path / "missing.json") is None


def test_default_output_paths_always_target_results_folder():
    assert judge_default_output_path("input_json/a.json") == "results\\a_gptscore_judged.json"
    assert judge_default_output_path("input_json/a_pairs.json") == "results\\a_gptscore_judged.json"
    assert score_default_output_path("input_json/a_gptscore_judged.json") == "results\\a_gptscore_scored.json"
