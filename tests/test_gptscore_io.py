import json

from gptscore.io_utils import load_pairs_file, select_pair_items


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

