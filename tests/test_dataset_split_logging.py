from collections import Counter


def test_split_counter_treats_qa_and_alter_separately():
    from wad_dataset import summarize_task_counts_from_indices

    metadata = [
        {"QA": {"Q": "q1", "A": "a1"}, "alter": "alt1"},
        {"alter": "alt2"},
        {"QA": {"Q": "q2", "A": "a2"}, "alter": "alt3"},
        {"alter": "alt4"},
    ]

    counts = summarize_task_counts_from_indices(metadata, [0, 1, 3])

    assert counts == Counter({"alter": 2, "QA": 1})
