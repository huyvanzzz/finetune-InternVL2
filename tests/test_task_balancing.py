from collections import Counter

from wad_dataset import (
    build_balanced_sample_weights,
    get_sample_task_type,
    resolve_eval_limit,
    summarize_task_types,
)


def test_get_sample_task_type_detects_qa_and_alter():
    qa_sample = {"QA": {"Q": "where", "A": "left"}}
    alter_sample = {"alter": "move forward"}

    assert get_sample_task_type(qa_sample) == "qa"
    assert get_sample_task_type(alter_sample) == "alter"


def test_summarize_task_types_counts_tasks():
    samples = [
        {"QA": {"Q": "a", "A": "b"}},
        {"alter": "x"},
        {"alter": "y"},
    ]

    assert summarize_task_types(samples) == {"qa": 1, "alter": 2}


def test_build_balanced_sample_weights_matches_target_task_mix():
    task_types = ["qa", "qa", "alter", "alter", "alter", "alter"]
    weights = build_balanced_sample_weights(
        task_types,
        task_target_weights={"qa": 0.4, "alter": 0.6},
    )

    assert len(weights) == len(task_types)

    total_weight = sum(weights)
    weight_by_task = Counter()
    for task_type, weight in zip(task_types, weights):
        weight_by_task[task_type] += weight

    assert abs((weight_by_task["qa"] / total_weight) - 0.4) < 1e-6
    assert abs((weight_by_task["alter"] / total_weight) - 0.6) < 1e-6


def test_resolve_eval_limit_disables_limit_for_none_or_non_positive_values():
    assert resolve_eval_limit(None) is None
    assert resolve_eval_limit(0) is None
    assert resolve_eval_limit(-1) is None
    assert resolve_eval_limit(200) == 200
