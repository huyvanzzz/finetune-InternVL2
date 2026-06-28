from scripts.check_data_leak import (
    classify_sample_type,
    compute_overlap_report,
    normalize_text,
    sample_field_value,
)


def test_classify_sample_type_detects_qa_and_alter():
    assert classify_sample_type({"QA": {"Q": "Where?", "A": "Left"}}) == "qa"
    assert classify_sample_type({"alter": "Slow down"}) == "alter"


def test_normalize_text_collapses_whitespace_and_case():
    assert normalize_text("  Hello   WORLD \n") == "hello world"
    assert normalize_text("") == ""
    assert normalize_text(None) == ""


def test_sample_field_value_returns_expected_text_fields():
    sample = {
        "frame_path": "abc.frame",
        "video": "abc.mp4",
        "summary": "A summary",
        "alter": "Slow down",
        "QA": {"Q": "Where?", "A": "Left"},
    }

    assert sample_field_value(sample, "frame_path") == "abc.frame"
    assert sample_field_value(sample, "video") == "abc.mp4"
    assert sample_field_value(sample, "summary") == "a summary"
    assert sample_field_value(sample, "alter") == "slow down"
    assert sample_field_value(sample, "qa_question") == "where?"
    assert sample_field_value(sample, "qa_answer") == "left"


def test_compute_overlap_report_counts_leaked_samples_per_type():
    train_samples = [
        {"frame_path": "a.frame", "video": "a.mp4", "alter": "Slow down"},
        {"frame_path": "b.frame", "video": "b.mp4", "QA": {"Q": "Where?", "A": "Left"}},
    ]
    test_samples = [
        {"frame_path": "a.frame", "video": "a.mp4", "alter": "Other"},
        {"frame_path": "c.frame", "video": "c.mp4", "QA": {"Q": "Where?", "A": "Right"}},
        {"frame_path": "d.frame", "video": "d.mp4", "alter": "No leak"},
    ]

    report = compute_overlap_report(train_samples, test_samples, key_name="frame_path")

    assert report["overlap_count"] == 1
    assert report["overlap_rate"] == 1 / 3
    assert report["per_type"]["alter"]["leaked"] == 1
    assert report["per_type"]["alter"]["total"] == 2
    assert report["per_type"]["qa"]["leaked"] == 0
    assert report["per_type"]["qa"]["total"] == 1
