import pytest


def test_bucket_paired_latency_rows_groups_same_token_count():
    from scripts.analyze_latency_pair import bucket_paired_latency_rows

    rows = [
        {"generated_token_delta": -1, "timing_delta": {"end_to_end_ms": -10.0}},
        {"generated_token_delta": 0, "timing_delta": {"end_to_end_ms": 5.0}},
        {"generated_token_delta": 2, "timing_delta": {"end_to_end_ms": 20.0}},
        {"generated_token_delta": 5, "timing_delta": {"end_to_end_ms": 50.0}},
    ]

    buckets = bucket_paired_latency_rows(rows)

    assert buckets["qformer_shorter"]["count"] == 1
    assert buckets["same_generated_token_count"]["count"] == 1
    assert buckets["qformer_longer_1_to_3"]["count"] == 1
    assert buckets["qformer_longer_4_plus"]["count"] == 1
    assert buckets["same_generated_token_count"]["end_to_end_ms_delta"]["mean"] == 5.0


def test_build_paired_latency_rows_reports_field_deltas():
    from scripts.analyze_latency_pair import build_paired_latency_rows

    no_qformer = {
        "samples": [
            {
                "id": 7,
                "generated_token_count": 12,
                "timing": {
                    "end_to_end_ms": 100.0,
                    "first_token_ms": 20.0,
                    "decode_after_first_token_ms": 70.0,
                    "decode_only_tokens_per_s": 15.0,
                },
            }
        ]
    }
    qformer = {
        "samples": [
            {
                "id": 7,
                "generated_token_count": 12,
                "timing": {
                    "end_to_end_ms": 80.0,
                    "first_token_ms": 25.0,
                    "decode_after_first_token_ms": 45.0,
                    "decode_only_tokens_per_s": 20.0,
                },
            }
        ]
    }

    rows = build_paired_latency_rows(no_qformer, qformer)

    assert rows[0]["id"] == 7
    assert rows[0]["generated_token_delta"] == 0
    assert rows[0]["timing_delta"]["end_to_end_ms"] == -20.0
    assert rows[0]["timing_delta"]["first_token_ms"] == 5.0
    assert rows[0]["timing_delta"]["decode_only_tokens_per_s"] == pytest.approx(5.0)
