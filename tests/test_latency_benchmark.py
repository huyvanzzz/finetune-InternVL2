import pytest


def test_decode_only_tokens_per_second_excludes_first_token():
    from scripts.benchmark_latency import compute_decode_only_tokens_per_s

    assert compute_decode_only_tokens_per_s(6, 0.5) == 10.0
    assert compute_decode_only_tokens_per_s(1, 0.5) == 0.0
    assert compute_decode_only_tokens_per_s(8, 0.0) == 0.0


def test_disabled_object_tracking_timing_is_reserved_for_v2():
    from scripts.benchmark_latency import disabled_object_tracking_timing

    timing = disabled_object_tracking_timing()

    assert timing == {
        "object_tracking_included": False,
        "object_tracking_ms": 0.0,
    }


def test_summary_aggregation_computes_mean_p50_and_p90():
    from scripts.benchmark_latency import summarize_latency_samples

    samples = [
        {"timing": {"end_to_end_ms": 10.0, "decode_only_tokens_per_s": 20.0, "vision_ms": 2.0, "trajectory_ms": 0.0, "llm_ms": 8.0, "object_tracking_ms": 0.0}},
        {"timing": {"end_to_end_ms": 20.0, "decode_only_tokens_per_s": 30.0, "vision_ms": 4.0, "trajectory_ms": 1.0, "llm_ms": 15.0, "object_tracking_ms": 0.0}},
        {"timing": {"end_to_end_ms": 30.0, "decode_only_tokens_per_s": 40.0, "vision_ms": 6.0, "trajectory_ms": 2.0, "llm_ms": 22.0, "object_tracking_ms": 0.0}},
    ]

    summary = summarize_latency_samples(samples)

    assert summary["end_to_end_ms"]["mean"] == 20.0
    assert summary["end_to_end_ms"]["p50"] == 20.0
    assert summary["end_to_end_ms"]["p90"] == 30.0
    assert summary["decode_only_tokens_per_s"]["mean"] == 30.0
    assert summary["breakdown_mean_ms"]["vision_ms"] == 4.0
    assert summary["breakdown_mean_ms"]["trajectory_ms"] == 1.0


def test_non_trajectory_sample_uses_zero_trajectory_timing():
    from scripts.benchmark_latency import build_sample_record

    record = build_sample_record(
        sample_id=3,
        question="q",
        prediction="a",
        generated_token_count=4,
        end_to_end_ms=12.0,
        vision_ms=3.0,
        trajectory_ms=None,
        llm_ms=9.0,
        first_token_ms=2.0,
        decode_after_first_token_ms=7.0,
        object_tracking_included=False,
        object_tracking_ms=0.0,
    )

    assert record["timing"]["trajectory_ms"] == 0.0
    assert record["timing"]["decode_only_tokens_per_s"] == pytest.approx(3 / 0.007)


def test_run_metadata_records_trajectory_fusion_mode():
    from scripts.benchmark_latency import build_run_metadata

    metadata = build_run_metadata(
        config_path="internvl_config_traj_concat.yaml",
        checkpoint="ckpt",
        split="test_alter",
        device="cuda:0",
        config={
            "model": {"architecture": "internvl", "qformer": {"enabled": True}},
            "trajectory": {"enabled": True, "fusion_mode": "concat"},
        },
        generation_config={"num_beams": 1, "do_sample": False},
        timing_mode="greedy_decode_token_timing",
        object_tracking_included=False,
    )

    assert metadata["trajectory_enabled"] is True
    assert metadata["trajectory_fusion_mode"] == "concat"
    assert metadata["object_tracking_included"] is False


def test_flash_attention_metadata_records_requested_available_and_active():
    from scripts.benchmark_latency import build_run_metadata

    metadata = build_run_metadata(
        config_path="internvl_config.yaml",
        checkpoint=None,
        split="test_alter",
        device="cuda:0",
        config={"model": {"architecture": "internvl", "flash_attention": {"enabled": True}}},
        generation_config={"num_beams": 1, "do_sample": False},
        timing_mode="greedy_decode_token_timing",
        object_tracking_included=False,
        flash_attention_status={
            "flash_attention_requested": True,
            "flash_attention_available": True,
            "flash_attention_active": True,
        },
    )

    assert metadata["flash_attention_requested"] is True
    assert metadata["flash_attention_available"] is True
    assert metadata["flash_attention_active"] is True


def test_auto_quantization_disables_bnb_on_cuda_13():
    from scripts.benchmark_latency import resolve_quantization_policy

    config = {"model": {"quantization": {"enabled": True}}}

    policy = resolve_quantization_policy(config, mode="auto", torch_cuda_version="13.0", bitsandbytes_available=True)

    assert policy == {
        "quantization_requested": True,
        "quantization_effective": False,
        "quantization_mode": "auto",
        "quantization_disable_reason": "bitsandbytes_cuda13_unsupported",
    }


def test_config_quantization_keeps_requested_policy_even_on_cuda_13():
    from scripts.benchmark_latency import resolve_quantization_policy

    config = {"model": {"quantization": {"enabled": True}}}

    policy = resolve_quantization_policy(config, mode="config", torch_cuda_version="13.0", bitsandbytes_available=True)

    assert policy["quantization_requested"] is True
    assert policy["quantization_effective"] is True
    assert policy["quantization_mode"] == "config"
    assert policy["quantization_disable_reason"] is None


def test_enable_flash_attention_sets_nested_vision_config():
    from scripts.benchmark_latency import enable_flash_attention_for_config

    class VisionConfig:
        use_flash_attn = False

    class ModelConfig:
        vision_config = VisionConfig()

    model_config = ModelConfig()

    changed = enable_flash_attention_for_config(model_config, requested=True)

    assert changed is True
    assert model_config.vision_config.use_flash_attn is True


def test_latency_generation_config_rejects_beam_search():
    from scripts.benchmark_latency import validate_latency_generation_config

    validate_latency_generation_config({"num_beams": 1, "do_sample": False})

    with pytest.raises(ValueError, match="num_beams=1"):
        validate_latency_generation_config({"num_beams": 3, "do_sample": False})

    with pytest.raises(ValueError, match="do_sample=false"):
        validate_latency_generation_config({"num_beams": 1, "do_sample": True})


def test_token_timing_streamer_splits_first_token_from_decode_tail():
    from scripts.benchmark_latency import TokenTimingStreamer

    events = iter([10.0, 10.2, 10.5, 10.9])
    streamer = TokenTimingStreamer(clock=lambda: next(events), sync_fn=lambda: None)

    streamer.start()
    streamer.put([1])
    streamer.put([2])
    streamer.put([3])

    assert streamer.first_token_ms == pytest.approx(200.0)
    assert streamer.decode_after_first_token_ms == pytest.approx(700.0)
