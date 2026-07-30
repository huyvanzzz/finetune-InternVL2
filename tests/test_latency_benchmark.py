import pytest
import torch


def test_decode_only_tokens_per_second_excludes_first_token():
    from scripts.benchmark_latency import compute_decode_only_tokens_per_s, compute_e2e_at_n_tokens_ms

    assert compute_decode_only_tokens_per_s(6, 0.5) == 10.0
    assert compute_decode_only_tokens_per_s(1, 0.5) == 0.0
    assert compute_decode_only_tokens_per_s(8, 0.0) == 0.0
    assert compute_e2e_at_n_tokens_ms(1000.0, 500.0, 10.0, target_tokens=20) == 2400.0
    assert compute_e2e_at_n_tokens_ms(1000.0, 500.0, 0.0, target_tokens=20) == 0.0


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
        question_token_count=5,
        num_image_patches=1,
        model_num_image_token=32,
        runtime_call_counts={
            "extract_feature_call_count": 1,
            "qformer_encode_call_count": 1,
            "set_qformer_text_call_count": 1,
            "clear_qformer_text_call_count": 1,
        },
        phase_breakdown_ms={
            "image_preprocess_ms": 2.0,
            "vision_forward_ms": 3.0,
            "qformer_text_encode_ms": 4.0,
            "qformer_forward_ms": 5.0,
            "qformer_projection_ms": 6.0,
        },
    )

    assert record["timing"]["trajectory_ms"] == 0.0
    assert record["timing"]["decode_only_tokens_per_s"] == pytest.approx(3 / 0.007)
    assert record["timing"]["ttft_ms"] == 2.0
    assert record["timing"]["e2e_at_20_tokens_ms"] == pytest.approx(12.0 - 7.0 + (19 / (3 / 0.007)) * 1000)
    assert record["timing"]["image_preprocess_ms"] == 2.0
    assert record["timing"]["vision_forward_ms"] == 3.0
    assert record["timing"]["qformer_text_encode_ms"] == 4.0
    assert record["timing"]["qformer_forward_ms"] == 5.0
    assert record["timing"]["qformer_projection_ms"] == 6.0
    assert record["question_token_count"] == 5
    assert record["num_image_patches"] == 1
    assert record["image_context_token_count"] == 32
    assert record["extract_feature_call_count"] == 1
    assert record["qformer_encode_call_count"] == 1
    assert record["set_qformer_text_call_count"] == 1
    assert record["clear_qformer_text_call_count"] == 1


def test_runtime_call_counter_hooks_count_qformer_methods_once():
    from scripts.benchmark_latency import (
        get_model_runtime_call_counts,
        install_runtime_call_counter_hooks,
        reset_model_latency_phases,
        reset_model_runtime_call_counts,
    )

    class Model:
        def extract_feature(self, pixel_values):
            return pixel_values

        def encode_qformer_texts(self, texts):
            return texts

        def set_qformer_text(self, input_ids, attention_mask):
            return (input_ids, attention_mask)

        def clear_qformer_text(self):
            return None

    model = Model()
    install_runtime_call_counter_hooks(model)
    reset_model_latency_phases(model)
    reset_model_runtime_call_counts(model)

    assert model.extract_feature("pixels") == "pixels"
    assert model.encode_qformer_texts(["prompt"]) == ["prompt"]
    assert model.set_qformer_text("ids", "mask") == ("ids", "mask")
    assert model.clear_qformer_text() is None

    assert get_model_runtime_call_counts(model) == {
        "extract_feature_call_count": 1,
        "qformer_encode_call_count": 1,
        "set_qformer_text_call_count": 1,
        "clear_qformer_text_call_count": 1,
    }
    assert model._latency_phase_ms["qformer_text_encode_ms"] >= 0.0


def test_runtime_call_counter_defaults_missing_methods_to_zero():
    from scripts.benchmark_latency import (
        get_model_runtime_call_counts,
        install_runtime_call_counter_hooks,
        reset_model_runtime_call_counts,
    )

    class Model:
        pass

    model = Model()
    install_runtime_call_counter_hooks(model)
    reset_model_runtime_call_counts(model)

    assert get_model_runtime_call_counts(model) == {
        "extract_feature_call_count": 0,
        "qformer_encode_call_count": 0,
        "set_qformer_text_call_count": 0,
        "clear_qformer_text_call_count": 0,
    }


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
        runtime_diagnostics={
            "torch_version": "2.11.0+cu128",
            "torch_cuda_version": "12.8",
            "cuda_device_name": "NVIDIA GeForce RTX 4090",
            "model_num_image_token": 32,
        },
    )

    assert metadata["trajectory_enabled"] is True
    assert metadata["trajectory_fusion_mode"] == "concat"
    assert metadata["object_tracking_included"] is False
    assert metadata["torch_cuda_version"] == "12.8"
    assert metadata["model_num_image_token"] == 32


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
            "flash_attention_layer_count": 2,
            "flash_attention_active_layer_count": 1,
            "flash_attention_inactive_reason": None,
        },
    )

    assert metadata["flash_attention_requested"] is True
    assert metadata["flash_attention_available"] is True
    assert metadata["flash_attention_active"] is True
    assert metadata["flash_attention_layer_count"] == 2
    assert metadata["flash_attention_active_layer_count"] == 1
    assert metadata["flash_attention_inactive_reason"] is None


def test_collect_flash_attention_status_counts_seen_and_active_layers(monkeypatch):
    import runtime_flash_attention

    class Layer:
        def __init__(self, use_flash_attn):
            self.use_flash_attn = use_flash_attn

    class VisionModel:
        def modules(self):
            return [Layer(True), Layer(False), object()]

    class Model:
        vision_model = VisionModel()

    monkeypatch.setattr(runtime_flash_attention, "flash_attention_available", lambda: True)

    status = runtime_flash_attention.collect_flash_attention_status(Model(), requested=True)

    assert status["flash_attention_active"] is True
    assert status["flash_attention_layer_count"] == 2
    assert status["flash_attention_active_layer_count"] == 1
    assert status["flash_attention_inactive_reason"] is None


def test_collect_flash_attention_status_detects_attn_implementation(monkeypatch):
    import runtime_flash_attention

    class Layer(torch.nn.Module):
        def __init__(self, implementation):
            super().__init__()
            self._attn_implementation = implementation

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fast = Layer("flash_attention_2")
            self.slow = Layer("eager")

    monkeypatch.setattr(runtime_flash_attention, "flash_attention_available", lambda: True)

    status = runtime_flash_attention.collect_flash_attention_status(Model(), requested=True)

    assert status["flash_attention_active"] is True
    assert status["flash_attention_layer_count"] == 2
    assert status["flash_attention_active_layer_count"] == 1
    assert status["flash_attention_inactive_reason"] is None


def test_collect_flash_attention_status_records_inactive_reason(monkeypatch):
    import runtime_flash_attention

    class Model:
        pass

    monkeypatch.setattr(runtime_flash_attention, "flash_attention_available", lambda: True)

    status = runtime_flash_attention.collect_flash_attention_status(Model(), requested=True)

    assert status["flash_attention_active"] is False
    assert status["flash_attention_layer_count"] == 0
    assert status["flash_attention_inactive_reason"] == "no_flash_attention_markers_detected"


def test_flash_attention_from_pretrained_kwargs_only_when_available(monkeypatch):
    import runtime_flash_attention

    config = {"model": {"flash_attention": {"enabled": True}}}
    monkeypatch.setattr(runtime_flash_attention, "flash_attention_available", lambda: True)

    assert runtime_flash_attention.flash_attention_from_pretrained_kwargs(config) == {
        "attn_implementation": "flash_attention_2"
    }

    monkeypatch.setattr(runtime_flash_attention, "flash_attention_available", lambda: False)

    assert runtime_flash_attention.flash_attention_from_pretrained_kwargs(config) == {}


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


def test_latency_generation_mode_defaults_to_greedy_decode_timing():
    from scripts.benchmark_latency import build_generation_config

    generation_config, validity = build_generation_config(
        generation_mode="latency_greedy",
        max_new_tokens=512,
        num_beams=1,
        do_sample=False,
    )

    assert generation_config == {
        "max_new_tokens": 512,
        "num_beams": 1,
        "do_sample": False,
    }
    assert validity == {
        "generation_mode": "latency_greedy",
        "decode_only_tokens_per_s_valid": True,
        "decode_only_tokens_per_s_warning": None,
    }


def test_restore_eval_generation_mode_matches_779_generation_contract():
    from scripts.benchmark_latency import build_generation_config

    generation_config, validity = build_generation_config(
        generation_mode="restore_eval",
        max_new_tokens=128,
        num_beams=1,
        do_sample=True,
    )

    assert generation_config == {
        "max_new_tokens": 128,
        "num_beams": 3,
        "do_sample": False,
        "repetition_penalty": 1.3,
        "early_stopping": True,
    }
    assert validity["decode_only_tokens_per_s_valid"] is False
    assert "restore_eval" in validity["decode_only_tokens_per_s_warning"]


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


def test_iter_latency_indices_can_disable_progress_for_clean_logs():
    from scripts.benchmark_latency import iter_latency_indices

    assert list(iter_latency_indices(3, progress=False)) == [0, 1, 2]
