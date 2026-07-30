import argparse
import json
import math
import os
import time
from statistics import mean
from typing import Dict, Iterable, List, Optional

from runtime_flash_attention import (
    collect_flash_attention_status,
    enable_flash_attention_for_config,
    flash_attention_requested,
)


TIMING_MODE = "greedy_decode_token_timing"
RESTORE_EVAL_TOKEN_WARNING = (
    "restore_eval uses beam search to match restore-779cc7b eval behavior; "
    "decode_only_tokens_per_s is recorded for debugging only, not as the primary latency metric."
)
RUNTIME_CALL_COUNT_FIELDS = {
    "extract_feature": "extract_feature_call_count",
    "encode_qformer_texts": "qformer_encode_call_count",
    "set_qformer_text": "set_qformer_text_call_count",
    "clear_qformer_text": "clear_qformer_text_call_count",
}


def compute_decode_only_tokens_per_s(generated_token_count: int, decode_after_first_token_seconds: float) -> float:
    if generated_token_count <= 1 or decode_after_first_token_seconds <= 0:
        return 0.0
    return (int(generated_token_count) - 1) / float(decode_after_first_token_seconds)


def disabled_object_tracking_timing() -> Dict[str, object]:
    return {
        "object_tracking_included": False,
        "object_tracking_ms": 0.0,
    }


def validate_latency_generation_config(generation_config: Dict):
    if int(generation_config.get("num_beams", 1)) != 1:
        raise ValueError("Latency benchmark decode-only tokens/s requires num_beams=1.")
    if bool(generation_config.get("do_sample", False)):
        raise ValueError("Latency benchmark decode-only tokens/s requires do_sample=false.")


def build_generation_config(
    generation_mode: str,
    max_new_tokens: int,
    num_beams: int,
    do_sample: bool,
):
    mode = str(generation_mode or "latency_greedy").strip().lower()
    if mode == "latency_greedy":
        generation_config = {
            "max_new_tokens": int(max_new_tokens),
            "num_beams": int(num_beams),
            "do_sample": bool(do_sample),
        }
        validate_latency_generation_config(generation_config)
        return generation_config, {
            "generation_mode": "latency_greedy",
            "decode_only_tokens_per_s_valid": True,
            "decode_only_tokens_per_s_warning": None,
        }
    if mode == "restore_eval":
        return {
            "max_new_tokens": int(max_new_tokens),
            "num_beams": 3,
            "do_sample": False,
            "repetition_penalty": 1.3,
            "early_stopping": True,
        }, {
            "generation_mode": "restore_eval",
            "decode_only_tokens_per_s_valid": False,
            "decode_only_tokens_per_s_warning": RESTORE_EVAL_TOKEN_WARNING,
        }
    raise ValueError(f"Unsupported generation mode: {generation_mode}")


def iter_latency_indices(sample_limit: int, progress: bool = True):
    indices = range(int(sample_limit))
    if not progress:
        return indices
    try:
        from tqdm import tqdm
    except Exception:
        return indices
    return tqdm(indices, desc="Benchmarking latency", unit="sample")


def bitsandbytes_available() -> bool:
    try:
        import importlib.metadata

        importlib.metadata.version("bitsandbytes")
    except Exception:
        return False
    return True


def resolve_quantization_policy(
    config: Dict,
    mode: str = "auto",
    torch_cuda_version: Optional[str] = None,
    bitsandbytes_available: Optional[bool] = None,
) -> Dict[str, object]:
    mode = str(mode or "auto").strip().lower()
    if mode not in {"auto", "config", "on", "off"}:
        raise ValueError(f"Unsupported quantization mode: {mode}")

    requested = bool(config.get("model", {}).get("quantization", {}).get("enabled", False))
    if mode == "on":
        requested = True
    elif mode == "off":
        requested = False

    effective = requested
    disable_reason = None
    if not requested:
        effective = False
    elif mode == "auto":
        bnb_ok = bitsandbytes_available if bitsandbytes_available is not None else globals()["bitsandbytes_available"]()
        if not bnb_ok:
            effective = False
            disable_reason = "bitsandbytes_not_installed"
        elif str(torch_cuda_version or "").startswith("13."):
            effective = False
            disable_reason = "bitsandbytes_cuda13_unsupported"

    return {
        "quantization_requested": bool(requested),
        "quantization_effective": bool(effective),
        "quantization_mode": mode,
        "quantization_disable_reason": disable_reason,
    }


def _percentile(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = max(0, min(len(ordered) - 1, math.ceil((percentile / 100.0) * len(ordered)) - 1))
    return ordered[index]


def _metric_summary(values: Iterable[float]) -> Dict[str, float]:
    values = [float(value) for value in values]
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p90": 0.0}
    return {
        "mean": round(mean(values), 6),
        "p50": round(_percentile(values, 50), 6),
        "p90": round(_percentile(values, 90), 6),
    }


def summarize_latency_samples(samples: List[Dict]) -> Dict:
    timing_rows = [sample["timing"] for sample in samples]
    breakdown_fields = ("object_tracking_ms", "vision_ms", "trajectory_ms", "llm_ms")
    return {
        "end_to_end_ms": _metric_summary(row["end_to_end_ms"] for row in timing_rows),
        "decode_only_tokens_per_s": _metric_summary(row["decode_only_tokens_per_s"] for row in timing_rows),
        "breakdown_mean_ms": {
            field: round(mean([float(row.get(field, 0.0)) for row in timing_rows]), 6) if timing_rows else 0.0
            for field in breakdown_fields
        },
    }


def build_sample_record(
    sample_id: int,
    question: str,
    prediction: str,
    generated_token_count: int,
    end_to_end_ms: float,
    vision_ms: float,
    trajectory_ms: Optional[float],
    llm_ms: float,
    first_token_ms: float,
    decode_after_first_token_ms: float,
    object_tracking_included: bool,
    object_tracking_ms: float,
    question_token_count: Optional[int] = None,
    num_image_patches: Optional[int] = None,
    model_num_image_token: Optional[int] = None,
    runtime_call_counts: Optional[Dict[str, int]] = None,
) -> Dict:
    decode_seconds = float(decode_after_first_token_ms) / 1000.0
    image_context_token_count = None
    if num_image_patches is not None and model_num_image_token is not None:
        image_context_token_count = int(num_image_patches) * int(model_num_image_token)
    runtime_call_counts = runtime_call_counts or {}

    return {
        "id": int(sample_id),
        "question": question,
        "prediction": prediction,
        "generated_token_count": int(generated_token_count),
        "question_token_count": int(question_token_count) if question_token_count is not None else None,
        "num_image_patches": int(num_image_patches) if num_image_patches is not None else None,
        "model_num_image_token": int(model_num_image_token) if model_num_image_token is not None else None,
        "image_context_token_count": image_context_token_count,
        **{
            field: int(runtime_call_counts.get(field, 0))
            for field in RUNTIME_CALL_COUNT_FIELDS.values()
        },
        "timing": {
            "end_to_end_ms": float(end_to_end_ms),
            "object_tracking_included": bool(object_tracking_included),
            "object_tracking_ms": float(object_tracking_ms),
            "vision_ms": float(vision_ms),
            "trajectory_ms": float(trajectory_ms or 0.0),
            "llm_ms": float(llm_ms),
            "first_token_ms": float(first_token_ms),
            "decode_after_first_token_ms": float(decode_after_first_token_ms),
            "decode_only_tokens_per_s": compute_decode_only_tokens_per_s(generated_token_count, decode_seconds),
        },
    }


def build_run_metadata(
    config_path: str,
    checkpoint: Optional[str],
    split: str,
    device: str,
    config: Dict,
    generation_config: Dict,
    timing_mode: str,
    object_tracking_included: bool,
    flash_attention_status: Optional[Dict[str, bool]] = None,
    quantization_policy: Optional[Dict[str, object]] = None,
    runtime_diagnostics: Optional[Dict[str, object]] = None,
    generation_validity: Optional[Dict[str, object]] = None,
) -> Dict:
    trajectory_cfg = config.get("trajectory", {})
    model_cfg = config.get("model", {})
    flash_attention_status = flash_attention_status or {
        "flash_attention_requested": flash_attention_requested(config),
        "flash_attention_available": False,
        "flash_attention_active": False,
        "flash_attention_layer_count": 0,
        "flash_attention_active_layer_count": 0,
        "flash_attention_inactive_reason": "not_collected",
    }
    return {
        "config_path": config_path,
        "checkpoint": checkpoint or "Base Model",
        "split": split,
        "device": device,
        "model_architecture": model_cfg.get("architecture", "unknown"),
        "qformer_enabled": bool(model_cfg.get("qformer", {}).get("enabled", False)),
        "trajectory_enabled": bool(trajectory_cfg.get("enabled", False)),
        "trajectory_fusion_mode": trajectory_cfg.get("fusion_mode"),
        "object_tracking_included": bool(object_tracking_included),
        **flash_attention_status,
        **(quantization_policy or {}),
        **(runtime_diagnostics or {}),
        **(generation_validity or {}),
        "generation_config": dict(generation_config),
        "timing_mode": timing_mode,
    }


def collect_runtime_diagnostics(torch_module, model) -> Dict[str, object]:
    diagnostics = {
        "torch_version": getattr(torch_module, "__version__", None),
        "torch_cuda_version": getattr(getattr(torch_module, "version", None), "cuda", None),
        "cuda_device_name": None,
        "model_num_image_token": getattr(model, "num_image_token", None),
    }
    try:
        if torch_module.cuda.is_available():
            diagnostics["cuda_device_name"] = torch_module.cuda.get_device_name(0)
    except Exception:
        diagnostics["cuda_device_name"] = None
    return diagnostics


def count_image_patches(sample: Dict) -> Optional[int]:
    pixel_values = sample.get("pixel_values")
    if pixel_values is None:
        return None
    try:
        return sum(int(getattr(pixel_value, "shape", [1])[0]) for pixel_value in pixel_values)
    except Exception:
        return None


def count_tokens(tokenizer, text: str) -> Optional[int]:
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return None


def cuda_synchronize_if_available():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


class PhaseTimer:
    def __enter__(self):
        cuda_synchronize_if_available()
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        cuda_synchronize_if_available()
        self.elapsed_ms = (time.perf_counter() - self.start) * 1000.0


def reset_model_latency_phases(model):
    if model is not None:
        model._latency_phase_ms = {"vision_ms": 0.0, "trajectory_ms": 0.0}


def reset_model_runtime_call_counts(model):
    if model is not None:
        model._latency_runtime_call_counts = {
            field: 0 for field in RUNTIME_CALL_COUNT_FIELDS.values()
        }


def get_model_runtime_call_counts(model) -> Dict[str, int]:
    counts = getattr(model, "_latency_runtime_call_counts", {}) if model is not None else {}
    return {
        field: int(counts.get(field, 0))
        for field in RUNTIME_CALL_COUNT_FIELDS.values()
    }


def install_runtime_call_counter_hooks(model):
    if model is None:
        return

    hooked_methods = getattr(model, "_latency_call_counter_hooked_methods", set())
    for method_name, counter_field in RUNTIME_CALL_COUNT_FIELDS.items():
        if method_name in hooked_methods:
            continue
        original_method = getattr(model, method_name, None)
        if not callable(original_method):
            continue

        def counted_method(*args, _original_method=original_method, _counter_field=counter_field, **kwargs):
            counts = getattr(model, "_latency_runtime_call_counts", None)
            if counts is not None:
                counts[_counter_field] = int(counts.get(_counter_field, 0)) + 1
            return _original_method(*args, **kwargs)

        setattr(model, method_name, counted_method)
        hooked_methods.add(method_name)

    model._latency_call_counter_hooked_methods = hooked_methods


def get_model_latency_phase(model, phase: str) -> float:
    return float(getattr(model, "_latency_phase_ms", {}).get(phase, 0.0))


def install_extract_feature_latency_hook(model):
    if model is None or getattr(model, "_latency_extract_feature_hooked", False):
        return
    if getattr(model, "_latency_extract_feature_handles_internal_breakdown", False):
        return
    original_extract_feature = getattr(model, "extract_feature", None)
    if original_extract_feature is None:
        return

    def timed_extract_feature(*args, **kwargs):
        with PhaseTimer() as timer:
            result = original_extract_feature(*args, **kwargs)
        phase_ms = getattr(model, "_latency_phase_ms", None)
        if phase_ms is not None:
            phase_ms["vision_ms"] = float(phase_ms.get("vision_ms", 0.0)) + timer.elapsed_ms
        return result

    model.extract_feature = timed_extract_feature
    model._latency_extract_feature_hooked = True


class TokenTimingStreamer:
    def __init__(self, clock=time.perf_counter, sync_fn=cuda_synchronize_if_available):
        self.clock = clock
        self.sync_fn = sync_fn
        self.start_time = None
        self.token_timestamps = []

    def start(self):
        self.sync_fn()
        self.start_time = self.clock()

    def put(self, value):
        self.sync_fn()
        self.token_timestamps.append(self.clock())

    def end(self):
        pass

    @property
    def first_token_ms(self) -> float:
        if self.start_time is None or not self.token_timestamps:
            return 0.0
        return (self.token_timestamps[0] - self.start_time) * 1000.0

    @property
    def decode_after_first_token_ms(self) -> float:
        if len(self.token_timestamps) <= 1:
            return 0.0
        return (self.token_timestamps[-1] - self.token_timestamps[0]) * 1000.0


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark model-side latency for VLM variants.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="test_alter", choices=["test_QA", "test_alter", "val"])
    parser.add_argument("--output_file", default="results/latency_benchmark.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--warmup_samples", type=int, default=5)
    parser.add_argument("--generation_mode", default="latency_greedy", choices=["latency_greedy", "restore_eval"])
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--quantization_mode", default="auto", choices=["auto", "config", "on", "off"])
    parser.add_argument("--object_tracking_mode", default="disabled", choices=["disabled"])
    parser.add_argument("--disable_progress", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    generation_config, generation_validity = build_generation_config(
        generation_mode=args.generation_mode,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        do_sample=bool(args.do_sample),
    )

    # Full runtime benchmarking is intentionally imported lazily so utility tests stay lightweight.
    import yaml
    import torch

    from scripts import test_infer
    from trajectory_branch import load_trajectory_branch

    args.checkpoint = test_infer.resolve_checkpoint_path(args.checkpoint)
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.object_tracking_mode != "disabled":
        raise ValueError("V1 only supports --object_tracking_mode disabled.")

    quantization_policy = resolve_quantization_policy(
        config,
        mode=args.quantization_mode,
        torch_cuda_version=torch.version.cuda,
    )
    response_format = test_infer.get_response_format(config)
    architecture = config["model"]["architecture"]
    backend = test_infer.get_backend(architecture) if architecture == "sailvl" else None
    runtime_config = dict(config)
    runtime_config["model"] = dict(config["model"])
    runtime_config["model"]["quantization"] = dict(config["model"]["quantization"])
    runtime_config["model"]["quantization"]["enabled"] = bool(quantization_policy["quantization_effective"])
    quantization_config = None
    if quantization_policy["quantization_effective"]:
        quantization_config = test_infer.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16
            if runtime_config["model"]["quantization"]["compute_dtype"] == "bfloat16"
            else torch.float16,
            bnb_4bit_use_double_quant=runtime_config["model"]["quantization"]["double_quant"],
            bnb_4bit_quant_type=runtime_config["model"]["quantization"]["type"],
        )

    if backend is not None and backend.name == "sailvl":
        model, tokenizer = backend.load_model_and_tokenizer(runtime_config, args.checkpoint)
        backend.attach_qformer_if_enabled(model, runtime_config)
        if args.checkpoint:
            backend.load_backend_artifacts(model, args.checkpoint, runtime_config)
    else:
        model_config = test_infer.AutoConfig.from_pretrained(
            runtime_config["model"]["name"],
            trust_remote_code=runtime_config["model"]["trust_remote_code"],
        )
        enable_flash_attention_for_config(model_config, flash_attention_requested(runtime_config))
        model = test_infer.AutoModel.from_pretrained(
            runtime_config["model"]["name"],
            config=model_config,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            device_map={"": 0},
            low_cpu_mem_usage=True,
            trust_remote_code=runtime_config["model"]["trust_remote_code"],
        )
        tokenizer = test_infer.AutoTokenizer.from_pretrained(
            runtime_config["model"]["name"],
            trust_remote_code=True,
            use_fast=False,
        )
        model.img_context_token_id = tokenizer.convert_tokens_to_ids(test_infer.IMG_CONTEXT_TOKEN)
        model.system_message = test_infer.SYSTEM_MESSAGE
        if test_infer.qformer_enabled(runtime_config):
            test_infer.attach_qformer_bridge(model, runtime_config)

    if args.checkpoint:
        if backend is None and test_infer.qformer_enabled(runtime_config):
            test_infer.load_qformer_bridge(model, args.checkpoint, strict=True)
        if getattr(model, "trajectory_enabled", False):
            load_trajectory_branch(model, args.checkpoint, strict=True)
        model.language_model = test_infer.PeftModel.from_pretrained(
            model.language_model,
            args.checkpoint,
            is_trainable=False,
            device_map={"": 0},
        )

    model.eval()
    if hasattr(model, "language_model"):
        model.language_model.eval()
    install_extract_feature_latency_hook(model)
    install_runtime_call_counter_hooks(model)
    flash_attention_status = collect_flash_attention_status(model, flash_attention_requested(runtime_config))

    frame_index, bbox_by_folder, trajectory_source = test_infer.prepare_auxiliary_data(runtime_config)
    data_file = "test_alter.json" if args.split == "test_alter" else "test_QA.json"
    dataset_dict = test_infer.load_dataset(runtime_config["data"]["name"], data_files={"test": data_file})
    test_dataset = test_infer.WADDatasetForInternVL(
        metadata_dataset=dataset_dict,
        frame_index=frame_index,
        bbox_by_folder=bbox_by_folder,
        trajectory_source=trajectory_source,
        split="test",
        response_format=response_format,
        direct_text_alter_prompt_mode=runtime_config["data"].get("direct_text_alter_prompt_mode", "fixed_legacy"),
        direct_text_qa_prompt_mode=runtime_config["data"].get("direct_text_qa_prompt_mode", "current_v1"),
        non_train_error_policy=runtime_config["data"].get("non_train_error_policy", "skip"),
        seed=runtime_config["data"].get("seed", 42),
    )

    samples = []
    sample_limit = len(test_dataset) if args.limit is None else min(args.limit, len(test_dataset))
    for idx in iter_latency_indices(sample_limit, progress=not args.disable_progress):
        sample = test_dataset[idx]
        if sample is None:
            continue
        question = str(sample["question"])
        num_image_patches = count_image_patches(sample)
        model_num_image_token = getattr(model, "num_image_token", None)
        question_token_count = count_tokens(tokenizer, question)
        object_tracking = disabled_object_tracking_timing()
        end_to_end_start = time.perf_counter()
        reset_model_latency_phases(model)
        reset_model_runtime_call_counts(model)
        token_streamer = TokenTimingStreamer()
        timed_generation_config = dict(generation_config)
        should_time_tokens = bool(generation_validity["decode_only_tokens_per_s_valid"])
        if should_time_tokens:
            timed_generation_config["streamer"] = token_streamer
        if backend is not None and backend.name == "sailvl":
            with PhaseTimer() as llm_timer:
                if should_time_tokens:
                    token_streamer.start()
                prediction = backend.generate_response(model, tokenizer, sample, timed_generation_config, runtime_config)
            vision_ms = get_model_latency_phase(model, "vision_ms")
            trajectory_ms = get_model_latency_phase(model, "trajectory_ms")
        else:
            with PhaseTimer() as vision_timer:
                pixel_values = torch.cat([torch.as_tensor(p) for p in sample["pixel_values"]], dim=0).to(torch.bfloat16).cuda()
                if getattr(model, "qformer_enabled", False):
                    qformer_text = sample.get("qformer_text", question.replace("<image>", "").strip())
                    q_ids, q_mask = model.encode_qformer_texts([qformer_text] * pixel_values.shape[0], device=pixel_values.device)
                    model.set_qformer_text(q_ids, q_mask)
            if getattr(model, "trajectory_enabled", False):
                model.set_trajectory_inputs(
                    sample["trajectory_label_ids"].unsqueeze(0).repeat(pixel_values.shape[0], 1).cuda(),
                    sample["trajectory_direction_ids"].unsqueeze(0).repeat(pixel_values.shape[0], 1).cuda(),
                    sample["trajectory_numeric_feats"].unsqueeze(0).repeat(pixel_values.shape[0], 1, 1).cuda(),
                    sample["trajectory_object_mask"].unsqueeze(0).repeat(pixel_values.shape[0], 1).cuda(),
                )
            with PhaseTimer() as llm_timer:
                if should_time_tokens:
                    token_streamer.start()
                prediction = test_infer.run_model_chat(
                    model,
                    tokenizer,
                    pixel_values,
                    str(sample["question"]),
                    timed_generation_config,
                )
            if getattr(model, "qformer_enabled", False):
                model.clear_qformer_text()
            if getattr(model, "trajectory_enabled", False):
                model.clear_trajectory_inputs()
            vision_ms = vision_timer.elapsed_ms + get_model_latency_phase(model, "vision_ms")
            trajectory_ms = get_model_latency_phase(model, "trajectory_ms")

        end_to_end_ms = (time.perf_counter() - end_to_end_start) * 1000.0
        generated_token_count = len(tokenizer.encode(prediction, add_special_tokens=False))
        first_token_ms = token_streamer.first_token_ms
        decode_after_first_token_ms = token_streamer.decode_after_first_token_ms
        runtime_call_counts = get_model_runtime_call_counts(model)

        if idx >= args.warmup_samples:
            samples.append(
                build_sample_record(
                    sample_id=idx,
                    question=str(sample["question"]),
                    prediction=prediction,
                    generated_token_count=generated_token_count,
                    end_to_end_ms=end_to_end_ms,
                    vision_ms=vision_ms,
                    trajectory_ms=trajectory_ms,
                    llm_ms=llm_timer.elapsed_ms,
                    first_token_ms=first_token_ms,
                    decode_after_first_token_ms=decode_after_first_token_ms,
                    object_tracking_included=object_tracking["object_tracking_included"],
                    object_tracking_ms=object_tracking["object_tracking_ms"],
                    question_token_count=question_token_count,
                    num_image_patches=num_image_patches,
                    model_num_image_token=model_num_image_token,
                    runtime_call_counts=runtime_call_counts,
                )
            )

    output = {
        "run_metadata": build_run_metadata(
            config_path=args.config,
            checkpoint=args.checkpoint,
            split=args.split,
            device=str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
            config=config,
            generation_config=generation_config,
            timing_mode=TIMING_MODE,
            object_tracking_included=False,
            flash_attention_status=flash_attention_status,
            quantization_policy=quantization_policy,
            runtime_diagnostics=collect_runtime_diagnostics(torch, model),
            generation_validity=generation_validity,
        ),
        "samples": samples,
        "summary": summarize_latency_samples(samples),
    }
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
