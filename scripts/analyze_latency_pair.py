import argparse
import json
import math
from statistics import mean, median
from typing import Dict, Iterable, List


TIMING_FIELDS = (
    "end_to_end_ms",
    "e2e_at_20_tokens_ms",
    "image_preprocess_ms",
    "vision_forward_ms",
    "qformer_text_encode_ms",
    "qformer_forward_ms",
    "qformer_projection_ms",
    "vision_ms",
    "trajectory_ms",
    "llm_ms",
    "first_token_ms",
    "ttft_ms",
    "decode_after_first_token_ms",
    "decode_only_tokens_per_s",
)
RUNTIME_CALL_COUNT_FIELDS = (
    "extract_feature_call_count",
    "qformer_encode_call_count",
    "set_qformer_text_call_count",
    "clear_qformer_text_call_count",
)


def percentile(values: List[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = max(0, min(len(ordered) - 1, math.ceil((percentile_value / 100.0) * len(ordered)) - 1))
    return ordered[index]


def summarize_values(values: Iterable[float]) -> Dict[str, float]:
    values = [float(value) for value in values]
    if not values:
        return {"mean": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0}
    return {
        "mean": round(mean(values), 6),
        "median": round(median(values), 6),
        "p10": round(percentile(values, 10), 6),
        "p90": round(percentile(values, 90), 6),
    }


def sample_by_id(run: Dict) -> Dict[int, Dict]:
    return {int(sample["id"]): sample for sample in run.get("samples", [])}


def build_paired_latency_rows(no_qformer_run: Dict, qformer_run: Dict) -> List[Dict]:
    no_qformer_samples = sample_by_id(no_qformer_run)
    qformer_samples = sample_by_id(qformer_run)
    rows = []
    for sample_id in sorted(set(no_qformer_samples) & set(qformer_samples)):
        no_qformer_sample = no_qformer_samples[sample_id]
        qformer_sample = qformer_samples[sample_id]
        timing_delta = {
            field: float(qformer_sample.get("timing", {}).get(field, 0.0))
            - float(no_qformer_sample.get("timing", {}).get(field, 0.0))
            for field in TIMING_FIELDS
        }
        runtime_call_count_delta = {
            field: int(qformer_sample.get(field, 0)) - int(no_qformer_sample.get(field, 0))
            for field in RUNTIME_CALL_COUNT_FIELDS
        }
        rows.append(
            {
                "id": sample_id,
                "no_qformer_generated_token_count": int(no_qformer_sample.get("generated_token_count", 0)),
                "qformer_generated_token_count": int(qformer_sample.get("generated_token_count", 0)),
                "generated_token_delta": int(qformer_sample.get("generated_token_count", 0))
                - int(no_qformer_sample.get("generated_token_count", 0)),
                "timing_delta": timing_delta,
                "runtime_call_count_delta": runtime_call_count_delta,
            }
        )
    return rows


def summarize_rows(rows: List[Dict]) -> Dict:
    return {
        "count": len(rows),
        "qformer_faster_count": sum(row["timing_delta"]["end_to_end_ms"] < 0 for row in rows),
        "generated_token_delta": summarize_values(row["generated_token_delta"] for row in rows),
        **{
            f"{field}_delta": summarize_values(row["timing_delta"].get(field, 0.0) for row in rows)
            for field in TIMING_FIELDS
        },
        **{
            f"{field}_delta": summarize_values(row["runtime_call_count_delta"].get(field, 0) for row in rows)
            for field in RUNTIME_CALL_COUNT_FIELDS
        },
    }


def bucket_paired_latency_rows(rows: List[Dict]) -> Dict[str, Dict]:
    buckets = {
        "qformer_shorter": [],
        "same_generated_token_count": [],
        "qformer_longer_1_to_3": [],
        "qformer_longer_4_plus": [],
    }
    for row in rows:
        token_delta = int(row["generated_token_delta"])
        if token_delta < 0:
            buckets["qformer_shorter"].append(row)
        elif token_delta == 0:
            buckets["same_generated_token_count"].append(row)
        elif token_delta <= 3:
            buckets["qformer_longer_1_to_3"].append(row)
        else:
            buckets["qformer_longer_4_plus"].append(row)
    return {name: summarize_rows(bucket_rows) for name, bucket_rows in buckets.items()}


def build_analysis(no_qformer_run: Dict, qformer_run: Dict) -> Dict:
    paired_rows = build_paired_latency_rows(no_qformer_run, qformer_run)
    return {
        "no_qformer_metadata": no_qformer_run.get("run_metadata", {}),
        "qformer_metadata": qformer_run.get("run_metadata", {}),
        "paired_count": len(paired_rows),
        "overall": summarize_rows(paired_rows),
        "buckets": bucket_paired_latency_rows(paired_rows),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze paired no-QFormer vs QFormer latency JSON files.")
    parser.add_argument("--no_qformer_json", required=True)
    parser.add_argument("--qformer_json", required=True)
    parser.add_argument("--output_file", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.no_qformer_json, "r", encoding="utf-8") as f:
        no_qformer_run = json.load(f)
    with open(args.qformer_json, "r", encoding="utf-8") as f:
        qformer_run = json.load(f)
    analysis = build_analysis(no_qformer_run, qformer_run)
    payload = json.dumps(analysis, ensure_ascii=False, indent=2)
    print(payload)
    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            f.write(payload)
            f.write("\n")


if __name__ == "__main__":
    main()
