import argparse
import json
from collections import Counter
from pathlib import Path

import yaml
from datasets import load_dataset


def normalize_text(value):
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())


def classify_sample_type(sample):
    qa = sample.get("QA")
    if qa and isinstance(qa, dict) and qa.get("Q"):
        return "qa"
    return "alter"


def sample_field_value(sample, key_name):
    if key_name == "frame_path":
        return sample.get("frame_path", "")
    if key_name == "video":
        return sample.get("video", "")
    if key_name == "summary":
        return normalize_text(sample.get("summary"))
    if key_name == "alter":
        return normalize_text(sample.get("alter"))
    if key_name == "qa_question":
        qa = sample.get("QA") or {}
        return normalize_text(qa.get("Q"))
    if key_name == "qa_answer":
        qa = sample.get("QA") or {}
        return normalize_text(qa.get("A"))
    raise ValueError(f"Unsupported key_name: {key_name}")


def build_value_to_indices(samples, key_name):
    mapping = {}
    for idx, sample in enumerate(samples):
        value = sample_field_value(sample, key_name)
        if not value:
            continue
        mapping.setdefault(value, []).append(idx)
    return mapping


def compute_overlap_report(train_samples, test_samples, key_name):
    train_values = build_value_to_indices(train_samples, key_name)
    test_values = build_value_to_indices(test_samples, key_name)
    overlapping_values = set(train_values).intersection(test_values)

    leaked_test_indices = set()
    for value in overlapping_values:
        leaked_test_indices.update(test_values[value])

    per_type = {}
    type_totals = Counter(classify_sample_type(sample) for sample in test_samples)
    type_leaks = Counter(classify_sample_type(test_samples[idx]) for idx in leaked_test_indices)
    for sample_type in ("qa", "alter"):
        total = type_totals.get(sample_type, 0)
        leaked = type_leaks.get(sample_type, 0)
        per_type[sample_type] = {
            "total": total,
            "leaked": leaked,
            "leak_rate": (leaked / total) if total else 0.0,
        }

    example_values = sorted(overlapping_values)[:10]
    return {
        "key_name": key_name,
        "train_unique_values": len(train_values),
        "test_unique_values": len(test_values),
        "overlap_unique_values": len(overlapping_values),
        "test_total": len(test_samples),
        "overlap_count": len(leaked_test_indices),
        "overlap_rate": (len(leaked_test_indices) / len(test_samples)) if test_samples else 0.0,
        "per_type": per_type,
        "example_overlaps": example_values,
    }


def load_split(dataset_name, filename):
    return list(
        load_dataset(
            dataset_name,
            data_files={"data": filename},
            split="data",
        )
    )


def summarize_split(samples):
    counts = Counter(classify_sample_type(sample) for sample in samples)
    return {
        "total": len(samples),
        "qa": counts.get("qa", 0),
        "alter": counts.get("alter", 0),
    }


def build_full_report(dataset_name):
    train_samples = load_split(dataset_name, "train.json")
    test_qa_samples = load_split(dataset_name, "test_QA.json")
    test_alter_samples = load_split(dataset_name, "test_alter.json")
    combined_test_samples = test_qa_samples + test_alter_samples

    key_names = ["frame_path", "video", "summary", "qa_question", "qa_answer", "alter"]
    split_payloads = {
        "test_QA": test_qa_samples,
        "test_alter": test_alter_samples,
        "test_all": combined_test_samples,
    }

    report = {
        "dataset_name": dataset_name,
        "split_summary": {
            "train": summarize_split(train_samples),
            "test_QA": summarize_split(test_qa_samples),
            "test_alter": summarize_split(test_alter_samples),
            "test_all": summarize_split(combined_test_samples),
        },
        "leak_checks": {},
    }

    for split_name, split_samples in split_payloads.items():
        report["leak_checks"][split_name] = {
            key_name: compute_overlap_report(train_samples, split_samples, key_name)
            for key_name in key_names
        }

    return report


def print_summary(report):
    print("=== DATA LEAK SUMMARY ===")
    print(f"Dataset: {report['dataset_name']}")
    for split_name, stats in report["split_summary"].items():
        print(f"- {split_name}: total={stats['total']} | qa={stats['qa']} | alter={stats['alter']}")

    print("\n=== OVERLAP BY SPLIT ===")
    for split_name, leak_checks in report["leak_checks"].items():
        print(f"\n[{split_name}]")
        for key_name, leak_report in leak_checks.items():
            print(
                f"  - {key_name}: leaked={leak_report['overlap_count']}/{leak_report['test_total']} "
                f"({leak_report['overlap_rate']:.2%}) | "
                f"qa={leak_report['per_type']['qa']['leaked']}/{leak_report['per_type']['qa']['total']} | "
                f"alter={leak_report['per_type']['alter']['leaked']}/{leak_report['per_type']['alter']['total']}"
            )


def main():
    parser = argparse.ArgumentParser(description="Check train/test leakage for WAD splits.")
    parser.add_argument("--config", default="internvl_config.yaml")
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--output_json", default="results/data_leak_report.json")
    args = parser.parse_args()

    dataset_name = args.dataset_name
    if dataset_name is None:
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        dataset_name = config["data"]["name"]

    report = build_full_report(dataset_name)
    print_summary(report)

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nSaved detailed report to: {output_path}")


if __name__ == "__main__":
    main()
