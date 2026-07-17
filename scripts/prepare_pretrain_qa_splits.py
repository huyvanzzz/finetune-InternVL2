import argparse
import json
import random
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for key in ("frame_path", "frame_id", "question", "gt"):
                if key not in row:
                    raise ValueError(f"{path}:{line_no} missing required key: {key}")
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stats(rows):
    frame_paths = {str(row["frame_path"]) for row in rows}
    question_ids = Counter(str(row.get("question_id", "")) for row in rows)
    return {
        "row_count": len(rows),
        "frame_count": len(frame_paths),
        "question_id_counts": dict(sorted(question_ids.items())),
    }


def split_rows(rows, seed: int):
    grouped = {}
    for row in rows:
        grouped.setdefault(str(row["frame_path"]), []).append(row)

    frame_paths = sorted(grouped)
    rng = random.Random(seed)
    rng.shuffle(frame_paths)

    total = len(frame_paths)
    test_count = max(1, round(total * 0.1)) if total >= 3 else 0
    val_count = max(1, round(total * 0.1)) if total >= 3 else 0
    if test_count + val_count >= total:
        val_count = max(0, total - test_count - 1)

    test_paths = set(frame_paths[:test_count])
    val_paths = set(frame_paths[test_count : test_count + val_count])
    train_paths = set(frame_paths[test_count + val_count :])
    return {
        "train": [row for row in rows if str(row["frame_path"]) in train_paths],
        "val": [row for row in rows if str(row["frame_path"]) in val_paths],
        "test": [row for row in rows if str(row["frame_path"]) in test_paths],
    }


def verify_splits(splits):
    frame_sets = {
        name: {str(row["frame_path"]) for row in rows}
        for name, rows in splits.items()
    }
    names = list(frame_sets)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            overlap = frame_sets[left] & frame_sets[right]
            if overlap:
                raise ValueError(f"frame_path leakage between {left} and {right}: {sorted(overlap)[:5]}")
    if not all(splits.values()):
        empty = [name for name, rows in splits.items() if not rows]
        raise ValueError(f"Empty split(s): {empty}")


def main():
    parser = argparse.ArgumentParser(description="Create fixed 80/10/10 pretrain QA splits grouped by frame_path.")
    parser.add_argument("--input", default="json/question_train.jsonl")
    parser.add_argument("--output_dir", default="json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    rows = read_jsonl(input_path)
    splits = split_rows(rows, seed=args.seed)
    verify_splits(splits)

    paths = {
        "train": output_dir / "question_train_split_train.jsonl",
        "val": output_dir / "question_train_split_val.jsonl",
        "test": output_dir / "question_train_split_test.jsonl",
    }
    for name, path in paths.items():
        write_jsonl(path, splits[name])

    payload = {
        "source_file": str(input_path),
        "seed": args.seed,
        "group_key": "frame_path",
        "ratios": {"train": 0.8, "val": 0.1, "test": 0.1},
        "files": {name: str(path) for name, path in paths.items()},
        "stats": {name: stats(rows) for name, rows in splits.items()},
    }
    manifest_path = output_dir / "question_train_split_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload["stats"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
