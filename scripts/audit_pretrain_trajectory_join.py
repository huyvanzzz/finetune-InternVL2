import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trajectory_branch import TrajectorySource


def load_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def audit_split(split_path: Path, source: TrajectorySource):
    rows = load_rows(split_path)
    object_count_distribution = Counter()
    missing_examples = []
    present_examples = []
    match_count = 0
    missing_count = 0

    for row in rows:
        frame_path = str(row["frame_path"])
        frame_id = int(row["frame_id"])
        key = (frame_path, frame_id)
        if source.has_record(frame_path, frame_id):
            match_count += 1
            object_count = len(source.lookup[key])
            object_count_distribution[object_count] += 1
            if len(present_examples) < 5:
                present_examples.append(
                    {
                        "frame_path": frame_path,
                        "frame_id": frame_id,
                        "question_id": row.get("question_id"),
                        "object_count": object_count,
                    }
                )
        else:
            missing_count += 1
            if len(missing_examples) < 10:
                missing_examples.append(
                    {
                        "frame_path": frame_path,
                        "frame_id": frame_id,
                        "question_id": row.get("question_id"),
                    }
                )

    return {
        "row_count": len(rows),
        "match_count": match_count,
        "missing_count": missing_count,
        "object_count_distribution": dict(sorted(object_count_distribution.items())),
        "present_examples": present_examples,
        "missing_examples": missing_examples,
    }


def main():
    parser = argparse.ArgumentParser(description="Audit pretrain QA splits against trajectory canonical records.")
    parser.add_argument("--trajectory-source", type=str, default="json", help="Trajectory source file or directory.")
    parser.add_argument(
        "--split-files",
        nargs="+",
        default=[
            "json/question_train_split_train.jsonl",
            "json/question_train_split_val.jsonl",
            "json/question_train_split_test.jsonl",
        ],
        help="Question split JSONL files to audit.",
    )
    args = parser.parse_args()

    source = TrajectorySource.from_file(args.trajectory_source)
    report = {}
    for split_file in args.split_files:
        split_path = Path(split_file)
        report[split_path.name] = audit_split(split_path, source)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
