import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trajectory_priority import build_grouped_records, load_jsonl, write_json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build top-k trajectory object JSON from BoTSORT detections."
    )
    parser.add_argument(
        "--input",
        default="image/results_botsort.jsonl",
        help="Input JSONL file containing per-object detections.",
    )
    parser.add_argument(
        "--output",
        default="image/results_botsort_top6.json",
        help="Output JSON file containing grouped top objects.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=6,
        help="Maximum number of objects to keep for each sample.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    rows = load_jsonl(input_path)
    records = build_grouped_records(rows, limit=args.limit)
    write_json(output_path, records)

    print(
        f"Built {len(records)} grouped samples from {len(rows)} rows -> {output_path}"
    )


if __name__ == "__main__":
    main()
