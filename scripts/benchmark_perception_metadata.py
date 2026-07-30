import argparse
import json
from pathlib import Path
from statistics import mean

from online_perception import OnlineFastPerceptionEngine, YoloDetectorAdapter, load_image


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark online object detection + tracking until top-6 trajectory metadata.")
    parser.add_argument("--img_root", required=True, help="Directory containing sequence folders of .jpg frames.")
    parser.add_argument("--output_file", default="results/perception_metadata_latency.json")
    parser.add_argument("--yolo_model_path", default="yolo11n.pt")
    parser.add_argument("--mode", default="online_fast", choices=["online_fast"])
    parser.add_argument("--frame_glob", default="*.jpg")
    parser.add_argument("--start_idx", type=int, default=None)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save_only_last_frame", action="store_true")
    return parser.parse_args()


def sorted_frame_paths(sequence_dir: Path, frame_glob: str):
    def frame_key(path: Path):
        try:
            return int(path.stem)
        except ValueError:
            return path.name

    return sorted(sequence_dir.glob(frame_glob), key=frame_key)


def summarize(records):
    timings = [record["timing"] for record in records]

    def mean_field(field):
        return round(mean(float(t.get(field, 0.0)) for t in timings), 6) if timings else 0.0

    return {
        "num_records": len(records),
        "object_detection_ms": {"mean": mean_field("object_detection_ms")},
        "tracking_ms": {"mean": mean_field("tracking_ms")},
        "top6_selection_ms": {"mean": mean_field("top6_selection_ms")},
        "object_tracking_ms": {"mean": mean_field("object_tracking_ms")},
    }


def main():
    args = parse_args()
    img_root = Path(args.img_root)
    sequence_dirs = sorted([path for path in img_root.iterdir() if path.is_dir()])
    start = args.start_idx if args.start_idx is not None else 0
    end = args.end_idx if args.end_idx is not None else len(sequence_dirs)
    sequence_dirs = sequence_dirs[start:end]
    if args.limit is not None:
        sequence_dirs = sequence_dirs[: args.limit]

    engine = OnlineFastPerceptionEngine(detector=YoloDetectorAdapter(args.yolo_model_path))
    records = []
    for sequence_dir in sequence_dirs:
        engine.reset_sequence(sequence_dir.name)
        frame_paths = sorted_frame_paths(sequence_dir, args.frame_glob)
        if not frame_paths:
            continue
        for frame_index, frame_path in enumerate(frame_paths):
            frame = load_image(frame_path)
            try:
                frame_id = int(frame_path.stem)
            except ValueError:
                frame_id = len(records)
            record = engine.update(frame=frame, frame_id=frame_id, folder_id=sequence_dir.name)
            if not args.save_only_last_frame or frame_index == len(frame_paths) - 1:
                records.append(record)

    output = {
        "run_metadata": {
            "mode": args.mode,
            "img_root": str(img_root),
            "yolo_model_path": args.yolo_model_path,
            "save_only_last_frame": bool(args.save_only_last_frame),
            "start_idx": start,
            "end_idx": end,
            "limit": args.limit,
        },
        "records": records,
        "summary": summarize(records),
    }
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved perception metadata latency to: {output_path}")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
