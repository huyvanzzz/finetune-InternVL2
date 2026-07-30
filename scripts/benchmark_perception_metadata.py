import argparse
import json
from pathlib import Path
from statistics import mean

from online_perception import OnlineFastPerceptionEngine, YoloDetectorAdapter, load_image


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark online object detection + tracking until top-6 trajectory metadata.")
    parser.add_argument("--img_root", default=None, help="Directory containing sequence folders of .jpg frames.")
    parser.add_argument("--config", default=None, help="Benchmark config; when provided, load frames through WADDataset/HF cache.")
    parser.add_argument("--split", default="test_alter", choices=["test_QA", "test_alter", "val"])
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


def iter_with_progress(iterable, desc: str, unit: str):
    try:
        from tqdm import tqdm
    except Exception:
        return iterable
    return tqdm(iterable, desc=desc, unit=unit)


def iter_local_sequence_records(args, engine):
    img_root = Path(args.img_root)
    sequence_dirs = sorted([path for path in img_root.iterdir() if path.is_dir()])
    start = args.start_idx if args.start_idx is not None else 0
    end = args.end_idx if args.end_idx is not None else len(sequence_dirs)
    sequence_dirs = sequence_dirs[start:end]
    if args.limit is not None:
        sequence_dirs = sequence_dirs[: args.limit]

    records = []
    for sequence_dir in iter_with_progress(sequence_dirs, desc="Benchmarking perception", unit="seq"):
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
    return records, {"data_source": "local_folder", "img_root": str(img_root), "start_idx": start, "end_idx": end}


def iter_config_dataset_records(args, engine):
    import yaml

    from scripts import test_infer

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    response_format = test_infer.get_response_format(config)
    frame_index, bbox_by_folder, trajectory_source = test_infer.prepare_auxiliary_data(config)
    data_file = "test_alter.json" if args.split == "test_alter" else "test_QA.json"
    dataset_dict = test_infer.load_dataset(config["data"]["name"], data_files={"test": data_file})
    dataset = test_infer.WADDatasetForInternVL(
        metadata_dataset=dataset_dict,
        frame_index=frame_index,
        bbox_by_folder=bbox_by_folder,
        trajectory_source=trajectory_source,
        split="test",
        response_format=response_format,
        direct_text_alter_prompt_mode=config["data"].get("direct_text_alter_prompt_mode", "fixed_legacy"),
        direct_text_qa_prompt_mode=config["data"].get("direct_text_qa_prompt_mode", "current_v1"),
        non_train_error_policy=config["data"].get("non_train_error_policy", "skip"),
        seed=config["data"].get("seed", 42),
    )

    sample_count = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    records = []
    for idx in iter_with_progress(range(sample_count), desc="Benchmarking perception", unit="sample"):
        sample_meta = dataset.metadata[idx]
        frame_path = sample_meta["frame_path"]
        frame_ids = dataset._select_frames_safe(frame_path)
        engine.reset_sequence(frame_path)
        frames = dataset._load_frames(frame_path, frame_ids)
        for frame_index, (frame_id, frame) in enumerate(zip(frame_ids, frames)):
            record = engine.update(frame=frame, frame_id=int(frame_id), folder_id=str(frame_path))
            record["sample_id"] = idx
            if not args.save_only_last_frame or frame_index == len(frame_ids) - 1:
                records.append(record)
    return records, {
        "data_source": "config_dataset",
        "config": args.config,
        "split": args.split,
        "dataset_name": config["data"]["name"],
    }


def main():
    args = parse_args()
    if args.config is None and args.img_root is None:
        raise ValueError("Provide either --config for WAD/HF dataset mode or --img_root for local folder mode.")

    output = {
        "run_metadata": {
            "mode": args.mode,
            "yolo_model_path": args.yolo_model_path,
            "save_only_last_frame": bool(args.save_only_last_frame),
            "limit": args.limit,
        },
    }
    engine = OnlineFastPerceptionEngine(detector=YoloDetectorAdapter(args.yolo_model_path))
    if args.config is not None:
        records, source_metadata = iter_config_dataset_records(args, engine)
    else:
        records, source_metadata = iter_local_sequence_records(args, engine)
    output["run_metadata"].update(source_metadata)
    output["records"] = records
    output["summary"] = summarize(records)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved perception metadata latency to: {output_path}")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
