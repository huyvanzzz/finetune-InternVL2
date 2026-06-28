import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))


def _latest_run_dir(output_root: Path) -> Path:
    runs = sorted([p for p in output_root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f"No run directories found under {output_root}")
    return runs[-1]


def _latest_adapter_dir(run_dir: Path) -> Path:
    candidates = sorted(run_dir.glob("*"), key=lambda p: p.stat().st_mtime)
    candidates = [p for p in candidates if p.is_dir() and (p / "adapter_config.json").exists()]
    if not candidates:
        raise FileNotFoundError(f"No adapter checkpoint directories found under {run_dir}")
    return candidates[-1]


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_adapter_weights(checkpoint_dir: Path) -> Path:
    for filename in ("adapter_model.safetensors", "adapter_model.bin"):
        candidate = checkpoint_dir / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No adapter weights found in {checkpoint_dir}")


def _collect_log_markers(run_dir: Path):
    log_files = sorted(run_dir.glob("output.*.log.txt"), key=lambda p: p.stat().st_mtime)
    sample_lines = []
    rng_lines = []
    for log_file in log_files:
        for line in log_file.read_text(encoding="utf-8", errors="replace").splitlines():
            if "[DEBUG_SAMPLE_IDS]" in line:
                sample_lines.append(line)
            if "[DEBUG_RNG]" in line:
                rng_lines.append(line)
    return {
        "sample_id_logs": sample_lines,
        "rng_logs": rng_lines,
    }


def _write_config(base_config_path: Path, output_root: Path, config_path: Path):
    with open(base_config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["training"]["output_dir"] = str(output_root).replace("\\", "/")
    config["training"]["num_epochs"] = 1
    config["training"]["batch_size"] = 1
    config["training"]["gradient_accumulation_steps"] = 1
    config["training"]["eval_steps"] = None
    config["training"]["save_steps"] = 4
    config["training"]["log_token_stats"] = False
    config["training"]["token_log_batches"] = 0
    config["training"]["debug_log_sample_ids"] = True
    config["training"]["debug_log_rng_digest"] = True
    config["data"]["train_limit"] = 8
    config["data"]["eval_limit"] = 0

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=False)


def _run_train(config_path: Path, checkpoint: str | None = None, start_epoch: int | None = None, start_step: int | None = None):
    cmd = ["python", "train.py", "--config", str(config_path)]
    if checkpoint:
        cmd += ["--checkpoint", checkpoint]
    if start_epoch is not None:
        cmd += ["--start_epoch", str(start_epoch)]
    if start_step is not None:
        cmd += ["--start_step", str(start_step)]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Smoke-check real train.py resume behavior on a tiny run.")
    parser.add_argument("--config", default="internvl_config.yaml")
    parser.add_argument("--output_json", default="results/resume_smoke.json")
    args = parser.parse_args()

    workspace = Path(tempfile.mkdtemp(prefix="resume_smoke_", dir="."))
    baseline_root = workspace / "baseline_outputs"
    resumed_root = workspace / "resumed_outputs"
    baseline_config = workspace / "baseline_config.yaml"
    resumed_config = workspace / "resumed_config.yaml"

    try:
        _write_config(Path(args.config), baseline_root, baseline_config)
        _write_config(Path(args.config), resumed_root, resumed_config)

        _run_train(baseline_config)
        baseline_run_dir = _latest_run_dir(baseline_root)
        step_checkpoint = baseline_run_dir / "epoch_1_step_4"
        final_checkpoint = baseline_run_dir / "epoch_1"
        if not step_checkpoint.exists():
            raise FileNotFoundError(f"Expected step checkpoint at {step_checkpoint}")

        _run_train(resumed_config, checkpoint=str(step_checkpoint), start_epoch=0, start_step=4)
        resumed_run_dir = _latest_run_dir(resumed_root)
        resumed_final_checkpoint = resumed_run_dir / "epoch_1"

        baseline_adapter_hash = _hash_file(_find_adapter_weights(final_checkpoint))
        resumed_adapter_hash = _hash_file(_find_adapter_weights(resumed_final_checkpoint))

        report = {
            "baseline_run_dir": str(baseline_run_dir),
            "resumed_run_dir": str(resumed_run_dir),
            "step_checkpoint_dir": str(step_checkpoint),
            "baseline_final_hash": baseline_adapter_hash,
            "resumed_final_hash": resumed_adapter_hash,
            "final_hash_equal": baseline_adapter_hash == resumed_adapter_hash,
            "baseline_logs": _collect_log_markers(baseline_run_dir),
            "resumed_logs": _collect_log_markers(resumed_run_dir),
        }

        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print(json.dumps(report, indent=2))
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    main()
