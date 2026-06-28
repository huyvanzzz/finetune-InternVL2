import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from resume_equivalence import (
    build_verdict,
    make_dataset,
    run_epoch_from_checkpoint,
    run_reference_epoch,
    run_resumed_epoch,
)


def run_cpu_toy(steps: int, resume_step: int, use_rng_state: bool):
    dataset = make_dataset(steps)
    baseline = run_reference_epoch(dataset, epoch_seed=42, dropout_p=0.5, checkpoint_after_steps=resume_step)
    resumed = run_resumed_epoch(dataset, baseline["checkpoint"], restore_rng_state=use_rng_state)
    comparison = build_verdict(baseline["records"][resume_step:], resumed["records"])
    return {
        "mode": "cpu_toy",
        "steps": steps,
        "resume_step": resume_step,
        "use_rng_state": use_rng_state,
        "comparison": comparison,
        "baseline_sample_ids": [record["sample_id"] for record in baseline["records"][resume_step:]],
        "resumed_sample_ids": resumed["sample_order"],
    }


def run_train_like(steps: int, resume_step: int, use_rng_state: bool):
    dataset = make_dataset(steps)
    epoch_zero = run_reference_epoch(dataset, epoch_seed=42, dropout_p=0.5, checkpoint_after_steps=resume_step)
    resumed = run_resumed_epoch(dataset, epoch_zero["checkpoint"], restore_rng_state=use_rng_state)
    comparison = build_verdict(epoch_zero["records"][resume_step:], resumed["records"])
    return {
        "mode": "train_like",
        "steps": steps,
        "resume_step": resume_step,
        "use_rng_state": use_rng_state,
        "comparison": comparison,
        "baseline_sample_ids": [record["sample_id"] for record in epoch_zero["records"][resume_step:]],
        "resumed_sample_ids": resumed["sample_order"],
    }


def main():
    parser = argparse.ArgumentParser(description="Verify resume equivalence behavior on CPU toy harness.")
    parser.add_argument("--mode", choices=["cpu_toy", "train_like"], default="cpu_toy")
    parser.add_argument("--checkpoint_dir", default=None, help="Unused for toy modes; reserved for future extensions.")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--resume_step", type=int, default=6)
    parser.add_argument("--use_rng_state", action="store_true")
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    if args.resume_step <= 0 or args.resume_step >= args.steps:
        raise SystemExit("--resume_step must be between 1 and steps-1.")

    if args.mode == "cpu_toy":
        result = run_cpu_toy(args.steps, args.resume_step, args.use_rng_state)
    else:
        result = run_train_like(args.steps, args.resume_step, args.use_rng_state)

    comparison = result["comparison"]
    print(f"Mode: {result['mode']}")
    print(f"Resume step: {result['resume_step']} / {result['steps']}")
    print(f"Use RNG state: {result['use_rng_state']}")
    print(f"Sample order equal: {comparison['sample_order_equal']}")
    print(f"Losses equal: {comparison['losses_equal']}")
    print(f"Weight checksums equal: {comparison['weight_checksums_equal']}")
    print(f"Optimizer/scheduler state equal: {comparison['scheduler_state_equal']}")
    print(f"Verdict: {comparison['verdict']}")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
