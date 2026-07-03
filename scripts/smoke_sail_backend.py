import argparse
import os
import sys

import torch
import yaml
from huggingface_hub import snapshot_download
from PIL import Image

sys.path.append(".")

from model_backends import get_backend
from model_backends.sailvl.preprocess import preprocess_sail_image


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke test SAIL-VL backend with one synthetic sample.")
    parser.add_argument("--config", default="sailvl_config.yaml")
    parser.add_argument("--checkpoint", default=None, help="Local checkpoint dir or Hugging Face repo id")
    return parser.parse_args()


def resolve_checkpoint_path(checkpoint):
    if not checkpoint:
        return None
    if os.path.exists(checkpoint):
        return checkpoint
    return snapshot_download(
        repo_id=checkpoint,
        allow_patterns=[
            "adapter_config.json",
            "adapter_model.safetensors",
            "adapter_model.bin",
            "qformer_bridge.safetensors",
            "qformer_bridge_config.json",
            "tokenizer*",
            "special_tokens_map.json",
            "added_tokens.json",
        ],
    )


def build_smoke_sample():
    prompt = (
        "Provide brief navigation guidance for a visually impaired user based on this image.\n"
        "Focus on immediate obstacles, the safe direction, and the next safe action.\n"
        "Output only the final guidance in natural language."
    )
    return {
        "question": f"<image>\n{prompt}",
        "answer": "move forward carefully",
        "qformer_text": prompt,
        "task_type": "alter",
        "selected_prompt_id": "T1",
        "selected_prompt_text": prompt,
        "frame_path": "synthetic/smoke_frame.jpg",
        "image": [Image.new("RGB", (448, 448), color=(160, 160, 160))],
    }


def main():
    args = parse_args()
    checkpoint_dir = resolve_checkpoint_path(args.checkpoint)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    architecture = config["model"]["architecture"]
    if architecture != "sailvl":
        raise SystemExit("SAIL smoke test only supports sailvl configs.")

    backend = get_backend(architecture)
    model, tokenizer = backend.load_model_and_tokenizer(config, checkpoint_dir)
    backend.attach_qformer_if_enabled(model, config)
    if checkpoint_dir:
        backend.load_backend_artifacts(model, checkpoint_dir, config)
    model.eval()

    sample = build_smoke_sample()
    pixel_values = preprocess_sail_image(sample["image"][0], config)
    num_tiles = len(pixel_values)
    total_image_tokens = num_tiles * model.num_image_token

    generation_config = dict(
        max_new_tokens=64,
        do_sample=False,
        temperature=0.0,
    )
    response = backend.generate_response(model, tokenizer, sample, generation_config, config)

    print("SAIL backend smoke test passed.")
    print(f"  config: {args.config}")
    print(f"  checkpoint: {checkpoint_dir or 'none'}")
    print(f"  qformer enabled: {getattr(model, 'qformer_enabled', False)}")
    print(f"  tiles: {num_tiles}")
    print(f"  image tokens: {total_image_tokens}")
    print("  question:")
    print(sample["question"])
    print("  qformer_text:")
    print(sample["qformer_text"])
    print("  response:")
    print(response)


if __name__ == "__main__":
    main()
