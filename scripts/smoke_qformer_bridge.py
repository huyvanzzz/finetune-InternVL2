import argparse
import sys

import torch
import yaml
from transformers import AutoModel, BitsAndBytesConfig

sys.path.append(".")
from qformer_bridge import attach_qformer_bridge, qformer_enabled, trainable_parameter_summary


def gpu_supports_bf16():
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(0)
    return major >= 8


def resolve_runtime_dtype(config, use_cpu=False):
    if use_cpu:
        return torch.float32
    wants_bf16 = config["model"]["quantization"].get("compute_dtype", "bfloat16") == "bfloat16"
    if wants_bf16 and gpu_supports_bf16():
        return torch.bfloat16
    return torch.float16


def main():
    parser = argparse.ArgumentParser(description="Smoke test InternVL Q-Former bridge.")
    parser.add_argument("--config", default="internvl_config.yaml")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not qformer_enabled(config):
        raise SystemExit("Q-Former is disabled; smoke test skipped.")
    runtime_dtype = resolve_runtime_dtype(config, use_cpu=args.cpu)
    print(f"Runtime dtype selected: {runtime_dtype}")

    quant_cfg = None
    if not args.cpu and config["model"]["quantization"]["enabled"]:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=runtime_dtype,
            bnb_4bit_use_double_quant=config["model"]["quantization"]["double_quant"],
            bnb_4bit_quant_type=config["model"]["quantization"]["type"],
        )

    model = AutoModel.from_pretrained(
        config["model"]["name"],
        torch_dtype=runtime_dtype,
        quantization_config=quant_cfg,
        low_cpu_mem_usage=True,
        trust_remote_code=config["model"]["trust_remote_code"],
    )
    model.runtime_dtype = runtime_dtype
    attach_qformer_bridge(model, config)
    model.vision_model.requires_grad_(False)
    model.eval()

    device = torch.device("cpu" if args.cpu else "cuda")
    if args.cpu:
        model.to(device)

    pixel_values = torch.zeros(
        1,
        3,
        448,
        448,
        dtype=runtime_dtype,
        device=device,
    )
    q_ids, q_mask = model.encode_qformer_texts(
        ["Analyze the scene and give guidance for a visually impaired user."],
        device=device,
    )
    model.set_qformer_text(q_ids, q_mask)
    with torch.no_grad():
        visual_embeds = model.extract_feature(pixel_values)
    model.clear_qformer_text()

    expected_tokens = config["model"]["qformer"]["num_query_tokens"]
    assert visual_embeds.shape[1] == expected_tokens, visual_embeds.shape
    assert visual_embeds.shape[-1] == model.config.llm_config.hidden_size, visual_embeds.shape

    print("Smoke test passed.")
    print(f"  visual_embeds: {tuple(visual_embeds.shape)}")
    print(f"  num_image_token: {model.num_image_token}")
    print("Trainable parameters:")
    for row in trainable_parameter_summary(model):
        print(f"  {row}")


if __name__ == "__main__":
    main()
