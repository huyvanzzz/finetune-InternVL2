# Huong Dan Chay `concat` Best-shot BF16 2 GPU

Nhanh nay dung cho:
- branch: `feature/trajectory-pretrain-qformer-concat-bestshot-bf16`
- config: `internvl_config_traj_concat_bestshot_bf16_2gpu.yaml`
- mode: `concat`
- data: `alter_only`
- trajectory: `384 / 4 / 768 / dropout 0.10`
- quantization: `tat 4bit`
- dtype: `bf16`
- LoRA: `r=16`

## 1. Vao server va checkout nhanh

```bash
cd /workspace
git clone https://github.com/huyvanzzz/finetune-InternVL2.git
cd finetune-InternVL2
git fetch origin feature/trajectory-pretrain-qformer-concat-bestshot-bf16
git checkout -B feature/trajectory-pretrain-qformer-concat-bestshot-bf16 origin/feature/trajectory-pretrain-qformer-concat-bestshot-bf16
```

Neu repo da co san:

```bash
cd /workspace/finetune-InternVL2
git fetch origin feature/trajectory-pretrain-qformer-concat-bestshot-bf16
git checkout -B feature/trajectory-pretrain-qformer-concat-bestshot-bf16 origin/feature/trajectory-pretrain-qformer-concat-bestshot-bf16
```

## 2. Tao frame index va chuan bi Q-Former

```bash
python build_frame_index.py
python scripts/prepare_qformer.py --config internvl_config_traj_concat_bestshot_bf16_2gpu.yaml
python scripts/smoke_qformer_bridge.py --config internvl_config_traj_concat_bestshot_bf16_2gpu.yaml
```

## 3. Cau hinh `accelerate`

Chay:

```bash
accelerate config
```

Chon:
- `This machine`
- `multi-GPU`
- `1` machine
- `NO` cho `distributed error check` neu muon nhanh hon
- `NO` cho `torch dynamo`
- `NO` cho `DeepSpeed`
- `NO` cho `FSDP`
- `NO` cho `Megatron-LM`
- `2` GPU
- GPU ids: `all` hoac `0,1`
- `NO` cho `numa efficiency`

## 4. Train finetune 2 GPU

Thay `<CONCAT_PRETRAIN_CKPT>` bang checkpoint pretrain `concat` tuong thich.

```bash
accelerate launch --num_processes 2 train.py \
  --config internvl_config_traj_concat_bestshot_bf16_2gpu.yaml \
  --pretrain_checkpoint <CONCAT_PRETRAIN_CKPT>
```

Neu resume finetune:

```bash
accelerate launch --num_processes 2 train.py \
  --config internvl_config_traj_concat_bestshot_bf16_2gpu.yaml \
  --checkpoint outputs/internvl3_2b_traj_concat_bestshot_bf16_2gpu/<run_name>/epoch_3
```

## 5. Test tat ca checkpoint theo epoch va in metric

Lenh nay tu dong:
- tim run moi nhat trong `outputs/internvl3_2b_traj_concat_bestshot_bf16_2gpu`
- loop qua moi folder `epoch_*`
- chay `test_alter`
- in metric tung epoch
- sinh 2 file JSON cho moi epoch

```bash
python - <<'PY'
import subprocess
import yaml
import json
from pathlib import Path

CONFIG_PATH = "internvl_config_traj_concat_bestshot_bf16_2gpu.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    runtime_cfg = yaml.safe_load(f)

base_output_dir = Path(runtime_cfg["training"]["output_dir"])
result_stem = Path(CONFIG_PATH).stem.replace("internvl_config_", "")
run_dirs = sorted([p for p in base_output_dir.glob("*") if p.is_dir()], key=lambda p: p.stat().st_mtime)
latest_run_dir = run_dirs[-1] if run_dirs else None

if latest_run_dir is None:
    print("Khong tim thay run nao. Train truoc da.")
    raise SystemExit(1)

checkpoints = sorted(
    [p for p in latest_run_dir.glob("epoch_*") if p.is_dir() and p.name.split("_")[-1].isdigit()],
    key=lambda p: int(p.name.split("_")[-1]),
)

if not checkpoints:
    print("Khong tim thay folder epoch_* trong run moi nhat.")
    raise SystemExit(1)

for checkpoint_dir in checkpoints:
    epoch_suffix = checkpoint_dir.name
    output_file = f"results/{result_stem}_eval_test_alter_{epoch_suffix}.json"
    print(f"\\n=== Evaluating {checkpoint_dir} ===")
    subprocess.run([
        "python", "scripts/test_infer.py",
        "--config", CONFIG_PATH,
        "--checkpoint", str(checkpoint_dir),
        "--split", "test_alter",
        "--output_file", output_file,
    ], check=True)
    with open(output_file, "r", encoding="utf-8") as f:
        result_payload = json.load(f)
    print("Metrics:", json.dumps(result_payload.get("metrics", {}), ensure_ascii=False, indent=2))
    print("Result JSON:", output_file)
    print("Pairs JSON:", output_file.replace(".json", "_pairs.json"))
PY
```

## 6. Hai file JSON sau moi epoch

Moi epoch se co:

- file ket qua chinh:
  `results/traj_concat_bestshot_bf16_2gpu_eval_test_alter_epoch_X.json`
- file cap `ground_truth / generation`:
  `results/traj_concat_bestshot_bf16_2gpu_eval_test_alter_epoch_X_pairs.json`

## 7. Check nhanh log dung setup chua

Dau run nen thay cac thong tin kieu:
- `distributed=True`
- `world_size=2`
- `quantization_enabled=False`
- `bf16=True`
- `trajectory_mode=concat`
- `alter_only=True`
- `seed=42`
- `lora_r=16`

## 8. Luu y

- Phai dung `pretrain checkpoint concat`, khong dung checkpoint `cls_add`.
- Notebook cua nhanh nay la:
  `run_qformer_concat_bestshot_bf16_2gpu.ipynb`
- Neu test mot epoch cu the, sua truc tiep `--checkpoint` trong lenh `test_infer.py`.
