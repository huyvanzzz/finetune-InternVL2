# Huong Dan Day Du Cho Nhanh `concat` Best-shot BF16 2 GPU

Nhanh nay dung cho:
- branch: `feature/trajectory-pretrain-qformer-concat-bestshot-bf16`
- config: `internvl_config_traj_concat_bestshot_bf16_2gpu.yaml`
- mode: `concat`
- data: `alter_only`
- trajectory: `384 / 4 / 768 / dropout 0.10`
- quantization: `tat 4bit`
- dtype: `bf16`
- LoRA: `r=32`

## 0. Luu y truoc khi len server

Nhanh nay hien dang la nhanh lam viec moi.
Neu ban muon server `git fetch` duoc branch nay, ban phai push branch tu may local len GitHub truoc.

Len may local:

```powershell
cd D:\NCKH_VLM\finetune-InternVL2\.worktrees\trajectory-pretrain-qformer-concat-bestshot-bf16
git status
git push -u origin feature/trajectory-pretrain-qformer-concat-bestshot-bf16
```

Neu chua push, server se bao loi kieu:

```bash
fatal: couldn't find remote ref feature/trajectory-pretrain-qformer-concat-bestshot-bf16
```

## 1. SSH vao server tu may local

Neu chua co key:

```powershell
ssh-keygen -t ed25519 -f $env:USERPROFILE\.ssh\id_ed25519
Get-Content $env:USERPROFILE\.ssh\id_ed25519.pub
```

Copy public key do vao Vast.ai phan `SSH Keys`.

Neu da co key, SSH vao server bang lenh Vast cho:

```powershell
ssh -i $env:USERPROFILE\.ssh\id_ed25519 -p <SSH_PORT> root@<SERVER_IP>
```

Vi du:

```powershell
ssh -i $env:USERPROFILE\.ssh\id_ed25519 -p 13414 root@210.64.18.177
```

Neu lan dau ket noi, go:

```text
yes
```

Neu bi `Permission denied (publickey)`:
- check lai dung file `id_ed25519`
- check lai public key trong Vast
- neu key moi add sau khi tao instance, nen tao lai instance moi

## 2. Sau khi vao server

Nen dung `tmux` de job van chay du ban mat mang hay tat may local:

```bash
tmux new -s concat_bestshot
```

Neu da co session:

```bash
tmux attach -t concat_bestshot
```

Bat chuot cuon trong tmux:

```bash
tmux set -g mouse on
```

## 3. Kiem tra nhanh may va CUDA

```bash
nvidia-smi
nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda)"
ls -d /usr/local/cuda* 2>/dev/null
```

Neu muon cai `flash-attn` tu source cho nhanh hon, nen co:
- `nvcc = 13.0`
- `torch.version.cuda = 13.0`

Neu `nvcc` va `torch.version.cuda` lech nhau, de bi loi build `flash-attn`.

## 4. Lay code dung nhanh

Neu chua clone repo:

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
git pull origin feature/trajectory-pretrain-qformer-concat-bestshot-bf16
```

## 5. Cai moi truong

Neu server da co `/venv/main` va ban dang o prompt `(main)`, van nen cai lai dung thu tu:

```bash
cd /workspace/finetune-InternVL2
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install pytest
```

Neu muon dung `flash-attn`:

```bash
pip install ninja
MAX_JOBS=4 pip install --no-build-isolation flash-attn
```

Neu `flash-attn` loi mismatch CUDA, bo qua tam thoi cung duoc; nhanh nay uu tien chat luong `bf16`, khong phu thuoc 4bit.

## 6. Kiem tra dependency

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -c "import accelerate; print(accelerate.__version__)"
python -c "import transformers; print(transformers.__version__)"
python -c "import peft; print(peft.__version__)"
python -c "import yaml; print('yaml ok')"
python -m py_compile train.py wad_dataset.py qformer_bridge.py trajectory_branch.py trajectory_trainability.py
```

Neu muon chay them test source-level:

```bash
python -m pytest tests/test_finetune_alter_only.py tests/test_pretrain_handoff.py tests/test_qformer_bridge_dual.py tests/test_trajectory_branch.py -q
```

## 7. Tao frame index va chuan bi Q-Former

```bash
python build_frame_index.py
python scripts/prepare_qformer.py --config internvl_config_traj_concat_bestshot_bf16_2gpu.yaml
python scripts/smoke_qformer_bridge.py --config internvl_config_traj_concat_bestshot_bf16_2gpu.yaml
```

## 8. Cau hinh `accelerate`

Chay mot lan:

```bash
accelerate config
```

Chon:
- `This machine`
- `multi-GPU`
- `1` machine
- `NO` cho `distributed operations check`
- `NO` cho `torch dynamo`
- `NO` cho `DeepSpeed`
- `NO` cho `FSDP`
- `NO` cho `Megatron-LM`
- `2` GPU
- GPU ids: `all` hoac `0,1`
- `NO` cho `numa efficiency`

## 9. Train finetune 2 GPU

Phai dung `pretrain checkpoint concat`, khong dung checkpoint `cls_add`.

```bash
accelerate launch --num_processes 2 train.py \
  --config internvl_config_traj_concat_bestshot_bf16_2gpu.yaml \
  --pretrain_checkpoint <CONCAT_PRETRAIN_CKPT>
```

Vi du neu checkpoint o Hugging Face:

```bash
accelerate launch --num_processes 2 train.py \
  --config internvl_config_traj_concat_bestshot_bf16_2gpu.yaml \
  --pretrain_checkpoint huyvanzzz/pretrain_concat
```

Neu resume finetune:

```bash
accelerate launch --num_processes 2 train.py \
  --config internvl_config_traj_concat_bestshot_bf16_2gpu.yaml \
  --checkpoint outputs/internvl3_2b_traj_concat_bestshot_bf16_2gpu/<run_name>/epoch_3
```

## 10. Kiem tra log dau run

Dau run nen thay cac thong tin kieu:
- `distributed=True`
- `world_size=2`
- `quantization_enabled=False`
- `bf16=True`
- `trajectory_mode=concat`
- `alter_only=True`
- `seed=42`
- `lora_r=32`

## 11. Test tat ca checkpoint theo epoch va in metric

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

## 12. 2 file JSON sau moi epoch

Moi epoch se co:

- file ket qua chinh:
  `results/traj_concat_bestshot_bf16_2gpu_eval_test_alter_epoch_X.json`
- file cap `ground_truth / generation`:
  `results/traj_concat_bestshot_bf16_2gpu_eval_test_alter_epoch_X_pairs.json`

## 13. Tai checkpoint ve may local

Tao folder local:

```powershell
mkdir D:\NCKH_VLM\checkpoints\concat_bestshot -Force
```

Tai ca folder `epoch_5` vi du:

```powershell
scp -i $env:USERPROFILE\.ssh\id_ed25519 -P <SSH_PORT> -r root@<SERVER_IP>:/workspace/finetune-InternVL2/outputs/internvl3_2b_traj_concat_bestshot_bf16_2gpu/<run_name>/epoch_5 D:\NCKH_VLM\checkpoints\concat_bestshot\
```

Tai rieng file metric:

```powershell
scp -i $env:USERPROFILE\.ssh\id_ed25519 -P <SSH_PORT> root@<SERVER_IP>:/workspace/finetune-InternVL2/outputs/internvl3_2b_traj_concat_bestshot_bf16_2gpu/<run_name>/metrics.json D:\NCKH_VLM\checkpoints\concat_bestshot\
```

## 14. Neu mat mang hoac tat may local

- Neu train dang chay trong `tmux`, job van chay tren server.
- Vao lai:

```bash
tmux attach -t concat_bestshot
```

## 15. Notebook dung cho nhanh nay

Notebook tuong ung la:

```text
run_qformer_concat_bestshot_bf16_2gpu.ipynb
```

Notebook nay da duoc sua de:
- train bang `accelerate launch --num_processes 2`
- co `PRETRAIN_CHECKPOINT`
- test tat ca `epoch_*`
- in metric tung epoch
- bao duong dan den 2 file JSON cua tung epoch
