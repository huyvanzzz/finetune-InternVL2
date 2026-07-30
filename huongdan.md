# Huong Dan Chay Latency Benchmark Tren GPU Moi

File nay dung cho nhanh:

```text
feature/latency-benchmark-matrix
```

Muc tieu: thue 1 GPU moi, setup moi truong, checkout dung nhanh, roi chay latency benchmark tren **1 GPU duy nhat** bang `CUDA_VISIBLE_DEVICES=0`.

## 1. Chuan bi SSH key `id_ed25519`

Tren may local Windows, kiem tra da co key chua:

```powershell
Test-Path $env:USERPROFILE\.ssh\id_ed25519
Test-Path $env:USERPROFILE\.ssh\id_ed25519.pub
```

Neu chua co, tao key moi:

```powershell
ssh-keygen -t ed25519 -f $env:USERPROFILE\.ssh\id_ed25519
```

In public key de copy vao Vast.ai / server SSH keys:

```powershell
Get-Content $env:USERPROFILE\.ssh\id_ed25519.pub
```

File private key dung de SSH la:

```powershell
$env:USERPROFILE\.ssh\id_ed25519
```

## 2. SSH vao server

Tu may local:

```powershell
ssh -i $env:USERPROFILE\.ssh\id_ed25519 -p 54387 root@211.72.37.229
```

Sau khi vao server, tao tmux de job khong bi dung khi mat ket noi:

```bash
tmux new -s latency
```

Neu can detach khoi tmux ma van giu job chay:

```text
Ctrl+b roi nhan d
```

Vao lai tmux:

```bash
tmux attach -t latency
```

## 3. Kiem tra GPU va CUDA

```bash
nvidia-smi
nvcc --version || true
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Ky vong:

```text
torch.cuda.is_available() = True
```

Neu server la CUDA 13.x, `bitsandbytes` co the khong ho tro binary. Script benchmark da co `--quantization_mode auto` mac dinh de tu tat 4-bit khi CUDA 13 khong hop.

## 4. Lay code dung nhanh

Neu chua clone repo:

```bash
cd /workspace
git clone https://github.com/huyvanzzz/finetune-InternVL2.git
cd finetune-InternVL2
```

Neu repo da co san:

```bash
cd /workspace/finetune-InternVL2
```

Checkout nhanh latency:

```bash
git fetch origin feature/latency-benchmark-matrix
git checkout -B feature/latency-benchmark-matrix origin/feature/latency-benchmark-matrix
git pull origin feature/latency-benchmark-matrix
```

Neu checkout bi chan boi file `__pycache__/*.pyc`, chay:

```bash
git restore '*.pyc'
git checkout -B feature/latency-benchmark-matrix origin/feature/latency-benchmark-matrix
git pull origin feature/latency-benchmark-matrix
```

Kiem tra dung nhanh:

```bash
git branch --show-current
git log --oneline -n 3
```

Can thay:

```text
feature/latency-benchmark-matrix
```

## 5. Cai dependency

Neu server da co venv `(main)` thi dung luon. Neu chua co:

```bash
python -m venv /venv/main
source /venv/main/bin/activate
```

Cap nhat pip:

```bash
pip install --upgrade pip setuptools wheel
```

Cai core packages:

```bash
pip install transformers==4.46.2 datasets accelerate timm safetensors huggingface_hub sentencepiece peft pyyaml pillow tqdm scikit-learn
```

Cai torch theo CUDA cua server. Neu server co CUDA 12.8:

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
```

Neu server image da co torch CUDA dung san thi co the bo qua lenh cai torch.

Cai optional packages:

```bash
pip install bitsandbytes
pip install ninja
MAX_JOBS=4 pip install --no-build-isolation flash-attn || true
```

Luu y:
- `flash-attn` tot cho latency neu cai duoc, nhung benchmark van chay neu no loi.
- `bitsandbytes` co the loi tren CUDA 13.x. Script benchmark mac dinh se auto disable 4-bit trong case nay.

## 6. Verify dependency

```bash
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda, 'available:', torch.cuda.is_available())"
python -c "import transformers, accelerate, peft, yaml; print('core deps ok')"
python -c "import flash_attn; print('flash_attn ok')" || echo "flash_attn not available"
python -c "import bitsandbytes as bnb; print('bitsandbytes:', bnb.__version__)" || echo "bitsandbytes not available"
python -m py_compile scripts/benchmark_latency.py runtime_flash_attention.py qformer_bridge.py trajectory_branch.py wad_dataset.py
```

Neu `bitsandbytes` loi CUDA 13, van tiep tuc duoc vi `--quantization_mode auto` la mac dinh.

## 7. Chuan bi data phu tro

Build frame index neu server chua co:

```bash
python build_frame_index.py
```

Kiem tra Q-Former cho config can chay, vi du concat:

```bash
python scripts/prepare_qformer.py --config internvl_config_traj_concat.yaml
python scripts/smoke_qformer_bridge.py --config internvl_config_traj_concat.yaml
```

Voi mode khac, doi config tuong ung.

## 8. Chay benchmark 1 GPU

Tat ca lenh dung:

```bash
CUDA_VISIBLE_DEVICES=0
```

Khong dung `accelerate launch`, khong dung `--num_processes 2`, khong chay song song tren GPU nay neu muon so latency sach.

Mac dinh:
- `generation_mode=latency_greedy`
- `num_beams=1`
- `do_sample=false`
- `warmup_samples=5`
- `quantization_mode=auto`
- khong tinh object/tracking, `object_tracking_ms=0.0`

Neu muon check behavior giong eval setup cu cua `restore-779cc7b`, them:

```bash
--generation_mode restore_eval
```

Mode nay se ep `num_beams=3`, `do_sample=false`, `repetition_penalty=1.3`, `early_stopping=true`.
Chi dung mode nay de doi chieu output/behavior; khong dung `decode_only_tokens_per_s` cua mode nay lam metric latency chinh.

### InternVL Q-Former

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.benchmark_latency \
  --config internvl_config.yaml \
  --checkpoint minhdang0901/intern-qformer-2707-epoch3 \
  --split test_alter \
  --output_file results/latency_internvl_qformer_epoch3_full.json \
  --generation_mode latency_greedy \
  --num_beams 1 \
  --warmup_samples 5
```

### InternVL No Q-Former

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.benchmark_latency \
  --config internvl_config_no_qformer.yaml \
  --checkpoint abcdsayhi19/internvl3_2b_no_qformer_2507_epoch3 \
  --split test_alter \
  --output_file results/latency_internvl_no_qformer_epoch3_full.json \
  --generation_mode latency_greedy \
  --num_beams 1 \
  --warmup_samples 5
```

### SAIL-VL Q-Former

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.benchmark_latency \
  --config sailvl_config.yaml \
  --checkpoint abcdsayhi19/sailvl_1d5_2b_qformer_epoch2 \
  --split test_alter \
  --output_file results/latency_sailvl_qformer_epoch2_full.json \
  --generation_mode latency_greedy \
  --num_beams 1 \
  --warmup_samples 5
```

### SAIL-VL No Q-Former

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.benchmark_latency \
  --config sailvl_config_no_qformer.yaml \
  --checkpoint Ares628/run_sail_no_qformer_epoch3 \
  --split test_alter \
  --output_file results/latency_sailvl_no_qformer_epoch3_full.json \
  --generation_mode latency_greedy \
  --num_beams 1 \
  --warmup_samples 5
```

### InternVL Trajectory Concat

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.benchmark_latency \
  --config internvl_config_traj_concat.yaml \
  --checkpoint minhdang0901/intern-qformer-concat-1807-epoch3 \
  --split test_alter \
  --output_file results/latency_internvl_traj_concat_epoch3_full.json \
  --generation_mode latency_greedy \
  --num_beams 1 \
  --warmup_samples 5
```

### InternVL Trajectory CLS

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.benchmark_latency \
  --config internvl_config_traj_cls.yaml \
  --checkpoint minhdang0901/intern-pretrain-finetune-cls-2307-epoch2 \
  --split test_alter \
  --output_file results/latency_internvl_traj_cls_epoch2_full.json \
  --generation_mode latency_greedy \
  --num_beams 1 \
  --warmup_samples 5
```

## 9. Xem ket qua

Xem summary cua mot file:

```bash
python - <<'PY'
import json
p="results/latency_internvl_traj_concat_epoch3_full.json"
d=json.load(open(p, encoding="utf-8"))
print(json.dumps(d["summary"], indent=2, ensure_ascii=False))
print(json.dumps(d["run_metadata"], indent=2, ensure_ascii=False))
PY
```

Trong `run_metadata`, can check:
- `generation_mode`
- `decode_only_tokens_per_s_valid`
- `flash_attention_requested`
- `flash_attention_available`
- `flash_attention_active`
- `flash_attention_layer_count`
- `flash_attention_active_layer_count`
- `torch_version`
- `torch_cuda_version`
- `cuda_device_name`
- `model_num_image_token`
- `quantization_requested`
- `quantization_effective`
- `quantization_disable_reason`

So Q-Former voi No Q-Former theo tung sample:

```bash
python -m scripts.analyze_latency_pair \
  --no_qformer_json results/latency_internvl_no_qformer_epoch3_full.json \
  --qformer_json results/latency_internvl_qformer_epoch3_full.json \
  --output_file results/latency_internvl_qformer_vs_no_qformer_analysis.json
```

Trong `summary`, lay 2 metric chinh:
- `end_to_end_ms`
- `decode_only_tokens_per_s`

## 10. Tai ket qua ve may local

Tu may local:

```powershell
mkdir D:\NCKH_VLM\latency_results -Force
scp -i $env:USERPROFILE\.ssh\id_ed25519 -P <SSH_PORT> root@<SERVER_IP>:/workspace/finetune-InternVL2/results/latency_*.json D:\NCKH_VLM\latency_results\
```

## 11. Loi thuong gap

Neu loi `No module named runtime_flash_attention`:

```bash
python -m scripts.benchmark_latency --help
```

Khong chay bang `python scripts/benchmark_latency.py`; hay chay bang `python -m scripts.benchmark_latency`.

Neu loi `bitsandbytes CUDA VERSION MISMATCH`:

```bash
git pull origin feature/latency-benchmark-matrix
```

Sau do chay lai voi mac dinh `--quantization_mode auto`. Neu van muon ep dung config 4-bit thi them:

```bash
--quantization_mode config
```

Nhung voi CUDA 13.x, khong nen ep 4-bit vi bitsandbytes co the khong co binary phu hop.

Neu OOM:
- Kiem tra co process khac khong: `nvidia-smi`
- Dam bao chi chay 1 benchmark tren GPU do.
- Thu lai voi `--quantization_mode auto` tren CUDA 12.8, hoac thue GPU VRAM lon hon.
