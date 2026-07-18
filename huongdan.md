# Huong dan chay pretrain trajectory tren Vast.ai

## 1. Chay o dau

- Ban mo Vast.ai tren trinh duyet de:
  - xem instance da `running` chua
  - lay lenh SSH dung
  - xem GPU, disk, billing
- Ban chay train that bang SSH tu may local vao server.
- Sau khi vao server, nen dung `tmux` de job van chay du ban tat may local.

## 2. SSH tu may local

Neu da co SSH key tren may local:

```powershell
ssh -i $env:USERPROFILE\.ssh\id_ed25519 -p <PORT> root@<HOST>
```

Vi du:

```powershell
ssh -i $env:USERPROFILE\.ssh\id_ed25519 -p 38213 root@220.130.209.122
```

Neu lan dau ket noi, go `yes`.

## 3. Sau khi vao server

Tao `tmux`:

```bash
tmux new -s pretrain
```

Kiem tra nhanh:

```bash
nvidia-smi
nvcc --version
python --version
python - <<'PY'
import torch
print(torch.__version__)
print("torch cuda =", torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
if torch.cuda.device_count() >= 2:
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.get_device_name(1))
PY
```

## 4. Lay code dung nhanh

```bash
git clone https://github.com/huyvanzzz/finetune-InternVL2.git
cd finetune-InternVL2
git checkout feature/pretrain-server-setup
```

Neu da clone roi:

```bash
cd finetune-InternVL2
git fetch origin
git checkout feature/pretrain-server-setup
git pull origin feature/pretrain-server-setup
```

## 5. Cai moi truong dung thu tu

Khong cai `flash-attn` chung trong `requirements.txt`.
Neu truoc do da cai sai moi truong, don sach truoc:

```bash
pip uninstall -y flash-attn
pip uninstall -y torch torchvision torchaudio triton bitsandbytes accelerate transformers peft huggingface-hub tokenizers datasets evaluate timm scikit-learn scipy nltk rouge-score sentencepiece safetensors pandas numpy pyarrow
pip uninstall -y -r requirements.txt
pip cache purge
```

Thu tu dung la:

```bash
pip install --upgrade pip wheel
pip install --no-cache-dir "setuptools<82"
python -c "import torch; print(torch.__version__, torch.version.cuda)"
pip install --no-cache-dir --no-deps -r requirements.txt
MAX_JOBS=4 pip install --no-build-isolation --no-cache-dir flash-attn==2.6.3
```

Ly do:

- `requirements.txt` chi giu dependency Python thong thuong.
- `requirements.txt` duoc cai bang `--no-deps` de khong cho pip resolver tu y doi bo `torch` da cai truoc do.
- `flash-attn` can `torch` co san truoc khi build.
- Muc tieu chay nhanh nhat voi `flash-attn` tren branch nay la dung CUDA toolkit `12.8` va bo `torch` `cu128`.
- Neu instance moi da co san `torch` va `torch.version.cuda` khop `nvcc`, uu tien giu nguyen bo do, khong cai de.
- Khong nang `setuptools` len `82+` vi `torch 2.11.0+cu128` yeu cau `setuptools < 82`.
- Cai `flash-attn` bang `--no-build-isolation` de tranh loi `No module named 'torch'`.

Neu instance chua co `torch` dung ban CUDA, moi cai them nhu sau:

```bash
pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
pip install --no-cache-dir "setuptools<82"
```

## 6. Verify moi truong

```bash
nvcc --version
python -c "import torch; print(torch.__version__)"
python -c "import torchvision, torchaudio; print(torchvision.__version__, torchaudio.__version__)"
python -c "import torch; print('torch cuda =', torch.version.cuda)"
python -c "import setuptools; print(setuptools.__version__)"
python -c "import transformers; print(transformers.__version__)"
python -c "import accelerate; print(accelerate.__version__)"
python -c "import bitsandbytes as bnb; print(bnb.__version__)"
python -c "import flash_attn; print(flash_attn.__version__)"
```

Neu `flash_attn` import loi, gui lai dung traceback do.

Neu `torch.version.cuda` khong ra `12.8`, hoac `nvcc --version` khong ra `12.8`, thi khong nen build `flash-attn` tren instance do.

## 7. Kiem tra split QA

Repo hien dung san:

- `json/question_train_split_train.jsonl`
- `json/question_train_split_val.jsonl`
- `json/question_train_split_test.jsonl`
- `json/question_train_split_manifest.json`

Kiem tra:

```bash
ls json/question_train_split_*.jsonl
ls json/question_train_split_manifest.json
wc -l json/question_train_split_train.jsonl
wc -l json/question_train_split_val.jsonl
wc -l json/question_train_split_test.jsonl
```

Khong can tao lai neu 4 file nay da co.

## 8. Kiem tra nhanh code truoc khi train

```bash
python -m py_compile train_pretrain.py pretrain_dataset.py qformer_bridge.py trajectory_branch.py
pytest tests/test_pretrain_dataset.py tests/test_train_pretrain.py -q
```

## 9. Cau hinh accelerate

Chay mot lan:

```bash
accelerate config
```

Khuyen nghi cho 2x RTX 4090:

- compute environment: `This machine`
- distributed type: `multi-GPU`
- number of processes: `2`
- mixed precision: `bf16`
- dynamo: `no`

## 10. Smoke test dau tien

Mode mac dinh la `cls_add`:

```bash
accelerate launch --num_processes 2 train_pretrain.py \
  --config internvl_pretrain_config_traj_cls.yaml
```

Neu can resume:

```bash
accelerate launch --num_processes 2 train_pretrain.py \
  --config internvl_pretrain_config_traj_cls.yaml \
  --checkpoint outputs/pretrain_traj_cls/<run_id>/last
```

## 11. Khi tat may local

- Neu job dang chay trong `tmux` tren server, ban tat may local thi job van tiep tuc.
- Luc quay lai, SSH vao server roi attach lai:

```bash
tmux attach -t pretrain
```

## 12. Xem checkpoint o dau

Mac dinh checkpoint duoc luu trong `outputs/...` ben trong repo tren server.
Vi du:

```bash
ls outputs
find outputs -maxdepth 3 -type d
```

## 13. Neu can cap nhat code moi

```bash
cd finetune-InternVL2
git fetch origin
git checkout feature/pretrain-server-setup
git pull origin feature/pretrain-server-setup
```

Lenh nay chi lay phan code moi, khong clone lai tu dau.
