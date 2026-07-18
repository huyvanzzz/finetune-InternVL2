# Runbook pretrain trajectory tren server 2x RTX 4090

## Runtime truth

- Branch: `feature/pretrain-server-setup`
- Entrypoint: `train_pretrain.py`
- Mode mac dinh: `cls_add`
- Config mac dinh: `internvl_pretrain_config_traj_cls.yaml`
- Pretrain khong train LoRA.
- Trainable chi gom:
  - trajectory backbone
  - trajectory cls head
  - trajectory token projector
  - `qformer_input_proj`
  - `qformer_to_mlp1_proj`

## Dependency policy

Khong cai `torch` va `flash-attn` chung trong `requirements.txt`.

Thu tu dung:

```bash
pip install --upgrade pip setuptools wheel
pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1
pip install --no-cache-dir --no-deps -r requirements.txt
MAX_JOBS=4 pip install --no-build-isolation --no-cache-dir flash-attn==2.6.3
```

Ly do:

- `flash-attn` can `torch` co san truoc khi build.
- Cach cai tren tranh loi `ModuleNotFoundError: No module named 'torch'`.
- `requirements.txt` vi vay chi giu dependency Python thong thuong va duoc cai bang `--no-deps` de khong cho pip doi bo `torch`.
- De uu tien toc do cao nhat voi `flash-attn`, branch nay nen dung CUDA toolkit `12.8` va PyTorch `cu128`.

## Verify sau khi cai

```bash
python -c "import torch; print(torch.__version__)"
python -c "import torchvision, torchaudio; print(torchvision.__version__, torchaudio.__version__)"
python -c "import torch; print('torch cuda =', torch.version.cuda)"
python -c "import bitsandbytes as bnb; print(bnb.__version__)"
python -c "import accelerate; print(accelerate.__version__)"
python -c "import flash_attn; print(flash_attn.__version__)"
```

Neu `torch.version.cuda` hoac `nvcc --version` khong khop `12.8`, can doi template/container thay vi co build cuong ep tren instance do.

## Speed baseline cho 2x4090

- quantization: 4-bit
- dtype: bf16
- gradient checkpointing: off truoc
- world size: 2
- `num_workers`: 4 moi process
- mode dau tien de smoke va benchmark: `cls_add`

Neu OOM hoac VRAM qua sat, moi bat `gradient_checkpointing=true`.

## Split QA

Runtime dung 3 file co san:

- `json/question_train_split_train.jsonl`
- `json/question_train_split_val.jsonl`
- `json/question_train_split_test.jsonl`

Manifest:

- `json/question_train_split_manifest.json`

## Smoke run

```bash
accelerate launch --num_processes 2 train_pretrain.py \
  --config internvl_pretrain_config_traj_cls.yaml
```

## Resume

```bash
accelerate launch --num_processes 2 train_pretrain.py \
  --config internvl_pretrain_config_traj_cls.yaml \
  --checkpoint outputs/pretrain_traj_cls/<run_id>/last
```

Log can thay:

- resolved checkpoint path
- `training_state.json`
- `early_stopping_state.json`
- restored epoch / step / best val loss

## Debug can kiem

Log dau run nen co:

- world size / rank / device
- batch size / accumulation / global batch
- FlashAttention runtime status
- trainable parameter summary
- optimizer param groups
- sample question / answer / qformer text / trajectory
- tong token input
