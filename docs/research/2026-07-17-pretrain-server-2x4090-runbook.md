# Runbook pretrain trajectory trên server 2x RTX 4090

## Mặc định

- Branch: `feature/pretrain-server-setup`
- Entry point: `train_pretrain.py`
- Mode mặc định: `cls_add`
- Config mặc định: `internvl_pretrain_config_traj_cls.yaml`
- Pretrain không train LoRA.
- Trainable chỉ gồm trajectory branch và hai bridge projections.
- Speed baseline dùng `4-bit + bf16 + gradient_checkpointing=false`.
- `num_workers=4` là 4 worker mỗi process, không phải toàn server.
- QA pretrain dùng 3 file split cố định:
  - `json/question_train_split_train.jsonl`
  - `json/question_train_split_val.jsonl`
  - `json/question_train_split_test.jsonl`
  - manifest: `json/question_train_split_manifest.json`

## Setup nhanh

```bash
git clone https://github.com/huyvanzzz/finetune-InternVL2.git
cd finetune-InternVL2
git checkout feature/pretrain-server-setup
pip install -r requirements.txt
```

Kiểm tra GPU:

```bash
nvidia-smi
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
PY
```

## Accelerate config

Chạy một lần:

```bash
accelerate config
```

Khuyến nghị cho 2 RTX 4090:

- compute environment: local machine
- distributed type: multi-GPU
- number of processes: 2
- mixed precision: bf16
- dynamo: no

## Smoke run

Nếu cần tạo lại split từ file nguồn:

```bash
python scripts/prepare_pretrain_qa_splits.py \
  --input json/question_train.jsonl \
  --output_dir json \
  --seed 42
```

Chạy smoke ngắn bằng `cls_add`:

```bash
accelerate launch --num_processes 2 train_pretrain.py \
  --config internvl_pretrain_config_traj_cls.yaml
```

Nếu OOM, bật fallback VRAM bằng cách đổi trong config:

```yaml
training:
  gradient_checkpointing: true
```

## Train các mode

`cls_add` là baseline chính:

```bash
accelerate launch --num_processes 2 train_pretrain.py \
  --config internvl_pretrain_config_traj_cls.yaml
```

`concat`:

```bash
accelerate launch --num_processes 2 train_pretrain.py \
  --config internvl_pretrain_config_traj_concat.yaml
```

`dual`:

```bash
accelerate launch --num_processes 2 train_pretrain.py \
  --config internvl_pretrain_config_traj_dual.yaml
```

## Resume

Resume từ checkpoint local:

```bash
accelerate launch --num_processes 2 train_pretrain.py \
  --config internvl_pretrain_config_traj_cls.yaml \
  --checkpoint outputs/pretrain_traj_cls/<run_id>/last
```

Log cần thấy:

- resolved checkpoint path
- `training_state.json`
- `early_stopping_state.json`
- restored `best_epoch`, `best_val_loss`, `bad_epochs`

## Debug cần kiểm khi bắt đầu run

Log đầu run phải có:

- world size, rank, device
- batch size, accumulation, global batch
- quantization, bf16, gradient checkpointing
- split stats train/val/test
- FlashAttention runtime status
- trainable parameter summary
- optimizer param group health
- sample question/answer/qformer text/trajectory
- token count

## Benchmark nên chạy

Chạy ngắn để so:

- `gradient_checkpointing=false`
- `gradient_checkpointing=true` nếu OOM hoặc muốn thử batch lớn hơn
- `num_workers=4/process`
- `num_workers=8/process`

So sánh bằng:

- samples/sec
- optimizer updates/sec
- step time
- VRAM peak
- data loading time nếu log profiling cho thấy bottleneck
