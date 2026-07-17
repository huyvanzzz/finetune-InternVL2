# Hướng dẫn chạy pretrain trajectory qua SSH trên Vast.ai

## 1. Chạy ở đâu?

- Các lệnh `ssh`, `git`, `tmux`, `accelerate launch`, `train_pretrain.py`:
  chạy trên server Vast.ai.
- Bạn gõ các lệnh đó từ terminal trên máy local bằng cách SSH vào server.
- Nếu dùng `tmux`, bạn có thể tắt máy local mà job vẫn tiếp tục chạy trên Vast.ai.

## 2. Có cần mở Vast.ai trên trình duyệt không?

Có, nhưng chỉ để:

- xem instance đã ở trạng thái `running` hay chưa
- lấy đúng lệnh SSH
- xem log tổng quan như GPU, disk, billing, trạng thái instance
- khi cần stop/restart instance

Train thật vẫn nên chạy bằng SSH.

## 3. Trước khi SSH cần kiểm tra gì trên Vast.ai?

Trên giao diện Vast.ai, kiểm tra:

- instance không còn ở trạng thái `scheduling`
- instance đã là `running`
- có nút hoặc mục `Connect` / `SSH`
- có lệnh SSH đầy đủ, ví dụ:

```bash
ssh -p 11419 root@ssh2.vast.ai
```

Nếu còn `scheduling` thì SSH thường sẽ bị `Connection refused`.

## 4. Bước 1: SSH từ máy local vào server

Trên máy local, mở PowerShell hoặc terminal rồi chạy đúng lệnh Vast.ai cung cấp.

Ví dụ:

```powershell
ssh -p 11419 root@ssh2.vast.ai
```

Nếu vào được server, bạn sẽ thấy prompt kiểu Linux như:

```bash
root@...:~#
```

## 5. Bước 2: Tạo phiên `tmux` để giữ job chạy nền

Ngay sau khi SSH vào server, chạy:

```bash
tmux new -s pretrain
```

Từ đây trở đi, bạn nên chạy toàn bộ lệnh trong phiên `tmux` này.

## 6. Bước 3: Kiểm tra server đã ổn chưa

Chạy lần lượt:

```bash
pwd
nvidia-smi
python --version
python -c "import torch; print('cuda=', torch.cuda.is_available(), 'num_gpus=', torch.cuda.device_count())"
python -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_name(1))"
df -h
```

Bạn cần thấy tối thiểu:

- có 2 GPU `RTX 4090`
- `torch.cuda.is_available() = True`
- `torch.cuda.device_count() = 2`
- disk còn đủ chỗ

## 7. Bước 4: Lấy code đúng nhánh

Nếu chưa clone repo:

```bash
git clone https://github.com/huyvanzzz/finetune-InternVL2.git
cd finetune-InternVL2
git checkout feature/pretrain-server-setup
```

Nếu đã clone từ trước:

```bash
cd finetune-InternVL2
git fetch origin
git checkout feature/pretrain-server-setup
git pull origin feature/pretrain-server-setup
```

## 8. Bước 5: Cài môi trường

Chạy:

```bash
pip install -r requirements.txt
```

Nên verify thêm:

```bash
python -c "import torch; print(torch.__version__)"
python -c "import transformers; print(transformers.__version__)"
python -c "import accelerate; print(accelerate.__version__)"
python -c "import bitsandbytes as bnb; print(bnb.__version__)"
python -c "import flash_attn; print(flash_attn.__version__)"
```

Nếu `flash_attn` import lỗi thì gửi lại lỗi đó, tôi sẽ giúp sửa tiếp.

## 9. Bước 6: Kiểm tra file split đã có chưa

Repo hiện tại dùng sẵn 3 file:

- `json/question_train_split_train.jsonl`
- `json/question_train_split_val.jsonl`
- `json/question_train_split_test.jsonl`

Kiểm tra:

```bash
ls json/question_train_split_*.jsonl
ls json/question_train_split_manifest.json
```

Kiểm tra nhanh số dòng:

```bash
wc -l json/question_train_split_train.jsonl
wc -l json/question_train_split_val.jsonl
wc -l json/question_train_split_test.jsonl
```

Nếu 3 file này đã có thì không cần tạo lại.

## 10. Bước 7: Kiểm tra syntax trước khi train

Chạy:

```bash
python -m py_compile train_pretrain.py pretrain_dataset.py qformer_bridge.py trajectory_branch.py
```

Nếu muốn kiểm thêm test nhanh:

```bash
pytest tests/test_pretrain_dataset.py tests/test_train_pretrain.py -q
```

## 11. Bước 8: Cấu hình `accelerate`

Chạy:

```bash
accelerate config
```

Khuyến nghị ban đầu cho 2x RTX 4090:

- compute environment: `This machine`
- distributed type: `multi-GPU`
- number of processes: `2`
- mixed precision: `bf16`

## 12. Bước 9: Smoke test đầu tiên

Nên chạy mode `cls_add` trước.

```bash
accelerate launch --num_processes 2 train_pretrain.py \
  --config internvl_pretrain_config_traj_cls.yaml
```

Khi chạy, hãy kiểm tra log đầu có các phần sau:

- `Pretrain split stats`
- `FlashAttention runtime`
- `Trainable parameter summary`
- `Optimizer param groups`
- `sample question`
- `qformer_text`
- `target answer`
- `trajectory sample`
- `total input tokens`

Nếu các block này hiện ra bình thường thì luồng cơ bản đang ổn.

## 13. Chạy từng mode

`cls_add`:

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

## 14. Resume checkpoint

Ví dụ resume từ checkpoint `last`:

```bash
accelerate launch --num_processes 2 train_pretrain.py \
  --config internvl_pretrain_config_traj_cls.yaml \
  --checkpoint outputs/pretrain_traj_cls/<run_id>/last
```

Khi resume đúng, log nên có:

- đường dẫn checkpoint đã resolve
- có hoặc không có `training_state.json`
- có hoặc không có `early_stopping_state.json`
- `start_epoch`
- `start_step`
- `global_optimizer_step`

## 15. Cách thoát mà job vẫn chạy

Trong `tmux`, nhấn:

```text
Ctrl+b rồi nhấn d
```

Lúc đó bạn đã detach khỏi phiên `tmux`, nhưng job vẫn chạy trên server.

Sau này SSH lại và vào tiếp:

```bash
tmux attach -t pretrain
```

## 16. Nếu tắt máy local thì job có chạy tiếp không?

Có, nếu:

- bạn đã chạy job bên trong `tmux`
- instance Vast.ai vẫn còn sống

Không, nếu:

- bạn chạy trực tiếp trong SSH mà không dùng `tmux` hoặc `screen`
- hoặc bạn stop instance trên Vast.ai

## 17. Cách kiểm tra nhanh job còn sống không

Sau khi SSH lại vào server:

```bash
tmux ls
nvidia-smi
```

Nếu job đang chạy, thường sẽ thấy:

- có session `pretrain`
- `nvidia-smi` có process Python đang dùng GPU

## 18. Nếu gặp lỗi, cần chụp gì gửi lại?

Nếu có lỗi, hãy gửi lại:

1. lệnh bạn đã chạy
2. toàn bộ traceback hoặc log lỗi
3. kết quả của:

```bash
nvidia-smi
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

4. nếu lỗi SSH thì gửi cả:

```powershell
ssh -p <PORT> root@ssh2.vast.ai
```

và trạng thái instance trên Vast.ai là `scheduling` hay `running`

## 19. Checklist ngắn gọn

Chạy đúng thứ tự này:

1. Vào Vast.ai, đợi instance `running`
2. Copy lệnh SSH
3. SSH từ máy local vào server
4. Tạo `tmux`
5. Kiểm tra GPU, Python, Torch
6. Clone/pull repo và checkout đúng nhánh
7. Kiểm tra 3 file split
8. `accelerate config`
9. Chạy smoke test `cls_add`
10. Xem log verify
11. Nếu ổn thì chạy train thật
