# Hướng dẫn chạy pretrain trajectory qua SSH trên Vast.ai

## 1. Chạy ở đâu?

- Bạn mở Vast.ai trên trình duyệt để:
  - kiểm tra instance đã `running` chưa
  - lấy lệnh SSH đúng
  - xem trạng thái GPU, disk, billing
  - stop hoặc restart instance khi cần
- Bạn chạy train thật bằng SSH từ máy local vào server Vast.ai.
- Khi đã vào server, nên dùng `tmux` để job vẫn chạy dù bạn tắt máy local.

## 2. Cách SSH mặc định nên dùng

Với Windows của bạn, cách dễ và ổn định nhất là dùng thẳng private key bằng `-i`.

Ví dụ:

```powershell
ssh -i $env:USERPROFILE\.ssh\id_ed25519 -p 38354 root@220.130.209.122
```

Không cần cấu hình `ssh-agent` nếu bạn chưa muốn làm thêm.

## 3. Nếu máy local chưa có SSH key

Kiểm tra:

```powershell
dir $env:USERPROFILE\.ssh
```

Nếu chưa thấy `id_ed25519` và `id_ed25519.pub`, tạo mới:

```powershell
ssh-keygen -t ed25519 -C "vastai"
```

Sau đó:

- nhấn `Enter` để lưu ở đường dẫn mặc định
- nếu muốn nhanh thì để passphrase trống

Tiếp theo in public key ra:

```powershell
Get-Content $env:USERPROFILE\.ssh\id_ed25519.pub
```

Copy toàn bộ dòng bắt đầu bằng `ssh-ed25519 ...` rồi dán vào Vast.ai ở phần SSH Keys.

## 4. Trước khi SSH cần kiểm tra gì trên Vast.ai?

Trên giao diện Vast.ai, kiểm tra:

- instance không còn ở trạng thái `scheduling`
- instance đã là `running`
- mục `Connect` hoặc `Direct SSH` đã hiện lệnh SSH

Ví dụ:

```powershell
ssh -i $env:USERPROFILE\.ssh\id_ed25519 -p 38213 root@220.130.209.122
```

Nếu instance còn `scheduling` thì SSH có thể bị `Connection refused`.

## 5. Bước 1: SSH từ máy local vào server

Mở PowerShell trên máy local và chạy:

```powershell
ssh -i $env:USERPROFILE\.ssh\id_ed25519 -p 38213 root@220.130.209.122
```

Nếu lần đầu vào host đó, SSH có thể hỏi:

```text
Are you sure you want to continue connecting (yes/no/[fingerprint])?
```

Khi đó gõ:

```text
yes
```

Nếu vào được server, bạn sẽ thấy prompt Linux kiểu:

```bash
root@...:~#
```

## 6. Bước 2: Tạo phiên `tmux`

Ngay sau khi SSH vào server, chạy:

```bash
tmux new -s pretrain
```

Từ đây trở đi, nên chạy toàn bộ lệnh trong phiên `tmux` này.

## 7. Bước 3: Kiểm tra server đã ổn chưa

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
- `torch.cuda.is_available()` là `True`
- `torch.cuda.device_count()` là `2`
- disk còn đủ chỗ

## 8. Bước 4: Lấy code đúng nhánh

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

## 9. Bước 5: Cài môi trường

Chạy:

```bash
pip install -r requirements.txt
```

Verify thêm:

```bash
python -c "import torch; print(torch.__version__)"
python -c "import transformers; print(transformers.__version__)"
python -c "import accelerate; print(accelerate.__version__)"
python -c "import bitsandbytes as bnb; print(bnb.__version__)"
python -c "import flash_attn; print(flash_attn.__version__)"
```

Nếu `flash_attn` import lỗi thì gửi lại lỗi đó.

## 10. Bước 6: Kiểm tra file split đã có chưa

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

Nếu các file này đã có thì không cần tạo lại.

## 11. Bước 7: Kiểm tra syntax trước khi train

Chạy:

```bash
python -m py_compile train_pretrain.py pretrain_dataset.py qformer_bridge.py trajectory_branch.py
```

Nếu muốn test nhanh:

```bash
pytest tests/test_pretrain_dataset.py tests/test_train_pretrain.py -q
```

## 12. Bước 8: Cấu hình `accelerate`

Chạy:

```bash
accelerate config
```

Khuyến nghị ban đầu cho 2x RTX 4090:

- compute environment: `This machine`
- distributed type: `multi-GPU`
- number of processes: `2`
- mixed precision: `bf16`

## 13. Bước 9: Smoke test đầu tiên

Nên chạy mode `cls_add` trước:

```bash
accelerate launch --num_processes 2 train_pretrain.py \
  --config internvl_pretrain_config_traj_cls.yaml
```

Khi chạy, kiểm tra log đầu có các phần:

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

## 14. Chạy từng mode

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

## 15. Resume checkpoint

Ví dụ resume từ checkpoint `last`:

```bash
accelerate launch --num_processes 2 train_pretrain.py \
  --config internvl_pretrain_config_traj_cls.yaml \
  --checkpoint outputs/pretrain_traj_cls/<run_id>/last
```

Khi resume đúng, log nên có:

- đường dẫn checkpoint đã resolve
- `training_state.json`
- `early_stopping_state.json`
- `start_epoch`
- `start_step`
- `global_optimizer_step`

## 16. Cách thoát mà job vẫn chạy

Trong `tmux`, nhấn:

```text
Ctrl+b rồi nhấn d
```

Lúc đó bạn đã detach khỏi `tmux`, nhưng job vẫn chạy trên server.

Khi quay lại:

```bash
tmux attach -t pretrain
```

## 17. Nếu tắt máy local thì job có chạy tiếp không?

Có, nếu:

- job đang chạy bên trong `tmux`
- instance Vast.ai vẫn còn sống

Không, nếu:

- bạn chạy trực tiếp trong SSH mà không dùng `tmux`
- hoặc bạn stop instance trên Vast.ai

## 18. Cách kiểm tra nhanh job còn sống không

Sau khi SSH lại vào server:

```bash
tmux ls
nvidia-smi
```

Nếu job còn chạy, thường sẽ thấy:

- có session `pretrain`
- `nvidia-smi` có process Python đang dùng GPU

## 19. Nếu gặp lỗi, cần gửi lại gì?

Nếu có lỗi, hãy gửi lại:

1. lệnh bạn đã chạy
2. toàn bộ traceback hoặc log lỗi
3. kết quả của:

```bash
nvidia-smi
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

4. nếu lỗi SSH thì gửi cả lệnh:

```powershell
ssh -i $env:USERPROFILE\.ssh\id_ed25519 -p 38213 root@220.130.209.122
```

và trạng thái instance trên Vast.ai là `scheduling` hay `running`

## 20. Checklist ngắn gọn

Làm đúng thứ tự này:

1. Vào Vast.ai, đợi instance `running`
2. Lấy lệnh SSH
3. SSH từ máy local bằng `ssh -i ...`
4. Tạo `tmux`
5. Kiểm tra GPU, Python, Torch
6. Clone hoặc pull repo và checkout đúng nhánh
7. Kiểm tra 3 file split
8. `accelerate config`
9. Chạy smoke test `cls_add`
10. Xem log verify
11. Nếu ổn thì chạy train thật
