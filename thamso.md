Báo cáo quyết định
Batch size mặc định hiện tại:cls_add: batch_size=32
concat: batch_size=16
dual: batch_size=16

Gradient accumulation và effective global batch:Cả 3 mode: gradient_accumulation_steps=1
Nếu chạy đúng server 2 GPU bằng accelerate --num_processes 2:
cls_add: effective global batch = 32 * 1 * 2 = 64
concat: effective global batch = 16 * 1 * 2 = 32
dual: effective global batch = 16 * 1 * 2 = 32
Đây là config tối ưu throughput, không công bằng tuyệt đối để so model vì global batch khác nhau.

DataLoader defaults:num_workers=4 mỗi process
pin_memory=true
persistent_workers=true
prefetch_factor=2

Gradient checkpointing:Mặc định đang tắt: gradient_checkpointing=false
Chỉ nên bật nếu OOM hoặc muốn thử batch lớn hơn.

Data split:Tỷ lệ: train/val/test = 80/10/10
Group bằng: frame_path
Hiện tại split được tạo trong runtime thành train_dataset, val_dataset, test_dataset và log split_stats.
Tôi chưa tạo 3 file JSON vật lý riêng và chưa ghi manifest split ra file. Đây là phần chưa implement đầy đủ so với plan nếu bạn muốn có artifact split cố định trên disk.

Lấy trực tiếp từ plan:accelerate path
không train LoRA
trainable chỉ trajectory branch + 2 bridge projections
gradient_checkpointing=false
split 80/10/10 theo frame_path
num_workers=4/process
token-weighted validation loss reduce
FlashAttention runtime verifier
checkpoint metadata/debug
debug grad norm, grad none, NaN/Inf
cls_add là default chính

Tôi tự chọn/default thêm:
cls_add batch_size=32
concat/dual batch_size=16
gradient_accumulation_steps=1
prefetch_factor=2
profile_steps=5
gradient_debug_steps=2
chỉ log tối đa 20 FlashAttention modules / gradient missing names để tránh log quá dài
global_optimizer_step được thêm vào training_state.json
Logic tự thêm/thay đổi/đơn giản hóa:Thêm fallback single-process nếu không chạy qua accelerate.
Rank phụ dùng SilentLogger để không ghi log/checkpoint cạnh tranh rank 0.
Resume vẫn theo epoch/update-step; không làm exact mid-epoch sampler/RNG resume.
Validation loss đổi từ mean theo batch sang token-weighted loss.
Checkpoint hiện có được giữ, chỉ thêm pretrain_run_metadata.json và global_optimizer_step.

Chưa implement đầy đủ / cần test server:Chưa benchmark thực tế trên 2x4090.
Chưa xác nhận FlashAttention thật sự bật trên Vast runtime.
Chưa tạo split manifest/file JSON cố định.
Chưa test OOM với batch 32/16/16.
Chưa verify distributed correctness bằng 2 GPU thật.
Chưa tạo VRAM-safe config riêng; hiện chỉ có speed baseline, nếu OOM thì sửa gradient_checkpointing=true.

File đã sửa/tạo:[train_pretrain.py](D:/NCKH_VLM/finetune-InternVL2/.worktrees/pretrain-server-setup/train_pretrain.py): thêm accelerate, rank-0 save/log, token-weighted val loss, FlashAttention verifier, optimizer/gradient debug, metadata checkpoint, global optimizer step.
[pretrain_dataset.py](D:/NCKH_VLM/finetune-InternVL2/.worktrees/pretrain-server-setup/pretrain_dataset.py): thêm split train/val/test theo frame_path, split stats, test dataset.
[internvl_pretrain_config_traj_cls.yaml](D:/NCKH_VLM/finetune-InternVL2/.worktrees/pretrain-server-setup/internvl_pretrain_config_traj_cls.yaml): server speed config cho cls_add, batch 32.
[internvl_pretrain_config_traj_concat.yaml](D:/NCKH_VLM/finetune-InternVL2/.worktrees/pretrain-server-setup/internvl_pretrain_config_traj_concat.yaml): server speed config cho concat, batch 16.
[internvl_pretrain_config_traj_dual.yaml](D:/NCKH_VLM/finetune-InternVL2/.worktrees/pretrain-server-setup/internvl_pretrain_config_traj_dual.yaml): server speed config cho dual, batch 16.
[tests/test_pretrain_dataset.py](D:/NCKH_VLM/finetune-InternVL2/.worktrees/pretrain-server-setup/tests/test_pretrain_dataset.py): test split 3-way và stats.
[tests/test_train_pretrain.py](D:/NCKH_VLM/finetune-InternVL2/.worktrees/pretrain-server-setup/tests/test_train_pretrain.py): test DataLoader kwargs, optimizer health, Flash verifier, metadata, token-weighted loss.
[2026-07-17-pretrain-server-2x4090-runbook.md](D:/NCKH_VLM/finetune-InternVL2/.worktrees/pretrain-server-setup/docs/research/2026-07-17-pretrain-server-2x4090-runbook.md): lệnh chạy server/smoke/resume/benchmark.

Command smoke test đầu tiên:
cd finetune-InternVL2
git checkout feature/pretrain-server-setup

accelerate launch --num_processes 2 train_pretrain.py \
  --config internvl_pretrain_config_traj_cls.yaml
Tôi đã verify local:
python -m py_compile train_pretrain.py pretrain_dataset.py
python -m pytest tests/test_pretrain_dataset.py tests/test_train_pretrain.py tests/test_pretrain_handoff.py tests/test_trajectory_branch.py tests/test_qformer_bridge_dual.py tests/test_trajectory_checkpoint_resolution.py -q
Kết quả: 47 passed.