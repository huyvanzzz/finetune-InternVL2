# Research chốt cấu hình chống overfit cho pretrain trajectory

Ngày: 2026-07-19  
Nhánh tham chiếu: `feature/pretrain-server-setup`  
Scope: pretrain-only, generation-only, InternVL-QFormer trajectory branch.

## Câu hỏi cần chốt

Cần chốt một cấu hình pretrain cuối cùng cho trajectory branch, ưu tiên chống overfit mạnh nhất có thể nhưng không làm branch random-init bị underfit. Không xét thêm KL/teacher/contrastive/matching loss trong vòng này.

Kết luận ngắn: dùng preset `anti_overfit_final` với regularization mạnh hơn hiện tại một chút ở `trajectory.dropout` và `weight_decay`, giữ LR hiện tại vì đang ở vùng an toàn, không tăng LR.

## Baseline repo hiện tại

`Repo-evidenced` Config hiện tại trong ba file `internvl_pretrain_config_traj_{cls,concat,dual}.yaml`:

| Trường | `cls_add` | `concat` | `dual` |
|---|---:|---:|---:|
| `trajectory_learning_rate` | `3e-4` | `3e-4` | `3e-4` |
| `bridge_learning_rate` | `5e-5` | `5e-5` | `5e-5` |
| `weight_decay` | `0.01` | `0.01` | `0.01` |
| `trajectory.dropout` | `0.05` | `0.05` | `0.05` |
| `loss_mode` | `label_smoothing` | `label_smoothing` | `label_smoothing` |
| `label_smoothing` | `0.10` | `0.10` | `0.10` |
| `warmup_ratio/min/max` | `0.1 / 20 / 100` | same | same |
| `num_epochs` | `50` | `50` | `50` |
| `early_stopping_patience` | `8` | `8` | `8` |
| `early_stopping_min_delta` | `0.001` | `0.001` | `0.001` |
| `batch_size` | `24` | `16` | `16` |
| `gradient_accumulation_steps` | `1` | `1` | `1` |

`Repo-evidenced` Runtime trainable policy trong `train_pretrain.py`:

- Frozen: vision encoder, Q-Former, Q-Former query tokens, `mlp1`, base LLM.
- Trainable: `qformer_input_proj`, `qformer_to_mlp1_proj`, `trajectory_backbone`, `trajectory_cls_head`, `trajectory_token_projector`.
- Không train LoRA ở pretrain.
- Loss là generation CE hoặc label-smoothed CE trên answer tokens.

## Evidence từ nguồn chính

| Nguồn | Evidence liên quan | Gần repo mình | Nhãn |
|---|---|---:|---|
| PyTorch `CrossEntropyLoss` docs: https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html | `label_smoothing` là tham số chính thức trong CE; phù hợp token-level generation loss hiện tại. | Trực tiếp ở implementation loss | `Paper/Docs-evidenced` |
| PyTorch `Dropout` docs: https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html | Dropout là regularization chuẩn, disable ở eval, tác động đúng vào trainable trajectory modules nếu đặt trong branch/head. | Trực tiếp ở `trajectory_branch.py` | `Docs-evidenced` |
| PyTorch `AdamW` docs: https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html | AdamW dùng decoupled weight decay; đây là regularization optimizer-level phù hợp khi train adapter/connector nhỏ. | Trực tiếp, repo dùng AdamW | `Docs-evidenced` |
| Hugging Face optimization docs: https://huggingface.co/docs/transformers/en/main_classes/optimizer_schedules | Có AdamW + cosine/warmup scheduler; warmup/cosine là pattern chuẩn cho Transformers. | Trực tiếp ở scheduler hiện tại | `Docs-evidenced` |
| BLIP-2 paper: https://arxiv.org/html/2301.12597 | Frozen image encoder + frozen LLM, train bridge/Q-Former để nối modality sang language. Đây là precedent kiến trúc cho frozen-backbone connector pretraining. | Gần | `Paper-evidenced` |
| LAVIS BLIP-2 stage 1 config: https://raw.githubusercontent.com/salesforce/LAVIS/main/lavis/projects/blip2/train/pretrain_stage1.yaml | Pretrain với `init_lr=1e-4`, `weight_decay=0.05`, cosine warmup, frozen ViT, 10 epoch. | Gần nhưng data lớn hơn nhiều | `Repo-evidenced` |
| LAVIS BLIP-2 stage 2 config: https://raw.githubusercontent.com/salesforce/LAVIS/main/lavis/projects/blip2/train/pretrain_stage2.yaml | Stage 2 generative pretraining tiếp tục dùng `init_lr=1e-4`, `weight_decay=0.05`, frozen ViT, warmup. | Gần hơn vì generation-to-LLM | `Repo-evidenced` |
| MiniGPT-4 stage 1 config: https://raw.githubusercontent.com/Vision-CAIR/MiniGPT-4/main/train_configs/minigpt4_stage1_pretrain.yaml | Train connector/alignment với `init_lr=1e-4`, `weight_decay=0.05`, cosine warmup, max epoch 4 trên data rất lớn. | Gần, nhưng scale data khác xa | `Repo-evidenced` |
| MiniGPT-4 stage 2 config: https://raw.githubusercontent.com/Vision-CAIR/MiniGPT-4/main/train_configs/minigpt4_stage2_finetune.yaml | Alignment/instruction stage dùng LR thấp hơn `3e-5`, `weight_decay=0.05`, warmup. | Gần về language alignment | `Repo-evidenced` |
| LLaVA v1.5 pretrain script: https://raw.githubusercontent.com/haotian-liu/LLaVA/main/scripts/v1_5/pretrain.sh | Projector-only pretrain dùng LR cao `1e-3`, `weight_decay=0`, 1 epoch, large-scale pretrain. | Gián tiếp; mục tiêu speed/alignment hơn anti-overfit | `Repo-evidenced` |
| MiniGPT-4 paper: https://arxiv.org/pdf/2304.10592 | Frozen visual encoder/Q-Former/LLM + train projection layer là precedent cho connector-only training. | Gần | `Paper-evidenced` |

## Diễn giải evidence

`Research-supported inference` Không có paper/repo nào trùng hoàn toàn bài toán này: trajectory structured branch, top-6 objects, 1 frame, QA ngắn, frozen InternVL + Q-Former bridge. Vì vậy số cụ thể phải là thiết kế có bằng chứng hỗ trợ, không thể gọi là paper-evidenced tuyệt đối.

`Research-supported inference` BLIP-2/MiniGPT-4 cho thấy connector/frozen-backbone pretraining thường dùng WD mạnh hơn `0.01`, thường là `0.05`, nhưng họ có dữ liệu lớn hơn rất nhiều và train Q-Former/projection trên image-text quy mô lớn. Với repo mình, đẩy thẳng `weight_decay=0.05` cộng `dropout=0.10` có nguy cơ underfit trajectory random-init.

`Research-supported inference` LLaVA dùng LR projector rất cao `1e-3` và `weight_decay=0`, nhưng đó là projector-only, data lớn, 1 epoch, không tối ưu cho chống overfit. Không nên lấy LLaVA làm lý do tăng LR trong run duy nhất.

`Research-supported inference` `trajectory_lr=3e-4` hiện tại không thấp đối với branch random-init 5-6M params. Nó cao hơn BLIP-2/MiniGPT LR chính `1e-4`, nhưng trainable branch nhỏ hơn và không update backbone lớn. Giữ `3e-4` là hợp lý hơn tăng LR.

`Research-supported inference` `bridge_lr=5e-5` là an toàn. Bridge projection nằm sát Q-Former/LLM seam; LR thấp hơn trajectory LR 6 lần giúp giảm rủi ro embedding drift.

## Quyết định preset cuối: `anti_overfit_final`

Đây là cấu hình nên dùng cho một run duy nhất.

| Trường | Giá trị cuối | Quyết định | Lý do |
|---|---:|---|---|
| `trajectory_learning_rate` | `3e-4` | Giữ | Đủ học cho random-init trajectory branch; tăng LR sẽ chống overfit kém hơn. |
| `bridge_learning_rate` | `5e-5` | Giữ | Bridge gần seam Q-Former/LLM, cần update chậm. |
| `learning_rate` | `1e-5` | Giữ | Không có trainable base/LoRA ở pretrain; chủ yếu là fallback group. |
| `weight_decay` | `0.02` | Tăng từ `0.01` | Siết hơn baseline nhưng không mạnh như `0.05` của BLIP/MiniGPT để tránh underfit. |
| `trajectory.dropout` | `0.10` | Tăng từ `0.05` | Chống co-adaptation tốt hơn cho branch 5-6M params và QA ngắn. |
| `loss_mode` | `label_smoothing` | Giữ | Phù hợp generation-only. |
| `label_smoothing` | `0.10` | Giữ | Đủ mạnh; tăng nữa dễ làm answer ngắn bị mềm quá mức. |
| `warmup_ratio` | `0.10` | Giữ | Chuẩn conservative cho Transformer/adapter training. |
| `warmup_min_steps` | `20` | Giữ | Tránh warmup quá ngắn nếu subset nhỏ. |
| `warmup_max_steps` | `100` | Giữ | Với 40k-55k samples, 100 warmup steps đủ và không ăn quá nhiều epoch đầu. |
| `lr_scheduler` | `cosine` | Giữ | Precedent rộng từ HF/LAVIS/LLaVA. |
| `num_epochs` | `50` | Giữ | Là trần trên, không phải bắt buộc train đủ. |
| `early_stopping_patience` | `8` | Giữ | User muốn tránh dừng sớm vì val noise; hợp với run duy nhất. |
| `early_stopping_min_delta` | `0.001` | Giữ | Vừa đủ tránh coi noise rất nhỏ là improvement. |
| `early_stopping_burn_in_epochs` | `1` | Giữ | Không dừng trước khi branch có cơ hội học. |
| `restore_best_checkpoint` | `true` | Giữ | Chốt artifact tốt nhất theo val loss. |
| `max_grad_norm` | `1.0` | Giữ | Safety chuẩn, tránh spike. |
| `batch_size cls_add` | `24` per GPU | Giữ | Đã phù hợp memory thực tế sau OOM ở batch lớn hơn. |
| `batch_size concat/dual` | `16` per GPU | Giữ | An toàn hơn vì token count 38. |
| `gradient_accumulation_steps` | `1` | Giữ | Throughput tốt trên 2x4090, không tăng regularization trực tiếp. |

## YAML khuyến nghị cuối

Áp dụng chung cho cả `cls_add`, `concat`, `dual`; chỉ khác `fusion_mode`, `output_dir`, và `batch_size`.

```yaml
trajectory:
  dropout: 0.10

training:
  num_epochs: 50
  learning_rate: 1e-5
  trajectory_learning_rate: 3e-4
  bridge_learning_rate: 5e-5
  warmup_ratio: 0.1
  warmup_min_steps: 20
  warmup_max_steps: 100
  weight_decay: 0.02
  loss_mode: "label_smoothing"
  label_smoothing: 0.10
  early_stopping_patience: 8
  early_stopping_min_delta: 0.001
  early_stopping_burn_in_epochs: 1
  restore_best_checkpoint: true
  lr_scheduler: "cosine"
  max_grad_norm: 1.0
  gradient_accumulation_steps: 1
```

Mode-specific:

```yaml
# cls_add
training:
  batch_size: 24

# concat / dual
training:
  batch_size: 16
```

## Vì sao không chọn cấu hình mạnh hơn

`Research-supported inference` Không chọn `trajectory.dropout=0.20`: dropout xuất hiện ở numeric MLP, object MLP, TransformerEncoder và heads. Với structured object features, dropout quá cao có thể phá tín hiệu bbox/movement vốn đã ít chiều.

`Research-supported inference` Không chọn `label_smoothing=0.15-0.20`: answer ngắn, nhiều answer dạng direction/object name. Smoothing quá mạnh làm model bớt tự tin quá mức cần thiết.

`Research-supported inference` Không chọn `weight_decay=0.05`: có precedent ở BLIP-2/MiniGPT-4, nhưng scale data của họ lớn hơn nhiều. Với trajectory branch random-init và QA ngắn, `0.02` là điểm cân bằng tốt hơn cho một run duy nhất.

`Research-supported inference` Không tăng `trajectory_lr` lên `5e-4`: tăng LR có thể học nhanh hơn nhưng đi ngược mục tiêu chống overfit. `3e-4` hiện đã không thấp.

`Research-supported inference` Không thêm KL/teacher/auxiliary loss: không có precedent trực tiếp đủ mạnh cho trajectory QA pretrain hiện tại; thêm loss sẽ mở thêm biến số và khó quy lỗi nếu run duy nhất fail.

## Kết luận cuối

`Design choice for v1` Preset cuối nên là:

- `trajectory_learning_rate=3e-4`
- `bridge_learning_rate=5e-5`
- `weight_decay=0.02`
- `trajectory.dropout=0.10`
- `loss_mode=label_smoothing`
- `label_smoothing=0.10`
- `warmup_ratio=0.1`, `warmup_min_steps=20`, `warmup_max_steps=100`
- `num_epochs=50`
- `early_stopping_patience=8`
- `early_stopping_min_delta=0.001`
- `max_grad_norm=1.0`
- `batch_size=24` cho `cls_add`, `16` cho `concat/dual`
- `gradient_accumulation_steps=1`

`Open gap` Không có bằng chứng trực tiếp từ paper cho đúng bài toán trajectory object QA một frame. Tuy vậy, preset trên là lựa chọn tốt nhất từ evidence hiện có vì nó kết hợp: frozen-backbone connector precedent, regularization chuẩn cho generation, và ràng buộc runtime thật của repo.

