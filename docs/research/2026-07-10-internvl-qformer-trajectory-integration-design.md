# Thiết kế tổng hợp `trajectory branch` cho `InternVL-qformer`

## 1. Mục tiêu của bộ tài liệu

Tài liệu này gom lại toàn bộ kết luận research/design để sau đó có thể implement `trajectory branch` cho `InternVL-qformer` mà không phải quyết định lại kiến trúc từ đầu.

Phạm vi đã khóa:

- chỉ xét `InternVL-qformer`
- có đủ hai hướng:
  - `CLS-token / additive`
  - `concat-token`
- có đủ hai chế độ huấn luyện:
  - `finetune-only`
  - `pretrain riêng -> finetune tiếp`

Mục tiêu tối thượng:

- chất lượng tốt hơn
- nhưng tham số tăng thêm và thời gian train/test tăng ở mức vừa phải

---

## 2. Code truth phải giữ

## 2.1. `InternVL-qformer` hiện tại

`Repo-evidenced`

Luồng hiện tại:

```text
ViT
-> pixel shuffle
-> qformer_input_proj
-> Q-Former (32 query tokens)
-> qformer_to_mlp1_proj
-> mlp1
-> 32 token ở hidden space LLM
-> thay thế IMG_CONTEXT embeddings
```

Ý nghĩa:

- Q-Former hiện tại đang là bottleneck `32 token`
- mọi thay đổi trajectory nên tôn trọng seam này nếu muốn chi phí thấp

## 2.2. `TrackingMeetsLMM`

`Paper-evidenced` và `Repo-evidenced`

Repo/paper gốc cho thấy:

- trajectory được encode bằng branch riêng
- trong encoder vẫn học trên chuỗi token
- cuối branch có `modality head`
- output cuối của branch gốc là **một vector/token toàn cục**
- paper còn có:
  - `self-supervised pretraining strategy`
  - `automated annotation pipeline`

Kết luận:

- precedent mạnh cho nhánh `CLS-token`
- precedent gián tiếp cho việc dùng pretrain riêng trước khi finetune

## 2.3. Precedent ngoài repo gốc

`Paper-evidenced`

- `BLIP-2`: Q-Former là bottleneck nhẹ, có hai giai đoạn pretrain.
- `Slot-VLM`: semantically decomposed tokens hữu ích khi cần reasoning ở mức object/event.
- `Dense Connector`: tăng token có thể tăng chất lượng nhưng cũng tăng overhead; connector mỏng là hướng thực dụng.

Kết luận tổng:

- `CLS-token` có precedent gần hơn
- `concat-token` có precedent gián tiếp tốt nếu muốn giữ object-level detail

---

## 3. Backbone trajectory chung

## 3.1. Input contract

Mỗi sample:

- tối đa `6 object`
- mỗi object có:
  - `label_id`
  - `cx, cy, w, h`
  - `area`
  - `distance_norm`
  - `direction_weight`
  - `sin(movement_angle), cos(movement_angle)`
  - `speed_percent`

Tensor:

```text
label_ids      : (B, 6)
numeric_feats  : (B, 6, 10)
object_mask    : (B, 6)
```

## 3.2. Sort và thiếu object

`Design choice for v1`

- sort object theo priority rank trước khi lấy 6 slot
- slot rỗng được zero-pad
- `object_mask` đánh dấu slot thật/rỗng
- dùng `slot embedding` cho rank `1..6`

## 3.3. Backbone khuyến nghị

`Design choice for v1`

```text
label_id -> Embedding(d_cat=32)
numeric_feats(10) -> MLP(10->64->64)
concat(cat, numeric) -> object MLP(96->128->128)
add slot embedding
-> set encoder nhỏ (2 layer, 4 heads, d_traj=128, ffn=256)
-> traj_tokens_base: (B, 6, 128)
```

Đây là phần **dùng chung** cho cả hai hướng.

---

## 4. Hướng 1: `CLS-token / additive`

## 4.1. Luồng chốt

```text
traj_tokens_base: (B, 6, 128)
-> traj_cls_head
-> traj_cls: (B, 1, pixel_shuffle_dim)
-> broadcast add vào 32 qformer tokens ở pre-mlp1 seam
-> mlp1
-> 32 token visual cuối
```

## 4.2. Vì sao chọn `pre-mlp1 seam`

`Design choice for v1`

So với `post-mlp1 seam`, `pre-mlp1 seam` được khuyến nghị mạnh hơn vì:

- không tăng token length
- sạch hơn về code path
- gần tinh thần `TrackingMeetsLMM` hơn
- chi phí train/test thấp hơn

## 4.3. Điểm mạnh và điểm yếu

Điểm mạnh:

- gọn
- rẻ
- dễ cắm vào code hiện tại

Điểm yếu:

- nén trajectory về 1 token khá sớm
- có thể mất bớt object-level detail

---

## 5. Hướng 2: `concat-token`

## 5.1. Luồng chốt

```text
traj_tokens_base: (B, 6, 128)
-> traj_token_projector
-> traj_tokens: (B, 6, llm_hidden)

qformer path
-> qformer_tokens_llm: (B, 32, llm_hidden)

concat
-> (B, 38, llm_hidden)
-> thay thế IMG_CONTEXT embeddings
```

## 5.2. Path chính không dùng `CLS`

`Design choice for v1`

- nhánh concat chính bỏ hẳn `CLS`
- nếu cần global token thì chỉ giữ như một ablation riêng, không phải path chính

## 5.3. Điểm mạnh và điểm yếu

Điểm mạnh:

- giữ được 6 object token đến seam cuối
- phù hợp hơn nếu trajectory cần bổ sung chi tiết cụ thể

Điểm yếu:

- sequence dài hơn
- train/test chậm hơn hướng `CLS-token`

---

## 6. So sánh decision-ready giữa hai hướng

| Mục | `CLS-token` | `concat-token` |
|---|---|---|
| Output cuối của trajectory branch | `traj_cls: (B, 1, D)` | `traj_tokens: (B, 6, D)` |
| Dùng chung backbone | Có | Có |
| Tăng token length downstream | Không | Có, từ `32 -> 38` |
| Gần `TrackingMeetsLMM` | Gần hơn | Xa hơn |
| Giữ object-level detail | Thấp hơn | Cao hơn |
| Độ sạch khi implement v1 | Cao | Trung bình |
| Rủi ro tăng train/test time | Thấp hơn | Cao hơn |

Kết luận thực dụng:

- `CLS-token` là hướng nên implement trước
- `concat-token` là hướng nên giữ như variant thứ hai để thử trần chất lượng cao hơn

---

## 7. Setup `finetune-only`

## 7.1. Hướng ưu tiên code trước

`Design choice for v1`

Ưu tiên code trước:

- `InternVL-qformer + trajectory CLS-token + pre-mlp1 additive seam`

Lý do:

- ít thay đổi nhất lên code hiện tại
- không tăng token count
- compute tăng vừa phải

## 7.2. Module trainable trong `finetune-only`

Giữ trainable như hiện tại:

- LoRA của language model
- `qformer_input_proj`
- `qformer_to_mlp1_proj`

Thêm trainable:

- trajectory backbone chung
- head riêng của từng hướng:
  - `traj_cls_head` hoặc
  - `traj_token_projector`

Giữ frozen:

- vision encoder
- Q-Former gốc
- `mlp1`

---

## 8. Setup `pretrain riêng -> finetune tiếp`

## 8.1. Objective pretrain chính

`Design choice for v2`

Objective chính được khuyến nghị:

- `trajectory-only pseudo-instruction generation`

Mục tiêu:

- buộc trajectory branch học semantic gần với `alter`
- ví dụ học:
  - obstacle chính ở đâu
  - đường clear hay blocked
  - nên đi tiếp, chậm lại hay tránh

## 8.2. Objective fallback

`Design choice for v2`

Nếu pseudo-text khó dựng sạch, objective fallback là:

- `trajectory-to-visual-token alignment`

Ví dụ:

- align trajectory output với teacher visual summary từ qformer/image branch
- dùng `L2 + cosine`

## 8.3. Module trainable ở pha pretrain

Khuyến nghị:

- train trajectory backbone chung
- train head riêng của từng hướng
- train seam projector liên quan

Giữ frozen:

- vision encoder
- Q-Former
- `mlp1`
- language model
- LoRA

## 8.4. Khi sang finetune sẽ load gì

Load lại:

- trajectory backbone chung
- head riêng
- seam projector

Sau đó bật finetune:

- trajectory modules
- LoRA
- `qformer_input_proj`
- `qformer_to_mlp1_proj`

---

## 9. Đánh giá chi phí train/test

## 9.1. `finetune-only`

`Research-supported inference`

- `CLS-token`:
  - tăng train time nhẹ
  - tăng test time nhẹ
- `concat-token`:
  - tăng train time nhiều hơn `CLS-token`
  - tăng test time nhiều hơn `CLS-token`
  - nhưng vẫn rẻ hơn nhiều so với quay lại 256 visual tokens

## 9.2. `pretrain -> finetune`

`Research-supported inference`

- pha pretrain riêng có thể rẻ hơn full finetune vì:
  - không mở LoRA
  - không mở Q-Former
  - chỉ học trajectory branch
- nhưng tổng wall-clock của cả pipeline chắc chắn dài hơn `finetune-only`

Kết luận:

- nếu cần hướng triển khai nhanh và an toàn: làm `finetune-only` trước
- nếu trajectory branch chứng minh được giá trị: mới mở thêm pha `pretrain -> finetune`

---

## 10. Khuyến nghị cuối cùng

### 10.1. Nên implement hướng nào trước

`Design choice for v1`

**Khuyến nghị mạnh:** implement trước:

1. backbone trajectory chung
2. nhánh `CLS-token`
3. seam `pre-mlp1 additive`
4. `finetune-only`

### 10.2. Giữ hướng nào làm bước hai

Sau khi nhánh đầu ổn, bước hai nên là:

1. tái dùng backbone chung
2. thêm `concat-token`
3. đo xem object-level detail có giúp chất lượng đủ nhiều để bù phần latency tăng không

### 10.3. Vì sao đây là thứ tự hợp lý nhất

- đúng mục tiêu “chất lượng tốt nhưng không tăng thời gian quá nhiều”
- bám khá gần precedent của `TrackingMeetsLMM`
- vẫn mở sẵn đường cho hướng `concat-token`
- không làm hai nhánh tách thành hai hệ khác nhau

