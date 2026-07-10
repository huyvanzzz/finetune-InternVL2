# Thiết kế nhánh `trajectory branch` cho `InternVL-qformer` theo hướng `CLS-token`

## 1. Mục tiêu của tài liệu

Tài liệu này chốt riêng nhánh `trajectory branch` theo hướng:

- đầu ra cuối chỉ còn **đúng 1 token** `traj_cls`
- shape mục tiêu là `(B, 1, D)`
- token này dùng cho **additive conditioning**
- ưu tiên giữ chi phí train/test tăng thêm ở mức thấp

Phạm vi đã khóa:

- chỉ xét `InternVL-qformer`
- chưa code trong đợt này
- vẫn phải thiết kế sao cho tương thích với hai chế độ:
  - `finetune-only`
  - `pretrain riêng -> finetune tiếp`

`Design choice for v1`

- Hai hướng `CLS-token` và `concat-token` sẽ **dùng chung một trajectory backbone**.
- Nhánh này chỉ khác ở **global head** và **điểm chèn cuối**.

---

## 2. Code truth cần khóa trước khi thiết kế

### 2.1. `InternVL-qformer` hiện tại đang chạy thế nào

`Repo-evidenced`

- Ở [qformer_bridge.py](D:/NCKH_VLM/finetune-InternVL2/qformer_bridge.py), bridge hiện tại override `extract_feature`.
- Luồng hiện tại là:

```text
ViT tokens
-> pixel shuffle
-> qformer_input_proj
-> Q-Former
-> qformer_to_mlp1_proj
-> mlp1
-> 32 token ở hidden space của LLM
```

- `num_query_tokens` hiện tại là `32`.
- Sau khi attach bridge, `model.num_image_token = 32`.
- `mlp1` vẫn là lớp cuối để đưa token visual sang hidden size của language model.

`Repo-evidenced`

- Ở [train.py](D:/NCKH_VLM/finetune-InternVL2/train.py), số `<IMG_CONTEXT>` token được sinh theo `self.model.num_image_token`.
- Vì vậy, nếu giữ đầu ra là `32` token thì không phải đổi logic prompt placeholder.
- Nếu tăng số token, downstream prompt/token budget sẽ tăng tương ứng.

### 2.2. `TrackingMeetsLMM` thực sự làm gì

`Paper-evidenced` và `Repo-evidenced`

- `TrackingMeetsLMM` dùng trajectory encoder riêng.
- Sau trunk transformer vẫn còn **nhiều token**.
- Ở `modality_heads['traj']`, repo gốc mới:
  - `LayerNorm`
  - `SelectElement(index=0)`
  - `Dropout`
  - `Linear`
- Nghĩa là repo gốc giữ lại **CLS token** ở cuối branch để tạo embedding toàn cục.

Kết luận quan trọng:

- Tinh thần repo gốc là:
  - trong encoder vẫn học quan hệ nhiều object/token
  - nhưng output cuối của trajectory branch là **một token toàn cục**
- Đây là precedent rất gần cho hướng `CLS-token`.

### 2.3. Vì sao cần phân biệt rõ `trước mlp1` và `sau qformer output`

`Repo-evidenced`

Trong repo hiện tại, có hai seam hợp lý nếu muốn cộng trajectory vào luồng `qformer`:

1. **Pre-mlp1 seam**
   - sau `qformer_to_mlp1_proj`
   - trước `mlp1`
   - tensor đang ở không gian `pixel_shuffle_dim`

2. **Post-mlp1 seam**
   - sau `mlp1`
   - tensor đã ở hidden space của LLM

Tài liệu này sẽ chốt seam khuyến nghị mạnh cho hướng `CLS-token`.

---

## 3. Backbone trajectory dùng chung cho cả hai hướng

## 3.1. Input contract

`Repo-evidenced` từ các báo cáo trước và dữ liệu hiện tại

Mỗi sample trajectory dùng tối đa `6 object`, mỗi object có:

- `label_id`
- `cx, cy, w, h`
- `area`
- `distance_norm`
- `direction_weight`
- `sin(movement_angle), cos(movement_angle)`
- `speed_percent`

Tensor đầu vào:

```text
label_ids      : (B, 6)
numeric_feats  : (B, 6, 10)
object_mask    : (B, 6)
```

### 3.2. Quy tắc thiếu object

`Design choice for v1`

- Nếu ít hơn `6 object`:
  - pad zero cho `label_ids`
  - pad zero cho `numeric_feats`
  - `object_mask = 0` ở slot rỗng
- Không sinh số object động ở v1.

Lý do:

- dễ batching
- dễ giữ compute cố định
- tương thích với cả `finetune-only` và `pretrain`

### 3.3. Quy tắc sort trước khi vào 6 slot

`Design choice for v1`

- Trước khi đưa vào encoder, object phải được sắp theo **thứ tự ưu tiên cố định**.
- Thứ tự này không phải “positional theo trái-phải”, mà là **priority rank**.
- Priority score upstream có thể dùng từ file trajectory chuẩn bị trước:
  - diện tích bbox chuẩn hóa
  - khoảng cách
  - hướng ưu tiên gần `12h`

Kết luận:

- slot `0` luôn là object ưu tiên cao nhất
- slot `1` là object ưu tiên tiếp theo
- ...
- slot rỗng bị mask

### 3.4. Có dùng slot embedding không

`Research-supported inference`

- Vì đầu vào đã được sort theo priority rank, slot index mang nghĩa “mức ưu tiên”.
- Do đó nên dùng **slot embedding nhẹ** cho 6 vị trí này.

`Design choice for v1`

- dùng `slot_embedding: (6, d_traj)`
- đây là embedding cho **thứ hạng sau sort**, không phải positional encoding của ảnh

### 3.5. Kiến trúc backbone chung

`Design choice for v1`

Khuyến nghị backbone chung:

1. **Category branch**
   - `label_id -> nn.Embedding`
   - `d_cat = 32`

2. **Numeric branch**
   - `numeric_feats (10-dim) -> MLP`
   - khuyến nghị:

```text
10 -> 64 -> 64
```

3. **Object fusion**
   - nối `cat_emb` và `numeric_emb`
   - qua `object MLP`

```text
(32 + 64) -> 128 -> 128
```

4. **Set encoder**
   - input: `object_tokens + slot_embedding`
   - output: `traj_tokens_base`
   - khuyến nghị:
     - `d_traj = 128`
     - `2 encoder layer`
     - `4 attention heads`
     - `ffn_dim = 256`
     - dropout nhẹ `0.1`

Output trung gian chung:

```text
traj_tokens_base: (B, 6, 128)
```

### 3.6. Vì sao dùng MLP thay vì linear ở numeric/object fusion

`Research-supported inference`

- `Linear` thuần quá yếu cho phần trộn các feature số dị loại.
- Numeric branch cần học:
  - tương tác trong cùng object
  - ví dụ `w*h`, `distance` + `direction`, `speed` + `angle`
- Vì vậy `MLP` nhỏ là hợp lý hơn `Linear` ở hai chỗ:
  - `numeric_feats -> numeric_emb`
  - `concat(cat, numeric) -> object_token`

---

## 4. Thiết kế riêng cho hướng `CLS-token`

### 4.1. Nhánh này lấy `traj_cls` từ đâu

`Design choice for v1`

Để giữ backbone chung với nhánh `concat-token`, không nên tách hẳn một trunk riêng chỉ cho `CLS`.

Thay vào đó:

```text
traj_tokens_base: (B, 6, 128)
-> traj_cls_head
-> traj_cls: (B, 1, d_head)
```

Khuyến nghị `traj_cls_head`:

1. `LayerNorm` trên `traj_tokens_base`
2. một `learned cls query` kích thước `(1, 1, 128)`
3. một block attention nhẹ:
   - query là `traj_cls_query`
   - key/value là `traj_tokens_base`
4. `MLP`
5. projector cuối sang dim của seam cần chèn

Kết luận:

- `traj_cls` vẫn là một token toàn cục
- nhưng backbone 6 object token vẫn được tái dùng nguyên cho nhánh concat

### 4.2. Vì sao không bắt buộc mang `CLS` xuyên cả trunk

`Repo-evidenced` + `Design choice for v1`

- `TrackingMeetsLMM` thật sự có `CLS` trong trunk rồi mới chọn ở head.
- Nhưng ở repo này, yêu cầu mạnh hơn là:
  - hai hướng dùng chung backbone
  - path concat chính không dùng `CLS`

Vì vậy, dùng `traj_cls_head` sau backbone là một nhượng bộ kiến trúc hợp lý:

- vẫn giữ tinh thần “một token toàn cục”
- nhưng tránh phải duy trì hai trunk khác nhau

### 4.3. Dim đầu ra của `traj_cls`

Có hai lựa chọn đúng với hai seam:

1. Nếu chèn ở **pre-mlp1 seam**:

```text
traj_cls: (B, 1, pixel_shuffle_dim)
```

2. Nếu chèn ở **post-mlp1 seam**:

```text
traj_cls: (B, 1, llm_hidden_size)
```

---

## 5. Chốt điểm chèn khuyến nghị cho hướng `CLS-token`

## 5.1. Ứng viên A: `pre-mlp1 seam`

Luồng:

```text
qformer output
-> qformer_to_mlp1_proj
-> qformer_tokens_pre_mlp1: (B, 32, pixel_shuffle_dim)

traj_tokens_base
-> traj_cls_head
-> traj_cls_pre_mlp1: (B, 1, pixel_shuffle_dim)

broadcast add:
qformer_tokens_pre_mlp1 + traj_cls_pre_mlp1
-> mlp1
-> (B, 32, llm_hidden)
```

Ưu điểm:

- không tăng số token downstream
- không phải sửa placeholder từ `32` thành số lớn hơn
- trajectory enriches visual query **trước** khi vào không gian LLM
- gần tinh thần `TrackingMeetsLMM` hơn: enrich branch trước khi bơm sang LMM
- compute tăng rất ít

Nhược điểm:

- trajectory phải học vào không gian `pixel_shuffle_dim`, ít “language-aligned” hơn ngay từ đầu

## 5.2. Ứng viên B: `post-mlp1 seam`

Luồng:

```text
qformer output
-> qformer_to_mlp1_proj
-> mlp1
-> qformer_tokens_llm: (B, 32, llm_hidden)

traj_tokens_base
-> traj_cls_head
-> traj_cls_llm: (B, 1, llm_hidden)

broadcast add:
qformer_tokens_llm + traj_cls_llm
```

Ưu điểm:

- trajectory được chèn thẳng trong hidden space của LLM
- dễ nghĩ hơn về alignment ngôn ngữ

Nhược điểm:

- can thiệp muộn hơn vào pipeline
- xa hơn tinh thần “enrich visual query before connector”
- dễ đụng nhiều hơn tới semantics của token đã qua `mlp1`

## 5.3. Khuyến nghị chốt cho v1

`Design choice for v1`

**Khuyến nghị mạnh:** dùng **`pre-mlp1 seam`** cho nhánh `CLS-token`.

Lý do:

- sạch hơn về integration
- không tăng token length
- gần `TrackingMeetsLMM` hơn ở tinh thần fusion
- chi phí train/test thấp nhất trong hai seam

`Fallback`

- giữ `post-mlp1 seam` làm phương án phụ nếu sau này thấy `pre-mlp1` bị yếu về language alignment

---

## 6. Setup `finetune-only` cho nhánh `CLS-token`

Đây là hướng ưu tiên code trước.

### 6.1. Module trainable khuyến nghị

`Design choice for v1`

Giữ nguyên phần trainable hiện có của `InternVL-qformer`:

- LoRA của language model
- `qformer_input_proj`
- `qformer_to_mlp1_proj`

Thêm trainable cho trajectory:

- `label embedding`
- `numeric MLP`
- `object MLP`
- `slot embedding`
- `set encoder`
- `traj_cls_head`

Giữ frozen:

- vision encoder
- Q-Former gốc
- `mlp1`

### 6.2. Vì sao setup này hợp lý

`Research-supported inference`

- trajectory là modality mới, nên backbone trajectory phải trainable
- nhưng để tránh thời gian tăng quá nhiều:
  - không mở vision encoder
  - không mở Q-Former gốc
  - không mở `mlp1`
- chỉ học trajectory backbone + seam projector + LoRA hiện có

### 6.3. Tác động hiệu năng kỳ vọng

`Research-supported inference`

- So với `InternVL-qformer` hiện tại:
  - train time tăng **nhẹ đến vừa**
  - test time tăng **nhẹ**
- Tăng thêm chủ yếu ở:
  - `numeric MLP`
  - `object MLP`
  - `2-layer set encoder`
  - `traj_cls_head`
- Vì vẫn giữ `32` token downstream, phần LLM-side gần như không phình ra.

---

## 7. Setup `pretrain riêng -> finetune tiếp`

## 7.1. Mục tiêu của pha pretrain

`Paper-evidenced` từ `TrackingMeetsLMM`

- Paper gốc nói rõ:
  - có `self-supervised pretraining strategy`
  - có `automated annotation pipeline`
  - mục tiêu là làm tracking encoder mang thêm context trước khi finetune task cuối

`Design choice for repo hiện tại`

- Với repo này, pretrain nên làm cho trajectory branch học:
  - object importance
  - direction
  - distance
  - movement
  - safety/path-state cues
- và học theo dạng **text-aligned** chứ không chỉ regression thuần số.

## 7.2. Objective pretrain chính

`Design choice for v2`

**Objective chính được khuyến nghị:**  
`trajectory-only pseudo-instruction generation`

Ý tưởng:

- từ tracking/top-6 object tạo ra pseudo prompt + pseudo target tự động
- pseudo target ưu tiên gần kiểu `alter` ngắn:
  - hướng nào cần chú ý
  - object nào là đáng chú ý nhất
  - đường có clear hay blocked
  - có nên đi tiếp / chậm lại / tránh

Ví dụ loại supervision:

- `What is the most relevant obstacle and where is it?`
- `Is the path clear or blocked?`
- `What short guidance should be given now?`

Module trainable ở pha này:

- trajectory backbone chung
- `traj_cls_head`
- seam projector cho `pre-mlp1`

Giữ frozen:

- vision encoder
- Q-Former
- `mlp1`
- language model
- LoRA

Lý do:

- đây là cách rẻ nhất để ép trajectory branch học semantic có thể dùng cho text
- đồng thời không làm pretrain biến thành một đợt finetune đầy đủ

## 7.3. Objective fallback

`Design choice for v2`

Nếu pseudo text còn nhiễu hoặc khó dựng nhanh, objective fallback là:

- `trajectory-to-visual-token alignment`

Ví dụ:

- dùng qformer visual tokens của cùng frame làm teacher cố định
- học cho `traj_cls` dự đoán embedding toàn cục tương ứng
- có thể dùng `L2 + cosine alignment`

Objective này an toàn hơn về dữ liệu, nhưng yếu hơn objective chính ở mặt language alignment.

## 7.4. Sang pha finetune sẽ load lại gì

`Design choice for v2`

Checkpoint pretrain nên lưu:

- trajectory backbone chung
- `traj_cls_head`
- seam projector

Khi sang finetune:

- load lại toàn bộ trajectory weights
- bật train tiếp:
  - trajectory modules
  - LoRA
  - `qformer_input_proj`
  - `qformer_to_mlp1_proj`

Kết luận:

- backbone trajectory được tái dùng nguyên
- không cần tách thành một hệ khác với `finetune-only`

---

## 8. Đánh giá hiệu năng và độ rủi ro

## 8.1. Vì sao nhánh `CLS-token` là hướng rẻ hơn

`Research-supported inference`

- output cuối vẫn chỉ là `32` visual tokens của Q-Former
- trajectory chỉ thêm một global bias token rồi broadcast-add
- không tăng sequence length đi vào LLM

Vì vậy:

- train latency tăng ít hơn nhánh concat
- test latency cũng tăng ít hơn nhánh concat

## 8.2. Hidden size khuyến nghị để không đội compute

`Design choice for v1`

- `d_cat = 32`
- `d_num = 64`
- `d_traj = 128`
- `2 encoder layer`, `4 heads`

Đây là mức đủ để học object interaction mà vẫn gọn hơn nhiều so với mở thêm nhánh lớn kiểu vision encoder thứ hai.

## 8.3. Rủi ro chính

- `pre-mlp1 seam` có thể hơi yếu nếu trajectory semantic khó map vào không gian pre-mlp1
- `traj_cls` có thể nén mất một phần object-level detail

Vì vậy:

- nhánh `CLS-token` hợp nhất cho **v1 nhanh, rẻ, sạch**
- nhưng có thể không phải trần chất lượng cao nhất

---

## 9. Kết luận chốt cho nhánh `CLS-token`

### 9.1. Thiết kế chốt

```text
top-6 objects
-> label embedding + numeric MLP
-> object MLP
-> set encoder
-> traj_tokens_base: (B, 6, 128)
-> traj_cls_head
-> traj_cls_pre_mlp1: (B, 1, pixel_shuffle_dim)
-> broadcast add vào 32 qformer tokens ở pre-mlp1 seam
-> mlp1
-> 32 visual tokens vào LLM
```

### 9.2. Khuyến nghị thực dụng

`Design choice for v1`

Nếu ưu tiên:

- thời gian train/test không tăng nhiều
- sửa code ít
- vẫn bám khá gần tinh thần `TrackingMeetsLMM`

thì nhánh **`CLS-token + pre-mlp1 additive seam`** là hướng nên implement đầu tiên.

### 9.3. Vai trò của pretrain

`Design choice for v2`

- `finetune-only` vẫn là baseline nên làm trước
- `pretrain riêng -> finetune tiếp` nên xem là bước tăng chất lượng sau
- objective chính nên là `trajectory-only pseudo-instruction generation`

