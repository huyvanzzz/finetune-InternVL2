# Thiết kế nhánh `trajectory branch` cho `InternVL-qformer` theo hướng `concat-token`

## 1. Mục tiêu của tài liệu

Tài liệu này chốt riêng nhánh `trajectory branch` theo hướng:

- giữ đầu ra cuối là **chuỗi token**
- output chính là `traj_tokens`
- shape mục tiêu là `(B, 6, D)`
- dùng để **nối thêm token** vào downstream visual tokens
- path chính **không dùng CLS**

Phạm vi đã khóa:

- chỉ xét `InternVL-qformer`
- chưa code trong đợt này
- vẫn phải tương thích với:
  - `finetune-only`
  - `pretrain riêng -> finetune tiếp`

---

## 2. Code truth cần giữ

### 2.1. `InternVL-qformer` hiện có 32 token visual sau bridge

`Repo-evidenced`

- Bridge hiện tại sinh `32` query tokens qua Q-Former.
- Sau `qformer_to_mlp1_proj -> mlp1`, model nhận:

```text
qformer_tokens_llm: (B, 32, llm_hidden)
```

- Các token này sẽ thay thế dải `<IMG_CONTEXT>` trong input embeddings.

### 2.2. Vì sao concat ở đây là tự nhiên

`Repo-evidenced` + `Research-supported inference`

- Nếu muốn nối thêm trajectory token mà không viết lại LLM, seam sạch nhất là:
  - giữ nguyên 32 token của Q-Former
  - sinh thêm 6 token trajectory ở cùng hidden size
  - `concat` theo chiều token

Ví dụ:

```text
(B, 32, D) + (B, 6, D) -> (B, 38, D)
```

### 2.3. `TrackingMeetsLMM` không trực tiếp làm concat-token

`Paper-evidenced` và `Repo-evidenced`

- Repo gốc encode nhiều token rồi cuối cùng rút về `CLS`.
- Nghĩa là `TrackingMeetsLMM` không phải precedent trực tiếp cho output token-level cuối cùng.

Nhưng repo/paper đó vẫn hỗ trợ hai điểm:

- giữ nhiều object/token trong quá trình encode là hợp lý
- trajectory branch có thể tách riêng rồi mới fuse ở cuối

---

## 3. Backbone chung với nhánh `CLS-token`

## 3.1. Input contract

Giống hoàn toàn nhánh `CLS-token`:

```text
label_ids      : (B, 6)
numeric_feats  : (B, 6, 10)
object_mask    : (B, 6)
```

Feature số:

- `cx, cy, w, h`
- `area`
- `distance_norm`
- `direction_weight`
- `sin(movement_angle), cos(movement_angle)`
- `speed_percent`

## 3.2. Sort và padding

`Design choice for v1`

- sort theo priority rank trước khi lấy 6 slot
- pad zero cho slot thiếu
- `object_mask` đánh dấu slot thật/rỗng

## 3.3. Backbone chung khuyến nghị

Giống nhánh `CLS-token`:

1. `label_id -> embedding`, `d_cat = 32`
2. `numeric_feats -> MLP 10 -> 64 -> 64`
3. `concat(cat, numeric) -> object MLP -> 128 -> 128`
4. `slot embedding` cho 6 rank slot
5. `set encoder` nhỏ:
   - `d_traj = 128`
   - `2 layer`
   - `4 heads`
   - `ffn_dim = 256`

Output backbone chung:

```text
traj_tokens_base: (B, 6, 128)
```

Kết luận:

- nhánh concat không cần backbone riêng
- nó dùng đúng backbone chung rồi chỉ thay head/fusion cuối

---

## 4. Thiết kế riêng cho hướng `concat-token`

## 4.1. Path chính không dùng `CLS`

`Design choice for v1`

Path chính chốt như sau:

```text
traj_tokens_base: (B, 6, 128)
-> token projector
-> traj_tokens: (B, 6, d_fuse)
```

Không thêm `CLS` vào path chính.

Lý do:

- người dùng đã chốt rằng hướng concat phải giữ token-level information
- thêm `CLS` vào path chính sẽ làm mờ ranh giới giữa hai hướng

## 4.2. Token projector cuối

`Design choice for v1`

Vì downstream của `InternVL-qformer` làm việc ở hidden size của LLM, nên cần projector:

```text
128 -> llm_hidden
```

Khuyến nghị head cuối:

```text
LayerNorm
-> Linear(128, llm_hidden)
```

Nếu muốn mềm hơn có thể:

```text
LayerNorm
-> Linear(128, llm_hidden)
-> GELU
-> Linear(llm_hidden, llm_hidden)
```

Nhưng v1 nên ưu tiên bản nhẹ hơn để tiết kiệm thời gian.

## 4.3. Điểm concat khuyến nghị

`Design choice for v1`

Concat nên diễn ra **sau khi Q-Former đã sinh xong 32 token và sau `mlp1`**, vì đây là lúc cả hai nhánh đều có thể ở cùng hidden size của LLM.

Luồng khuyến nghị:

```text
qformer path -> qformer_tokens_llm: (B, 32, D)
trajectory path -> traj_tokens: (B, 6, D)
concat(dim=1)
-> visual_tokens_all: (B, 38, D)
```

Sau đó:

- cập nhật số `<IMG_CONTEXT>` tương ứng
- chèn cả 38 token vào input embeddings

### 4.4. Vì sao không concat ở `pre-mlp1 seam`

`Research-supported inference`

- Nhánh concat cần token-level output dễ hiểu và dễ điều khiển.
- Nếu concat ở `pre-mlp1 seam`, trajectory token vẫn còn ở không gian `pixel_shuffle_dim`, chưa thật sự là token downstream-ready.
- Concat sau `mlp1` sạch hơn nhiều:
  - cùng hidden size
  - cùng dạng token với qformer output
  - dễ debug hơn

---

## 5. Precedent research cho hướng giữ token-level output

### 5.1. `Slot-VLM`

`Paper-evidenced`

- `Slot-VLM` nhấn mạnh việc giữ **semantically decomposed tokens** rồi đưa chúng vào LLM.
- Paper cũng nêu BLIP-2/Q-Former là một bottleneck cố định 32 token.

Điểm liên quan:

- precedent tốt cho giả thuyết:
  - khi cần reasoning giàu object-level detail, giữ token-level output là hợp lý

### 5.2. `Dense Connector`

`Paper-evidenced`

- `Dense Connector` cho thấy tăng số token có thể đem lại thông tin phong phú hơn, nhưng token count tăng sẽ kéo theo overhead.

Điểm liên quan:

- precedent tốt cho tradeoff:
  - concat token thường giàu thông tin hơn
  - nhưng latency/compute sẽ tăng hơn path global

### 5.3. Kết luận research cho nhánh concat

`Research-supported inference`

- Không thấy precedent nào trùng hẳn với:
  - top-6 tracked objects
  - 1 frame
  - concat ra sau Q-Former của InternVL
- Nhưng có precedent đủ tốt cho ba ý:
  - giữ token-level representation
  - dùng connector/projector mỏng
  - chấp nhận tăng token count có kiểm soát để lấy thêm semantic detail

---

## 6. Setup `finetune-only` cho nhánh `concat-token`

### 6.1. Module trainable khuyến nghị

Giống nhánh `CLS-token`, cộng thêm head riêng:

- LoRA của language model
- `qformer_input_proj`
- `qformer_to_mlp1_proj`
- trajectory backbone chung:
  - label embedding
  - numeric MLP
  - object MLP
  - slot embedding
  - set encoder
- `traj_token_projector`

Giữ frozen:

- vision encoder
- Q-Former gốc
- `mlp1`

### 6.2. Vì sao setup này hợp lý

`Research-supported inference`

- trajectory backbone cần trainable để học semantics riêng
- projector cần trainable để map sang hidden space LLM
- nhưng không cần mở vision/Q-Former/`mlp1`, nên chi phí vẫn được kiểm soát

---

## 7. Setup `pretrain riêng -> finetune tiếp`

## 7.1. Objective pretrain chính

`Design choice for v2`

Nhánh concat vẫn nên dùng **cùng objective chính** với nhánh `CLS-token`:

- `trajectory-only pseudo-instruction generation`

Nhưng ở đây fusion path của pretrain nên bám đúng branch concat:

```text
traj_tokens_base
-> traj_token_projector
-> concat vào token path downstream
-> sinh pseudo target
```

Lý do:

- tránh mismatch giữa pretrain path và finetune path
- trajectory học luôn cách tồn tại như một chuỗi token thay vì bị ép thành global token rồi đổi kiến trúc ở finetune

## 7.2. Objective fallback

`Design choice for v2`

Fallback:

- `trajectory token alignment` với qformer visual token hoặc hidden summary từ image branch

Ví dụ:

- align từng `traj_token` với teacher token summary theo cosine/L2
- hoặc dùng pooling nhẹ làm teacher alignment

Fallback này dễ setup hơn, nhưng vẫn yếu hơn objective sinh pseudo guidance.

## 7.3. Checkpoint lưu gì

Lưu:

- trajectory backbone chung
- `traj_token_projector`

Khi sang finetune:

- load lại trajectory weights
- bật train tiếp trajectory + LoRA + hai projector qformer hiện có

---

## 8. Tradeoff hiệu năng

## 8.1. Vì sao nhánh concat nặng hơn nhánh `CLS-token`

`Research-supported inference`

- nhánh `CLS-token` vẫn giữ `32` visual tokens
- nhánh concat tăng thành `38` token

Điều này kéo theo:

- attention của downstream LLM side tăng
- thời gian train tăng hơn nhánh `CLS-token`
- thời gian test cũng tăng hơn nhánh `CLS-token`

### 8.2. Nhưng mức tăng vẫn kiểm soát được

`Research-supported inference`

- tăng từ `32` lên `38` token không phải bùng nổ theo kiểu quay về `256` token của no-qformer
- trajectory backbone được giữ nhỏ:
  - `d_traj = 128`
  - `2 layer`
  - `4 heads`

Kết luận:

- nhánh concat đắt hơn nhánh `CLS-token`
- nhưng vẫn ở mức “hợp lý để thử”, nhất là nếu mục tiêu là giữ object-level detail

### 8.3. Điểm mạnh chính của nhánh concat

- giữ riêng 6 object token đến tận seam cuối
- phù hợp hơn nếu kỳ vọng trajectory phải bổ sung chi tiết cụ thể:
  - object nào
  - hướng nào
  - motion nào

### 8.4. Điểm yếu chính

- sequence dài hơn
- dễ tăng thời gian train/test hơn path `CLS-token`
- nếu dữ liệu trajectory còn nhiễu, 6 token riêng có thể mang cả nhiễu xuống downstream

---

## 9. Kết luận chốt cho nhánh `concat-token`

### 9.1. Thiết kế chốt

```text
top-6 objects
-> label embedding + numeric MLP
-> object MLP
-> set encoder
-> traj_tokens_base: (B, 6, 128)
-> traj_token_projector
-> traj_tokens: (B, 6, llm_hidden)

qformer path
-> 32 token ở hidden space LLM

concat
-> (B, 38, llm_hidden)
-> chèn vào dải IMG_CONTEXT
```

### 9.2. Vai trò của nhánh concat

`Design choice for v1`

Nhánh này không phải hướng rẻ nhất, nhưng là hướng có **trần biểu diễn cao hơn** vì không ép trajectory về một token toàn cục quá sớm.

### 9.3. Khuyến nghị thực dụng

- Nếu ưu tiên **ít tăng thời gian**: làm `CLS-token` trước.
- Nếu muốn thử hướng có khả năng giữ object-level detail tốt hơn: làm `concat-token` sau.

