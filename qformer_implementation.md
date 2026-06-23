# Chi tiết triển khai Q-Former Bridge trong dự án finetune-InternVL2

## 1. Tổng quan kiến trúc

Dự án tích hợp **Q-Former từ InstructBLIP** vào **InternVL2** thông qua một module gọi là `QFormer Bridge`. Đây là một **Prompt-Aware Visual Feature Compressor** — thay vì đưa thẳng hàng trăm visual token từ ViT vào LLM, Q-Former nén chúng thành **32 query token cố định**, được conditioning bởi text prompt.

### So sánh Pipeline: Gốc vs. Có Q-Former

````carousel
**Pipeline gốc của InternVL2 (không có Q-Former):**

```
Ảnh đầu vào (448×448)
        │
   [ViT Encoder] (frozen)
        │  last_hidden_state: [N_tiles, 1025, 1024]
        │  bỏ CLS token → [N_tiles, 1024, 1024]
        │
   [pixel_shuffle] (downsample_ratio=0.5)
        │  → [N_tiles, 256, 4096]  ← pixel_shuffle_dim
        │
   [mlp1] (frozen): Linear(4096 → 2048) + GELU + Linear(2048 → 2048)
        │  → [N_tiles, 256, 2048]  ← llm_hidden_size
        │
   Ghép vào input embeddings của LLM
   (mỗi tile = 256 visual tokens)
```

Với 1 ảnh = 2 tiles → **512 visual tokens** vào LLM.
<!-- slide -->
**Pipeline có Q-Former Bridge (`bridge_mode: prompt_aware_preproj_mlp1`):**

```
Ảnh đầu vào (448×448)          Text Prompt (câu hỏi)
        │                               │
   [ViT Encoder] (frozen)       [Q-Former Tokenizer]
        │  [N_tiles, 1024, 1024]        │  [N_tiles, seq_len]
        │                               │
   [pixel_shuffle] (frozen)             │
        │  [N_tiles, 256, 4096]         │
        │                               │
   [qformer_input_proj] (trainable!)    │
     LayerNorm(4096) + Linear(4096→768) │
        │  [N_tiles, 256, 768]          │
        │                               │
        └─────────[Q-Former]────────────┘
                (Salesforce/instructblip-flan-t5-xl, frozen)
                  query_tokens: [N_tiles, 32, 768]
                  encoder_hidden_states = vit_proj
                  Cross-Attention điều kiện bởi text
                        │
                  query_output: [N_tiles, 32, 768]
                        │
           [qformer_to_mlp1_proj] (trainable!)
             LayerNorm(768) + Linear(768→4096)
                        │  [N_tiles, 32, 4096]
                        │
                   [mlp1] (frozen)
             Linear(4096→2048) + GELU + Linear(2048→2048)
                        │  [N_tiles, 32, 2048]
                        │
           Ghép vào input embeddings của LLM
           (mỗi tile = 32 visual tokens thay vì 256!)
```

Với 1 ảnh = 2 tiles → chỉ **64 visual tokens** vào LLM (giảm 8×).
````

---

## 2. Các chiều tensor chi tiết

| Bước | Tensor shape | Ghi chú |
|---|---|---|
| Input ảnh | `[N_tiles, 3, 448, 448]` | N_tiles = số tile sau dynamic preprocess |
| ViT output (sau bỏ CLS) | `[N_tiles, 1024, 1024]` | 1024 patches, hidden_size=1024 |
| Sau pixel_shuffle | `[N_tiles, 256, 4096]` | `pixel_shuffle_dim = 1024 × (1/0.5)²` = 4096 |
| Sau `qformer_input_proj` | `[N_tiles, 256, 768]` | chiều về qformer_encoder_dim=768 |
| query_tokens (expand) | `[N_tiles, 32, 768]` | 32 learnable query tokens |
| Q-Former output | `[N_tiles, 32, 768]` | Cross-Attn giữa query và encoder_hidden_states |
| Sau `qformer_to_mlp1_proj` | `[N_tiles, 32, 4096]` | lên lại pixel_shuffle_dim |
| Sau `mlp1` | `[N_tiles, 32, 2048]` | llm_hidden_size=2048 |

---

## 3. Cấu trúc module (trong `qformer_bridge.py`)

### 3.1 `attach_qformer_bridge(model, config)` — Khởi tạo

```python
# Tải Q-Former từ Salesforce/instructblip-flan-t5-xl
qformer, query_tokens, qformer_tokenizer, blip_config = _load_qformer_from_source(
    "Salesforce/instructblip-flan-t5-xl", cache_dir="./qformer_cache"
)

# Tính các chiều
pixel_shuffle_dim  = vit_hidden_size * (1/downsample_ratio)²  # 1024 * 4 = 4096
qformer_encoder_dim = blip_config.qformer_config.encoder_hidden_size  # 768
qformer_hidden_size = blip_config.qformer_config.hidden_size           # 768

# Gắn các thành phần vào model
model.qformer             = qformer           # frozen
model.qformer_query_tokens = query_tokens[:, :32, :]  # frozen
model.qformer_tokenizer    = qformer_tokenizer

# Hai projection layer DUY NHẤT được train
model.qformer_input_proj = nn.Sequential(
    nn.LayerNorm(4096),
    nn.Linear(4096, 768),
)  # float32

model.qformer_to_mlp1_proj = nn.Sequential(
    nn.LayerNorm(768),
    nn.Linear(768, 4096),
)  # float32

# Override extract_feature bằng version có Q-Former
model.extract_feature = MethodType(_extract_feature_with_qformer, model)
# Thêm helper methods
model.encode_qformer_texts = MethodType(_encode_qformer_texts, model)
model.set_qformer_text     = MethodType(_set_qformer_text, model)
model.clear_qformer_text   = MethodType(_clear_qformer_text, model)

# Quan trọng: ghi đè num_image_token
model.num_image_token = 32  # thay cho 256 gốc
```

> [!IMPORTANT]
> `model.num_image_token = 32` là thay đổi **then chốt**. Con số này được dùng trong `CollaterFn` để biết phải chèn bao nhiêu `<IMG_CONTEXT>` token vào prompt text. Nếu số này sai → shape mismatch khi ghép visual embeddings.

### 3.2 `_extract_feature_with_qformer(self, pixel_values)` — Forward Pass

Hàm này **thay thế** `extract_feature` gốc của `InternVLChatModel`:

```python
def _extract_feature_with_qformer(self, pixel_values):
    # Bước 1: ViT + pixel_shuffle (giống gốc, không qua mlp1)
    vit_embeds = _extract_vit_tokens(self, pixel_values)
    # shape: [N_tiles, 256, 4096]

    # Bước 2: Chiếu vào không gian Q-Former
    _ensure_bridge_device(self, vit_embeds)  # sync device/dtype
    encoder_hidden_states = self.qformer_input_proj(vit_embeds.to(float32))
    # shape: [N_tiles, 256, 768]

    encoder_attention_mask = torch.ones([N_tiles, 256])  # tất cả attended

    # Bước 3: Lấy text conditioning từ state (đặt trước bằng set_qformer_text)
    qformer_input_ids      = self._qformer_input_ids       # [N_tiles, seq_len]
    qformer_attention_mask = self._qformer_attention_mask  # [N_tiles, seq_len]
    # Nếu không set → dùng empty string ""

    # Bước 4: Expand query tokens
    query_tokens = self.qformer_query_tokens.expand(N_tiles, -1, -1)
    # shape: [N_tiles, 32, 768]

    # Ghép attention mask: [query_mask | text_mask]
    query_attention_mask = torch.ones([N_tiles, 32])
    qformer_attention_mask = cat([query_attention_mask, qformer_attention_mask], dim=1)
    # shape: [N_tiles, 32+seq_len]

    # Bước 5: Q-Former cross-attention
    query_outputs = self.qformer(
        input_ids=qformer_input_ids,
        attention_mask=qformer_attention_mask,  # full attention over query+text
        query_embeds=query_tokens,              # learnable queries
        encoder_hidden_states=encoder_hidden_states,  # visual as keys/values
        encoder_attention_mask=encoder_attention_mask,
    )
    query_output = query_outputs[0][:, :32, :]  # chỉ lấy query positions
    # shape: [N_tiles, 32, 768]

    # Bước 6: Project ngược về pixel_shuffle_dim rồi qua mlp1
    mlp1_inputs = self.qformer_to_mlp1_proj(query_output.to(float32))
    # shape: [N_tiles, 32, 4096]
    return self.mlp1(mlp1_inputs.to(mlp1_dtype))
    # shape: [N_tiles, 32, 2048]  ← đây là final visual embeddings
```

### 3.3 Text Conditioning Flow (Prompt-Aware)

Q-Former cần biết text prompt **trước** khi gọi `extract_feature`. Đây là flow trong `train.py`:

```
CollaterFn.__call__()
    │
    ├── for each sample: lấy question text
    │       qformer_text = sample.get("qformer_text", question)
    │       # = câu hỏi không có tag <image>
    │       qformer_texts.extend([qformer_text] * total_tiles)
    │       # nhân theo số tiles!
    │
    └── model.encode_qformer_texts(qformer_texts)
            → (input_ids, attention_mask) lưu vào qformer_inputs
                    │
                    ▼
train_model() loop:
    model.set_qformer_text(qformer_inputs[0].cuda(), qformer_inputs[1].cuda())
    outputs = model(pixel_values=..., input_ids=..., ...)
        └── model.forward() gọi self.extract_feature(pixel_values)
                └── _extract_feature_with_qformer() đọc self._qformer_input_ids
    model.clear_qformer_text()
```

> [!WARNING]
> Text được nhân theo **số tiles** (total_tiles), không phải số ảnh. Mỗi tile xử lý độc lập qua Q-Former với cùng 1 text. Nếu `qformer_input_ids.shape[0] != N_tiles` → ValueError ngay.

---

## 4. Trainable vs Frozen Parameters

| Component | Trạng thái | Lý do |
|---|---|---|
| **ViT Encoder** (`vision_model`) | ❄️ Frozen | Tốn VRAM, feature đã đủ tốt |
| **Q-Former** (`qformer`) | ❄️ Frozen | Pretrained cross-attn từ InstructBLIP |
| **query_tokens** | ❄️ Frozen | Pretrained từ InstructBLIP |
| **mlp1** | ❄️ Frozen | Không train để tránh overfit |
| **`qformer_input_proj`** | 🔥 **Trainable** | Cầu nối ViT → Q-Former |
| **`qformer_to_mlp1_proj`** | 🔥 **Trainable** | Cầu nối Q-Former → mlp1 |
| **LLM LoRA adapters** | 🔥 **Trainable** | r=16, α=32, target: wqkv/wo/w1/w2/w3 |
| **LLM base weights** | ❄️ Frozen (4-bit) | NF4 quantization |

Tổng params trainable rất nhỏ: chỉ ~**2 projection layers + LoRA**

---

## 5. Save/Load Checkpoint

### Save (cuối mỗi epoch hoặc save_steps)

```python
# LoRA adapter
model.language_model.save_pretrained(save_dir)
# → adapter_config.json, adapter_model.safetensors

# Q-Former bridge (chỉ 2 proj layers)
save_qformer_bridge(model, save_dir)
# → qformer_bridge.safetensors  (chỉ qformer_input_proj + qformer_to_mlp1_proj)
# → qformer_bridge_config.json  (metadata)

# Optimizer + Scheduler
torch.save(optimizer.state_dict(), "optimizer.pt")
torch.save(lr_scheduler.state_dict(), "scheduler.pt")
```

### Load Resume

```python
# Tải lại LoRA
model.language_model = PeftModel.from_pretrained(model.language_model, resume_dir, is_trainable=True)

# Tải lại bridge weights
load_qformer_bridge(model, resume_dir, strict=True)
align_qformer_bridge_runtime(model)  # đồng bộ device/dtype
```

> [!NOTE]
> Q-Former weights gốc (frozen) **không được lưu** vào checkpoint — chúng luôn được tải lại từ HuggingFace Hub (`Salesforce/instructblip-flan-t5-xl`). Chỉ 2 projection layers được lưu.

---

## 6. Device & Dtype Management

Đây là phần phức tạp do mix nhiều dtype:

| Module | Dtype |
|---|---|
| ViT | `bfloat16` |
| Q-Former | `bfloat16` (sync với ViT) |
| `qformer_input_proj` | `float32` (tránh overflow khi train) |
| `qformer_to_mlp1_proj` | `float32` |
| query_tokens | `bfloat16` |
| mlp1 | `bfloat16` |
| LLM | 4-bit (NF4) + compute bfloat16 |

Hàm `_ensure_bridge_device(model, reference_tensor)` được gọi mỗi forward để sync Q-Former, query_tokens về cùng device/dtype với ViT output. Proj layers giữ float32 để tránh underflow trong gradient.

---

## 7. Vấn đề tiềm ẩn & Điểm cần chú ý

> [!CAUTION]
> **Vấn đề 1: `num_image_token` bị ghi đè**  
> `attach_qformer_bridge()` set `model.num_image_token = 32`. Nhưng trong `InternVLChatModel.__init__()`, giá trị gốc được tính là `(448/14)² × 0.5² = 256`. Việc ghi đè này ảnh hưởng toàn bộ tokenization pipeline (số `<IMG_CONTEXT>` tokens).

> [!WARNING]
> **Vấn đề 2: Text phải khớp số tile**  
> `qformer_texts.extend([qformer_text] * total_tiles)` — text phải được nhân đúng theo `total_tiles`, không phải số ảnh. Nếu dùng `num_frames > 1`, cần cẩn thận về cách đếm total_tiles.

> [!NOTE]
> **Vấn đề 3: mlp1 frozen nhưng vẫn nhận gradient**  
> Khi `freeze_mlp1=True`, mlp1 không train. Nhưng `qformer_to_mlp1_proj` → `mlp1` vẫn có gradient flow **qua** mlp1 để tính gradient cho proj. Điều này bình thường.

> [!TIP]
> **`align_qformer_bridge_runtime()`** nên được gọi sau mỗi lần move model sang device mới. Nó đồng bộ Q-Former, proj layers, query_tokens về cùng device với `mlp1` hoặc `vision_model`.

---

## 8. Flow đầy đủ từ ảnh đến loss

```
[1 sample từ WAD Dataset]
  frame_path → last_frame_id → PIL Image
        │
  [process_image(img)]  ← data.py
    dynamic_preprocess → N_tiles crops (448×448)
    transform → [N_tiles, 3, 448, 448] tensor
        │
  pixel_values = [process_image(img)]  ← list of tensors

[CollaterFn.__call__()]
  question = "<image>\nDescribe the scene..."
  → tokenize → input_ids (với NUM_IMAGE_TOKEN=32 IMG_CONTEXT tokens/tile)
  → qformer_text = "Describe the scene..."
  → encode_qformer_texts([qformer_text] * N_tiles) → (q_ids, q_mask)
  → pixel_values_batch = cat(pixel_values)  # [total_tiles, 3, 448, 448]

[train_model() loop]
  model.set_qformer_text(q_ids, q_mask)
  outputs = model(
      input_ids=...,        # [B, seq_len] với 32 IMG_CONTEXT per tile
      pixel_values=...,     # [total_tiles, 3, 448, 448]
      labels=...,           # -100 cho input, token_id cho answer
      image_flags=...,
  )
      │
      ├── input_embeds = LLM.get_input_embeddings()(input_ids)
      │
      ├── vit_embeds = model.extract_feature(pixel_values)
      │       = _extract_feature_with_qformer(pixel_values)
      │       → [total_tiles, 32, 2048]
      │
      ├── Ghép: input_embeds[selected==IMG_CONTEXT_TOKEN] = vit_embeds.reshape(-1, 2048)
      │   (32 × total_tiles positions được thay bằng visual embeddings)
      │
      └── LLM(inputs_embeds=...) → logits → CrossEntropyLoss với labels
  
  model.clear_qformer_text()
  loss.backward()
```
