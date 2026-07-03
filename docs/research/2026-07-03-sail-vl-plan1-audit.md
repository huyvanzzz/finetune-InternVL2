# Báo cáo Audit: Repo Hiện Tại Và Nền Tảng Cho `SAIL-VL-1.5-2B`

## Mục tiêu

Tài liệu này tổng hợp kết quả của `Plan 1`:

- đọc kỹ repo hiện tại để chốt `semantic contract` của nhánh `InternVL`
- xác định ranh giới để thêm model mới mà không làm hỏng `InternVL`
- research `SAIL-VL-1.5-2B` từ nguồn chính thống
- đối chiếu hai bên để xác định mức độ sẵn sàng cho `Plan 2`

Tài liệu này **chưa** chốt:

- config train cuối cho `SAIL`
- kiến trúc `Q-Former` cuối cho `SAIL`
- cách tổ chức file/code cuối
- hyperparameter / LoRA setup cuối

Lưu ý quan trọng:

- phạm vi hiện tại là `alter-only`
- các phần liên quan `QA` được xem là ngoài phạm vi quyết định của tài liệu này

---

## 1. Repo hiện tại đang chạy theo semantic contract nào?

### 1.1 Contract đầu vào hiện tại

**Repo evidence**

- `wad_dataset.py` chọn frame qua `_select_frames_safe(...)`, nhưng trong `__getitem__` chỉ lấy `last_frame_id = frame_ids[-1]` và load duy nhất frame cuối ([wad_dataset.py](../../wad_dataset.py)).
- `question` luôn được build theo dạng `"<image>\n{text_content}"`.
- `qformer_text` luôn là `text_content.strip()`.
- `answer` được map bởi `format_ground_truth(...)` trong [preprocessing.py](../../preprocessing.py).

### 1.2 Prompt contract hiện tại cho `alter`

**Repo evidence**

- Prompt cũ được giữ trong `ALTER_DIRECT_TEXT_LEGACY_PROMPT`.
- Prompt mới gồm 4 template `T1..T4` trong `ALTER_DIRECT_TEXT_TEMPLATES`.
- `balanced_v1` hoạt động như sau:
  - `train`: gán 4 prompt theo phân bố cân bằng trên tập `alter`
  - `val/test`: cố định về `T1`
- Config hiện tại ở cả hai YAML đều để:
  - `response_format: "direct_text"`
  - `direct_text_alter_prompt_mode: "balanced_v1"`
  - `train_task_filter: "alter_only"`
  - `val_task_filter: "alter_only"`

### 1.3 Supervision contract hiện tại cho `alter`

**Repo evidence**

- Trong [preprocessing.py](../../preprocessing.py), nếu không có `QA` thì `instruction = metadata['alter']`.
- Với setup hiện tại, `response_format == "direct_text"`, nên target train cuối cùng chính là `instruction`.
- Vì train/val hiện tại đều `alter_only`, model thực chất đang học trực tiếp text `alter`.

### 1.4 Output contract hiện tại ở train và infer

**Train**

- `train.py` đưa `question` vào conversation template, chèn visual tokens vào vị trí `<image>`, rồi supervise phần `answer` sau prompt.
- Với `qformer`, `qformer_text` được repeat theo số tile rồi đưa vào `model.encode_qformer_texts(...)`.

**Infer**

- `scripts/test_infer.py` dùng cùng semantic prompt:
  - lấy `sample['question']`
  - nếu bật `qformer`, lấy `sample['qformer_text']`
  - generate ra text cuối cùng
- Metrics khi `response_format == "direct_text"` được tính trên `raw_text`.

### 1.5 Sơ đồ semantic pipeline hiện tại

```text
raw metadata sample
-> chọn frame cuối
-> process_image(frame cuối) -> pixel_values
-> build question text
-> build qformer_text
-> build answer text
-> collate chèn IMG_CONTEXT vào prompt chat
-> model forward / generate
-> output text cuối
-> metrics trên raw_text
```

### 1.6 Bảng contract hiện tại

| Lớp | Contract hiện tại |
| --- | --- |
| Ảnh đầu vào | Frame cuối của sample |
| Prompt chính | `question = <image> + text_content` |
| Prompt visual-side | `qformer_text = text_content.strip()` |
| Alter train | 4 prompt balanced |
| Alter val/test | cố định `T1` |
| Target direct_text | `alter` |
| Eval output | text tự nhiên, không JSON |

---

## 2. Contract nào bắt buộc phải giữ để so sánh công bằng giữa `InternVL` và `SAIL`?

### 2.1 Những gì phải giống ở mức semantic

Để so sánh công bằng, các phần sau nên được giữ giống nhau:

- cùng nguồn metadata
- cùng split train/val/test
- cùng rule chọn **frame cuối**
- cùng `response_format`
- cùng logic build `question`
- cùng logic build `qformer_text`
- cùng prompt family/mode cho `alter`
- cùng target supervision `alter`
- cùng metrics và split eval

### 2.2 Những gì không cần ép giống

Không nên ép giống các chi tiết nội bộ sau:

- preprocess tensor-level giống hệt `InternVL`
- số tile chính xác giống hệt
- chat template native bên trong model
- visual token count nội bộ
- cách projector native của model hoạt động

### 2.3 Kết luận quan trọng về fairness

`Công bằng` ở đây nên được định nghĩa là:

- **same raw sample + same semantic prompt/answer contract + same eval policy**
- **model-native preprocess / projector / chat internals được phép khác**

Nếu ép `SAIL` dùng lại toàn bộ preprocess tensor của `InternVL`, ta có nguy cơ làm sai kiến trúc gốc của `SAIL`.

---

## 3. Boundary contract để không phá `InternVL`

### 3.1 Phần nào đang dùng chung được

Có thể tái sử dụng ý tưởng hoặc contract từ các phần sau:

- task typing theo `alter`
- task filtering train/val
- prompt mode cho `direct_text + alter`
- answer mapping trong [preprocessing.py](../../preprocessing.py)
- metrics pipeline
- resume/debug utilities
- test split policy

### 3.2 Phần nào đang hardcode cho `InternVL`

**Repo evidence**

- `train.py` hardcode:
  - `IMG_START_TOKEN = "<img>"`
  - `IMG_END_TOKEN = "</img>"`
  - `IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"`
  - `SYSTEM_MESSAGE = "You are a navigation assistant for visually impaired users."`
- `CollaterFn` trong [train.py](../../train.py) build prompt dựa trên `model.template`, chèn image token theo `self.model.num_image_token`.
- `wad_dataset.py` đang gọi trực tiếp `process_image(...)` từ [data.py](../../data.py), nghĩa là dataset hiện tại đã gắn preprocess với `InternVL-like tensor path`.
- `qformer_bridge.py` patch trực tiếp `extract_feature`, `encode_qformer_texts`, `set_qformer_text`, `clear_qformer_text` lên model.
- `internvl_config.yaml` và `internvl_config_no_qformer.yaml` đang mang LoRA target modules theo `InternVL2.5`:
  - `wqkv`, `wo`, `w1`, `w2`, `w3`

### 3.3 Coupling points cần tránh sửa trực tiếp ở Plan 2

Những điểm sau là coupling points:

- [wad_dataset.py](../../wad_dataset.py): preprocess ảnh đang gắn chặt với `process_image`
- [train.py](../../train.py): collate + load model + LoRA target modules + chat token insertion
- [scripts/test_infer.py](../../scripts/test_infer.py): infer runtime đang gắn với `InternVL`
- [qformer_bridge.py](../../qformer_bridge.py): bridge hiện tại giả định `InternVL-like pixel_shuffle -> mlp1`
- [model/conversation.py](../../model/conversation.py): prompt template của `InternVL`

### 3.4 Do-not-touch paths cho Plan 2

Plan 2 nên coi các đường sau là “không sửa để generic hóa sớm”:

- `train.py`
- `scripts/test_infer.py`
- `qformer_bridge.py`
- `model/`

### 3.5 Kết luận boundary

**Đã xác minh đủ chắc**

- repo hiện tại chưa ở dạng multi-backend
- luồng `InternVL` đang có nhiều assumption ăn vào model family
- nếu muốn thêm `SAIL` mà không vỡ nhánh cũ, hướng an toàn nhất là `tách nhánh riêng`

---

## 4. Audit riêng `qformer` / `no-qformer` hiện tại

### 4.1 `Q-Former` runtime contract hiện tại

**Repo evidence**

- `qformer_enabled(config)` đọc từ `model.qformer.enabled`.
- Mặc định:
  - `freeze_qformer = true`
  - `freeze_mlp1 = true`
  - `prompt_aware = true`
  - `num_query_tokens = 32`
- `attach_qformer_bridge(...)` tạo:
  - `qformer_input_proj`
  - `qformer_to_mlp1_proj`
  - patch `extract_feature`

Luồng thực tế:

```text
vision_model
-> pixel_shuffle
-> qformer_input_proj
-> Q-Former (có text instruction)
-> qformer_to_mlp1_proj
-> mlp1
```

### 4.2 Instruction vào `Q-Former` bằng đường nào?

**Repo evidence**

- `qformer_text` được tạo ngay ở dataset side.
- `CollaterFn` repeat `qformer_text` theo tổng số tile.
- `model.encode_qformer_texts(...)` mã hóa text.
- `model.set_qformer_text(...)` gắn text token vào model trước forward/generate.

=> `Q-Former` hiện tại là **instruction-aware** thật sự, không phải chỉ là visual compressor.

### 4.3 Projection nào đang trainable?

Theo `attach_qformer_bridge(...)`:

- `qformer_input_proj`: trainable
- `qformer_to_mlp1_proj`: trainable
- `qformer`: frozen theo config hiện tại
- `qformer_query_tokens`: frozen nếu `freeze_qformer=true`
- `mlp1`: frozen nếu `freeze_mlp1=true`

### 4.4 `no-qformer` runtime contract

Nếu `model.qformer.enabled = false`:

- không attach bridge
- `qformer_text` có thể vẫn tồn tại trong sample, nhưng train/infer runtime không dùng đến
- visual path native của model giữ nguyên

### 4.5 `LoRA` contract hiện tại

**Repo evidence**

- `train.py` gọi `prepare_model_for_kbit_training(model.language_model)`
- sau đó bọc `model.language_model` bằng PEFT LoRA
- `InternVL` target modules hiện tại:
  - `wqkv`, `wo`, `w1`, `w2`, `w3`

### 4.6 Invariant cần giữ cho `SAIL`

Nếu sau này muốn `SAIL qformer` có cùng semantic behavior, các invariant sau nên được giữ:

- `question` và `qformer_text` phải cùng một semantic prompt
- `qformer_text` phải là instruction text thật sự, không được để rỗng mặc định nếu đang train instruction-aware
- `direct_text` target vẫn là final answer text
- `train alter` vẫn dùng multi-prompt, `val/test alter` vẫn có prompt cố định để comparable
- `LoRA` vẫn nên bọc vào language model side, còn bridge là trainable projector-side

---

## 5. `SAIL-VL-1.5-2B` nói gì từ nguồn chính thống?

### 5.1 Fact từ model card / config / remote code

**Source evidence**

- Hugging Face model card nói rõ:
  - `SAIL-VL-1.5-2B = SAILViT-Huge + Qwen2.5-1.5B + 2-layer MLP + 2x2 token merge`
- `config.json` cho thấy:
  - `architectures = ["SailVLModel"]`
  - `model_type = "internvl_chat"`
  - `template = "sailvl-chat"`
  - `downsample_ratio = 0.5`
  - `dynamic_image_size = true`
  - `force_image_size = 448`
  - `use_thumbnail = true`
  - `min_dynamic_patch = 1`
  - `max_dynamic_patch = 12`
  - `vision_config.hidden_size = 1536`
  - `llm_config.architectures = ["Qwen2ForCausalLM"]`
  - `llm_config.hidden_size = 1536`

### 5.2 Fact từ native visual path

**Source evidence**

- `modeling_sailvl.py` cho thấy:

```text
vision_model
-> pixel_shuffle
-> mlp1
-> vit_embeds
```

- `extract_feature(...)` của `SAIL` dùng:
  - `vision_model`
  - `pixel_shuffle`
  - `mlp1`

=> `SAIL` có một visual seam rất rõ ràng trước khi đưa embeddings vào language model.

### 5.3 Fact từ native chat path

**Source evidence**

- `chat(...)` của `SAIL`:
  - nếu cần, tự thêm `<image>\n`
  - dùng `get_conv_template(self.template)`
  - chèn `<img> + <IMG_CONTEXT> * num_image_token + </img>`
- `generate(...)` của `SAIL` scatter visual embeddings vào vị trí `IMG_CONTEXT_TOKEN` trong input embeddings.

=> Về mặt runtime prompt/image token, `SAIL` gần `InternVL` hơn rất nhiều so với `Qwen2-VL`.

### 5.4 Fact từ README sử dụng native

**Source evidence**

- README ghi rõ:
  - “The basic usage and dynamic crop strategy of SAIL-VL follows InternVL2”
- Sample code:
  - `dynamic_preprocess(... max_num=10, image_size=448, use_thumbnail=False)`
  - `load_image(... use_thumbnail=True, max_num=10)`
  - `AutoModel.from_pretrained(... trust_remote_code=True)`
  - `AutoTokenizer.from_pretrained(... trust_remote_code=True, use_fast=False)`

### 5.5 Fact từ conversation template

**Source evidence**

- `conversation.py` của `SAIL` register template `sailvl-chat`
- native `system_message` của template là tiếng Trung

### 5.6 Fact liên quan `LoRA`

**Source evidence**

- `modeling_qwen2.py` của `SAIL` có naming chuẩn `Qwen2`:
  - `q_proj`, `k_proj`, `v_proj`, `o_proj`
  - `gate_proj`, `up_proj`, `down_proj`

=> Nếu làm LoRA native cho `SAIL`, khả năng cao target modules sẽ cần theo naming `Qwen2`, không thể tái sử dụng nguyên bộ `InternVL` targets hiện tại.

---

## 6. `SAIL` khớp đến đâu với contract của repo?

### 6.1 Bảng compatibility matrix

| Hạng mục | Repo hiện tại cần giữ | `SAIL` có khớp tự nhiên? | Ghi chú |
| --- | --- | --- | --- |
| Chọn frame cuối | Có | Có thể giữ | Dataset-level, không phụ thuộc model |
| `question` semantic | Có | Có | Có thể giữ nguyên contract |
| `qformer_text` semantic | Có | Có | Có thể giữ nếu thêm bridge riêng |
| `direct_text` target | Có | Có | Target là dataset-level |
| Multi-prompt `alter` | Có | Có | Dataset-level |
| Eval prompt cố định | Có | Có | Dataset-level |
| Native preprocess tensor | Không cần giống | Có thể khác | Nên để native `SAIL` |
| Native chat token path | Cần tương thích | Có | `SAIL` đã dùng `<image>/<IMG_CONTEXT>` style |
| `no-qformer` path | Cần | Có | Vì `SAIL` đã có `vision -> mlp1 -> LLM` native |
| `qformer` insertion point | Cần rõ ràng | Có khả năng cao | Có seam trước `mlp1` |
| LoRA | Cần | Có | Nhưng target modules sẽ khác |

### 6.2 `Safe reuse`

Những phần có khả năng tái sử dụng về mặt logic:

- semantic prompt contract
- prompt mode `balanced_v1`
- answer mapping
- split / metrics policy
- qformer instruction philosophy
- resume / eval philosophy

### 6.3 `Must isolate`

Những phần khả năng cao phải tách riêng cho `SAIL`:

- preprocess ảnh native
- model load path
- tokenizer load path
- collate/runtime path
- LoRA target modules
- qformer bridge implementation
- checkpoint output dir / config nhận diện model
- notebook/entrypoint chạy `SAIL`

### 6.4 `Repo-backed inference`

Từ repo hiện tại + fact từ `SAIL`, có thể suy ra:

- `SAIL` là ứng viên hợp lý cho hướng “thêm một nhánh mới mà vẫn giữ semantic input/output giống nhau”
- `SAIL` hợp hơn `Qwen2-VL` cho bài toán `Q-Former` vì nó có `pixel_shuffle -> mlp1` seam rõ ràng
- hướng tối ưu cho Plan 2 là:
  - giữ `InternVL` nguyên
  - thêm `SAIL` theo nhánh riêng
  - không generic hóa sớm runtime hiện tại

---

## 7. Các câu hỏi còn mở cho Plan 2

### 7.1 Open questions được phép defer

Những câu hỏi sau **chưa được phép chốt** ở Plan 1:

- preprocess native cuối cho `SAIL` nên để `max_num = 10` hay `12`
- bridge `Q-Former` cho `SAIL` nên chèn trước `mlp1` theo cách nào cụ thể
- có giữ `mlp1` frozen hay cho trainable một phần
- LoRA targets cuối cho `SAIL` nên là full `Qwen2 set` hay một subset
- tổ chức file/runtime cuối nên là `dispatch layer` hay `sail-specific entrypoints`
- notebook / config tree cuối cho `SAIL`

### 7.2 Những điều đã đủ chắc để sang Plan 2

**Đã xác minh đủ chắc**

- semantic contract của repo hiện tại đã rõ
- fairness contract đã rõ
- `InternVL` cần được bảo vệ bằng cách không sửa generic hóa thẳng vào runtime cũ
- `SAIL` đủ gần `InternVL-like runtime` để xứng đáng đầu tư setup tiếp
- `SAIL` có native path hợp lý cho cả `no-qformer` và `qformer`

---

## 8. Kết luận tổng hợp

### Đã xác minh đủ chắc

- Repo hiện tại đang train/infer theo contract: `frame cuối + question text + qformer_text + direct_text target`.
- `alter` hiện tại là task trung tâm trong train/val.
- `Q-Former` hiện tại là **prompt-aware** thật sự.
- `InternVL` runtime đang có nhiều hardcode family-specific, không nên generic hóa sớm.
- `SAIL-VL-1.5-2B` có visual/runtime structure đủ gần để thêm thành nhánh riêng.

### Có khả năng đúng nhưng cần research sâu hơn ở Plan 2

- `SAIL qformer` hợp lý nhất khi chèn ở seam `vision -> pixel_shuffle -> [new bridge] -> mlp1`.
- `SAIL no-qformer` có thể giữ native path và vẫn comparable ở mức semantic.
- LoRA cho `SAIL` nên theo naming `Qwen2`.

### Chưa được phép chốt ở pha này

- config train cuối
- exact qformer architecture của `SAIL`
- final file/folder layout
- final notebook layout
- final hyperparameters

## Nguồn chính

- Repo local:
  - [wad_dataset.py](../../wad_dataset.py)
  - [train.py](../../train.py)
  - [scripts/test_infer.py](../../scripts/test_infer.py)
  - [qformer_bridge.py](../../qformer_bridge.py)
  - [data.py](../../data.py)
  - [preprocessing.py](../../preprocessing.py)
  - [internvl_config.yaml](../../internvl_config.yaml)
  - [internvl_config_no_qformer.yaml](../../internvl_config_no_qformer.yaml)
- SAIL model card: https://huggingface.co/BytedanceDouyinContent/SAIL-VL-1d5-2B
- SAIL remote code:
  - https://huggingface.co/BytedanceDouyinContent/SAIL-VL-1d5-2B/blob/main/modeling_sailvl.py
  - https://huggingface.co/BytedanceDouyinContent/SAIL-VL-1d5-2B/blob/main/conversation.py
  - https://huggingface.co/BytedanceDouyinContent/SAIL-VL-1d5-2B/blob/main/config.json
  - https://huggingface.co/BytedanceDouyinContent/SAIL-VL-1d5-2B/blob/main/modeling_qwen2.py
- SAIL paper: https://arxiv.org/abs/2501.05952
