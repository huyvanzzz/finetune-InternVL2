# Audit `train` và `test/infer` của InternVL: `779cc7b` vs code hiện tại

## Tóm tắt

Đợt audit này chỉ xét hai luồng có thể làm lệch kết quả cuối:

- `train`: dữ liệu đi vào, prompt/target, split/sampler, collate/input build, setup model, eval-in-train, resume semantics
- `test/infer`: prompt test, generation config, checkpoint load, metric input/output trên `test_alter`

Những kết luận quan trọng nhất:

1. **Khác biệt lớn nhất nằm ở prompt contract của `direct_text`, không phải ở optimizer hay generation config.**
   - `alter` prompt cũ ở `779` là “based on the final frame”, còn code hiện tại dùng legacy prompt “based on this image”.
   - `QA` prompt hiện tại cũng đã khác đáng kể so với `779`.
   - Vì setup mixed đang train cả `QA + alter`, đây là thay đổi semantics train rõ ràng nhất.

2. **Split/sampler của config mixed hiện tại đã gần quay lại `779`.**
   - `train_task_filter=all`, `val_task_filter=all`, `stratify_split=false`, `task_balancing=false`.
   - Điều này có nghĩa: hiện tại không còn lệch lớn ở task filtering hay weighted sampling.

3. **Validation semantics đã đổi rõ rệt dù train composition gần giống.**
   - `779` cap `val` mặc định còn 200 mẫu.
   - Hiện tại `eval_limit=0`, nên `val` dùng full 1013 mẫu.
   - `779` resample sample lỗi ở mọi split; hiện tại `val/test` log rồi skip.

4. **Test/infer generation config về cơ bản không đổi.**
   - `max_new_tokens=512`, `num_beams=3`, `do_sample=False`, `repetition_penalty=1.3`, `early_stopping=True` đều giữ nguyên.
   - Metric target cho `direct_text` vẫn là `raw_text`.

5. **Nếu bạn train từ đầu, phần resume/runtime hygiene mới gần như không phải nguồn lệch chính.**
   - Nếu có resume giữa epoch thì semantics resume hiện tại khác `779`.
   - Nếu không resume, chênh lệch chủ yếu nên ưu tiên truy ở `prompt contract`, `val policy`, và `dataset object semantics`.

## Ma trận khác biệt của `train`

### 1. Data đi vào train/val

| Trục | `779cc7b` | Hiện tại | Loại khác biệt | Đánh giá |
|---|---|---|---|---|
| Nguồn train metadata | `train.json` mixed tự nhiên | `train.json` mixed tự nhiên sau union filter `all/all` | gần như giống | low |
| Chọn frame train | lấy `frame_ids=[4,6,8]`, rồi chỉ load frame cuối | vẫn lấy `frame_ids=[4,6,8]`, rồi chỉ load frame cuối | giống | low |
| Mapping answer | `QA -> QA.A`, `alter -> alter` | giữ nguyên | giống | low |
| Tạo train/val dataset | tạo một `train_dataset` trên full `train`, rồi tách `Subset(train_dataset, train_indices/val_indices)` | split ra `train_samples/val_samples`, rồi tạo **hai dataset instance riêng** với `split='train'` và `split='val'` | semantic | medium |
| Sample lỗi ở train/val | lỗi ở mọi split đều `random-resample` | `train` resample; `val` log `[DATA ERROR]` rồi `return None` | semantic | high cho val, low cho train |

**Evidence**

- `779cc7b:wad_dataset.py:72-90`
- `779cc7b:wad_dataset.py:96-156`
- `779cc7b:wad_dataset.py:250-258`
- `779cc7b:preprocessing.py:58,102-104`
- `current .worktrees/sailvl-core/wad_dataset.py:193-196`
- `current .worktrees/sailvl-core/wad_dataset.py:233-269`
- `current .worktrees/sailvl-core/wad_dataset.py:291-298`
- `current .worktrees/sailvl-core/wad_dataset.py:412-430`

### 2. Prompt / supervision contract khi train

| Trục | `779cc7b` | Hiện tại | Loại khác biệt | Đánh giá |
|---|---|---|---|---|
| `alter` prompt train | `Describe the scene ... based on the final frame.` + 2 dòng guidance | legacy prompt: `Describe the scene ... based on this image.` + 2 dòng guidance | semantic | **critical** |
| `QA` prompt train | `Describe the scene ... based on the final frame.` + `Focus on obstacles...` + `Question: ...` + `Answer the question directly in natural language.` | `Describe the scene ... based on this frame.` + `Focus on obstacles, nearby people or vehicles, free walking space, direction, and safety.` + `Question: ...` | semantic | **critical** |
| `qformer_text` | mirror đúng `text_content` đang train | vẫn mirror đúng prompt train | giống về cơ chế, khác về nội dung | high vì prompt đã đổi |
| Target train cuối | `direct_text` -> instruction text | giữ nguyên `direct_text` -> instruction text | giống | low |

**Evidence**

- `779cc7b:wad_dataset.py:113-145`
- `current .worktrees/sailvl-core/wad_dataset.py:18-22`
- `current .worktrees/sailvl-core/wad_dataset.py:216-229`
- `current .worktrees/sailvl-core/preprocessing.py:58,102-104`

**Runtime-confirmed**

- Current mixed config thực tế dựng ra prompt `alter`:
  - `<image>\nDescribe the scene for a visually impaired user based on this image.\nFocus on immediate obstacles, safe direction, and what action the user should take.\nProvide only the final spoken guidance in natural language.`
- Current mixed config thực tế dựng ra prompt `QA`:
  - `<image>\nDescribe the scene for a visually impaired user based on this frame.\nFocus on obstacles, nearby people or vehicles, free walking space, direction, and safety.\nQuestion: describe the current scene`

### 3. Split / sampling contract

| Trục | `779cc7b` | Hiện tại | Loại khác biệt | Đánh giá |
|---|---|---|---|---|
| Task filter | không có filter path | có infra `train_task_filter/val_task_filter`, nhưng config mixed đang là `all/all` | semantic-capable, runtime gần giống | low |
| Split type | `train_test_split(..., random_state=seed)` thường | `stratify_split=false` nên cũng split thường | giống ở runtime mixed | low |
| Sampler | shuffle thường | có `build_train_sampler()`, nhưng `task_balancing.enabled=false` nên shuffle thường | semantic-capable, runtime gần giống | low |
| Val size | `eval_limit` mặc định 200 | `eval_limit=0` -> full val 1013 | semantic | **high** |

**Evidence**

- `779cc7b:wad_dataset.py:250-265`
- `779cc7b:internvl_config.yaml:57,61-78,86-88`
- `current .worktrees/sailvl-core/wad_dataset.py:85`
- `current .worktrees/sailvl-core/wad_dataset.py:380-390`
- `current .worktrees/sailvl-core/wad_dataset.py:412-434`
- `current .worktrees/sailvl-core/train.py:351-389`
- `current .worktrees/sailvl-core/internvl_config_mixed.yaml:57-60,66-67,82-83,99-102`

**Runtime-confirmed**

- Current mixed config summary:
  - `train_task_filter=all`
  - `val_task_filter=all`
  - `direct_text_alter_prompt_mode=fixed_legacy`
  - `stratify_split=false`
  - `eval_limit=0`
- Current split stats:
  - `train_len=9111`, `val_len=1013`
  - `train: QA=1415, alter=7696`
  - `val: QA=138, alter=875`

### 4. Collate / input build contract

| Trục | `779cc7b` | Hiện tại | Loại khác biệt | Đánh giá |
|---|---|---|---|---|
| Chèn `<image>`/`<IMG_CONTEXT>` | thay `<image>` bằng `num_patches * num_image_token` image tokens | giữ nguyên cơ chế | giống | low |
| Lặp `qformer_text` | lặp theo `total_tiles` | giữ nguyên | giống | low |
| Batch lỗi / sample `None` | vì val/test resample nên collate hầu như không thấy `None` | collate hiện tại có thể nhận batch chứa `None` ở val/test path gián tiếp qua `return None` từ dataset | semantic-capable | medium |
| Token/image logging | chỉ log token stats | thêm prompt/sample logs, data verify logs | bookkeeping | low |

**Evidence**

- `779cc7b:train.py:205-334`
- `current .worktrees/sailvl-core/train.py:241-348`

### 5. Model setup lúc train

| Trục | `779cc7b` | Hiện tại | Loại khác biệt | Đánh giá |
|---|---|---|---|---|
| Base model | `OpenGVLab/InternVL2_5-2B` | giữ nguyên | giống | low |
| Q-Former attach | attach bridge + align runtime | giữ nguyên ở InternVL path | giống | low |
| LoRA init/load | `prepare_model_for_kbit_training` + `PeftModel.from_pretrained` hoặc `get_peft_model` | giữ nguyên | giống | low |
| Projection param group | có `proj_learning_rate` cho `qformer_input_proj`, `qformer_to_mlp1_proj` | giữ nguyên | giống | low |
| Vision freeze | `freeze_encoder=true` | giữ nguyên | giống | low |
| Gradient checkpointing | bật | hiện tại vẫn bật, nhưng đi qua helper `enable_gradient_checkpointing` | bookkeeping cho InternVL | low |

**Evidence**

- `779cc7b:internvl_config.yaml:8-45`
- `779cc7b:train.py:610-621`
- `current .worktrees/sailvl-core/train.py:559-568`
- `current .worktrees/sailvl-core/train.py:848-867`
- `current .worktrees/sailvl-core/internvl_config_mixed.yaml:8-45`

### 6. Optimization / eval-in-train / resume

| Trục | `779cc7b` | Hiện tại | Loại khác biệt | Đánh giá |
|---|---|---|---|---|
| `batch_size`, `grad_accum`, `lr`, `proj_lr`, `eval_steps`, `save_steps` | `2`, `8`, `2e-4`, `5e-4`, `2000`, `2000` | giữ nguyên | giống | low |
| `num_epochs` | `3` | `5` | khác config, không ảnh hưởng nếu so cùng epoch | low |
| Resume giữa epoch | skip batch bằng `next(batch_iterator)` | skip batch + restore `runtime_state.pt` nếu có | semantic **chỉ khi resume** | medium |
| Optimizer/scheduler load | load rồi sanitize/move device đơn giản | thêm sanitize, device-align, rng state restore, optimizer mismatch checks | semantic **chỉ khi resume** | medium |

**Evidence**

- `779cc7b:internvl_config.yaml:61-88`
- `779cc7b:train.py:378-536`
- `current .worktrees/sailvl-core/train.py:520-616`
- `current .worktrees/sailvl-core/train.py:638-749`
- `current .worktrees/sailvl-core/internvl_config_mixed.yaml:66-102`

## Ma trận khác biệt của `test/infer`

### 1. Test dataset contract

| Trục | `779cc7b` | Hiện tại | Loại khác biệt | Đánh giá |
|---|---|---|---|---|
| Split `test_alter` | load `test_alter.json` khi truyền `--split test_alter` | giữ nguyên | giống | low |
| Test loader | `DataLoader(... batch_size=1, shuffle=False)` | giữ nguyên | giống | low |
| Sample lỗi ở test | dataset cũ resample nên không có skip path rõ ràng | dataset hiện tại `return None`, test loop `skipped_samples += 1` rồi `continue` | semantic | medium |

**Evidence**

- `779cc7b:scripts/test_infer.py:212,236,253`
- `current .worktrees/sailvl-core/wad_dataset.py:291-298`
- `current .worktrees/sailvl-core/scripts/test_infer.py:294,310,315-316,372-373`

### 2. Prompt infer contract

| Trục | `779cc7b` | Hiện tại | Loại khác biệt | Đánh giá |
|---|---|---|---|---|
| `alter` prompt test | dựa trên prompt cũ “final frame” | dùng `fixed_legacy` hiện tại “this image” | semantic | **critical** |
| `qformer_text` infer | mirror prompt test | vẫn mirror prompt test | giống cơ chế, khác nội dung | high |
| Prompt mode infra | không có | có `direct_text_alter_prompt_mode`, nhưng mixed config đang `fixed_legacy` | semantic-capable | low |

**Evidence**

- `779cc7b:wad_dataset.py:113-145`
- `current .worktrees/sailvl-core/wad_dataset.py:18-22`
- `current .worktrees/sailvl-core/wad_dataset.py:216-229`
- `current .worktrees/sailvl-core/scripts/test_infer.py:248-250`

### 3. Generation contract

| Trục | `779cc7b` | Hiện tại | Loại khác biệt | Đánh giá |
|---|---|---|---|---|
| `max_new_tokens` | `512` | `512` | giống | low |
| `num_beams` | `3` | `3` | giống | low |
| `do_sample` | `False` | `False` | giống | low |
| `repetition_penalty` | `1.3` | `1.3` | giống | low |
| `early_stopping` | `True` | `True` | giống | low |
| Decode path | `run_model_chat -> model.generate` rồi split theo `template.sep` | giữ nguyên cho InternVL path | giống | low |

**Evidence**

- `779cc7b:scripts/test_infer.py:46-68`
- `779cc7b:scripts/test_infer.py:259-264`
- `current .worktrees/sailvl-core/scripts/test_infer.py:57-79`
- `current .worktrees/sailvl-core/scripts/test_infer.py:323-328`

### 4. Checkpoint load contract

| Trục | `779cc7b` | Hiện tại | Loại khác biệt | Đánh giá |
|---|---|---|---|---|
| Load base model | `AutoModel.from_pretrained(...)` + quantization | giữ nguyên với InternVL path | giống | low |
| Load Q-Former bridge | `attach_qformer_bridge`, rồi `load_qformer_bridge` nếu checkpoint có | giữ nguyên | giống | low |
| Load LoRA | `PeftModel.from_pretrained(... is_trainable=False)` | giữ nguyên | giống | low |
| Device align / eval logs | hiện ít hơn | hiện có thêm `align_language_model_devices`, eval-state logs | bookkeeping | low |

**Evidence**

- `779cc7b:scripts/test_infer.py:169-192`
- `current .worktrees/sailvl-core/scripts/test_infer.py:223-246`

### 5. Metric input/output contract

| Trục | `779cc7b` | Hiện tại | Loại khác biệt | Đánh giá |
|---|---|---|---|---|
| Metric target field | `direct_text -> raw_text` | giữ nguyên | giống | low |
| Kết quả chính | `output_file` JSON | giữ nguyên | giống | low |
| Pairs file phụ | không có | thêm `*_pairs.json` lưu `ground_truth/generation` | bookkeeping | low |

**Evidence**

- `779cc7b:scripts/test_infer.py:302`
- `current .worktrees/sailvl-core/scripts/test_infer.py:369`
- `current .worktrees/sailvl-core/scripts/test_infer.py:393-399`

## Danh sách khác biệt có rủi ro cao

### F1. Prompt `alter` đã đổi giữa `779` và current

- `Severity`: critical
- `Area`: train + test prompt
- `Old behavior`: `Describe the scene ... based on the final frame.`
- `Current behavior`: `Describe the scene ... based on this image.`
- `Why it matters`: đổi instruction wording ở cả train lẫn test, nên model không còn học và được đánh giá trên cùng textual contract như `779`
- `Expected impact`: rất dễ làm lệch kết quả, nhất là với task `alter` vốn target ngắn và dễ nhạy prompt
- `Evidence`: `779cc7b:wad_dataset.py:113-145`; `current .worktrees/sailvl-core/wad_dataset.py:18-22,216-229`
- `Label`: `code-evidenced`, `runtime-confirmed`

### F2. Prompt `QA` đã đổi mạnh trong mixed training

- `Severity`: critical
- `Area`: train prompt
- `Old behavior`: QA prompt cũ vẫn neo theo “final frame” và kết bằng `Answer the question directly in natural language.`
- `Current behavior`: QA prompt mới dùng “this frame” + wording khác hẳn về focus
- `Why it matters`: mixed setup train `QA + alter`, nên việc thay QA prompt có thể đổi hẳn regularization/supervision signal mà Q-Former nhìn thấy
- `Expected impact`: cao, đặc biệt nếu trước đây mixed-task từng cho kết quả tốt hơn
- `Evidence`: `779cc7b:wad_dataset.py:126-139`; `current .worktrees/sailvl-core/wad_dataset.py:218-223`
- `Label`: `code-evidenced`, `runtime-confirmed`

### F3. Validation policy đã đổi từ `200` mẫu sang full `1013`

- `Severity`: high
- `Area`: eval-in-train
- `Old behavior`: `eval_limit` mặc định `200`
- `Current behavior`: `eval_limit=0`, nên val full `1013`
- `Why it matters`: dù không đổi weight update trực tiếp, nó đổi hẳn cách bạn nhìn overfit/early stopping và có thể làm so sánh “epoch tốt nhất” lệch khỏi `779`
- `Expected impact`: cao với kết luận theo val curve; thấp với weight dynamics thuần train
- `Evidence`: `779cc7b:wad_dataset.py:263-265`; `779cc7b:internvl_config.yaml:86-88`; `current .worktrees/sailvl-core/wad_dataset.py:431-435`; `current .worktrees/sailvl-core/internvl_config_mixed.yaml:60`
- `Label`: `code-evidenced`, `runtime-confirmed`

### F4. Val/test error handling đổi từ resample sang skip

- `Severity`: high
- `Area`: val + test dataset semantics
- `Old behavior`: sample lỗi sẽ random-resample, nên val/test effective set ít đổi theo lỗi đọc ảnh
- `Current behavior`: `val/test` log lỗi rồi `return None`, vòng eval/test sẽ skip
- `Why it matters`: tập val/test hiệu dụng có thể thay đổi giữa các lần chạy nếu có sample lỗi
- `Expected impact`: trung bình đến cao nếu dataset/frame index/tar access không ổn định
- `Evidence`: `779cc7b:wad_dataset.py:156-159`; `current .worktrees/sailvl-core/wad_dataset.py:291-298`; `current .worktrees/sailvl-core/scripts/test_infer.py:310,315-316,372-373`
- `Label`: `code-evidenced`

### F5. Resume semantics đã đổi, nhưng chỉ đáng nghi nếu bạn resume

- `Severity`: medium
- `Area`: train resume
- `Old behavior`: skip batch đơn thuần
- `Current behavior`: thêm `runtime_state.pt`, restore RNG, sanitize optimizer/scheduler
- `Why it matters`: nếu so một run resume với một run train liền mạch, current gần `equivalent` hơn `779`; nếu toàn bộ run đều train từ đầu thì finding này không giải thích được lệch lớn
- `Expected impact`: chỉ cao khi resume giữa epoch
- `Evidence`: `779cc7b:train.py:432-485`; `current .worktrees/sailvl-core/train.py:577-649`
- `Label`: `code-evidenced`

## Khác biệt thấp hoặc gần như không phải thủ phạm chính

- Base model, LoRA targets, Q-Former attach path, `prepare_model_for_kbit_training` về cơ bản giữ nguyên giữa hai mốc.
- Current mixed config đã tắt task balancing và tắt stratified split, nên **sampler/filter không còn là khác biệt chính** so với `779`.
- Test generation config gần như giữ nguyên hoàn toàn.
- Metric target cho `direct_text` vẫn là `raw_text`.
- File `*_pairs.json` mới chỉ là artifact phụ, không đổi cách chấm điểm.

## Thứ tự ablation nên làm

1. **Khôi phục đúng prompt contract của `779` trước**
   - `alter` prompt phải quay đúng về wording “final frame”
   - `QA` prompt phải quay đúng về wording cũ của `779`
   - làm cả ở train dataset và test dataset

2. **Khôi phục đúng validation policy của `779`**
   - cap `val` về `200`
   - nếu muốn so ngang `779`, nên giữ cùng policy sample lỗi cho val/test hoặc ít nhất log rõ có skip sample nào không

3. **Giữ nguyên mixed split/sampler như hiện tại**
   - `all/all`
   - `stratify_split=false`
   - `task_balancing=false`
   - vì ở runtime mixed hiện tại, phần này đã khá sát `779`

4. **Chỉ sau đó mới kiểm tra ảnh hưởng của dataset instance riêng vs `Subset`**
   - nếu prompt/val policy đã khớp mà điểm vẫn lệch, đây là khác biệt train semantics tiếp theo đáng test

5. **Chỉ ưu tiên nhánh resume nếu run đang so sánh có dùng resume thật**
   - nếu không resume, đừng để nhánh này làm nhiễu chẩn đoán

## Kết luận ngắn

Nếu chỉ giới hạn vào `train` và `test_alter`, thì **nguồn lệch kết quả đáng nghi nhất hiện tại không phải optimizer hay generation config**, mà là:

1. prompt `alter` đã khác `779`
2. prompt `QA` trong mixed training đã khác `779`
3. validation policy (`200` vs full `1013`) đã khác
4. val/test error handling (`resample` vs `skip`) đã khác

Ngược lại, các phần sau **đang khá giống hoặc không đủ mạnh để giải thích lệch lớn**:

- base model / LoRA / Q-Former attach
- generation config của `test_alter`
- metric target field
- split/sampler mixed hiện tại (`all/all`, no stratify, no balancing)
