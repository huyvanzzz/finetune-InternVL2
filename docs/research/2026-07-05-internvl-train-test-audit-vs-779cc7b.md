# Audit train/test InternVL: `779cc7b` vs code hiện tại

## 1. Tóm tắt

Đợt audit này chỉ tập trung vào hai luồng có thể làm lệch kết quả cuối:

- `train`: dữ liệu, prompt, input batch, setup model, optimizer/eval-in-train, resume
- `test_alter`: prompt infer, generation config, checkpoint load, metric I/O

Kết luận ngắn:

1. **Thành phần dữ liệu train mixed hiện tại đã khớp `779` ở mức split cơ bản**.
   - `raw_train_len = 10124`
   - `raw_train_counts = alter 8571 / qa 1553`
   - `train_len = 9111`
   - `val_len_full = 1013`
   - `train_counts = alter 7696 / qa 1415`
   - `val_counts_full = alter 875 / qa 138`
   - Nhãn: `runtime-confirmed`

2. **Prompt `alter` hiện tại đã khớp `779`, nhưng prompt `QA` vẫn chưa khớp hoàn toàn**.
   - `alter`: đã quay về `"based on the final frame"` đúng `779`
   - `QA`: hiện tại vẫn bắt đầu bằng `"based on this frame"`, trong khi `779` thực sự bắt đầu bằng `"based on the final frame"` rồi mới nối phần QA
   - Đây là khác biệt semantic quan trọng nhất ở nhánh mixed hiện tại
   - Nhãn: `runtime-confirmed` + `code-evidenced`

3. **Validation trong train không còn giống `779`**.
   - `779`: val bị cap còn 200 samples từ `build_dataset`, và `eval_model()` còn có guard `if total_eval_batchs == 200`
   - Hiện tại: val full 1013 samples, `eval_model()` không còn cap 200 batches
   - Điểm này không trực tiếp đổi `test_alter` nếu so cùng epoch, nhưng có thể đổi tín hiệu chọn epoch/checkpoint và diễn giải xu hướng train
   - Nhãn: `code-evidenced` + `runtime-confirmed`

4. **`test_alter` hiện tại có một khác biệt semantic quan trọng ở error handling**.
   - `779`: sample lỗi sẽ resample
   - Hiện tại trong `scripts/test_infer.py`: dataset test đang dùng default `non_train_error_policy=skip`, `TestCollaterFn` lọc `None`, nên sample lỗi bị bỏ qua
   - Nếu test có sample lỗi, điểm cuối có thể đổi vì số mẫu chấm không còn giống nhau
   - Nhãn: `code-evidenced`

5. **Generation contract của `test_alter` về cơ bản giữ nguyên**.
   - `max_new_tokens=512`
   - `num_beams=3`
   - `do_sample=False`
   - `repetition_penalty=1.3`
   - `early_stopping=True`
   - Đây không phải nguồn lệch chính
   - Nhãn: `code-evidenced`

## 2. Ma trận khác biệt của train

### 2.1. Data train/val

**Finding T1**

- `Hành vi cũ @779`
  - Load `train.json` mixed tự nhiên, split bằng `train_test_split(..., random_state=seed)`, không filter task
- `Hành vi hiện tại`
  - Mixed config cũng load `train.json`, `train_task_filter=all`, `val_task_filter=all`, `stratify_split=false`
- `Loại khác biệt`
  - `bookkeeping`
- `Vì sao có thể ảnh hưởng kết quả`
  - Với mixed config hiện tại thì gần như không ảnh hưởng, vì union filter vẫn giữ toàn bộ mẫu
- `Hướng ảnh hưởng kỳ vọng`
  - Không đáng kể
- `Bằng chứng`
  - `779`:
    - `wad_dataset.py:250`
  - Hiện tại:
    - `wad_dataset.py:442`
    - `internvl_config_mixed.yaml:56-64`
  - Runtime:
    - `raw_train_len=10124`
    - `train_len=9111`
    - `val_len=1013`
    - `train_counts=alter 7696 / qa 1415`
    - `val_counts=alter 875 / qa 138`

**Finding T2**

- `Hành vi cũ @779`
  - `train_subset` và `val_subset` là `Subset(train_dataset, ...)`, tức cùng bọc trên một dataset gốc có `split='train'`
- `Hành vi hiện tại`
  - `train_dataset` và `val_dataset` là hai dataset instance riêng, với `split='train'` và `split='val'`
- `Loại khác biệt`
  - `semantic`
- `Vì sao có thể ảnh hưởng kết quả`
  - Vì code hiện tại có logic phụ thuộc `split` như prompt mode theo split và `non_train_error_policy`, nên việc `val` trở thành dataset riêng không còn là thay đổi thuần cấu trúc nữa
- `Hướng ảnh hưởng kỳ vọng`
  - Ảnh hưởng thấp đến trung bình; mạnh nhất khi val/test error handling hoặc prompt mode theo split khác nhau
- `Bằng chứng`
  - `779`:
    - `wad_dataset.py:254-267`
  - Hiện tại:
    - `wad_dataset.py:474-504`

**Finding T3**

- `Hành vi cũ @779`
  - Mọi split đều resample khi sample lỗi
- `Hành vi hiện tại`
  - `mixed` config đã được set về `non_train_error_policy: resample`, nên `train` và `val` hiện tại quay lại gần kiểu cũ
- `Loại khác biệt`
  - `bookkeeping` đối với mixed config hiện tại
- `Vì sao có thể ảnh hưởng kết quả`
  - Trước khi quay về `resample`, `val` có thể bị `return None`; hiện tại mixed config không còn khác biệt này
- `Hướng ảnh hưởng kỳ vọng`
  - Không còn là nguồn lệch chính trong mixed config hiện tại
- `Bằng chứng`
  - `779`:
    - `wad_dataset.py:167`
  - Hiện tại:
    - `wad_dataset.py:323-330`
    - `internvl_config_mixed.yaml:62-64`

### 2.2. Prompt / supervision contract

**Finding T4**

- `Hành vi cũ @779`
  - Mọi sample `direct_text` đều bắt đầu từ:
    - `"Describe the scene for a visually impaired user based on the final frame."`
  - Nếu là QA thì nối thêm:
    - focus QA
    - `Question: ...`
    - `Answer the question directly in natural language.`
- `Hành vi hiện tại`
  - `alter` đã quay về đúng chuỗi cũ
  - `QA` lại dùng template `legacy_779` riêng bắt đầu bằng:
    - `"Describe the scene for a visually impaired user based on this frame."`
- `Loại khác biệt`
  - `semantic`
- `Vì sao có thể ảnh hưởng kết quả`
  - Trong mixed train, nhánh QA là nguồn supervision phụ. Việc prompt QA khác wording cũ làm thay đổi phân bố instruction, nhất là với Q-Former prompt-aware
- `Hướng ảnh hưởng kỳ vọng`
  - Khả năng cao làm lệch kết quả, đặc biệt nếu lợi ích của mixed đến từ regularization qua QA
- `Bằng chứng`
  - `779`:
    - `wad_dataset.py:113`
    - `wad_dataset.py:131-136`
  - Hiện tại:
    - `wad_dataset.py:24-28`
    - `wad_dataset.py:36-41`
    - `wad_dataset.py:159-161`
  - Runtime:
    - current QA prompt sample:
      - `"based on this frame"`
    - reconstructed `779` QA prompt sample:
      - `"based on the final frame"`

**Finding T5**

- `Hành vi cũ @779`
  - `alter` prompt fixed, không multi-template
- `Hành vi hiện tại`
  - Mixed config đang dùng `direct_text_alter_prompt_mode: fixed_779`
- `Loại khác biệt`
  - `bookkeeping`
- `Vì sao có thể ảnh hưởng kết quả`
  - Không còn ảnh hưởng trong mixed config hiện tại vì wording đã khớp
- `Hướng ảnh hưởng kỳ vọng`
  - Không đáng kể
- `Bằng chứng`
  - `internvl_config_mixed.yaml:62`
  - Runtime prompt sample của `alter` khớp hoàn toàn với reconstructed `779`

**Finding T6**

- `Hành vi cũ @779`
  - `qformer_text` chính là `text_content.strip()` của prompt train
- `Hành vi hiện tại`
  - Vẫn là `text_content.strip()`
- `Loại khác biệt`
  - `bookkeeping`
- `Vì sao có thể ảnh hưởng kết quả`
  - Không phải nguồn lệch; `qformer_text` chỉ lệch khi prompt gốc lệch
- `Hướng ảnh hưởng kỳ vọng`
  - Phụ thuộc hoàn toàn vào T4
- `Bằng chứng`
  - `779`:
    - `wad_dataset.py:159`
  - Hiện tại:
    - `wad_dataset.py:305`

### 2.3. Split / sampling contract

**Finding T7**

- `Hành vi cũ @779`
  - Split random thường, không `stratify`
- `Hành vi hiện tại`
  - Mixed config đang `stratify_split: false`
- `Loại khác biệt`
  - `bookkeeping`
- `Vì sao có thể ảnh hưởng kết quả`
  - Không còn là nguồn lệch trong mixed config hiện tại
- `Hướng ảnh hưởng kỳ vọng`
  - Không đáng kể
- `Bằng chứng`
  - `779`:
    - `wad_dataset.py:250`
  - Hiện tại:
    - `wad_dataset.py:104-108`
    - `wad_dataset.py:442-446`
    - `internvl_config_mixed.yaml:59`

**Finding T8**

- `Hành vi cũ @779`
  - `train_loader` dùng `shuffle=True`, không sampler đặc biệt
- `Hành vi hiện tại`
  - Code đã có `WeightedRandomSampler`, nhưng mixed config `task_balancing.enabled=false`, nên thực tế vẫn rơi về `shuffle=True`
- `Loại khác biệt`
  - `bookkeeping`
- `Vì sao có thể ảnh hưởng kết quả`
  - Với mixed config hiện tại thì không ảnh hưởng
- `Hướng ảnh hưởng kỳ vọng`
  - Không đáng kể
- `Bằng chứng`
  - `779`:
    - `train.py` không có `WeightedRandomSampler`
  - Hiện tại:
    - `train.py:360-399`
    - `train.py:936-942`
    - `internvl_config_mixed.yaml:86-89`

**Finding T9**

- `Hành vi cũ @779`
  - Val bị cap về 200 samples ngay từ `build_dataset`
- `Hành vi hiện tại`
  - Mixed config để `eval_limit: 0`, nên val full 1013 samples
- `Loại khác biệt`
  - `semantic`
- `Vì sao có thể ảnh hưởng kết quả`
  - Làm đổi hoàn toàn `val_loss` theo step, tín hiệu theo dõi overfit, và khả năng chọn checkpoint/epoch
- `Hướng ảnh hưởng kỳ vọng`
  - Cao đối với diễn giải train và model selection; gián tiếp ảnh hưởng kết quả cuối nếu bạn chọn epoch theo val
- `Bằng chứng`
  - `779`:
    - `wad_dataset.py:263-267`
  - Hiện tại:
    - `wad_dataset.py:466-470`
    - `internvl_config_mixed.yaml:60`
  - Runtime:
    - old val cap 200 → `alter 171 / qa 29`
    - current val full 1013 → `alter 875 / qa 138`

### 2.4. Collate / input build

**Finding T10**

- `Hành vi cũ @779`
  - Collate chèn `<img> + <IMG_CONTEXT> * num_image_token * num_patches + </img>`
  - `qformer_text` được repeat theo `total_tiles`
- `Hành vi hiện tại`
  - InternVL path giữ đúng logic đó
- `Loại khác biệt`
  - `bookkeeping`
- `Vì sao có thể ảnh hưởng kết quả`
  - Không phải nguồn lệch chính
- `Hướng ảnh hưởng kỳ vọng`
  - Không đáng kể
- `Bằng chứng`
  - `779`:
    - `train.py` collate section
  - Hiện tại:
    - `train.py` collate section
  - Runtime smoke:
    - `input_ids_shape=(2, 48)`
    - `pixel_values_shape=(2, 3, 448, 448)`
    - `qformer_input_shape=(2, 1)`
    - `total_image_tokens=32` trong patched smoke 1 tile

**Finding T11**

- `Hành vi cũ @779`
  - Collate không có prompt/sample debug extras
- `Hành vi hiện tại`
  - Có thêm token logs, prompt logs, sample-id logs
- `Loại khác biệt`
  - `bookkeeping`
- `Vì sao có thể ảnh hưởng kết quả`
  - Chỉ tăng observability, không đổi input semantic
- `Hướng ảnh hưởng kỳ vọng`
  - Không đáng kể
- `Bằng chứng`
  - `train.py:263-359`

### 2.5. Model setup lúc train

**Finding T12**

- `Hành vi cũ @779`
  - Base model: `OpenGVLab/InternVL2_5-2B`
  - Q-Former bridge prompt-aware
  - LoRA target modules: `wqkv`, `wo`, `w1`, `w2`, `w3`
  - `prepare_model_for_kbit_training()` trên `language_model`
  - Vision encoder freeze
- `Hành vi hiện tại`
  - Mixed config và InternVL path vẫn giữ đúng các thành phần này
- `Loại khác biệt`
  - `bookkeeping`
- `Vì sao có thể ảnh hưởng kết quả`
  - Không phải nguồn lệch chính
- `Hướng ảnh hưởng kỳ vọng`
  - Không đáng kể
- `Bằng chứng`
  - `preprocessing.py` và `qformer_bridge.py` hiện tại không có diff so với `779`
  - `git diff --name-only 779... -- preprocessing.py qformer_bridge.py` trả về rỗng

**Finding T13**

- `Hành vi cũ @779`
  - Gọi trực tiếp `model.gradient_checkpointing_enable()`
- `Hành vi hiện tại`
  - Gọi `enable_gradient_checkpointing()` helper, helper này thử trên `model`, rồi fallback sang `language_model`
- `Loại khác biệt`
  - `inference-only`
- `Vì sao có thể ảnh hưởng kết quả`
  - Nếu InternVL path kích hoạt cùng module như cũ thì không sao; nếu helper fallback khác đi thì activation point của checkpointing có thể thay đổi
- `Hướng ảnh hưởng kỳ vọng`
  - Thấp, nhưng chưa đủ bằng chứng runtime để loại bỏ hoàn toàn
- `Bằng chứng`
  - `779`:
    - `train.py:597`
  - Hiện tại:
    - `training_runtime.py`
    - `train.py:859-862`

### 2.6. Optimization / eval-in-train / resume

**Finding T14**

- `Hành vi cũ @779`
  - `num_epochs=3`
  - `batch_size=2`
  - `gradient_accumulation_steps=8`
  - `eval_steps=2000`
  - `save_steps=2000`
- `Hành vi hiện tại`
  - Mixed config:
    - `num_epochs=5`
    - `batch_size=2`
    - `gradient_accumulation_steps=8`
    - `eval_steps=2000`
    - `save_steps=2000`
- `Loại khác biệt`
  - `bookkeeping` nếu so cùng epoch
- `Vì sao có thể ảnh hưởng kết quả`
  - Không ảnh hưởng khi bạn đang so cùng epoch
- `Hướng ảnh hưởng kỳ vọng`
  - Bỏ qua trong phân tích nguyên nhân chính
- `Bằng chứng`
  - `779` config vs `internvl_config_mixed.yaml`

**Finding T15**

- `Hành vi cũ @779`
  - `eval_model()` dừng ở `total_eval_batchs == 200`
  - Nhưng do val đã cap 200 samples, với `batch_size=2` thì hiệu lực thực tế là khoảng 100 eval batches
- `Hành vi hiện tại`
  - `eval_model()` không còn batch cap
- `Loại khác biệt`
  - `semantic`
- `Vì sao có thể ảnh hưởng kết quả`
  - Làm đổi trực tiếp val loss during training
- `Hướng ảnh hưởng kỳ vọng`
  - Cao đối với val-based checkpoint selection
- `Bằng chứng`
  - `779`:
    - `train.py:367`
  - Hiện tại:
    - `train.py` phần `eval_model()` không còn guard đó

**Finding T16**

- `Hành vi cũ @779`
  - Resume giữa epoch chỉ skip `start_step` batches
- `Hành vi hiện tại`
  - Code có hỗ trợ `runtime_state.pt`, RNG digest, optimizer sanitization, nhưng mixed config đang đặt `resume_runtime_mode: legacy_skip_only`
- `Loại khác biệt`
  - `bookkeeping` đối với mixed config hiện tại
- `Vì sao có thể ảnh hưởng kết quả`
  - Với mixed config hiện tại, resume path đã quay về kiểu cũ; nếu sau này đổi mode thì semantic mới khác
- `Hướng ảnh hưởng kỳ vọng`
  - Thấp trong mixed hiện tại; chỉ đáng lo nếu resume giữa epoch với config khác
- `Bằng chứng`
  - `779`:
    - `train.py:477-478`
  - Hiện tại:
    - `train.py:145-151`
    - `train.py:586`
    - `train.py:627-635`
    - `internvl_config_mixed.yaml:94`

## 3. Ma trận khác biệt của test_alter

### 3.1. Test dataset contract

**Finding E1**

- `Hành vi cũ @779`
  - `test_alter` load từ `test_alter.json`
  - `batch_size=1`
  - test sample lỗi không bị bỏ qua ở collate; dataset path cũ resample
- `Hành vi hiện tại`
  - Vẫn load `test_alter.json`
  - `evaluation.batch_size=4`
  - `TestCollaterFn` lọc `None`
  - dataset test không được truyền `non_train_error_policy`, nên default là `skip`
- `Vì sao có thể làm đổi điểm test_alter`
  - Nếu có sample lỗi, current path sẽ chấm trên ít mẫu hơn; old path sẽ resample để vẫn đủ mẫu
- `Hướng ảnh hưởng`
  - Trung bình đến cao nếu data/test có sample lỗi; thấp nếu test sạch hoàn toàn
- `Bằng chứng`
  - `779`:
    - `scripts/test_infer.py:233`
    - `scripts/test_infer.py:238`
  - Hiện tại:
    - `scripts/test_infer.py:290`
    - `scripts/test_infer.py:296`
    - `scripts/test_infer.py:314`
    - `wad_dataset.py:175`
    - `wad_dataset.py:323-328`
  - Runtime:
    - `test_alter_len = 1007`

### 3.2. Prompt infer contract

**Finding E2**

- `Hành vi cũ @779`
  - `test_alter` dùng prompt alter fixed cũ
- `Hành vi hiện tại`
  - Mixed config truyền `direct_text_alter_prompt_mode=fixed_779` vào dataset test
- `Vì sao có thể làm đổi điểm test_alter`
  - Với `test_alter`, prompt semantic hiện tại khớp cũ nên đây không phải nguồn lệch
- `Hướng ảnh hưởng`
  - Không đáng kể
- `Bằng chứng`
  - `scripts/test_infer.py:290`
  - Runtime reconstructed current alter prompt khớp reconstructed `779`

**Finding E3**

- `Hành vi cũ @779`
  - `qformer_text` infer bám đúng prompt alter
- `Hành vi hiện tại`
  - Vẫn vậy
- `Vì sao có thể làm đổi điểm test_alter`
  - Không phải nguồn lệch
- `Hướng ảnh hưởng`
  - Không đáng kể
- `Bằng chứng`
  - `scripts/test_infer.py:332-339`
  - `wad_dataset.py:305`

### 3.3. Generation contract

**Finding E4**

- `Hành vi cũ @779`
  - `max_new_tokens=512`
  - `num_beams=3`
  - `do_sample=False`
  - `repetition_penalty=1.3`
  - `early_stopping=True`
- `Hành vi hiện tại`
  - Giữ nguyên
- `Vì sao có thể làm đổi điểm test_alter`
  - Không phải nguồn lệch chính
- `Hướng ảnh hưởng`
  - Không đáng kể
- `Bằng chứng`
  - `779`:
    - `scripts/test_infer.py:261-264`
  - Hiện tại:
    - `scripts/test_infer.py:325-328`

**Finding E5**

- `Hành vi cũ @779`
  - Evaluate sample-by-sample với `batch_size=1`
- `Hành vi hiện tại`
  - `evaluation.batch_size=4`, nhưng vẫn loop từng sample trong batch
- `Vì sao có thể làm đổi điểm test_alter`
  - Nếu không có sample lỗi và generation deterministic, semantic gần như giữ nguyên; khác biệt chủ yếu ở throughput
- `Hướng ảnh hưởng`
  - Thấp
- `Bằng chứng`
  - `779`:
    - `scripts/test_infer.py:238`
  - Hiện tại:
    - `scripts/test_infer.py:296`

### 3.4. Checkpoint load contract

**Finding E6**

- `Hành vi cũ @779`
  - Load base model, load bridge nếu qformer, rồi load LoRA adapter
- `Hành vi hiện tại`
  - InternVL path vẫn giữ contract đó; thêm prompt-state log, device/eval-state log
- `Vì sao có thể làm đổi điểm test_alter`
  - Các log/device checks này thiên về hygiene hơn là đổi semantics
- `Hướng ảnh hưởng`
  - Thấp
- `Bằng chứng`
  - `scripts/test_infer.py` old vs current

### 3.5. Metric input/output contract

**Finding E7**

- `Hành vi cũ @779`
  - Metric target field là `raw_text` khi `response_format=direct_text`
  - Save `prediction` và `ground_truth`
- `Hành vi hiện tại`
  - Vẫn chấm `raw_text`
  - Thêm file `_pairs.json` chứa `ground_truth/generation`
- `Vì sao có thể làm đổi điểm test_alter`
  - Không đổi semantics chấm điểm
- `Hướng ảnh hưởng`
  - Không đáng kể
- `Bằng chứng`
  - `scripts/test_infer.py` old/current

## 4. Danh sách khác biệt rủi ro cao

1. **Prompt QA trong mixed train chưa khớp `779`**
   - Current: `"based on this frame"`
   - Old: `"based on the final frame"`
   - Nhãn: `runtime-confirmed`

2. **Validation during training đã đổi từ cap 200 samples sang full 1013 samples**
   - Có thể đổi mạnh val-loss trajectory, checkpoint selection, và diễn giải overfit
   - Nhãn: `runtime-confirmed`

3. **`test_alter` hiện tại skip sample lỗi thay vì resample**
   - Đây là khác biệt semantic trực tiếp ở evaluation nếu test có sample lỗi
   - Nhãn: `code-evidenced`

4. **Dataset train/val hiện là hai dataset instance riêng thay vì `Subset(train_dataset)`**
   - Tự nó chưa chắc gây lệch lớn, nhưng vì code hiện tại có logic phụ thuộc `split`, đây không còn là khác biệt cấu trúc thuần túy
   - Nhãn: `code-evidenced`

## 5. Danh sách khác biệt rủi ro thấp

1. **Split mixed cơ bản hiện tại khớp `779`**
   - same train/val counts
   - same QA/alter distribution

2. **Weighted sampler có tồn tại trong code nhưng mixed config đang tắt**
   - không làm đổi behavior hiện tại

3. **Generation config của `test_alter` giữ nguyên**
   - `beam`, `sampling`, `repetition_penalty`, `early_stopping` khớp

4. **Preprocessing và Q-Former bridge không có diff so với `779`**
   - không phải nguồn lệch chính

5. **Resume mixed config đã quay lại `legacy_skip_only`**
   - không còn là khác biệt chính nếu resume theo mixed config này

6. **Token/prompt/sample-id/debug logs**
   - tăng observability, không đổi semantic input/output

## 6. Thứ tự ablation nên làm

1. **Sửa prompt QA mixed để khớp `779` thật sự**
   - đây là khác biệt semantic rõ nhất còn sót lại trong train

2. **Nếu muốn so hoàn toàn công bằng với `779`, khôi phục val cap 200 cho mixed**
   - ít nhất là để so val-loss trajectory và checkpoint selection

3. **Sửa `scripts/test_infer.py` để test path dùng cùng error policy resample như `779`**
   - tránh việc test sample lỗi bị bỏ qua

4. **Chỉ sau khi 3 điểm trên đã khớp, mới đánh giá tiếp resume giữa epoch hoặc các hygiene khác**

## Ghi chú về độ tin cậy

- `runtime-confirmed`
  - counts train/val hiện tại và reconstructed `779`
  - prompt mẫu hiện tại
  - collate smoke hiện tại
- `code-evidenced`
  - test error handling hiện tại
  - generation contract
  - dataset instance shape
- `inference-only`
  - helper gradient checkpointing hiện tại có thể khác activation path cũ, nhưng chưa có log runtime cụ thể cho mốc so sánh này

