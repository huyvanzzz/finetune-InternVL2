# Audit `train` và `test_alter`: `779cc7b` vs code hiện tại

## Tóm tắt

Đợt audit này chỉ tập trung vào hai luồng có thể làm lệch kết quả cuối:

- `train`
- `test_alter`

Mốc cũ:

- commit `779cc7b2284c8fa480ef1d5cc91a89c5f21ee862`

Mốc hiện tại:

- code path `InternVL` mixed trên `feature/sailvl-core`
- config đang đối chiếu: `internvl_config_mixed.yaml`

Kết luận ngắn:

1. Sau các lần sửa gần đây, **những khác biệt train có rủi ro cao nhất đã được kéo về rất gần mốc `779`**:
   - split train/val mixed tự nhiên
   - không stratify
   - không weighted sampler
   - prompt `alter` fixed legacy kiểu `779`
   - prompt `QA` legacy kiểu `779`
   - `resume_runtime_mode=legacy_skip_only`
   - test dataset dùng lại `resample`

2. Tuy vậy, **repo hiện tại vẫn chưa giống hệt `779`**. Vẫn còn một số khác biệt train/test thật sự tồn tại, gồm cả nhỏ lẫn lớn.

3. Trong các khác biệt còn lại, **thứ đáng nghi nhất cho việc lệch kết quả mạnh không còn là prompt cũ/mới nữa**, mà là:
   - resume giữa epoch
   - một số thay đổi trong train loop / optimizer-state hygiene / save-load path
   - test path hiện đại hóa (`batch_size=4`, lọc `None`, ghi thêm artifact, thêm eval wrapper)
   - khác biệt ở identity của sample trong split hiện tại (`questionId` cục bộ theo split) so với `Subset` cũ

4. Nếu điểm vẫn lệch mạnh sau khi đã kéo data/prompt về cũ, thì **khả năng cao nguyên nhân nằm ở phần runtime train/resume nhiều hơn là ở prompt wording đơn lẻ**.

---

## 1. Ma trận khác biệt của `train`

### 1.1 Data train/val

### Finding T1
- `Severity`: high
- `Area`: data
- `Hành vi cũ @779`: load toàn bộ `train.json`, split bằng `train_test_split(..., random_state=seed)` trực tiếp trên index của dataset gốc, rồi bọc bằng `Subset`.  
  `Code-evidenced`: `wad_dataset.py` @779 dòng 238-269.
- `Hành vi hiện tại`: load `train.json`, sau đó lọc theo `train_task_filter/val_task_filter`, rồi split trên `filtered_train_samples`, tạo **2 dataset instance riêng** cho `train` và `val`.  
  `Code-evidenced`: `wad_dataset.py` dòng 415-503.
- `Runtime-confirmed`: config mixed hiện tại đang để:
  - `train_task_filter=all`
  - `val_task_filter=all`
  - `stratify_split=false`
- `Runtime-confirmed`: số lượng hiện tại:
  - raw: `alter=8571`, `qa=1553`
  - train: `alter=7696`, `qa=1415`
  - val: `alter=875`, `qa=138`
- `Vì sao có thể ảnh hưởng`: nếu filter/split khác, toàn bộ distribution train/val sẽ khác.
- `Hướng ảnh hưởng kỳ vọng`: hiện tại với mixed config mới, **semantics split đã khớp rất sát 779**.

### Finding T2
- `Severity`: low
- `Area`: data
- `Hành vi cũ @779`: `Subset(train_dataset, train_indices)` giữ nguyên mapping index của dataset gốc; `questionId` thực tế đi theo index gốc được `Subset` chuyển vào dataset.  
  `Code-evidenced`: `wad_dataset.py` @779 dòng 156-163, 256-258.
- `Hành vi hiện tại`: tạo dataset train/val mới từ list đã cắt; `questionId=str(idx)` là **idx cục bộ trong split mới**.  
  `Code-evidenced`: `wad_dataset.py` dòng 286-297, 471-494.
- `Vì sao có thể ảnh hưởng`: không đổi gradient, nhưng đổi trace log / sample-id debug / so sánh resume.
- `Hướng ảnh hưởng kỳ vọng`: chủ yếu bookkeeping, không phải nguyên nhân chính làm lệch chất lượng.

### Finding T3
- `Severity`: medium
- `Area`: data
- `Hành vi cũ @779`: mọi exception trong `__getitem__` đều `random.randint(...); return self.__getitem__(new_idx)` cho cả train/val/test.  
  `Code-evidenced`: `wad_dataset.py` @779 dòng 166-168.
- `Hành vi hiện tại`: có nhánh `skip` cho `val/test`, nhưng mixed config hiện tại đã đặt `non_train_error_policy=resample`; thêm nữa `scripts/test_infer.py` nay đã truyền giá trị này vào dataset test.  
  `Code-evidenced`: `wad_dataset.py` dòng 323-330; `scripts/test_infer.py` dòng 283-292.
- `Vì sao có thể ảnh hưởng`: nếu `skip` thì tập test thực chấm có thể khác.
- `Hướng ảnh hưởng kỳ vọng`: **đã được kéo về gần 779**; khác biệt này hiện không còn là thủ phạm lớn.

### 1.2 Prompt / supervision contract

### Finding T4
- `Severity`: high
- `Area`: prompt
- `Hành vi cũ @779`: prompt `alter` cố định:
  - `Describe the scene for a visually impaired user based on the final frame.`
  - phần hướng dẫn immediate obstacles / safe direction / action  
  `Code-evidenced`: `wad_dataset.py` @779 dòng 112-145.
- `Hành vi hiện tại`: mixed config đặt `direct_text_alter_prompt_mode=fixed_779`; runtime prompt mẫu hiện tại khớp đúng prompt trên.  
  `Runtime-confirmed`: smoke prompt hiện tại in ra `selected_prompt_id=legacy_779` với nội dung `final frame`.
- `Vì sao có thể ảnh hưởng`: prompt alter đi trực tiếp vào toàn bộ nhánh alter.
- `Hướng ảnh hưởng kỳ vọng`: **đã khớp 779**.

### Finding T5
- `Severity`: high
- `Area`: prompt
- `Hành vi cũ @779`: prompt `QA` cũng dùng base `Describe the scene ... based on the final frame.`, sau đó thêm:
  - focus obstacles / nearby people / free walking space / direction / safety
  - `Question: ...`
  - `Answer the question directly in natural language.`  
  `Code-evidenced`: `wad_dataset.py` @779 dòng 124-139.
- `Hành vi hiện tại`: mixed config đặt `direct_text_qa_prompt_mode=legacy_779`; runtime sample QA hiện tại khớp lại prompt cũ này.  
  `Runtime-confirmed`: smoke prompt QA hiện tại in ra `qa_legacy_779` với `based on the final frame`.
- `Vì sao có thể ảnh hưởng`: mixed train dùng QA khá nhiều mẫu, nên prompt QA khác sẽ đổi gradient thật.
- `Hướng ảnh hưởng kỳ vọng`: **khác biệt này đã được sửa về cũ**.

### Finding T6
- `Severity`: low
- `Area`: prompt
- `Hành vi cũ @779`: `qformer_text = text_content.strip()`.  
  `Code-evidenced`: `wad_dataset.py` @779 dòng 157-160.
- `Hành vi hiện tại`: vẫn là `qformer_text = text_content.strip()`.  
  `Code-evidenced`: `wad_dataset.py` dòng 286-289.
- `Vì sao có thể ảnh hưởng`: nếu khác sẽ đổi visual instruction cho Q-Former.
- `Hướng ảnh hưởng kỳ vọng`: không khác.

### 1.3 Split / sampling

### Finding T7
- `Severity`: high
- `Area`: split
- `Hành vi cũ @779`: split thường, không `stratify`.  
  `Code-evidenced`: `wad_dataset.py` @779 dòng 246-254.
- `Hành vi hiện tại`: mixed config đặt `stratify_split=false`, và runtime in ra `Using plain random split without task stratification.`  
  `Code-evidenced`: `wad_dataset.py` dòng 439-449.
- `Vì sao có thể ảnh hưởng`: stratified split đổi composition train/val.
- `Hướng ảnh hưởng kỳ vọng`: **đã khớp 779**.

### Finding T8
- `Severity`: high
- `Area`: sampler
- `Hành vi cũ @779`: `DataLoader(... shuffle=True)` cho train, không sampler riêng.  
  `Code-evidenced`: `train.py` @779 dòng 643-645.
- `Hành vi hiện tại`: có hẳn `build_train_sampler()` và `WeightedRandomSampler`, nhưng mixed config đặt `task_balancing.enabled=false`, nên runtime path quay về `shuffle=True`.  
  `Code-evidenced`: `train.py` dòng 361-394, 936-942.
- `Vì sao có thể ảnh hưởng`: weighted sampling đổi hẳn distribution effective trong epoch.
- `Hướng ảnh hưởng kỳ vọng`: với mixed config mới, **đã khớp 779 về semantics**.

### Finding T9
- `Severity`: low
- `Area`: val-in-train
- `Hành vi cũ @779`: config cũ `internvl_config.yaml` để `eval_limit` mặc định hữu hiệu là `200`; trong `eval_model` cũng break sau `200` batch.  
  `Code-evidenced`: `wad_dataset.py` @779 dòng 262-267; `train.py` @779 dòng 339-370.
- `Hành vi hiện tại`: mixed config đặt `eval_limit=0` nên val dataset full `1013`; `eval_model` hiện vẫn break sau 200 batch.  
  `Code-evidenced`: `internvl_config_mixed.yaml` dòng 60; `train.py` dòng 468-470 trong file hiện tại có log limit ở dataset, còn `eval_model` break logic vẫn tồn tại.
- `Vì sao có thể ảnh hưởng`: chủ yếu đổi tập val được duyệt trong eval-in-train và thời gian chạy.
- `Hướng ảnh hưởng kỳ vọng`: ảnh hưởng chủ yếu tới log/monitoring, không phải nguyên nhân mạnh nhất của lệch `test_alter`.

### 1.4 Collate / input build

### Finding T10
- `Severity`: medium
- `Area`: collate
- `Hành vi cũ @779`: collate chèn `<img><IMG_CONTEXT>...` theo `model.num_image_token * num_patches`, ghép `question + answer + eos`, và repeat `qformer_text` theo số tile.  
  `Code-evidenced`: `train.py` @779 dòng 242-304.
- `Hành vi hiện tại`: logic cốt lõi này vẫn giữ nguyên.  
  `Code-evidenced`: `train.py` dòng 267-359.
- `Runtime-confirmed`: collate smoke hiện tại cho batch 2 mẫu:
  - `input_ids_shape=(2, 58)`
  - `labels_shape=(2, 58)`
  - `attention_mask_shape=(2, 58)`
  - `pixel_values_shape=(2, 3, 448, 448)`
  - `qformer_input_shape=(2, 1)`
  - `total_image_tokens_sample0=32`
- `Vì sao có thể ảnh hưởng`: nếu khác thì sequence train thực sự khác.
- `Hướng ảnh hưởng kỳ vọng`: phần input build cốt lõi hiện **không phải khác biệt lớn**.

### Finding T11
- `Severity`: low
- `Area`: collate
- `Hành vi cũ @779`: chỉ có log token stats.  
  `Code-evidenced`: `train.py` @779 dòng 265-282.
- `Hành vi hiện tại`: thêm log prompt sample, sample-id debug, RNG digest debug.  
  `Code-evidenced`: `train.py` dòng 336-357, 595-660.
- `Vì sao có thể ảnh hưởng`: bản thân log không đổi gradient, nhưng nếu bật quá nhiều có thể làm train chậm hơn.
- `Hướng ảnh hưởng kỳ vọng`: bookkeeping / observability là chính.

### 1.5 Model setup lúc train

### Finding T12
- `Severity`: low
- `Area`: model
- `Hành vi cũ @779`: base model `OpenGVLab/InternVL2_5-2B`, Q-Former bật, LoRA target như config cũ.  
  `Code-evidenced`: `internvl_config.yaml` @779; `train.py` @779 dòng 579-623.
- `Hành vi hiện tại`: mixed config vẫn giữ cùng backbone, cùng Q-Former, cùng target modules LoRA.  
  `Code-evidenced`: `internvl_config_mixed.yaml`.
- `Vì sao có thể ảnh hưởng`: nếu backbone hay LoRA target khác thì lệch lớn.
- `Hướng ảnh hưởng kỳ vọng`: hiện không khác về semantic chính.

### Finding T13
- `Severity`: low
- `Area`: model
- `Hành vi cũ @779`: `preprocessing.py` và `qformer_bridge.py` không có diff so với logic đang dùng cho InternVL.  
  `Code-evidenced`: `git diff 779... -- preprocessing.py qformer_bridge.py` trả về rỗng.
- `Hành vi hiện tại`: không đổi.
- `Vì sao có thể ảnh hưởng`: answer mapping / bridge là hai chỗ nếu đổi sẽ lệch mạnh.
- `Hướng ảnh hưởng kỳ vọng`: không phải nguyên nhân.

### Finding T14
- `Severity`: medium
- `Area`: model/runtime
- `Hành vi cũ @779`: `gradient_checkpointing_enable()` được gọi trực tiếp khi config bật.  
  `Code-evidenced`: `train.py` @779 dòng 595-597.
- `Hành vi hiện tại`: đi qua wrapper `enable_gradient_checkpointing(...)`.  
  `Code-evidenced`: `train.py` dòng 853-856.
- `Vì sao có thể ảnh hưởng`: nếu wrapper đổi target module được bật checkpointing thì runtime/path có thể khác.
- `Hướng ảnh hưởng kỳ vọng`: cần xem lại nếu nghi model runtime khác, nhưng chưa có bằng chứng nó đổi semantics trên InternVL.

### 1.6 Optimization / eval-in-train / resume

### Finding T15
- `Severity`: high
- `Area`: resume
- `Hành vi cũ @779`: resume giữa epoch chỉ `next(batch_iterator)` để skip batch đã đi qua; không restore runtime RNG state.  
  `Code-evidenced`: `train.py` @779 dòng 476-480.
- `Hành vi hiện tại`: code có hỗ trợ `runtime_state.pt`, restore RNG, debug digest... nhưng mixed config đang ép `resume_runtime_mode=legacy_skip_only`, nên khi chạy mixed sẽ quay lại hành vi cũ hơn.  
  `Code-evidenced`: `train.py` dòng 146-151, 583-594, 670-691; `internvl_config_mixed.yaml` dòng 94.
- `Vì sao có thể ảnh hưởng`: đây là khác biệt cực mạnh nếu có lúc bạn resume bằng mode mới; còn nếu mixed config hiện tại thật sự chạy `legacy_skip_only` thì phần này đã gần lại 779.
- `Hướng ảnh hưởng kỳ vọng`: **nếu cùng checkpoint nhưng resume giữa epoch bằng mode mới/cũ khác nhau thì có thể lệch mạnh**.

### Finding T16
- `Severity`: medium
- `Area`: optimizer
- `Hành vi cũ @779`: có sanitize optimizer state và move state về đúng device, nhưng chưa có `copy.deepcopy`, mismatch counter, hay defensive checks trước `optimizer.step()`.  
  `Code-evidenced`: `train.py` @779 dòng 152-219, 442-460, 544-563.
- `Hành vi hiện tại`: thêm:
  - `copy.deepcopy` khi export state
  - đếm mismatch device
  - move state trước `optimizer.step()`
  - runtime checks chặt hơn  
  `Code-evidenced`: `train.py` dòng 152-235, 717-733.
- `Vì sao có thể ảnh hưởng`: chủ yếu là hygiene/fix portability; bình thường không nên đổi gradient nếu mọi thứ vốn hợp lệ.
- `Hướng ảnh hưởng kỳ vọng`: medium-low, trừ khi trước đó optimizer state thực sự bị sai device/dtype.

### Finding T17
- `Severity`: low
- `Area`: optimization
- `Hành vi cũ @779`: `num_epochs=3`, `evaluation.batch_size=1`.  
  `Code-evidenced`: `internvl_config.yaml` @779.
- `Hành vi hiện tại`: mixed config đang để `num_epochs=5`, `evaluation.batch_size=4`.  
  `Code-evidenced`: `internvl_config_mixed.yaml` dòng 68-104.
- `Vì sao có thể ảnh hưởng`: khi so cùng epoch thì `num_epochs` tổng không phải nguyên nhân chính; `eval batch size` chỉ tác động nhẹ tới throughput test path.
- `Hướng ảnh hưởng kỳ vọng`: thấp nếu đang so checkpoint cùng epoch.

---

## 2. Ma trận khác biệt của `test_alter`

### 2.1 Test dataset contract

### Finding E1
- `Severity`: medium
- `Area`: test dataset
- `Hành vi cũ @779`: `test_dataset = WADDatasetForInternVL(... split='test', response_format=response_format)`; mọi lỗi sample sẽ tự resample do dataset code cũ.  
  `Code-evidenced`: `scripts/test_infer.py` @779 dòng 228-234; `wad_dataset.py` @779 dòng 166-168.
- `Hành vi hiện tại`: dataset test được truyền thêm:
  - `direct_text_alter_prompt_mode`
  - `direct_text_qa_prompt_mode`
  - `non_train_error_policy`
  - `seed`  
  `Code-evidenced`: `scripts/test_infer.py` dòng 283-292.
- `Vì sao có thể ảnh hưởng`: giờ test path phụ thuộc explicit vào config mixed, thay vì ngầm ăn default cũ.
- `Hướng ảnh hưởng kỳ vọng`: với mixed config hiện tại, đây là **điều tốt** vì đưa test semantics về gần 779 hơn.

### Finding E2
- `Severity`: medium
- `Area`: test dataset
- `Hành vi cũ @779`: `TestCollaterFn` trả nguyên `batch`.  
  `Code-evidenced`: `scripts/test_infer.py` @779 dòng 76-82.
- `Hành vi hiện tại`: `TestCollaterFn` lọc `[sample for sample in batch if sample is not None]`.  
  `Code-evidenced`: `scripts/test_infer.py` dòng 81-86.
- `Vì sao có thể ảnh hưởng`: nếu có sample lỗi và policy `skip`, test set thực chấm sẽ khác.
- `Hướng ảnh hưởng kỳ vọng`: với mixed config hiện tại (`resample`) thì rủi ro giảm mạnh, nhưng code path vẫn khác về mặt cơ chế.

### 2.2 Prompt infer contract

### Finding E3
- `Severity`: high
- `Area`: prompt infer
- `Hành vi cũ @779`: prompt test phụ thuộc hoàn toàn vào `WADDatasetForInternVL`; vì dataset cũ dùng fixed prompt train-style nên infer cũng đi fixed prompt đó.  
  `Code-evidenced`: `scripts/test_infer.py` @779 dòng 228-234 + `wad_dataset.py` @779 dòng 112-145.
- `Hành vi hiện tại`: mixed config đã truyền rõ `fixed_779 + legacy_779`, nên prompt infer hiện tại khớp lại train semantic cũ hơn.  
  `Runtime-confirmed`: prompt alter hiện tại là fixed `final frame`; QA prompt hiện tại là `legacy_779`.
- `Vì sao có thể ảnh hưởng`: prompt infer khác train là một nguồn lệch score lớn.
- `Hướng ảnh hưởng kỳ vọng`: khác biệt này hiện đã được kéo về cũ.

### 2.3 Generation contract

### Finding E4
- `Severity`: low
- `Area`: generation
- `Hành vi cũ @779`:  
  - `max_new_tokens=512`
  - `num_beams=3`
  - `do_sample=False`
  - `repetition_penalty=1.3`
  - `early_stopping=True`  
  `Code-evidenced`: `scripts/test_infer.py` @779 dòng 259-265.
- `Hành vi hiện tại`: y hệt.  
  `Code-evidenced`: `scripts/test_infer.py` dòng 325-331.
- `Vì sao có thể ảnh hưởng`: generation config mà đổi thì điểm test đổi mạnh.
- `Hướng ảnh hưởng kỳ vọng`: không khác.

### Finding E5
- `Severity`: low
- `Area`: generation
- `Hành vi cũ @779`: batch test cố định `1`, loop mỗi batch lấy `sample = batch[0]`.  
  `Code-evidenced`: `scripts/test_infer.py` @779 dòng 236-255.
- `Hành vi hiện tại`: `evaluation.batch_size=4`, rồi lặp từng sample trong batch.  
  `Code-evidenced`: `scripts/test_infer.py` dòng 295-347.
- `Vì sao có thể ảnh hưởng`: bình thường không đổi output logic vì vẫn generate từng sample một, nhưng có thể đổi throughput và cách skip sample rỗng.
- `Hướng ảnh hưởng kỳ vọng`: thấp.

### 2.4 Checkpoint load contract

### Finding E6
- `Severity`: medium
- `Area`: checkpoint load
- `Hành vi cũ @779`: load base model, override `system_message`, attach bridge nếu bật qformer, rồi load LoRA + bridge.  
  `Code-evidenced`: `scripts/test_infer.py` @779 dòng 167-203.
- `Hành vi hiện tại`: thêm:
  - `log_runtime_prompt_state`
  - `log_eval_state`
  - `align_language_model_devices`
  - path backend cho SAIL  
  `Code-evidenced`: `scripts/test_infer.py` dòng 27-48, 199-252.
- `Vì sao có thể ảnh hưởng`: trên InternVL, phần backend SAIL không liên quan; nhưng device alignment thêm vào có thể đổi lỗi/runtime chứ không nên đổi semantic output.
- `Hướng ảnh hưởng kỳ vọng`: medium-low.

### 2.5 Metric input/output contract

### Finding E7
- `Severity`: low
- `Area`: metrics/output
- `Hành vi cũ @779`: lưu `results.json` với `question / prediction / ground_truth`.  
  `Code-evidenced`: `scripts/test_infer.py` @779 dòng 312-322.
- `Hành vi hiện tại`: vẫn lưu file đó, và lưu thêm file `_pairs.json` cho cặp `ground_truth / generation`.  
  `Code-evidenced`: `scripts/test_infer.py` dòng 369-391.
- `Vì sao có thể ảnh hưởng`: không đổi điểm, chỉ thêm artifact.
- `Hướng ảnh hưởng kỳ vọng`: bookkeeping.

---

## 3. Danh sách khác biệt rủi ro cao

1. `Resume giữa epoch`
- Nếu có lần chạy dùng runtime-state restore khác mode `legacy_skip_only`, hoặc checkpoint được tạo ở mode khác, quỹ đạo train có thể lệch mạnh.

2. `Khác biệt còn sót ở train loop hiện đại`
- Dù data/prompt đã kéo về cũ, code train hiện tại vẫn có thêm nhiều tầng runtime hygiene quanh optimizer/resume/device state.

3. `Cơ chế test hiện đại hóa`
- `batch_size=4`, lọc `None`, truyền explicit policy vào dataset test.
- Phần này chủ yếu tốt hơn, nhưng vẫn là khác biệt thực so với `779`.

4. `questionId` / split identity hiện tại`
- Không đổi gradient, nhưng làm trace resume/debug không còn map 1-1 với cách cũ.

---

## 4. Danh sách khác biệt rủi ro thấp

- `preprocessing.py` không đổi
- `qformer_bridge.py` không đổi
- generation config `test_alter` không đổi
- prompt `alter` mixed hiện đã khớp lại
- prompt `QA` mixed hiện đã khớp lại
- split mixed hiện đã khớp lại
- sampler mixed hiện đã khớp lại
- `qformer_text` mirror prompt như cũ
- thêm log token/prompt/sample-id/RNG digest
- thêm file `_pairs.json` khi test

---

## 5. Những lý do khả nghi nhất khiến kết quả vẫn lệch nhiều

### H1. Resume/runtime semantics vẫn là nghi phạm số 1
- `Code-evidenced`
- Nếu trước đây một run đã từng resume giữa epoch theo code mới hoặc từ checkpoint sinh ra dưới runtime khác, thì chỉ riêng prompt kéo về cũ là chưa đủ.

### H2. Bạn đang so checkpoint được tạo dưới hai runtime train khác nhau
- `Inference-only`
- Dù config mixed hiện tại gần `779`, checkpoint cũ đã train ra trước đó có thể đã chịu tác động của code path trung gian khác.

### H3. Test path hiện đã gần `779` hơn, nên nếu điểm vẫn lệch mạnh thì thủ phạm nhiều khả năng nằm ở train
- `Code-evidenced + runtime-confirmed`

### H4. Phần khác biệt “nhỏ nhưng cộng dồn” trong optimizer/resume/device hygiene có thể đủ làm quỹ đạo LoRA rẽ hướng
- `Inference-only`
- Đây không phải kiểu khác 1 dòng prompt là đủ giải thích, mà là sai khác động học train.

---

## 6. Thứ tự ablation nên làm tiếp

1. Chạy lại **train từ đầu, không resume**, bằng mixed config hiện tại, để loại hẳn yếu tố resume.
2. Nếu vẫn lệch mạnh, so trực tiếp log train 1 epoch đầu:
   - sample ids
   - prompt mẫu
   - avg loss theo step
3. Nếu vẫn lệch, tạo một mode train “779-minimal” hơn nữa:
   - tắt thêm toàn bộ logging/debug hiện đại không cần thiết
   - giữ đúng `evaluation.batch_size=1`
   - giữ đúng `print_samples=5`
4. Nếu vẫn lệch, mới quay sang audit sâu hơn phần optimizer/load/save path hoặc môi trường.

---

## 7. Kết luận cuối

Tính tới thời điểm audit này:

- **data/prompt/split/sampler** của mixed InternVL hiện tại đã được kéo về rất gần `779`
- **test prompt / generation contract** cũng đã gần lại đáng kể
- vì vậy, nếu kết quả vẫn lệch mạnh, thì **nghi phạm lớn nhất không còn là prompt wording**, mà là:
  - quỹ đạo train đã khác do resume/runtime
  - hoặc checkpoint đang so không thật sự được sinh ra dưới cùng semantics train như `779`

Nói ngắn gọn:

> Nếu bạn muốn kiểm chứng nguyên nhân nhanh nhất, hãy ưu tiên một run mixed mới từ đầu, không resume, trên code hiện tại đã kéo về `779`, rồi so lại với mốc cũ.
