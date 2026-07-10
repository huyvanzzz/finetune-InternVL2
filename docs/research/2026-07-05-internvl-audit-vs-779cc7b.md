# Audit chi tiết giữa snapshot `779cc7b` và `feature/sailvl-core`

## Summary

Mục tiêu của đợt audit này là đối chiếu **toàn bộ semantics train / infer / setup model** giữa:

- `A`: snapshot repo tại commit `779cc7b2284c8fa480ef1d5cc91a89c5f21ee862`
- `B`: code path InternVL hiện tại trên `feature/sailvl-core`, cụ thể là notebook mixed:
  - `run_qformer.ipynb`
  - `run_no_qformer.ipynb`
  - `internvl_config_mixed.yaml`
  - `internvl_config_no_qformer_mixed.yaml`

Mục tiêu của báo cáo này không phải tìm “khác biệt lớn” thôi, mà là ghi lại **tất cả khác biệt tìm thấy**, kể cả nhỏ, rồi phân loại:

- `bookkeeping / logging / hygiene`
- `semantic`
- `unknown-until-runtime`

### Kết luận tổng quát

1. `feature/sailvl-core` hiện tại **không trùng semantics** với snapshot `779cc7b`.
2. Phần chênh lớn nhất nằm ở:
   - notebook entry contract
   - data / prompt / split contract
   - validation / test protocol
   - resume / runtime hygiene
3. `preprocessing.py` và `qformer_bridge.py` **không có diff thực chất** so với `779cc7b`, nên khả năng cao không phải gốc rễ của sự lệch kết quả.
4. Nếu chỉ ưu tiên theo khả năng làm lệch quality, thứ tự đáng nghi nhất là:
   - prompt QA
   - val / infer protocol
   - epoch budget
   - notebook split test đang chạy (`test_QA` cũ vs `test_alter` hiện tại)
   - task filtering / split / sampler stack mới

### Runtime truth check đã làm

- Đã xác nhận bằng notebook cells rằng `feature/sailvl-core` đang trỏ vào:
  - `BRANCH = "feature/sailvl-core"`
  - `CONFIG_PATH = "internvl_config_mixed.yaml"` hoặc `internvl_config_no_qformer_mixed.yaml`
- Đã chạy smoke check config mixed hiện tại và xác nhận:
  - `train_task_filter=all`
  - `val_task_filter=all`
  - `prompt_mode=fixed_legacy`
  - `stratify_split=false`
  - `eval_limit=0`
  - `num_epochs=5`
  - `eval_batch_size=4`
- Không thể build dataset đến cuối trên máy local này vì thiếu:
  - `./wad_dataset/frame_index.pkl`
- Vì vậy:
  - `notebook/config`: `runtime-confirmed`
  - `dataset prompt sample`: `code-evidenced`, chưa `runtime-confirmed` trên máy local này

---

## Behavior Difference Matrix

## 1. Notebook entry contract

| Trục | Snapshot `779cc7b` | `feature/sailvl-core` hiện tại | Diff type | Đánh giá |
|---|---|---|---|---|
| Repo branch | `main` | `feature/sailvl-core` | semantic | Khác toàn bộ code path được pull trên Kaggle |
| Q-Former config handoff | `run_qformer.ipynb` không đặt `CONFIG_PATH`, train bằng `python train.py` => dùng default `internvl_config.yaml` | có `CONFIG_PATH = "internvl_config_mixed.yaml"` và pass `--config` rõ ràng | semantic | đổi setup train thực tế |
| no-QFormer config handoff | `run_no_qformer.ipynb` tạo file `internvl_config_no_qformer.yaml` từ `internvl_config.yaml` ngay trong notebook | dùng file có sẵn `internvl_config_no_qformer_mixed.yaml` | semantic | đổi nguồn config và cách notebook định nghĩa setup |
| Test split | cả 2 notebook cũ test `test_QA` | cả 2 notebook mixed hiện tại test `test_alter` | semantic | đổi tập đánh giá cuối |
| Test output file | `*_test_QA.json` | `*_test_alter.json` | bookkeeping + semantic context | đổi artifact và target task |
| Notebook compile check | cả cũ và mới đều có `py_compile` | giống | bookkeeping | không đổi semantics |

### Bằng chứng

- `779cc7b:run_qformer.ipynb`
  - `10: "BRANCH = \"main\""`
  - `98: "cmd = [\"python\", \"train.py\"]"`
  - `130: "--split", "test_QA"`
- `779cc7b:run_no_qformer.ipynb`
  - `10: "BRANCH = \"main\""`
  - `12: "CONFIG_PATH = \"internvl_config_no_qformer.yaml\""`
  - `154: "--split", "test_QA"`
- `.worktrees/sailvl-core/run_qformer.ipynb`
  - `10: BRANCH = "feature/sailvl-core"`
  - `12: CONFIG_PATH = "internvl_config_mixed.yaml"`
  - `174: "--split", "test_alter"`
- `.worktrees/sailvl-core/run_no_qformer.ipynb`
  - `10: BRANCH = "feature/sailvl-core"`
  - `12: CONFIG_PATH = "internvl_config_no_qformer_mixed.yaml"`
  - `143: "--split", "test_alter"`

---

## 2. Data contract

| Trục | Snapshot `779cc7b` | `feature/sailvl-core` hiện tại | Diff type | Đánh giá |
|---|---|---|---|---|
| Frame selection | lấy `[4,6,8]`, rồi chỉ nạp frame cuối vào model | giống | same | không phải nguồn lệch |
| Sample fields | `question`, `answer`, `qformer_text`, `pixel_values`, `questionId`, `image` | thêm `task_type`, `selected_prompt_id`, `selected_prompt_text`, `frame_path` | bookkeeping | chủ yếu thêm observability |
| Train/val construction | tạo `train_dataset` trên full `metadata["train"]`, sau đó split indices và wrap bằng `Subset` | split xong rồi tạo `train_dataset` và `val_dataset` là hai dataset instance riêng | semantic | đổi lifecycle dataset và split identity |
| Error handling | mọi split đều `random-resample` khi `__getitem__` lỗi | `val`/`test` log `[DATA ERROR]` và `return None`; train mới resample | semantic | có thể đổi val/test composition thực tế khi có sample lỗi |
| Task typing metadata | không có | có `task_type` / `task_types` | bookkeeping + phục vụ sampler | có thể kích hoạt code path mới |
| Train limit | không có | có `train_limit` hook | semantic-capable | hiện tại config không bật, nhưng code path đã thay đổi |

### Bằng chứng

- `779cc7b:wad_dataset.py`
  - `93-100`: vẫn lấy `frame_ids = self._select_frames_safe(frame_path)` và `last_frame_id = frame_ids[-1]`
  - `153-160`: sample dict cũ
  - `165-166`: `new_idx = random.randint(...)` và recursive retry
  - `248-259`: split bằng `Subset`
- `.worktrees/sailvl-core/wad_dataset.py`
  - `175-183`: vẫn frame selection cũ
  - `271-280`: sample dict mới
  - `290-297`: `val` / `test` return `None` nếu lỗi
  - `438-455`: tạo train dataset riêng
  - `447-455`: tạo val dataset riêng

---

## 3. Prompt / supervision contract

| Trục | Snapshot `779cc7b` | `feature/sailvl-core` hiện tại | Diff type | Đánh giá |
|---|---|---|---|---|
| Alter prompt | fixed legacy prompt, mở đầu bằng `based on the final frame` | `fixed_legacy` nhưng text đã đổi thành `based on this image` | semantic | thay wording alter |
| QA prompt | `Describe ... based on the final frame` + `Focus ...` + `Question:` + `Answer the question directly in natural language.` | `Describe ... based on this frame` + `Focus ...` + `Question:`, không còn dòng `Answer ...` | semantic | khác hợp đồng supervision QA |
| Prompt mode infra | không có prompt mode | có `fixed_legacy`, `fixed_v1`, `balanced_v1` | bookkeeping + semantic-capable | code path mới có thể đổi behavior nếu config đổi |
| `qformer_text` | mirror `text_content.strip()` | vẫn mirror `text_content.strip()` | same | không đổi nguyên tắc |
| Prompt train/test alter | cũ: train alter fixed, infer test theo dataset text logic cũ | mới: train alter fixed legacy theo config mixed, test alter cũng theo config mode | semantic nhẹ | logic rộng hơn và config-driven hơn |
| JSON prompt path | tồn tại trong code | tồn tại trong code, không phải mode hiện đang dùng | same | không phải nguồn lệch cho direct_text |

### Bằng chứng

- `779cc7b:wad_dataset.py`
  - `113`: `Describe the scene for a visually impaired user based on the final frame.`
  - `128-131`: QA focus + question
  - `136`: `Answer the question directly in natural language.`
- `.worktrees/sailvl-core/wad_dataset.py`
  - `18-21`: `ALTER_DIRECT_TEXT_LEGACY_PROMPT` mới dùng `based on this image`
  - `220-222`: QA prompt mới dùng `based on this frame`
  - không còn dòng `Answer the question directly in natural language.`

### Nhận định

- Đây là một trong các delta có khả năng cao nhất làm lệch kết quả mixed, vì QA vẫn nằm trong train/val setup mixed hiện tại.

---

## 4. Split / sampling contract

| Trục | Snapshot `779cc7b` | `feature/sailvl-core` hiện tại | Diff type | Đánh giá |
|---|---|---|---|---|
| Task filter | không có | `train_task_filter`, `val_task_filter` tồn tại và đang set `all` | semantic-capable | config mixed hiện tại tắt filter, nhưng code path đã đổi |
| Split type | `train_test_split(... random_state=seed)` thường | `stratify_split: false` nên cũng split thường | same at runtime | giống trong config mixed hiện tại |
| Stratify infra | không có | có `should_use_stratified_split()` | semantic-capable | không bật trong mixed config |
| Weighted sampler | không có | có `build_train_sampler()` + `WeightedRandomSampler` nếu enable | semantic-capable | mixed config hiện tại tắt |
| Eval limit | default `config['data'].get('eval_limit', 200)` | `resolve_eval_limit(0) => None`, tức full val | semantic | đổi protocol val |
| Train size | `train_split=0.9` | vẫn `0.9` | same | không phải nguồn lệch |

### Bằng chứng

- `779cc7b:wad_dataset.py`
  - `248-252`: `train_test_split(... random_state=seed)`
  - `263`: `eval_limit = config['data'].get('eval_limit', 200)`
- `779cc7b:train.py`
  - `644`: `train_loader ... shuffle=True`
  - không có `WeightedRandomSampler`
- `.worktrees/sailvl-core/internvl_config_mixed.yaml`
  - `57-58`: `train_task_filter: "all"`, `val_task_filter: "all"`
  - `59`: `stratify_split: false`
  - `60`: `eval_limit: 0`
  - `85-88`: `task_balancing.enabled: false`
- `.worktrees/sailvl-core/train.py`
  - `337-374`: code path `WeightedRandomSampler`
- `.worktrees/sailvl-core/wad_dataset.py`
  - `87-90`: `should_use_stratified_split`
  - `431-434`: val limit chỉ áp nếu `eval_limit` khác `None`

### Nhận định

- Mặc dù mixed config hiện tại đã tắt task balancing và stratify, val policy vẫn khác rõ với snapshot cũ vì `eval_limit=0` => full val.

---

## 5. Model setup contract

| Trục | Snapshot `779cc7b` | `feature/sailvl-core` hiện tại | Diff type | Đánh giá |
|---|---|---|---|---|
| Base model | `OpenGVLab/InternVL2_5-2B` | giống | same | không đổi |
| Architecture | `internvl` | giống | same | không đổi |
| Quantization | 4-bit NF4, compute `bfloat16` | giống | same | không đổi |
| Vision freeze | `freeze_encoder: true` và `model.vision_model.requires_grad_(False)` | giống | same | không đổi |
| Q-Former bridge code | không thay đổi thực chất | không thay đổi thực chất | same | không phải nguồn lệch |
| LoRA target modules | `wqkv`, `wo`, `w1`, `w2`, `w3` | giống | same | không đổi |
| Gradient checkpointing call | `model.gradient_checkpointing_enable()` trực tiếp | `enable_gradient_checkpointing(...)` helper | bookkeeping/semantic-possible | với InternVL path khả năng giống, nhưng đã đổi wrapper |
| Backend dispatch | chỉ InternVL path | có `get_backend(...)`, nhưng InternVL path vẫn không qua backend | bookkeeping | không đổi runtime InternVL trực tiếp |
| `prepare_model_for_kbit_training` | có | giống | same | không đổi |
| LoRA load/init | `build_fresh_lora_model` hoặc `PeftModel.from_pretrained` | vẫn giống nguyên tắc | same | không đổi logic cốt lõi |

### Bằng chứng

- `779cc7b:internvl_config.yaml`
  - model name, quant, lora, qformer đều giống logic hiện tại
- `779cc7b:train.py`
  - `597`: `model.gradient_checkpointing_enable()`
  - `603`: freeze vision
  - `610`: `prepare_model_for_kbit_training`
  - `617`: `PeftModel.from_pretrained`
- `.worktrees/sailvl-core/train.py`
  - `805-808`: `enable_gradient_checkpointing(...)`
  - `810-818`: freeze vision + attach qformer
  - `820`: `prepare_model_for_kbit_training`
  - `827`: `PeftModel.from_pretrained`

### Nhận định

- Về cốt lõi model runtime InternVL, khác biệt rất nhỏ.
- Đây không phải nhóm finding đáng nghi nhất.

---

## 6. Optimization contract

| Trục | Snapshot `779cc7b` | `feature/sailvl-core` hiện tại | Diff type | Đánh giá |
|---|---|---|---|---|
| `num_epochs` | `3` | `5` cho qformer mixed | semantic | thay epoch budget rõ ràng |
| Train batch size | `2` | `2` | same | không đổi |
| Grad accumulation | `8` | `8` | same | không đổi |
| LR / proj LR | `2e-4` / `5e-4` | giống | same | không đổi |
| Warmup / scheduler | giống | giống | same | không đổi |
| Eval cadence | `2000` | `2000` | same | không đổi |
| Save cadence | `2000` | `2000` | same | không đổi |
| Metrics file | có `metrics.json` trong `train.py` snapshot này | vẫn có và phong phú hơn | bookkeeping | không đổi optimization |

### Bằng chứng

- `779cc7b:internvl_config.yaml`
  - `61`: `num_epochs: 3`
  - `62-78`: training core params
- `.worktrees/sailvl-core/internvl_config_mixed.yaml`
  - `66`: `num_epochs: 5`
  - `67-83`: phần lớn params giống cũ

### Nhận định

- Khác biệt quan trọng nhất ở nhóm này là `3 epoch` vs `5 epoch`.

---

## 7. Validation / infer / metrics contract

| Trục | Snapshot `779cc7b` | `feature/sailvl-core` hiện tại | Diff type | Đánh giá |
|---|---|---|---|---|
| Val loss cap | trong `eval_model`, stop sau `200` batches | không còn hard stop 200 | semantic | đổi protocol val |
| Eval batch size config | `1` | `4` | semantic | đổi protocol đánh giá |
| Test loader batch size | hard-coded `1` | lấy từ `config['evaluation'].get('batch_size', 1)` | semantic | có thể đổi throughput và empty-batch behavior |
| Test split notebook | `test_QA` | `test_alter` | semantic | đổi tập benchmark |
| Generation config | `max_new_tokens=512`, `num_beams=3`, `do_sample=False`, `repetition_penalty=1.3`, `early_stopping=True` | giống | same | không đổi |
| `print_samples` | default parser = `5` | default parser = `None`, rồi fallback từ config `print_samples=3` | bookkeeping + observability | không đổi model, chỉ đổi log |
| Output artifact | chỉ `output_file` json | thêm file `*_pairs.json` lưu `ground_truth/generation` | bookkeeping | không đổi metric |
| Empty sample handling khi test | `TestCollaterFn` trả thẳng batch | collater lọc `None`, loop đếm `skipped_samples` | semantic nhẹ | có thể đổi số sample thực chấm nếu có sample lỗi |

### Bằng chứng

- `779cc7b:train.py`
  - `367`: `if total_eval_batchs == 200:`
- `779cc7b:internvl_config.yaml`
  - `88`: `evaluation.batch_size: 1`
- `779cc7b:scripts/test_infer.py`
  - `238`: `batch_size=1`
  - `260-264`: generation config
  - `111`: `--output_file`
- `.worktrees/sailvl-core/internvl_config_mixed.yaml`
  - `101`: `batch_size: 4`
  - `102`: `print_samples: 3`
- `.worktrees/sailvl-core/scripts/test_infer.py`
  - `123`: `print_samples` parser default `None`
  - `186-187`: fallback từ config
  - `296`: `batch_size=config['evaluation'].get('batch_size', 1)`
  - `324-328`: generation config giống cũ
  - `393-401`: ghi thêm file pairs

---

## 8. Resume / runtime hygiene contract

| Trục | Snapshot `779cc7b` | `feature/sailvl-core` hiện tại | Diff type | Đánh giá |
|---|---|---|---|---|
| Resume step skip | có skip batch theo `start_step` | vẫn có | same-ish | vẫn giữ logic cũ |
| RNG runtime state | không có `runtime_state.pt` | có `save_runtime_state_file` / `load_runtime_state_file` / `restore_full_runtime_state` | semantic cho resume, bookkeeping với train mới | đổi behavior resume |
| Debug sample IDs | không có | có `debug_log_sample_ids` | bookkeeping | observability |
| Debug RNG digest | không có | có `debug_log_rng_digest` | bookkeeping | observability |
| Prompt logging | chỉ token stats | có thêm prompt sample logging | bookkeeping | observability |
| Run summary / data summary | không có | có | bookkeeping | observability |
| Optimizer sanitization | đã có | vẫn có + deepcopy khi export | bookkeeping / resume hygiene | giảm lỗi resume cross-env |
| `set_epoch()` dataset | không có | có gọi `train_loader.dataset.set_epoch(epoch)` | semantic-capable | chỉ tác động khi prompt mode balance |

### Bằng chứng

- `779cc7b:train.py`
  - có `start_step` skip
  - không import `resume_state`, `resume_debug_tools`
- `.worktrees/sailvl-core/train.py`
  - `24-26`: import runtime state helpers
  - `514-515`: debug flags
  - `597-601`: load runtime state
  - `610`: `train_loader.dataset.set_epoch(epoch)`
  - `629-633`: restore runtime state khi resume mid-epoch
  - `736-760`: save runtime state vào checkpoint

### Nhận định

- Nhóm này rất quan trọng nếu so sánh `resume vs uninterrupted`.
- Nếu chỉ so `train from scratch`, đây không phải nhóm khả nghi nhất.

---

## Exhaustive Difference Log

## High severity

### F01
- `Severity`: high
- `Area`: notebook
- `Old behavior`: notebook Q-Former cũ chạy branch `main`, train bằng default `internvl_config.yaml`, test `test_QA`
- `Current behavior`: notebook Q-Former mixed chạy `feature/sailvl-core`, train bằng `internvl_config_mixed.yaml`, test `test_alter`
- `Why it can matter`: đổi cả code snapshot, config, và benchmark split
- `Evidence`: `run_qformer.ipynb` cũ line `10`, `98`, `130`; mới line `10`, `12`, `126`, `174`
- `Expected impact`: rất cao, vì “kết quả” đang được đo trên một protocol khác

### F02
- `Severity`: high
- `Area`: prompt
- `Old behavior`: QA prompt cũ thêm dòng `Answer the question directly in natural language.`
- `Current behavior`: QA prompt mới bỏ dòng này, và đổi `final frame` thành `this frame`
- `Why it can matter`: mixed train vẫn có QA, nên delta này đổi supervision contract
- `Evidence`: `779cc7b:wad_dataset.py` line `113`, `131`, `136`; current `wad_dataset.py` line `220-222`
- `Expected impact`: có thể đổi khả năng regularize của QA lên alter

### F03
- `Severity`: high
- `Area`: validation
- `Old behavior`: val limit mặc định 200 trong dataset + hard stop 200 batch trong eval loop
- `Current behavior`: `eval_limit: 0`, không còn hard stop 200
- `Why it can matter`: val curve không còn cùng protocol
- `Evidence`: `779cc7b:wad_dataset.py:263`; `779cc7b:train.py:367`; current config line `60`; current `wad_dataset.py:431-434`
- `Expected impact`: val loss / best checkpoint có thể lệch rõ

### F04
- `Severity`: high
- `Area`: optimization
- `Old behavior`: `num_epochs=3`
- `Current behavior`: qformer mixed `num_epochs=5`
- `Why it can matter`: đổi epoch budget và vị trí checkpoint tốt nhất
- `Evidence`: `779cc7b:internvl_config.yaml:61`; current mixed config line `66`
- `Expected impact`: best epoch / overfit profile bị đổi

### F05
- `Severity`: high
- `Area`: infer
- `Old behavior`: notebook test `test_QA`
- `Current behavior`: notebook test `test_alter`
- `Why it can matter`: nếu so “kết quả” mà khác split thì không còn cùng bài toán
- `Evidence`: old notebooks line `130/154`; current notebooks line `174/143`
- `Expected impact`: đổi metric report và cách diễn giải kết quả

## Medium severity

### F06
- `Severity`: medium
- `Area`: prompt
- `Old behavior`: alter fixed prompt cũ dùng `based on the final frame`
- `Current behavior`: alter fixed legacy prompt mới dùng `based on this image`
- `Why it can matter`: wording supervision thay đổi
- `Evidence`: `779cc7b:wad_dataset.py:113` + else alter block; current `wad_dataset.py:18-21`
- `Expected impact`: nhẹ-đến-vừa, nhưng vẫn là semantic delta

### F07
- `Severity`: medium
- `Area`: split
- `Old behavior`: split bằng `Subset` trên một train dataset
- `Current behavior`: split xong mới tạo hai dataset instance train/val riêng
- `Why it can matter`: đổi handling theo `split`, nhất là `val`/`test` error path và prompt state
- `Evidence`: `779cc7b:wad_dataset.py:248-259`; current `wad_dataset.py:438-455`
- `Expected impact`: vừa

### F08
- `Severity`: medium
- `Area`: data
- `Old behavior`: sample lỗi ở mọi split đều random-resample
- `Current behavior`: `val/test` trả `None` và bỏ qua
- `Why it can matter`: số lượng sample thực được val/test có thể khác
- `Evidence`: `779cc7b:wad_dataset.py:165-166`; current `wad_dataset.py:290-297`, `scripts/test_infer.py:96-97`, `309-313`
- `Expected impact`: vừa, đặc biệt nếu data lỗi không ít

### F09
- `Severity`: medium
- `Area`: infer
- `Old behavior`: `test_loader` hard-coded `batch_size=1`
- `Current behavior`: lấy `evaluation.batch_size`, hiện tại là `4`
- `Why it can matter`: đổi throughput và cách xử lý empty batches; thông thường ít đổi generation logic, nhưng đổi protocol infer
- `Evidence`: `779cc7b:scripts/test_infer.py:238`; current config line `101`; current `scripts/test_infer.py:296`
- `Expected impact`: nhẹ-đến-vừa

### F10
- `Severity`: medium
- `Area`: sampling
- `Old behavior`: không có hệ sampler task balancing
- `Current behavior`: code có `WeightedRandomSampler`, dù mixed config đang tắt
- `Why it can matter`: khi so sánh chính xác, phải phân biệt code path sẵn có vs runtime đang tắt
- `Evidence`: current `train.py:337-374`
- `Expected impact`: hiện tại thấp vì config tắt, nhưng là semantic infrastructure mới

### F11
- `Severity`: medium
- `Area`: resume
- `Old behavior`: resume chỉ dựa vào optimizer/scheduler + skip batch
- `Current behavior`: thêm `runtime_state.pt` để restore RNG position
- `Why it can matter`: đổi semantics resume mid-epoch
- `Evidence`: current `train.py:597-633`, `736-760`
- `Expected impact`: cao cho resume, thấp cho train from scratch

## Low severity

### F12
- `Severity`: low
- `Area`: model
- `Old behavior`: `gradient_checkpointing_enable()` gọi trực tiếp
- `Current behavior`: qua helper `enable_gradient_checkpointing(...)`
- `Why it can matter`: có thể đổi cho model khác, nhưng InternVL path khả năng tương đương
- `Evidence`: `779cc7b:train.py:597`; current `train.py:805-808`
- `Expected impact`: thấp

### F13
- `Severity`: low
- `Area`: logging
- `Old behavior`: chỉ có token stats
- `Current behavior`: thêm prompt sample logs, run summary, data summary
- `Why it can matter`: chủ yếu observability
- `Evidence`: current `train.py:239`, `312`, `378-415`, current config lines `92-97`
- `Expected impact`: thấp

### F14
- `Severity`: low
- `Area`: infer
- `Old behavior`: `print_samples` parser default `5`
- `Current behavior`: lấy từ config, mặc định `3`
- `Why it can matter`: chỉ đổi số sample in ra
- `Evidence`: `779cc7b:scripts/test_infer.py` parser line `111`; current `scripts/test_infer.py:123`, `186-187`
- `Expected impact`: không đổi model quality

### F15
- `Severity`: low
- `Area`: infer
- `Old behavior`: chỉ save `output_file`
- `Current behavior`: save thêm `*_pairs.json`
- `Why it can matter`: artifact mới phục vụ GPTScore về sau, không đổi metric
- `Evidence`: current `scripts/test_infer.py:160-170`, `393-401`
- `Expected impact`: không đổi model quality

### F16
- `Severity`: low
- `Area`: sample contract
- `Old behavior`: không có `selected_prompt_id` / `selected_prompt_text` / `frame_path`
- `Current behavior`: có
- `Why it can matter`: chủ yếu phục vụ debug và fairness audit
- `Evidence`: current `wad_dataset.py:271-280`
- `Expected impact`: thấp

### F17
- `Severity`: low
- `Area`: model
- `Old behavior`: `preprocessing.py` và `qformer_bridge.py` như snapshot
- `Current behavior`: không có diff thực chất
- `Why it can matter`: giúp loại trừ một vùng nghi vấn
- `Evidence`: `git diff 779cc7b -- preprocessing.py qformer_bridge.py` không trả về diff
- `Expected impact`: none

### F18
- `Severity`: low
- `Area`: config
- `Old behavior`: notebook no-qformer cũ tự generate config từ `internvl_config.yaml`
- `Current behavior`: dùng file mixed có sẵn
- `Why it can matter`: đổi UX và source of truth config, nhưng nếu nội dung trùng nhau thì ảnh hưởng semantics là gián tiếp
- `Evidence`: `779cc7b:run_no_qformer.ipynb` có cell write YAML; current run_no_qformer dùng `CONFIG_PATH = "internvl_config_no_qformer_mixed.yaml"`
- `Expected impact`: thấp-đến-vừa, phụ thuộc nội dung config thực tế

---

## High-risk differences

Thứ tự ưu tiên theo mức độ khả nghi:

1. **QA prompt contract đổi**
2. **Validation policy đổi (`200` cap -> full val, batch size `1 -> 4`)**
3. **Epoch budget đổi (`3 -> 5`)**
4. **Notebook benchmark split đổi (`test_QA -> test_alter`)**
5. **Alter prompt wording đổi (`final frame -> this image`)**
6. **Val/test sample error handling đổi (`resample -> skip`)**

---

## Low-risk / bookkeeping-only differences

Những mục dưới đây có, nhưng khả năng thấp là nguyên nhân chính:

- thêm `selected_prompt_*`, `frame_path`
- thêm run summary / prompt log / data summary
- thêm pairs output file
- parser `print_samples` đọc từ config
- helper gradient checkpointing wrapper
- backend dispatch scaffold, vì InternVL path hiện tại vẫn đi trực tiếp

---

## Most likely reasons results diverged

Nếu bạn hỏi “vì sao kết quả hiện tại khác so với mốc `779cc7b`?”, thứ tự hợp lý nhất là:

1. **Bạn đang không còn train / validate / evaluate theo cùng notebook protocol nữa.**
2. **QA prompt đã đổi, trong khi mixed train hiện tại vẫn sử dụng QA.**
3. **Val protocol đã đổi rất mạnh, nên không thể so sánh trực tiếp val curve cũ/mới.**
4. **Epoch budget đã đổi, nên best checkpoint có thể rơi vào vị trí khác.**
5. **Test split notebook cũ và mới đang khác nhau (`test_QA` vs `test_alter`).**

---

## Recommended ablation order

Nếu mục tiêu là quay về gần nhất với behavior snapshot `779cc7b`, thứ tự nên làm là:

1. **Khóa notebook current về cùng benchmark split với baseline khi cần đối chiếu.**
   - nếu so training semantics thì tách biệt vấn đề test split

2. **Phục hồi đúng QA prompt snapshot `779cc7b`.**

3. **Phục hồi validation protocol**
   - `eval_limit=200`
   - nếu cần, eval batch size `1`
   - nếu muốn sát hơn nữa, hard-cap eval loop 200 batch

4. **Phục hồi epoch budget về `3`.**

5. **Chỉ sau đó mới soi tới các delta nhỏ hơn**
   - alter wording
   - dataset instance vs `Subset`
   - val/test skip-on-error
   - resume hygiene stack

---

## Ghi chú về mức độ xác minh

- `code-evidenced`
  - prompt
  - split
  - config
  - sampler
  - model setup
  - val/test protocol
  - resume/runtime stack
- `runtime-confirmed`
  - notebook/config mixed hiện tại thực sự đang trỏ vào `feature/sailvl-core`
  - config mixed hiện tại thực sự là `all/all`, `fixed_legacy`, `stratify_split=false`, `eval_limit=0`, `num_epochs=5`, `evaluation.batch_size=4`
- `inference-only`
  - tác động chính xác của một số delta nhỏ lên metric, vì máy local này không build được dataset đến cuối do thiếu `./wad_dataset/frame_index.pkl`

---

## Decision-ready conclusion

Nếu lấy `779cc7b` làm baseline chuẩn, thì `feature/sailvl-core` hiện tại **chưa giống** baseline ở nhiều điểm semantics quan trọng. Hai vùng nghi nhất không phải ở Q-Former bridge hay preprocessing, mà ở:

- prompt QA
- val / infer protocol
- epoch budget
- notebook split target

Nếu cần truy nguồn lệch kết quả một cách có hệ thống, nên ưu tiên restore / ablate theo thứ tự trong phần `Recommended ablation order`, thay vì tiếp tục so sánh kết quả cũ/mới như thể chúng đang dùng cùng một setup.
