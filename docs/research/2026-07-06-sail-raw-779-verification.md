# Verify `SAIL on raw-779`

## 1. Tóm tắt

- Repo/worktree đã kiểm: `D:\NCKH_VLM\finetune-InternVL2-raw-779`
- Branch hiện tại: `feature/sail-on-raw-779`
- `HEAD` hiện tại: `5759de0`
- Commit mốc yêu cầu: `779cc7b2284c8fa480ef1d5cc91a89c5f21ee862`
- Kết luận tổng quát:
  - Phần `SAIL no-qformer`, `SAIL qformer`, notebook SAIL, file `pairs.json`, và log split `QA/alter` đều **đã có code và đã được chứng minh ở mức static/test**.
  - Worktree hiện tại **đi lên từ `779`** nhưng **không còn đứng đúng tại `HEAD = 779`** nữa.
  - Các phần runtime thật như `resume thật`, `save/load checkpoint thật`, `infer thật sinh pairs file`, `dataset build local` vẫn **chưa được chứng minh đủ**.
  - Có ít nhất một điểm cần nói thẳng là **lệch với yêu cầu kiểm tra nghiêm ngặt**: `HEAD` không còn bằng `779`.

## 2. Bảng đối chiếu yêu cầu -> trạng thái

| Yêu cầu | Trạng thái | Bằng chứng | Ghi chú |
|---|---|---|---|
| Worktree đích là `D:\NCKH_VLM\finetune-InternVL2-raw-779` | Đã xác minh | `git worktree list` | Đúng worktree mục tiêu |
| `HEAD` của worktree đúng bằng `779...` | Sai hoặc lệch yêu cầu | `git rev-parse HEAD` trả `5759de0`; `git log --oneline -5` | Nhánh hiện tại được phát triển từ `779` nhưng `HEAD` không còn là `779` |
| Nhánh hiện tại thực sự bám gốc `779` | Đã xác minh | `git merge-base 5759de0 779...` trả đúng `779...` | Đúng theo nghĩa “base từ 779” |
| Có đủ backend files cho SAIL | Đã xác minh | Tồn tại `backend_dispatch.py`, `checkpoint_metadata.py`, `model_backends/sailvl/*` | Không còn là stub file-level |
| `train.py` giữ InternVL path cũ làm mặc định, chỉ dispatch sang SAIL khi `model.architecture == sailvl` | Đã xác minh | `train.py` có `get_backend_for_config`, `if backend is not None ... else ...` | Đã nối được cả hai path |
| `scripts/test_infer.py` hỗ trợ cả InternVL và SAIL | Đã xác minh | `scripts/test_infer.py` có `get_backend_for_config`, `backend.load_model_and_tokenizer`, `backend.generate_response` | Flow cũ InternVL vẫn còn |
| Có config mixed riêng cho `SAIL no-qformer` | Đã xác minh | `sailvl_config_no_qformer_mixed.yaml` + test `test_sail_backend_contract.py` | `qformer.enabled: false` |
| Có config mixed riêng cho `SAIL qformer` | Đã xác minh | `sailvl_config_mixed.yaml` + test `test_sail_backend_contract.py` | `qformer.enabled: true`, `num_query_tokens: 32` |
| Q-Former path của SAIL có attach bridge | Đã xác minh | `model_backends/sailvl/runtime.py`, `model_backends/sailvl/qformer_bridge.py`, test `test_attach_qformer_if_enabled_delegates_to_sail_bridge` | Có code path và unit test |
| Q-Former path của SAIL có encode/set/clear `qformer_text` | Đã xác minh | `model_backends/sailvl/qformer_bridge.py`, test `test_sail_train_collate_encodes_qformer_text_when_enabled` | Đã chứng minh ở mức collate/runtime contract |
| Bridge metadata có `bridge_backend: "sailvl"` | Đã xác minh | `model_backends/sailvl/qformer_bridge.py`, `runtime.py`, test `test_save_backend_artifacts_writes_sail_bridge_metadata` | Đúng metadata backend |
| `checkpoint_metadata.py` sanitize `base_model_name_or_path`/README metadata | Đã xác minh | `checkpoint_metadata.py` | Có code sanitize rõ ràng |
| Semantics nền là mixed `QA + alter` | Đã xác minh | `wad_dataset.py` vẫn load `train.json`, split random trên toàn bộ metadata | Không thấy filter task framework mới trong config SAIL |
| Prompt/semantic contract vẫn dùng logic `779` ở dataset hiện tại | Đã xác minh | `wad_dataset.py` diff với `779`: chỉ thêm helper split count/log, không thấy đổi prompt logic | Xác minh bằng static diff, chưa có runtime sample check ở đợt này |
| Split random kiểu cũ | Đã xác minh | `wad_dataset.py` dùng `train_test_split(... random_state=seed)` | Không có `stratify` |
| Không task balancing framework mới | Đã xác minh | `sailvl_config_mixed.yaml`, `sailvl_config_no_qformer_mixed.yaml` không có `task_balancing`, `train.py` vẫn `shuffle=True` | Đúng theo base `779` |
| Test nhắm `test_alter` | Đã xác minh | `run_sail_qformer.ipynb`, `run_sail_no_qformer.ipynb`, grep `--split\", \"test_alter\"` | Notebook test không trỏ `test_QA` |
| Có cả `SAIL no-qformer` và `SAIL qformer` | Đã xác minh | 2 config + 2 notebook + backend qformer bridge | Có code path riêng |
| Cùng semantic `question / answer / qformer_text` với contract cũ | Đã xác minh | `wad_dataset.py` trả chung semantic fields, SAIL collate dùng `image` + giữ `question/answer/qformer_text` | Chủ yếu chứng minh bằng static code + test collate |
| Có notebook riêng cho cả hai | Đã xác minh | `run_sail_qformer.ipynb`, `run_sail_no_qformer.ipynb`, test `test_sail_notebooks.py` | Đã tồn tại và parse pass |
| Resume path hỗ trợ local/HF checkpoint ở mức code path | Đã xác minh | `train.py` có `resolve_checkpoint_path`, `resolve_resume_config`, `PeftModel.from_pretrained`, `backend.load_backend_artifacts` | Mới ở mức code path, chưa runtime thật |
| `build_dataset()` in tổng `train/val` và số mẫu `QA/alter` | Đã xác minh | `wad_dataset.py` có `summarize_task_counts_from_indices` + log `Final split stats | ...` + test `test_dataset_split_logging.py` | Dataset smoke local bị chặn do thiếu `frame_index.pkl`, nên chưa thấy log runtime thật trên máy local |
| `scripts/test_infer.py` tạo file `*_pairs.json` | Đã xác minh | `write_prediction_pairs()` + gọi ở cuối `main()` | Chưa chạy infer thật để thấy file sinh ra |
| Cell test trong notebook chạy `test_alter`, không nhầm `test_QA` | Đã xác minh | `test_sail_notebooks.py` pass | Static verified |

## 3. Những gì đã được chứng minh

- Worktree mục tiêu là đúng thư mục `raw-779`.
- Nhánh hiện tại thực sự có gốc từ commit `779`.
- Backend SAIL đã được port vào base `779` với dispatcher riêng.
- Có đủ hai mode:
  - `SAIL no-qformer`
  - `SAIL qformer`
- `train.py` và `scripts/test_infer.py` đều đã có đường `backend` cho SAIL, không chỉ là file stub.
- Notebook SAIL tồn tại và đúng nhánh/config/test split ở mức static parse.
- `scripts/test_infer.py` có code tạo `*_pairs.json`.
- `wad_dataset.py` có log split stats `QA/alter`.
- Bộ test hiện tại pass:
  - `python -m pytest tests/test_sail_entrypoints.py tests/test_sail_backend_contract.py tests/test_sail_runtime.py tests/test_sail_notebooks.py tests/test_dataset_split_logging.py -q`
  - Kết quả: `13 passed`
- `py_compile` cho các file chính pass:
  - `train.py`
  - `scripts/test_infer.py`
  - `backend_dispatch.py`
  - `checkpoint_metadata.py`
  - `model_backends/base.py`
  - `model_backends/sailvl/runtime.py`
  - `model_backends/sailvl/preprocess.py`
  - `model_backends/sailvl/qformer_bridge.py`
  - `wad_dataset.py`

## 4. Những gì chưa được chứng minh đủ

- `resume` thật với checkpoint SAIL:
  - Có code path, nhưng chưa có smoke runtime thật trên local/Kaggle.
- `save/load checkpoint` thật:
  - Có unit test metadata và code path save/load, nhưng chưa có chạy thật qua checkpoint directory.
- `infer` thật sinh `*_pairs.json`:
  - Có code path, nhưng chưa có một lần chạy infer thật để nhìn thấy file sinh ra.
- `build_dataset()` runtime-local:
  - Không xác minh được do thiếu artifact `./wad_dataset/frame_index.pkl`.
  - Notebook có cell `build_frame_index.py`, nên nhiều khả năng Kaggle path sẽ ổn, nhưng local audit hiện tại chưa chứng minh được.
- `smoke train` thật:
  - Chưa chạy 1 batch train thật cho cả `SAIL no-qformer` và `SAIL qformer` trên worktree này trong đợt verify.
- `smoke infer` thật:
  - Chưa chạy 1 sample infer thật trên worktree này trong đợt verify.
- `prepare_qformer.py` và `smoke_qformer_bridge.py` trong notebook qformer:
  - Đã được notebook tham chiếu đúng, nhưng chưa chạy trong đợt verify hiện tại.

## 5. Những gì sai hoặc có nguy cơ sai

### Sai rõ ràng

- Nếu áp dụng đúng tiêu chí nghiêm ngặt “`HEAD` của worktree phải đúng bằng `779...`”, thì trạng thái hiện tại **không đạt**:
  - `git rev-parse HEAD` trả `5759de0`
  - không còn là `779...`

### Có nguy cơ sai / chưa đủ bằng chứng

- Yêu cầu “verify bằng runtime smoke local `build_dataset()`” hiện **bị chặn** vì thiếu `./wad_dataset/frame_index.pkl`.
- Yêu cầu “resume có hỗ trợ local/HF checkpoint” mới được xác minh ở mức code path, chưa có bằng chứng runtime.
- Yêu cầu “pairs file tạo ra khi test” mới được xác minh ở mức code path, chưa có file thật từ một lần chạy thật.
- Bộ test hiện tại cover tốt static contract, nhưng chưa cover:
  - resume thật
  - smoke train thật
  - smoke infer thật
  - save/load checkpoint thật
  - runtime pairs file thật

## 6. Coverage của bộ test hiện tại

| Test file | Cover chính | Mức độ |
|---|---|---|
| `test_sail_entrypoints.py` | Dispatcher `backend_dispatch` và chọn backend `sailvl` | Static/API-level |
| `test_sail_backend_contract.py` | Tồn tại config, bật/tắt qformer, registry API backend | Static contract |
| `test_sail_runtime.py` | Collate `image -> pixel_values`, qformer text repetition, attach bridge delegate, bridge metadata save | Unit/runtime-light |
| `test_sail_notebooks.py` | Notebook tồn tại, đúng branch/config, test split là `test_alter`, qformer notebook có prepare/smoke cell | Static notebook audit |
| `test_dataset_split_logging.py` | Helper đếm `QA/alter` theo split indices | Unit logic |

### Những gì bộ test chưa cover

- `train.py` chạy thật với config SAIL
- `scripts/test_infer.py` chạy thật với config SAIL
- `build_dataset()` đọc được artifact thật
- `resume` thật
- checkpoint thật từ Hugging Face/local
- `pairs.json` được tạo thật sau infer
