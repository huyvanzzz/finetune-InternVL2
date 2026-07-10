# Audit ky khac biet giua train InternVL hien tai va commit `9b07f63`

## Summary

Muc tieu cua dot audit nay la truy vet vi sao ket qua InternVL hien tai lech manh so voi setup o commit `9b07f63e502c29498ac012ac346645bd3d031100`, nhung so sanh theo **semantics train thuc su** thay vi chi nhin diff file.

Ba moc duoc doi chieu:

- `A`: commit `9b07f63`
- `B`: code InternVL hien tai tren nhanh `main`
- `C`: code InternVL dang duoc notebook hien tai chay tren nhanh `feature/sailvl-core`

Ket luan ngan gon:

- `B` khac `A` rat manh o **data composition**, **prompt contract**, **sampling**, va **validation policy**.
- `C` da duoc dua ve gan `A` hon o **mixed QA+alter**, **khong weighted sampler**, **khong stratify**, **alter fixed legacy prompt**.
- Tuy vay, `C` van **chua trung semantics** voi `A` o it nhat 4 diem co the doi ket qua:
  - `QA prompt` da bi doi
  - `val` khong con cap `200` nhu setup cu
  - `num_epochs` tu `3` thanh `5`
  - `evaluation.batch_size` tu `1` thanh `4`

Noi cach khac: neu muc tieu la lap lai dung behavior gan voi `9b07f63`, thi current mixed tren `feature/sailvl-core` moi chi khop mot phan, chu **chua phai ban tai lap dung setup cu**.

---

## Comparison target da xac minh

### A. Commit `9b07f63`

Bang chung chinh:

- `git show 9b07f63:internvl_config.yaml`
- `git show 9b07f63:train.py`
- `git show 9b07f63:wad_dataset.py`

### B. Current `main`

Bang chung chinh:

- [internvl_config.yaml](/D:/NCKH_VLM/finetune-InternVL2/internvl_config.yaml)
- [train.py](/D:/NCKH_VLM/finetune-InternVL2/train.py)
- [wad_dataset.py](/D:/NCKH_VLM/finetune-InternVL2/wad_dataset.py)

### C. Current `feature/sailvl-core`

Bang chung chinh:

- [.worktrees/sailvl-core/run_qformer.ipynb](/D:/NCKH_VLM/finetune-InternVL2/.worktrees/sailvl-core/run_qformer.ipynb)
- [.worktrees/sailvl-core/internvl_config_mixed.yaml](/D:/NCKH_VLM/finetune-InternVL2/.worktrees/sailvl-core/internvl_config_mixed.yaml)
- [.worktrees/sailvl-core/train.py](/D:/NCKH_VLM/finetune-InternVL2/.worktrees/sailvl-core/train.py)
- [.worktrees/sailvl-core/wad_dataset.py](/D:/NCKH_VLM/finetune-InternVL2/.worktrees/sailvl-core/wad_dataset.py)

Notebook `run_qformer.ipynb` tren worktree nay da tro thang vao:

- `BRANCH = "feature/sailvl-core"`
- `CONFIG_PATH = "internvl_config_mixed.yaml"`

Nen khi noi “current InternVL ma notebook dang chay”, thuc te phai hieu la cot `C`, khong phai `B`.

---

## Behavior Difference Matrix

## 1. Data contract

| Truc | A: `9b07f63` | B: current `main` | C: `feature/sailvl-core` |
|---|---|---|---|
| Frame selection | Chon `[4,6,8]` roi lay frame cuoi, tuc chi dua frame cuoi vao model | Van dung final-frame semantic | Van dung final-frame semantic |
| Train source | `train.json`, mixed tu nhien | `train.json` sau khi filter `alter_only` | `train.json` mixed `all` |
| Val source | Tach tu `train.json` bang random split thuong | Tach tu `train.json`, co he filter moi | Tach tu `train.json`, mixed `all` |
| Sample fields | `question`, `answer`, `qformer_text`, `pixel_values`, `questionId`, `image` | Them `task_type`, `selected_prompt_id`, `selected_prompt_text`, `frame_path` | Giong `B` |
| Error handling | Loi sample thi random-resample cho moi split | Val/test co the `return None`, train moi resample | Giong `B` |

### Danh gia

- `A -> B`: doi semantics lon vi tu mixed sang `alter_only`.
- `A -> C`: final-frame van giong, nhung error handling va sample contract da doi.
- Muc do anh huong:
  - `A -> B`: `High`
  - `A -> C`: `Low-Medium`

---

## 2. Prompt / supervision contract

### `alter` prompt

`A: 9b07f63`

- `Describe the scene for a visually impaired user based on the final frame.`
- `Focus on immediate obstacles, safe direction, and what action the user should take.`
- `Provide only the final spoken guidance in natural language.`

`B: current main`

- prompt mode `balanced_v1`
- train `alter` khong con dung mot prompt co dinh; no chay theo family multi-prompt

`C: feature/sailvl-core`

- `direct_text_alter_prompt_mode: fixed_legacy`
- da quay lai alter fixed prompt cu

### `QA` prompt

`A: 9b07f63`

- `Describe the scene for a visually impaired user based on the final frame.`
- `Focus on obstacles, nearby people or vehicles, free walking space, direction, and safety.`
- `Question: ...`
- `Answer the question directly in natural language.`

`B: current main`

- `Based on this image, answer the following question for a visually impaired user directly in natural language.`
- `Question: ...`

`C: feature/sailvl-core`

- QA prompt da doi lan nua thanh:
  - `Describe the scene for a visually impaired user based on this frame.`
  - `Focus on obstacles, nearby people or vehicles, free walking space, direction, and safety.`
  - `Question: ...`
- Nhung **van khong giong hiet** `A` vi:
  - `final frame` thanh `this frame`
  - mat dong `Answer the question directly in natural language.`

### `qformer_text`

- `A`: `qformer_text = text_content.strip()`, mirror prompt text dang train
- `B`: mirror prompt moi, co the la multi-prompt cho `alter`
- `C`: mirror prompt fixed cho `alter`, QA mirror prompt QA moi

### Danh gia

- `A -> B`: doi semantics prompt rat lon.
- `A -> C`: alter da gan trung, nhung QA prompt van khac ro.
- Day la mot trong nhung nguon gay lech ket qua co kha nang cao nhat neu train mixed.

Muc do anh huong:

- `alter prompt`: `High` giua `A` va `B`, `Low` giua `A` va `C`
- `QA prompt`: `High` giua `A` va `C`

---

## 3. Split / sampling contract

### A. Commit `9b07f63`

- `train_test_split(indices, train_size=..., random_state=seed)`
- khong `task_filter`
- khong `stratify`
- khong `WeightedRandomSampler`
- `train_loader(..., shuffle=True)`

### B. Current `main`

- `train_task_filter: alter_only`
- `val_task_filter: alter_only`
- co `task_balancing.enabled: true`
- `build_train_sampler()` co the tao `WeightedRandomSampler`
- split co the di qua `stratify`

### C. `feature/sailvl-core`

- `train_task_filter: all`
- `val_task_filter: all`
- `stratify_split: false`
- `task_balancing.enabled: false`
- log runtime xac nhan:
  - `Mixed setup confirmed: both QA and alter are eligible for train/val.`
  - `Using plain random split without task stratification.`

### Diem khac can luu y

- `A` dung **mot train dataset** roi tach bang `Subset(train_dataset, train_indices/val_indices)`.
- `C` dung **hai dataset instance rieng** cho train va val sau khi tach sample list.

Khac biet nay khong phai luc nao cung doi ket qua, nhung no co the doi:

- cach xu ly exception
- cach gan state runtime theo `split`
- cach phat prompt o train/val

### Danh gia

- `A -> B`: `High`
- `A -> C`: `Low-Medium`

Neu chi xet split/sampler, thi `C` da tro ve gan `A` kha nhieu.

---

## 4. Optimization contract

| Muc | A: `9b07f63` | B: current `main` | C: `feature/sailvl-core` |
|---|---|---|---|
| `num_epochs` | `3` | `5` | `5` |
| `batch_size` train | `2` | `2` | `2` |
| `grad_accum` | `8` | `8` | `8` |
| `learning_rate` | `2e-4` | `2e-4` | `2e-4` |
| `proj_learning_rate` | `5e-4` | `5e-4` | `5e-4` |
| `eval_steps` | `2000` | `2000` | `2000` |
| `save_steps` | `2000` | `2000` | `2000` |
| optimizer groups | proj + con lai | giong y tuong cu | giong y tuong cu |

### Danh gia

- Cot `C` giu duoc phan lon optimization contract co ban.
- Khac biet lon nhat o day la:
  - `3 epoch` vs `5 epoch`

Neu ban dang so ket qua “best epoch 1” cua setup cu voi ket qua sau nhieu epoch o setup moi, thi day la khac biet co y nghia.

Muc do anh huong:

- `num_epochs`: `Medium-High`
- cac phan optimizer con lai: `Low`

---

## 5. Validation / test contract

### A. Commit `9b07f63`

- `internvl_config.yaml` khong dat `eval_limit`
- `wad_dataset.py` co default:
  - `eval_limit = config['data'].get('eval_limit', 200)`
- neu khong override, val se bi cat con `200` mau
- trong `eval_model`, con co them hard stop:
  - `if total_eval_batchs == 200: break`
- `evaluation.batch_size: 1`

### B. Current `main`

- `eval_limit: 0` => full val
- `eval_model` khong con hard stop 200 batch
- `evaluation.batch_size: 4`

### C. `feature/sailvl-core`

- `eval_limit: 0` => full val
- `eval_model` khong con hard stop 200 batch
- `evaluation.batch_size: 4`

### Danh gia

Day la mot khac biet quan trong nhung rat de bi bo qua:

- `A` khong danh gia tren full val nhu `C`
- `A` va `C` khong chi khac so mau val, ma con khac ca `eval batch size`

Tac dong ky vong:

- metric/val loss co the on dinh khac
- thoi diem ban thay “overfit” hoac “best epoch” co the bi doi
- khong nen so sanh truc tiep val curve cua `A` va `C` nhu the chung cung mot protocol

Muc do anh huong:

- `val cap 200` vs `full val`: `High`
- `eval batch size 1` vs `4`: `Medium`

---

## 6. Resume / runtime / debug contract

### Current code (`B` va `C`) da them

- runtime RNG save/restore
- optimizer sanitization
- sample-id debug
- prompt logging
- run summary logging
- data verify logging
- metrics JSON logging

### Danh gia

Phan lon nhung thay doi nay la:

- `debug/log/resume hygiene`
- khong chu dich doi supervision contract
- khong doi prompt text/answer text
- khong doi split policy neu config khong doi

Tuy nhien, co mot ngoai le nho:

- current code path co `None`-batch skip cho mot so backend / sample error path
- current dataset val/test khong random-resample giong cu

Nhung so voi cac khac biet data/prompt/val policy, phan nay it kha nang la nguyen nhan chinh.

Muc do anh huong:

- `Low`

---

## High-risk differences

Nhung khac biet duoi day nhieu kha nang nhat lam lech ket qua so voi `9b07f63`:

1. **Data composition**
- `A`: mixed tu nhien `QA + alter`
- `B`: `alter_only`
- `C`: mixed tro lai
- Day la delta lon nhat giua `A` va `B`.

2. **QA prompt contract**
- `A` va `C` van khac nhau o QA prompt.
- Neu train mixed, day la khac biet co the tac dong rat ro len regularization/task interaction.

3. **Validation policy**
- `A`: val bi cap hieu dung quanh `200`
- `C`: full val
- Neu ban dang so “val loss khong giong setup cu”, day la mot ung vien rat manh.

4. **Epoch budget**
- `A`: `3`
- `C`: `5`
- Khac biet nay co the doi mo hinh “best epoch”.

5. **Eval batch size**
- `A`: `1`
- `C`: `4`
- Thuong nhe hon data/prompt, nhung van du de doi profile danh gia.

---

## Medium-likelihood differences

1. **Dataset implementation style**
- `A`: `Subset` tren cung mot dataset instance
- `C`: train/val la hai dataset instance rieng

2. **Val/test error handling**
- `A`: sample loi thi random-resample
- `C`: val/test co the bo qua sample loi

3. **Them field debug trong sample**
- it kha nang doi ket qua, chu yeu doi observability

---

## Low / probably bookkeeping only

Nhung muc sau nhieu kha nang chi la “bookkeeping”:

- optimizer state sanitization
- runtime RNG state save/load
- debug sample id
- debug prompt/token logs
- metrics JSON logging
- run summary logging

Nhung muc nay co the doi **resume fidelity** hoac **kha nang debug**, nhung khong phai ung vien chinh de giai thich viec train tu dau cho ra ket qua khac.

---

## Most likely reasons current result diverged from `9b07f63`

Neu chi uu tien theo suc nang gay lech semantics train, thu tu hop ly nhat la:

1. **Current `main` khac `9b07f63` rat manh vi no da chuyen sang alter-only + prompt mode moi + weighted sampling.**

2. **Ngay ca current mixed tren `feature/sailvl-core` van chua trung `9b07f63` do QA prompt da bi doi.**

3. **Validation protocol da doi tu val cap 200 sang full val, nen curve val hien tai khong so sanh truc tiep voi setup cu duoc.**

4. **So epoch danh gia khac (`3` vs `5`) co the lam “best checkpoint” bi doi hoan toan.**

5. **Eval batch size `1 -> 4` va dataset construction moi co the dong gop them, nhung kha nang thap hon 4 diem tren.**

---

## Recommended next ablation order

Neu muc tieu la tim nhanh nguon gay lech ket qua, thu tu nen la:

1. **Khoa dung lai QA prompt giong `9b07f63`.**
- Day la delta semantics lon nhat con sot lai giua mixed `C` va baseline `A`.

2. **Khoa lai validation policy giong `9b07f63`.**
- dat `eval_limit=200`
- neu can, danh gia voi `evaluation.batch_size=1`
- dam bao protocol val cung kieu truoc khi so curve

3. **Khoa lai epoch budget giong `9b07f63`.**
- chay `3 epoch`

4. **Chi sau do moi danh gia cac delta nho hon.**
- dataset-instance vs `Subset`
- val/test sample error behavior
- resume/debug hygiene

---

## Decision-ready conclusion

Neu cau hoi la: **“Setup InternVL hien tai da that su giong `9b07f63` chua?”**

Cau tra loi la:

- `main`: **khong**
- `feature/sailvl-core` mixed: **gan hon nhieu, nhung van chua**

Neu cau hoi la: **“Dau la khac biet nhieu kha nang nhat dang lam ket qua lech?”**

Thu tu manh nhat hien tai la:

1. QA prompt contract
2. Validation policy (`200` vs `full`)
3. Epoch budget (`3` vs `5`)
4. Eval batch size (`1` vs `4`)

Con cac thay doi resume/debug/runtime moi chu yeu la phan phu tro, it kha nang la nguyen nhan chinh.
