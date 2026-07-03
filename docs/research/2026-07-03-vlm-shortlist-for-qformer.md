# Research: Shortlist VLM `~2B-2.6B` Phu Hop Hon Cho Huong `Q-Former`

## Muc tieu

Muc tieu cua dot research nay la tim mot VLM open-weight quanh `2B-2.6B`:

- co benchmark/leaderboard manh
- co visual connector hop hon `Q-Former` so voi `Qwen2-VL`
- co kha nang fine-tune / LoRA / infer thuc te
- co the tich hop vao repo ma khong phai dap di xay lai luong `InternVL` hien tai

Nguon xep hang chinh:

- OpenVLM Leaderboard cua OpenCompass / VLMEvalKit
- file detailed results `OpenVLM.json` duoc VLMEvalKit cong khai

Nguon kien truc chinh:

- model card / docs chinh thuc tren Hugging Face
- inspect `config.json` / `auto_map` / `model_type` khi can

## Cach loc candidate

Pool ban dau duoc loc theo:

- open-weight
- co score cong khai tren OpenVLM Leaderboard
- size `~2B-2.6B`

Sau do moi cham them theo 4 truc:

1. `Leaderboard strength`
2. `Q-Former fit`
3. `Repo integration fit`
4. `Rui ro chinh`

### Rubric `Q-Former fit`

- `Cao`: visual bridge kieu `MLP` / `projector` / `token merge` / `pixel shuffle`, co visual embeddings ro rang truoc khi vao LLM
- `Trung binh`: co bridge ro nhung runtime phuc tap hon hoac phu thuoc custom path
- `Thap`: visual path gan chat vao dynamic-resolution / placeholder-token contract / special positional scheme, viec chen `Q-Former` gan nhu doi kien truc goc

## Candidate pool da verify

Nhung model duoi day la nhung candidate co y nghia nhat sau khi loc theo size va benchmark:

- `SAIL-VL-1.5-2B`
- `InternVL3-2B`
- `SmolVLM2-2.2B`
- `InternVL2.5-2B`
- `Qwen2-VL-2B`

Mot so model khac trong cung vung size co xuat hien tren leaderboard, nhung khong vao shortlist cuoi vi it nhat mot trong cac ly do sau:

- benchmark kem hon ro
- connector khong ro rang cho huong `Q-Former`
- runtime / integration phuc tap hon nhung khong doi lai du loi ich

## Bang benchmark va kien truc

Bang duoi day uu tien 6 chi so de so sanh nhanh:

- `MMMU`
- `MMBench EN v1.1`
- `MathVista`
- `OCRBench`
- `AI2D`
- `MMStar`

So lieu benchmark ben duoi duoc trich tu detailed results cua OpenVLM Leaderboard.

| Model | Size | Connector hien tai | MMMU | MMBench EN v1.1 | MathVista | OCRBench | AI2D | MMStar |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SAIL-VL-1.5-2B | 2.47B | 2-layer MLP + 2x2 token merge | 46.4 | 79.4 | 67.0 | 891 | 83.7 | 62.6 |
| InternVL3-2B | 2.09B | ViT-MLP-LLM + pixel unshuffle | 48.7 | 79.0 | 57.6 | 831 | 78.6 | 61.1 |
| SmolVLM2-2.2B | 2.25B | pixel-shuffle + linear connector | 41.6 | 69.1 | 51.5 | 725 | 69.7 | 46.0 |
| InternVL2.5-2B | 2.00B | ViT-MLP-LLM + pixel unshuffle | 43.2 | 72.4 | 51.1 | 802 | 74.9 | 54.3 |
| Qwen2-VL-2B | 2.00B | dynamic visual path + image token scatter | 42.2 | 73.2 | 48.0 | 797 | 74.7 | 47.5 |

## Phan tich tung ung vien

### 1. `SAIL-VL-1.5-2B`

**Benchmark strength:** rat cao trong nhom size nay.

- Theo model card, `SAIL-VL-1.5-2B` dung `AimV2-Large + Qwen2.5-1.5B + 2-layer MLP + 2x2 token merge`
- Model card cong bo bang chi tiet cho thay no vuot `InternVL2.5-2B`, `InternVL3-2B`, `Ovis2-2B`, va `Qwen2.5-VL-3B` tren bo danh gia tong hop cua ho
- Tren OpenVLM detailed results, no cung rat manh o:
  - `MMBench EN v1.1 = 79.4`
  - `MathVista = 67.0`
  - `OCRBench = 891`
  - `AI2D = 83.7`
  - `MMStar = 62.6`

**Q-Former fit:** cao.

Ly do:

- connector hien tai da la `2-layer MLP`
- co `token merge 2x2`
- visual embeddings duoc bien doi truoc khi vao LLM
- ve mat logic, day la dang connector gan voi huong `Q-Former bridge` hon Qwen2-VL

**Repo integration fit:** trung binh-den-cao.

Ly do:

- diem tru: model dung `auto_map`, co file remote code rieng (`configuration_sailvl.py`, `modeling_sailvl.py`)
- diem cong: van theo `internvl_chat` family, nen boundary kien truc gan voi repo hien tai hon SmolVLM2

**Rui ro chinh:**

- phu thuoc `trust_remote_code`
- phai audit ky custom modeling truoc khi chen `Q-Former`

**Verdict:** `recommended`

### 2. `InternVL3-2B`

**Benchmark strength:** cao va on dinh.

- OpenVLM detailed results cho thay no tang ro so voi `InternVL2.5-2B`:
  - `MMMU: 48.7` vs `43.2`
  - `MMBench EN v1.1: 79.0` vs `72.4`
  - `MathVista: 57.6` vs `51.1`
  - `OCRBench: 831` vs `802`
  - `MMStar: 61.1` vs `54.3`
- Khong manh bang `SAIL-VL-1.5-2B` tren cac chi so OCR / MathVista / AI2D, nhung tong the rat can bang

**Q-Former fit:** rat cao.

Ly do:

- model card xac nhan van giu paradigm `ViT-MLP-LLM`
- van dung `randomly initialized MLP projector`
- van co `pixel unshuffle`
- day la family gan nhat voi flow `Q-Former` hien tai trong repo

**Repo integration fit:** rat cao.

Ly do:

- ban chat la duong tiep noi tu `InternVL2.5-2B`
- repo hien tai da co phan lon assumptions theo `InternVL`
- cong suc migration nho nhat trong tat ca candidate

**Rui ro chinh:**

- khong tao ra mot backbone "khac ho" de so sanh qua nhieu
- neu muc tieu la fairness giua nhieu family, no van hoi "gan InternVL"

**Verdict:** `backup`

### 3. `SmolVLM2-2.2B`

**Benchmark strength:** kha, nhung thap hon ro so voi hai ung vien tren.

- `MMMU = 41.6`
- `MMBench EN v1.1 = 69.1`
- `MathVista = 51.5`
- `OCRBench = 725`
- `AI2D = 69.7`
- `MMStar = 46.0`

No khong phai model manh nhat trong nhom nay, nhung van la mot baseline nho gon, co y nghia nghien cuu.

**Q-Former fit:** cao.

Ly do:

- Hugging Face mo ta ro rang `SigLIP + pixel-shuffle connector + SmolLM2`
- blog chinh thuc xac nhan SmolVLM theo implementation cua Idefics3 nhung nen manh hon o pixel shuffle
- day la visual connector ro rang, hop ly cho viec chen `Q-Former`

**Repo integration fit:** cao.

Ly do:

- native `transformers` support
- `model_type = smolvlm`
- `auto_map = None`
- infer / train path dua tren `AutoProcessor` va `AutoModelForVision2Seq` ro rang
- model card / blog cung co huong dan PEFT/LoRA truc tiep

**Rui ro chinh:**

- benchmark khong du manh neu muc tieu la tranh top-tier 2B models
- de thanh "fair new backbone", nhung kho thanh "best-performing new backbone"

**Verdict:** `optional stretch-style alternative`

Ghi chu: day khong phai stretch theo size; no la `stretch theo huong nghien cuu`, tuc la mot ung vien "khac family, de tich hop, de kiem soat" du benchmark kem hon.

### 4. `InternVL2.5-2B`

**Benchmark strength:** van tot, nhung da bi vuot boi `InternVL3-2B` va `SAIL-VL-1.5-2B`.

**Q-Former fit:** rat cao, vi day la backbone ban dang dung.

**Repo integration fit:** cao nhat, vi code hien tai da chay tren no.

**Rui ro chinh:**

- neu tiep tuc chon no lam "model moi", research value khong tang nhieu
- da co ung vien manh hon trong cung vung size va van hop `Q-Former`

**Verdict:** `reject as new backbone, keep as current baseline`

### 5. `Qwen2-VL-2B`

**Benchmark strength:** kha, nhung khong noi bat hon top candidate.

**Q-Former fit:** thap.

Ly do:

- visual path gan chat vao contract `image token` / `image_grid_thw`
- dynamic-resolution va position handling phuc tap hon ro
- chen `Q-Former` vao de bien thanh connector moi se can thay doi runtime nhieu hon cac model kieu `MLP connector`

**Repo integration fit:** trung binh.

Ly do:

- native transformers support tot
- nhung chinh vi visual contract cua no khac ban chat voi huong `Q-Former`, cong tich hop se lon nhung gia tri nghien cuu khong can xung

**Verdict:** `reject`

## Bang diem quyet dinh

| Model | Leaderboard strength | Q-Former fit | Repo integration fit | Rui ro chinh | Verdict |
| --- | --- | --- | --- | --- | --- |
| SAIL-VL-1.5-2B | Rat cao | Cao | Trung binh-cao | custom remote code | Recommended |
| InternVL3-2B | Cao | Rat cao | Rat cao | gan current family | Backup |
| SmolVLM2-2.2B | Trung binh | Cao | Cao | benchmark kem hon | Optional alternative |
| InternVL2.5-2B | Trung binh-cao | Rat cao | Rat cao | khong tao them research value | Reject as new backbone |
| Qwen2-VL-2B | Trung binh | Thap | Trung binh | visual contract khong hop Q-Former | Reject |

## Shortlist cuoi

### Recommended: `SAIL-VL-1.5-2B`

Nen chon neu muc tieu la:

- muon mot backbone moi, manh tren leaderboard
- van thuoc nhom kien truc hop `Q-Former`
- chap nhan phai audit custom remote code ky hon `InternVL`

Day la ung vien can bang tot nhat giua:

- benchmark
- do "hop Q-Former"
- gia tri nghien cuu khi doi backbone

### Backup: `InternVL3-2B`

Nen chon neu muc tieu la:

- muon nang cap tu current baseline an toan nhat
- muon giu kien truc gan nhu da co san trong repo
- muon toi thieu hoa rui ro implementation

Neu uu tien cao nhat la "lam duoc nhanh, chac, it vo nhat", thi day moi la lua chon thuc dung nhat.

### Optional alternative: `SmolVLM2-2.2B`

Nen chon neu muc tieu la:

- muon mot family khac ro rang hon
- muon native transformers support sach hon
- chap nhan benchmark thap hon de doi lai implementation gon hon

No phu hop hon vai tro `ablation backbone` hon la `main competitive backbone`.

## Ket luan

Sau khi doi chieu leaderboard + kien truc + integration:

- neu muc tieu la tim `mot backbone moi` vua manh vua hop `Q-Former`, lua chon tot nhat hien tai la **`SAIL-VL-1.5-2B`**
- neu muc tieu la tim `lua chon an toan nhat de implement ngay`, lua chon tot nhat la **`InternVL3-2B`**
- neu muc tieu la tim `mot family khac de so sanh nghien cuu` va uu tien native `transformers`, lua chon hop ly la **`SmolVLM2-2.2B`**

Nghia la:

- `Khong can tiep tuc voi Qwen2-VL cho huong Q-Former`
- `Khong can doi backbone neu uu tien cao nhat la implementation safety`; khi do `InternVL3-2B` da la buoc nang cap hop ly nhat
- `Chi nen doi sang SAIL-VL-1.5-2B` neu ban thuc su muon mot backbone moi co benchmark rat tot ma van giu duoc connector style hop `Q-Former`

## Nguon chinh

- OpenVLM Leaderboard: https://huggingface.co/spaces/opencompass/open_vlm_leaderboard
- VLMEvalKit README (link detailed results `OpenVLM.json`): https://github.com/open-compass/vlmevalkit
- InternVL2.5-2B model card: https://huggingface.co/OpenGVLab/InternVL2_5-2B
- InternVL3-2B model card: https://huggingface.co/OpenGVLab/InternVL3-2B
- SAIL-VL-1.5-2B model card: https://huggingface.co/BytedanceDouyinContent/SAIL-VL-1.5-2B
- SmolVLM blog / architecture notes: https://huggingface.co/blog/smolvlm
