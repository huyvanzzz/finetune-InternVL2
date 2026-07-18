# Audit pretrain runtime hien tai cho `cls_add`

## Tom tat

Audit nay khoa lai runtime truth cua nhanh `feature/pretrain-server-setup` cho pretrain `InternVL-qformer` mode `cls_add`, tap trung vao cau hoi: neu run hien tai gap OOM hoac runtime issue sau vai batch, kha nang cao nhat dang nam o duong nao.

Ket luan ngan:

- Setup runtime hien tai da kha sach o mat mode gating, dtype bridge, va FlashAttention.
- `image token count` hien tai khong phai nghi pham so 1.
- Nguon gay van de kha nang cao hon nam o `runtime memory peak` cua duong `vision -> qformer -> mlp1 -> language_model`, cong voi overhead cua `DDP + 4-bit + bf16 + dynamic text length`.
- `pin_memory + workers` va `allocator / fragmentation` la hai nhom phu can verify tiep, nhung chua co bang chung manh bang nhom tren.

## Runtime truth hien tai

### Config va batching

Theo [internvl_pretrain_config_traj_cls.yaml](/D:/NCKH_VLM/finetune-InternVL2/.worktrees/pretrain-server-setup/internvl_pretrain_config_traj_cls.yaml):

- `batch_size = 32`
- `gradient_accumulation_steps = 1`
- `use_accelerate = true`
- `bf16 = true`
- `fp16 = false`
- `gradient_checkpointing = false`
- `num_workers = 4`
- `pin_memory = true`
- `persistent_workers = true`
- `prefetch_factor = 2`
- `quantization.enabled = true`
- `quantization.bits = 4`
- `quantization.compute_dtype = bfloat16`
- `vision.min_pixels = 256`
- `vision.max_pixels = 1280`

Voi `accelerate launch --num_processes 2`, runtime thuc te la:

- micro-batch moi GPU: `32`
- world size: `2`
- global batch: `64`

### Mode gating va trainable modules

Theo [train_pretrain.py](/D:/NCKH_VLM/finetune-InternVL2/.worktrees/pretrain-server-setup/train_pretrain.py) va [qformer_bridge.py](/D:/NCKH_VLM/finetune-InternVL2/.worktrees/pretrain-server-setup/qformer_bridge.py):

- frozen:
  - `vision_model`
  - `qformer`
  - `qformer_query_tokens`
  - `mlp1`
  - `language_model`
- trainable chung:
  - `qformer_input_proj`
  - `qformer_to_mlp1_proj`
  - `trajectory_backbone`
- mode `cls_add` chi train:
  - `trajectory_cls_head`
- mode `cls_add` khong train:
  - `trajectory_token_projector`

Day la mot thay doi quan trong: runtime hien tai da loai bo tinh trang mang theo head cua mode `concat` khi dang train `cls_add`.

### Image token va text token path hien tai

Theo [qformer_bridge.py](/D:/NCKH_VLM/finetune-InternVL2/.worktrees/pretrain-server-setup/qformer_bridge.py):

1. `pixel_values`
2. `vision_model`
3. bo CLS vision token
4. `pixel_shuffle`
5. `qformer_input_proj`
6. `qformer`
7. `qformer_to_mlp1_proj`
8. cong `traj_cls` vao `mlp1_inputs` neu la `cls_add`
9. `mlp1`
10. ep `visual_tokens` ve `dtype` cua embedding LLM
11. chen vao `language_model`

Theo [train_pretrain.py](/D:/NCKH_VLM/finetune-InternVL2/.worktrees/pretrain-server-setup/train_pretrain.py) va [pretrain_dataset.py](/D:/NCKH_VLM/finetune-InternVL2/.worktrees/pretrain-server-setup/pretrain_dataset.py):

- `question` di vao ca:
  - Q-Former text path: `qformer_text = "Question: ..."`
  - LLM prompt path: `"<image>\\nQuestion: ..."`
- `answer` di vao target text voi format `Answer: ...`

## Audit duong di cua cac tensor memory-critical

### 1. `pixel_values -> vision_model -> vit_embeds`

Theo [pretrain_dataset.py](/D:/NCKH_VLM/finetune-InternVL2/.worktrees/pretrain-server-setup/pretrain_dataset.py), moi sample chi mang `1 frame`, nhung frame nay duoc `process_image(...)` cat thanh nhieu tiles.

Theo log hien co:

- `frames=1`
- `tiles_per_frame=[3]`
- `query_tokens_per_tile=32`
- `total_image_tokens=96`

Dieu nay cho thay:

- image path hien tai dang expand theo `tiles`
- nhung voi run hien tai, so tile debug duoc nhin thay dang co dinh o `3`
- neu dieu nay dung cho nhieu batch lien tiep, thi phan image token count khong phai nguon "phinh" dot ngot

### 2. `vit_embeds -> qformer_input_proj -> qformer -> qformer_to_mlp1_proj -> mlp1`

Day la cum nghi pham manh nhat o mat memory:

- `qformer_input_proj` va `qformer_to_mlp1_proj` duoc giu o `float32`
- `trajectory_backbone` va `trajectory_cls_head` cung duoc dat ve `float32`
- `language_model` va input embedding side chay o `bf16`
- cuoi cung `visual_tokens` moi duoc ep ve `llm_embed_dtype`

Nghia la:

- trong bridge path co mot doan activation chay o `float32`
- du doan peak memory co the xuat hien truoc khi tensor cuoi duoc cast ve `bf16`

### 3. `mlp1 output + trajectory cls -> language_model`

Cho `cls_add`, `traj_cls` duoc cong vao `mlp1_inputs`, sau do moi qua `mlp1` va chen vao LLM embedding stream.

Vi `num_image_token` van la `32`, mode `cls_add` khong tang token count downstream.

Do do, neu co OOM thi:

- khong nen uu tien nghi "vi them trajectory nen token tang"
- nen uu tien nghi memory peak activation o bridge path hoac language side

## Log hien tai: cai gi co dinh, cai gi dang dong

### Nhung diem co dinh da khoa duoc

Tu log user da gui:

- `fusion_mode=cls_add`
- `num_image_token=32`
- `Mode-gated trajectory heads | cls_head_trainable=True | token_projector_trainable=False`
- `global_batch=64`
- `FlashAttention runtime | supported=24 | flash=24 | fallback=0`
- `tiles_per_frame=[3]`
- `total_image_tokens=96`

Nhung diem nay dang nhat quan voi code.

### Nhung diem dang dong

Tu log user da gui:

- `dynamic token length` dao dong quanh `163-166`
- `Profile step elapsed_sec` giam dan tu ~`3.06s` xuong ~`1.14s`
- `Gradient health` o buoc dau cho thay:
  - `trajectory = 0.0`
  - `bridge ~= 1.4`

Y nghia:

- text length co dao dong, nhung bien do hien nhin thay con nho
- image tile count debug duoc nhin thay dang co dinh
- step time giam dan o cac buoc dau la binh thuong vi warm startup / cache / compile path

## Vi sao `image token count` hien tai khong phai nghi pham so 1

Ly do chinh:

1. mode `cls_add` giu `num_image_token = 32`, khong tang thanh `38`
2. log hien tai cho thay `tiles_per_frame=[3]` va `total_image_tokens=96`, khop logic va chua cho thay dot bien
3. OOM xuat hien sau vai batch khong hop nhat voi gia thuyet "ngay tu dau image token da qua lon"

Neu sau nay co bang chung rang mot so batch khac co `tiles_per_frame > 3`, khi do gia thuyet nay phai mo lai. Nhung voi evidence hien tai, no khong nen la nghi pham uu tien so 1.

## Nhom nguyen nhan kha nang cao

### Rat kha nang

#### 1. Peak memory o bridge path truoc khi cast ve `bf16`

Bang chung:

- `qformer_input_proj`, `qformer_to_mlp1_proj`, `trajectory_backbone`, `trajectory_cls_head` dang o `float32`
- `visual_tokens` chi duoc cast ve `llm_embed_dtype` sau `mlp1`
- duong `vision -> qformer -> pre-mlp1 add -> mlp1` la noi co nhieu activation tam thoi

Day la nghi pham manh nhat o muc code path.

#### 2. Overhead cua `DDP + 4-bit + bf16 + 2 GPU`

Bang chung:

- runtime dung `Accelerator(...)` + `accelerate.prepare(...)`
- multi-GPU run co them memory va sync overhead so voi 1 GPU
- van de truoc do voi unused params da cho thay runtime nay rat nhay voi DDP state

Neu OOM chi xuat hien tren `2 GPU` ma khong co tren `1 GPU`, day la nhom nghi pham cuc manh.

### Co the

#### 3. Dynamic text length + sequence allocation

Bang chung:

- `dynamic token length` dang dao dong
- dao dong hien nhin thay nho, nen kho la culprit duy nhat
- nhung van co the dong gop vao peak memory neu ket hop voi bridge path va DDP

#### 4. `pin_memory + workers + persistent_workers`

Bang chung:

- `num_workers=4/process`, tong cong ~`8 workers`
- `pin_memory=true`, `persistent_workers=true`
- nhung cai nay thuong anh huong host RAM / data transfer nhieu hon VRAM model

No co the lam runtime "nang" hon hoac phuc tap hon, nhung chua du bang chung de coi la culprit chinh cho GPU OOM.

#### 5. Allocator / fragmentation

Bang chung:

- config co `cuda_alloc_conf = expandable_segments:True`
- OOM xuat hien sau vai batch co the phu hop voi fragmentation hoac memory peak khong deu

Nhung hien tai chua co log `allocated/reserved/max_reserved` nen moi dung o muc gia thuyet duoc ho tro mot phan.

### It kha nang

#### 6. Image token count hien tai tu no gay OOM

Voi evidence hien co, gia thuyet nay yeu hon cac nhom tren.

#### 7. Head cua mode khac bi train nham

Da duoc sua bang mode gating. Hien tai no khong con la nghi pham chinh nua.

## Thu tu verify tiep theo truoc khi sua code

1. Verify tren `1 GPU` voi cung config `cls_add`
- Neu 1 GPU cung OOM o step tuong tu, nghi pham chinh nghieng ve bridge path / activation peak.
- Neu 1 GPU khong OOM, nghi pham nghieng manh ve `DDP + accelerate` overhead.

2. Log them memory theo step
- `torch.cuda.memory_allocated()`
- `torch.cuda.memory_reserved()`
- `torch.cuda.max_memory_allocated()`
- `torch.cuda.max_memory_reserved()`

3. Log them thong tin dong theo step dau
- `tiles_per_frame`
- `total_image_tokens`
- `dynamic token length`
- de xac nhan khong co batch bat thuong ve tile count

4. Neu can, benchmark lai voi:
- `num_workers=0`
- `pin_memory=false`
- de loai bo nhom dataloader interaction

## Ket luan

Runtime hien tai da dung va sach hon rat nhieu so voi vong truoc:

- mode gating da dung
- dtype mismatch `bf16/float32` o bridge output da duoc xu ly
- FlashAttention dang bat that
- image token path hien tai khong cho thay su "phinh" dot ngot

Voi bang chung tu code va log hien co, nhom nghi pham uu tien nhat la:

1. peak memory o bridge path `float32` truoc khi cast ve `bf16`
2. overhead cua `DDP + 4-bit + bf16`
3. dynamic sequence allocation dong gop them vao peak

Neu buoc sau can sua code, muc tieu hop ly nhat la tap trung vao do memory per step va phan bridge/runtime allocation, khong nen uu tien nghi image token count la nguyen nhan so 1.
