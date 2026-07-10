# Audit cách setup `trajectory encoder` trong `TrackingMeetsLMM`

## Tóm tắt

Mục tiêu của báo cáo này là bóc tách đúng cách repo `mbzuai-oryx/TrackingMeetsLMM` đang đưa thông tin trajectory vào mô hình, chưa bàn sang cách port vào `restore-779`.

Kết luận ngắn:

- Repo này **không dùng Q-Former cho trajectory**.
- Trajectory đi qua một **trajectory encoder riêng** dựa trên `ImageBindModel`, tách thành hai modality:
  - `traj`
  - `ego`
- Ảnh đi theo một nhánh riêng từ CLIP.
- Ba nhánh `image`, `traj`, `ego` được projector về cùng không gian `v_embed_dim=768`, rồi **cộng lại** trước khi đi qua `visual_blocks`.
- Output sau fusion không được nối trực tiếp vào prompt text, mà được dùng làm **dynamic adapter signal** để chèn vào các layer cuối của LLaMA Adapter.

---

## 1. Pipeline tổng của repo gốc

### 1.1 Trajectory encoder tham gia ở pha nào

`Fact`

- Ở `main_finetune.py`, model `LLaMA_adapter` được dùng cho fine-tune và dataloader luôn trả thêm `trajs`, `egos`.
- Ở `engine_finetune.py`, mỗi batch đều gọi:
  - `model(examples, labels, imgs, trajs, egos)`
- Ở `main_pretrain.py` và `engine_pretrain.py`, flow tương tự cũng dùng `trajs`, `egos`.
- Ở `demo.py`, inference cũng truyền:
  - `model.generate(images, prompts, trajs, egos, ...)`

Kết luận:

- Trajectory encoder tham gia ở cả:
  - `pretrain`
  - `finetune`
  - `inference`

### 1.2 Sơ đồ pipeline tổng

`Fact`

```text
JSON sample
-> dataset đọc image + conversations + trajectories
-> pack thành:
   - image tensor
   - traj tensor (6, 5, 5)
   - ego tensor (1, 5, 5)
   - tokenized text
-> model.forward(...)
-> forward_visual(...)
   - CLIP image branch
   - traj encoder branch
   - ego encoder branch
   - projector + fusion
-> visual_query
-> inject vào các layer cuối của LLaMA Adapter
-> sinh logits / generate text
```

---

## 2. Dataset contract của trajectory

### 2.1 Field đầu vào

`Fact`

- File `data/dataset.py` và `demo.py` đều đọc field:
  - `image`
  - `conversations`
  - `trajectories` (nếu có)

- `question` lấy từ:
  - `data_item['conversations'][0]['value']`
- `answer` lấy từ:
  - `data_item['conversations'][1]['value']`

### 2.2 Cách tách `ego` và `traj`

`Fact`

- `trajectories` là một list các trajectory.
- Phần tử đầu tiên (`i == 0`) được đưa vào `ego_tensor`.
- Các phần tử tiếp theo (`i > 0`) được đưa vào `traj_tensor`.

### 2.3 Shape và rule pack tensor

`Fact`

- `traj_tensor` được khởi tạo với shape:
  - `(6, 5, 5)`
- `ego_tensor` được khởi tạo với shape:
  - `(1, 5, 5)`

- Rule fill:
  - với `traj_tensor`, chỉ nhận các phần tử `i > 0` và `i < 6`
  - với mỗi trajectory, chỉ lấy tối đa `5` timestep
  - với mỗi timestep, chỉ lấy tối đa `5` giá trị

Điều này đồng nghĩa:

- tối đa có `1 ego track`
- tối đa có `5 object tracks` thực tế được dùng trong `traj_tensor` nếu tính riêng các track sau ego, vì tensor có 6 slot nhưng slot `0` không dùng cho object branch
- mỗi track tối đa `5` timestep
- mỗi timestep tối đa `5` feature values

### 2.4 Padding / truncation

`Fact`

- Tensor được khởi tạo bằng zeros.
- Nếu trajectory ngắn hơn, phần còn lại giữ nguyên zero padding.
- Nếu trajectory dài hơn:
  - số track vượt quá giới hạn sẽ bị bỏ
  - timestep vượt quá 5 sẽ bị cắt
  - số chiều mỗi timestep vượt quá 5 cũng bị cắt

### 2.5 Ý nghĩa của shape `(6,5,5)` và `(1,5,5)`

`Fact`

- Code chỉ cho thấy đây là hardcode shape, không có chú thích giải thích semantics từng chiều.

`Repo-structure inference`

- Nhiều khả năng:
  - chiều đầu là số track
  - chiều giữa là số timestep
  - chiều cuối là vector feature của mỗi timestep

`Open question`

- 5 giá trị ở mỗi timestep là gì cụ thể:
  - tọa độ?
  - velocity?
  - heading?
  - kích thước bbox?
  - hay feature tracking đã chuẩn hóa từ 3DMOTFormer?

Code công khai không đủ để kết luận chắc.

---

## 3. Kiến trúc trajectory encoder

### 3.1 Backbone trajectory

`Fact`

- Trong `llama/llama_adapter.py`:
  - `self.traj_encoder = imagebind_model.imagebind_huge()`

- `imagebind_huge()` trong `encoder/models/imagebind_model.py` tạo:
  - `ImageBindModel(out_embed_dim=1024, traj_drop_path=0.7)`

### 3.2 Hai modality riêng: `traj` và `ego`

`Fact`

- `ImageBindModel` chỉ khai báo 2 modality:
  - `TRAJ="traj"`
  - `EGO="ego"`

- `traj` và `ego` có:
  - preprocessor riêng
  - trunk riêng
  - head riêng
  - postprocessor riêng

### 3.3 Shared weights hay không

`Fact`

- `traj` và `ego` **không shared module object hoàn toàn**.
- Chúng có:
  - `traj_stem` riêng
  - `ego_stem` riêng
  - transformer trunk riêng
  - head riêng

`Repo-structure inference`

- Dù kiến trúc đối xứng gần như y hệt nhau, weights giữa `traj` và `ego` là tách riêng.

### 3.4 Patchify / tokenize trajectory

`Fact`

- `TrajPreprocessor.forward()` và `EgoPreprocessor.forward()` đều:
  - `unfold(..., kernel_size=5, step=5)` trên chiều cuối
  - `reshape` thành token sequence
  - qua `traj_stem.proj`
  - thêm `cls_token`
  - thêm positional embedding

- `traj` dùng:
  - `img_size=[6, 5, 5]`
- `ego` dùng:
  - `img_size=[1, 5, 5]`

### 3.5 Hidden size và output dim

`Fact`

- Trong `ImageBindModel`:
  - `traj_embed_dim = 512`
  - `out_embed_dim = 1024`

- Sau trunk:
  - head chọn `CLS token`
  - qua linear lên `1024`
  - postprocessor normalize output

Kết luận:

- output cuối của trajectory encoder trước projector ở `llama_adapter.py` là vector `1024-dim` cho mỗi sample, cho cả:
  - `traj`
  - `ego`

---

## 4. Cách fuse vào LMM

### 4.1 Các projector trước fusion

`Fact`

- Trong `llama_adapter.py`:
  - image branch:
    - `clip_proj: clip_dim -> 768`
    - `clip_proj_norm`
  - traj branch:
    - `traj_proj: 1024 -> 768`
  - ego branch:
    - `ego_proj: 1024 -> 768`

### 4.2 Cách tạo feature từng nhánh

`Fact`

- Ảnh:
  - mỗi frame đi qua `clip_encode_image`
  - giữ toàn bộ spatial tokens
  - concat token của nhiều frame theo chiều sequence

- Ego:
  - `traj_encoder({'ego': ego})['ego']`
  - normalize
  - `unsqueeze(1)`
  - `ego_proj`

- Traj:
  - `traj_encoder({'traj': traj})['traj']`
  - normalize
  - `unsqueeze(1)`
  - `traj_proj`

### 4.3 Rule fusion

`Fact`

- Code hiện tại set:
  - `outputs = [clip_feats, traj_feats, ego_feats]`
  - `outputs_weights = [inputs['Image'][1], inputs['traj'][1], inputs['ego'][1]]`
- Ở call site, cả 3 weight hiện đều đang là `1`
- Fusion thực hiện bằng:
  - `visual_feats = sum(output * output_weight for ...)`

Kết luận:

- Fusion là **cộng trực tiếp** các branch feature sau khi đã projector về cùng dim.
- Không có cross-attention riêng giữa trajectory và image ở bước này.

### 4.4 Visual query và refine

`Fact`

- Model có:
  - `visual_query = nn.Embedding(query_len, v_embed_dim)`
  - mặc định `query_len = 10`

- Sau đó:
  - concat `visual_query` với `visual_feats`
  - đi qua `visual_blocks` (ViT blocks)
  - chỉ giữ lại `query_len` token đầu
  - projector lên dim của LLaMA

Kết luận:

- Trajectory không được đưa trực tiếp thành token text.
- Nó được gộp vào một latent visual query chung.

### 4.5 Injection vào LLM

`Fact`

- `visual_query` sau projector được cộng với `adapter_query` ở các layer cuối của LLaMA:
  - chỉ áp dụng cho `query_layer` layer cuối
  - mặc định `query_layer = 31`

- Mỗi layer cuối nhận:
  - `dynamic_adapter = adapter[layer_idx] + visual_query`

Kết luận:

- Injection diễn ra theo kiểu **adapter prompt trong hidden space của LLM**, không phải prepend token vào text input.

---

## 5. Trainability của trajectory encoder

### 5.1 Fine-tune

`Fact`

- `get_trainable_params('finetune')`:
  - freeze toàn bộ tham số trước
  - chỉ mở trainable cho param có tên chứa:
    - `traj`
    - `ego`
  - ngoài ra trong `llama.*` chỉ mở:
    - `norm`
    - `bias`

Kết luận:

- Ở fine-tune, trajectory/ego branch là phần trainable chính.
- Image branch CLIP không phải trọng tâm trainable.

### 5.2 Pretrain

`Fact`

- `get_trainable_params('pretrain')` cũng chỉ mở các param có tên chứa:
  - `traj`
  - `ego`

`Repo-structure inference`

- Repo hiện tại thật ra đang dùng trajectory branch như phần learnable chính để mang thêm tín hiệu tracking vào mô hình nền.

---

## 6. Hard assumptions của repo

### 6.1 Assumption về dữ liệu trajectory

`Fact`

- Hardcode shape:
  - `traj_tensor = (6,5,5)`
  - `ego_tensor = (1,5,5)`

- Hardcode rule:
  - trajectory đầu tiên là ego
  - các trajectory còn lại là object tracks

### 6.2 Assumption về prompt / dữ liệu DriveLM

`Fact`

- Dataset giả định format instruction-following kiểu:
  - `conversations[0].value` = question
  - `conversations[1].value` = answer

- Dữ liệu ảnh dùng field `image`, có thể là một path hoặc list path.

### 6.3 Assumption về image branch

`Fact`

- Ảnh resize về `224x224`
- Dùng CLIP `ViT-L/14`
- Nếu sample có nhiều ảnh, họ stack frame và concat feature theo sequence

### 6.4 Assumption về fusion

`Fact`

- Weight của `Image`, `traj`, `ego` hiện đang bằng 1 ở call site.
- Không thấy policy adaptive weighting trong code công khai.

### 6.5 Assumption về external preprocessing

`Fact`

- README nói trajectory đến từ 3DMOTFormer.

`Open question`

- Repo không public pipeline đầy đủ để đi từ raw tracking sang đúng tensor `(track, time, 5)` đang dùng ở train.
- Vì vậy phần “semantic meaning” của 5 số mỗi timestep vẫn chưa tái lập được chỉ từ code này.

---

## 7. Những điểm có thể tái dùng cho `restore-779`

Các ý dưới đây chỉ là **điểm có thể tái dùng về mặt concept**, chưa phải đề xuất integration.

`Fact`

- Repo gốc chứng minh một hướng hợp lệ là:
  - tách trajectory thành branch riêng
  - encode trajectory và ego riêng
  - projector về cùng hidden space với image branch
  - fuse ở latent space thay vì nhét trajectory vào prompt text

`Repo-structure inference`

- Những ý tưởng có khả năng tái dùng cho `restore-779`:
  - giữ `ego` tách khỏi `object trajectories`
  - hard-cap số track và timestep để tensor shape ổn định
  - encode trajectory bằng một encoder riêng, không ép dùng chung image encoder
  - fuse trajectory ở hidden space thay vì string hóa trajectory

---

## 8. Những điểm chưa được phép kết luận về integration

`Open question`

- Có nên reuse nguyên `ImageBind` hay không
- Có nên giữ shape `(6,5,5)` hay thiết kế shape khác cho WAD
- 5 feature values mỗi timestep của WAD nên là gì
- Có nên fuse trajectory:
  - trước `mlp1`
  - sau `pixel_shuffle`
  - hay như một branch song song với Q-Former
- Trajectory encoder trong repo của bạn nên đi cùng:
  - InternVL no-qformer
  - InternVL qformer
  - hay cả hai với seam khác nhau

Những điểm này cần một pha thảo luận riêng sau báo cáo này.

---

## Kết luận ngắn

`TrackingMeetsLMM` đang dùng một thiết kế khá rõ:

- trajectory không đi vào text prompt
- trajectory không đi qua Q-Former
- trajectory được encode bằng một encoder riêng cho `traj` và `ego`
- sau đó fuse với image feature trong latent space
- rồi mới inject vào LLM bằng adapter-style hidden prompts

Đây là một baseline nghiên cứu tốt để tham khảo về **ý tưởng branch trajectory riêng**, nhưng chưa đủ để kết luận rằng repo `restore-779` nên copy nguyên kiến trúc này.
