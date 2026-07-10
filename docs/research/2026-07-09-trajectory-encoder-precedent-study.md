# Nghiên cứu precedent cho trajectory branch kiểu "3 phần encoder"

## 1. Câu hỏi cần verify

Mục tiêu của báo cáo này là kiểm tra xem ý tưởng trajectory branch hiện đang bàn có precedent đến đâu trong literature và repo công khai.

Kiến trúc cần verify là:

- một nhánh cho `category/object-type`
- một nhánh cho `numerical geometry/motion`
- một `object/set encoder` để trộn các object token

Các câu hỏi chính:

1. Đã có ai dùng `category embedding + numeric projection + set/sequence encoder` chưa?
2. Trong driving / multimodal VLM, đã có ai tách object/trajectory branch riêng chưa?
3. Nếu không có precedent trùng hẳn, từng mảnh của ý tưởng có được hỗ trợ mạnh không?
4. Taxonomy `18 nhóm` hiện tại có đi ngược precedent phổ biến không?

---

## 2. Kiến trúc đang bàn là gì

`Data-evidenced`

Baseline hiện tại đang bàn, ở mức rất ngắn:

- input: top-6 object từ tracking
- mỗi object có:
  - `label/group`
  - `bbox`
  - `area`
  - `distance_norm`
  - `direction_weight`
  - `movement_angle`
  - `speed_percent`
- ý tưởng encoder:
  - `Layer A`: `group_id -> learned embedding`
  - `Layer B`: vector số -> `MLP/projection`
  - `Layer C`: ghép hai nhánh thành `object token`
  - `Layer D`: đưa dãy object token vào `Transformer / set encoder`
  - `Layer E`: dùng branch này làm đầu vào structured cho mô hình đa phương thức phía sau

Điểm cần verify không phải là toàn bộ pipeline VLM, mà là **riêng tính hợp lý của trajectory/object branch** này.

---

## 3. Các paper/repo liên quan

### 3.1 Tabular / mixed-feature modeling

1. **Entity Embeddings of Categorical Variables**  
   ArXiv 2016: https://arxiv.org/abs/1604.06737

2. **FT-Transformer: Revisiting Deep Learning Models for Tabular Data**  
   OpenReview / arXiv 2021: https://openreview.net/pdf?id=i_Q1yrOegLY

3. **SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training**  
   PDF: https://table-representation-learning.github.io/assets/papers/saint_improved_neural_networks.pdf

4. **On Embeddings for Numerical Features in Tabular Deep Learning**  
   OpenReview 2022: https://openreview.net/forum?id=pfI7u0eJAIr

### 3.2 Trajectory / motion / set encoding

5. **Set Transformer**  
   PMLR 2019: https://proceedings.mlr.press/v97/lee19d.html

6. **VectorNet: Encoding HD Maps and Agent Dynamics From Vectorized Representation**  
   CVPR 2020: https://openaccess.thecvf.com/content_CVPR_2020/html/Gao_VectorNet_Encoding_HD_Maps_and_Agent_Dynamics_From_Vectorized_Representation_CVPR_2020_paper.html

### 3.3 Driving / navigation multimodal

7. **TrackingMeetsLMM**  
   Repo: https://github.com/mbzuai-oryx/TrackingMeetsLMM  
   Local audit trong repo hiện tại: `docs/research/2026-07-09-trackingmeetslmm-trajectory-encoder-audit.md`

8. **Towards Long-Tailed 3D Detection**  
   CoRL / PMLR: https://proceedings.mlr.press/v205/peri23a.html

### 3.4 Object-centric multimodal modeling

9. **VideoOrion: Tokenizing Object Dynamics in Videos**  
   ICCV 2025: https://openaccess.thecvf.com/content/ICCV2025/html/Feng_VideoOrion_Tokenizing_Object_Dynamics_in_Videos_ICCV_2025_paper.html

10. **Chat-Scene: Bridging 3D Scene and Large Language Models with Object Identifiers**  
    ArXiv / NeurIPS 2024: https://arxiv.org/abs/2312.08168

11. **Direction-aware 3D Large Multimodal Models**  
    CVPR 2026: https://openaccess.thecvf.com/content/CVPR2026/html/Liu_Direction-aware_3D_Large_Multimodal_Models_CVPR_2026_paper.html

---

## 4. Bảng đối chiếu precedent

| Paper/repo | Domain | Input type | Category embedding | Numeric projection riêng | Object/set encoder | Downstream multimodal/driving | Mức độ liên quan | Nhận xét ngắn |
|---|---|---|---|---|---|---|---|---|
| Entity Embeddings of Categorical Variables | Tabular DL | categorical features | Có | Không phải trọng tâm | Không | Không | Gián tiếp | Chứng minh mạnh cho việc map category id thành learned embedding |
| FT-Transformer | Tabular DL | mixed categorical + numerical features | Có | Có | Có | Không | Gần | Precedent rất mạnh cho pattern “feature embedding -> transformer” |
| SAINT | Tabular DL | mixed categorical + continuous features | Có | Có | Có | Không | Gần | Nói rất rõ việc chiếu cả continuous và categorical vào cùng embedding space rồi dùng attention |
| On Embeddings for Numerical Features in Tabular Deep Learning | Tabular DL | numerical features | Không | Có | Có thể dùng cùng transformer-like models | Không | Gián tiếp | Củng cố riêng cho nhánh numeric embedding/projection |
| Set Transformer | Set modeling | unordered set elements | Không | Không phải trọng tâm | Có | Không | Gián tiếp | Precedent nền tảng cho việc dùng attention để model tương tác giữa các phần tử trong set |
| VectorNet | Trajectory prediction | vectorized agent/map features | Không rõ category branch | Có, theo vector agent/map | Có (local graph + global graph) | Không phải VLM | Gần | Rất mạnh cho ý tưởng encode object/agent bằng vector số rồi model tương tác toàn cục |
| TrackingMeetsLMM | Driving MLLM | image + traj + ego tensors | Không | Có | Có trajectory encoder riêng | Có, driving | Gần | Chứng minh mạnh cho branch trajectory riêng trong driving MLLM, nhưng không có category embedding branch kiểu mình bàn |
| Towards Long-Tailed 3D Detection | 3D driving detection | fine-grained classes + hierarchy | Có hierarchy class/superclass | Không phải trọng tâm | Không | Có, driving perception | Gián tiếp | Hỗ trợ cho ý tưởng dùng superclass / hierarchy thay vì label quá vụn |
| VideoOrion | Video-LLM | tracked object dynamics + context | Có object semantics theo detect-segment-track | Có object projector | Có object token branch | Có, multimodal | Gần | Precedent mạnh cho object-centric branch riêng từ tracking rồi đưa token vào LLM |
| Chat-Scene | 3D multimodal LLM | object proposals + object identifiers | Có object identifiers | Có object-level embeddings 2D/3D | Dùng object sequence | Có, multimodal | Gần | Chứng minh mạnh cho object-centric token sequence, nhưng không theo kiểu tabular cat+num tách rời |
| Direction-aware 3D LMMs | 3D multimodal LLM | detector/instance-based object tokens | Có ở mức detector-based object tokens | Có geometry/pose supplementation | Có ở mức object token input | Có, multimodal | Gián tiếp | Hữu ích để xác nhận detector-based object token pipelines là một nhánh đã được công nhận |

---

## 5. Những phần đã có bằng chứng mạnh

### 5.1 `Layer A`: category id -> learned embedding

`Paper-evidenced`

- `Entity Embeddings of Categorical Variables` cho thấy việc map biến categorical vào learned embedding là hoàn toàn chuẩn và hữu ích, nhất là khi dữ liệu sparse hoặc high-cardinality.
- `FT-Transformer` và `SAINT` đều dùng tư duy “mỗi feature được chiếu thành embedding”, trong đó categorical features là đối tượng tự nhiên của learned embeddings.

Kết luận:

- dùng `group_id -> embedding` **không phải ý tưởng mới**
- đây là precedent mạnh

### 5.2 `Layer B`: geometry/motion vector -> MLP/projection

`Paper-evidenced`

- `On Embeddings for Numerical Features in Tabular Deep Learning` hỗ trợ trực tiếp việc biến numeric feature thành vector embedding thay vì giữ scalar thô.
- `SAINT` nêu rất rõ chuyện chiếu cả continuous và categorical vào cùng dense space.
- `VectorNet` chứng minh trong bài toán trajectory, vector hình học/động học hoàn toàn có thể là đối tượng encode chính.

Kết luận:

- nhánh `numeric projection` là **có precedent mạnh**

### 5.3 `Layer C`: ghép category embedding + numeric projection thành object token

`Research-supported inference`

- `FT-Transformer` và `SAINT` không nói theo ngôn ngữ “object token”, nhưng về mặt cơ chế thì đã có pattern:
  - categorical embedding
  - numerical projection
  - đưa vào cùng không gian embedding
- Với bài toán của mình, mỗi object đóng vai trò như một “row nhỏ” hoặc một “entity”.

Kết luận:

- ý tưởng ghép `category branch + numeric branch` thành `object token` **được hỗ trợ khá mạnh theo precedent gần**
- nhưng chưa thấy paper driving-VLM công khai nào làm đúng câu này theo cùng ngôn ngữ

### 5.4 `Layer D`: object token -> set/sequence encoder

`Paper-evidenced`

- `Set Transformer` là precedent nền rất rõ cho việc dùng attention để xử lý một tập phần tử và học tương tác giữa chúng.
- `VectorNet` là precedent rất mạnh trong trajectory/motion forecasting cho chuyện:
  - encode từng thực thể
  - sau đó model tương tác ở tầng toàn cục

Kết luận:

- dùng set/sequence encoder cho top-k objects là **có precedent mạnh**

### 5.5 `Layer E`: branch object/trajectory -> downstream multimodal model

`Repo-evidenced`

- `TrackingMeetsLMM` chứng minh mạnh rằng trong driving MLLM, trajectory branch riêng là hợp lệ.
- `VideoOrion` chứng minh object-centric branch từ detect-segment-track rồi biến thành object tokens là một hướng hợp lệ trong multimodal LLM.
- `Chat-Scene` chứng minh object-level token sequence là một abstraction mạnh cho multimodal reasoning.

Kết luận:

- việc có một branch structured/object-centric đưa sang multimodal model **có precedent khá mạnh**

---

## 6. Những phần chỉ có bằng chứng gián tiếp

### 6.1 Tổ hợp đúng kiểu “cat branch + num branch + set encoder” trong driving VLM

`Open gap`

Tôi **không tìm thấy precedent trực tiếp** công khai nào trong driving VLM / driving MLLM làm đúng trọn bộ:

- `category embedding`
- `numeric motion/geometry projection`
- `object token fusion`
- `set/sequence encoder`
- rồi nối vào VLM/LLM downstream

Điều tôi tìm thấy là:

- từng mảnh riêng lẻ có precedent mạnh
- một số object-centric multimodal model rất gần
- một số trajectory/dynamics model rất gần

Nhưng **bản ghép đúng như mình đang bàn** chưa thấy paper công khai nào trùng khít.

### 6.2 Taxonomy 18 nhóm hiện tại

`Research-supported inference`

Literature không buộc phải chọn một trong hai cực:

- raw fine-grained labels
- superclass rất thô

`Towards Long-Tailed 3D Detection` hỗ trợ khá rõ chuyện:

- class hierarchy có ích
- tail classes có thể được phân tích theo semantic superclass
- nhầm giữa các lớp gần nhau về ngữ nghĩa đôi khi nên được coi là lỗi nhẹ hơn

Điều này ủng hộ hướng:

- không cần giữ label quá vụn
- dùng `mixed head-label + tail-superclass` là hợp lý

Kết luận:

- bản `18 nhóm` hiện tại **không đi ngược precedent phổ biến**
- nhưng cũng chưa được literature “chứng minh là tối ưu”

---

## 7. Những phần còn là thiết kế mới

### 7.1 Chính xác hóa object feature set cho WAD

`Open gap`

Phần sau vẫn là thiết kế mới của mình, chưa thấy precedent trực tiếp cho đúng bài toán này:

- chọn đúng bộ feature:
  - `cx, cy, w, h, area`
  - `distance_norm`
  - `direction_weight`
  - `movement_angle`
  - `speed_percent`
- chỉ dùng `1 frame`
- bỏ `ego`
- chỉ giữ `top-6 object`

Điều này không có nghĩa là sai, mà nghĩa là:

- literature chỉ hỗ trợ ở mức pattern
- cấu hình cụ thể này vẫn là thiết kế tùy biến cho repo của mình

### 7.2 Cách taxonomy 18 nhóm được chốt

`Open gap`

Literature hỗ trợ mạnh cho:

- category embedding
- grouped hierarchy
- head-tail thinking

Nhưng không có paper nào nói rằng đúng với file label hiện tại của mình thì:

- `18 nhóm`
- và đúng mapping hiện tại

là tối ưu.

Tức là:

- hướng là hợp lý
- mapping cụ thể vẫn là engineering design choice

---

## 8. Kết luận: mức độ đáng tin của hướng hiện tại

### Kết luận chính

**Kết luận B**: từng mảnh của kiến trúc đều có precedent khá mạnh, nhưng tổ hợp hiện tại vẫn là một thiết kế ghép mới.

### Giải thích ngắn

`Paper-evidenced`

- category embedding: có precedent mạnh
- numeric projection: có precedent mạnh
- set/object encoder: có precedent mạnh
- object-centric / trajectory branch trong multimodal model: có precedent khá mạnh

`Open gap`

- chưa tìm thấy precedent trực tiếp công khai cho đúng tổ hợp:
  - `category branch`
  - `numeric branch`
  - `object token fusion`
  - `set encoder`
  - `driving VLM downstream`
  - trong đúng cấu hình `single-frame top-6 tracked objects`

### Đánh giá cuối cùng

Hướng hiện tại **đáng tin để đi tiếp ở mức research engineering**, vì:

- nó không phải ý tưởng “ảo giác từ không khí”
- mỗi tầng chính đều có chỗ dựa từ literature

Nhưng cũng cần giữ thái độ đúng:

- đây **không phải** một kiến trúc đã được paper driving-VLM nào “đóng dấu sẵn”
- nó là một **tổ hợp mới nhưng có nền precedent tốt**

Nếu muốn đi tiếp một cách thận trọng, bước hợp lý sau báo cáo này là:

1. giữ kiến trúc ở mức đơn giản nhất
2. tránh thêm quá nhiều nhánh phụ
3. dùng ablation để kiểm tra:
   - chỉ numeric
   - numeric + category
   - numeric + category + set encoder

Đó sẽ là cách công bằng nhất để xác nhận liệu precedent lý thuyết này có thực sự chuyển thành lợi ích trong repo của mình hay không.
