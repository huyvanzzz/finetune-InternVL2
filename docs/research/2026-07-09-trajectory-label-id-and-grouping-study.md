# Nghiên cứu lại vấn đề `map id` và việc có cần gộp label hay không

## 1. Câu hỏi cần verify

Mục tiêu của báo cáo này là trả lời lại thật rõ hai câu hỏi đang còn chưa chắc:

1. Với trajectory/object branch dạng structured input, có nên đưa `label/category` vào model bằng cách `map sang id rồi học embedding` hay không?
2. Nếu đã dùng `id embedding`, có nên giữ riêng toàn bộ label hay gộp bớt các label hiếm thành taxonomy nhỏ hơn?

Điểm cần tách bạch ngay từ đầu:

- `map id` và `gộp label` là **hai quyết định khác nhau**.
- Có thể:
  - `map id` nhưng **không gộp**
  - `map id` và **có gộp**
  - **không** `map id`, mà dùng `text label`

Báo cáo này chỉ nghiên cứu để ra quyết định, chưa code, chưa chốt integration.

---

## 2. Baseline dữ liệu hiện tại

`Data-evidenced`

Nguồn dữ liệu local đang bám:

- file gốc: `.worktrees/restore-779cc7b/image/results_botsort.jsonl`
- file top-6 đã lọc theo priority: `.worktrees/restore-779cc7b/image/results_botsort_top6.json`

### 2.1. Cấu trúc object hiện tại

Mỗi object trong file top-6 hiện có các trường chính:

- `label`
- `boxs`
- `distance_norm`
- `relative_position`
- `movement_angle`
- `speed_percent`
- `area`
- `proximity_score`
- `direction_weight`
- `priority_score`

Ở mức ý tưởng branch đang bàn, mỗi object sẽ có:

- category signal: `label`
- geometry signal: từ bbox
- motion/proximity signal:
  - `distance_norm`
  - `movement_angle`
  - `speed_percent`
  - `direction_weight`

### 2.2. Quy mô label thực tế

#### Trên file nguồn `results_botsort.jsonl`

- Tổng object rows: `825`
- Số label khác nhau: `78`
- Số label xuất hiện đúng `1` lần: `31`
- Số label xuất hiện không quá `2` lần: `43`
- Số label xuất hiện không quá `5` lần: `59`

Top label:

- `person`: 153
- `car_(automobile)`: 132
- `streetlight`: 81
- `street_sign`: 65
- `bicycle`: 47
- `pole`: 41
- `wheel`: 29
- `cone`: 24
- `signboard`: 17
- `motorcycle`: 15

#### Trên file top-6 `results_botsort_top6.json`

- Tổng object sau lọc: `447`
- Số label khác nhau: `51`
- Số label xuất hiện đúng `1` lần: `17`
- Số label xuất hiện không quá `2` lần: `27`
- Số label xuất hiện không quá `5` lần: `35`

Top label:

- `person`: 107
- `car_(automobile)`: 47
- `streetlight`: 42
- `bicycle`: 32
- `street_sign`: 31
- `pole`: 27
- `cone`: 18
- `vent`: 11
- `bus_(vehicle)`: 10
- `trousers`: 10

### 2.3. Kết luận dữ liệu nền

`Data-evidenced`

- Label space hiện tại **không quá lớn** ở mức implementation: top-6 chỉ còn `51` label.
- Nhưng dữ liệu có **đuôi dài khá nặng**:
  - hơn một nửa label top-6 xuất hiện không quá 5 lần
  - hơn một nửa label top-6 xuất hiện không quá 2 lần
- Vì vậy:
  - câu hỏi `map id` là câu hỏi về **cách biểu diễn category**
  - câu hỏi `gộp label` là câu hỏi về **xử lý long-tail**

Đây là hai bài toán liên quan nhau, nhưng không phải một.

---

## 3. Các hướng biểu diễn category

## 3.1. Hướng A: giữ label như text rồi dùng tokenizer / text embedding

Ví dụ:

- `person`
- `street_sign`
- `car_(automobile)`

được đưa vào dưới dạng text token hoặc text serialization.

### Precedent tìm được

`Paper-evidenced`

- `TabLLM` cho thấy có thể biến dữ liệu bảng thành text rồi giao cho language model xử lý bằng serialization.  
  Nguồn: https://proceedings.mlr.press/v206/hegselmann23a.html
- Các hướng “language models on tabular data” cũng nghiên cứu text serialization như một tuyến hợp lệ khi đầu vào được biến thành câu hoặc template text.  
  Nguồn tổng quan: https://openreview.net/forum?id=6o3vVBWYis

### Đánh giá cho case của repo này

`Research-supported inference`

Hướng này **có precedent**, nhưng precedent mạnh nhất của nó nằm ở bối cảnh:

- toàn bộ sample được serialize thành text
- rồi đưa vào một language model xử lý như NLP input

Nó **không phải precedent mạnh nhất** cho case hiện tại, vì branch đang bàn là:

- compact structured branch
- mỗi sample chỉ có `top-6 object`
- muốn tạo object token gọn, ổn định và dễ học

Rủi ro thực tế của hướng này trong case hiện tại:

- tokenizer có thể tách label thành nhiều subword không nhất quán
- label kỹ thuật như `car_(automobile)` hoặc `street_sign` không phải dạng text tự nhiên đẹp
- branch structured nhỏ sẽ phải dựa khá nhiều vào prior của tokenizer/LM thay vì học một embedding category gọn và trực tiếp

Kết luận cho Hướng A:

- hợp lệ nếu muốn **serialize toàn bộ object branch thành text**
- nhưng **không phải lựa chọn tự nhiên nhất** cho một structured object encoder nhỏ

## 3.2. Hướng B: map label/category sang id rồi học embedding

Ví dụ:

- `person -> 0`
- `car_(automobile) -> 1`
- `street_sign -> 2`

Sau đó model học embedding vector cho từng id.

### Precedent tìm được

`Paper-evidenced`

- `Entity Embeddings of Categorical Variables` là precedent rất trực tiếp cho `category id -> learned embedding`, và paper này còn nhấn mạnh hướng đó hữu ích khi dữ liệu sparse hoặc high-cardinality.  
  Nguồn: https://arxiv.org/abs/1604.06737
- `FT-Transformer` biến cả feature categorical và numerical thành embeddings, rồi mới đưa qua Transformer.  
  Nguồn: https://openreview.net/pdf?id=i_Q1yrOegLY
- `SAINT` cũng đi theo hướng chiếu riêng categorical và continuous features vào dense space trước khi attention.  
  Nguồn: https://arxiv.org/abs/2106.01342

### Đánh giá cho case của repo này

`Paper-evidenced` + `Research-supported inference`

Đây là hướng có precedent mạnh nhất cho object branch dạng structured vì:

- category là một biến rời rạc hữu hạn
- số label top-6 hiện tại chỉ khoảng `51`, nên embedding table nhỏ
- category embedding có thể được ghép với geometry/motion vector để tạo object token

Điểm quan trọng:

- `map id` **không có nghĩa** là model mất semantic signal
- semantic nằm ở **embedding được học**
- nếu hai label thường xuất hiện trong pattern gần nhau, embedding có thể tự học để ở gần nhau hơn

Kết luận cho Hướng B:

- với kiến trúc structured branch đang bàn, đây là hướng có **cơ sở mạnh nhất**

## 3.3. Hướng C: bỏ category signal, chỉ dùng feature số

Tức là chỉ giữ:

- bbox / area
- distance
- direction prior
- motion cues

mà không đưa label/category vào.

### Precedent và đánh giá

`Research-supported inference`

Trong trajectory forecasting thuần hình học, chuyện chỉ dùng vector số là có precedent. Ví dụ `VectorNet` tập trung mạnh vào vectorized geometry/dynamics.  
Nguồn: https://openaccess.thecvf.com/content_CVPR_2020/html/Gao_VectorNet_Encoding_HD_Maps_and_Agent_Dynamics_From_Vectorized_Representation_CVPR_2020_paper.html

Nhưng với bài toán hiện tại, bỏ category signal sẽ làm mất thông tin phân biệt:

- người
- xe
- biển báo
- cọc tiêu
- thùng rác

Trong khi object-centric multimodal papers thường nhấn mạnh explicit semantic/object tokens:

- `Chat-Scene`: object identifiers và object-centric representations  
  https://arxiv.org/abs/2312.08168
- `VideoOrion`: object-centric branch với object tokens mang semantics và spatial-temporal information  
  https://openaccess.thecvf.com/content/ICCV2025/papers/Feng_VideoOrion_Tokenizing_Object_Dynamics_in_Videos_ICCV_2025_paper.pdf

Kết luận cho Hướng C:

- có thể dùng làm baseline ablation
- nhưng **không phải lựa chọn khuyến nghị chính** nếu mục tiêu là object-aware reasoning

---

## 4. Research về long-tail và grouping

## 4.1. Literature có ủng hộ grouping/hierarchy không?

`Paper-evidenced`

Có, nhưng theo kiểu:

- hierarchy/superclass là **một công cụ để xử lý long-tail**
- chứ không phải luật mặc định bắt buộc phải gộp mọi taxonomy nhỏ

Một số precedent chính:

- `Adaptive Hierarchical Representation Learning for Long-Tailed Object Detection (AHRL)` chỉ ra rằng các class hiếm dễ cluster với các class tương tự và dùng coarse-to-fine grouping/hierarchy để cải thiện biểu diễn.  
  Nguồn: https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Adaptive_Hierarchical_Representation_Learning_for_Long-Tailed_Object_Detection_CVPR_2022_paper.pdf
- `Towards Long-Tailed 3D Detection` nhấn mạnh trong driving, quan hệ giữa class với superclass có ý nghĩa thực dụng; paper còn đánh giá một phần credit dựa trên quan hệ liên lớp.  
  Nguồn: https://proceedings.mlr.press/v205/peri23a.html

### Ý nghĩa rút ra

`Research-supported inference`

Literature ủng hộ các ý sau:

- long-tail là vấn đề thật
- hierarchy hoặc superclass có thể hữu ích
- nhầm giữa class gần nhau về ngữ nghĩa đôi khi ít nghiêm trọng hơn nhầm hoàn toàn

Nhưng literature **không nói** rằng:

- cứ ít mẫu là phải gộp ngay
- hay đúng taxonomy `18 nhóm` hiện tại là tối ưu

## 4.2. Có precedent mạnh cho việc “giữ riêng tất cả label” không?

`Paper-evidenced`

Trong rất nhiều mô hình categorical embedding hoặc object classification, mỗi class vẫn có id riêng. Điều này là bình thường nếu:

- label space chưa quá lớn
- embedding table chưa thành vấn đề

Với top-6 hiện tại:

- `51` label là không lớn về mặt tham số
- chi phí embedding riêng cho từng label gần như không đáng kể

### Vấn đề thật không nằm ở kích thước bảng embedding

`Data-evidenced`

Vấn đề lớn hơn là:

- nhiều label quá hiếm
- tail labels có rất ít cơ hội học embedding ổn định

Nói cách khác:

- lý do để cân nhắc grouping là **vấn đề dữ liệu và generalization**
- không phải vì `51 label` là quá nhiều về mặt model size

## 4.3. Có precedent mạnh cho taxonomy gộp kiểu hiện tại không?

`Open gap`

Tôi **không tìm thấy precedent trực tiếp** nào nói rằng với đúng kiểu dữ liệu hiện tại thì nên:

- giữ một số label head riêng
- rồi gộp tail về đúng taxonomy `18 nhóm` như đã phác thảo

Literature chỉ ủng hộ ở mức:

- có thể dùng hierarchy
- có thể dùng grouping coarse-to-fine
- có thể tận dụng quan hệ semantic giữa class head và tail

Nhưng:

- đúng nhóm nào gộp với nhóm nào
- bao nhiêu nhóm là hợp lý
- bỏ hẳn nhóm nào khỏi branch

vẫn là **design choice của repo**, chưa phải fact được paper đóng dấu sẵn.

---

## 5. Bảng đối chiếu precedent

| Paper/repo | Domain | Loại input category | Dùng id embedding | Có numeric feature branch | Có bàn long-tail / hierarchy | Có object/set encoder | Mức độ liên quan | Nhận xét ngắn |
|---|---|---|---|---|---|---|---|---|
| Entity Embeddings of Categorical Variables | Tabular DL | categorical variables | Có | Không phải trọng tâm | Không | Không | Trực tiếp | Precedent mạnh nhất cho `category -> id embedding` |
| FT-Transformer | Tabular DL | categorical + numerical features | Có | Có | Không | Có | Trực tiếp | Rất gần pattern `category embedding + numeric embedding + transformer` |
| SAINT | Tabular DL | categorical + continuous features | Có | Có | Không | Có | Trực tiếp | Củng cố mạnh cho mixed-feature tokenization |
| TabLLM | Tabular + LLM | category dưới dạng text serialization | Không bắt buộc id embedding | Có thể serialize kèm số | Không | Không phải trọng tâm | Gần | Ủng hộ hướng dùng text, nhưng theo kiểu serialize cả sample thành ngôn ngữ tự nhiên |
| Chat-Scene | 3D MLLM | object identifiers / object-centric representations | Có object identifier | Có object-level features | Không | Có object sequence | Gần | Ủng hộ mạnh cho object-level symbolic identity thay vì bỏ category |
| VideoOrion | Video-LLM | tracked objects / object tokens | Có semantics ở object tokens | Có spatial-temporal dynamics | Không | Có object branch | Gần | Ủng hộ object-centric branch riêng, nhưng không trả lời trực tiếp chuyện taxonomy gộp |
| Adaptive Hierarchical Representation Learning for LTOD | Long-tail object detection | fine-grained classes | Không phải trọng tâm chính | Không phải trọng tâm | Có | Không | Gần | Ủng hộ grouping/hierarchy như công cụ xử lý tail |
| Towards Long-Tailed 3D Detection | Driving 3D detection | driving classes + superclass relations | Không phải trọng tâm chính | Không phải trọng tâm | Có | Không | Gần | Ủng hộ việc tận dụng inter-class relations trong driving long-tail |

---

## 6. Những phần đã có bằng chứng mạnh

## 6.1. `Map id` cho category là hợp lý

`Paper-evidenced`

Nếu branch đang bàn là:

- structured
- object-level
- compact
- không serialize cả sample thành câu

thì `label/category -> id -> learned embedding` là lựa chọn có precedent mạnh nhất.

## 6.2. `Map id` không đồng nghĩa với `gộp`

`Paper-evidenced`

Có thể hoàn toàn:

- giữ nguyên `51` label top-6
- map mỗi label sang một id riêng
- học embedding riêng cho từng label

Không có gì trong literature buộc phải gộp chỉ vì đã dùng id embedding.

## 6.3. Grouping/hierarchy là công cụ hợp lệ cho long-tail

`Paper-evidenced`

Grouping hoặc superclass reasoning có cơ sở trong long-tail object detection và driving detection. Tức là ý tưởng “tail labels có thể chia sẻ signal qua superclass” là hợp lý.

---

## 7. Những phần còn là design choice

## 7.1. Exact taxonomy `18 nhóm`

`Open gap`

Tôi chưa có bằng chứng đủ mạnh để nói rằng taxonomy `18 nhóm` hiện tại là mặc định đúng hoặc tối ưu.

Điều chưa được literature chốt hộ mình:

- có nên gộp `wheel` vào đâu
- có nên bỏ `misc`
- có nên giữ `animal` riêng
- có nên để `street_sign` và `signboard` tách hay gộp

Những chuyện này hiện vẫn là engineering choice.

## 7.2. Có nên gộp ngay ở phiên bản đầu hay không

`Research-supported inference`

Với đúng dữ liệu top-6 hiện tại:

- label space chỉ `51`
- embedding table nhỏ
- grouping hiện tại lại chưa có precedent trực tiếp

nên việc gộp ngay từ đầu **không phải lựa chọn an toàn nhất về mặt khoa học**.

Lựa chọn an toàn hơn là:

1. baseline trước bằng `raw labels riêng hết`
2. nếu cần, thêm ablation `head riêng + tail gộp`
3. chỉ sau đó mới quyết định taxonomy nhỏ có đáng giữ lâu dài hay không

## 7.3. Dùng text label thay vì id embedding cho branch này

`Research-supported inference`

Nếu sau này muốn làm một nhánh hoàn toàn theo hướng LLM-native serialization thì text label là một hướng đáng thử.

Nhưng với compact structured branch hiện đang bàn, tôi **không thấy đủ cơ sở để ưu tiên text label hơn id embedding**.

---

## 8. Kết luận khuyến nghị cho repo hiện tại

## Kết luận cuối

**Kết luận D** ở câu hỏi về grouping:

- với dữ liệu hiện tại, **nên giữ riêng toàn bộ label trước thay vì gộp ngay**

Đồng thời, ở câu hỏi về biểu diễn category:

- **nên dùng `map id -> learned embedding`**

Tức là khuyến nghị đầy đủ cho repo hiện tại là:

1. **Giữ category signal**
2. **Map mỗi label sang một id riêng**
3. **Chưa gộp taxonomy ở baseline đầu tiên**
4. Xem grouping là **ablation hoặc v2**, không xem là mặc định đã được chứng minh

### Vì sao đây là khuyến nghị mạnh nhất lúc này

`Paper-evidenced` + `Data-evidenced` + `Research-supported inference`

- `map id` có precedent mạnh
- `51` label top-6 không lớn đến mức phải gộp vì lý do kích thước model
- grouping có cơ sở như một công cụ long-tail, nhưng exact grouping hiện tại chưa có đủ bằng chứng trực tiếp

### Thứ tự thử hợp lý nhất sau báo cáo này

Nếu sau này đi đến pha code và ablation, thứ tự hợp lý nhất là:

1. `raw label riêng hết + id embedding`
2. `raw label riêng hết + id embedding + set/object encoder`
3. `head riêng + tail gộp` như một ablation có kiểm soát
4. chỉ giữ taxonomy nhỏ nếu ablation chứng minh nó thật sự tốt hơn

### Một câu chốt ngắn

`Map id` là quyết định có cơ sở mạnh.  
`Gộp label` hiện tại thì chưa đủ cơ sở để coi là mặc định đúng.

---

## 9. Nguồn chính đã dùng

- Entity Embeddings of Categorical Variables: https://arxiv.org/abs/1604.06737
- FT-Transformer: https://openreview.net/pdf?id=i_Q1yrOegLY
- SAINT: https://arxiv.org/abs/2106.01342
- TabLLM: https://proceedings.mlr.press/v206/hegselmann23a.html
- Chat-Scene: https://arxiv.org/abs/2312.08168
- VideoOrion: https://openaccess.thecvf.com/content/ICCV2025/papers/Feng_VideoOrion_Tokenizing_Object_Dynamics_in_Videos_ICCV_2025_paper.pdf
- Adaptive Hierarchical Representation Learning for Long-Tailed Object Detection: https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Adaptive_Hierarchical_Representation_Learning_for_Long-Tailed_Object_Detection_CVPR_2022_paper.pdf
- Towards Long-Tailed 3D Detection: https://proceedings.mlr.press/v205/peri23a.html
