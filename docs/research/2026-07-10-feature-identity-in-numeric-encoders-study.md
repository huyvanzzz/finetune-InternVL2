# Nghiên cứu riêng: trong numeric encoder, “tên/ý nghĩa của từng giá trị” được encode như thế nào

## 1. Câu hỏi cần verify

Mục tiêu của báo cáo này là trả lời câu hỏi rất cụ thể sau:

- ở các paper dùng `numeric features`
- nhất là structured/tabular/trajectory/object branches
- nếu họ **không đưa field name/key ra thành text**

thì model biết “giá trị này là distance, giá trị kia là speed” bằng cách nào.

Điểm phải khóa ngay từ đầu:

- “không có field name text” **không đồng nghĩa** với “model hoàn toàn tự đoán từ số trần”
- trong nhiều kiến trúc, field identity vẫn tồn tại nhưng được encode **ngầm trong kiến trúc**

Mục tiêu của research là tách bạch rõ 4 khả năng:

1. `Slot-only`
2. `Field-specific parameters`
3. `Explicit field embedding`
4. `Textual/verbalized field semantics`

---

## 2. Vấn đề này trong repo hiện tại thực sự là gì

`Data-evidenced`

Trong context repo hiện tại, branch đang bàn là:

- `top-6 object`
- `1 frame`
- không `ego`
- downstream cuối là `alter`

Với mỗi object, nếu đi theo nhánh numeric thì ta có thể dùng các tín hiệu như:

- `cx`
- `cy`
- `w`
- `h`
- `area`
- `distance_norm`
- `direction_weight`
- `movement_angle`
- `speed_percent`

Câu hỏi không phải là “các số này có ích không”, mà là:

- model biết `chiều thứ 1` là `cx` bằng cách nào
- model biết `chiều thứ 6` là `distance_norm` bằng cách nào
- nếu không nói ra bằng text, thì thông tin field identity đang nằm ở đâu

Đây là câu hỏi cực quan trọng, vì nếu hiểu sai chỗ này thì rất dễ tưởng rằng các paper numeric encoder chỉ “ném một đống số vào rồi để model tự giác ngộ”.

---

## 3. Bốn mức encode field identity

## 3.1. Mức A: `vị trí thuần`

Model chỉ biết nghĩa của giá trị vì:

- cột/chiều luôn đứng ở vị trí cố định

Ví dụ:

- chiều 1 luôn là `cx`
- chiều 2 luôn là `cy`
- chiều 3 luôn là `w`

Tức là field identity đến từ **format cố định của input**, không có machinery riêng cho từng field ngoài chuyện “nó luôn ở cùng một chỗ”.

## 3.2. Mức B: `vị trí + tham số riêng`

Model không thấy field name dưới dạng text, nhưng mỗi field/cột có:

- projection riêng
- weight riêng
- tokenizer riêng
- bias riêng

Khi đó field identity được encode ngầm trong tham số của kiến trúc.

Đây là dạng rất phổ biến trong tabular DL hiện đại.

## 3.3. Mức C: `field identity tường minh trong latent space`

Field identity được làm tường minh hơn bằng:

- feature embedding
- column embedding
- field token
- channel marker

Nó vẫn chưa phải text/natural language, nhưng đã vượt xa kiểu chỉ dựa vào vị trí cố định.

## 3.4. Mức D: `field identity được verbalize`

Field name hoặc semantics của field được đưa sang language side, ví dụ:

- `The age is 42`
- `distance = 0.23`
- `channel_x_accel`

Đây là cách mạnh nhất nếu muốn tận dụng trực tiếp language prior của LLM.

---

## 4. Các cụm research đã quét

## 4.1. Tabular / mixed-feature modeling

Nguồn chính:

- FT-Transformer  
  https://openreview.net/pdf?id=i_Q1yrOegLY
- TabTransformer  
  https://arxiv.org/pdf/2012.06678
- SAINT  
  https://arxiv.org/abs/2106.01342
- On Embeddings for Numerical Features in Tabular Deep Learning  
  https://arxiv.org/abs/2203.05556
- Representation Learning for Tabular Data survey  
  https://arxiv.org/html/2504.16109v1

Trọng tâm:

- numerical feature semantics nằm ở:
  - slot position
  - per-feature tokenizer/projection
  - column/feature embedding

## 4.2. Tabular SSL / masked feature learning

Nguồn chính:

- MET  
  https://table-representation-learning.github.io/assets/papers/met_masked_encoding_for_tabula.pdf
- TabRet  
  https://openreview.net/pdf?id=qnRlh8BdMY

Trọng tâm:

- coordinate/feature identity có được mô hình hóa riêng không
- unseen columns được xử lý ra sao

## 4.3. Trajectory / vectorized structured inputs

Nguồn chính:

- VectorNet  
  https://arxiv.org/abs/2005.04259

Trọng tâm:

- trajectory/vectorized states thường dựa vào fixed vector schema hay có field identity machinery rõ hơn

## 4.4. Object-centric multimodal / driving MLLM

Nguồn chính:

- TrackingMeetsLMM  
  audit local: [2026-07-09-trackingmeetslmm-trajectory-encoder-audit.md](D:/NCKH_VLM/finetune-InternVL2/docs/research/2026-07-09-trackingmeetslmm-trajectory-encoder-audit.md)
- TOKEN  
  https://arxiv.org/html/2407.00959v1
- DriveMLM  
  https://arxiv.org/html/2312.09245v2

Trọng tâm:

- khi tiến sang reasoning/planning, raw numeric semantics có còn để ngầm hoàn toàn không

## 4.5. Time-series / sensor-to-LLM

Nguồn chính:

- SensorLLM  
  https://arxiv.org/abs/2410.10624
- Towards Time-Series Reasoning with LLMs  
  https://arxiv.org/html/2409.11376v1
- OpenTSLM  
  https://arxiv.org/abs/2510.02410
- TimeSense  
  https://arxiv.org/abs/2511.06344

Trọng tâm:

- channel identity của signals có được coi là tự đủ hay không
- khi sang LLM, có cần markers/alignment/textual scaffolding không

## 4.6. LLM for tables / verbalization

Nguồn chính:

- TabLLM  
  https://proceedings.mlr.press/v206/hegselmann23a/hegselmann23a.pdf
- Towards Better Serialization of Tabular Data for Few-shot Classification  
  https://arxiv.org/html/2312.12464v1

Trọng tâm:

- khi muốn LLM hiểu field semantics tốt hơn, literature có đẩy thẳng field names sang text không

---

## 5. Bảng đối chiếu paper/repo

| Paper/repo | Domain | Loại input số | Field identity kiểu nào | Có category/context hỗ trợ không | Có auxiliary objective không | Paper có nói trực tiếp về feature/field semantics không | Mức độ liên quan | Nhận xét ngắn |
|---|---|---|---|---|---|---|---|---|
| FT-Transformer | Tabular DL | numerical + categorical features | B | Có categorical features | Không phải trọng tâm | Gián tiếp | Trực tiếp | Feature Tokenizer dùng hàm riêng `f_j` và bias riêng `b_j` cho từng feature; không phải số trần thuần |
| TabTransformer | Tabular DL | categorical + continuous features | C cho categorical, A cho continuous trong bản gốc | Có | Có MLM/RTD pretraining | Có khá trực tiếp | Trực tiếp | Có `column embedding` với unique identifier cho từng cột categorical; continuous features thì concat raw vào MLP ở bản gốc |
| SAINT | Tabular DL | continuous + categorical features | B | Có | Có contrastive pretraining | Gián tiếp | Gần | Mỗi feature được chiếu độc lập vào dense space; field identity không verbalize nhưng không phải để tự đoán hoàn toàn |
| On Embeddings for Numerical Features in Tabular Deep Learning | Tabular DL | scalar numerical features | B | Không bắt buộc | Không | Không trực tiếp | Trực tiếp | Numerical features được map qua embedding modules; paper mạnh về value embedding, ít bàn semantic naming |
| MET | Tabular SSL | tabular coordinates/features | A/B | Phụ thuộc dataset | Có masked reconstruction | Có | Gần | Paper nói thẳng feature/coordinate structure của tabular không cố định và khó nhận ra a priori |
| TabRet | Tabular transfer learning | per-column tabular values | B | Không cần column descriptions | Có masked autoencoding + retokenizing | Có | Gần | Định nghĩa tokenizer `t_c` cho từng cột và xử lý cả unseen columns bằng retokenizing |
| VectorNet | Trajectory prediction | vectorized agent/map states | A/B | Có agent/map roles ở pipeline | Có masked auxiliary recovery | Không trực tiếp | Gần | Dựa rất mạnh vào vector schema cố định; semantic của từng chiều chủ yếu nằm trong structured representation design |
| TrackingMeetsLMM | Driving MLLM | trajectory tensors | A/B | Có image context; tách traj/ego | Có synthetic QA/pretraining | Gián tiếp | Gần | Không verbalize raw trajectory fields; field semantics chủ yếu nằm trong tensor contract và encoder riêng |
| TOKEN | Driving MM-LLM | object-centric latent tokens | C | Có object-centric semantics rõ | Không phải trọng tâm | Gián tiếp | Gần | Xu hướng chuyển từ raw/unstructured tokens sang semantically informed object tokens |
| DriveMLM | Driving LLM | planning states | C/D | Có sensor, rules, instructions | Có data engine + alignment | Gián tiếp | Gần | Chủ động chuẩn hóa decision states sang dạng dễ cho LLM xử lý, không để raw state semantics ngầm hoàn toàn |
| SensorLLM | Sensor-to-LLM | motion sensor time series | C/D | Có trend descriptions | Có sensor-language alignment | Có trực tiếp | Trực tiếp | Thêm special tokens đánh dấu channel boundaries để LLM bắt channel-specific features |
| Towards Time-Series Reasoning with LLMs | Time-series MLLM | raw time series + encoder | B/C | Có textual context và CoT tasks | Có encoder training + CoT fine-tuning | Có | Gần | Cho thấy perception bottleneck và cần encoder riêng để latent representation phản ánh slope/frequency |
| OpenTSLM | Time-series language models | native time series modality | B/C | Có text-time-series integration | Có multimodal training | Gián tiếp | Gần | Time series được coi là native modality, không chỉ là số trần đưa thẳng vào LM head |
| TimeSense | Time-series + LLM | time series | C | Có textual reasoning + temporal grounding | Có temporal sense module | Có | Gần | Nhấn mạnh cần giữ temporal sense để tránh bias theo textual cues |
| TabLLM | LLM for tables | serialized tabular rows | D | Có prompt/task text | Không phải trọng tâm | Có trực tiếp | Trực tiếp | Field names và values được serialize thẳng thành natural language string |

---

## 6. Những gì literature hỗ trợ trực tiếp

## 6.1. Nhiều paper không dùng text field names, nhưng cũng không để model “tự đoán từ số trần”

`Paper-evidenced`

Đây là điểm quan trọng nhất của báo cáo.

### FT-Transformer

Paper mô tả `Feature Tokenizer` như sau:

- mỗi feature `x_j` được map thành một embedding riêng `T_j`
- với hàm riêng `f_j`
- và bias riêng `b_j`

Nói cách khác:

- field identity nằm trong `j`
- và trong tham số riêng của feature đó

Đây là `Mức B`, không phải `Mức A` thuần.

### TabTransformer

Paper viết rất rõ:

- có `column embedding layer`
- mỗi categorical feature có embedding lookup table riêng
- còn có `column-specific and unique identifier`

Tức là field identity thậm chí đã đi tới `Mức C` cho categorical columns.

Paper còn nói:

- trong tabular data không có ordering của features như NLP
- nên họ **không dùng positional encoding kiểu NLP**
- mà dùng `unique identifier` cho cột

Đây là bằng chứng rất mạnh rằng ít nhất ở tabular DL, literature **không mặc định tin vào raw slot alone** khi đã có cách encode field identity tốt hơn.

### TabRet

TabRet còn đi xa hơn:

- định nghĩa tokenizer `t_c` cho từng cột `c`
- khi gặp unseen columns, thêm tokenizer mới và `retokenizing`

Điểm này cực mạnh vì nó cho thấy:

- nhiều công trình thật ra xem `field identity` là một phần kiến trúc cốt lõi
- chứ không phải thứ có thể bỏ qua

## 6.2. Với LLM/time-series reasoning, field/channel identity thường được làm tường minh hơn

`Paper-evidenced`

### SensorLLM

Paper nói trực tiếp:

- LLM bị hạn chế bởi `lack of semantic context`
- và `challenges in processing numerical inputs`
- họ thêm `special tokens` để đánh dấu `channel boundaries`

Tức là khi sang LLM:

- chỉ raw values là chưa đủ tin cậy
- channel identity được đẩy lên ít nhất `Mức C`

### TabLLM

TabLLM thì đi theo cực còn lại:

- serialize `feature names` và `values` thành text tự nhiên

Đây là `Mức D`, và nó cho thấy một hướng rất rõ trong literature:

- nếu muốn dùng language prior mạnh, nhiều công trình chọn verbalize field identity thay vì để ngầm hoàn toàn.

## 6.3. Các paper gần reasoning hơn thường thêm machinery để bridge semantic gap

`Paper-evidenced`

`Toward Time-Series Reasoning with LLMs`, `OpenTSLM`, `TimeSense`, `DriveMLM`, `TOKEN` đều cho thấy một pattern khá nhất quán:

- khi mục tiêu là reasoning/planning/language output
- field/channel/object semantics thường được làm rõ hơn qua:
  - encoder riêng
  - semantically informed tokens
  - channel markers
  - alignment stage
  - decision-state standardization

---

## 7. Những gì chỉ hỗ trợ gián tiếp

## 7.1. `Slot-only` thuần có thể tồn tại, nhưng không phải là pattern mạnh nhất trong literature hiện đại

`Research-supported inference`

Với trajectory/vectorized input kiểu `VectorNet`, rất có thể một phần semantics nằm ở:

- fixed schema của vector
- vai trò đã được quy ước trước trong pipeline

Điều này gần `Mức A`, nhưng trên thực tế vẫn thường có:

- agent roles
- map roles
- structured grouping

nên nó ít khi là “số trần hoàn toàn vô danh”.

## 7.2. Với repo hiện tại, baseline đơn giản nhất có thể là `slot cố định`, nhưng literature mạnh hơn nghiêng về `slot + field-specific machinery`

`Research-supported inference`

Nếu chỉ muốn một baseline rất đơn giản cho repo:

- giữ thứ tự chiều cố định

là đủ để chạy.

Nhưng nếu hỏi phương án nào được literature ủng hộ mạnh hơn, thì câu trả lời nhiều khả năng là:

- `slot + field-specific parameters`

chứ không phải `slot-only` thuần.

---

## 8. Những khoảng trống còn thiếu

## 8.1. Thiếu paper nói trực diện “slot-only là đủ tốt nhất”

`Open gap`

Tôi chưa thấy paper mạnh nào chủ động khẳng định:

- cứ giữ thứ tự chiều cố định là đủ
- không cần field-specific machinery nào đáng kể

Ngược lại, nhiều paper thành công hơn lại đang thêm tokenizer/column embedding/channel markers.

## 8.2. Thiếu precedent trực tiếp cho đúng bài toán `top-6 object, 1 frame, alter`

`Open gap`

Không có paper nào mình tìm được nói thẳng:

- với branch object numeric nhỏ như repo này
- phương án tốt nhất là A/B/C/D

Nên đoạn cuối vẫn phải là khuyến nghị engineering có căn cứ, chứ không phải fact tuyệt đối.

---

## 9. Kết luận khuyến nghị cho repo hiện tại

## Kết luận cuối

**Kết luận B**

- literature cho thấy field semantics **thường đã được encode ngầm trong kiến trúc**
- nhưng khi tiến gần hơn tới reasoning/LLM thì `explicit field identity` càng đáng cân nhắc

## 9.1. Trả lời trực tiếp 7 câu hỏi thực dụng

### 1. Có paper nào thật sự để numeric branch học hoàn toàn từ `slot position` mà không có field-specific machinery nào đáng kể không?

`Không thấy đây là pattern mạnh nhất trong các paper trọng tâm đã đọc.`

Có những bài gần kiểu đó ở vectorized trajectories, nhưng ngay cả chúng cũng thường dựa vào structured schema và pipeline roles, chứ không phải số vô danh hoàn toàn.

### 2. Trong tabular/trajectory papers, “không có field name text” thường có nghĩa là gì?

`Thường là có projection/embedding/tokenizer riêng cho từng field hoặc cột, chứ không phải chỉ có slot position thuần.`

Đây là kết luận rất mạnh từ:

- FT-Transformer
- TabTransformer
- TabRet

### 3. Có precedent mạnh cho `feature-specific projection` hoặc `field embedding` không?

`Có.`

Đây chính là một trong những precedent mạnh nhất của tabular DL hiện đại.

### 4. Trong các paper gần MLLM/reasoning hơn, field semantics có thường phải được làm tường minh hơn không?

`Có xu hướng như vậy.`

Ví dụ:

- SensorLLM thêm channel markers
- TabLLM verbalize field names
- TOKEN/DriveMLM chuẩn hóa scene/decision semantics rõ hơn cho LLM

### 5. Có paper nào nói rõ rằng lack of semantic context / field identity là một vấn đề khi nối numeric data với LLM không?

`Có.`

SensorLLM là ví dụ rất trực tiếp.

### 6. Với branch của repo hiện tại, phương án nào được literature ủng hộ mạnh nhất?

Khuyến nghị theo mức:

1. `baseline đơn giản nhất`: `slot cố định`
2. `baseline mạnh hơn nhưng vẫn gọn`: `slot + field-specific parameters`
3. `hướng mở rộng v2`: `explicit field embedding`
4. `text verbalization`: chỉ nên cân nhắc nếu muốn đi hẳn theo hướng language-facing branch

### 7. Nếu chưa đủ evidence để chốt một phương án duy nhất, phương án nào nên là baseline/v2?

- `Baseline đơn giản nhất`: A
- `Baseline mạnh hơn nhưng vẫn gọn`: B
- `Hướng mở rộng v2`: C

## 9.2. Một câu chốt ngắn

Nếu một paper “không có field name text”, điều đó **không có nghĩa** là họ để model tự đoán hoàn toàn từ raw numbers.  
Rất nhiều paper thực ra đã encode field identity ngầm qua tokenizer/projection/column embedding riêng.

---

## 10. Nguồn chính đã dùng

- FT-Transformer: https://openreview.net/pdf?id=i_Q1yrOegLY
- TabTransformer: https://arxiv.org/pdf/2012.06678
- SAINT: https://arxiv.org/abs/2106.01342
- On Embeddings for Numerical Features in Tabular Deep Learning: https://arxiv.org/abs/2203.05556
- MET: https://table-representation-learning.github.io/assets/papers/met_masked_encoding_for_tabula.pdf
- TabRet: https://openreview.net/pdf?id=qnRlh8BdMY
- VectorNet: https://arxiv.org/abs/2005.04259
- TrackingMeetsLMM: https://github.com/mbzuai-oryx/TrackingMeetsLMM
- TOKEN: https://arxiv.org/html/2407.00959v1
- DriveMLM: https://arxiv.org/html/2312.09245v2
- SensorLLM: https://arxiv.org/abs/2410.10624
- Towards Time-Series Reasoning with LLMs: https://arxiv.org/html/2409.11376v1
- OpenTSLM: https://arxiv.org/abs/2510.02410
- TimeSense: https://arxiv.org/abs/2511.06344
- TabLLM: https://proceedings.mlr.press/v206/hegselmann23a/hegselmann23a.pdf
- Towards Better Serialization of Tabular Data for Few-shot Classification: https://arxiv.org/html/2312.12464v1
