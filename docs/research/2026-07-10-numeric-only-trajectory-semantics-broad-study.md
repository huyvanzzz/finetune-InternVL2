# Nghiên cứu mở rộng: `numeric-only` trajectory encoder có đủ “ý nghĩa” cho bài toán hiện tại hay không

## 1. Câu hỏi cần verify

Mục tiêu của báo cáo này là trả lời thật rõ câu hỏi sau:

- nếu trajectory/object branch chỉ nhận `feature số`
- không có `category branch`
- không verbalize feature thành text
- không nói cho model biết “cột này là gì” bằng ngôn ngữ tự nhiên

thì model có thật sự học được thông tin cần thiết cho downstream reasoning hay không, nhất là khi đầu ra cuối cùng vẫn là text kiểu `alter`.

Điểm phải tách bạch ngay từ đầu:

1. Học được `prediction pattern` không đồng nghĩa với học được `language-aligned reasoning`.
2. Paper chứng minh numeric-only làm tốt ở trajectory prediction chưa đủ để kết luận nó đủ mạnh cho text generation/reasoning.
3. Nếu literature chỉ hỗ trợ tới representation hoặc decision-level semantics, báo cáo phải nói thẳng là chưa đủ bằng chứng cho language alignment.

---

## 2. `Numeric-only` trong repo hiện tại thực sự nghĩa là gì

`Data-evidenced`

Trong context repo hiện tại, `numeric-only` đang được hiểu là:

- input là `top-6 object`
- chỉ dùng `1 frame`
- không có `ego`
- không có `label/category`
- không có text serialization

Mỗi object về mặt ý tưởng chỉ giữ các tín hiệu số như:

- hình học từ bbox
  - `cx`
  - `cy`
  - `w`
  - `h`
  - `area`
- khoảng cách / ưu tiên
  - `distance_norm`
  - `direction_weight`
- động học
  - `movement_angle`
  - `speed_percent`

Tức là mô hình sẽ nhìn thấy:

- một số vector số cho từng object
- thứ tự slot cố định
- nhưng không được nói bằng text rằng object nào là người, xe, biển báo, vật cản, v.v.

---

## 3. Bốn mức “model hiểu feature số”

Để tránh kết luận quá tay, báo cáo này dùng 4 mức sau:

### Mức A: `slot-level statistical learning`

Model chỉ học rằng:

- cột thứ `k` có tương quan nào đó với target
- ví dụ một chiều tăng lên thì xác suất một nhãn nào đó tăng

Đây là mức thấp nhất. Nó chưa có nghĩa là model “hiểu” object hay scene.

### Mức B: `structured relational semantics`

Model học được rằng:

- nhiều cột số cùng mô tả một thực thể
- hoặc nhiều object vectors có quan hệ với nhau
- ví dụ object gần hơn, to hơn, hướng tiến vào hơn thì có thể quan trọng hơn

Đây là mức “hiểu có cấu trúc”, nhưng vẫn chưa phải reasoning bằng ngôn ngữ.

### Mức C: `decision-aligned semantics`

Representation học được đã đủ để hỗ trợ:

- planning
- navigation
- action recommendation
- trajectory/behavior decision

Đây là mức cao hơn prediction thuần, nhưng đầu ra vẫn chưa nhất thiết là text.

### Mức D: `language-aligned semantics`

Representation số được nối thành công với kiểu reasoning cần để:

- giải thích
- trả lời bằng ngôn ngữ
- sinh guidance đúng nghĩa, an toàn, dễ hiểu

Đây là mức khó nhất, và cũng là mức liên quan trực tiếp nhất tới task `alter`.

Điểm nguyên tắc:

- bằng chứng mức A không được dùng để kết luận mức D
- bằng chứng mức B/C cũng chưa tự động đủ để kết luận mức D

---

## 4. Các cụm research đã quét

## 4.1. Tabular / mixed-feature modeling

Nguồn chính:

- FT-Transformer  
  https://openreview.net/pdf?id=i_Q1yrOegLY
- SAINT  
  https://arxiv.org/abs/2106.01342
- On Embeddings for Numerical Features in Tabular Deep Learning  
  https://arxiv.org/abs/2203.05556
- Representation Learning for Tabular Data: A Comprehensive Survey  
  https://arxiv.org/html/2504.16109v1

Trọng tâm:

- numerical features được chiếu thành embeddings thế nào
- mô hình học cấu trúc giữa các cột số ra sao
- họ có nói trực tiếp gì về “ý nghĩa” của feature slot hay không

## 4.2. Trajectory / motion forecasting

Nguồn chính:

- VectorNet  
  https://arxiv.org/abs/2005.04259
- Trajectory Prediction Meets Large Language Models: A Survey  
  https://arxiv.org/html/2506.03408v1

Trọng tâm:

- numeric/vector input có đủ cho prediction không
- “đủ” ở đây là đủ cho prediction hay đủ cho reasoning

## 4.3. Object-centric multimodal / autonomous driving MLLM

Nguồn chính:

- TrackingMeetsLMM  
  audit local: [2026-07-09-trackingmeetslmm-trajectory-encoder-audit.md](D:/NCKH_VLM/finetune-InternVL2/docs/research/2026-07-09-trackingmeetslmm-trajectory-encoder-audit.md)
- TOKEN: Tokenize the World into Object-level Knowledge  
  https://arxiv.org/html/2407.00959v1
- DriveMLM  
  https://arxiv.org/html/2312.09245v2

Trọng tâm:

- khi đi sang MLLM, numeric/state features có đứng một mình không
- object-centric semantics có thường được đưa vào rõ ràng hơn không

## 4.4. Sensor-to-language / time-series-to-LLM

Nguồn chính:

- SensorLLM  
  https://aclanthology.org/2025.emnlp-main.19.pdf
- Towards Time-Series Reasoning with LLMs  
  https://arxiv.org/html/2409.11376v1
- OpenTSLM  
  https://arxiv.org/abs/2510.02410
- TimeSense  
  https://arxiv.org/abs/2511.06344

Trọng tâm:

- numeric/time-series có đủ semantic cho language reasoning không
- nếu chưa đủ thì literature thêm gì để bù

## 4.5. Structured representation learning / auxiliary objectives

Nguồn chính:

- MET: Masked Encoding for Tabular Data  
  https://table-representation-learning.github.io/assets/papers/met_masked_encoding_for_tabula.pdf

Trọng tâm:

- nếu numeric-only input chưa “lộ” đủ cấu trúc, literature có dùng masked reconstruction / auxiliary learning để ép model học quan hệ giữa feature slots không

## 4.6. Tabular foundation / LLM for tables

Nguồn chính:

- TabLLM  
  https://proceedings.mlr.press/v206/hegselmann23a/hegselmann23a.pdf

Trọng tâm:

- khi muốn tận dụng language prior, người ta có thường chuyển feature thành text không
- điều đó hàm ý gì về giới hạn của numeric-only khi đi vào LM reasoning

---

## 5. Bảng đối chiếu paper/repo

| Paper/repo | Domain | Loại numeric input | Numeric-only hay mixed | Có category/context hỗ trợ không | Có auxiliary objective không | Có nói trực tiếp về feature semantics không | Mức “hiểu” hỗ trợ | Mức độ liên quan | Nhận xét ngắn |
|---|---|---|---|---|---|---|---|---|---|
| FT-Transformer | Tabular DL | feature-level numerical values | Mixed | Có categorical features | Không phải trọng tâm | Không trực tiếp | A/B | Gần | Chứng minh numerical features có thể được embed và trộn tốt, nhưng không bàn language reasoning |
| SAINT | Tabular DL | continuous + categorical features | Mixed | Có | Có contrastive pretraining | Không trực tiếp | A/B | Gần | Hỗ trợ mạnh cho feature tokenization và representation learning, chưa bàn language alignment |
| On Embeddings for Numerical Features in Tabular Deep Learning | Tabular DL | scalar numerical features | Numeric-only hoặc mixed | Không bắt buộc | Không phải trọng tâm | Không trực tiếp | A | Trực tiếp | Nói rõ numerical features có thể map thành embeddings hữu ích; đây là bằng chứng mạnh cho slot-level learning |
| MET | Tabular SSL | tabular coordinates/features | Mixed hoặc numeric-heavy | Phụ thuộc dataset | Có masked reconstruction | Gián tiếp | A/B | Gần | Quan trọng vì paper nêu rõ tabular data thiếu semantic structure cố định như image/text, nên cần học quan hệ giữa coordinates |
| VectorNet | Trajectory prediction | vectorized agent/map states | Numeric-heavy | Có context map/agent roles trong pipeline | Có masked auxiliary recovery | Không trực tiếp | B/C | Gần | Rất mạnh cho prediction từ vector số, nhưng không nhắm language reasoning |
| Trajectory Prediction Meets LLMs Survey | Survey | nhiều dạng | Mixed | Thường có language/context | Không áp dụng | Có ở mức survey | C/D | Gián tiếp | Cho thấy xu hướng field đang dịch từ numeric-only sang semantically enriched systems |
| TrackingMeetsLMM | Driving MLLM | trajectory tensors | Numeric trajectory + image | Có visual context; riêng branch traj/ego | Có pretraining data synthesis | Không trực tiếp | C | Gần | Không đưa traj vào text, nhưng cũng không để numeric branch đứng một mình |
| TOKEN | Autonomous driving MM-LLM | object-level latent tokens | Mixed | Có object-centric semantics rõ | Không phải trọng tâm chính | Gián tiếp | C/D | Gần | Nhấn mạnh semantically informed object tokens dễ cho LLM diễn giải hơn dense/unstructured tokens |
| DriveMLM | Autonomous driving LLM | planning / decision states | Mixed | Có sensor, rules, user instructions | Có data engine + aligned decision states | Gián tiếp | C/D | Gián tiếp | Không đi theo numeric-only; chủ động biến decision states thành dạng dễ cho LLM xử lý |
| SensorLLM | Sensor-to-LLM | motion sensor time series | Numeric-only raw signals nhưng có alignment text | Có trend descriptions và channel markers | Có sensor-language alignment | Có, khá trực tiếp | C/D | Trực tiếp | Paper nêu rõ LLM bị hạn chế vì lack of semantic context và numerical-input challenges |
| Towards Time-Series Reasoning with LLMs | Time-series MLLM | raw time series + encoder | Numeric modality + text reasoning | Có CoT tasks và textual context | Có encoder pretraining + CoT fine-tuning | Có | B/C/D | Trực tiếp | Chỉ ra perception bottleneck và cho rằng encoder riêng + reasoning supervision là cần thiết |
| OpenTSLM | Time-series language models | native time series modality | Mixed with text | Có text-time-series integration | Có multimodal training | Gián tiếp | C/D | Gần | So sánh với treat-as-text baselines và cho thấy cần time-series như native modality để reasoning tốt hơn |
| TimeSense | Time-series + LLM | time series | Mixed | Có textual reasoning nhưng thêm temporal sense module | Có reconstruction-like temporal grounding | Có, khá trực tiếp | C/D | Gần | Nhấn mạnh chỉ dựa text supervision dễ bias về textual cues và mâu thuẫn với temporal context |
| TabLLM | Tabular + LLM | serialized rows | Textualized numeric/tabular data | Có natural-language serialization | Không phải trọng tâm chính | Gián tiếp | D | Gần | Quan trọng vì cho thấy khi muốn tận dụng LLM reasoning, nhiều công trình chọn verbalize dữ liệu thay vì để nguyên vector số |

---

## 6. Những gì literature hỗ trợ trực tiếp

## 6.1. Numeric-only đủ mạnh cho representation và prediction

`Paper-evidenced`

Các paper như:

- `On Embeddings for Numerical Features in Tabular Deep Learning`
- `FT-Transformer`
- `SAINT`
- `VectorNet`

ủng hộ khá mạnh cho kết luận rằng:

- feature số có thể được chiếu thành embeddings tốt
- mô hình có thể học quan hệ giữa các feature slots
- numeric/vector input hoàn toàn có thể đủ cho prediction hoặc classification

Đây là bằng chứng mạnh cho:

- `Mức A`
- một phần `Mức B`
- và trong một số bài trajectory/planning, lên tới `Mức C`

## 6.2. Tabular / structured numeric data không có semantic structure “tự nhiên” như text/image

`Paper-evidenced`

`MET` nói rất rõ rằng tabular SSL khó hơn image/text vì:

- mỗi dataset có cấu trúc feature/coordinate khác nhau
- semantic structure của coordinates không cố định và khó nhận ra a priori

Điều này cực kỳ quan trọng cho câu hỏi hiện tại:

- model **có thể** học từ feature slots
- nhưng numeric-only input không hề có semantic prior tự nhiên mạnh như text token hay image patch

Tức là:

- numeric-only không vô nghĩa
- nhưng cũng không nên giả định model “tự hiểu” các số như hiểu ngôn ngữ

## 6.3. Khi mục tiêu là reasoning bằng ngôn ngữ, literature mới thường thêm alignment hoặc semantic scaffolding

`Paper-evidenced`

Các nguồn mạnh nhất ở điểm này là:

- `SensorLLM`
  - nêu trực tiếp các vấn đề: lack of semantic context, challenges in processing numerical inputs
  - giải pháp: `Sensor-Language Alignment`
- `Towards Time-Series Reasoning with LLMs`
  - nêu `perception bottleneck`
  - giải pháp: encoder riêng + CoT reasoning tasks
- `TimeSense`
  - cảnh báo text labels supervision có thể bias model vào textual cues và bỏ qua temporal features
  - giải pháp: temporal grounding / preserved temporal sense

Điểm chung rất rõ:

- khi bài toán đi sang text reasoning
- literature không chỉ “ném vector số vào rồi mong LLM tự hiểu”
- họ thường thêm một trong các thứ sau:
  - encoder riêng
  - alignment stage
  - reconstruction / grounding objective
  - chain-of-thought hoặc task-aware tuning

## 6.4. Trong driving/object-centric MLLM, xu hướng là semantically informed tokens

`Paper-evidenced`

`TOKEN` nhấn mạnh object-level semantically informed representation dễ cho LLM diễn giải hơn dense/unstructured tokens.  
`DriveMLM` thì chủ động đưa planning states về dạng dễ cho LLM xử lý thay vì giữ raw low-level state 그대로.

Điều này không chứng minh numeric-only là sai, nhưng nó cho thấy:

- khi mục tiêu là reasoning/planning có ngôn ngữ
- field đang nghiêng về structured semantics rõ ràng hơn, không chỉ numeric vectors trần

---

## 7. Những gì chỉ hỗ trợ gián tiếp

## 7.1. Numeric-only có thể học được `structured semantics`

`Research-supported inference`

Từ `VectorNet`, `FT-Transformer`, `SAINT`, `MET`, có thể suy ra khá hợp lý rằng:

- nếu feature slots đủ ổn định
- normalization đủ tốt
- encoder đủ mạnh

thì numeric-only branch có thể học:

- object salience
- proximity-risk patterns
- motion relevance

Tức là có khả năng chạm tới `Mức B`, thậm chí hỗ trợ `Mức C` cho một số downstream decisions.

Nhưng đây vẫn là suy luận hợp lý, không phải paper nào cũng phát biểu trực diện theo đúng câu hỏi của repo này.

## 7.2. Numeric-only có thể là baseline hợp lệ cho repo hiện tại

`Research-supported inference`

Với context:

- top-6 object
- 1 frame
- bbox/distance/direction/motion

numeric-only hoàn toàn xứng đáng làm:

- một baseline hợp lệ
- một ablation quan trọng

để kiểm tra xem chỉ geometry/motion đã mang được bao nhiêu signal.

Nhưng từ literature hiện tại, tôi **không thấy đủ cơ sở để xem numeric-only là phương án chính mặc định** cho mục tiêu cuối là text guidance.

---

## 8. Những khoảng trống còn thiếu

## 8.1. Thiếu bằng chứng trực tiếp cho `Mức D`

`Open gap`

Tôi chưa thấy paper nào nói trực tiếp rằng:

- chỉ với structured numeric-only object input
- không category
- không verbalization
- không alignment stage

thì model vẫn đủ mạnh để làm `language-aligned reasoning` tốt một cách ổn định.

Đây là khoảng trống lớn nhất.

## 8.2. Thiếu bằng chứng trực tiếp cho bài toán dạng `alter`

`Open gap`

Task của repo bạn không chỉ là:

- dự đoán trajectory
- hay phân loại activity

mà là sinh guidance ngắn, an toàn, định hướng cho người khiếm thị.

Tôi không thấy literature nào chứng minh trực tiếp numeric-only object branch là đủ cho loại output như vậy nếu đứng một mình.

## 8.3. Thiếu bằng chứng rằng numeric-only sẽ generalize tốt ở long-tail safety cases

`Open gap`

Ngay cả khi numeric-only học được pattern ở các case thường gặp, literature hiện có chưa cho tôi đủ lý do để tin rằng:

- nó sẽ robust ở các trường hợp hiếm
- nó sẽ giữ được alignment đúng giữa risk pattern và wording guidance

đặc biệt khi không có category identity để phân biệt bản chất object.

---

## 9. Kết luận khuyến nghị cho repo hiện tại

## Kết luận cuối

**Kết luận C**:

- literature nghiêng khá rõ về việc `numeric-only` **không nên** là phương án chính đứng một mình nếu mục tiêu cuối là reasoning/text generation
- nó nên đi kèm:
  - semantic/context branch
  - hoặc alignment / auxiliary supervision

## 9.1. Trả lời trực tiếp 7 câu hỏi thực dụng

### 1. Có paper nào nói trực tiếp rằng numeric-only input vẫn đủ để model học “ý nghĩa” nhờ feature-slot consistency không?

`Chưa có theo đúng cách nói đó.`  
`Paper-evidenced` gần nhất là nhóm tabular/SSL như `MET`, `FT-Transformer`, `On Embeddings...`, nhưng họ chủ yếu chứng minh representation/prediction, không diễn đạt bằng ngôn ngữ “LLM hiểu ý nghĩa cột số”.

### 2. Có paper nào nói numeric-only là đủ cho downstream language reasoning không?

`Không thấy bằng chứng trực tiếp mạnh.`  
Ngược lại, các paper gần language reasoning hơn thường thêm alignment, encoder riêng, semantically informed tokens hoặc supervised reasoning objectives.

### 3. Trong các paper trajectory/driving, numeric branch có thường đứng một mình không?

`Không phải xu hướng chính khi đi sang MLLM reasoning.`  
Khi bài toán tiến sang planning/explanation/language, nhiều hệ thống thêm:

- object-centric semantics
- route/context
- decision-state alignment
- sensor-language alignment

### 4. Literature có dấu hiệu nào cho thấy numeric-only dễ bị yếu ở đâu?

`Có.`  
Các điểm yếu lặp lại nhiều nhất là:

- thiếu semantic context
- thiếu object identity
- perception bottleneck
- khó align với textual reasoning

### 5. Literature gợi ý giải pháp bù nào mạnh nhất?

Các hướng được ủng hộ nhiều nhất:

- category/object semantics
- alignment stage với text
- auxiliary pretraining
- masking / reconstruction
- CoT hoặc task-aware tuning cho reasoning

### 6. Với repo hiện tại, numeric-only nên được xem là gì?

`Khuyến nghị: baseline hợp lệ và ablation quan trọng.`  
Không nên coi là phương án chính mặc định ngay từ đầu.

### 7. Nếu chưa đủ evidence cho numeric-only làm phương án chính, nó thiếu ở tầng nào?

Thiếu mạnh nhất ở:

- `language alignment`

và thiếu vừa ở:

- `object semantics`

Trong khi phần:

- `representation / prediction`

thì có bằng chứng tốt hơn nhiều.

## 9.2. Khuyến nghị thực dụng cho hướng thiết kế tiếp theo

`Research-supported inference`

Nếu sau này đi tới pha code, thứ tự an toàn nhất là:

1. làm `numeric-only` như baseline
2. so với `numeric + category`
3. nếu muốn giữ numeric-only lâu hơn, nên cân nhắc thêm:
   - masked / reconstruction objective
   - alignment objective
   - hoặc bridge tốt hơn sang language side

Một câu chốt ngắn:

`Numeric-only` có cơ sở tốt cho học representation và prediction.  
Nhưng với bài toán cuối là `alter`, literature hiện tại nghiêng về việc nó **không nên đứng một mình** nếu muốn kỳ vọng reasoning bằng ngôn ngữ đủ mạnh và ổn định.

---

## 10. Nguồn chính đã dùng

- FT-Transformer: https://openreview.net/pdf?id=i_Q1yrOegLY
- SAINT: https://arxiv.org/abs/2106.01342
- On Embeddings for Numerical Features in Tabular Deep Learning: https://arxiv.org/abs/2203.05556
- MET: https://table-representation-learning.github.io/assets/papers/met_masked_encoding_for_tabula.pdf
- Representation Learning for Tabular Data: A Comprehensive Survey: https://arxiv.org/html/2504.16109v1
- VectorNet: https://arxiv.org/abs/2005.04259
- Trajectory Prediction Meets Large Language Models: A Survey: https://arxiv.org/html/2506.03408v1
- SensorLLM: https://aclanthology.org/2025.emnlp-main.19.pdf
- Towards Time-Series Reasoning with LLMs: https://arxiv.org/html/2409.11376v1
- OpenTSLM: https://arxiv.org/abs/2510.02410
- TimeSense: https://arxiv.org/abs/2511.06344
- TOKEN: https://arxiv.org/html/2407.00959v1
- DriveMLM: https://arxiv.org/html/2312.09245v2
