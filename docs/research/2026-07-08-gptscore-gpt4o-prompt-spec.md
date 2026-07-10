# Thiết Kế Prompt Judge `GPTScore` v1 Cho `alter` Với `GPT-4o`

## 1. Mục tiêu của tài liệu này

Tài liệu này chốt cách thiết kế **prompt judge** cho `GPTScore` v1 trên task `alter`, với các ràng buộc đã khóa:

- chỉ chấm cặp `ground_truth` và `generation`
- chỉ dùng `GPT-4o`
- chỉ dùng **một prompt, một JSON**
- đầu ra chính là:
  - `gate flags`
  - `criterion-level judgments`
  - `overall_score`
  - `brief rationale`

Tài liệu này **không** bàn sang:

- prompt train/infer của model
- code runner/API client
- calibration bằng human labels

Mục tiêu cuối là: sau tài liệu này, implementer có thể viết prompt thật và schema thật mà không phải tự quyết thêm các decision quan trọng.

## 2. Vì sao `GPT-4o` phù hợp cho v1

### 2.1 Phù hợp với đầu ra JSON chặt

`Research-supported inference`

Theo tài liệu chính thức của OpenAI về Structured Outputs, mô hình có thể bám đúng JSON Schema và việc này giúp giảm gánh nặng ép format bằng prompt mạnh tay.  
Nguồn:

- https://developers.openai.com/api/docs/guides/structured-outputs
- https://openai.com/index/introducing-structured-outputs-in-the-api/

Ý nghĩa cho bài toán này:

- ta nên dùng **schema để khóa format**
- prompt nên tập trung vào **logic chấm**, không nên lãng phí nhiều token để nhắc đi nhắc lại “hãy trả JSON hợp lệ”

### 2.2 Phù hợp với judge one-shot dạng rubric

`Research-supported inference`

Các hướng như G-Eval, SAJA, Rulers, AutoCalibrate đều cho thấy LLM judge hiệu quả hơn khi:

- có rubric rõ
- có dimension-level outputs
- có protocol chấm cố định

Các nguồn hỗ trợ:

- G-Eval: https://aclanthology.org/2023.emnlp-main.153/
- SAJA: https://aclanthology.org/2026.acl-industry.45.pdf
- Rulers: https://arxiv.org/html/2601.08654v2
- AutoCalibrate: https://aclanthology.org/2024.lrec-main.237.pdf

Ý nghĩa cho v1:

- không nên bắt đầu bằng prompt ngắn kiểu “chấm giúp tôi câu này”
- nên dùng prompt dạng **rubric cố định + protocol cố định**

### 2.3 Không nên tối ưu prompt theo kiểu “mẹo vặt”

`Research-supported inference`

Các nghiên cứu về LLM-as-a-judge nhấn mạnh rằng prompt design có thể gây:

- rubric execution drift
- prompt-format sensitivity
- score misalignment với human intent

Nguồn hỗ trợ:

- Rulers: https://arxiv.org/html/2601.08654v2
- AutoCalibrate: https://aclanthology.org/2024.lrec-main.237.pdf
- Calibrating LLM-Based Evaluator: scoring criteria rõ tốt hơn chỉ nêu output space

Kết luận:

- prompt v1 nên ưu tiên **ổn định, audit được, ít quyết định ngầm**
- không nên thiết kế theo hướng “càng thông minh càng tốt” mà nên thiết kế theo hướng “càng khó hiểu sai càng tốt”

## 3. Những failure modes prompt phải chống

### 3.1 Lexical-overlap bias

`Data-evidenced` + `Research-supported inference`

`alter` có lặp mẫu câu và target khá template hóa, nên judge rất dễ thưởng quá mức cho generation chỉ vì giống từ với GT.  
Prompt phải nói rõ:

- không phạt paraphrase đúng nghĩa
- không thưởng chỉ vì wording giống

Nguồn nền:

- [2026-07-08-gptscore-design-for-alter.md](D:/NCKH_VLM/finetune-InternVL2/docs/research/2026-07-08-gptscore-design-for-alter.md)
- Blind to the Human Touch: https://arxiv.org/abs/2602.07673

### 3.2 Rubric execution drift

`Research-supported inference`

Judge có thể “hiểu đúng rubric ở đầu prompt” nhưng khi chấm lại bỏ qua một số rule, nhất là:

- `N/A`
- gate
- distinction giữa `Safety` và `Hazard / Path-State`

Prompt phải ép thứ tự xử lý đủ rõ để giảm drift.

### 3.3 Over-inference khi GT thiếu tín hiệu

`Data-evidenced`

Không phải mẫu `alter` nào cũng có:

- direction anchor
- action demand
- hazard/path-state rõ

Prompt phải cấm judge:

- tự bịa tín hiệu GT không có
- tự nâng một criterion thành applicable chỉ vì generation nhắc tới nó

### 3.4 Hiểu sai `N/A` thành “miễn trách nhiệm”

`Design choice for v1`

Rule đã khóa:

- `Direction = N/A` không che hallucinated direction nguy hiểm
- `Action = N/A` không che hallucinated action nguy hiểm

Prompt phải nói cực rõ:

- `N/A` chỉ có nghĩa là **không chấm fidelity/usefulness trực tiếp cho criterion đó**
- nếu generation tự thêm hướng/hành động nguy hiểm, vẫn phải phạt qua `Safety` hoặc `gate`

### 3.5 Verbose rationale làm judge “tự kể chuyện”

`Research-supported inference`

Nếu để rationale quá dài, judge dễ:

- tự thêm giả định
- rationalize một điểm không chắc
- bịa evidence ngoài GT/generation

Prompt v1 nên giới hạn rationale:

- 1 câu ngắn cho mỗi criterion
- 1 đoạn ngắn cho `overall_rationale`

## 4. So sánh 3 kiến trúc prompt ứng viên

### 4.1 Minimal strict judge

Mô tả:

- prompt ngắn
- nêu rubric cực gọn
- ép JSON cứng

Ưu điểm:

- rẻ token
- ít nguy cơ dài dòng
- dễ format ổn định

Nhược điểm:

- judge dễ hiểu thiếu boundary như:
  - `Safety` vs `Hazard`
  - `N/A` vs hallucination nguy hiểm
  - `applicable` vs `not applicable`

Đánh giá:

- không nên là kiến trúc chính cho v1
- có thể giữ làm fallback nếu sau này prompt dài gây vấn đề chi phí

### 4.2 Structured rubric judge

Mô tả:

- prompt có đủ:
  - mục tiêu
  - rubric
  - gate
  - `N/A`
  - scale
  - schema

Ưu điểm:

- logic rõ hơn
- ít bỏ sót rule quan trọng

Nhược điểm:

- nếu chỉ liệt kê luật mà không ép thứ tự xử lý, judge vẫn có thể “đọc nhưng không thực thi đúng”

Đánh giá:

- tốt hơn minimal
- nhưng vẫn chưa phải lựa chọn tốt nhất

### 4.3 Procedure-first structured rubric judge

Mô tả:

- prompt không chỉ đưa rubric, mà còn ép judge xử lý theo thứ tự:
  1. xác định GT signals
  2. xác định criterion nào applicable
  3. check gate
  4. chấm từng criterion
  5. xuất criterion scores + gate flags dưới JSON cố định
  6. để code/pipeline tính `overall_score`

Ưu điểm:

- sát nhất với logic repo hiện tại
- giảm over-inference
- giảm nguy cơ `N/A` bị dùng sai
- dễ audit khi xem JSON

Nhược điểm:

- dài hơn
- cần schema tốt để tránh verbosity

Đánh giá:

- **đây là kiến trúc khuyến nghị chính cho v1**

### 4.4 Kết luận so sánh

`Design choice for v1`

| Kiến trúc | Mức phù hợp cho v1 | Lý do chính |
|---|---|---|
| Minimal strict judge | Thấp | quá dễ bỏ sót rule semantics |
| Structured rubric judge | Trung bình | logic rõ nhưng chưa khóa procedure |
| Procedure-first structured rubric judge | Cao | cân bằng tốt nhất giữa đúng logic, auditability, và one-shot JSON |

Fallback nên giữ:

- **Structured rubric judge**

nếu sau này procedure-first prompt quá dài hoặc làm latency/token cost tăng không đáng.

## 5. Kiến trúc prompt khuyến nghị

## 5.1 Hình dạng tổng thể

`Design choice for v1`

Prompt nên có 2 lớp:

1. **System instruction**
   - khóa vai trò judge
   - khóa nguyên tắc scoring
   - khóa anti-bias rules
2. **User payload**
   - truyền `ground_truth`
   - truyền `generation`
   - nhắc lại đúng task đang chấm

Không nên:

- nhồi toàn bộ rubric vào user message
- trộn lẫn role, task, data, schema trong cùng một khối dài

### 5.2 System instruction phải làm 4 việc

`Design choice for v1`

1. Khóa **evaluation objective**
   - so sánh generation với GT về đúng nghĩa và an toàn
   - không chấm wording similarity đơn thuần

2. Khóa **semantic boundaries**
   - `Safety` khác `Hazard / Path-State`
   - `Direction` cần `direction anchor`
   - `Action` cần `action demand`
   - `N/A` không che hallucination nguy hiểm

3. Khóa **procedure**
   - judge phải xác định GT signals trước
   - rồi mới quyết định applicable
   - rồi mới check gate
   - rồi mới chấm score

4. Khóa **output discipline**
   - chỉ xuất dữ liệu đúng schema
   - rationale ngắn
   - không kể reasoning dài dòng

### 5.3 User payload nên tối giản

`Design choice for v1`

User payload chỉ nên có:

- `Task: evaluate a generated navigation guidance against the ground truth instruction for a visually impaired user.`
- `Ground truth: ...`
- `Generation: ...`

Không nên thêm:

- câu giải thích lại rubric dài dòng
- ví dụ few-shot trong v1
- `question/context` mặc định

Lý do:

- `alter` là task ngắn
- JSON schema đã lo phần format
- rubric chính nên nằm ở system instruction

### 5.4 Procedure khóa cứng trong prompt

`Design choice for v1`

Prompt phải yêu cầu judge làm đúng thứ tự sau:

1. Xác định GT có:
   - `direction anchor` hay không
   - `action demand` hay không
   - `hazard_or_path_state` hay không
2. Từ đó quyết định criterion nào `applicable`
3. Kiểm tra gate:
   - `polarity_reversal`
   - `unsafe_action`
   - `unsafe_direction_reversal`
4. Chấm từng criterion bằng score `0..3`
5. Xuất criterion scores và gate flags
6. Để code/pipeline tính `overall_score`
7. Xuất JSON

Đây là điểm quan trọng nhất của v1. Nếu bỏ thứ tự này, prompt vẫn có thể “đúng ý” nhưng dễ chấm sai logic.

## 6. Những section bắt buộc phải có trong prompt

## 6.1 Role

`Design choice for v1`

Prompt phải nói judge là:

- một evaluator cho navigation guidance dành cho người khiếm thị
- safety-first
- reference-based

Không nên mô tả judge quá chung chung là “helpful evaluator”.

## 6.2 Evaluation objective

Prompt phải nói rõ:

- mục tiêu là đánh giá generation có đúng và an toàn so với GT hay không
- không đánh giá độ hay văn phong chung chung
- không đánh giá theo lexical overlap đơn thuần

## 6.3 Criterion definitions

Prompt phải định nghĩa đủ 5 criteria:

- `Safety Correctness`
- `Hazard / Path-State Fidelity`
- `Direction Fidelity`
- `Action Usefulness`
- `Spoken-Guidance Quality`

Mỗi criterion nên được mô tả bằng 1-3 câu ngắn, không dùng định nghĩa dài kiểu essay.

## 6.4 Applicable / N/A rules

Prompt phải có rule riêng cho:

- `Direction`
  - applicable khi GT có `o'clock`, `left/right`, `ahead/front/straight ahead`, hoặc anchor vị trí đủ rõ
- `Action`
  - applicable khi GT có `action demand` đủ rõ, không phải cứ có hazard/path-state là mặc định applicable

Prompt cũng phải nói rõ:

- `N/A` không phải điểm 0
- `N/A` không loại bỏ lỗi safety nếu generation tự bịa direction/action nguy hiểm

Rule chặt hơn cho `Action applicable`:

- applicable khi GT thật sự:
  - yêu cầu một bước hành động trực tiếp như `slow down`, `avoid`, `move forward`, `walk slowly`
  - hoặc mô tả `clear/safe path` theo cách đã chốt một bước nên làm như `please move forward without worry`
  - hoặc nêu hazard/path-state theo cách khiến bước hành động là một phần cốt lõi của guidance, chứ không chỉ là thông tin scene
- không applicable khi GT chỉ:
  - mô tả scene/vị trí ngắn
  - nêu object/hazard presence nhưng chưa chốt người dùng nên làm gì
  - cung cấp ngữ cảnh như `this is a crossroad` hoặc `the front intersection is available for turning` mà chưa thành guidance step

Nói ngắn gọn:

- `Action applicable` cần `action demand`
- `hazard/path-state` chỉ là bằng chứng phụ, không tự động biến mọi mẫu thành `Action applicable`

## 6.5 Gate rules

Prompt phải định nghĩa và mô tả ngắn:

- `polarity_reversal`
- `unsafe_action`
- `unsafe_direction_reversal`

Và phải nói rõ:

- gate được check trước
- gate là tín hiệu để code/pipeline cap `overall_score`

Nguyên tắc chốt:

- gate chỉ dùng cho **lỗi safety-critical rõ ràng**
- lỗi fidelity nặng vẫn có thể bị chấm `Fail = 0` ở criterion mà **không cần bật gate**
- với `hazard hallucination` hoặc `hazard omission`:
  - mặc định xử lý trong `Hazard / Path-State Fidelity`
  - chỉ khi lỗi đó kéo sang đảo kết luận an toàn hoặc tạo hành động nguy hiểm thì mới bật:
    - `polarity_reversal`
    - hoặc `unsafe_action`

Rule cap khuyến nghị cho pipeline:

- `polarity_reversal`
  - pipeline đặt `overall_score = 0.0`
- `unsafe_action`
  - pipeline đặt `overall_score = 0.0`
- `unsafe_direction_reversal`
  - pipeline đặt `overall_score = min(mean_before_gate, 1.0)`

Nếu nhiều gate cùng bật:

- dùng mức cap nghiêm khắc nhất
- pipeline không được “bù lại” bằng wording hay spoken quality tốt

Nhưng ở v1, judge **không tự suy diễn cap** theo kiểu `overall_score <= ...`.
Judge chỉ cần trả:

- criterion scores
- gate flags

Phần cap cuối cùng sẽ do code/pipeline tính một cách deterministic.

## 6.6 Scale anchors

Prompt phải khóa:

- `0 = Fail`
- `1 = Weak`
- `2 = Acceptable`
- `3 = Strong`

Và nhấn mạnh:

- score là primary output
- label text chỉ là optional explanation layer nếu cần

## 6.7 Anti-over-inference rules

Prompt phải cấm judge:

- bịa chi tiết GT không có
- dùng generation để suy ngược rằng GT “chắc là có”
- phạt generation chỉ vì không giống từ GT
- thưởng generation chỉ vì trùng wording

## 7. JSON output contract

Từ đây phải tách rõ **hai lớp output khác nhau**:

1. **Judge output**
   - là JSON trả trực tiếp từ `GPT-4o`
   - chỉ chứa semantic judgment:
     - `signals_in_gt`
     - `criteria`
     - `gate`
     - rationale ngắn
   - **không** tự tính `mean_before_gate`
   - **không** tự tính `applied_gate_cap`
   - **không** tự tính `overall_score`

2. **Pipeline output**
   - là JSON cuối do code/pipeline tạo ra sau khi nhận judge output
   - dùng rule deterministic để tính:
     - `mean_before_gate`
     - `applied_gate_cap`
     - `overall_score`

Rule chốt:

- judge chỉ làm phần **nhận định ngữ nghĩa**
- code/pipeline làm phần **tổng hợp điểm cuối**

## 7.1 Judge output khuyến nghị

`Design choice for v1`

```json
{
  "gate": {
    "polarity_reversal": false,
    "unsafe_action": false,
    "unsafe_direction_reversal": false
  },
  "signals_in_gt": {
    "has_direction_anchor": true,
    "has_action_demand": true,
    "has_hazard_or_path_state": true
  },
  "criteria": {
    "safety_correctness": {
      "applicable": true,
      "score": 3,
      "rationale": "..."
    },
    "hazard_path_state_fidelity": {
      "applicable": true,
      "score": 2,
      "rationale": "..."
    },
    "direction_fidelity": {
      "applicable": true,
      "score": 3,
      "rationale": "..."
    },
    "action_usefulness": {
      "applicable": true,
      "score": 2,
      "rationale": "..."
    },
    "spoken_guidance_quality": {
      "applicable": true,
      "score": 2,
      "rationale": "..."
    }
  },
  "overall_rationale": "..."
}
```

Judge output **không** nên có:

- `mean_before_gate`
- `applied_gate_cap`
- `overall_score`

vì ba field này không phải semantic judgment; chúng là kết quả tổng hợp deterministic của pipeline.

## 7.2 Rule cho từng field của judge output

`Design choice for v1`

- `gate.*`
  - bắt buộc
  - boolean
  - chỉ gồm:
    - `polarity_reversal`
    - `unsafe_action`
    - `unsafe_direction_reversal`
- `signals_in_gt.*`
  - bắt buộc
  - boolean
- `criteria.<name>.applicable`
  - bắt buộc
  - boolean
- `criteria.<name>.score`
  - số `0..3` khi applicable
  - `null` khi `applicable = false`
- `criteria.<name>.rationale`
  - bắt buộc
  - 1 câu ngắn
- `overall_rationale`
  - ngắn, 1-2 câu

Rule chốt:

- judge chỉ trả criterion scores theo map:
  - `Fail = 0`
  - `Weak = 1`
  - `Acceptable = 2`
  - `Strong = 3`
- code/pipeline sẽ tính:
  - `mean_before_gate` = mean trên các criterion applicable
  - nếu `polarity_reversal` hoặc `unsafe_action`: `overall_score = 0.0`
  - else nếu `unsafe_direction_reversal`: `overall_score = min(mean_before_gate, 1.0)`
  - else: `overall_score = mean_before_gate`
- `N/A` chỉ loại criterion khỏi `mean_before_gate`, không miễn trừ lỗi safety

## 7.3 Pipeline output khuyến nghị

`Design choice for v1`

```json
{
  "judge_output": {
    "gate": {
      "polarity_reversal": false,
      "unsafe_action": false,
      "unsafe_direction_reversal": false
    },
    "signals_in_gt": {
      "has_direction_anchor": true,
      "has_action_demand": true,
      "has_hazard_or_path_state": true
    },
    "criteria": {
      "safety_correctness": {
        "applicable": true,
        "score": 3,
        "rationale": "..."
      },
      "hazard_path_state_fidelity": {
        "applicable": true,
        "score": 2,
        "rationale": "..."
      },
      "direction_fidelity": {
        "applicable": true,
        "score": 3,
        "rationale": "..."
      },
      "action_usefulness": {
        "applicable": true,
        "score": 2,
        "rationale": "..."
      },
      "spoken_guidance_quality": {
        "applicable": true,
        "score": 2,
        "rationale": "..."
      }
    },
    "overall_rationale": "..."
  },
  "mean_before_gate": 2.4,
  "applied_gate_cap": null,
  "overall_score": 2.4
}
```

## 7.4 Rule cho pipeline output

`Design choice for v1`

- `mean_before_gate`
  - do code/pipeline tính
  - là mean trên các criterion applicable trước khi áp gate
- `applied_gate_cap`
  - do code/pipeline tính
  - `null` nếu không có gate nào cap điểm
  - nếu có gate, trả về giá trị cap cuối cùng đã áp, ví dụ `1.0` hoặc `0.0`
- `overall_score`
  - do code/pipeline tính
  - là điểm sau khi áp gate cap

Nếu xuất `mean_before_gate`, `applied_gate_cap`, `overall_score` trong JSON cuối, cần ghi rõ:

- đây là field do code/pipeline tính từ output judge
- không phải phần judge tự áp cap một cách mơ hồ

## 7.5 Rationale length

`Design choice for v1`

Khuyến nghị:

- criterion rationale: tối đa 1 câu
- overall rationale: tối đa 2 câu

Không nên yêu cầu:

- chain-of-thought dài
- trích dẫn quote dài
- explanation nhiều đoạn

## 8. Bộ sanity cases để kiểm prompt

`Design choice for v1`

| Case | Applicable dự kiến | Gate dự kiến | `overall_score` kỳ vọng |
|---|---|---|---|
| 1. Hazard + direction + action, gen đúng hoàn toàn | cả 5 trừ khi spoken quá ngắn | không | `2.5 - 3.0` |
| 2. Hazard + direction + action, gen sai direction nhẹ | direction applicable | có thể không gate | `1.5 - 2.5` |
| 3. Hazard + direction + action, gen đảo chiều nguy hiểm | direction + safety applicable | `unsafe_direction_reversal` | `0.0 - 1.0` |
| 4. GT chỉ `slow down`, gen đúng, không direction | direction `N/A`, action applicable | không | `2.0 - 3.0` |
| 5. GT chỉ `slow down`, gen tự bịa hướng nguy hiểm | direction có thể `N/A`, safety vẫn phạt | `unsafe_direction_reversal` có thể bật | `0.0 - 1.0` |
| 6. GT `clear path`, gen paraphrase đúng nghĩa | hazard/path-state applicable, direction có thể `N/A` | không | `2.0 - 3.0` |
| 7. GT `clear path`, gen bịa obstacle chính | hazard/path-state applicable | thường không gate; phạt mạnh ở `Hazard / Path-State Fidelity`, trừ khi kéo sang kết luận nguy hiểm | `0.0 - 1.5` |
| 8. GT `crossroad`, gen trung tính | direction/action có thể `N/A` | không | `1.5 - 2.5` |
| 9. GT `crossroad`, gen khuyên hành động nguy hiểm | action có thể `N/A`, safety vẫn phạt | `unsafe_action` có thể bật | `0.0 - 1.0` |
| 10. Gen đúng nghĩa nhưng dài dòng | spoken quality thấp hơn | không | thấp hơn case tương đương khoảng `0.5 - 1.0` điểm |
| 11. Hazard đúng nhưng action thiếu | action applicable, hazard vẫn đúng | thường không gate | `1.5 - 2.5` |
| 12. GT `clear path`, gen hallucinate blocked/obstacle | hazard/path-state applicable | thường không gate; chỉ bật gate nếu kéo sang kết luận an toàn sai hoặc hành động nguy hiểm | `0.0 - 1.0` |
| 13. Wording rất giống GT nhưng sai hướng | direction applicable | `unsafe_direction_reversal` hoặc direction fail nặng | `0.0 - 1.0` |

Ý nghĩa của bộ này:

- prompt chỉ được coi là ổn nếu nó xử lý đúng các case `N/A nhưng vẫn phạt safety`
- đây là nơi dễ lộ bug prompt nhất

Ba case mới rất quan trọng:

- case 11 kiểm tra judge có nhầm `hazard đúng` thành `mọi thứ đều ổn` hay không
- case 12 kiểm tra judge có bắt được lỗi `clear-path -> blocked-path hallucination` như một lỗi fidelity nặng, và chỉ bật gate khi nó kéo sang lỗi safety-critical
- case 13 kiểm tra prompt có chống lexical-overlap bias đủ mạnh hay không

## 9. Bảng quyết định prompt design choice -> vì sao dùng với `GPT-4o`

`Design choice for v1`

| Prompt design choice | Vì sao nên dùng với `GPT-4o` |
|---|---|
| Dùng Structured Outputs/schema để khóa JSON | giảm gánh nặng ép format bằng prompt và tăng độ ổn định output |
| Đặt rubric logic ở system instruction | giúp role và luật ít bị “trôi” hơn data payload |
| User payload chỉ chứa `GT` và `generation` | giảm nhiễu, giảm token, giảm nguy cơ lẫn luật với dữ liệu |
| Procedure-first ordering | giảm rubric execution drift và over-inference |
| Có `signals_in_gt` explicit | buộc judge quyết định applicability từ GT trước khi chấm |
| Score `0..3` là primary output | dễ benchmark và khớp logic repo hiện tại |
| Rationale ngắn | giảm self-justification và suy diễn quá mức |
| Rule chống lexical overlap bias viết rõ | đặc biệt quan trọng vì `alter` khá template hóa |
| Rule `N/A` không che hallucination nguy hiểm viết rõ | tránh lỗi logic nghiêm trọng nhất của v1 |

## 10. Những gì chưa nên chốt ở giai đoạn này

`Design choice for v1`

- exact wording cuối cùng của prompt
- few-shot examples trong prompt
- có cần `label` text song song với `score` hay không
- có cần `confidence` field hay không
- có cần self-consistency nhiều lượt gọi API hay không
- có cần human-calibrated post-hoc remapping hay không

Các phần này nên để sang bước sau, sau khi đã có prompt v1 và stress-test bằng sanity cases.

## 11. Kết luận cuối cùng

Nếu phải chốt một hướng prompt judge phù hợp nhất cho `GPTScore` v1 ở repo hiện tại, thì hướng nên chọn là:

- `GPT-4o`
- `Structured Outputs`
- `one-shot, one-JSON`
- `procedure-first structured rubric judge`
- `score-first output`
- `N/A` có kiểm soát
- `gate` đứng trên mean score

Nói ngắn gọn:

> Prompt tốt cho `GPTScore` v1 không phải là prompt “bắt model chấm điểm thật khéo”, mà là prompt buộc judge đi đúng thứ tự:  
> hiểu GT có tín hiệu gì, biết criterion nào mới được chấm, không dùng `N/A` để bỏ qua lỗi nguy hiểm, rồi mới xuất điểm dưới một JSON ổn định.

## 12. Nguồn chính đã dùng

- Structured model outputs | OpenAI API  
  https://developers.openai.com/api/docs/guides/structured-outputs

- Introducing Structured Outputs in the API | OpenAI  
  https://openai.com/index/introducing-structured-outputs-in-the-api/

- G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment  
  https://aclanthology.org/2023.emnlp-main.153/

- SAJA: A Simple Approach to Judge Alignment for LLM-as-a-Judge  
  https://aclanthology.org/2026.acl-industry.45.pdf

- From Rubrics to Reliable Scores: Evidence-Grounded Text Evaluation with LLM Judges  
  https://arxiv.org/html/2601.08654v2

- Calibrating LLM-Based Evaluator  
  https://aclanthology.org/2024.lrec-main.237.pdf

- Learning to Judge: LLMs Designing and Applying Evaluation Rubrics  
  https://aclanthology.org/2026.findings-eacl.335.pdf

- Thiết Kế `GPTScore` Cho `alter`: Bản Dễ Hiểu, Chặt, Và Dùng Được Để Implement  
  [2026-07-08-gptscore-design-for-alter.md](D:/NCKH_VLM/finetune-InternVL2/docs/research/2026-07-08-gptscore-design-for-alter.md)

- Thiết Kế Thang Điểm Cho Từng Tiêu Chí `GPTScore` Của `alter`  
  [2026-07-08-gptscore-scale-design.md](D:/NCKH_VLM/finetune-InternVL2/docs/research/2026-07-08-gptscore-scale-design.md)
