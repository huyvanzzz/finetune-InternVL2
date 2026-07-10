# Thiết Kế Thang Điểm Cho Từng Tiêu Chí `GPTScore` Của `alter`

## 1. Vì sao phải chốt lại thang điểm

Tài liệu trước đã chốt được các tiêu chí chính cho `GPTScore` của `alter`:

- `Safety Correctness`
- `Hazard / Path-State Fidelity`
- `Direction Fidelity`
- `Action Usefulness`
- `Spoken-Guidance Quality`

Nhưng vẫn còn một câu hỏi rất quan trọng:

> Mỗi tiêu chí nên dùng mấy mức chấm, và mỗi mức phải được hiểu như thế nào?

Nếu làm phần này không chặt, sẽ có 3 rủi ro:

1. Judge chấm không ổn định vì ranh giới giữa các mức quá mơ hồ.
2. Báo cáo nhìn có vẻ chi tiết nhưng thực ra “điểm đẹp mà không đáng tin”.
3. Engineer sau này phải tự nghĩ lại cách đổi band sang điểm cuối, dẫn tới lệch thiết kế ban đầu.

Vì vậy, mục tiêu của tài liệu này là chốt rõ:

- nên dùng thang `3`, `4`, hay `5` mức
- tại sao chọn thang đó
- từng mức của từng tiêu chí nghĩa là gì
- cách tổng hợp thành `overall` như thế nào

Ghi chú quan trọng:

- `Fail` không phải là nhãn có sẵn trong dữ liệu `alter`
- `Fail` là mức đánh giá dành cho **generation khi so với GT**

## 2. Nền dữ liệu `alter` nói gì về việc chấm điểm

### 2.1 Không phải tiêu chí nào cũng áp dụng cho mọi mẫu

`Data-evidenced`

Từ các lượt audit heuristic bảo thủ trên train `alter` hiện tại:

- `direction anchor` đủ rõ xuất hiện khoảng `76% - 78%`
- `hazard/path-state` đủ rõ xuất hiện khoảng `67% - 75%`
- `action demand` đủ rõ xuất hiện khoảng `56% - 59%`

Điều này có nghĩa:

- không phải mẫu nào cũng có đủ thông tin để chấm `Direction`
- không phải mẫu nào cũng có đủ thông tin để chấm `Hazard`
- không phải mẫu nào cũng có đủ thông tin để chấm `Action`

Trong khi đó:

- `Safety`
- `Spoken-Guidance Quality`

gần như luôn có thể chấm.

### 2.2 Bảng nền về tính `applicable`

`Data-evidenced` + `Design choice for v1`

| Tiêu chí | Trạng thái mặc định |
|---|---|
| `Safety Correctness` | gần như luôn applicable |
| `Hazard / Path-State Fidelity` | conditionally applicable |
| `Direction Fidelity` | conditionally applicable |
| `Action Usefulness` | conditionally applicable |
| `Spoken-Guidance Quality` | gần như luôn applicable |

Ý nghĩa trực tiếp:

- thang điểm không thể được thiết kế theo kiểu “mẫu nào cũng chấm đủ 5 tiêu chí như nhau”
- phần `overall` bắt buộc phải hỗ trợ `N/A`

## 3. Survey: `3 mức`, `4 mức`, hay `5 mức`?

## 3.1 Những gì research nói chắc được

### 3.1.1 Scale design thật sự ảnh hưởng tới LLM judge

`Research-supported inference`

Paper *Evaluating the Consistency of LLM Evaluators* cho thấy LLM evaluators có vấn đề về:

- `Self-Consistency`
- `Inter-scale Consistency`

và “different scoring scales can result in divergent evaluation results, especially using non-numerical scoring scales.” Nguồn: https://aclanthology.org/2025.coling-main.710.pdf

Nói đơn giản:

- đổi thang điểm là có thể đổi hành vi judge
- không thể coi scale chỉ là phần trình bày

### 3.1.2 Có bằng chứng thực nghiệm rằng `0-5` có thể align tốt với human hơn khi gộp nhiều task

`Research-supported inference`

Paper *Grading Scale Impact on LLM-as-a-Judge: Human-LLM Alignment Is Highest on 0-5 Grading Scale* báo rằng:

- choice of scale làm thay đổi human–LLM agreement đáng kể
- khi gộp trên nhiều benchmark, thang `0-5` cho alignment mạnh nhất

Nguồn: https://arxiv.org/abs/2601.03444

Nhưng cần đọc rất cẩn thận:

- đây là kết quả **aggregate** trên nhiều task
- không phải bằng chứng trực tiếp rằng bài toán `alter` nên dùng 6 mức số `0..5`
- càng không phải bằng chứng rằng mọi criterion rubric đều nên dùng thang 5

### 3.1.3 Rubric-based judging cũng có bias theo vị trí của các mức

`Research-supported inference`

Paper *Am I More Pointwise or Pairwise? Revealing Position Bias in Rubric-Based LLM-as-a-Judge* cho thấy rubric-based evaluation cũng có position bias, vì model đang thực hiện một dạng multiple-choice trên các mức rubric. Nguồn: https://arxiv.org/abs/2602.02219

Ý nghĩa:

- càng nhiều mức, càng nhiều lựa chọn
- càng nhiều lựa chọn, gánh nặng phân biệt và rủi ro bias vị trí có thể càng tăng

### 3.1.4 Detailed criterion definitions thường giúp ổn định hơn

`Research-supported inference`

Trong *Evaluating the Consistency of LLM Evaluators*, criterion granularity có xu hướng tăng consistency khi định nghĩa tiêu chí chi tiết hơn. Nguồn: https://aclanthology.org/2025.coling-main.710.pdf

Ý nghĩa:

- judge ổn hơn khi anchor descriptions rõ
- nên đầu tư vào mô tả từng mức, không chỉ chọn số mức

## 3.2 So sánh thực dụng giữa `3`, `4`, và `5`

### `3 mức`

Ưu điểm:

- dễ viết anchor
- dễ calibrate
- judge ít phải phân biệt quá nhiều vùng trung gian

Nhược điểm:

- dễ dồn quá nhiều case vào mức giữa
- khó tách:
  - `omission nhẹ`
  - `mismatch đáng kể`
  - `nguy hiểm rõ`

Phù hợp khi:

- tiêu chí gần nhị phân
- hoặc muốn một bản cực đơn giản

### `4 mức`

Ưu điểm:

- đủ để tách:
  - `fail`
  - `weak`
  - `acceptable`
  - `strong`
- tránh một mức giữa quá mơ hồ kiểu “trung bình”
- vẫn chưa quá mịn để judge phải phân biệt nhiều ranh giới nhỏ

Nhược điểm:

- không trực tiếp có “điểm đẹp” kiểu `0-5`
- vẫn cần anchor viết kỹ, nếu không `weak` và `acceptable` dễ lẫn

Phù hợp khi:

- cần rubric có khả năng phân biệt lỗi nặng, lỗi vừa, ổn, tốt
- muốn giữ v1 đơn giản nhưng không quá thô

### `5 mức`

Ưu điểm:

- giàu biểu đạt hơn
- hợp với một số setting tổng quát nơi cần alignment cao với human panel trên nhiều task

Nhược điểm:

- judge phải phân biệt nhiều ranh giới hơn
- anchor burden tăng
- dễ có hai mức “ở giữa” gần nhau một cách giả tạo
- với task safety ngắn như `alter`, nhiều mức có thể không được dùng ổn định

Phù hợp khi:

- đã có human calibration
- đã có anchor examples mạnh
- hoặc mục tiêu là benchmark chung nhiều task hơn là task-specific safety rubric

## 3.3 Kết luận so sánh

`Design choice for v1`

### Khuyến nghị chính cho v1

Dùng **`4 mức` cho tất cả các tiêu chí ordinal**.

### Lý do

1. `3 mức` quá thô cho `Direction`, `Hazard`, `Action`.
2. `5 mức` quá mịn cho v1 của một task safety-specific, trong khi chưa có calibration với human.
3. `4 mức` đủ để tách lỗi nặng khỏi lỗi vừa, và phân biệt “ổn” với “tốt”.
4. `4 mức` tránh mức giữa trung tính quá mơ hồ.

### Fallback

Nếu sau này calibration thực tế cho thấy `weak` và `acceptable` bị judge lẫn nhiều, fallback nên là:

- rút về **`3 mức`**

chứ không nên nhảy lên `5 mức`.

### Khi nào mới nên cân nhắc `5 mức`

Chỉ nên xem xét `5 mức` nếu có:

- human annotation set đủ tốt
- bằng chứng calibration rằng mức thứ 5 đem lại tín hiệu thực sự hữu ích
- tỷ lệ judge nhầm lẫn giữa các mức liền kề đủ thấp

## 4. Không nên dùng “label trừu tượng” đơn thuần

## 4.1 `abstract labels` không đủ

`Design choice for v1`

Nếu chỉ dùng:

- `poor`
- `fair`
- `good`
- `excellent`

thì judge và người đọc rất dễ hiểu khác nhau.

Ví dụ:

- `good` của `Direction` là sai nhẹ hay đúng hoàn toàn?
- `fair` của `Hazard` là thiếu hazard chính hay chỉ paraphrase kém?

Do đó, label ngắn chỉ nên là lớp hiển thị. Phần quan trọng hơn phải là:

- **behavioral anchors**

## 4.2 Nên dùng `behavioral anchors`

`Research-supported inference`

*Autorubric* mô tả rubric là một scoring instrument gồm criteria và performance-level descriptions. Nguồn: https://arxiv.org/pdf/2603.00077

Áp sang repo này:

- mỗi mức phải được định nghĩa bằng hành vi / lỗi / chất lượng cụ thể
- không nên chỉ gắn tên mức mà không nói mức đó thực sự nghĩa là gì

## 4.3 Nguyên tắc viết anchor descriptions

`Design choice for v1`

Mỗi anchor nên:

1. bám vào semantics của `alter`, không dùng mô tả chung chung
2. nói rõ dấu hiệu để vào mức đó
3. nói rõ lỗi nào là “nặng”
4. không thưởng chỉ vì wording giống GT
5. cho phép paraphrase đúng nghĩa vẫn lên mức cao

## 4.4 Rule phân biệt `Safety` và `Hazard / Path-State`

`Research-supported inference` + `Design choice for v1`

Đây là cặp tiêu chí dễ bị chồng nhau nhất, nên cần khóa rule rõ:

- `Hazard / Path-State Fidelity`
  - đánh giá generation có mô tả đúng obstacle / risk / clear-path state mà GT nêu không
- `Safety Correctness`
  - đánh giá generation có chuyển tải kết luận an toàn đúng chiều không

Ma trận rule thực dụng:

| Tình huống | Hazard / Path-State | Safety |
|---|---|---|
| mô tả sai obstacle nhưng chưa làm đổi guidance an toàn | giảm mạnh | giảm vừa hoặc giữ ở mức trung bình |
| mô tả sai obstacle và kéo theo guidance nguy hiểm | fail hoặc weak | fail và có thể bật gate |
| mô tả đúng obstacle nhưng hành động quá yếu | acceptable/strong | không được strong |
| GT là `clear/unobstructed`, generation bịa obstacle | fail | weak hoặc fail tùy mức độ kéo lệch guidance |

Rule chốt:

- `Hazard` hỏi “scene được mô tả đúng chưa?”
- `Safety` hỏi “guidance cuối có an toàn chưa?”

Ví dụ nhanh:

- GT: `There are stairs in front, be careful and walk slowly.`
- Gen: `There are stairs ahead.`  
  -> `Hazard / Path-State` có thể vẫn cao vì stairs còn đúng, nhưng `Safety` không nên cao tương ứng vì generation chưa làm tròn vai guidance an toàn.

- GT: `the current road is clear, please move forward without worry.`
- Gen: `There is a car in front. Stop immediately.`  
  -> `Hazard / Path-State` fail vì bịa sai scene; `Safety` cũng giảm mạnh vì guidance đã bị kéo sang một kết luận an toàn khác hẳn GT.

## 4.5 Rule `Direction` và `Action` applicable

`Data-evidenced` + `Design choice for v1`

### `Direction Fidelity` applicable khi nào

Applicable nếu GT có neo định hướng rõ:

- `o'clock`
- `left/right`
- `ahead/front/straight ahead`
- vị trí tương đối có vai trò thực trong guidance

`N/A` nếu:

- GT chỉ nói `clear path`
- GT chỉ nói `slow down`
- GT chỉ nêu hazard chung mà không gắn hướng

Nhưng cần khóa thêm một rule:

- `Direction = N/A` không được dùng để che một direction hallucination nguy hiểm.
- Nếu GT không có direction anchor rõ mà generation vẫn tự thêm hướng theo chiều rủi ro, lỗi đó vẫn phải bị phạt qua `Safety` và có thể kích hoạt `unsafe_direction_reversal`.

### `Action Usefulness` applicable khi nào

Applicable nếu GT:

- có action cue trực tiếp
- hoặc có hazard/path-state đủ rõ để guidance hành động là một phần cốt lõi của câu

`N/A` nếu:

- GT chỉ là mô tả vị trí/ngữ cảnh ngắn
- chưa thực sự chốt một bước nên làm

Nhưng cần khóa thêm một rule:

- `Action = N/A` không được dùng để che một action hallucination nguy hiểm.
- Nếu GT không thật sự đòi action mà generation lại khuyên một bước đi sai chiều an toàn, lỗi đó vẫn phải bị phạt qua `Safety` và có thể kích hoạt `unsafe_action`.

Rule ngắn gọn:

- `Direction` cần `direction anchor`
- `Action` cần `action demand`

Ví dụ nhanh:

- `at 11 o'clock direction, there are pedestrians passing by. be careful to avoid.`  
  -> `Direction` applicable, `Action` cũng applicable.

- `the current road is narrow, please slow down.`  
  -> `Action` applicable, nhưng `Direction` thường là `N/A`.

- `this is a crossroad.`  
  -> `Direction` thường `N/A`, `Action` cũng thường `N/A`.

- `the current road is narrow, please slow down.` + gen `Turn right and move quickly.`  
  -> `Direction` có thể `N/A` theo nghĩa fidelity, nhưng vẫn phải phạt safety mạnh.

- `this is a crossroad.` + gen `Run across quickly.`  
  -> `Action` có thể `N/A` theo nghĩa usefulness trực tiếp, nhưng vẫn phải phạt safety mạnh.

## 5. Khuyến nghị thang cụ thể cho từng criterion

## 5.1 Cùng một số mức cho mọi criterion hay không

`Design choice for v1`

Khuyến nghị:

- dùng **cùng một thang `4 mức`** cho tất cả 5 criteria

### Vì sao

- dễ implement
- dễ hướng dẫn judge
- dễ tổng hợp `overall`
- dễ calibrate hơn việc mỗi criterion một kiểu thang riêng

Không phải criterion nào cũng quan trọng ngang nhau, nhưng:

- chuyện **quan trọng** được xử lý bởi `gate`
- không nhất thiết phải xử lý bằng số mức khác nhau

## 5.2 Bảng quyết định cho từng criterion

`Design choice for v1`

| Criterion | Số mức | Khuyến nghị |
|---|---:|---|
| `Safety Correctness` | 4 | giữ 4 mức, dù gần nhị phân, vì vẫn cần tách “sai nguy hiểm”, “hơi lệch”, “ổn”, “rất đúng” |
| `Hazard / Path-State Fidelity` | 4 | cần 4 mức để tách omission nhẹ khỏi hallucination nguy hiểm |
| `Direction Fidelity` | 4 | cần 4 mức để tách `minor omission`, `mismatch`, `unsafe reversal` |
| `Action Usefulness` | 4 | cần 4 mức để tách “không sai nhưng vô ích” khỏi “hữu ích rõ rệt” |
| `Spoken-Guidance Quality` | 4 | vẫn dùng 4 mức để đồng bộ, dù là criterion nhẹ hơn |

## 6. Bảng anchor cụ thể cho từng criterion

## 6.1 `Safety Correctness`

`Design choice for v1`

| Mức | Label | Nghĩa |
|---:|---|---|
| 0 | `Fail` | Đảo polarity an toàn, khuyên hành động nguy hiểm, hoặc làm người dùng hiểu sai tình huống theo hướng rủi ro rõ rệt |
| 1 | `Weak` | Không đảo polarity hoàn toàn nhưng diễn đạt khiến guidance kém an toàn, thiếu cảnh báo quan trọng, hoặc tạo cảm giác an toàn quá mức |
| 2 | `Acceptable` | Giữ được ý nghĩa an toàn chính, có thể còn thiếu độ chắc chắn hoặc chưa tối ưu về mức cảnh báo |
| 3 | `Strong` | Phản ánh đúng mức độ an toàn/nguy hiểm và truyền tải guidance an toàn rõ ràng |

## 6.2 `Hazard / Path-State Fidelity`

`Design choice for v1`

| Mức | Label | Nghĩa |
|---:|---|---|
| 0 | `Fail` | Bịa hazard/path-state chính theo cách làm đổi bản chất scene, hoặc bỏ mất hazard/path-state cốt lõi |
| 1 | `Weak` | Nêu sai hoặc thiếu một phần quan trọng của hazard/path-state, nhưng chưa đến mức đảo ngược hoàn toàn |
| 2 | `Acceptable` | Giữ đúng ý chính về hazard/path-state, có thể bỏ sót chi tiết phụ hoặc paraphrase chưa gọn |
| 3 | `Strong` | Phản ánh đúng hazard/path-state cốt lõi, không hallucinate, không bỏ sót ý chính |

Ghi chú:

- với mẫu `clear path`, criterion này vẫn applicable
- khi đó “hazard” được hiểu rộng là `hazard / path-state`

## 6.3 `Direction Fidelity`

`Design choice for v1`

| Mức | Label | Nghĩa |
|---:|---|---|
| 0 | `Fail` | Sai hướng theo cách nguy hiểm hoặc làm guidance đổi bản chất rõ rệt (`unsafe reversal`) |
| 1 | `Weak` | Sai hướng đáng kể (`meaningful mismatch`) nhưng chưa đủ để kết luận đảo ngược nguy hiểm hoàn toàn |
| 2 | `Acceptable` | Thiếu hướng hoặc hơi mơ hồ (`minor omission`), nhưng không làm đổi ý nghĩa chính |
| 3 | `Strong` | Hướng/vị trí tương đối khớp rõ ràng (`exact` hoặc tương đương an toàn) |

Ghi chú:

- nếu GT không có direction đủ rõ thì criterion này là `N/A`

## 6.4 `Action Usefulness`

`Design choice for v1`

| Mức | Label | Nghĩa |
|---:|---|---|
| 0 | `Fail` | Khuyên hành động sai chiều an toàn hoặc hành động rõ ràng không phù hợp |
| 1 | `Weak` | Không sai rõ nhưng hành động mơ hồ, kém hữu ích, hoặc thiếu chỉ dẫn cần thiết trong bối cảnh đó |
| 2 | `Acceptable` | Hành động nhìn chung hợp lý và dùng được, dù chưa thật sự tối ưu hoặc chưa đủ cụ thể |
| 3 | `Strong` | Hành động rõ, phù hợp, hữu ích ngay lập tức cho người dùng |

Ghi chú:

- criterion này chỉ applicable khi GT thực sự chứa đủ tín hiệu để kỳ vọng một khuyến nghị hành động
- nếu GT chỉ là mô tả vị trí rất ngắn và không chốt hành vi nên làm, criterion này có thể là `N/A`

## 6.5 `Spoken-Guidance Quality`

`Design choice for v1`

| Mức | Label | Nghĩa |
|---:|---|---|
| 0 | `Fail` | Câu khó hiểu, lủng củng, hoặc không còn mang dạng spoken guidance usable |
| 1 | `Weak` | Câu hiểu được nhưng khá vụng, dài dòng, hoặc thiếu trực tiếp |
| 2 | `Acceptable` | Câu đủ rõ, đủ tự nhiên, dùng được làm spoken guidance |
| 3 | `Strong` | Câu ngắn gọn, trực tiếp, dễ nghe, đúng vai trò trợ lý điều hướng |

## 7. Judge nên trả `band` hay `score`?

## 7.1 Kết luận

`Design choice for v1`

Judge nên trả:

- `band/label` trước
- hệ thống map band sang điểm nội bộ sau

Không nên yêu cầu judge trả score số trực tiếp cho từng criterion làm đầu ra chính.

### Vì sao

- band dễ gắn với anchor hơn
- ít tạo cảm giác chính xác giả
- dễ audit hơn khi judge sai

## 7.2 Điểm số khuyến nghị cho v1

`Design choice for v1`

Với thang `4 mức`:

- `Fail = 0`
- `Weak = 1`
- `Acceptable = 2`
- `Strong = 3`

Fallback nếu buộc phải rút về `3 mức`:

- `Fail = 0`
- `Acceptable = 1`
- `Strong = 2`

Ở mức triển khai, có thể để judge trả `label` rồi code đổi thẳng sang số `0..3`.
Không cần thêm một lớp “map ngược” từ điểm cuối về band tổng.

## 8. Cách tính điểm cuối

## 8.1 Quy tắc tổng hợp

`Design choice for v1`

1. Judge trả `gate flags`
2. Judge trả `criterion labels`
3. Những criterion `N/A` bị loại khỏi mean
4. Tính `mean_internal` trên các criterion applicable
5. Áp `gate cap`
6. Xuất ra `overall_score`

## 8.2 `N/A` handling

`Design choice for v1`

Nếu criterion không applicable:

- không tính `0`
- không tính vào mẫu số
- không ảnh hưởng breakdown của criterion khác

Nhưng:

- `N/A` chỉ loại criterion đó khỏi phép lấy trung bình.
- `N/A` không được xóa lỗi safety nếu generation tự hallucinate direction/action nguy hiểm.

Ví dụ:

- chỉ có 4 criterion applicable
- tổng là `3 + 2 + 2 + 1 = 8`
- `mean_internal = 8 / 4 = 2.0`

## 8.3 `gate` cap

`Design choice for v1`

Nếu có một trong các lỗi nặng:

- `polarity_reversal`
- `unsafe_action`
- `unsafe_direction_reversal`

thì `overall` phải bị cap xuống mạnh, kể cả khi một vài criterion mềm nhìn vẫn ổn.

Khuyến nghị v1:

- nếu có `polarity_reversal` hoặc `unsafe_action`, ép `overall_score = 0.0`
- nếu có `unsafe_direction_reversal` nhưng chưa rơi vào hai case trên, `overall_score` tối đa chỉ được là `1.0`

Lý do:

- không để câu “nói mượt” cứu một guidance nguy hiểm

Không cần map ngược `mean_internal` về `overall_band` trong v1.

Lý do:

- mục tiêu cuối là so sánh model bằng điểm
- band theo từng criterion đã đủ để giải thích lỗi
- thêm một lớp `overall_band` chỉ làm pipeline rối hơn mà không giúp benchmark tốt hơn

## 9. Ví dụ sanity-check trên các case điển hình

## 9.1 Sai polarity an toàn

- GT: `There are stairs in front, be careful and walk slowly.`
- Gen: `The road is clear, please move forward without worry.`

Kỳ vọng:

- `Safety Correctness = Fail`
- `Hazard / Path-State Fidelity = Fail`
- `Action Usefulness = Fail`
- gate bật
- `overall_score = 0.0`

## 9.2 Đúng nghĩa nhưng paraphrase

- GT: `the current road is clear, please move forward without worry.`
- Gen: `The path ahead is unobstructed. You can continue forward carefully.`

Kỳ vọng:

- `Safety = Strong`
- `Hazard / Path-State = Strong` hoặc `Acceptable`
- `Action = Acceptable` hoặc `Strong`
- `Spoken Quality = Acceptable` hoặc `Strong`

Không được phạt chỉ vì wording khác.

## 9.3 Thiếu direction vì GT không có direction

- GT: `the current road is narrow, please slow down.`
- Gen: `Please slow down. The path is narrow.`

Kỳ vọng:

- `Direction = N/A`
- không bị phạt vì không nêu hướng

## 9.4 Hallucination hazard

- GT: `the current road is clear, please move forward without worry.`
- Gen: `There is a car in front. Stop immediately.`

Kỳ vọng:

- `Hazard / Path-State Fidelity = Fail`
- `Safety Correctness = Weak` hoặc `Fail`
- gate có thể bật nếu hallucination làm đổi bản chất guidance

## 9.5 Action mơ hồ nhưng không nguy hiểm

- GT: `at 11 o'clock direction, there are pedestrians passing by. be careful to avoid.`
- Gen: `There are pedestrians ahead.`

Kỳ vọng:

- `Hazard = Acceptable`
- `Direction = Acceptable` hoặc `Strong`
- `Action = Weak`
- `Safety = Acceptable`

## 10. Kết luận cuối cùng

### Khuyến nghị chính cho v1

`Design choice for v1`

- dùng **`4 mức`** cho mọi criterion ordinal
- judge trả **band** thay vì score số trực tiếp
- mọi mức phải có **behavioral anchors**
- `overall_score` tính bằng mean trên các criterion applicable
- `gate` có quyền cap mạnh `overall_score`

### Vì sao không chọn `3 mức` làm mặc định

- quá thô cho `Direction`, `Hazard`, `Action`
- khó tách omission nhẹ khỏi mismatch đáng kể

### Vì sao chưa chọn `5 mức` cho v1

- research có tín hiệu rằng `0-5` có thể align tốt trên aggregate multi-task
- nhưng chưa đủ để kết luận bài toán `alter` task-specific nên dùng 5 mức
- với v1, 5 mức làm tăng gánh nặng anchor và rủi ro dao động

Nói ngắn gọn:

> Nếu mục tiêu là một `GPTScore` đầu tiên vừa đủ tinh để phân biệt các lỗi quan trọng, nhưng chưa quá mịn đến mức judge dao động mạnh, thì `4 mức + behavioral anchors + gate + N/A handling` là thiết kế hợp lý nhất cho `alter`.

## 11. Nguồn chính đã dùng

- Grading Scale Impact on LLM-as-a-Judge: Human-LLM Alignment Is Highest on 0-5 Grading Scale  
  https://arxiv.org/abs/2601.03444

- Evaluating the Consistency of LLM Evaluators  
  https://aclanthology.org/2025.coling-main.710.pdf

- Autorubric: Unifying Rubric-based LLM Evaluation  
  https://arxiv.org/pdf/2603.00077

- Am I More Pointwise or Pairwise? Revealing Position Bias in Rubric-Based LLM-as-a-Judge  
  https://arxiv.org/abs/2602.02219

- Thiết Kế `GPTScore` Cho `alter`: Bản Dễ Hiểu, Chặt, Và Dùng Được Để Implement  
  [2026-07-08-gptscore-design-for-alter.md](D:/NCKH_VLM/finetune-InternVL2/docs/research/2026-07-08-gptscore-design-for-alter.md)
