# Nghiên Cứu `alter` GT Và Đề Xuất Tiêu Chí `GPTScore` v1

## 1. Đặc trưng dữ liệu `alter`

### 1.1 Đường build target hiện tại

`Data-evidenced`

Trong file `preprocessing.py`, hàm `format_ground_truth()` gọi `map_metadata_to_ground_truth()`, và với task `alter` thì `instruction` được lấy trực tiếp từ trường `metadata['alter']`. Ở chế độ `direct_text`, target cuối cùng chỉ là `instruction.strip()`, không có JSON wrapper và không có hậu xử lý ngữ nghĩa bổ sung.

Diễn giải thực tế:

- `GT` đang chấm chính là câu `alter`
- metric mới nên ưu tiên chất lượng của câu hướng dẫn cuối, không phải chất lượng của các trường `location/weather/scene`

### 1.2 Thống kê thực tế trên train/test `alter`

`Data-evidenced`

Trên tập train mixed hiện tại:

- `alter` train: `8571` mẫu
- `alter` test: `1007` mẫu
- độ dài trung bình: `85.35` ký tự
- độ dài trung bình: `14.99` từ
- median: `13` từ
- tỷ lệ unique exact text: `0.799`
- tương đương khoảng `20.1%` mẫu là exact duplicate

Các exact duplicate lặp nhiều nhất:

- `101` lần: `at 11 o'clock direction, there are pedestrians passing by, pay attention to avoid.`
- `78` lần: `at 11 o'clock direction, there are pedestrians passing by, be careful to avoid.`
- `75` lần: `the current road is clear, please move forward without worry.`
- `44` lần: `the current road is unobstructed. please move forward without worry.`
- `42` lần: `the road ahead is clear, please move forward without worry.`

### 1.3 Pattern ngôn ngữ chính

`Data-evidenced`

Tỷ lệ xuất hiện một số pattern quan trọng trên train `alter`:

- có hướng kiểu `o'clock`: `72.61%`
- có `ahead/front/straight ahead`: `39.70%`
- có từ/cụm thể hiện tránh né như `avoid`: `42.54%`
- có từ/cụm thể hiện đi chậm: `13.09%`
- có từ/cụm thể hiện đi tiếp như `move forward/keep walking`: `11.92%`
- có tín hiệu đường an toàn/thoáng như `clear/unobstructed/without worry`: `16.51%`
- có nhắc `pedestrian/passers/people`: `39.38%`
- có nhắc phương tiện như `car/truck/vehicle/motorcycle/bus`: `36.20%`
- có nhắc `stairs/steps`: `15.34%`

### 1.4 Ví dụ đại diện

`Data-evidenced`

Ví dụ kiểu chỉ hành động ngắn:

- `the current road is narrow, please slow down.`

Ví dụ kiểu hazard + action:

- `ahead there are pedestrians gathering in the middle of the road. please slow down and avoid the pedestrians towards 1 o'clock.`

Ví dụ kiểu đường an toàn:

- `the current road is clear, please move forward without worry.`

Ví dụ kiểu cảnh báo bậc thang:

- `There are stairs in front, be careful and walk slowly.`

Ví dụ test `alter`:

- `There is a road sign five steps ahead. Please keep walking slowly on the current road.`
- `at 11 o'clock direction, there are horizontal passing vehicles.`
- `at one o'clock direction, there are pedestrians taking pictures. beware of avoiding them.`

### 1.5 Kết luận về bản chất GT

`Data-evidenced`

`alter` GT không phải caption mở và cũng không phải câu trả lời tự do thiên về diễn đạt. Nó chủ yếu là:

- chỉ dẫn điều hướng ngắn
- cảnh báo vật cản/rủi ro tức thời
- gợi ý hành động an toàn ngay bước kế tiếp

Kết luận trực tiếp:

- GT ưu tiên `độ đúng an toàn` hơn `độ giống câu chữ`
- metric chỉ bám similarity bề mặt sẽ bỏ sót lỗi nguy hiểm như sai hướng hoặc khuyên hành động không an toàn

## 2. Những gì metric cần thưởng/phạt

### 2.1 Semantic contract của `alter`

`Data-evidenced`

Từ dữ liệu thực tế, có thể tách câu `alter` thành 5 thành phần semantic chính:

1. `Hazard/Object`
   - có nhắc đúng vật cản hoặc tác nhân rủi ro không
2. `Direction/Location`
   - có nói đúng hướng tương đối như `11 o'clock`, `ahead`, `front`, `left/right` không
3. `Action`
   - có khuyến nghị hành động phù hợp như `slow down`, `avoid`, `move forward`, `walk slowly` không
4. `Safety polarity`
   - có nói đúng chiều an toàn/nguy hiểm không
   - ví dụ `clear road -> move forward` khác hoàn toàn với `stairs ahead -> be careful`
5. `Conciseness / spoken-guidance quality`
   - có giữ được phong cách ngắn, trực tiếp, dùng được cho người khiếm thị không

### 2.2 Thành phần nào là cốt lõi, thành phần nào là phụ

`Data-evidenced` + `Research-supported inference`

Cốt lõi:

- `Safety correctness`
- `Direction fidelity`
- `Hazard fidelity`
- `Action usefulness`

Phụ:

- `Conciseness / spoken-guidance quality`
- mức độ giống wording reference

Lý do:

- dữ liệu có lặp wording khá mạnh, nên chấm “giống câu” dễ thưởng nhầm cho các câu template
- trong bài toán hỗ trợ người khiếm thị, một câu khác wording nhưng cùng hazard, cùng hướng, cùng hành động an toàn vẫn nên được chấm cao

### 2.3 Các lỗi nặng mà metric phải phạt mạnh

`Design choice for v1` dựa trên `Data-evidenced`

- bịa ra vật cản/hazard không có trong GT
- bỏ sót vật cản chính trong GT khi GT là cảnh báo nguy hiểm
- nói sai hướng tương đối của hazard quan trọng
- khuyên hành động sai chiều an toàn
- biến tình huống nguy hiểm thành tình huống “an toàn, cứ đi tiếp”

Các lỗi nhẹ hơn:

- wording khác nhưng ý đúng
- thiếu một chi tiết phụ không ảnh hưởng hành động
- câu hơi dài nhưng vẫn an toàn và đúng

## 3. Rubric `GPTScore` đề xuất

### 3.1 Cơ sở research

`Research-supported inference`

Nguồn chính hỗ trợ thiết kế rubric:

- **G-Eval**: ủng hộ cách dùng LLM-as-judge với rubric dạng form-filling thay vì chỉ cho điểm chung chung, đồng thời cảnh báo bias của LLM judge đối với văn bản do LLM sinh ra.
- **FLASK**: ủng hộ cách chấm fine-grained theo từng skill/tiêu chí thay vì chỉ một điểm overall.
- **ASSISTER**: nhấn mạnh assistive navigation cần câu hướng dẫn coherent với cảnh hiện tại và nhiệm vụ hỗ trợ tức thời.
- **WalkVLM**: mô tả walking assistance là bài toán tạo “concise yet informative reminders”.
- **LaF-GRPO / NIG-VI**: nhấn mạnh instruction cho VI cần human-centered, precise directional guidance, obstacle adaptation, và safety.

Suy ra cho repo này:

- nên dùng `LLM-as-judge` có rubric cấu trúc
- nên chấm theo nhiều tiêu chí nhỏ
- nên đặt `safety` và `direction correctness` cao hơn `fluency`

### 3.2 Bộ tiêu chí v1

`Design choice for v1`

Đề xuất chấm 5 tiêu chí, mỗi tiêu chí theo thang `0-5`:

1. `Safety Correctness`
   - Đo việc generation có bảo toàn ý nghĩa an toàn/nguy hiểm của GT không.
   - `0-1`: khuyên hành động nguy hiểm hoặc đảo ngược nghĩa an toàn.
   - `2-3`: có ý an toàn nhưng còn thiếu/chưa chắc.
   - `4-5`: phản ánh đúng mức độ nguy hiểm và khuyến nghị hợp lý.

2. `Hazard Fidelity`
   - Đo việc generation có nhắc đúng vật cản/rủi ro chính trong GT không.
   - Trừ mạnh nếu hallucinate vật cản mới hoặc bỏ sót hazard chính.

3. `Direction Fidelity`
   - Đo việc generation có giữ đúng hướng/vị trí tương đối của hazard hoặc hướng di chuyển không.
   - Sai `11 o'clock` thành `1 o'clock` phải bị phạt nặng.

4. `Action Usefulness`
   - Đo tính thực dụng của hành động khuyến nghị.
   - Câu phải trả lời được “người dùng nên làm gì ngay bây giờ”.

5. `Conciseness / Spoken-Guidance Quality`
   - Đo việc câu có ngắn gọn, trực tiếp, dễ nghe và phù hợp vai trò trợ lý điều hướng không.
   - Tiêu chí này là phụ; không được cứu một câu sai an toàn.

### 3.3 Trọng số đề xuất

`Design choice for v1`

Đề xuất trọng số:

- `Safety Correctness`: `0.35`
- `Direction Fidelity`: `0.25`
- `Hazard Fidelity`: `0.20`
- `Action Usefulness`: `0.15`
- `Conciseness / Spoken-Guidance Quality`: `0.05`

Lý do:

- với dữ liệu `alter`, sai hazard/hướng/hành động gây lệch bản chất hơn nhiều so với việc câu không tự nhiên lắm

### 3.4 Quy tắc chặn điểm cho lỗi nguy hiểm

`Design choice for v1`

Để tránh việc một câu nói trôi chảy nhưng nguy hiểm vẫn được điểm cao, v1 nên có hard rule:

- nếu judge xác định `unsafe_action = true` hoặc `wrong_direction` ở hazard chính, thì `overall_score` bị cap tối đa ở mức trung bình thấp
- khuyến nghị cap: `overall_score <= 2.0/5`

### 3.5 Ví dụ cách chấm

`Design choice for v1`

Ví dụ 1:

- GT: `There are stairs in front, be careful and walk slowly.`
- Gen: `The road is clear, please move forward without worry.`

Kỳ vọng:

- `Safety Correctness`: rất thấp
- `Hazard Fidelity`: rất thấp
- `Direction Fidelity`: thấp
- `Action Usefulness`: rất thấp
- `overall_score`: rất thấp
- flags: `unsafe_action=true`, `hallucinated_clear_path=true`

Ví dụ 2:

- GT: `the current road is clear, please move forward without worry.`
- Gen: `The path ahead is unobstructed. You can continue forward carefully.`

Kỳ vọng:

- wording khác nhưng ý gần
- `Safety`, `Hazard`, `Direction`, `Action`: cao
- `Conciseness`: khá cao

Ví dụ 3:

- GT: `at 11 o'clock direction, there are pedestrians passing by. be careful to avoid.`
- Gen: `At 1 o'clock there are pedestrians. Please avoid them.`

Kỳ vọng:

- `Hazard Fidelity`: khá ổn
- `Direction Fidelity`: thấp vì sai hướng
- `overall_score`: bị kéo xuống đáng kể

Ví dụ 4:

- GT: `the current road is narrow, please slow down.`
- Gen: `The road is narrow and there are vehicles and stairs around you, turn right immediately.`

Kỳ vọng:

- bị phạt hallucination do thêm hazard không có trong GT
- bị phạt vì action có thể quá mức so với GT

## 4. Spec đầu vào/đầu ra cho bộ chấm v1

### 4.1 Đầu vào tối thiểu

`Design choice for v1`

Judge v1 nên nhận tối thiểu:

```json
{
  "ground_truth": "...",
  "generation": "...",
  "question": "optional"
}
```

Khuyến nghị:

- `ground_truth`: bắt buộc
- `generation`: bắt buộc
- `question`: optional, chỉ dùng nếu muốn judge nhìn thêm ngữ cảnh yêu cầu

V1 chưa cần ảnh và chưa cần metadata bbox.

### 4.2 Đầu ra tối thiểu

`Design choice for v1`

```json
{
  "safety_correctness": 0,
  "hazard_fidelity": 0,
  "direction_fidelity": 0,
  "action_usefulness": 0,
  "conciseness_spoken_quality": 0,
  "overall_score": 0.0,
  "flags": {
    "hallucinated_hazard": false,
    "wrong_direction": false,
    "unsafe_action": false,
    "missed_main_hazard": false
  },
  "brief_rationale": ""
}
```

Gợi ý implementation sau này:

- 5 tiêu chí con dùng integer `0-5`
- `overall_score` là weighted average trên thang `0-5`
- `brief_rationale` dài `1-3` câu

### 4.3 V1 là metric kiểu gì

`Design choice for v1`

V1 nên là:

- `reference-based`
- `GT vs generation`
- `safety-first`

Không nên là:

- pure wording similarity
- image-based judge
- pairwise preference giữa hai model ngay từ đầu

Lý do:

- bạn đã có sẵn `pairs.json`
- mục tiêu trước mắt là so sánh generation với GT một cách công bằng hơn ROUGE/TF-IDF
- thêm ảnh vào judge sẽ tăng chi phí, tăng biến động, và làm khó tái lập ngay ở phiên bản đầu

## 5. Kết luận chốt cho pha implementation sau

### 5.1 Những gì đã đủ chắc

`Data-evidenced`

- `alter` GT là navigation instruction ngắn, thiên về safety-aware guidance
- dữ liệu có lặp exact text đáng kể
- hướng tương đối, hazard, và action là ba trục semantic nổi bật

### 5.2 Những gì được literature hỗ trợ

`Research-supported inference`

- LLM-as-judge nên dùng rubric có cấu trúc thay vì chấm điểm cảm tính
- chấm fine-grained đáng tin hơn chấm overall đơn
- assistive navigation nên ưu tiên an toàn, hướng chính xác, và tính thực dụng của chỉ dẫn

### 5.3 Quyết định v1

`Design choice for v1`

- dùng `GPTScore` dạng rubric 5 tiêu chí
- ưu tiên `Safety > Direction > Hazard > Action > Fluency`
- thêm safety flags để bắt lỗi nghiêm trọng
- không thưởng mạnh cho similarity bề mặt

## 6. Nguồn đã dùng

- G-Eval (ACL Anthology): https://aclanthology.org/2023.emnlp-main.153/
- FLASK (arXiv): https://arxiv.org/abs/2307.10928
- ASSISTER (ECCV 2022): https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136960269.pdf
- WalkVLM (arXiv): https://arxiv.org/abs/2412.20903
- LaF-GRPO / NIG-VI (AAAI): https://ojs.aaai.org/index.php/AAAI/article/download/40804/44765
