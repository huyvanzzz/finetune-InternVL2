# Báo Cáo Dễ Hiểu: Nên Thiết Kế `GPTScore` Cho `alter` Như Thế Nào?

## 1. Mục tiêu của báo cáo này

Báo cáo này trả lời một câu hỏi rất thực tế:

> Nếu muốn xây dựng một `GPTScore` để so sánh `ground truth` và `generation` cho task `alter`, thì nên thiết kế nó như thế nào cho hợp lý nhất?

Ở đây tôi **chưa đi vào prompt cụ thể**, mà chỉ tập trung vào:

- nên chấm kiểu gì
- nên tránh kiểu gì
- nên ưu tiên tiêu chí nào
- có nên dùng trọng số ngay hay không

## 2. Trước hết, task `alter` thực chất là gì?

Từ dữ liệu hiện tại, `alter` không phải là caption mở kiểu “hãy mô tả bức ảnh”.

Nó gần với:

- câu chỉ dẫn điều hướng ngắn
- câu cảnh báo vật cản / nguy hiểm
- câu nói cho người khiếm thị biết **nên làm gì ngay bây giờ**

Ví dụ:

- `There are stairs in front, be careful and walk slowly.`
- `the current road is clear, please move forward without worry.`
- `at 11 o'clock direction, there are pedestrians passing by. be careful to avoid.`

Nghĩa là bài toán này ưu tiên:

1. đúng về an toàn
2. đúng về hướng
3. đúng về vật cản / hazard
4. đúng về hành động nên làm

Chứ không ưu tiên hàng đầu việc câu có giống từ ngữ với GT hay không.

## 3. Vì sao không nên chấm kiểu “giống câu là được”?

Có hai lý do chính.

### 3.1 Vì dữ liệu `alter` lặp mẫu câu khá nhiều

Trong tập train `alter`, có khá nhiều câu lặp exact text.  
Điều này làm cho các metric kiểu text overlap rất dễ bị “ảo tưởng”:

- câu giống wording thì được điểm cao
- nhưng chưa chắc đúng bản chất an toàn

Ví dụ nguy hiểm:

- GT: `There are stairs in front, be careful and walk slowly.`
- Gen: `The road is clear, please move forward without worry.`

Hai câu đều ngắn, đều tự nhiên, nhưng câu sinh ra là **nguy hiểm nghiêm trọng**.

### 3.2 Vì research về navigation instruction cũng nói metric bề mặt không đủ

Paper về evaluation navigation instruction cho thấy BLEU, ROUGE, METEOR, CIDEr không đánh giá tốt instruction grounded.

Ý nghĩa với bài toán của bạn:

- metric bề mặt không đủ đáng tin
- phải đánh giá theo **khả năng hỗ trợ điều hướng an toàn**

Nguồn:

- https://aclanthology.org/2021.eacl-main.111.pdf

## 4. Vậy `GPTScore` nên làm kiểu gì?

## Kết luận ngắn gọn

Thiết kế hợp lý nhất hiện tại là:

- `pointwise`
- `reference-based`
- `safety-first`
- `2 tầng: gate trước, rubric sau`

Nói dễ hiểu:

1. Đầu tiên xem câu sinh ra có mắc lỗi nguy hiểm không
2. Nếu không mắc lỗi nguy hiểm, mới chấm tiếp chất lượng chi tiết

## 5. Vì sao không nên dùng `pairwise` ở v1?

`Pairwise` nghĩa là đưa hai câu A/B vào rồi hỏi model câu nào tốt hơn.

Nghe có vẻ hợp lý, nhưng research gần đây cho thấy:

- pairwise dễ bị position bias
- dễ bị distractor bias hơn
- dễ bị lật preference khi câu có thêm đặc điểm gây nhiễu

Nói ngắn gọn:

- `pairwise` không ổn định như mình tưởng
- với bài toán safety như của bạn, nó không phải lựa chọn an toàn nhất ở phiên bản đầu

Nguồn:

- https://arxiv.org/abs/2504.14716
- https://aclanthology.org/2025.ijcnlp-long.18.pdf

Vì thế, với v1, nên dùng:

- **pointwise judge**
- tức là mỗi lần chỉ chấm một cặp:
  - `ground_truth`
  - `generation`

## 6. Vì sao không nên khóa trọng số ngay?

Hiện tại nếu viết kiểu:

- safety = 0.35
- direction = 0.25
- hazard = 0.20
- action = 0.15
- fluency = 0.05

thì nhìn có vẻ hợp lý, nhưng vấn đề là:

- chưa có human study đủ kỹ
- chưa có calibration set đủ chắc
- dễ tạo cảm giác “chính xác giả”

Nói đơn giản:

- số đẹp không đồng nghĩa với chắc
- nhất là trong bài toán có lỗi nguy hiểm

Vì vậy, ở giai đoạn này, **không nên ép về weighted average ngay**.

## 7. Thiết kế tốt hơn: `Gate + Rubric`

Đây là hướng tôi khuyên dùng.

### 7.1 Tầng 1: `Safety Gate`

Trước tiên, judge phải kiểm tra xem generation có mắc lỗi nặng không.

Các lỗi nặng nên kiểm:

- `unsafe_action`
  - khuyên người dùng làm điều nguy hiểm
- `wrong_main_direction`
  - nói sai hướng của vật cản/hazard chính
- `hallucinated_main_hazard`
  - bịa ra vật cản chính không có trong GT
- `missed_main_hazard`
  - bỏ sót vật cản chính trong GT
- `polarity_reversal`
  - GT là nguy hiểm nhưng generation lại nói như thể an toàn

Nếu dính các lỗi này, thì câu đó phải bị xếp thấp ngay.

Điểm quan trọng:

- không nên để một câu “nói hay” cứu được một câu “sai an toàn”

### 7.2 Tầng 2: `Rubric`

Nếu câu không mắc lỗi nặng, mới chấm tiếp các tiêu chí mềm hơn:

- `Safety Correctness`
- `Hazard Fidelity`
- `Direction Fidelity`
- `Action Usefulness`
- `Spoken-Guidance Quality`

Tầng này dùng để phân biệt:

- câu đúng nhưng mediocre
- câu đúng và rất hữu ích
- câu đúng nhưng hơi dài / hơi kém tự nhiên

## 8. Sai hướng nên xử lý thế nào?

Đây là điểm rất quan trọng.

Tôi không nghĩ nên chấm kiểu quá máy móc:

- lệch 1 giờ thì trừ ít
- lệch 2 giờ thì trừ nhiều
- lệch 3 giờ thì trừ rất nhiều

Lý do:

- mức độ nguy hiểm không chỉ phụ thuộc khoảng cách số giờ
- nó còn phụ thuộc:
  - hazard có phải là hazard chính không
  - action đi kèm có bị nguy hiểm không
  - câu có đảo ngược bản chất cảnh không

Ví dụ:

- sai `11 o'clock` thành `1 o'clock` với hazard chính: lỗi nặng
- thiếu hướng trong câu `road is clear`: có thể chỉ là lỗi nhẹ hơn

Vì vậy nên chia sai hướng thành các mức:

1. `Exact`
   - đúng hướng chính
2. `Minor omission`
   - thiếu hoặc hơi mơ hồ, nhưng chưa làm đổi ý nghĩa an toàn
3. `Mismatch`
   - sai hướng đáng kể
4. `Unsafe reversal`
   - sai hướng tới mức làm hành động trở nên nguy hiểm

Thiết kế này hợp lý hơn kiểu trừ điểm tuyến tính cứng nhắc.

## 9. GPTScore có giải quyết được vấn đề “tương đồng từ ngữ” không?

Có thể giảm, nhưng **không tự động giải quyết hoàn toàn**.

Research mới cho thấy LLM judge vẫn có overlap bias:

- câu càng giống reference thì dễ được ưu ái hơn
- khi overlap thấp, judge có thể đánh giá kém ổn định hơn

Nguồn:

- https://arxiv.org/abs/2602.07673

Nghĩa là:

- GPTScore tốt hơn ROUGE/TF-IDF ở chỗ hiểu nghĩa tốt hơn
- nhưng nếu thiết kế không cẩn thận, nó vẫn có thể bị bias theo wording

Kết luận:

- GPTScore **có thể giúp**, nhưng không phải phép màu
- thiết kế judge phải chủ động chống lexical bias

## 10. Vậy output của `GPTScore` v1 nên là gì?

Tôi nghĩ output hợp lý nhất lúc này chưa nên là một số thập phân quá chi tiết.

Nên là output có cấu trúc như sau:

```json
{
  "gate": {
    "unsafe_action": false,
    "wrong_main_direction": false,
    "hallucinated_main_hazard": false,
    "missed_main_hazard": false,
    "polarity_reversal": false
  },
  "rubric": {
    "safety_correctness": "strong",
    "hazard_fidelity": "acceptable",
    "direction_fidelity": "strong",
    "action_usefulness": "strong",
    "spoken_guidance_quality": "acceptable"
  },
  "overall_band": "strong",
  "brief_rationale": "..."
}
```

### Vì sao `band` lại hợp lý hơn `score` ở giai đoạn đầu?

Vì:

- dễ đọc
- dễ audit
- ít tạo cảm giác chính xác giả
- hợp với bài toán safety hơn

Ví dụ band:

- `Fail`
- `Weak`
- `Acceptable`
- `Strong`
- `Excellent`

Sau này nếu muốn, vẫn có thể map band sang score.

## 11. Thiết kế đề xuất cuối cùng

Nếu phải chốt hướng thiết kế `GPTScore` hợp lý nhất cho hiện tại, tôi đề xuất:

### Nên dùng

- `pointwise`
- `reference-based`
- `GT vs generation`
- `safety-first`
- `gate + rubric`
- `overall_band`
- `brief_rationale`

### Chưa nên dùng ngay

- `pairwise A/B judge`
- `weighted average` cố định
- `single overall numeric score` làm đầu ra chính
- `text similarity` làm logic cốt lõi
- `image-based judge` ở phiên bản đầu

## 12. Kết luận dễ hiểu nhất

Nếu nói thật ngắn gọn:

> Với task `alter`, GPTScore hợp lý nhất không phải là “câu này giống GT bao nhiêu điểm”, mà là:
> 
> 1. câu này có nguy hiểm không  
> 2. có đúng vật cản và đúng hướng không  
> 3. có đưa ra hành động hữu ích không  
> 4. sau đó mới xét câu có gọn và dễ nghe không

Nên thiết kế tốt nhất lúc này là:

> **Safety gate trước, rubric sau, band cuối cùng**

thay vì:

> **một công thức cộng điểm có trọng số ngay từ đầu**

## 13. Nguồn tham khảo chính

- On the Evaluation of Vision-and-Language Navigation Instructions  
  https://aclanthology.org/2021.eacl-main.111.pdf

- ASSISTER: Assistive Navigation via Conditional Instruction Generation  
  https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136960269.pdf

- LaF-GRPO: In-Situ Navigation Instruction Generation for the Visually Impaired  
  https://ojs.aaai.org/index.php/AAAI/article/download/40804/44765

- Blind to the Human Touch: Overlap Bias in LLM-Based Summary Evaluation  
  https://arxiv.org/abs/2602.07673

- Pairwise or Pointwise? Evaluating Feedback Protocols for Bias in LLM-Based Evaluation  
  https://arxiv.org/abs/2504.14716

- A Systematic Study of Position Bias in LLM-as-a-Judge  
  https://aclanthology.org/2025.ijcnlp-long.18.pdf

- A Survey on LLM-as-a-Judge  
  https://www.sciencedirect.com/science/article/pii/S2666675825004564
