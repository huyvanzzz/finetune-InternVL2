# Nghiên Cứu Sửa Prompt GPTScore `alter`: Giảm Overfire Ở Direction Và Siết `Action Usefulness`

## 1. Mục tiêu của tài liệu này

Tài liệu này chốt lại những gì cần sửa trong **prompt judge GPTScore cho `alter`** để tránh hai vấn đề đã lộ ra khi chạy thật:

- prompt đang có xu hướng **phạt direction quá cứng**
- prompt đang có xu hướng **coi `Action Usefulness` applicable quá rộng**

Trọng tâm của đợt này không phải sửa code pipeline, không phải sửa schema, và không phải đổi 3 gate hiện có. Mục tiêu là trả lời:

1. Khi nào direction sai chỉ là lỗi fidelity?
2. Khi nào direction sai mới đủ để bật `unsafe_direction_reversal`?
3. Vì sao `10h` và `11h` không nên bị đối xử giống `10h` và `3h`?
4. `Action Usefulness` nên applicable theo rule nào để không kéo điểm xuống oan?

## 2. Nền bằng chứng đã dùng

### 2.1 Prompt production hiện tại

`Code-evidenced`

Prompt runtime hiện tại nằm ở:

- [.worktrees/restore-779cc7b/gptscore/prompts/gptscore_alter_system_prompt.txt](D:/NCKH_VLM/finetune-InternVL2/.worktrees/restore-779cc7b/gptscore/prompts/gptscore_alter_system_prompt.txt)

Các rule quan trọng đang có:

- `Direction Fidelity` chỉ applicable khi GT có direction anchor rõ
- `unsafe_direction_reversal` được mô tả là:
  - generation introduces or changes direction information in a way that creates a clear safety-critical risk
- `Action Usefulness` applicable khi GT có action demand thực sự

Về mặt wording, prompt hiện tại **đã có ý đúng**, nhưng chưa đủ rõ để GPT-4o luôn phân biệt được:

- lệch hướng nhẹ
- lệch hướng có ý nghĩa
- lỗi hướng thực sự nguy hiểm

### 2.2 Luật tính điểm hiện tại của pipeline

`Code-evidenced`

Pipeline scoring nằm ở:

- [.worktrees/restore-779cc7b/gptscore/scoring.py](D:/NCKH_VLM/finetune-InternVL2/.worktrees/restore-779cc7b/gptscore/scoring.py)

Rule hiện tại:

- `polarity_reversal` -> `overall_score = 0.0`
- `unsafe_action` -> `overall_score = 0.0`
- `unsafe_direction_reversal` -> `overall_score = min(mean_before_gate, 1.0)`

Vì vậy nếu prompt bật `unsafe_direction_reversal` hơi rộng, điểm sẽ tụt rất mạnh ngay cả khi generation chỉ sai một phần.

### 2.3 Bằng chứng từ các file judged/scored đã chạy

`Runtime-confirmed`

Đã dùng hai nhóm kết quả thật:

- file judged 20 mẫu với `gpt-4o-mini`
- file scored 50 mẫu với `gpt-4o`

Từ file 50 mẫu:

- `unsafe_direction_reversal = 19/50`
- `polarity_reversal = 7/50`
- `unsafe_action = 7/50`
- `direction_fidelity = Fail` xuất hiện `24/50`
- `has_action_demand = true` xuất hiện `48/50`

Hai tín hiệu mạnh nhất là:

1. `unsafe_direction_reversal` đang bật khá nhiều
2. `has_action_demand` đang gần như bật cho mọi mẫu

Đây là dấu hiệu prompt hiện tại còn đang **quá rộng ở decision boundary**.

## 3. Các pattern chấm quá cứng đã quan sát

## 3.1 Case lệch một nấc đồng hồ nhưng bị cap mạnh

`Runtime-confirmed`

Ví dụ thật:

- GT: `at ten o'clock direction, there are passers-by staying. please pay attention to safety.`
- Gen: `at 11 o'clock direction, there are pedestrians passing by, pay attention to avoid.`

Judge hiện tại:

- `direction_fidelity = Fail`
- `unsafe_direction_reversal = true`
- `overall_score = 1.0` do bị cap

Nhận xét:

- Đây **không phải case đúng**
- nhưng cũng **chưa đủ bằng chứng để mặc định coi là safety-critical** theo cùng mức với các case kiểu:
  - GT có obstacle ở `1h`, gen bảo obstacle ở `11h`
  - GT bảo rẽ `3h`, gen đổi sang hướng khác hẳn

Kết luận:

- `10h -> 11h` là một **direction mismatch có ý nghĩa**
- nhưng **không nên tự động là gate**

## 3.2 Case bỏ hẳn direction và đảo safety thì gate là hợp lý

`Runtime-confirmed`

Ví dụ thật:

- GT: `at one o'clock direction, there are pedestrians taking pictures. beware of avoiding them.`
- Gen: `the current road is unobstructed. please move forward without worry.`

Judge hiện tại:

- `polarity_reversal = true`
- `unsafe_action = true`
- `unsafe_direction_reversal = true`
- `overall_score = 0.0`

Nhận xét:

- Ở đây lỗi chính là:
  - xóa hazard
  - đảo safety
  - khuyên hành động nguy hiểm
- direction sai không phải lỗi chính, nhưng vẫn có thể đồng xuất hiện với gate khác

Kết luận:

- Case này bị điểm rất thấp là hợp lý
- nhưng không nên lấy case này để suy ra rằng **mọi direction mismatch đều đáng cap**

## 3.3 Case GT không có direction nhưng generation tự bịa hướng

`Runtime-confirmed`

Ví dụ thật:

- GT: `there is a crossroads ahead. a pedestrian passes about five steps ahead. be careful to avoid.`
- Gen: `at 11 o'clock direction, there are pedestrians passing by, pay attention to avoid.`

Judge hiện tại:

- `unsafe_direction_reversal = true`
- `overall_score = 0.6`

Nhận xét:

- GT không có `o'clock` direction
- generation tự thêm direction mới
- đây **có thể là lỗi nghiêm trọng hơn lỗi fidelity thông thường**, nhưng chưa phải lúc nào cũng là gate

Kết luận:

- cần tách rõ:
  - unsupported direction hallucination nhưng chưa làm đổi safety outcome
  - unsupported direction hallucination làm guidance trở nên nguy hiểm rõ ràng

## 3.4 `Action Usefulness` đang bị applicable quá rộng

`Runtime-confirmed`

Trong file scored 50 mẫu:

- `has_action_demand = true` ở `48/50` mẫu

Điều này quá rộng so với ý nghĩa ban đầu của tiêu chí.

Ví dụ ít tranh cãi:

- GT: `the current direction of the road is clear, please walk without worry.`
  - `Action` applicable là hợp lý
- GT: `at 11 o'clock direction, there is a fork in the road where one can go straight and turn left.`
  - `Action` không applicable là hợp lý
- GT: `at 1 o'clock there is a sign to the bus stop.`
  - `Action` không applicable là hợp lý

Nhận xét:

- prompt hiện tại nói “clear/safe path” có thể tạo `light action demand`
- rule này hợp lý
- nhưng judge đang có xu hướng mở rộng thêm sang nhiều case chỉ là hazard/location description

Kết luận:

- wording của `Action Usefulness` cần siết lại
- không thể để judge hiểu theo kiểu “cứ có hazard/path-state là gần như luôn applicable”

## 4. Taxonomy direction error nên dùng cho prompt mới

## 4.1 Mức 1: Minor directional deviation

`Design choice for prompt revision`

Đây là các case:

- lệch một nấc đồng hồ liền kề như `10h -> 11h`, `1h -> 2h`
- đổi diễn đạt direction nhưng vẫn gần cùng vùng không gian
- direction không giữ hoàn toàn chính xác, nhưng chưa có bằng chứng rõ là làm đổi hành động an toàn

Nên xử lý:

- trừ ở `Direction Fidelity`
- có thể kéo `Safety Correctness` xuống nhẹ nếu ngữ cảnh nhạy
- **không tự động bật** `unsafe_direction_reversal`

## 4.2 Mức 2: Meaningful directional mismatch

`Design choice for prompt revision`

Đây là các case:

- direction sai đáng kể
- bỏ mất direction anchor chính
- đổi từ một vùng sang vùng khác làm guidance giảm độ tin cậy rõ rệt

Nên xử lý:

- `Direction Fidelity = Fail` hoặc `Weak` tùy case
- `Safety Correctness` có thể giảm
- **chưa chắc là gate**

Gate chỉ nên bật nếu direction error này kéo theo hậu quả an toàn rõ ràng.

## 4.3 Mức 3: Safety-critical directional error

`Design choice for prompt revision`

Đây là các case:

- direction sai làm đổi hành động an toàn
- direction sai đẩy người dùng sang vùng nguy hiểm
- GT không có direction nhưng generation tự bịa direction theo cách làm guidance nguy hiểm
- direction sai đi kèm hazard/action khiến người dùng nhiều khả năng đi sai đường theo nghĩa nguy hiểm

Nên xử lý:

- `unsafe_direction_reversal = true`
- `Direction Fidelity = Fail`
- `Safety Correctness` không được cao

## 5. Boundary mới giữa `Direction Fidelity` và `Safety Correctness`

## 5.1 Khi nào direction sai chỉ là lỗi fidelity

`Design choice for prompt revision`

Direction sai chỉ nên dừng ở mức fidelity khi:

- generation vẫn giữ đúng safety intent chính
- action safety không bị đảo
- sai direction chưa đủ làm đổi kết luận điều hướng theo nghĩa nguy hiểm

Ví dụ:

- GT bảo chú ý an toàn ở `10h`
- gen nói `11h`
- vẫn còn giữ warning, vẫn không trấn an sai, vẫn không khuyên hành động nguy hiểm

Trường hợp này:

- `Direction Fidelity` giảm mạnh là hợp lý
- nhưng `unsafe_direction_reversal` chưa nên tự động bật

## 5.2 Khi nào direction sai bắt đầu ảnh hưởng `Safety Correctness`

`Design choice for prompt revision`

Direction sai nên làm giảm `Safety Correctness` khi:

- người dùng có thể hiểu sai vùng cần tránh
- generation vẫn “nghe có vẻ an toàn”, nhưng direction shift làm guidance kém an toàn hơn

Tức là:

- direction không còn là lỗi thuần mô tả
- nó bắt đầu ăn vào ý nghĩa an toàn

## 5.3 Khi nào direction sai mới đủ để bật gate

`Design choice for prompt revision`

Chỉ bật `unsafe_direction_reversal` khi có đủ hai điều kiện:

1. direction information bị đổi hoặc bị bịa theo cách đáng kể
2. lỗi đó tạo ra **rủi ro safety-critical rõ ràng**

Nói ngắn:

- direction sai **không đủ**
- phải là direction sai **và** làm guidance nguy hiểm rõ ràng

Đây là boundary quan trọng nhất cần thêm vào prompt.

## 6. Rule mới đề xuất cho `Action Usefulness`

## 6.1 Rule hiện tại cần siết ở đâu

`Design choice for prompt revision`

Rule hiện tại đúng ở ý tưởng, nhưng judge đang đọc hơi rộng.

Prompt mới nên nói rõ hơn:

- hazard-only description **không tự động** tạo action demand
- path-state chỉ tạo `light action demand` khi nó thực sự:
  - permits continuing forward
  - hoặc clearly recommends the next step
- warning chung kiểu `pay attention to safety` không phải lúc nào cũng tương đương với một action demand mạnh

## 6.2 Boundary khả dụng nên dùng

`Design choice for prompt revision`

`Action Usefulness` applicable khi GT:

- có imperative rõ
- có safe-to-proceed statement rõ
- có warning/hazard đủ rõ để người đọc kỳ vọng một next-step guidance thực sự

`Action Usefulness` không nên applicable khi GT:

- chỉ mô tả location/object
- chỉ thông báo cấu trúc không gian
- chỉ nêu direction anchor mà chưa thành guidance step

Ví dụ:

- `at 1 o'clock there is a sign to the bus stop`
  - `Action = N/A`
- `there is a fork in the road where one can go straight and turn left`
  - mặc định nên `Action = N/A`
- `the current road is clear, please walk without worry`
  - `Action` applicable

## 7. So sánh 3 hướng sửa prompt

## 7.1 Hướng A: Sửa tối thiểu

- chỉ thêm 1-2 câu rằng direction lệch nhẹ không tự động là gate

Ưu điểm:

- ít sửa

Nhược điểm:

- quá yếu
- dễ còn ambiguity cũ

Kết luận:

- không nên chọn làm hướng chính

## 7.2 Hướng B: Thêm taxonomy direction vào phần criterion/gate

- bổ sung prompt với 3 mức:
  - minor
  - meaningful
  - safety-critical

Ưu điểm:

- giải quyết đúng ambiguity cốt lõi
- không cần đổi schema

Nhược điểm:

- prompt dài hơn một chút

Kết luận:

- đây là hướng khuyến nghị chính

## 7.3 Hướng C: Thêm procedure riêng cho direction

- ép judge phải quyết định trước:
  - minor deviation
  - meaningful mismatch
  - safety-critical error

Ưu điểm:

- giảm overfire tốt hơn

Nhược điểm:

- procedural hơn
- prompt dài hơn

Kết luận:

- có thể kết hợp nhẹ với hướng B
- không cần thêm field JSON mới

## 8. Prompt revision spec cần chốt

## 8.1 Những câu cần thêm

`Design choice for prompt revision`

Trong phần `Direction Fidelity` nên thêm các ý sau:

- not every direction mismatch is safety-critical
- small directional deviations may still be non-critical if they do not change the safety-relevant action or conclusion
- treat `unsafe_direction_reversal` only as a gate for direction errors that create a clear safety-critical risk, not for every incorrect anchor

Trong phần gate nên thêm:

- a nearby clock-direction drift such as one adjacent clock step is not by itself sufficient for `unsafe_direction_reversal`
- unsupported direction hallucination should trigger the gate only when it materially changes the safe guidance or creates a clear navigation risk

Trong phần `Action Usefulness` nên thêm:

- do not mark this criterion applicable merely because the ground truth mentions an object, obstacle, or scene structure
- warnings or path descriptions count only when they clearly imply a next-step guidance expectation

## 8.2 Những câu nên bỏ hoặc thay

`Design choice for prompt revision`

Không nên để wording mơ hồ kiểu:

- direction changes that could mislead the user

nếu không nói rõ mức độ nào mới là gate, vì `could mislead` quá rộng.

Thay vào đó nên dùng wording kiểu:

- direction changes that create a clear safety-critical navigation risk

## 9. Bộ sanity cases bắt buộc để review prompt bằng tay

## 9.1 Direction-focused cases

1. `10h -> 11h`, warning vẫn giữ, action không đảo
   - `Direction Fidelity` giảm
   - không auto gate

2. `1h -> 11h`, hazard chính bị dời sang vùng khác rõ rệt
   - có thể rất nặng
   - có thể gate nếu safety effect rõ

3. GT không có direction, generation tự thêm `11h`, nhưng safety intent chưa thành nguy hiểm rõ
   - direction hallucination nặng
   - chưa chắc gate

4. GT không có direction, generation tự thêm direction và khuyên đi theo hướng đó
   - có thể gate

## 9.2 Action-focused cases

5. GT chỉ mô tả location, không phải guidance step
   - `Action = N/A`

6. GT clear-path, gen vẫn giữ safe-to-proceed
   - `Action` applicable

7. GT hazard-only, gen có action nhưng không đúng hazard
   - `Action` không nên cứu được lỗi hazard/safety

## 10. Kết luận cuối

Kết luận quan trọng nhất của đợt research này là:

- prompt hiện tại không sai ở ý tưởng tổng thể
- nhưng decision boundary cho `unsafe_direction_reversal` còn quá mơ hồ
- vì vậy judge đang có xu hướng **overfire gate ở các case direction chưa chắc safety-critical**
- đồng thời `Action Usefulness` cũng đang bị applicable quá rộng

Hướng sửa đúng không phải là “nới tay chung chung”, mà là:

1. thêm taxonomy direction rõ hơn
2. tách mạnh `direction wrong` khỏi `dangerous direction`
3. siết lại `Action Usefulness` để không gần như luôn applicable

Nếu chỉ được chốt một nguyên tắc sửa prompt cho vòng tới, nguyên tắc đó nên là:

> Sai direction chỉ nên bật `unsafe_direction_reversal` khi lỗi hướng làm đổi guidance theo nghĩa an toàn một cách rõ ràng; lệch hướng nhẹ như `10h -> 11h` nên bị xử lý trước ở `Direction Fidelity`, không nên tự động bị cap toàn cục.
