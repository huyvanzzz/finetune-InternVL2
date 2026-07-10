# Thiết Kế `GPTScore` Cho `alter`: Bản Dễ Hiểu, Chặt, Và Dùng Được Để Implement

## 1. Mục tiêu của tài liệu này

Tài liệu này trả lời câu hỏi:

> Nếu muốn xây dựng một `GPTScore` để so sánh `ground truth` và `generation` cho task `alter`, thì nên thiết kế như thế nào cho hợp lý nhất?

Mục tiêu ở đây là chốt:

- nên chấm theo cấu trúc nào
- tiêu chí nào nên có
- trường hợp nào được phép là `N/A`
- khi nào phải dùng `gate`
- `overall` nên được tính ra sao ở bản v1

Tài liệu này **chưa** đi vào prompt judge cụ thể và **chưa** code bộ chấm.

Ghi chú quan trọng:

- `Fail` là mức đánh giá dành cho generation khi so với GT
- `Fail` không phải là loại nhãn có sẵn trong dữ liệu `alter`

## 2. `alter` GT thực chất đang là gì

### 2.1 `alter` đang được lấy như thế nào

`Data-evidenced`

Trong [preprocessing.py](D:/NCKH_VLM/finetune-InternVL2/preprocessing.py) hiện tại:

- `format_ground_truth()` gọi `map_metadata_to_ground_truth()`
- nếu mẫu có `QA` thì instruction lấy từ `QA.A`
- nếu không thì instruction lấy trực tiếp từ `metadata['alter']`
- với `response_format == "direct_text"`, đầu ra cuối cùng là `instruction.strip()`

Ý nghĩa:

- với task `alter`, GT dùng để chấm chính là câu `alter`
- không có lớp hậu xử lý ngữ nghĩa nào chen vào giữa `alter` và text cuối được chấm

### 2.2 `alter` không phải caption mở

`Data-evidenced`

Từ dữ liệu đang dùng, `alter` gần với:

- câu chỉ dẫn điều hướng ngắn
- câu cảnh báo vật cản hoặc nguy hiểm tức thời
- câu nói cho người khiếm thị biết nên làm gì ngay bây giờ

Ví dụ điển hình:

- `There are stairs in front, be careful and walk slowly.`
- `the current road is clear, please move forward without worry.`
- `at 11 o'clock direction, there are pedestrians passing by. be careful to avoid.`

Vì vậy, bản chất bài toán này ưu tiên:

1. đúng về an toàn
2. đúng về hazard / path state
3. đúng về hướng
4. đúng về hành động nên làm

chứ không ưu tiên hàng đầu việc câu có giống từ ngữ với GT hay không.

### 2.3 Dữ liệu cho thấy không phải mẫu nào cũng có đủ mọi tín hiệu

`Data-evidenced`

Từ các lượt audit heuristic bảo thủ trên train `alter` hiện tại:

- tổng số mẫu `alter`: `8571`
- proxy có `direction anchor` đủ rõ: khoảng `76% - 78%`
- proxy có `hazard/path-state` đủ rõ: khoảng `67% - 75%`
- proxy có `action demand` đủ rõ: khoảng `56% - 59%`

Các ví dụ thật cho thấy sự đa dạng này:

- có direction rõ:
  - `at 11 o'clock direction, there are pedestrians passing by. be careful to avoid.`
  - `There are stairs in front, be careful and walk slowly.`
- không có direction đủ rõ:
  - `the current road is narrow, please slow down.`
  - `road is narrow, please pay attention to safety.`
- có action demand rõ:
  - `ahead there are pedestrians gathering in the middle of the road. please slow down and avoid the pedestrians towards 1 o'clock.`
  - `the current road is clear, please move forward without worry.`
- không có action demand đủ rõ:
  - `this is a crossroad.`
  - `the front intersection is available for turning.`
- clear-path / path-state:
  - `road ahead is clear and safe to pass.`
  - `the current direction of the road is unobstructed. please walk without worry.`

Kết luận rất quan trọng:

- không phải GT nào cũng nêu rõ `direction`
- không phải GT nào cũng nêu rõ `hazard/path-state`
- không phải GT nào cũng nêu rõ `action demand`

Vì vậy, một `GPTScore` ép chấm đủ mọi tiêu chí trên mọi mẫu sẽ dễ phạt oan.

## 3. Vì sao không nên chấm theo wording similarity

### 3.1 Vì `alter` có lặp mẫu câu

`Data-evidenced`

Trong dữ liệu train `alter`, exact duplicate không hề ít. Điều này khiến các metric kiểu overlap rất dễ bị “ảo tưởng”:

- câu gần wording GT thì được điểm cao
- nhưng chưa chắc đúng về an toàn

Ví dụ:

- GT: `There are stairs in front, be careful and walk slowly.`
- Gen: `The road is clear, please move forward without worry.`

Câu sinh ra trôi chảy, ngắn gọn, nhưng sai bản chất an toàn.

### 3.2 Vì research về navigation instruction cũng cảnh báo điều này

`Research-supported inference`

Paper *On the Evaluation of Vision-and-Language Navigation Instructions* cho thấy BLEU, ROUGE, METEOR, CIDEr không hiệu quả cho grounded navigation instructions; ở mức hệ thống nếu có reference thì SPICE tốt hơn, còn ở mức từng instruction thì một compatibility model phản ánh human wayfinding tốt hơn nhiều metric text similarity. Nguồn: https://aclanthology.org/2021.eacl-main.111.pdf

Ý nghĩa khi áp sang `alter`:

- similarity bề mặt không đủ cho bài toán điều hướng an toàn
- metric mới phải bám vào semantic usefulness và safety correctness

## 4. Survey: thiết kế judge kiểu nào hợp lý hơn

## 4.1 `pointwise` hay `pairwise`

### Kết luận ngắn

`Design choice for v1`

Ở v1 nên dùng:

- `pointwise`
- mỗi lần chấm một cặp `GT + generation`

không nên dùng `pairwise` làm thiết kế mặc định.

### Vì sao

`Research-supported inference`

Paper *Pairwise or Pointwise? Evaluating Feedback Protocols for Bias in LLM-Based Evaluation* cho thấy choice of feedback protocol ảnh hưởng mạnh tới độ tin cậy; pairwise dễ bị distractor features hơn, và pairwise preferences bị flip khoảng `35%` trong setting của họ, cao hơn nhiều so với absolute scores (`9%`). Nguồn: https://arxiv.org/abs/2504.14716

Paper *Judging the Judges: A Systematic Study of Position Bias in LLM-as-a-Judge* cũng cho thấy pairwise/listwise settings bị position bias rõ rệt. Nguồn: https://aclanthology.org/2025.ijcnlp-long.18.pdf

Áp sang bài toán này:

- ta không cần so A với B để chọn winner
- ta cần biết một generation có ổn so với GT hay không
- vậy `pointwise` tự nhiên hơn và ít mở thêm nguồn bias không cần thiết

## 4.2 `reference-based` hay `reference-free`

### Kết luận ngắn

`Design choice for v1`

Ở v1 nên dùng:

- `reference-based`
- đầu vào chính là `ground_truth` và `generation`

### Vì sao

`Research-supported inference`

Đúng là paper về VLN evaluation gợi ý compatibility model hoặc trajectory-aware evaluation khi có trajectory/image context. Nhưng trong repo hiện tại, mục tiêu trước mắt là có một judge dễ triển khai để so sánh chất lượng generation cuối cùng. Ta chưa có một image-grounded judge đã được validate cho bộ dữ liệu này.

Do đó, lựa chọn an toàn hơn cho v1 là:

- chưa dùng ảnh
- chưa dùng reference-free judge
- dùng `GT vs generation` để chấm semantic correctness

Đây không phải giải pháp hoàn hảo nhất về mặt khoa học, nhưng là giải pháp hợp lý nhất để triển khai sớm và kiểm soát được hành vi của judge.

## 4.3 `numeric score` hay `band/rubric`

### Kết luận ngắn

`Design choice for v1`

Ở v1 nên ưu tiên:

- rubric có cấu trúc
- mức đánh giá theo band hoặc theo thang rời rạc ít mức

thay vì chỉ yêu cầu một số overall duy nhất.

### Vì sao

`Research-supported inference`

Paper *G-Eval* dùng chain-of-thought và form-filling paradigm thay vì hỏi một điểm mơ hồ, cho thấy cấu trúc đánh giá rõ ràng giúp judge align tốt hơn với human judgments. Nguồn: https://aclanthology.org/2023.emnlp-main.153/

Paper *FLASK* nhấn mạnh evaluation fine-grained theo skill-level giúp tăng interpretability và reliability so với coarse overall preference. Nguồn: https://arxiv.org/abs/2307.10928

Áp sang `alter`:

- cần biết model sai ở `hazard`, `direction`, hay `action`
- nếu chỉ ra một số overall, rất khó debug

## 4.4 Có nên khóa trọng số cứng ngay không

### Kết luận ngắn

`Design choice for v1`

Không nên lấy weighted average cứng làm mặc định ở v1.

### Vì sao

`Research-supported inference`

Hiện chưa có:

- human calibration set đủ rõ
- study đủ kỹ để nói `safety=0.35`, `direction=0.25` là đúng
- validation rằng human raters thực sự đồng ý với bộ trọng số đó

Thêm nữa, paper *Validating LLM-as-a-Judge Systems under Rating Indeterminacy* cho thấy nhiều bài toán rating có nhiều diễn giải hợp lệ, và forced single-label choices có thể tạo bias validation rất mạnh. Nguồn: https://papers.nips.cc/paper_files/paper/2025/file/a309239c11a28c597d050bd4a1752d32-Paper-Conference.pdf

Áp sang `alter`:

- nếu chưa có calibration, weighted average cứng tạo cảm giác “chính xác giả”
- tốt hơn nên dùng:
  - criterion-level judgments
  - `N/A` cho tiêu chí không áp dụng
  - mean trên các tiêu chí `applicable`

## 5. Những trường hợp khó mà GPTScore phải xử lý đúng

## 5.1 GT không có hướng thì chấm `Direction` thế nào

### Kết luận

`Design choice for v1`

- không được chấm `0` chỉ vì generation không nói hướng
- `Direction` phải được gắn `N/A` nếu GT không cung cấp đủ tín hiệu định hướng

### Lý do

`Data-evidenced`

Không phải mọi mẫu `alter` đều chứa `o'clock`, `left/right`, `ahead/front`. Có nhiều mẫu là `clear path`, `slow down`, hoặc warning ngắn.

Nếu GT không nêu direction rõ:

- judge không có căn cứ mạnh để yêu cầu generation phải khớp direction
- lúc này tiêu chí `Direction Fidelity` nên là `N/A`

## 5.2 GT không có hazard rõ thì chấm `Hazard` thế nào

### Kết luận

`Design choice for v1`

- nếu GT không nêu hazard cụ thể nhưng nêu path state như `clear` / `unobstructed`, vẫn có thể chấm ở mức `path-state fidelity`
- nếu GT hoàn toàn không cung cấp hazard/path-state đáng tin, `Hazard` nên là `N/A`

### Lý do

GT kiểu:

- `the current road is clear, please move forward without worry.`

thực ra vẫn chứa thông tin semantic quan trọng: trạng thái đường đi đang an toàn/thoáng.

Vì vậy tiêu chí này không nên hiểu hẹp là “có vật cản hay không”, mà nên hiểu là:

- `Hazard / Path-State Fidelity`

## 5.3 GT nói “đường an toàn” thì có được phạt vì không nêu hazard không

### Kết luận

`Design choice for v1`

Không.

Nếu GT là câu kiểu “đường đang clear”, generation không cần tự bịa thêm hazard để được điểm tốt. Ngược lại:

- bịa ra hazard mới có thể là lỗi nặng hơn

Judge nên thưởng cho việc:

- giữ đúng polarity an toàn
- khuyên hành động phù hợp
- không hallucinate vật cản trái ngược với GT

## 5.4 Paraphrase đúng nghĩa nhưng ít giống wording GT thì sao

### Kết luận

`Design choice for v1`

Paraphrase đúng nghĩa vẫn phải được chấm cao.

Ví dụ:

- GT: `the current road is clear, please move forward without worry.`
- Gen: `The path ahead is unobstructed. You can continue forward carefully.`

Nên được chấm tốt dù wording khác.

### Lý do

`Research-supported inference`

LLM judges thường tốt hơn ROUGE ở chỗ hiểu nghĩa và paraphrase, nhưng vẫn có overlap bias. Paper *Blind to the Human Touch* cho thấy judges ngày càng thiên vị output do LLM sinh khi lexical overlap với human response giảm, và simple comparison là chưa đủ. Nguồn: https://arxiv.org/abs/2602.07673

Suy ra:

- GPTScore có thể giảm lexical-overlap bias
- nhưng phải thiết kế prompt/rubric chủ động chống việc “giống từ” được thưởng quá mức

## 5.5 Sai hướng có nên trừ tuyến tính theo “số giờ” không

### Kết luận

`Design choice for v1`

Không nên dùng công thức tuyến tính cứng kiểu:

- lệch 1 giờ trừ ít
- lệch 2 giờ trừ nhiều
- lệch 3 giờ trừ rất nhiều

Ở v1 nên dùng các mức semantic:

- `exact`
- `minor omission`
- `meaningful mismatch`
- `unsafe reversal`

### Lý do

Sai hướng không chỉ là chuyện lệch bao nhiêu “giờ”, mà còn phụ thuộc:

- đó có phải hazard chính không
- câu có kéo theo hành động nguy hiểm không
- lỗi đó có đảo ngược bản chất an toàn hay không

Ví dụ:

- `11 o'clock` thành `1 o'clock` với hazard chính có thể là lỗi nặng
- thiếu hướng trong câu `road is clear` có thể chỉ là lỗi nhẹ hoặc không áp dụng

## 5.6 Hallucination nhỏ và hallucination nguy hiểm khác nhau thế nào

### Kết luận

`Design choice for v1`

Phải tách ít nhất hai mức:

- `minor extra detail`
- `safety-critical hallucination`

Ví dụ:

- thêm một chi tiết phụ không làm đổi hành động: lỗi nhẹ
- bịa ra `clear path` khi GT nói có stairs phía trước: lỗi nặng
- bịa vật cản chính hoặc đảo polarity an toàn: phải đi vào `gate`

## 6. Thiết kế khuyến nghị nhất cho `GPTScore v1`

## 6.1 Kiến trúc tổng thể

`Design choice for v1`

Thiết kế hợp lý nhất ở thời điểm này là:

- `pointwise`
- `reference-based`
- `safety-first`
- `2 tầng: gate trước, rubric sau`
- `overall` tính trên các tiêu chí `applicable`

Nói ngắn gọn:

1. Trước tiên kiểm tra xem generation có lỗi nguy hiểm rõ ràng không
2. Nếu không, mới chấm chi tiết theo rubric
3. Chỉ tổng hợp trên các tiêu chí mà GT thực sự cung cấp tín hiệu

## 6.2 Tầng 1: Safety Gate

`Design choice for v1`

`Gate` nên dùng để bắt các lỗi nghiêm trọng mà không nên được “câu văn trôi chảy” cứu lại.

Các cờ nên có ở v1:

- `polarity_reversal`
  - GT nguy hiểm nhưng generation nói như thể an toàn, hoặc ngược lại
- `unsafe_action`
  - generation khuyên một hành động có khả năng gây nguy hiểm rõ
- `main_hazard_hallucination`
  - bịa ra hazard chính không có trong GT theo cách làm đổi guidance
- `main_hazard_omission`
  - bỏ mất hazard chính khi GT rõ ràng là warning/hazard sample
- `unsafe_direction_reversal`
  - sai hướng theo cách làm guidance đổi bản chất an toàn

Quy tắc:

- nếu một gate nặng bật lên, `overall` phải bị cap mạnh hoặc rơi thẳng về `fail`

## 6.3 Tầng 2: Applicable Rubric

`Design choice for v1`

Sau gate, judge chấm từng tiêu chí sau:

1. `Safety Correctness`
2. `Hazard / Path-State Fidelity`
3. `Direction Fidelity`
4. `Action Usefulness`
5. `Spoken-Guidance Quality`

Mỗi tiêu chí nên có:

- `label`
  - ví dụ: `strong`, `acceptable`, `weak`, `fail`
- `applicable`
  - `true/false`
- `short_rationale`

## 6.4 Tiêu chí nào luôn chấm được, tiêu chí nào có thể `N/A`

`Design choice for v1`

### Gần như luôn applicable

- `Safety Correctness`
- `Spoken-Guidance Quality`

Lý do:

- hầu như câu `alter` nào cũng mang hàm ý hỗ trợ điều hướng
- câu nào cũng có thể đánh giá ở mức “an toàn hay không” và “có phù hợp làm spoken guidance hay không”

### Chỉ chấm khi GT có đủ tín hiệu

- `Direction Fidelity`
  - chỉ applicable nếu GT nêu hướng/vị trí tương đối đủ rõ
- `Hazard / Path-State Fidelity`
  - applicable nếu GT nêu hazard cụ thể hoặc path state như `clear/unobstructed`
- `Action Usefulness`
  - applicable khi GT có action rõ, warning rõ, hoặc path-state/hazard đủ để kỳ vọng một khuyến nghị hành động

Ghi chú:

- ở v1, `Action Usefulness` có thể vẫn khá thường applicable
- nhưng không nên ép mọi câu phải có action nếu GT chỉ mang tính mô tả vị trí ngắn và không thật sự chốt hành vi

## 6.4.1 Rule phân biệt `Safety` và `Hazard / Path-State`

`Research-supported inference` + `Design choice for v1`

Hai tiêu chí này liên quan chặt, nhưng không nên gộp làm một:

- `Hazard / Path-State Fidelity` hỏi:
  - generation có mô tả đúng vật cản, rủi ro, hoặc trạng thái đường đi mà GT nói tới không?
- `Safety Correctness` hỏi:
  - generation có dẫn tới thông điệp an toàn đúng chiều không?

Rule thực dụng cho v1:

1. Sai `hazard/path-state` nhưng chưa làm đổi hành động an toàn:
   - phạt mạnh ở `Hazard / Path-State`
   - `Safety` có thể chỉ giảm vừa phải
2. Sai `hazard/path-state` và kéo theo guidance nguy hiểm:
   - phạt mạnh ở cả `Hazard / Path-State` và `Safety`
   - có thể bật `gate`
3. Đúng `hazard/path-state` nhưng action quá yếu hoặc thiếu cảnh báo:
   - `Hazard / Path-State` vẫn có thể cao
   - `Safety` không được cao tương ứng
4. Với mẫu `clear/unobstructed`:
   - `Hazard / Path-State` vẫn applicable vì GT đã cung cấp path state
   - nếu generation bịa ra obstacle làm đổi bản chất scene thì đây vừa là lỗi hazard vừa có thể kéo theo lỗi safety

Nói ngắn:

- `Hazard` đánh giá “nhìn tình huống có đúng không”
- `Safety` đánh giá “nói ra có an toàn không”

Sanity-check ngắn:

- GT: `There are stairs in front, be careful and walk slowly.`
- Gen: `There are stairs ahead.`  
  -> `Hazard / Path-State` có thể cao vì vẫn nhận ra stairs, nhưng `Safety` không nên cao tương ứng vì guidance còn thiếu phần cảnh báo/hành động.

- GT: `the current road is clear, please move forward without worry.`
- Gen: `There is a car in front. Stop immediately.`  
  -> `Hazard / Path-State` fail vì bịa sai scene, và `Safety` cũng giảm mạnh vì generation kéo người dùng sang một guidance khác hẳn GT.

## 6.4.2 Rule `Direction` applicable

`Data-evidenced` + `Design choice for v1`

`Direction Fidelity` chỉ applicable khi GT cung cấp một neo định hướng đủ rõ.

Các tín hiệu đủ để coi là applicable:

- `o'clock`
- `left / right`
- `ahead / front / straight ahead`
- mô tả vị trí tương đối có vai trò thật trong guidance

Các case nên coi là `N/A`:

- GT chỉ nói `road is clear`
- GT chỉ nói `slow down`
- GT chỉ nêu hazard chung chung nhưng không gắn với vị trí/hướng

Rule chốt:

- có neo định hướng rõ trong GT -> `Direction` applicable
- không có neo định hướng rõ -> `Direction = N/A`

Không nên ép model phải “bịa thêm direction” nếu GT không có direction.

Rule rất quan trọng:

- `Direction = N/A` không có nghĩa là generation được phép tự bịa thêm hướng nguy hiểm mà không bị phạt.
- Nếu GT không có direction anchor rõ, nhưng generation tự thêm direction làm guidance lệch theo hướng rủi ro, thì lỗi đó vẫn phải bị phạt qua `Safety`, và nếu đủ nặng thì bật `unsafe_direction_reversal`.

Ví dụ:

- `at 11 o'clock direction, there are pedestrians passing by. be careful to avoid.`  
  -> `Direction` applicable.

- `the current road is narrow, please slow down.`  
  -> `Direction = N/A`, vì GT chưa chỉ ra trái/phải/trước/sau hay `o'clock`.

- `the front intersection is available for turning.`  
  -> đây là case biên. Nếu judge hiểu `front intersection` là anchor thật sự tác động đến guidance, có thể coi là applicable; nếu chỉ là mô tả vị trí lỏng, nên để `N/A`. Với v1, nên ưu tiên bảo thủ: chỉ tính applicable khi anchor đủ rõ để so đúng/sai hướng có ý nghĩa.

- GT: `the current road is narrow, please slow down.`
- Gen: `Turn right and move quickly.`  
  -> `Direction` có thể vẫn là `N/A` theo nghĩa không chấm fidelity trực tiếp với GT, nhưng generation vẫn phải bị phạt nặng ở `Safety`, và có thể bật `unsafe_direction_reversal`.

## 6.4.3 Rule `Action` applicable

`Research-supported inference` + `Design choice for v1`

`Action Usefulness` không nên mặc định applicable trên mọi mẫu.

Nó applicable khi GT làm ít nhất một trong các việc sau:

- khuyên hành động trực tiếp
  - `slow down`, `avoid`, `move forward`, `walk slowly`
- cảnh báo nguy cơ đủ mạnh để người dùng cần một hành động tương ứng
- mô tả `clear/safe path` theo cách hàm ý một hành động hợp lệ như tiếp tục đi

Nó có thể là `N/A` khi:

- GT chỉ là mô tả vị trí ngắn, không thực sự đưa ra hay hàm ý hành động
- GT chỉ nêu direction hoặc object presence mà chưa tạo thành guidance step

Rule chốt:

1. GT có imperative/action cue rõ -> `Action` applicable
2. GT không có imperative nhưng có hazard/path-state đủ rõ để suy ra bước nên làm -> `Action` vẫn applicable
3. GT chỉ mô tả scene/vị trí mà chưa thành guidance step -> `Action = N/A`

Điểm quan trọng:

- `Action` không phải tiêu chí “generation có động từ hay không”
- `Action` là tiêu chí “generation có đưa ra bước hành động hữu ích khi GT thật sự đòi hỏi điều đó hay không”

Rule rất quan trọng:

- `Action = N/A` không có nghĩa là generation được phép tự thêm một hành động nguy hiểm rồi thoát phạt.
- Nếu GT không thật sự đòi action, nhưng generation lại khuyên một bước đi sai chiều an toàn, thì lỗi đó vẫn phải bị phạt qua `Safety`, và nếu đủ nặng thì bật `unsafe_action`.

Ví dụ:

- `the current road is clear, please move forward without worry.`  
  -> `Action` applicable, vì GT đã chốt một bước nên làm là tiếp tục đi.

- `There are stairs in front, be careful and walk slowly.`  
  -> `Action` applicable, vì GT vừa cảnh báo vừa yêu cầu hành động cụ thể.

- `this is a crossroad.`  
  -> `Action = N/A`, vì GT chưa đủ để đòi một bước hành động cụ thể.

- `the front intersection is available for turning.`  
  -> thường nên để `Action = N/A` ở v1, trừ khi ngữ cảnh GT thực sự đã biến nó thành một bước guidance.

- GT: `this is a crossroad.`
- Gen: `Run across quickly.`  
  -> `Action` có thể vẫn là `N/A` theo nghĩa GT không đủ để chấm usefulness trực tiếp, nhưng generation vẫn phải bị phạt nặng ở `Safety`, và có thể bật `unsafe_action`.

## 6.5 `overall` nên tính thế nào

`Design choice for v1`

Khuyến nghị v1:

- **không** dùng weighted average cứng làm mặc định
- **có** dùng mean trên các tiêu chí `applicable`
- `gate` có quyền cap hoặc fail kết quả cuối

Một cách gọn:

1. Đổi các label criterion sang thang số nội bộ, ví dụ:
   - `fail = 0`
   - `weak = 1`
   - `acceptable = 2`
   - `strong = 3`
2. Tính mean trên các tiêu chí `applicable`
3. Áp rule gate:
   - nếu có `unsafe_action` hoặc `polarity_reversal`, `overall_score = 0.0`
   - nếu có `unsafe_direction_reversal` nhưng chưa rơi vào hai case trên, `overall_score` tối đa là `1.0`
4. Xuất thẳng `overall_score`; không cần sinh thêm `overall_band` ở v1

Điểm quan trọng:

- số nội bộ chỉ là công cụ tổng hợp
- đầu ra chính nên là `overall_score + rationale + criterion breakdown`
- `N/A` chỉ loại một criterion khỏi phép lấy trung bình; nó không xóa lỗi safety nếu generation tự hallucinate direction/action nguy hiểm

## 7. Spec đầu vào/đầu ra cho bộ chấm v1

## 7.1 Đầu vào tối thiểu

`Design choice for v1`

V1 chỉ cần:

- `ground_truth`
- `generation`

`question/context` là optional:

- chưa bắt buộc ở v1
- chỉ thêm nếu sau này thử nghiệm cho thấy judge cần để phân biệt vài case mơ hồ

## 7.2 Đầu ra tối thiểu

`Design choice for v1`

Đầu ra khuyến nghị:

```json
{
  "gate": {
    "polarity_reversal": false,
    "unsafe_action": false,
    "main_hazard_hallucination": false,
    "main_hazard_omission": false,
    "unsafe_direction_reversal": false
  },
  "criteria": {
    "safety_correctness": {
      "applicable": true,
      "label": "strong",
      "rationale": "..."
    },
    "hazard_path_state_fidelity": {
      "applicable": true,
      "label": "acceptable",
      "rationale": "..."
    },
    "direction_fidelity": {
      "applicable": false,
      "label": null,
      "rationale": "GT không nêu hướng rõ."
    },
    "action_usefulness": {
      "applicable": true,
      "label": "strong",
      "rationale": "..."
    },
    "spoken_guidance_quality": {
      "applicable": true,
      "label": "acceptable",
      "rationale": "..."
    }
  },
  "overall_score": 2.25,
  "overall_reason": "..."
}
```

## 8. Những gì nên chốt mạnh ở v1, nên để v2, và chưa nên làm

## 8.1 Khuyến nghị mạnh cho v1

`Design choice for v1`

- `pointwise`
- `reference-based`
- `gate + rubric`
- criterion-level `applicable / N/A`
- `overall_score` là mean trên criteria applicable, sau đó chịu cap bởi gate
- output chính là `score + breakdown + rationale`

## 8.2 Có thể thêm ở v2

`Design choice for v1`

- dùng `question/context` như input phụ nếu judge cần thêm disambiguation
- calibrate trọng số bằng human study nhỏ
- thêm confidence score
- thêm self-consistency / multi-pass judging
- thêm image-based judge hoặc grounded judge nếu có hạ tầng phù hợp

## 8.3 Chưa nên làm ngay

`Design choice for v1`

- khóa trọng số cứng
- dùng một overall numeric score duy nhất mà không có criterion breakdown
- dùng pairwise A/B judge làm metric mặc định
- ép mọi tiêu chí phải chấm trên mọi mẫu
- phạt direction theo công thức tuyến tính cứng theo “số giờ”

## 9. Kết luận cuối cùng

Nếu phải chốt một hướng thiết kế `GPTScore` hợp lý nhất cho `alter` ở thời điểm hiện tại, thì hướng nên chọn là:

- chấm từng cặp `GT + generation`
- ưu tiên safety hơn wording similarity
- dùng `gate` để bắt lỗi nguy hiểm
- dùng rubric theo tiêu chí semantic
- cho phép `N/A` ở các tiêu chí mà GT không cung cấp tín hiệu
- tính `overall_score` bằng trung bình trên các tiêu chí thật sự applicable, không dùng weighted average cứng mặc định

Nói thật ngắn:

> Với `alter`, GPTScore tốt không phải là “câu này giống GT bao nhiêu điểm”, mà là:
> 
> 1. câu này có sai nguy hiểm không  
> 2. nếu không sai nguy hiểm, nó có đúng hazard / direction / action ở những phần GT thực sự nói tới không  
> 3. sau đó mới xét câu có gọn, tự nhiên, và hữu ích hay không

## 10. Nguồn chính đã dùng

- On the Evaluation of Vision-and-Language Navigation Instructions  
  https://aclanthology.org/2021.eacl-main.111.pdf

- ASSISTER: Assistive Navigation via Conditional Instruction Generation  
  https://eshed1.github.io/papers/assister_eccv2022.pdf

- LaF-GRPO: In-Situ Navigation Instruction Generation for the Visually Impaired via GRPO with LLM-as-Follower Reward  
  https://ojs.aaai.org/index.php/AAAI/article/download/40804/44765

- G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment  
  https://aclanthology.org/2023.emnlp-main.153/

- FLASK: Fine-grained Language Model Evaluation based on Alignment Skill Sets  
  https://arxiv.org/abs/2307.10928

- Pairwise or Pointwise? Evaluating Feedback Protocols for Bias in LLM-Based Evaluation  
  https://arxiv.org/abs/2504.14716

- Judging the Judges: A Systematic Study of Position Bias in LLM-as-a-Judge  
  https://aclanthology.org/2025.ijcnlp-long.18.pdf

- Blind to the Human Touch: Overlap Bias in LLM-Based Summary Evaluation  
  https://arxiv.org/abs/2602.07673

- Validating LLM-as-a-Judge Systems under Rating Indeterminacy  
  https://papers.nips.cc/paper_files/paper/2025/file/a309239c11a28c597d050bd4a1752d32-Paper-Conference.pdf
