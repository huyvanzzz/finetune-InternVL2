# Phân Tích: Vì Sao `alter-only` Xuất Hiện `test_alter` Degradation Sớm, Và Vì Sao `QA` Ảnh Hưởng Khác Nhau Giữa Q-Former / no-QFormer

## 1. Câu hỏi nghiên cứu
Hiện tượng cần giải thích trong repo này là:

- `alter-only` có dấu hiệu `test_alter` xấu đi rất sớm, thường từ epoch 1, ở cả nhánh Q-Former và no-QFormer, dù `val_alter` vẫn có thể tiếp tục giảm.
- Khi trộn `QA + alter`, nhánh Q-Former có vẻ bớt bị `test_alter` degradation sớm hơn, dù giai đoạn đầu có thể học chậm hơn.
- no-QFormer không hưởng lợi tương tự từ `QA`, và về kết quả cuối thường vẫn hợp với trực giác `alter-only` hơn.
- Một nghi vấn bổ sung là prompt `alter` hiện tại chỉ có một template cố định cho toàn bộ task, có thể làm tăng shortcut learning hoặc sensitivity với wording.

**Quy tắc diễn giải trong tài liệu này**

- `val_alter` nên được hiểu là tín hiệu `validation fit` trên phân phối gần train.
- `test_alter` nên được hiểu là tín hiệu `generalization` ngoài phân phối gần train hơn.
- Khi `val_alter` và `test_alter` đi khác hướng, không nên gọi hiện tượng đó là `overfit cổ điển` một cách mặc định; cần ưu tiên diễn giải dưới góc `test degradation` hoặc `generalization gap`.

**Repo/Data Evidence**

- `test_alter` không overlap ảnh/video/summary, nhưng có `25.32%` exact overlap ở trường `alter` text.

## 2. Sự thật đã xác minh từ repo

### 2.1 Prompt train thực tế
Trong [wad_dataset.py](../../wad_dataset.py):

- Với `QA`, prompt train `direct_text` là:
  - `Based on this image, answer the following question for a visually impaired user directly in natural language.`
  - rồi nối thêm `Question: ...`
- Với `alter`, prompt train `direct_text` là:
  - `Describe the scene for a visually impaired user based on this image.`
  - `Focus on immediate obstacles, safe direction, and what action the user should take.`
  - `Provide only the final spoken guidance in natural language.`

**Repo/Data Evidence**

- `QA` và `alter` đang dùng hai instruction patterns tách biệt rõ, nhưng mỗi task chỉ có đúng một template chính trong code.

### 2.2 Ground truth đang được map như thế nào
Trong [preprocessing.py](../../preprocessing.py):

- Nếu sample có `QA`, `instruction = metadata['QA']['A']`
- Nếu không có `QA` mà có `alter`, `instruction = metadata['alter']`
- Với `response_format = direct_text`, target cuối cùng chỉ là `instruction`

**Repo/Data Evidence**

- `QA` và `alter` khác nhau không chỉ ở prompt đầu vào, mà còn ở phân phối target text đầu ra.

### 2.3 Q-Former khác no-QFormer ở đâu
Trong [train.py](../../train.py) và [qformer_bridge.py](../../qformer_bridge.py):

- Với Q-Former:
  - `qformer_text` được encode qua `encode_qformer_texts(...)`
  - rồi được đưa vào `set_qformer_text(...)`
  - nghĩa là instruction không chỉ đi vào LLM prompt, mà còn đi vào khâu trích visual feature.
- Với no-QFormer:
  - không có `attach_qformer_bridge(...)`
  - instruction chủ yếu chỉ tác động qua prompt phía LLM.

**Repo/Data Evidence**

- Cùng một prompt cố định, Q-Former sẽ nhạy hơn với wording vì wording chạm cả visual querying lẫn generation.
- no-QFormer chủ yếu chịu ảnh hưởng ở response mode / language side.

## 3. Sự thật đã xác minh từ data

Thống kê trên `train.json` của dataset `minhdang0901/WAD_Images_All_Size`:

- Tổng train sample: `10124`
- `QA`: `1553`
- `alter`: `8571`

### 3.1 Độ dài text

- `QA question`
  - mean: `5.51` từ
  - median: `4`
  - p90: `9`
  - max: `29`
- `QA answer`
  - mean: `24.64` từ
  - median: `23`
  - p90: `41`
  - max: `85`
- `alter`
  - mean: `14.99` từ
  - median: `13`
  - p90: `24`
  - max: `49`

### 3.2 Mức độ lặp

- `QA question`
  - unique: `384`
  - unique ratio: `0.2473`
  - top 10 share: `57.57%`
- `QA answer`
  - unique: `1533`
  - unique ratio: `0.9871`
  - top 10 share: `1.61%`
- `alter`
  - unique: `6814`
  - unique ratio: `0.7950`
  - top 10 share: `6.02%`
  - top 50 share: `11.45%`
  - top 100 share: `14.47%`

Khi normalize lỏng hơn cho `alter` bằng cách bỏ khác biệt dấu câu:

- unique ratio giảm còn `0.7648`
- top 10 share tăng lên `6.98%`
- top 50 share tăng lên `12.93%`

### 3.3 Ví dụ lặp nhiều nhất

`QA question` lặp nhiều:

- `describe the current scene` xuất hiện `512` lần
- `which direction is the main route?` xuất hiện `88` lần
- `describe the current road situation` xuất hiện `55` lần

`alter` lặp nhiều:

- `at 11 o'clock direction, there are pedestrians passing by, pay attention to avoid.` xuất hiện `101` lần
- `at 11 o'clock direction, there are pedestrians passing by, be careful to avoid.` xuất hiện `78` lần
- `the current road is clear, please move forward without worry.` xuất hiện `75` lần

### 3.4 Mẫu representative

`QA`

- `current road condition` -> `there are steps about five steps ahead, be careful.`
- `can you tell me what is in front?` -> `there is a billboard with a car advertisement in front of you.`

`alter`

- `ahead there are pedestrians gathering in the middle of the road. please slow down and avoid the pedestrians towards 1 o'clock.`
- `There are stairs in front, be careful and walk slowly.`
- `the current road is narrow, please slow down.`

**Repo/Data Evidence**

- `alter` ngắn hơn `QA answer` rõ rệt.
- `alter` có mật độ template lặp đáng kể ở phần output.
- `QA question` thì lặp rất mạnh ở đầu vào, nhưng `QA answer` lại đa dạng hơn nhiều.

## 4. Nguồn đã research và ý nghĩa của từng nguồn

### 4.1 InstructBLIP
Nguồn:

- https://arxiv.org/abs/2305.06500
- https://arxiv.org/html/2305.06500

Điểm liên quan:

- Đề xuất `instruction-aware visual feature extraction`
- Instruction được đưa không chỉ vào LLM mà còn vào Q-Former
- Dùng balanced sampling để cải thiện optimization giữa nhiều dataset/task

Hỗ trợ cho:

- `H3`: Q-Former có thể tận dụng `QA` như một dạng instruction-conditioned visual supervision
- `H5`: mixed-task có thể vừa gây khó optimization lúc đầu, vừa có ích cho representation về sau

Mức độ liên quan:

- **Trực tiếp** với kiến trúc Q-Former và việc instruction đi vào visual extraction
- **Gián tiếp** với bài toán khiếm thị / overfit cụ thể của repo này

### 4.2 Octavius
Nguồn:

- https://arxiv.org/abs/2311.02684
- https://arxiv.org/html/2311.02684v3

Điểm liên quan:

- Chỉ ra `task interference` và `tug-of-war problem` trong MLLM dùng PEFT
- Nêu rõ hiện tượng này nặng hơn khi nhiều task cùng tối ưu trên một lượng parameter nhỏ

Hỗ trợ cho:

- `H4`: no-QFormer chịu conflict ở LLM/LoRA side nhiều hơn
- `H5`: hiện tượng observed có thể là tổ hợp của overfit cục bộ và task interference

Mức độ liên quan:

- **Gián tiếp nhưng mạnh** với setup LoRA + multi-task trong repo này

### 4.3 MultiInstruct
Nguồn:

- https://arxiv.org/abs/2212.10773

Điểm liên quan:

- Mỗi task có `5 expert-written instructions`
- Kết luận rằng training trên diverse tasks và diverse instructions làm giảm sensitivity với variations của instruction

Hỗ trợ cho:

- `H2`: một prompt `alter` cố định duy nhất có thể làm sensitivity cao hơn
- `H5`: thiếu instruction diversity có thể góp phần vào overfit hoặc shortcut learning

Mức độ liên quan:

- **Gián tiếp nhưng khá sát** với nghi vấn fixed prompt trong repo này

### 4.4 ReFine3D
Nguồn:

- https://arxiv.org/html/2606.18472v1

Điểm liên quan:

- Đề xuất `text synonymization / text diversity regularization`
- Lập luận rằng text diversification giúp giảm overfitting và tăng robustness trong low-data adaptation

Hỗ trợ cho:

- `H2`: prompt/output text diversity có thể là regularizer chống overfit
- phần experiment về paraphrase / multi-template prompt

Mức độ liên quan:

- **Gián tiếp** vì paper ở bối cảnh 3D VLM, nhưng ý tưởng regularization bằng text diversity khá phù hợp

## 5. Các vấn đề nhiều khả năng là đúng

### H1. `alter-only` có thể xuất hiện `test_alter` degradation sớm vì target text lặp và hẹp hơn
**Current evidence from repo/data**

- `alter` ngắn hơn `QA answer` rõ rệt: mean `14.99` vs `24.64` từ.
- `alter` có nhiều câu lặp exact hoặc near-duplicate.
- top `50` alter texts đã chiếm khoảng `11.45%` train set; khi normalize lỏng thì top `50` lên `12.93%`.

**Literature support**

- ReFine3D gợi ý rằng thiếu diversity phía text có thể góp phần làm adaptation kém robust hơn trong low-data adaptation.

**Why it matches the observed behavior**

- Khi chỉ train `alter`, model gặp một objective hẹp hơn và nhiều output template lặp hơn.
- Điều này có thể làm model học response pattern rất nhanh trên miền gần train/val, nhưng chưa chắc chuyển hóa thành khả năng generalize tốt trên `test_alter`.

**What would falsify it**

- Nếu sau khi tăng text diversity của `alter` mà hiện tượng `test_alter` degrade sớm gần như không đổi, thì H1 yếu đi.

**Nhãn**

- `Repo/Data Evidence`
- `Literature-Supported Inference`

### H2. Prompt `alter` cố định duy nhất làm tăng instruction sensitivity / shortcut learning
**Current evidence from repo/data**

- Toàn bộ `alter` đang dùng cùng một template cố định trong code.
- Q-Former còn dùng chính text này làm `qformer_text`, nên prompt ảnh hưởng cả visual extractor.

**Literature support**

- MultiInstruct cho thấy diverse instructions giúp giảm sensitivity với instruction wording.
- ReFine3D hỗ trợ ý rằng text diversification có thể regularize adaptation.

**Why it matches the observed behavior**

- Model có thể học shortcut kiểu “gặp prompt này thì phát guidance style A” thay vì học một hàm guidance tổng quát.
- Ở Q-Former, fixed prompt còn có thể khóa cả `visual query policy`.

**What would falsify it**

- Nếu paraphrase prompt `alter` ở inference hoặc train không làm metric thay đổi đáng kể, thì ảnh hưởng của fixed prompt có thể nhỏ hơn mình nghi.

**Nhãn**

- `Repo/Data Evidence`
- `Literature-Supported Inference`
- `Open Hypothesis`

### H3. Q-Former tận dụng `QA` như instruction-conditioned visual supervision
**Current evidence from repo/data**

- Trong Q-Former branch, instruction text đi vào `encode_qformer_texts(...)`, `set_qformer_text(...)`, rồi tác động trực tiếp lên feature extraction.
- `QA question` dù lặp wording mạnh nhưng vẫn tạo nhiều loại truy vấn thị giác cụ thể hơn `alter-only`.

**Literature support**

- InstructBLIP cho thấy instruction-aware Q-Former giúp trích visual features phù hợp với task instruction.

**Why it matches the observed behavior**

- Mixed `QA + alter` có thể làm optimization khó hơn ở early phase, nhưng về sau giúp bridge học tốt hơn cách khóa attention vào chi tiết liên quan.
- Điều đó giải thích pattern: Q-Former mixed có thể thua lúc đầu nhưng thắng về sau.

**What would falsify it**

- Nếu thay `qformer_text` bằng prompt cố định hoặc rỗng mà kết quả mixed gần như không đổi, thì lợi ích từ QA-conditioned visual extraction có thể không phải nguyên nhân chính.

**Nhãn**

- `Repo/Data Evidence`
- `Literature-Supported Inference`

### H4. no-QFormer chịu task interference ở LLM side nhiều hơn, nên `QA` không regularize đủ
**Current evidence from repo/data**

- no-QFormer không có trainable instruction-aware visual bridge.
- Khi thêm `QA`, phần trainable chính vẫn chủ yếu là LoRA ở language model.

**Literature support**

- Octavius mô tả rõ tug-of-war problem khi nhiều task cùng tối ưu bằng PEFT trên số lượng parameter nhỏ.

**Why it matches the observed behavior**

- `QA` và `alter` có hai output modes khác nhau:
  - `QA`: targeted answer
  - `alter`: guidance / navigation utterance
- Nếu không có visual adapter đủ mạnh để biến QA thành useful representation learning, `QA` có thể chủ yếu tăng conflict ở decoder.

**What would falsify it**

- Nếu no-QFormer mixed-task vẫn cải thiện mạnh sau khi chỉ thay prompt/data scheduling mà không đổi visual side, thì task interference có thể không phải lời giải chính.

**Nhãn**

- `Repo/Data Evidence`
- `Literature-Supported Inference`

### H5. Hiện tượng observed là tổ hợp của `data repetition + prompt sensitivity + task interference`
**Current evidence from repo/data**

- `alter` text có repetition đáng kể.
- `alter` chỉ có một prompt cố định.
- `QA` và `alter` khác nhau cả ở input style lẫn output distribution.
- Q-Former và no-QFormer tiếp nhận instruction theo hai cách khác nhau.

**Literature support**

- InstructBLIP hỗ trợ phần instruction-aware visual extraction.
- Octavius hỗ trợ phần task interference.
- MultiInstruct và ReFine3D hỗ trợ phần prompt/text diversity.

**Why it matches the observed behavior**

- Không hypothesis riêng lẻ nào hiện tại đủ giải thích trọn vẹn toàn bộ pattern.
- Tổ hợp ba yếu tố nói trên hợp với quan sát nhất:
  - `alter-only` có `test_alter` degradation sớm
  - Q-Former mixed tốt hơn về sau
  - no-QFormer mixed không giúp tương xứng

**What would falsify it**

- Nếu một ablation duy nhất giải thích được gần như toàn bộ khác biệt, ví dụ chỉ cần thay prompt diversity là mọi pattern biến mất, thì H5 có thể quá rộng.

**Nhãn**

- `Literature-Supported Inference`
- `Open Hypothesis`

### H6. Vấn đề chính có thể là `generalization gap` giữa `val_alter` và `test_alter`, không phải overfit cổ điển trên val
**Current evidence from repo/data**

- Theo quan sát mới, `val_alter` vẫn có thể tiếp tục giảm trong khi `test_alter` xấu đi.
- `val_alter` được lấy từ dữ liệu gần train hơn, trong khi `test_alter` là split riêng.
- `alter` text trong train có độ lặp đáng kể, nên model có thể fit tốt pattern gần train mà vẫn fail trên test.

**Literature support**

- Octavius hỗ trợ phần `task interference`.
- MultiInstruct và ReFine3D hỗ trợ hướng nhìn theo `instruction/text robustness`.
- Không paper nào trong nhóm hiện tại trực tiếp chứng minh đúng case này, nên đây vẫn là suy luận cần verify.

**Why it matches the observed behavior**

- Nếu `val_alter` giảm nhưng `test_alter` xấu đi, hiện tượng phù hợp hơn với:
  - `generalization degradation`
  - `shortcut/style over-specialization`
  - hoặc `loss-metric mismatch`
- Nó không còn khớp hoàn toàn với định nghĩa overfit cổ điển là `val` cũng phải xấu đi.

**What would falsify it**

- Nếu kiểm tra lại log cho thấy `val_alter` thực chất cũng xấu đi sớm như `test_alter`, hoặc nếu thay metric test mà pattern biến mất hoàn toàn, thì H6 cần được viết lại.

**Nhãn**

- `Repo/Data Evidence`
- `Open Hypothesis`

## 6. Bảng kết luận theo mức độ chắc chắn

### Khá chắc đúng

| Nhận định | Loại bằng chứng | Mức độ chắc chắn |
|---|---|---|
| `alter` ngắn hơn và template-heavy hơn `QA answer` | Repo/Data Evidence | Cao |
| Q-Former nhận instruction không chỉ ở LLM mà còn ở visual extraction | Repo/Data Evidence | Cao |
| `val_alter` và `test_alter` không nên bị gộp thành cùng một loại tín hiệu khi diễn giải hiện tượng hiện tại | Repo/Data Evidence | Cao |

### Có khả năng cao

| Nhận định | Loại bằng chứng | Mức độ chắc chắn |
|---|---|---|
| `alter-only` có thể xuất hiện `test_alter` degradation sớm một phần vì output distribution hẹp hơn | Literature-Supported Inference + Repo/Data Evidence | Khá cao |
| fixed prompt `alter` có thể làm tăng sensitivity và shortcut learning, từ đó làm xấu generalization | Literature-Supported Inference + Repo/Data Evidence | Khá cao |
| `QA` giúp Q-Former vì nó làm giàu instruction-conditioned visual supervision | Literature-Supported Inference + Repo/Data Evidence | Khá cao |
| no-QFormer mixed-task chịu tug-of-war ở LoRA/LLM side nhiều hơn | Literature-Supported Inference + Repo/Data Evidence | Khá cao |
| hiện tượng chính có thể là `generalization gap` giữa `val_alter` và `test_alter` hơn là overfit cổ điển trên val | Repo/Data Evidence + Open Hypothesis | Khá cao |

### Chưa đủ bằng chứng

| Nhận định | Loại bằng chứng | Mức độ chắc chắn |
|---|---|---|
| fixed prompt là nguyên nhân lớn nhất gây `test_alter` degradation | Open Hypothesis | Trung bình |
| chỉ cần thêm prompt diversity là sẽ giải quyết phần lớn hiện tượng | Open Hypothesis | Trung bình-thấp |
| `QA` là regularizer chính, còn repetition/prompt chỉ là yếu tố phụ | Open Hypothesis | Trung bình-thấp |

## 7. Các thí nghiệm nên chạy tiếp

### 7.1 Prompt sensitivity test trên alter
**Mục tiêu**

- Đo xem model phụ thuộc bao nhiêu vào wording của prompt `alter`.

**Thiết lập**

- Giữ nguyên checkpoint và data.
- Tạo `3-5` paraphrase prompt `alter` tương đương nghĩa.
- Chạy `test_alter` cho từng prompt.

**Kết quả nào sẽ xác nhận**

- Metric biến động đáng kể giữa các wording khác nhau.

**Kết quả nào sẽ bác bỏ**

- Metric gần như giữ nguyên giữa các wording.

### 7.2 Fixed prompt vs multi-template alter training
**Mục tiêu**

- Kiểm tra xem fixed prompt có làm `test_alter` degrade sớm hơn không.

**Thiết lập**

- So sánh hai run:
  - `1 prompt alter cố định`
  - `nhiều template alter` cùng semantics
- Chạy cho cả Q-Former và no-QFormer.

**Kết quả nào sẽ xác nhận**

- multi-template làm `test_alter` ổn định hơn sớm, hoặc làm giảm khoảng cách giữa `val_alter` và `test_alter`.

**Kết quả nào sẽ bác bỏ**

- hai setup cho curve gần như giống nhau.

### 7.3 Alter repetition audit
**Mục tiêu**

- Định lượng chính xác mức template hóa của `alter`.

**Thiết lập**

- Audit exact duplicate và near-duplicate.
- Báo cáo cumulative share của top `10/50/100/500` alter texts.

**Kết quả nào sẽ xác nhận**

- một phần đáng kể tập `alter` nằm trong số ít template / near-template.

**Kết quả nào sẽ bác bỏ**

- repetition hóa ra thấp hơn nhiều khi normalize semantic / punctuation.

### 7.4 Q-Former vs no-QFormer dưới cùng policy prompt augmentation
**Mục tiêu**

- Đo nhánh nào nhạy hơn với fixed prompt.

**Thiết lập**

- Dùng cùng một policy prompt variation cho cả hai nhánh.
- So sánh mức thay đổi của `test_alter` và khoảng cách `val_alter` - `test_alter`.

**Kết quả nào sẽ xác nhận**

- Q-Former thay đổi mạnh hơn, ủng hộ giả thuyết prompt tác động cả visual query.

**Kết quả nào sẽ bác bỏ**

- no-QFormer thay đổi tương đương hoặc mạnh hơn.

### 7.5 Alter-only vs mixed-task vs curriculum
**Mục tiêu**

- Tách vai trò `regularization sớm` khỏi `enrichment muộn`.

**Thiết lập**

- So sánh:
  - `alter-only`
  - `QA + alter` từ đầu
  - `alter-only` một giai đoạn đầu rồi mới thêm `QA`

**Kết quả nào sẽ xác nhận**

- curriculum hoặc mixed từ đầu giúp Q-Former tốt hơn rõ rệt về late-stage nhưng không nhất thiết thắng ngay early-stage.

**Kết quả nào sẽ bác bỏ**

- mọi policy cho kết quả gần như giống nhau.

### 7.6 Per-task validation logging
**Mục tiêu**

- Xem vấn đề nằm ở `generalization gap` của `alter` hay là instability chung.

**Thiết lập**

- Tách validation theo task:
  - `val_alter`
  - `val_qa`
- Log riêng từng loss hoặc metric.

**Kết quả nào sẽ xác nhận**

- mixed-task cho thấy `val_qa` ổn hơn nhưng `val_alter` vẫn đi khác `test_alter`, hoặc ngược lại.

**Kết quả nào sẽ bác bỏ**

- cả hai task có curve giống nhau, không support hypothesis task-specific.

## Ghi chú cuối
Tại thời điểm viết tài liệu này, nhận định mạnh nhất có bằng chứng là:

- `alter` đúng là có độ lặp và độ hẹp cao hơn đáng kể ở output side.
- Q-Former và no-QFormer khác nhau ở chỗ instruction có đi vào visual extractor hay không.
- literature hiện có support tốt cho:
  - task interference trong PEFT MLLM
  - lợi ích của instruction-aware Q-Former
  - lợi ích của instruction/text diversity đối với robustness

- dữ kiện `val_alter` tiếp tục giảm nhưng `test_alter` xấu đi cần được diễn giải dưới góc `generalization gap`, không nên gắn thẳng nhãn `overfit cổ điển` nếu chưa verify thêm.

Điều chưa thể khẳng định chắc mà không làm ablation là:

- fixed prompt có phải thủ phạm lớn nhất không
- `QA` giúp chủ yếu vì regularization hay vì visual enrichment
- tỷ trọng đóng góp của từng yếu tố trong ba yếu tố: `repetition`, `fixed prompt`, `task interference`
