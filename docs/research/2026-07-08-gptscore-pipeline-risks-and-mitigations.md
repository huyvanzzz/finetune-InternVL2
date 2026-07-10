# Thiết kế pipeline `GPTScore` ổn định và công bằng cho `alter`

## 1. Mục tiêu của tài liệu này

Tài liệu này trả lời câu hỏi:

> Khi đưa prompt judge `GPTScore` vào code thật, cần thiết kế pipeline như thế nào để:
> - chạy ổn định
> - parse chắc chắn
> - không để lỗi nhỏ của LLM làm vỡ cả flow
> - giữ fairness khi tổng hợp điểm

Đợt này chỉ làm `research + synthesis + spec`, chưa code runner.

Định hướng cốt lõi:

- `judge` chỉ trả semantic judgment
- `pipeline` mới là nơi tính điểm cuối
- sample lỗi phải bị tách riêng, không được đánh đồng thành sample điểm thấp

## 2. Sự thật nền từ repo và từ file `pairs`

### 2.1 `alter` GT đang đi vào judge từ đâu

`Data-evidenced`

Trong [preprocessing.py](D:/NCKH_VLM/finetune-InternVL2/preprocessing.py):

- `format_ground_truth()` gọi `map_metadata_to_ground_truth()`
- với task `alter`, instruction lấy trực tiếp từ `metadata["alter"]`
- với `response_format == "direct_text"`, GT cuối cùng là `instruction.strip()`

Ý nghĩa:

- pipeline `GPTScore` v1 có thể xem `ground_truth` là câu `alter` cuối cùng đã được xuất ra để chấm
- không có lớp biến đổi ngữ nghĩa nào ở giữa cần phải "đoán thêm"

### 2.2 File `qformer_eval_test_alter_pairs.json` cho thấy rủi ro gì

`Data-evidenced`

Từ file [qformer_eval_test_alter_pairs.json](D:/Downloads/qformer_eval_test_alter_pairs.json):

- tổng số cặp: `1007`
- `ground_truth` rỗng: `0`
- `generation` rỗng: `0`

Nhưng phân bố `generation` rất lệch:

- một câu xuất hiện `403` lần:
  - `at 11 o'clock direction, there are pedestrians passing by, pay attention to avoid.`
- một số câu khác cũng lặp mạnh:
  - `at 11 o'clock direction there is a tree, be careful to avoid it.`: `89`
  - `the road ahead is clear, please move forward without worry.`: `48`
  - `the current road is clear, please move forward without worry.`: `30`

Ngoài ra:

- các generation có pattern `clock direction`: `892 / 1007`
- các generation có pattern `avoid / be careful / pay attention`: `889 / 1007`
- các generation có pattern `clear / unobstructed / move forward / without worry`: `110 / 1007`

Ý nghĩa đối với pipeline:

- đây không còn chỉ là bài toán "LLM judge có parse được JSON hay không"
- pipeline còn phải hỗ trợ audit tốt cho các lỗi kiểu:
  - lặp template
  - generic safe reassurance
  - sai scene nhưng câu vẫn trơn và có vẻ hợp lý

Tài liệu này **không** dùng các mẫu generation đó để sửa prompt judge. Nó chỉ dùng chúng để chốt:

- loại artifact nào cần lưu
- loại sanity check nào cần có
- sample-level failure nào cần được xem là rủi ro fairness cao

### 2.3 Judge output và pipeline output phải tách rõ

`Design choice for v1`

Đã chốt từ các tài liệu GPTScore hiện tại:

- `judge output` chỉ gồm semantic judgment
- `pipeline output` mới gồm điểm tổng hợp và các field runtime

Nếu không tách hai lớp này, sau này rất dễ:

- prompt lấn sang làm thay cả cách tính điểm
- sample parse lỗi bị "tưởng như" đã được chấm
- khó audit vì sao điểm đổi

## 3. Kiến trúc pipeline khuyến nghị cho v1

`Design choice for v1`

Nên khóa pipeline thành 4 tầng rõ ràng:

1. `input loader`
2. `judge caller`
3. `validator / normalizer`
4. `aggregator`

### 3.1 Tầng 1: `input loader`

Nhiệm vụ:

- đọc `pairs.json` hoặc `pairs.jsonl`
- xác nhận mỗi sample có đủ field tối thiểu
- normalize nhẹ phần text trước khi gọi API

Input tối thiểu khuyến nghị:

- `sample_id`
- `ground_truth`
- `generation`

Có thể giữ thêm:

- `frame_path`
- `questionId`
- `model_name`
- `checkpoint`

nhưng những field đó là metadata, không phải input bắt buộc cho judge v1.

### 3.2 Tầng 2: `judge caller`

Nhiệm vụ:

- gọi `GPT-4o`
- dùng prompt production đã chốt
- dùng Structured Outputs với `strict` schema
- xử lý retry cho lỗi transport / timeout / rate limit

Khuyến nghị v1:

- `non-streaming`
- `one pair -> one judge call`
- pin model cụ thể, không dùng alias mơ hồ nếu có thể

### 3.3 Tầng 3: `validator / normalizer`

Nhiệm vụ:

- kiểm JSON judge output
- phát hiện schema fail
- phát hiện semantic fail
- đưa output về một dạng nội bộ ổn định

Tầng này phải tồn tại vì:

- Structured Outputs giảm lỗi format, nhưng không nên tin mù 100%
- output đúng schema vẫn có thể sai logic rubric

### 3.4 Tầng 4: `aggregator`

Nhiệm vụ:

- map `label -> score`
- bỏ `null` ra khỏi mean
- tính `mean_before_gate`
- áp `gate cap`
- sinh `overall_score`

Tầng này phải deterministic và không được phụ thuộc vào suy luận của judge.

## 4. Ba lớp kiểm tra bắt buộc

`Design choice for v1`

Pipeline không được dùng model output theo kiểu "parse được là xong".

Cần 3 lớp kiểm:

### 4.1 Schema-level validation

Kiểm tra:

- thiếu field bắt buộc
- thừa field nếu schema đang strict
- criterion key bị mất
- `label` không thuộc enum hợp lệ
- `applicable = false` nhưng `label != null`
- `applicable = true` nhưng `label = null`

Đây là lớp chặn đầu tiên.

### 4.2 Semantic-level validation

Kiểm tra:

- `Direction = N/A` nhưng rationale lại bảo generation sai hướng nguy hiểm mà gate vẫn không bật
- `Action = N/A` nhưng rationale mô tả unsafe action mà gate vẫn không bật
- gate mô tả mâu thuẫn với criterion-level judgments
- `signals_in_gt` mâu thuẫn rõ ràng với applicable decisions

Đây là lớp giữ fairness.

### 4.3 Aggregation-level validation

Kiểm tra:

- map label sang số đúng
- chỉ average trên criterion applicable
- `gate cap` áp dụng đúng policy
- không có trường hợp `overall_score` được tính cho sample chưa qua validation

Đây là lớp chặn bug ở phần code tổng hợp.

## 5. Failure modes chính và cách xử lý

## 5.1 Nhóm A: lỗi input artifact

### Lỗi có thể xảy ra

`Data-evidenced` + `Design choice for v1`

- thiếu `ground_truth`
- thiếu `generation`
- chuỗi rỗng
- duplicate rows
- encoding lỗi
- whitespace / newline quá bẩn

### Vì sao nguy hiểm

- sample vô nghĩa vẫn bị đưa đi chấm
- cùng 1 sample bị tính 2 lần
- output sau cùng nhìn có vẻ hợp lệ nhưng benchmark bị bẩn

### Cách phát hiện

- validate trước khi gọi API
- strip text
- flag empty string
- dedupe theo `sample_id` nếu có
- nếu không có `sample_id`, cần chốt policy dedupe riêng, không được tự ý đồng sample

### Cách xử lý khuyến nghị

- sample invalid -> đưa vào `results_errors.jsonl`
- `judge_status = skipped_invalid_input`
- `validation_status = input_invalid`
- không crash whole run

## 5.2 Nhóm B: schema / parse failures

### Lỗi có thể xảy ra

`Research-supported inference`

- output không parse được
- thiếu criterion
- criterion key sai tên
- `label` ngoài tập:
  - `Fail`
  - `Weak`
  - `Acceptable`
  - `Strong`
  - `null`
- `overall_score` xuất hiện trong judge JSON dù đã cấm

### Vì sao nguy hiểm

- nếu pipeline cố gắng "sửa tạm" một cách mơ hồ, fairness sẽ vỡ
- parse fail mà auto chuyển thành `Fail` là rất sai

### Cách phát hiện

- strict JSON Schema
- schema validator sau parse
- explicit enum check

### Cách xử lý khuyến nghị

- retry có giới hạn ở cấp sample
- nếu vẫn fail:
  - không chấm điểm
  - lưu raw response
  - lưu error reason
  - đưa vào `results_errors.jsonl`

Không được:

- tự động gán `overall_score = 0`
- tự "đoán" label thay judge

## 5.3 Nhóm C: refusal / incomplete output / transport failures

### Lỗi có thể xảy ra

`Research-supported inference`

- refusal
- timeout
- rate limit
- output bị truncated
- request bị incomplete
- max output hit

### Bằng chứng nền

`Research-supported inference`

OpenAI Structured Outputs docs có nói rõ:

- model có thể trả refusal thay vì parsed object
- với Structured Outputs strict, schema không hỗ trợ sẽ bị request error

OpenAI Responses API docs cũng thể hiện object / item có thể có trạng thái `incomplete`.

Nguồn:

- Structured Outputs guide: https://developers.openai.com/api/docs/guides/structured-outputs
- Responses API reference: https://developers.openai.com/api/reference/resources/responses/methods/create/

### Cách phát hiện

- HTTP / SDK exception
- refusal field / refusal item
- status incomplete
- parse error sau khi response kết thúc

### Cách xử lý khuyến nghị

- v1 dùng `non-streaming`
- retry với exponential backoff cho:
  - timeout
  - transient network errors
  - rate limit
  - incomplete transport responses
- refusal thì:
  - `judge_status = refused`
  - không retry vô hạn
  - không tự gán điểm

### Policy

- refusal != score thấp
- incomplete != score thấp
- parse fail != score thấp

## 5.4 Nhóm D: semantic drift dù JSON hợp lệ

### Lỗi có thể xảy ra

`Research-supported inference`

Judge có thể trả JSON hợp lệ nhưng vẫn sai theo rubric:

- gate bật quá dễ
- gate bỏ sót
- `Direction` bị để `N/A` sai
- `Action` bị để applicable quá rộng
- rationale tự mâu thuẫn với label

### Vì sao nguy hiểm

Đây là loại lỗi khó thấy nhất:

- file JSON vẫn đẹp
- code vẫn chạy
- nhưng điểm sai một cách âm thầm

### Cách phát hiện

- semantic validator sau parse
- bộ sanity cases có expected behavior
- file audit mẫu để đọc tay

### Cách xử lý khuyến nghị

- sample semantic-invalid -> không đưa vào scoring aggregate
- đánh dấu:
  - `judge_status = returned_json`
  - `validation_status = semantic_failed`
- lưu lý do fail

Không nên:

- cố gắng "sửa nhẹ" kết quả semantic mà không có rule deterministic

## 5.5 Nhóm E: fairness drift do mode collapse / template repetition

### Lỗi có thể xảy ra

`Data-evidenced` + `Research-supported inference`

File `pairs` hiện tại cho thấy generation có thể lặp template rất mạnh. Điều này tạo ra một loại rủi ro riêng:

- judge có thể lặp lại cùng một quyết định cho hàng trăm mẫu mà không được kiểm tra phân phối
- benchmark cuối nhìn "ổn định" nhưng thực ra chỉ đang chấm một vài template lặp
- một câu generic như `road is clear` có thể xuất hiện trên rất nhiều scene khác nhau

### Vì sao nguy hiểm

Đây không phải lỗi parse, mà là lỗi fairness và auditability:

- nếu chỉ nhìn trung bình, ta rất khó biết model đang thật sự hiểu scene hay đang sụp vào vài template
- pipeline nếu không lưu đủ artifact sẽ khiến giai đoạn review tay gần như bất khả thi

### Cách phát hiện

- thống kê top repeated generations ở tầng input hoặc sau scoring
- log phân bố generation theo exact text
- log tỉ lệ sample có:
  - generic reassurance
  - repeated direction template
  - repeated obstacle template

### Cách xử lý khuyến nghị

- không sửa điểm chỉ vì bị lặp
- nhưng phải lưu đủ artifact để audit:
  - input pair
  - judge output
  - pipeline output
- nên có thêm summary sau run:
  - `top_repeated_generations`
  - `repeat_coverage`

Rule quan trọng:

- `mode collapse` không phải là parse failure
- cũng không nên bị "phạt ngầm" trong aggregator
- nó là tín hiệu để phân tích chất lượng model ở lớp báo cáo / audit

## 5.6 Nhóm F: fairness và reproducibility drift

### Lỗi có thể xảy ra

`Research-supported inference`

- alias model thay đổi ngầm
- prompt wording đổi nhẹ
- schema đổi
- logic map label đổi
- cùng một pairs file nhưng benchmark 2 đợt không còn comparable

### Vì sao nguy hiểm

Lỗi này không làm crash pipeline, nhưng làm mất khả năng so sánh.

### Cách phát hiện

- lưu metadata cho mỗi run:
  - model id
  - prompt version
  - prompt hash
  - schema hash
  - pipeline version
  - run timestamp

### Cách xử lý khuyến nghị

- pin exact model snapshot nếu hệ thống hỗ trợ
- pin 1 prompt production file cụ thể
- pin 1 schema cụ thể
- mọi thay đổi prompt / schema phải tăng version artifact

## 6. Policy fail-sample vs fail-whole-run

`Design choice for v1`

Đây là policy quan trọng nhất để tránh vỡ flow và tránh làm sai fairness.

### 6.1 Nên fail sample, không fail whole run, khi:

- input sample không hợp lệ
- parse fail
- refusal
- incomplete response
- semantic validation fail

### 6.2 Chỉ nên fail whole run khi:

- schema config toàn cục bị sai
- API credentials / endpoint sai toàn bộ
- prompt / schema version không khớp contract
- aggregator bug làm sample pass cũng không thể tính điểm đúng

### 6.3 Rule ưu tiên

Nếu phải chọn giữa:

- "giữ lại mọi sample bằng mọi giá"
- và "giữ fairness"

thì ưu tiên fairness:

- thiếu 1 sample còn hơn là tự đặt điểm sai cho nó

## 7. Judge output và pipeline output

## 7.1 Judge output

`Design choice for v1`

Judge output là nguồn semantic duy nhất, chỉ gồm:

- `gate`
- `signals_in_gt`
- `criteria`
- `overall_rationale`

Judge output không được có:

- `mean_before_gate`
- `applied_gate_cap`
- `overall_score`

Nếu có các field trên, đó là schema / prompt drift.

## 7.2 Pipeline output

`Design choice for v1`

Pipeline output là artifact sau khi đã:

- parse
- validate
- aggregate

Nên có ít nhất:

- `sample_id`
- `judge_status`
- `validation_status`
- `judge_output`
- `mean_before_gate`
- `applied_gate_cap`
- `overall_score`

Có thể thêm:

- `error_category`
- `error_message`
- `retry_count`
- `model_id`
- `prompt_version`
- `prompt_hash`
- `schema_hash`

## 8. Rule tính điểm cuối

`Design choice for v1`

Pipeline map như sau:

- `Fail -> 0`
- `Weak -> 1`
- `Acceptable -> 2`
- `Strong -> 3`
- `null` bị loại khỏi mean

### 8.1 `mean_before_gate`

Tính trung bình trên các criterion:

- có `applicable = true`
- có `label` hợp lệ

### 8.2 `applied_gate_cap`

Khuyến nghị lưu thành field riêng để dễ audit.

Giá trị có thể là:

- `none`
- `score=0.0_by_polarity_reversal`
- `score=0.0_by_unsafe_action`
- `score<=1.0_by_unsafe_direction_reversal`

Tên cụ thể có thể chốt lúc code, nhưng ý nghĩa phải giữ ổn định.

### 8.3 `overall_score`

Rule deterministic:

- nếu `polarity_reversal = true` -> `overall_score = 0.0`
- else nếu `unsafe_action = true` -> `overall_score = 0.0`
- else nếu `unsafe_direction_reversal = true` -> `overall_score = min(mean_before_gate, 1.0)`
- else -> `overall_score = mean_before_gate`

Quan trọng:

- sample parse lỗi hoặc refusal không được tính `overall_score`
- sample đó phải nằm ở luồng error / invalid

## 9. Retry policy khuyến nghị

`Design choice for v1`

Không nên retry tất cả mọi loại lỗi.

### 9.1 Nên retry

- timeout
- transient network error
- rate limit
- incomplete transport response
- parse/schema fail do output không trọn vẹn

### 9.2 Không nên retry nhiều lần

- refusal rõ ràng
- input invalid
- semantic validation fail lặp lại sau một lần retry schema-sạch

### 9.3 Khuyến nghị v1

- `max_retries` nhỏ, ví dụ `2` hoặc `3`
- exponential backoff
- lưu số retry đã dùng

Lý do:

- retry quá nhiều dễ tăng chi phí và làm mờ audit trail

## 10. Output artifacts khuyến nghị

`Design choice for v1`

Nên tách tối thiểu 2 artifact:

### 10.1 `results_scored.jsonl`

Chỉ chứa các sample:

- parse ok
- validation ok
- aggregate ok

Mỗi dòng nên có:

- input pair tối thiểu
- judge output
- pipeline output
- metadata run

### 10.2 `results_errors.jsonl`

Chứa các sample:

- input invalid
- refusal
- timeout hết retry
- parse fail
- semantic validation fail

Mỗi dòng nên có:

- input pair tối thiểu
- raw response nếu có
- error category
- error detail
- retry count

### 10.3 Run summary

Cuối run nên có tổng kết:

- `total_samples`
- `scored_samples`
- `input_invalid`
- `refused_samples`
- `parse_or_schema_failed`
- `semantic_validation_failed`
- `transport_failed_after_retries`
- `retries_used_total`

Nên cân nhắc thêm các field audit:

- `top_repeated_generations`
- `repeat_coverage`

để phát hiện mode collapse mà không phải mở lại toàn bộ file sample-level.

## 11. Danh sách failure mode theo mức độ ưu tiên

`Design choice for v1`

| Failure mode | Severity | Likelihood | Detection | Khuyến nghị |
|---|---|---|---|---|
| Input pair thiếu field / rỗng | high | medium | input validation | fail sample |
| Parse / schema fail | high | medium | schema validator | retry giới hạn, sau đó fail sample |
| Refusal | medium | low-medium | refusal field / item | mark refused, không auto-score |
| Timeout / rate limit / incomplete | high | medium | transport status / SDK errors | retry backoff |
| Semantic drift dù JSON hợp lệ | critical | medium | semantic validator + sanity cases | fail sample, audit tay |
| Mode collapse / repeated template | high | high | repeat statistics + audit samples | không sửa điểm, nhưng phải log để audit |
| Prompt / schema / model drift | critical | medium | version hash metadata | pin version, lưu artifact metadata |
| Duplicate sample | medium | medium | input dedupe policy | skip hoặc flag duplicate |

## 12. Preflight checklist trước khi code runner

`Design choice for v1`

Trước khi code thật, nên chốt:

1. Prompt production file nào là source of truth
2. JSON Schema strict nào là source of truth
3. Rule semantic validator nào là bắt buộc
4. Retry policy nào là chốt
5. Output artifact names nào là chốt
6. Sanity cases nào dùng để test prompt và validator
7. Summary statistics nào phải có để bắt mode collapse

Nếu các mục này chưa khóa, code runner sẽ rất dễ đổi behavior ngầm.

## 13. Những gì nên test ở pha code sau

`Design choice for v1`

### 13.1 Schema tests

- valid judge output parse được
- missing criterion bị reject
- invalid label bị reject
- `applicable = false` nhưng `label != null` bị reject

### 13.2 Pipeline scoring tests

- map `label -> score` đúng
- `null` bị loại khỏi mean
- 3 gate cap đúng policy

### 13.3 Failure-handling tests

- refusal không crash run
- incomplete response vào retry path
- parse fail vào `results_errors.jsonl`

### 13.4 Fairness tests

- sample parse lỗi không bị tính như score thấp
- duplicate sample không bị tính hai lần nếu chốt policy dedupe

### 13.5 Sanity-case tests

- chạy bộ review cases GPTScore hiện có
- check semantic validator và gate policy

### 13.6 Auditability tests

- summary có thể log top repeated generations
- sample-level artifacts đủ để lần ngược từ summary về từng cặp `GT / generation`

## 14. Kết luận chốt

Nếu muốn pipeline `GPTScore` v1 vừa ổn định vừa công bằng, cần giữ 5 nguyên tắc:

1. `judge output` và `pipeline output` phải tách rõ
2. parse fail / refusal / semantic fail phải là một nhóm riêng, không được giả vờ thành score thấp
3. Structured Outputs là lớp giúp ổn định, nhưng không đủ để bỏ qua validation
4. sample-level failure phải được cách ly để không vỡ whole run
5. pipeline phải hỗ trợ audit tốt cho mode collapse và repeated template, chứ không chỉ cho parse lỗi

Nói ngắn gọn:

> Mục tiêu đúng không phải là "LLM phải trả JSON đẹp mỗi lần", mà là:
> - nếu nó trả đẹp thì điểm tính ra phải công bằng
> - nếu nó trả lỗi thì pipeline vẫn không vỡ
> - nếu nó không chấm được thì phải nói rõ là "không chấm được", không được biến thành điểm 0
> - và nếu model sinh lặp template trên diện rộng thì pipeline phải giúp mình nhìn ra điều đó

## 15. Nguồn chính đã dùng

- OpenAI Structured Outputs guide  
  https://developers.openai.com/api/docs/guides/structured-outputs

- OpenAI Responses API reference  
  https://developers.openai.com/api/reference/resources/responses/methods/create/

- Ground-truth mapping trong repo hiện tại  
  [preprocessing.py](D:/NCKH_VLM/finetune-InternVL2/preprocessing.py)

- File cặp GT / generation thực tế  
  [qformer_eval_test_alter_pairs.json](D:/Downloads/qformer_eval_test_alter_pairs.json)

- Các tài liệu GPTScore hiện có trong repo  
  [2026-07-08-gptscore-design-for-alter.md](D:/NCKH_VLM/finetune-InternVL2/docs/research/2026-07-08-gptscore-design-for-alter.md)  
  [2026-07-08-gptscore-scale-design.md](D:/NCKH_VLM/finetune-InternVL2/docs/research/2026-07-08-gptscore-scale-design.md)  
  [2026-07-08-gptscore-gpt4o-production-prompt.md](D:/NCKH_VLM/finetune-InternVL2/docs/research/2026-07-08-gptscore-gpt4o-production-prompt.md)
