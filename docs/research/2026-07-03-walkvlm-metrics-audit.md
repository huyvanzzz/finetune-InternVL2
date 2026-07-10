# Audit metric `keyword-density` và `GPTScore` trong repo `walkvlm`

## 1. Mục tiêu và phạm vi

Tài liệu này truy vết lại hai metric được dùng trong base repo `xiaoyuan1996/walkvlm`:

- `keyword-density`
- `GPTScore`

Mục tiêu:

- viết lại chính xác hai metric này được tạo như thế nào trong repo gốc
- tách rõ phần nào đã có bằng chứng từ code
- tách rõ phần nào chỉ thấy call-site nhưng thiếu source
- đối chiếu xem hai metric này có còn công bằng nếu đem sang repo hiện tại hay không

Phạm vi phân tích chỉ tập trung vào task `alter`.

## 2. Nguồn đã đọc và mức độ bằng chứng

### 2.1 Nguồn `walkvlm` đã đọc trực tiếp

`Code-evidenced`

- `README.md` ở root repo `walkvlm`
- `WalkVLM-LR/README.md`
- `WalkVLM-LR/test.py`
- `WalkVLM-LR/inference.py`
- `WalkVLM-LR/vlm_grpo_template/src/open_r1/grpo_query_gene.py`
- `WalkVLM-LR/vlm_grpo_template/src/open_r1/trainer/grpo_trainer.py`
- `WalkVLM-LR/vlm_grpo_template/src/open_r1/sft.py`

### 2.2 Phát hiện quan trọng

`Code-evidenced`

- `WalkVLM-LR/test.py` import `from GPTScore import evaluate_image`
- `README.md` root có liệt kê `WalkVLM-LR/GPTScore.py` trong code structure
- nhưng trong clone công khai hiện tại, file `WalkVLM-LR/GPTScore.py` không tồn tại

Hệ quả:

- `keyword-density` có thể reverse-engineer đầy đủ từ code
- `GPTScore` chỉ có thể reverse-engineer ở mức usage contract và giới hạn tái lập

## 3. Luồng đánh giá metric trong `walkvlm`

### 3.1 Luồng test offline

`Code-evidenced`

Trong `WalkVLM-LR/test.py`, pipeline đánh giá chạy như sau:

1. Load tập validation từ file jsonl.
2. Bỏ qua sample nào không có trường `alter`.
3. Dùng ảnh cuối cùng có đường dẫn:
   - `wad_dataset/src_data/<frame_path>/8.jpg`
4. Lấy:
   - `reference = example["alter"]`
   - `keywords = example["keywords"]`
   - `qwen_text = example["qwen72b_output"]`
5. Model sinh `generated_text`.
6. Tính 3 nhóm metric:
   - ROUGE với `reference` và `generated_text`
   - `keyword-density` trên `generated_text` và `keywords`
   - `GPTScore` qua `evaluate_image(appid, appkey, source, qwen_text, generated_text)`
7. Cuối cùng lấy trung bình trên toàn bộ sample hợp lệ.

### 3.2 Luồng reward khi GRPO train

`Code-evidenced`

Trong `grpo_query_gene.py` và `grpo_trainer.py`:

- reward function `keywords_reward` gọi lại `calculate_keyword_density(...)`
- `keywords` được lấy từ dataset và truyền vào prompt metadata
- trainer repeat `keywords` theo `num_generations` để tính reward cho mỗi completion

Ý nghĩa:

- `keyword-density` không chỉ là metric test
- nó còn được dùng như một reward signal trong hướng fine-tune GRPO

### 3.3 Luồng inference

`Code-evidenced`

`WalkVLM-LR/inference.py` phục vụ sinh output từ ảnh, nhưng không phải nơi trung tâm của hai metric này. Metric logic xuất hiện rõ nhất ở `test.py`.

## 4. `keyword-density` được tạo như thế nào

### 4.1 Định nghĩa tổng quát

`Code-evidenced`

`keyword-density` trong `walkvlm` không phải là đếm exact keyword xuất hiện bao nhiêu lần trong câu.

Nó là một tỉ lệ semantic coverage:

- lấy các token ứng viên từ câu sinh ra
- embedding từng token bằng CLIP text encoder
- embedding từng keyword bằng CLIP text encoder
- nếu cosine similarity giữa token và bất kỳ keyword nào >= `0.9` thì token đó được tính là một keyword-hit
- density = `số token hit / tổng số token ứng viên`

### 4.2 Các bước cụ thể trong code

`Code-evidenced`

Hàm `calculate_keyword_density(text, keywords, model, processor, threshold=0.9)` trong `WalkVLM-LR/test.py` làm như sau:

1. Chuẩn hóa text:
   - lower-case
   - bỏ dấu câu bằng regex `re.sub(r'[^\w\s]', '', text.lower())`
2. Tách unigram:
   - `word_tokens = text.split()`
3. Deduplicate unigram:
   - `unique_word_tokens = set(word_tokens)`
4. Tạo bigram:
   - `generate_ngrams(text, 2)`
5. Hợp nhất tập token ứng viên:
   - `all_tokens = unique_word_tokens union set(bigrams)`
6. Với mỗi `token` trong `all_tokens`:
   - encode CLIP cho token
   - với mỗi `keyword` trong `keywords`:
     - encode CLIP cho keyword
     - tính `cosine_similarity`
     - nếu `similarity >= 0.9` thì tăng `keyword_count` lên 1 và `break`
7. Tính:
   - `keyword_density = keyword_count / len(all_tokens)`

### 4.3 Mẫu số và tử số là gì

`Code-evidenced`

- Tử số:
  - số token ứng viên đã match semantic với ít nhất một keyword
- Mẫu số:
  - tổng số token ứng viên sau khi deduplicate, bao gồm:
    - unigram unique
    - bigram unique

Hệ quả:

- metric này thường cao hơn exact-match keyword theo nghĩa semantic
- metric này không phát hiện tần suất lặp lại keyword, vì token đã deduplicate trước

### 4.4 Metric này đo cái gì

`Repo-structure inference`

Metric này đo mức độ câu sinh ra có “chứa nhiều thành phần semantically gần với keyword mong muốn” hay không.

Nó nghiêng về:

- compactness/cô đọng
- semantic inclusion của từ khóa

Nó không trực tiếp đo:

- thứ tự thông tin
- tính đầy đủ về hành động
- độ an toàn của hướng dẫn
- tính tự nhiên của câu trả lời

### 4.5 Pseudo-code ngắn gọn

`Code-evidenced`

```text
clean_text = lowercase(remove_punctuation(text))
unigrams = unique(split(clean_text))
bigrams = unique(generate_bigrams(clean_text))
candidate_tokens = unigrams ∪ bigrams

hit_count = 0
for token in candidate_tokens:
    token_vec = CLIP_text(token)
    for keyword in keywords:
        keyword_vec = CLIP_text(keyword)
        if cosine(token_vec, keyword_vec) >= 0.9:
            hit_count += 1
            break

keyword_density = hit_count / len(candidate_tokens)
```

### 4.6 Phiên bản test metric và train reward

#### Trong `test.py`

`Code-evidenced`

- threshold mặc định: `0.9`
- output là một scalar `keyword_density`
- được average qua tập test

#### Trong `grpo_query_gene.py`

`Code-evidenced`

- logic tính gần như giống hệt
- `keywords_reward` trả về `max(0.0, keyword_density)`
- reward này được đặt tên `keywords`

#### Kết luận so sánh

`Code-evidenced`

- test metric và train reward hiện tại là cùng một logic cốt lõi
- không thấy có version reward khác bản chất so với version test

## 5. `GPTScore` được biết tới mức nào

### 5.1 Những gì có thể xác minh trực tiếp

`Code-evidenced`

Trong `WalkVLM-LR/test.py`:

```python
from GPTScore import evaluate_image
...
gpt_score = evaluate_image(appid, appkey, source, qwen_text, generated_text)
```

Trong cùng file:

- `appid = ""`
- `appkey = ""`
- `source = ""`

Trong README:

- repo có liệt kê `GPTScore.py` là một phần của code structure
- README test command ghi rõ cần cấu hình thủ công “parameters for the GPT-4 API”

Từ đây có thể xác minh:

- `GPTScore` không phải metric thuần local
- nó phụ thuộc ít nhất một external API credential
- nó nhận 5 tham số:
  - `appid`
  - `appkey`
  - `source`
  - `qwen_text`
  - `generated_text`

### 5.2 Những gì có thể suy ra hợp lý từ call-site

`Repo-structure inference`

Rất có khả năng `GPTScore` là một judge score do model/API bên ngoài chấm:

- `qwen_text = example["qwen72b_output"]`
- `generated_text` là output của model đang đánh giá

Nên metric này nhiều khả năng đang đo:

- mức độ `generated_text` giống hoặc đạt chất lượng so với một teacher text `qwen72b_output`
- hoặc một API judge đọc hai câu và trả về điểm

Điều quan trọng là:

- call-site không đưa `alter` vào `evaluate_image(...)`
- nên `GPTScore` có khả năng cao là không chấm trực tiếp prediction so với ground-truth human `alter`
- thay vào đó nó chấm prediction so với một teacher/model khác

### 5.3 Giới hạn không thể vượt qua

`Missing-source limitation`

Vì file `GPTScore.py` không có trong repo công khai, hiện tại không thể khẳng định:

- prompt judge thật sự là gì
- score scale là gì
- có dùng ảnh thực tế trong request hay không
- có so sánh song song `qwen_text` và `generated_text` hay có thêm xử lý trung gian
- có post-processing, retry, parsing JSON, normalization nào không

Do đó, mọi kết luận về internal implementation của `GPTScore.py` đều không được phép khẳng định.

## 6. Đầu vào dữ liệu mà hai metric đang phụ thuộc

### 6.1 Trong `walkvlm`

`Code-evidenced`

Hai metric này phụ thuộc thêm vào các field mà repo hiện tại của bạn không sử dụng:

- `keywords`
- `qwen72b_output`

Ngoài ra `test.py` cũng giả định:

- ảnh cuối cùng có đường dẫn `.../<frame_path>/8.jpg`

### 6.2 Trong repo hiện tại của bạn

`Code-evidenced`

Qua grep repo hiện tại:

- không thấy `keywords`
- không thấy `qwen72b_output`
- metric hiện tại trong `scripts/metrics.py` chỉ có:
  - ROUGE-1
  - ROUGE-2
  - ROUGE-L
  - TF-IDF

Repo hiện tại đang test trên:

- output `direct_text`
- ground truth `alter`

## 7. Đối chiếu tính công bằng với repo hiện tại

| Metric | Có thể đem sang repo hiện tại ngay? | Blocker | Fairness risk |
|---|---|---|---|
| `keyword-density` | Chưa ngay lập tức | thiếu field `keywords` trong pipeline hiện tại | nếu tự sinh keyword bằng cách khác thì metric không còn trung thành với base `walkvlm` |
| `GPTScore` | Gần như không | thiếu `GPTScore.py`, thiếu API contract đầy đủ, thiếu `qwen72b_output` | metric phụ thuộc teacher/model ngoài, nên không còn là so sánh thuần prediction-vs-ground-truth |

### 7.1 Đánh giá `keyword-density`

`Repo-structure inference`

Nếu muốn dùng `keyword-density` một cách công bằng, cần giữ được ít nhất:

- cùng nguồn `keywords`
- cùng CLIP model
- cùng threshold `0.9`
- cùng logic deduplicate unigram + bigram

Nếu repo hiện tại không có `keywords`, có 3 trường hợp:

1. Lấy lại đúng field `keywords` từ data gốc nếu tồn tại trong dataset upstream.
2. Tự sinh `keywords` bằng một model/heuristic khác.
3. Bỏ metric này.

Trong 3 trường hợp đó:

- chỉ trường hợp 1 mới gần công bằng với base `walkvlm`
- trường hợp 2 sẽ biến metric thành một metric mới

### 7.2 Đánh giá `GPTScore`

`Repo-structure inference`

Ngay cả khi có thể phục dựng lại `GPTScore.py`, metric này vẫn có rủi ro công bằng cao vì:

- nó phụ thuộc `qwen72b_output`
- nghĩa là phụ thuộc vào teacher/model trung gian
- không phải intrinsic metric chỉ dựa trên `alter`

Vì vậy, nếu mục tiêu là so sánh công bằng giữa các backbone trong repo hiện tại, `GPTScore` có khả năng là metric phụ trợ hơn là metric chính.

## 8. Kết luận tổng hợp

### 8.1 Điều đã xác minh được

`Code-evidenced`

- `keyword-density` có implementation đầy đủ trong `walkvlm`
- metric này là semantic coverage metric dựa trên CLIP, unigram unique, bigram unique, và cosine threshold `0.9`
- metric này được dùng cả ở test offline và GRPO reward
- `GPTScore` được repo gọi tới qua `evaluate_image(...)`
- `GPTScore` cần external credential/API
- repo công khai hiện tại thiếu file `GPTScore.py`

### 8.2 Điều chỉ có thể suy ra

`Repo-structure inference`

- `GPTScore` nhiều khả năng là một teacher/judge metric dựa trên `qwen72b_output` và `generated_text`
- `GPTScore` không có vẻ là metric prediction-vs-human-reference thuần
- `keyword-density` có mục tiêu khuyến khích output cô đọng nhưng vẫn bao phủ semantic keyword

### 8.3 Giới hạn không thể vượt qua nếu chỉ dựa trên repo công khai

`Missing-source limitation`

- không thể viết lại nội bộ `GPTScore.py` một cách trung thực
- không thể khẳng định score scale, judge prompt, hay cách API trả điểm
- không thể khẳng định `GPTScore` có dùng ảnh trong request hay không

## 9. Gợi ý sử dụng cho repo hiện tại

Không phải để fix ngay, chỉ là kết luận research:

1. `keyword-density` là metric có thể tái hiện nếu bạn khôi phục được field `keywords` và giữ nguyên CLIP setup.
2. `GPTScore` hiện chưa đủ điều kiện để đưa vào benchmark chính của repo hiện tại, vì nó thiếu source và phụ thuộc teacher signal.
3. Nếu ưu tiên tính công bằng và khả năng tái lập, bộ metric hiện tại nên vẫn ưu tiên:
   - ROUGE
   - TF-IDF
   - và nếu cần, thêm metric mới nhưng phải có source đầy đủ trong repo.
