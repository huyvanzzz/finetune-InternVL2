# GPTScore

Thu muc nay chua pipeline GPTScore cho task `alter`.

## Workflow

1. Dung `scripts/test_infer.py` de sinh `*_pairs.json`
2. Chay judge:

```bash
python -m gptscore.run_judge --input results/qformer_eval_test_alter_pairs.json --provider openai
```

Mac dinh prompt profile la `variant_action_looser`.
Co the chon `baseline_142` neu muon:

```bash
python -m gptscore.run_judge --input results/qformer_eval_test_alter_pairs.json --provider openai --prompt-profile baseline_142
```

Hoac voi OpenRouter:

```bash
python -m gptscore.run_judge --input results/qformer_eval_test_alter_pairs.json --provider openrouter
```

3. Neu co mau loi, sua tay trong file `*_gptscore_judged.json`
4. Chay buoc scoring:

```bash
python -m gptscore.score_results --input results/qformer_eval_test_alter_gptscore_judged.json
```

## Ghi chu scoring

Pipeline nay khong con gate.

Diem cuoi duoc tinh bang trung binh tren cac criterion `applicable`:

- `Fail -> 0`
- `Weak -> 1`
- `Acceptable -> 2`
- `Strong -> 3`
- `label = null` bi loai khoi mean

## Chay nhieu dot trong cung 1 file judged

Co the chay theo tung doan va noi vao cung 1 file:

```bash
python -m gptscore.run_judge --input results/qformer_eval_test_alter_pairs.json --provider openrouter --offset 0 --limit 100
python -m gptscore.run_judge --input results/qformer_eval_test_alter_pairs.json --provider openrouter --offset 100 --limit 200
```

Mac dinh output van la mot file judged duy nhat suy ra tu input.

- Neu file judged da ton tai, pipeline se load lai va bo qua cac `id` da cham
- Pipeline flush file lien tuc trong luc chay, nen neu het API / dung giua chung thi van giu duoc cac item da cham

## API keys

- Tao `.env` o root repo hoac `gptscore/.env`
- Co the bat dau tu file `gptscore/.env.example`
