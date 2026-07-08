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

## Ghi chu gate

Pipeline nay chi con 2 gate:

- `polarity_reversal`
- `unsafe_action`

Loi direction duoc cham qua criteria, khong con gate cap rieng.

## API keys

- Tao `.env` o root repo hoac `gptscore/.env`
- Co the bat dau tu file `gptscore/.env.example`
