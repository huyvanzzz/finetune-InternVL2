python -m gptscore.run_judge --input input_json/qformer_eval_test_alter_pairs.json --provider openrouter --model openai/gpt-4o --offset 0 --limit 100
python -m gptscore.run_judge --input results/qformer_eval_test_alter_pairs.json --provider openrouter --model openai/gpt-4o --offset 100 --limit 200
python -m gptscore.score_results --input results/qformer_eval_test_alter_gptscore_judged.json