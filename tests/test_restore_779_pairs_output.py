import json
from pathlib import Path

from scripts.pairs_output import write_prediction_pairs


def test_write_prediction_pairs_creates_minimal_pairs_file(tmp_path):
    output_file = tmp_path / "eval_results.json"
    detailed_results = [
        {
            "id": 0,
            "question": "<image> ignored for alter gptscore",
            "prediction": "there are stairs ahead, walk slowly.",
            "ground_truth": "There are stairs in front, be careful and walk slowly.",
        }
    ]

    pairs_path = write_prediction_pairs(
        str(output_file),
        checkpoint_label="dummy-checkpoint",
        split_name="test_alter",
        detailed_results=detailed_results,
    )

    saved = json.loads(Path(pairs_path).read_text(encoding="utf-8"))
    assert saved["checkpoint"] == "dummy-checkpoint"
    assert saved["split"] == "test_alter"
    assert saved["pairs"] == [
        {
            "id": 0,
            "ground_truth": "There are stairs in front, be careful and walk slowly.",
            "generation": "there are stairs ahead, walk slowly.",
        }
    ]
