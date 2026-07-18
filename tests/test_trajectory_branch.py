import json
from pathlib import Path

import pytest
import torch

from trajectory_branch import (
    TRAJECTORY_CANONICAL_FILENAME,
    TRAJECTORY_CANONICAL_JSONL_FILENAME,
    TRAJECTORY_BRANCH_CONFIG_NAME,
    TRAJECTORY_BRANCH_WEIGHTS_NAME,
    TrajectoryBackbone,
    TrajectoryCLSHead,
    TrajectoryConcatHead,
    TrajectorySource,
    attach_trajectory_branch,
    load_trajectory_branch,
    resolve_trajectory_source_path,
    save_trajectory_branch,
)


def _write_canonical(path: Path, records):
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def test_trajectory_source_builds_stable_vocab_and_exact_join(tmp_path):
    source_path = tmp_path / "traj.json"
    _write_canonical(
        source_path,
        [
            {
                "folder_id": "sample_a.frame",
                "frame_id": 8,
                "objects": [
                    {
                        "label": "chair",
                        "relative_position": "12 o'clock",
                        "x1": 0.1,
                        "y1": 0.2,
                        "x2": 0.5,
                        "y2": 0.4,
                        "movement_angle": 15.0 / 180.0,
                        "speed_percent": 0.3,
                    },
                    {
                        "label": "pedestrian",
                        "relative_position": "10 o'clock",
                        "x1": 0.2,
                        "y1": 0.3,
                        "x2": 0.6,
                        "y2": 0.7,
                        "movement_angle": 2.0 / 180.0,
                        "speed_percent": 0.1,
                    },
                ],
            },
            {
                "folder_id": "sample_b.frame",
                "frame_id": 6,
                "objects": [],
            },
        ],
    )

    source = TrajectorySource.from_file(str(source_path))
    encoded = source.encode("sample_a.frame", 8)

    assert source.label_to_id == {"<PAD_UNK>": 0, "chair": 1, "pedestrian": 2}
    assert source.direction_to_id == {"<PAD_UNK>": 0, "10 o'clock": 1, "12 o'clock": 2}
    assert encoded["trajectory_label_ids"].tolist() == [1, 2, 0, 0, 0, 0]
    assert encoded["trajectory_direction_ids"].tolist() == [2, 1, 0, 0, 0, 0]
    assert encoded["trajectory_object_mask"].tolist() == [1, 1, 0, 0, 0, 0]
    assert encoded["trajectory_numeric_feats"].shape == (6, 6)
    assert source.has_record("sample_a.frame", 8) is True
    assert source.has_record("sample_a.frame", 7) is False


def test_resolve_trajectory_source_path_accepts_directory(tmp_path):
    image_dir = tmp_path / "image"
    image_dir.mkdir()
    resolved = resolve_trajectory_source_path(str(image_dir))
    assert Path(resolved) == image_dir / TRAJECTORY_CANONICAL_JSONL_FILENAME


def test_resolve_trajectory_source_path_prefers_json_when_present(tmp_path):
    image_dir = tmp_path / "image"
    image_dir.mkdir()
    (image_dir / TRAJECTORY_CANONICAL_FILENAME).write_text("[]", encoding="utf-8")
    resolved = resolve_trajectory_source_path(str(image_dir))
    assert Path(resolved) == image_dir / TRAJECTORY_CANONICAL_FILENAME


def test_trajectory_source_returns_empty_object_sample_for_missing_record(tmp_path):
    source_path = tmp_path / "traj.json"
    _write_canonical(
        source_path,
        [
            {
                "folder_id": "sample_a.frame",
                "frame_id": 8,
                "objects": [],
            }
        ],
    )
    source = TrajectorySource.from_file(str(source_path))
    encoded = source.encode("missing.frame", 9)

    assert encoded["trajectory_label_ids"].tolist() == [0, 0, 0, 0, 0, 0]
    assert encoded["trajectory_direction_ids"].tolist() == [0, 0, 0, 0, 0, 0]
    assert encoded["trajectory_object_mask"].tolist() == [0, 0, 0, 0, 0, 0]
    assert torch.equal(encoded["trajectory_numeric_feats"], torch.zeros(6, 6))


def test_trajectory_source_accepts_flat_jsonl_and_normalizes_fields(tmp_path):
    source_path = tmp_path / TRAJECTORY_CANONICAL_JSONL_FILENAME
    rows = [
        {
            "folder_id": "sample_a.frame",
            "frame_id": 8,
            "track_id": 1,
            "label": "chair",
            "boxs": [0.1, 0.2, 0.5, 0.4],
            "relative_position": "12 o'clock",
            "movement_angle": 15.0,
            "speed_percent": 0.3,
        },
        {
            "folder_id": "sample_a.frame",
            "frame_id": 8,
            "track_id": 2,
            "label": "pedestrian",
            "boxs": [0.2, 0.3, 0.6, 0.7],
            "relative_position": "10 o'clock",
            "movement_angle": 2.0,
            "speed_percent": 0.1,
        },
    ]
    source_path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    source = TrajectorySource.from_file(str(source_path))
    encoded = source.encode("sample_a.frame", 8)

    assert encoded["trajectory_label_ids"].tolist() == [1, 2, 0, 0, 0, 0]
    assert encoded["trajectory_direction_ids"].tolist() == [2, 1, 0, 0, 0, 0]
    assert encoded["trajectory_numeric_feats"][0].tolist() == pytest.approx([0.1, 0.2, 0.5, 0.4, 15.0 / 180.0, 0.3])
    assert encoded["trajectory_numeric_feats"][1].tolist() == pytest.approx([0.2, 0.3, 0.6, 0.7, 2.0 / 180.0, 0.1])


def test_trajectory_source_rejects_invalid_record_format(tmp_path):
    source_path = tmp_path / "traj.json"
    _write_canonical(
        source_path,
        [
            {
                "folder_id": "broken.frame",
                "frame_id": 8,
                "objects": [
                    {
                        "label": "chair",
                        "x1": 0.1,
                    }
                ],
            }
        ],
    )

    with pytest.raises(ValueError, match="missing required fields"):
        TrajectorySource.from_file(str(source_path))


def test_trajectory_backbone_and_heads_shape_contract():
    backbone = TrajectoryBackbone(vocab_size=8, direction_vocab_size=6)
    cls_head = TrajectoryCLSHead(output_dim=1024)
    concat_head = TrajectoryConcatHead(output_dim=896)

    label_ids = torch.tensor([[1, 2, 0, 0, 0, 0], [3, 4, 5, 6, 7, 0]], dtype=torch.long)
    direction_ids = torch.tensor([[1, 2, 0, 0, 0, 0], [3, 4, 5, 1, 2, 0]], dtype=torch.long)
    numeric_feats = torch.randn(2, 6, 6)
    object_mask = torch.tensor([[1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0]], dtype=torch.long)

    traj_tokens = backbone(label_ids=label_ids, direction_ids=direction_ids, numeric_feats=numeric_feats, object_mask=object_mask)

    assert traj_tokens.shape == (2, 6, 128)
    assert cls_head(traj_tokens, object_mask).shape == (2, 1, 1024)
    assert concat_head(traj_tokens).shape == (2, 6, 896)


def test_trajectory_backbone_handles_all_empty_mask_without_nan():
    backbone = TrajectoryBackbone(vocab_size=4, direction_vocab_size=4)
    cls_head = TrajectoryCLSHead(output_dim=1024)
    concat_head = TrajectoryConcatHead(output_dim=896)
    label_ids = torch.zeros(2, 6, dtype=torch.long)
    direction_ids = torch.zeros(2, 6, dtype=torch.long)
    numeric_feats = torch.zeros(2, 6, 6)
    object_mask = torch.zeros(2, 6, dtype=torch.long)

    traj_tokens = backbone(label_ids=label_ids, direction_ids=direction_ids, numeric_feats=numeric_feats, object_mask=object_mask)

    assert traj_tokens.shape == (2, 6, 128)
    assert torch.isnan(traj_tokens).sum().item() == 0
    assert torch.equal(cls_head(traj_tokens, object_mask), torch.zeros(2, 1, 1024))
    assert torch.equal(
        concat_head(traj_tokens) * object_mask.to(torch.float32).unsqueeze(-1),
        torch.zeros(2, 6, 896),
    )


def test_trajectory_backbone_records_stage_debug_when_enabled():
    backbone = TrajectoryBackbone(vocab_size=8, direction_vocab_size=6)
    backbone._enable_trajectory_grad_debug = True
    label_ids = torch.tensor([[1, 2, 0, 0, 0, 0]], dtype=torch.long)
    direction_ids = torch.tensor([[1, 2, 0, 0, 0, 0]], dtype=torch.long)
    numeric_feats = torch.randn(1, 6, 6)
    object_mask = torch.tensor([[1, 1, 0, 0, 0, 0]], dtype=torch.long)

    _ = backbone(label_ids=label_ids, direction_ids=direction_ids, numeric_feats=numeric_feats, object_mask=object_mask)

    debug = backbone._last_stage_debug
    assert debug["label_embeds_abs_mean"] > 0
    assert debug["direction_embeds_abs_mean"] > 0
    assert "numeric_embeds_abs_mean" in debug
    assert "tokens_before_mask_abs_mean" in debug
    assert "tokens_after_encoder_abs_mean" in debug


class _TinyTrajectoryModel(torch.nn.Module):
    def __init__(self, fusion_mode: str):
        super().__init__()
        self.trajectory_enabled = True
        self.trajectory_fusion_mode = fusion_mode
        self.trajectory_source_file = "image/results_botsort_top6_sorted.json"
        self.trajectory_backbone = TrajectoryBackbone(vocab_size=5, direction_vocab_size=5)
        self.trajectory_cls_head = TrajectoryCLSHead(output_dim=1024)
        self.trajectory_token_projector = TrajectoryConcatHead(output_dim=896)


def test_trajectory_save_load_round_trip_and_metadata(tmp_path):
    model = _TinyTrajectoryModel("cls_add")
    save_trajectory_branch(model, str(tmp_path))

    assert (tmp_path / TRAJECTORY_BRANCH_WEIGHTS_NAME).exists()
    metadata = json.loads((tmp_path / TRAJECTORY_BRANCH_CONFIG_NAME).read_text(encoding="utf-8"))
    assert metadata["fusion_mode"] == "cls_add"
    assert metadata["source_file"] == "image/results_botsort_top6_sorted.json"

    model2 = _TinyTrajectoryModel("cls_add")
    assert load_trajectory_branch(model2, str(tmp_path), strict=True) is True


def test_trajectory_load_fails_fast_on_mode_mismatch(tmp_path):
    model = _TinyTrajectoryModel("cls_add")
    save_trajectory_branch(model, str(tmp_path))

    mismatch = _TinyTrajectoryModel("concat")
    with pytest.raises(ValueError, match="Trajectory fusion mode mismatch"):
        load_trajectory_branch(mismatch, str(tmp_path), strict=True)


class _AttachDummy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qformer_enabled = True
        self.qformer_num_query_tokens = 32
        self.num_image_token = 32


def test_attach_trajectory_branch_updates_num_image_token_by_mode(tmp_path):
    source_path = tmp_path / "traj.json"
    _write_canonical(
        source_path,
        [{"folder_id": "sample.frame", "frame_id": 8, "objects": []}],
    )
    base_cfg = {
        "trajectory": {
            "enabled": True,
            "source_file": str(source_path),
            "num_objects": 6,
            "d_cat": 32,
            "d_numeric_hidden": 64,
            "d_traj": 128,
            "num_heads": 4,
            "num_layers": 2,
            "ffn_dim": 256,
        }
    }

    cls_model = _AttachDummy()
    attach_trajectory_branch(
        cls_model,
        {**base_cfg, "trajectory": {**base_cfg["trajectory"], "fusion_mode": "cls_add"}},
        pixel_shuffle_dim=1024,
        llm_hidden_size=896,
    )
    assert cls_model.num_image_token == 32

    concat_model = _AttachDummy()
    attach_trajectory_branch(
        concat_model,
        {**base_cfg, "trajectory": {**base_cfg["trajectory"], "fusion_mode": "concat"}},
        pixel_shuffle_dim=1024,
        llm_hidden_size=896,
    )
    assert concat_model.num_image_token == 38

    dual_model = _AttachDummy()
    attach_trajectory_branch(
        dual_model,
        {**base_cfg, "trajectory": {**base_cfg["trajectory"], "fusion_mode": "dual"}},
        pixel_shuffle_dim=1024,
        llm_hidden_size=896,
    )
    assert dual_model.num_image_token == 38


def test_pretrain_configs_use_upscaled_trajectory_architecture():
    expected_snippets = (
        "d_traj: 384",
        "num_layers: 4",
        "ffn_dim: 768",
    )

    for config_name in (
        "internvl_pretrain_config_traj_cls.yaml",
        "internvl_pretrain_config_traj_concat.yaml",
        "internvl_pretrain_config_traj_dual.yaml",
    ):
        content = Path(config_name).read_text(encoding="utf-8")
        for snippet in expected_snippets:
            assert snippet in content
