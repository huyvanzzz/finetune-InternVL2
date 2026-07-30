from types import SimpleNamespace

from scripts.smoke_qformer_bridge import expected_visual_token_count


def test_expected_visual_token_count_uses_model_num_image_token_for_trajectory():
    model = SimpleNamespace(trajectory_enabled=True, num_image_token=38)
    config = {"model": {"qformer": {"num_query_tokens": 32}}}

    assert expected_visual_token_count(model, config) == 38


def test_expected_visual_token_count_uses_qformer_config_without_trajectory():
    model = SimpleNamespace(trajectory_enabled=False, num_image_token=38)
    config = {"model": {"qformer": {"num_query_tokens": 32}}}

    assert expected_visual_token_count(model, config) == 32
