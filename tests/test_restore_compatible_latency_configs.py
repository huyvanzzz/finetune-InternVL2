import yaml


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_internvl_qformer_latency_config_uses_restore_compatible_prompt():
    config = load_yaml("internvl_config.yaml")

    assert config["data"]["direct_text_alter_prompt_mode"] == "fixed_779"
    assert config["data"]["direct_text_qa_prompt_mode"] == "legacy_779"
    assert config["data"]["non_train_error_policy"] == "resample"
    assert config["evaluation"]["batch_size"] == 1
    assert config["model"]["qformer"]["enabled"] is True
    assert config["model"]["qformer"]["num_query_tokens"] == 32


def test_internvl_no_qformer_config_only_changes_qformer_and_output_identity():
    qformer_config = load_yaml("internvl_config.yaml")
    no_qformer_config = load_yaml("internvl_config_no_qformer.yaml")

    q_model = dict(qformer_config["model"])
    n_model = dict(no_qformer_config["model"])
    q_model["qformer"] = dict(q_model["qformer"])
    n_model["qformer"] = dict(n_model["qformer"])
    q_model["qformer"]["enabled"] = False

    assert n_model == q_model
    assert no_qformer_config["data"] == qformer_config["data"]
    assert no_qformer_config["evaluation"] == qformer_config["evaluation"]
    assert no_qformer_config["experiment"]["name"] != qformer_config["experiment"]["name"]
    assert no_qformer_config["training"]["output_dir"] != qformer_config["training"]["output_dir"]


def assert_trajectory_config_matches_training_branch(config, fusion_mode):
    assert config["model"]["architecture"] == "internvl"
    assert config["model"]["name"] == "OpenGVLab/InternVL2_5-2B"
    assert config["model"]["qformer"] == {
        "enabled": True,
        "source_model": "Salesforce/instructblip-flan-t5-xl",
        "cache_dir": "./qformer_cache",
        "num_query_tokens": 32,
        "freeze_qformer": True,
        "freeze_mlp1": True,
        "prompt_aware": True,
        "max_text_length": 128,
        "train_lora_llm": True,
        "bridge_mode": "prompt_aware_preproj_mlp1",
    }
    assert config["trajectory"] == {
        "enabled": True,
        "fusion_mode": fusion_mode,
        "source_file": "json",
        "num_objects": 6,
        "d_cat": 32,
        "d_dir": 16,
        "d_numeric_hidden": 64,
        "d_traj": 384,
        "num_heads": 4,
        "num_layers": 4,
        "ffn_dim": 768,
    }
    assert config["data"] == {
        "name": "minhdang0901/WAD_Images_All_Size",
        "num_frames": 1,
        "train_split": 0.9,
        "seed": 42,
        "response_format": "direct_text",
        "alter_only": True,
        "direct_text_alter_prompt_mode": "fixed_779",
        "direct_text_qa_prompt_mode": "legacy_779",
    }
    assert config["training"]["batch_size"] == 2
    assert config["training"]["gradient_accumulation_steps"] == 8
    assert config["training"]["bf16"] is True
    assert config["training"]["fp16"] is False
    assert config["training"]["gradient_checkpointing"] is True
    assert config["evaluation"]["batch_size"] == 1


def test_trajectory_cls_config_matches_trajectory_restore_training_branch():
    config = load_yaml("internvl_config_traj_cls.yaml")

    assert_trajectory_config_matches_training_branch(config, "cls_add")


def test_trajectory_concat_config_matches_trajectory_restore_training_branch():
    config = load_yaml("internvl_config_traj_concat.yaml")

    assert_trajectory_config_matches_training_branch(config, "concat")
