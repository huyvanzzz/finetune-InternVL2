BACKEND_NAME = "internvl"


def _not_implemented(*args, **kwargs):
    raise NotImplementedError("InternVL backend adapter is not implemented in this module.")


load_model_and_tokenizer = _not_implemented
build_train_collate_fn = _not_implemented
build_eval_collate_fn = _not_implemented
forward_train_batch = _not_implemented
forward_eval_batch = _not_implemented
generate_response = _not_implemented
attach_qformer_if_enabled = _not_implemented
save_backend_artifacts = _not_implemented
load_backend_artifacts = _not_implemented
