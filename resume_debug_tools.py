import hashlib
import random

import numpy as np
import torch


def format_batch_sample_ids(samples) -> str:
    return ",".join(str(sample.get("questionId", "unknown")) for sample in samples)


def runtime_rng_digest(include_cuda: bool) -> str:
    digest = hashlib.sha256()
    digest.update(repr(random.getstate()).encode("utf-8"))

    numpy_state = np.random.get_state()
    digest.update(str(numpy_state[0]).encode("utf-8"))
    digest.update(numpy_state[1].tobytes())
    digest.update(str(numpy_state[2:]).encode("utf-8"))

    digest.update(torch.get_rng_state().cpu().numpy().tobytes())
    if include_cuda and torch.cuda.is_available():
        for state in torch.cuda.get_rng_state_all():
            digest.update(state.cpu().numpy().tobytes())
    return digest.hexdigest()[:16]
