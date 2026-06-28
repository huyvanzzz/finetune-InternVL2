import random

import numpy as np
import torch

from resume_debug_tools import format_batch_sample_ids, runtime_rng_digest


def test_format_batch_sample_ids_returns_compact_question_ids():
    samples = [
        {"questionId": "12"},
        {"questionId": "99"},
        {"questionId": "105"},
    ]

    assert format_batch_sample_ids(samples) == "12,99,105"


def test_runtime_rng_digest_changes_after_rng_consumption():
    random.seed(5)
    np.random.seed(5)
    torch.manual_seed(5)

    before = runtime_rng_digest(include_cuda=False)
    random.random()
    np.random.rand()
    torch.rand(1)
    after = runtime_rng_digest(include_cuda=False)

    assert before != after
