from resume_equivalence import (
    build_verdict,
    make_dataset,
    run_epoch_from_checkpoint,
    run_reference_epoch,
    run_resumed_epoch,
)


def test_mid_epoch_resume_reproduces_data_order_with_sampler_skip_only():
    dataset = make_dataset(12)
    uninterrupted = run_reference_epoch(dataset, epoch_seed=42, dropout_p=0.5, checkpoint_after_steps=6)
    resumed = run_resumed_epoch(dataset, uninterrupted["checkpoint"], restore_rng_state=False)

    assert resumed["sample_order"] == uninterrupted["sample_order"][6:]


def test_mid_epoch_resume_diverges_without_model_rng_restore():
    dataset = make_dataset(12)
    baseline = run_reference_epoch(dataset, epoch_seed=42, dropout_p=0.5, checkpoint_after_steps=6)
    resumed = run_resumed_epoch(dataset, baseline["checkpoint"], restore_rng_state=False)
    verdict = build_verdict(baseline["records"][6:], resumed["records"])

    assert verdict["sample_order_equal"] is True
    assert verdict["verdict"] == "diverges_due_to_rng"


def test_mid_epoch_resume_matches_after_full_rng_restore():
    dataset = make_dataset(12)
    baseline = run_reference_epoch(dataset, epoch_seed=42, dropout_p=0.5, checkpoint_after_steps=6)
    resumed = run_resumed_epoch(dataset, baseline["checkpoint"], restore_rng_state=True)
    verdict = build_verdict(baseline["records"][6:], resumed["records"])

    assert verdict["verdict"] == "fully_equivalent"


def test_epoch_boundary_resume_matches_without_extra_rng_restore():
    dataset = make_dataset(12)
    epoch_zero = run_reference_epoch(dataset, epoch_seed=42, dropout_p=0.5, checkpoint_after_steps=12)
    uninterrupted_epoch_one = run_epoch_from_checkpoint(dataset, epoch_zero["checkpoint"], epoch_seed=43)
    resumed_epoch_one = run_epoch_from_checkpoint(dataset, epoch_zero["checkpoint"], epoch_seed=43)
    verdict = build_verdict(uninterrupted_epoch_one["records"], resumed_epoch_one["records"])

    assert epoch_zero["sample_order"] != []
    assert verdict["verdict"] == "fully_equivalent"


def test_resume_with_dropout_disabled_matches_even_without_rng_restore():
    dataset = make_dataset(12)
    uninterrupted = run_reference_epoch(dataset, epoch_seed=42, dropout_p=0.0, checkpoint_after_steps=6)
    resumed = run_resumed_epoch(dataset, uninterrupted["checkpoint"], restore_rng_state=False)
    verdict = build_verdict(uninterrupted["records"][6:], resumed["records"])

    assert verdict["verdict"] == "fully_equivalent"
