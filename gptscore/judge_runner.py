from datetime import datetime, timezone

from gptscore.io_utils import save_json, select_pair_items
from gptscore.validation import validate_input_pair, validate_judge_output


def _default_emit(message):
    print(message)


def _default_progress_factory(iterable, total=None, desc=None):
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, total=total, desc=desc)


def _emit_preview(selected_pairs, preview_count, emit):
    if preview_count <= 0 or not selected_pairs:
        return
    emit(f"Previewing {min(preview_count, len(selected_pairs))} selected sample(s):")
    for pair in selected_pairs[:preview_count]:
        emit(f"  id={pair.get('id')}")
        emit(f"  ground_truth: {pair.get('ground_truth')}")
        emit(f"  generation: {pair.get('generation')}")


def judge_pairs_document(
    pairs_doc,
    provider,
    model,
    judge_callable,
    prompt_version,
    schema_version,
    limit=None,
    offset=0,
    sample_mode="head",
    sample_seed=0,
    preview_count=1,
    emit=None,
    show_progress=True,
    progress_factory=None,
    existing_judged=None,
    output_path=None,
    flush_every=1,
):
    selected_pairs = select_pair_items(
        pairs_doc,
        limit=limit,
        offset=offset,
        sample_mode=sample_mode,
        sample_seed=sample_seed,
    )
    emit = emit or _default_emit
    progress_factory = progress_factory or _default_progress_factory
    existing_judged = existing_judged or {}
    existing_items = list(existing_judged.get("items", []))
    processed_ids = {
        item.get("id")
        for item in existing_items
        if item.get("judge_status") is not None
    }

    judged = {
        "provider": provider,
        "model": model,
        "prompt_version": prompt_version,
        "schema_version": schema_version,
        "checkpoint": pairs_doc.get("checkpoint"),
        "split": pairs_doc.get("split"),
        "total_input_pairs": len(pairs_doc.get("pairs", [])),
        "selected_count": len(selected_pairs),
        "offset": int(offset or 0),
        "sample_mode": sample_mode,
        "sample_seed": sample_seed,
        "created_at": existing_judged.get("created_at") or datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "items": existing_items,
    }

    _emit_preview(selected_pairs, preview_count, emit)

    iterable = enumerate(selected_pairs)
    if show_progress:
        iterable = progress_factory(iterable, total=len(selected_pairs), desc="Judging pairs")

    appended_since_flush = 0
    for index, pair in iterable:
        pair_id = pair.get("id", index + int(offset or 0))
        if pair_id in processed_ids:
            continue

        item = {
            "id": pair_id,
            "ground_truth": pair.get("ground_truth"),
            "generation": pair.get("generation"),
            "judge_status": None,
            "raw_response": None,
            "judge_output": None,
            "error_type": None,
            "error_message": None,
        }

        input_errors = validate_input_pair(pair)
        if input_errors:
            item["judge_status"] = "input_invalid"
            item["error_type"] = "input_invalid"
            item["error_message"] = "; ".join(input_errors)
            judged["items"].append(item)
            continue

        try:
            judge_output, raw_response = judge_callable(
                str(pair["ground_truth"]),
                str(pair["generation"]),
            )
            item["raw_response"] = raw_response
            item["judge_output"] = judge_output
        except RuntimeError as exc:
            item["judge_status"] = "transport_error"
            item["error_type"] = "transport_error"
            item["error_message"] = str(exc)
            judged["items"].append(item)
            continue
        except ValueError as exc:
            item["judge_status"] = "parse_failed"
            item["error_type"] = "parse_failed"
            item["error_message"] = str(exc)
            judged["items"].append(item)
            continue

        output_errors = validate_judge_output(judge_output)
        if output_errors:
            item["judge_status"] = "schema_invalid"
            item["error_type"] = "schema_invalid"
            item["error_message"] = "; ".join(output_errors)
        else:
            item["judge_status"] = "scored"

        judged["items"].append(item)
        processed_ids.add(pair_id)
        appended_since_flush += 1

        if output_path and flush_every and appended_since_flush >= int(flush_every):
            judged["updated_at"] = datetime.now(timezone.utc).isoformat()
            save_json(output_path, judged)
            appended_since_flush = 0

    judged["selected_count"] = len(selected_pairs)
    judged["processed_count"] = len(processed_ids)
    judged["updated_at"] = datetime.now(timezone.utc).isoformat()
    if output_path:
        save_json(output_path, judged)

    return judged
