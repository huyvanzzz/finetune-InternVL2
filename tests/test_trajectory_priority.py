import math

from trajectory_priority import (
    bbox_area,
    build_grouped_records,
    compute_priority_score,
    direction_weight,
    select_top_objects,
)


def make_row(track_id, *, position, distance_norm, boxs, label="person"):
    return {
        "folder_id": "sample.frame",
        "frame_id": 8,
        "track_id": track_id,
        "label": label,
        "boxs": boxs,
        "distance_norm": distance_norm,
        "relative_position": position,
        "movement_angle": 0.0,
        "speed_percent": 0.0,
    }


def test_direction_weight_matches_clock_priority():
    assert direction_weight("12 o'clock") == 1.0
    assert math.isclose(direction_weight("11 o'clock"), 2 / 3)
    assert math.isclose(direction_weight("1 o'clock"), 2 / 3)
    assert math.isclose(direction_weight("10 o'clock"), 1 / 3)
    assert math.isclose(direction_weight("2 o'clock"), 1 / 3)
    assert direction_weight("3 o'clock") == 0.0
    assert direction_weight("unknown") == 0.0


def test_compute_priority_score_uses_area_proximity_and_direction():
    row = make_row(
        1,
        position="12 o'clock",
        distance_norm=0.25,
        boxs=[0.0, 0.0, 0.5, 0.5],
    )

    assert math.isclose(bbox_area(row["boxs"]), 0.25)
    assert math.isclose(compute_priority_score(row), 0.25 + 0.75 + 1.0)


def test_select_top_objects_keeps_highest_priority_first():
    rows = [
        make_row(1, position="3 o'clock", distance_norm=0.90, boxs=[0.0, 0.0, 0.2, 0.2]),
        make_row(2, position="12 o'clock", distance_norm=0.10, boxs=[0.0, 0.0, 0.2, 0.2]),
        make_row(3, position="11 o'clock", distance_norm=0.20, boxs=[0.0, 0.0, 0.4, 0.4]),
    ]

    selected = select_top_objects(rows, limit=2)

    assert [row["track_id"] for row in selected] == [2, 3]


def test_build_grouped_records_truncates_each_group_to_six_objects():
    rows = []
    for track_id in range(8):
        rows.append(
            make_row(
                track_id,
                position="12 o'clock" if track_id == 7 else "3 o'clock",
                distance_norm=0.9 - (track_id * 0.1),
                boxs=[0.0, 0.0, 0.1 + track_id * 0.01, 0.1 + track_id * 0.01],
            )
        )

    grouped = build_grouped_records(rows, limit=6)

    assert len(grouped) == 1
    assert grouped[0]["source_count"] == 8
    assert grouped[0]["selected_count"] == 6
    assert len(grouped[0]["objects"]) == 6
    assert grouped[0]["objects"][0]["track_id"] == 7
    assert grouped[0]["objects"][0]["rank"] == 1
