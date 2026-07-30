from online_perception import (
    OnlineFastPerceptionEngine,
    SimpleDetection,
    create_object_tracking_timing,
    select_top6_objects,
)


def test_top6_selector_prefers_active_large_and_near_objects():
    objects = [
        {"track_id": 1, "label": "far_small", "area_norm": 0.1, "distance_norm": 0.9, "active": True},
        {"track_id": 2, "label": "near_large", "area_norm": 0.8, "distance_norm": 0.1, "active": True},
        {"track_id": 3, "label": "inactive_large", "area_norm": 1.0, "distance_norm": 0.0, "active": False},
    ]

    selected = select_top6_objects(objects)

    assert [obj["track_id"] for obj in selected] == [2, 1, 3]


def test_online_engine_resets_history_between_sequences():
    class Detector:
        names = {0: "person"}

        def __call__(self, frame):
            return [
                SimpleDetection(
                    label="person",
                    confidence=0.9,
                    boxs=[0.1, 0.1, 0.3, 0.5],
                    class_id=0,
                )
            ]

    engine = OnlineFastPerceptionEngine(detector=Detector())

    first = engine.update(frame=object(), frame_id=0, folder_id="seq_a")
    engine.reset_sequence("seq_b")
    second = engine.update(frame=object(), frame_id=0, folder_id="seq_b")

    assert first["objects"][0]["track_id"] == 1
    assert second["objects"][0]["track_id"] == 1
    assert first["folder_id"] == "seq_a"
    assert second["folder_id"] == "seq_b"


def test_online_engine_outputs_trajectory_schema_and_timing():
    class Detector:
        names = {0: "person"}

        def __call__(self, frame):
            return [
                SimpleDetection(
                    label="person",
                    confidence=0.9,
                    boxs=[0.25, 0.2, 0.75, 0.9],
                    class_id=0,
                )
            ]

    engine = OnlineFastPerceptionEngine(detector=Detector())
    record = engine.update(frame=object(), frame_id=8, folder_id="seq")
    obj = record["objects"][0]

    assert record["frame_id"] == 8
    assert obj["label"] == "person"
    assert obj["relative_position"] == "12 o'clock"
    assert obj["movement_angle"] == 0.0
    assert obj["speed_percent"] == 0.0
    assert set(["object_detection_ms", "tracking_ms", "top6_selection_ms", "object_tracking_ms"]).issubset(record["timing"])
    assert record["timing"]["object_tracking_ms"] >= record["timing"]["object_detection_ms"]


def test_create_object_tracking_timing_disabled_and_online():
    disabled = create_object_tracking_timing("disabled")
    online = create_object_tracking_timing(
        "online_fast",
        {
            "object_detection_ms": 1.0,
            "tracking_ms": 2.0,
            "top6_selection_ms": 3.0,
            "object_tracking_ms": 6.0,
        },
    )

    assert disabled["object_tracking_included"] is False
    assert disabled["object_tracking_ms"] == 0.0
    assert online["object_tracking_included"] is True
    assert online["object_detection_ms"] == 1.0
    assert online["tracking_ms"] == 2.0
    assert online["top6_selection_ms"] == 3.0
    assert online["object_tracking_ms"] == 6.0
