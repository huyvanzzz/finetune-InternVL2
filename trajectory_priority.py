import json
from collections import defaultdict
from pathlib import Path


DIRECTION_WEIGHTS = {
    "12 o'clock": 1.0,
    "11 o'clock": 2 / 3,
    "1 o'clock": 2 / 3,
    "10 o'clock": 1 / 3,
    "2 o'clock": 1 / 3,
}


def direction_weight(relative_position):
    return float(DIRECTION_WEIGHTS.get(relative_position, 0.0))


def bbox_area(boxs):
    x1, y1, x2, y2 = boxs
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def proximity_score(distance_norm):
    return max(0.0, 1.0 - float(distance_norm))


def compute_priority_score(row):
    return (
        bbox_area(row["boxs"])
        + proximity_score(row["distance_norm"])
        + direction_weight(row["relative_position"])
    )


def enrich_row(row):
    enriched = dict(row)
    enriched["area"] = bbox_area(row["boxs"])
    enriched["proximity_score"] = proximity_score(row["distance_norm"])
    enriched["direction_weight"] = direction_weight(row["relative_position"])
    enriched["priority_score"] = compute_priority_score(row)
    return enriched


def select_top_objects(rows, limit=6):
    enriched_rows = [enrich_row(row) for row in rows]
    return sorted(
        enriched_rows,
        key=lambda row: (
            -row["priority_score"],
            -row["area"],
            row["distance_norm"],
            row["track_id"],
        ),
    )[:limit]


def build_grouped_records(rows, limit=6):
    grouped_rows = defaultdict(list)
    for row in rows:
        grouped_rows[(row["folder_id"], row["frame_id"])].append(row)

    records = []
    for (folder_id, frame_id), group_rows in sorted(grouped_rows.items()):
        selected_rows = select_top_objects(group_rows, limit=limit)
        objects = []
        for rank, row in enumerate(selected_rows, start=1):
            objects.append(
                {
                    "rank": rank,
                    "track_id": row["track_id"],
                    "label": row["label"],
                    "boxs": row["boxs"],
                    "distance_norm": row["distance_norm"],
                    "relative_position": row["relative_position"],
                    "movement_angle": row["movement_angle"],
                    "speed_percent": row["speed_percent"],
                    "area": row["area"],
                    "proximity_score": row["proximity_score"],
                    "direction_weight": row["direction_weight"],
                    "priority_score": row["priority_score"],
                }
            )

        records.append(
            {
                "folder_id": folder_id,
                "frame_id": frame_id,
                "source_count": len(group_rows),
                "selected_count": len(objects),
                "objects": objects,
            }
        )
    return records


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path, records):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
