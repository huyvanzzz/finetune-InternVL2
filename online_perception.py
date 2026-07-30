import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


TRAJECTORY_TOP_K = 6
DEFAULT_KEEP_LABELS = {
    "person",
    "car",
    "car_(automobile)",
    "motorcycle",
    "truck",
    "bus",
    "bus_(vehicle)",
    "bicycle",
    "traffic_light",
    "stop_sign",
    "bench",
    "chair",
    "trash_can",
    "dog",
    "cat",
}


@dataclass
class SimpleDetection:
    label: str
    confidence: float
    boxs: List[float]
    class_id: int = 0


@dataclass
class TrackState:
    track_id: int
    label: str
    boxs: List[float]
    confidence: float
    frame_id: int
    history: List[Dict] = field(default_factory=list)
    active: bool = True


def compute_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_a = max(0.0, float(box_a[2]) - float(box_a[0])) * max(0.0, float(box_a[3]) - float(box_a[1]))
    area_b = max(0.0, float(box_b[2]) - float(box_b[0])) * max(0.0, float(box_b[3]) - float(box_b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def bbox_center(box: Sequence[float]) -> List[float]:
    return [
        round((float(box[0]) + float(box[2])) / 2.0, 4),
        round((float(box[1]) + float(box[3])) / 2.0, 4),
    ]


def bbox_area_norm(box: Sequence[float]) -> float:
    return round(max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1])), 4)


def distance_norm_from_bbox(box: Sequence[float]) -> float:
    return round(1.0 - float(box[3]), 4)


def relative_position_from_bbox(box: Sequence[float]) -> str:
    cx = (float(box[0]) + float(box[2])) / 2.0
    if cx <= 0.2:
        return "10 o'clock"
    if cx <= 0.4:
        return "11 o'clock"
    if cx <= 0.6:
        return "12 o'clock"
    if cx <= 0.8:
        return "1 o'clock"
    return "2 o'clock"


def motion_from_history(history: List[Dict], current_box: Sequence[float]) -> Dict[str, float]:
    if not history:
        return {"movement_angle": 0.0, "speed_percent": 0.0}
    prev_center = history[-1]["center"]
    current_center = bbox_center(current_box)
    dx = current_center[0] - float(prev_center[0])
    dy = current_center[1] - float(prev_center[1])
    speed_percent = round(math.sqrt(dx * dx + dy * dy) * 100.0, 2)
    angle = round(math.degrees(math.atan2(dy, dx)) / 180.0, 4) if speed_percent > 0 else 0.0
    return {"movement_angle": angle, "speed_percent": speed_percent}


class IoUTracker:
    def __init__(self, iou_threshold: float = 0.3, max_history: int = 8):
        self.iou_threshold = float(iou_threshold)
        self.max_history = int(max_history)
        self.next_track_id = 1
        self.tracks: Dict[int, TrackState] = {}

    def reset(self):
        self.next_track_id = 1
        self.tracks = {}

    def update(self, detections: List[SimpleDetection], frame_id: int) -> List[TrackState]:
        unmatched_track_ids = set(self.tracks)
        updated_tracks: List[TrackState] = []

        for det in detections:
            best_track_id = None
            best_iou = 0.0
            for track_id in list(unmatched_track_ids):
                track = self.tracks[track_id]
                if track.label != det.label:
                    continue
                iou = compute_iou(track.boxs, det.boxs)
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id

            if best_track_id is not None and best_iou >= self.iou_threshold:
                track = self.tracks[best_track_id]
                unmatched_track_ids.remove(best_track_id)
            else:
                track = TrackState(
                    track_id=self.next_track_id,
                    label=det.label,
                    boxs=list(det.boxs),
                    confidence=float(det.confidence),
                    frame_id=int(frame_id),
                )
                self.next_track_id += 1

            track.history.append(
                {
                    "frame_id": track.frame_id,
                    "boxs": list(track.boxs),
                    "center": bbox_center(track.boxs),
                    "confidence": track.confidence,
                }
            )
            track.history = track.history[-self.max_history :]
            track.boxs = list(det.boxs)
            track.confidence = float(det.confidence)
            track.frame_id = int(frame_id)
            track.active = True
            self.tracks[track.track_id] = track
            updated_tracks.append(track)

        for track_id in unmatched_track_ids:
            self.tracks[track_id].active = False

        return updated_tracks


def track_to_object(track: TrackState) -> Dict:
    motion = motion_from_history(track.history, track.boxs)
    return {
        "track_id": int(track.track_id),
        "label": str(track.label),
        "probs": float(track.confidence),
        "boxs": [float(v) for v in track.boxs],
        "center": bbox_center(track.boxs),
        "area_norm": bbox_area_norm(track.boxs),
        "distance_norm": distance_norm_from_bbox(track.boxs),
        "relative_position": relative_position_from_bbox(track.boxs),
        "movement_angle": motion["movement_angle"],
        "speed_percent": motion["speed_percent"],
        "active": bool(track.active),
    }


def select_top6_objects(objects: List[Dict], top_k: int = TRAJECTORY_TOP_K) -> List[Dict]:
    ranked = sorted(
        objects,
        key=lambda obj: (
            bool(obj.get("active", False)),
            float(obj.get("area_norm", 0.0)),
            -float(obj.get("distance_norm", 1.0)),
            float(obj.get("probs", obj.get("confidence", 0.0))),
        ),
        reverse=True,
    )
    return ranked[: int(top_k)]


def create_object_tracking_timing(mode: str, timing: Optional[Dict[str, float]] = None) -> Dict[str, object]:
    if mode == "disabled":
        return {
            "object_tracking_included": False,
            "object_detection_ms": 0.0,
            "tracking_ms": 0.0,
            "top6_selection_ms": 0.0,
            "object_tracking_ms": 0.0,
        }
    timing = timing or {}
    return {
        "object_tracking_included": True,
        "object_detection_ms": float(timing.get("object_detection_ms", 0.0)),
        "tracking_ms": float(timing.get("tracking_ms", 0.0)),
        "top6_selection_ms": float(timing.get("top6_selection_ms", 0.0)),
        "object_tracking_ms": float(timing.get("object_tracking_ms", 0.0)),
    }


class YoloDetectorAdapter:
    def __init__(self, model_path: str = "yolo11n.pt", keep_labels: Optional[Iterable[str]] = None):
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise ImportError("online_fast perception requires `ultralytics`. Install it before running.") from exc
        self.model = YOLO(model_path)
        self.keep_labels = set(keep_labels or DEFAULT_KEEP_LABELS)

    def __call__(self, frame) -> List[SimpleDetection]:
        results = self.model(frame, verbose=False)
        detections = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            label = str(self.model.names[cls_id])
            if self.keep_labels and label not in self.keep_labels:
                continue
            detections.append(
                SimpleDetection(
                    label=label,
                    confidence=float(box.conf[0].item()),
                    boxs=[float(v) for v in box.xyxyn[0].detach().cpu().numpy().tolist()],
                    class_id=cls_id,
                )
            )
        return detections


class OnlineFastPerceptionEngine:
    def __init__(self, detector=None, *, iou_threshold: float = 0.3, max_history: int = 8, top_k: int = TRAJECTORY_TOP_K):
        self.detector = detector if detector is not None else YoloDetectorAdapter()
        self.tracker = IoUTracker(iou_threshold=iou_threshold, max_history=max_history)
        self.top_k = int(top_k)
        self.sequence_id = None

    def reset_sequence(self, sequence_id: str):
        self.sequence_id = str(sequence_id)
        self.tracker.reset()

    def update(self, frame, frame_id: int, folder_id: str) -> Dict:
        if self.sequence_id != str(folder_id):
            self.reset_sequence(str(folder_id))

        start = time.perf_counter()
        detections = self.detector(frame)
        detection_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        tracks = self.tracker.update(detections, frame_id=frame_id)
        objects = [track_to_object(track) for track in tracks]
        tracking_ms = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        top_objects = select_top6_objects(objects, self.top_k)
        top6_ms = (time.perf_counter() - start) * 1000.0
        timing = {
            "object_detection_ms": detection_ms,
            "tracking_ms": tracking_ms,
            "top6_selection_ms": top6_ms,
            "object_tracking_ms": detection_ms + tracking_ms + top6_ms,
        }

        return {
            "folder_id": str(folder_id),
            "frame_id": int(frame_id),
            "objects": top_objects,
            "timing": timing,
        }


def load_image(path: Path):
    try:
        import cv2
    except Exception as exc:
        raise ImportError("perception metadata runner requires `opencv-python` for image loading.") from exc
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image
