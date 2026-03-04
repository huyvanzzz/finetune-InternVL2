import json
from dataclasses import dataclass
from typing import List, Dict

from traitlets import Any

@dataclass
class POLMData:
    """POLM structure from bbox annotations"""
    object_type: str
    bbox: List[float]  # [x1, y1, x2, y2]
    relative_position: str
    distance_zone: str
    coming_to_user: bool
    speed: float
    danger_score: float
    def to_text(self) -> str:
        return (
            f"[OBJ] {self.object_type}, "
            f"({self.bbox[0]:.2f}, {self.bbox[1]:.2f}, "
            f"{self.bbox[2]:.2f}, {self.bbox[3]:.2f}), "
            f"pos={self.relative_position}, "
            f"dist={self.distance_zone}, "
            f"approaching={'yes' if self.coming_to_user else 'no'}, "
            f"speed={self.speed:.2f}."
        )

@dataclass
class GroundTruthData:
    """Ground truth for training"""
    location: str
    weather: str
    traffic: str
    scene: str
    instruction: str
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps({
            'location': self.location,
            'weather': self.weather,
            'traffic': self.traffic,
            'scene': self.scene,
            'instruction': self.instruction
        }, ensure_ascii=False)
    
def map_metadata_to_ground_truth(metadata: Dict) -> GroundTruthData:
    """Map WAD metadata to ground truth format"""
    
    # Location mapping
    area_map = {
        'Pedestrian Path': 'pedestrian_path',
        'Road': 'road',
        'Corridor': 'corridor',
        'Busy Street': 'busy_street',
        'Shopping Mall': 'shopping_mall',
        'Bicycle Lane': 'bicycle_lane',
        'Restaurant': 'restaurant',
        'Other': 'other'
    }
    
    # Weather mapping
    weather_map = {
        'Sunny': 'sunny',
        'Overcast': 'overcast',
        'Cloudy': 'cloudy',
        'Night': 'night',
        'Indoor': 'indoor',
        'Other': 'other'
    }
    
    # Traffic mapping
    traffic_map = {
        'High': 'high',
        'Mid': 'moderate',
        'Low': 'low'
    }
    
    location = area_map.get(metadata.get('area_type', 'Other'), 'other')
    weather = weather_map.get(metadata.get('weather_condition', 'Other'), 'other')
    traffic = traffic_map.get(metadata.get('traffic_flow_rating', 'Low'), 'low')
    scene = metadata.get('summary', '')
    
    # Instruction: alter or QA answer
    if metadata.get('QA') and isinstance(metadata['QA'], dict):
        instruction = metadata['QA'].get('A', '')
    elif metadata.get('alter'):
        instruction = metadata['alter']
    else:
        instruction = ''
    
    return GroundTruthData(
        location=location,
        weather=weather,
        traffic=traffic,
        scene=scene,
        instruction=instruction
    )