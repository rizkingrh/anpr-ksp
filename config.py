import numpy as np
from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class Config:
    """Configuration settings for the ANPR system"""
    # API Configuration
    API_URL: str = "http://192.168.30.208:8000/api/store"
    
    # Video Configuration
    # VIDEO_SOURCE: str = "rtsp://admin:ZUHDGC@192.168.30.160"
    VIDEO_SOURCE: str = "sample.mp4"  # For testing
    
    # Model Configuration
    MODEL_ID: str = "indonesia-license-plate-iqrtj/3"
    API_KEY: str = "bwIRBiOk7e1dT6URaaEh"
    OCR_MODEL_PATH: str = "cct-s-v1-global-model"
    
    # Directory Configuration
    OUTPUT_DIR: str = "captured_plates"
    
    # Detection Zones (you can easily modify these)
    POLYGON_ZONES: List[np.ndarray] = None
    ACTIVE_ZONE_INDEX: int = 2
    
    # Annotation Configuration
    TEXT_SCALE: float = 0.5
    TEXT_THICKNESS: int = 1
    BOX_THICKNESS: int = 2
    ZONE_THICKNESS: int = 2
    
    # Other Configuration
    CONFIDENCE_THRESHOLD: float = 0.5
    
    # For polygon zone in real location multiplied by 3
    def __post_init__(self):
        if self.POLYGON_ZONES is None:
            self.POLYGON_ZONES = [
                np.array([[853, 359], [1202, 506], [607, 702], [529, 394]]),
                np.array([[998, 359], [1254, 442], [607, 702], [521, 473]]),
                np.array([[903, 335], [1254, 442], [607, 702], [502, 403]]),
                np.array([[764, 308], [1273, 475], [413, 708], [333, 370]]),
                np.array([[2292, 924], [3819, 1425], [1239, 2124], [999, 1110]])
            ]