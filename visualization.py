import cv2
import supervision as sv
import numpy as np
from typing import List, Optional

class Visualizer:
    """Handles all visualization and annotation tasks"""
    
    def __init__(self, config):
        self.config = config
        self.setup_annotators()
        self.setup_zone()
    
    def setup_annotators(self):
        """Initialize supervision annotators"""
        self.label_annotator = sv.LabelAnnotator(
            text_scale=self.config.TEXT_SCALE,
            text_thickness=self.config.TEXT_THICKNESS,
            text_color=sv.Color.WHITE,
        )
        
        self.box_annotator = sv.BoxAnnotator(
            thickness=self.config.BOX_THICKNESS,
        )
        
    def setup_zone(self):
        """Setup the detection zone"""
        active_polygon = self.config.POLYGON_ZONES[self.config.ACTIVE_ZONE_INDEX]
        self.polygon_zone = sv.PolygonZone(polygon=active_polygon)
        
        self.zone_annotator = sv.PolygonZoneAnnotator(
            zone=self.polygon_zone,
            color=sv.Color.RED,
            thickness=self.config.ZONE_THICKNESS,
            text_thickness=self.config.TEXT_THICKNESS,
            text_scale=self.config.TEXT_SCALE
        )
    
    def annotate_frame(self, frame: np.ndarray, detections: sv.Detections, 
                      labels: List[str], current_plate_text: str = "") -> np.ndarray:
        """
        Annotate frame with detections and zone
        
        Args:
            frame: Input frame
            detections: Supervision detections
            labels: Detection labels
            current_plate_text: Current plate text to display
            
        Returns:
            Annotated frame
        """
        # Draw the polygon zone
        frame = self.zone_annotator.annotate(scene=frame)
        
        # Annotate with labels and bounding boxes
        frame = self.label_annotator.annotate(
            scene=frame, 
            detections=detections, 
            labels=labels
        )
        frame = self.box_annotator.annotate(
            scene=frame, 
            detections=detections
        )
        
        # Add plate text display
        if current_plate_text:
            cv2.putText(
                frame, f"Plate: {current_plate_text}", 
                (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4
            )
        
        return frame
    
    def get_zone_detections(self, tracked_detections: sv.Detections) -> sv.Detections:
        """Get detections that are within the zone"""
        in_zone_mask = self.polygon_zone.trigger(tracked_detections)
        return tracked_detections[in_zone_mask]