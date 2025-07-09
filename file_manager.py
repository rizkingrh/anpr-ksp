import os
import cv2
import datetime
from typing import Optional
import numpy as np

class FileManager:
    """Handles file operations and directory management"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.ensure_output_directory()
    
    def ensure_output_directory(self):
        """Create output directory if it doesn't exist"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")
    
    def save_plate_image(self, plate_image: np.ndarray, track_id: int) -> Optional[str]:
        """
        Save a license plate image with timestamp
        
        Args:
            plate_image: The plate image to save
            track_id: Tracking ID for the vehicle
            
        Returns:
            Path to saved file or None if failed
        """
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{self.output_dir}/plate_{track_id}_{timestamp}.jpg"
            
            success = cv2.imwrite(filename, plate_image)
            if success:
                print(f"Saved license plate to {filename}")
                return filename
            else:
                print(f"Failed to save image to {filename}")
                return None
                
        except Exception as e:
            print(f"Error saving image: {e}")
            return None