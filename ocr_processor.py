import cv2
import re
from fast_plate_ocr import ONNXPlateRecognizer
from typing import Optional
import numpy as np

class OCRProcessor:
    """Handles optical character recognition for license plates"""
    
    def __init__(self, model_path: str):
        self.reader = ONNXPlateRecognizer(model_path)
    
    def process_plate(self, plate_image: np.ndarray) -> str:
        """
        Process a license plate image and return cleaned OCR text
        
        Args:
            plate_image: BGR image of the license plate
            
        Returns:
            Cleaned OCR text
        """
        try:
            # Convert to grayscale for better OCR
            plate_img_bw = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
            
            # Run OCR
            ocr_result = self.reader.run(plate_img_bw)
            
            # Clean the result
            if ocr_result and len(ocr_result) > 0:
                cleaned_text = re.sub(r'[^A-Za-z0-9 ]+', '', ocr_result[0]).upper().strip()
                return cleaned_text
            
            return ""
            
        except Exception as e:
            print(f"Error processing OCR: {e}")
            return ""