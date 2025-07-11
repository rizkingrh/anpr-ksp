import requests
import base64
import json
from typing import Optional

class APIClient:
    """Handles API communication"""
    
    def __init__(self, api_url: str):
        self.api_url = api_url
    
    def send_plate_data(self, plate_text: str, image_path: str) -> bool:
        """
        Send plate data to the API
        
        Args:
            plate_text: OCR result text
            image_path: Path to the saved image
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Read and encode image
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Prepare data
            data = {
                "numberplate": str(plate_text),
                "image": encoded_string
            }
            
            # Send request
            response = requests.post(self.api_url, json=data, timeout=10)
            
            if response.status_code == 201:
                print(f"Successfully sent plate data: {plate_text}")
                return True
            else:
                print(f"Failed to send data! Status: {response.status_code}")
                print("Response text:", response.text)
                return False
                
        except FileNotFoundError:
            print(f"Error: Image not found at path {image_path}")
            return False
        except requests.exceptions.RequestException as e:
            print(f"Error sending request: {e}")
            return False
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return False