import cv2
import supervision as sv
from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame
from typing import Set, Optional

from config import Config
from ocr_processor import OCRProcessor
from api_client import APIClient
from file_manager import FileManager
from api_queue_manager import APIQueueManager
from visualization import Visualizer

class ANPRSystem:
    """Main ANPR system class that orchestrates all components"""
    
    def __init__(self, config: Config):
        self.config = config
        self.current_plate_text = ""
        self.captured_track_ids: Set[int] = set()
        self.confidence_threshold = config.CONFIDENCE_THRESHOLD
        
        # Initialize components
        self.ocr_processor = OCRProcessor(config.OCR_MODEL_PATH)
        self.api_client = APIClient(config.API_URL)
        self.file_manager = FileManager(config.OUTPUT_DIR)
        self.api_queue_manager = APIQueueManager(self.api_client)
        self.visualizer = Visualizer(config)
        
        # Initialize tracker
        self.tracker = sv.ByteTrack()
        
        # Initialize pipeline
        self.pipeline = None
    
    def start(self):
        """Start the ANPR system"""
        print("Starting ANPR system...")
        
        # Start API queue manager
        self.api_queue_manager.start()
        
        # Initialize and start the pipeline
        self.pipeline = InferencePipeline.init(
            model_id=self.config.MODEL_ID,
            api_key=self.config.API_KEY,
            video_reference=self.config.VIDEO_SOURCE,
            on_prediction=self._process_frame,
            confidence=self.confidence_threshold,
        )
        
        try:
            self.pipeline.start()
            self.pipeline.join()
        except KeyboardInterrupt:
            print("\nReceived interrupt signal...")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the ANPR system"""
        print("Stopping ANPR system...")
        
        if self.pipeline:
            self.pipeline.terminate()
        
        # Wait for API tasks and stop queue manager
        self.api_queue_manager.wait_for_completion()
        self.api_queue_manager.stop()
        
        # Close OpenCV windows
        cv2.destroyAllWindows()
        print("ANPR system stopped")
    
    def _process_frame(self, predictions: dict, video_frame: VideoFrame):
        """Process each frame from the video stream"""
        try:
            # Get labels and detections
            # labels = [p["class"] for p in predictions["predictions"]]
            labels = [f"{p['class']} ({p['confidence']:.2f})" for p in predictions["predictions"]]
            detections = sv.Detections.from_inference(predictions)
            # detections = detections[detections.confidence > self.confidence_threshold]
            
            # Update tracker
            tracked_detections = self.tracker.update_with_detections(detections)
            
            # Get detections in zone
            in_zone_detections = self.visualizer.get_zone_detections(tracked_detections)
            
            # Process frame
            frame = video_frame.image.copy()
            
            # Process detections in zone
            self._process_zone_detections(in_zone_detections, frame)
            
            # Annotate frame
            annotated_frame = self.visualizer.annotate_frame(
                frame, detections, labels, self.current_plate_text
            )
            
            # annotated_frame = cv2.resize(annotated_frame, (1280, 720))
            
            # Display frame
            cv2.imshow("ANPR", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop()
                
        except Exception as e:
            print(f"Error processing frame: {e}")
    
    def _process_zone_detections(self, in_zone_detections: sv.Detections, frame):
        """Process detections that are within the zone"""
        if len(in_zone_detections) == 0:
            return
        
        for i in range(len(in_zone_detections)):
            track_id = in_zone_detections.tracker_id[i]
            
            # Skip if already captured
            if track_id in self.captured_track_ids:
                continue
            
            # Mark as captured
            self.captured_track_ids.add(track_id)
            
            # Extract plate image
            bbox = in_zone_detections.xyxy[i]
            x1, y1, x2, y2 = map(int, bbox)
            plate_image = frame[y1:y2, x1:x2]
            
            plate_text = self.ocr_processor.process_plate(plate_image)
            
            if plate_text:
                self.current_plate_text = plate_text
                print(f"Detected plate: {plate_text}")
                
                image_path = self.file_manager.save_plate_image(plate_image, track_id)
                
                if image_path:
                    self.api_queue_manager.queue_request(plate_text, image_path)