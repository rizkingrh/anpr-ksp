import queue
import threading
from typing import Tuple, Optional
from .api_client import APIClient

class APIQueueManager:
    """Manages API requests in a separate thread to avoid blocking"""
    
    def __init__(self, api_client: APIClient):
        self.api_client = api_client
        self.api_queue = queue.Queue()
        self.worker_thread = None
        self.running = False
    
    def start(self):
        """Start the API worker thread"""
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        print("API queue manager started")
    
    def stop(self):
        """Stop the API worker thread"""
        self.running = False
        # Signal worker to stop
        self.api_queue.put(None)
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        print("API queue manager stopped")
    
    def queue_request(self, plate_text: str, image_path: str):
        """Queue an API request"""
        if self.running:
            self.api_queue.put((plate_text, image_path))
    
    def wait_for_completion(self):
        """Wait for all pending API requests to complete"""
        print("Waiting for pending API tasks to complete...")
        self.api_queue.join()
    
    def _worker(self):
        """Worker thread function"""
        while self.running:
            try:
                # Get task from queue
                task = self.api_queue.get(timeout=1)
                if task is None:  # None is our signal to exit
                    break
                
                plate_text, image_path = task
                self.api_client.send_plate_data(plate_text, image_path)
                
                # Mark task as done
                self.api_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in API worker thread: {e}")
                # Mark task as done even if it failed
                self.api_queue.task_done()