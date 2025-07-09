import sys
from config import Config
from anpr_system import ANPRSystem

def main():
    try:
        config = Config()
        
        # config.ACTIVE_ZONE_INDEX = 1  # Use different polygon zone
        # config.VIDEO_SOURCE = "sample_20detik720.mp4"  # Use video file instead of RTSP
        
        anpr_system = ANPRSystem(config)
        anpr_system.start()
        
    except Exception as e:
        print(f"Error starting ANPR system: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()