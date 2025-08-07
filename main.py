import sys
from config import Config
from utils.anpr_system import ANPRSystem
from utils.api_client import APIClient

def check_api_connection(config: Config) -> bool:
    print("🔄 Testing API connection...")
    api_client = APIClient(config.API_URL)
    
    if api_client.test_connection():
        print("✅ API connection successful!")
        return True
    
    return False

def main():
    try:
        config = Config()
        
        # Test API connection
        if not check_api_connection(config):
            print("\n🚫 ANPR System startup aborted due to API connection failure.")
            print("   Fix the API connection issue and try again.")
            sys.exit(1)
        
        print("\n🚀 Starting ANPR System...")
        
        # config.ACTIVE_ZONE_INDEX = 1  # Use different polygon zone
        # config.VIDEO_SOURCE = "sample_20detik720.mp4"  # Use video file instead of RTSP
        
        anpr_system = ANPRSystem(config)
        anpr_system.start()
        
    except Exception as e:
        print(f"❌ Error starting ANPR system: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
