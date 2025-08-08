import os
import time
from datetime import datetime, timedelta
import subprocess

INFERENCE_FILE_PATH = os.getcwd() + "\main.py"
PYTHON_PATH = '.venv/Scripts/python.exe' if os.path.exists(os.getcwd() + '\.venv') else 'python'

# Global variables to track system state
start_time = None
last_uptime_display = None

def log_message(message):
    """Log message with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def get_uptime():
    """Calculate uptime since system start"""
    if start_time is None:
        return "Unknown"
    
    uptime_duration = datetime.now() - start_time
    days = uptime_duration.days
    hours, remainder = divmod(uptime_duration.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if days > 0:
        return f"{days}d {hours}h {minutes}m {seconds}s"
    elif hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    else:
        return f"{minutes}m {seconds}s"

def display_uptime():
    """Display current uptime"""
    uptime = get_uptime()
    log_message(f"[UPTIME] System running for: {uptime}")

def startInference():
    """Start the inference process"""
    global start_time
    start_time = datetime.now()
    log_message("[START] Starting ANPR inference system...")
    try:
        process = subprocess.Popen([PYTHON_PATH, INFERENCE_FILE_PATH])
        log_message(f"[START] Process started with PID: {process.pid}")
        return process
    except Exception as e:
        log_message(f"[ERROR] Failed to start inference: {e}")
        return None

def terminateInference(process):
    """Terminate the inference process gracefully"""
    if process is None:
        return
    
    log_message(f"[STOP] Terminating process PID: {process.pid}")
    process.terminate()
    try:
        process.wait(timeout=10)
        log_message("[STOP] Process terminated gracefully")
    except subprocess.TimeoutExpired:
        log_message("[STOP] Process didn't terminate gracefully, forcing kill...")
        process.kill()
        log_message("[STOP] Process killed")

def should_restart_at_midnight():
    """Check if it's time for midnight restart"""
    now = datetime.now()
    return now.hour == 0 and now.minute == 0 and now.second == 0

def should_display_uptime():
    """Check if it's time to display uptime (every 4 hours)"""
    global last_uptime_display
    now = datetime.now()
    
    # Display uptime at 00:00, 04:00, 08:00, 12:00, 16:00, 20:00
    if now.minute == 0 and now.second == 0 and now.hour % 4 == 0:
        if last_uptime_display is None or (now - last_uptime_display).total_seconds() >= 3600:
            last_uptime_display = now
            return True
    return False

def restartInference():
    """Main scheduler function"""
    global start_time, last_uptime_display
    
    log_message("[SCHEDULER] ANPR System Scheduler started")
    log_message("[SCHEDULER] Schedule: Restart at midnight, Uptime display every 4 hours")
    
    inference = startInference()
    if inference is None:
        log_message("[ERROR] Failed to start initial process. Exiting.")
        return
    
    # Initialize uptime tracking
    last_uptime_display = datetime.now()
    
    try:
        while True:
            time.sleep(1)
            
            # Check if process is still running
            if inference.poll() is not None:
                log_message("[WARNING] Process has stopped unexpectedly. Restarting...")
                inference = startInference()
                continue
            
            # Check for midnight restart
            if should_restart_at_midnight():
                log_message("[SCHEDULER] Midnight restart triggered")
                uptime_before_restart = get_uptime()
                log_message(f"[SCHEDULER] Final uptime before restart: {uptime_before_restart}")
                
                terminateInference(inference)
                time.sleep(2)  # Brief pause before restart
                inference = startInference()
                
                log_message("[SCHEDULER] Midnight restart completed")
            
            # Check for uptime display
            elif should_display_uptime():
                display_uptime()
                
    except KeyboardInterrupt:
        log_message("[SCHEDULER] Shutdown signal received")
        terminateInference(inference)
        final_uptime = get_uptime()
        log_message(f"[SCHEDULER] Final system uptime: {final_uptime}")
        log_message("[SCHEDULER] Scheduler stopped")
    except Exception as e:
        log_message(f"[ERROR] Unexpected error: {e}")
        terminateInference(inference)

if __name__ == "__main__":
    restartInference()
