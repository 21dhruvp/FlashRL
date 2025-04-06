import mss
import numpy as np
import cv2
import pygetwindow as gw
import time

def get_ruffle_window_region():
    """Find the Ruffle player window with more robust detection"""
    try:
        # Try multiple common window title patterns
        patterns = ['Ruffle', 'Flash', 'SWF', 'World\'s Hardest Game']
        for pattern in patterns:
            windows = gw.getWindowsWithTitle(pattern)
            if windows:
                window = windows[0]
                
                # Ensure window is visible and has reasonable size
                if window.isMinimized:
                    window.restore()
                
                # Add small border to avoid capturing window decorations
                border = 5
                region = {
                    "top": window.top + border,
                    "left": window.left + border,
                    "width": max(window.width - 2*border, 100),
                    "height": max(window.height - 2*border, 100)
                }
                
                print(f"Found window: {window.title} at {region}")
                return region
        
        # If no window found, try active window
        active = gw.getActiveWindow()
        if active:
            print(f"Using active window: {active.title}")
            return {
                "top": active.top,
                "left": active.left,
                "width": active.width,
                "height": active.height
            }
            
    except Exception as e:
        print(f"Window detection error: {e}")
    
    # Fallback dimensions
    print("Using fallback screen region")
    return {"top": 100, "left": 100, "width": 820, "height": 630}

def capture_screen(region=None, retries=3):
    """Capture screen with error handling and retries"""
    if region is None:
        region = get_ruffle_window_region()
    
    for attempt in range(retries):
        try:
            with mss.mss() as sct:
                # Add small delay between attempts
                if attempt > 0:
                    time.sleep(0.1)
                
                screenshot = sct.grab(region)
                img = np.array(screenshot)
                return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                
        except Exception as e:
            print(f"Capture attempt {attempt+1} failed: {e}")
            if attempt == retries - 1:
                raise
                
    return np.zeros((region['height'], region['width'], 3), dtype=np.uint8)

def preprocess_frame(frame):
    """Convert frame to grayscale and resize"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (84, 84))
    return np.array(resized).flatten()

if __name__ == "__main__":
    # Test the capture
    print("Testing screen capture...")
    frame = capture_screen()
    
    if frame is not None:
        print(f"Captured frame shape: {frame.shape}")
        cv2.imshow("Captured Frame", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Failed to capture frame")