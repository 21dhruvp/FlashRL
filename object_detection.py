import cv2
import numpy as np

def detect_objects(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define color ranges in HSV.
    # Note: The "goal" color here (green) is ambiguous in the game (used for start, end, and checkpoints).
    color_ranges = {
        "player": ([0, 150, 150], [10, 255, 255]),      # Red player
        "enemy": ([100, 150, 150], [140, 255, 255]),    # Blue enemies
        "wall": ([0, 0, 0], [180, 255, 30]),            # Black walls
        "goal": ([50, 150, 150], [70, 255, 255]),       # Green goal (ambiguous)
        "coin": ([20, 150, 150], [40, 255, 255]),       # Yellow coins
    }

    detected = {}
    for obj, (lower, upper) in color_ranges.items():
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        
        # Find contours and extract centroids.
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        positions = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 10:  # Filter out noise.
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    positions.append((cx, cy))
        
        detected[obj] = positions

    return detected
