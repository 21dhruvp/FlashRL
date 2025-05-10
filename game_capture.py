import time
from typing import Dict

import bettercam
camera = bettercam.create()

import cv2
import mss
import numpy as np
import pygetwindow as gw

_PATTERNS = ['Ruffle', 'Flash', 'SWF', "World's Hardest Game"]

# ── locate the Ruffle window ─────────────────────────────────────────
def get_ruffle_window_region() -> Dict[str, int]:
    for p in _PATTERNS:
        wins = gw.getWindowsWithTitle(p)
        if wins:
            w = wins[0]
            if w.isMinimized:
                w.restore()
            b = 5
            return {"top": w.top + b, "left": w.left + b,
                    "width": max(w.width  - 2*b, 100),
                    "height": max(w.height - 2*b, 100)}
    return {"top": 100, "left": 100, "width": 820, "height": 630}

# ── grab one BGR frame ──────────────────────────────────────────────
def capture_screen(region=None, retries=3):

    if region is None:
        region = get_ruffle_window_region()
    
    # bettercam wants a (x, y, w, h) tuple
    rect = (region["left"], region["top"], region["width"], region["height"])

    for i in range(retries):
        try:
            frame = camera.grab()  # full screen
            if frame is not None:
                y1 = region["top"]
                y2 = y1 + region["height"]
                x1 = region["left"]
                x2 = x1 + region["width"]
                return frame[y1:y2, x1:x2]
        except Exception as e:
            print(f"[bettercam] capture attempt {i+1} failed → {e}")
            time.sleep(0.05)

    return np.zeros((region['height'], region['width'], 3), np.uint8)

# ── 84×84 grayscale for the agent ───────────────────────────────────
def preprocess_frame(frame):
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return small.flatten()

# ── manual colour test ─────────────────────────────────────────────
if __name__ == "__main__":
    img_bgr = capture_screen()
    cv2.imshow("BGR test (red square should look red)", img_bgr)
    cv2.waitKey(0); cv2.destroyAllWindows()
