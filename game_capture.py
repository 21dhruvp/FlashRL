import time
from typing import Dict

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

    for i in range(retries):
        try:
            with mss.mss() as sct:
                shot = sct.grab(region)               # BGRA buffer
                bgr  = np.frombuffer(shot.raw, np.uint8)  # raw = BGRA
                bgr  = bgr.reshape(shot.height, shot.width, 4)[:, :, :3]
                return bgr
        except Exception as e:
            print(f"[mss] grab attempt {i+1} failed → {e}")
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
