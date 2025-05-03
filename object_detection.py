"""
object_detection.py – streamlined, tuple‑based detections
---------------------------------------------------------
Returns (x, y) tuples so it plugs straight into Environment without any
additional changes.
"""

from __future__ import annotations
from typing import Dict, List, Tuple

import cv2
import numpy as np

# ─── HSV colour ranges tuned for World's Hardest Game ──────────────────────
COLOR_RANGES: Dict[str, Tuple[List[int], List[int]]] = {
    # player (red)
    "player": ([0, 180, 160], [8, 255, 255]),
    # enemies (blue)
    "enemy": ([100, 120, 80], [140, 255, 220]),
    # checkpoints & finish (green)
    "goal": ([50, 120, 120], [70, 255, 255]),
    # coins (yellow)
    "coin": ([20, 150, 150], [30, 255, 255]),
    # walls (dark)
    "wall": ([0, 0, 0], [180, 50, 50]),
}

# minimum contour area (px²) for a valid detection
MIN_AREA = {
    "player": 50,
    "enemy": 20,
    "goal": 30,
    "coin": 15,
    "wall": 40,
}

# kernel for small‑noise removal
_K3 = np.ones((3, 3), np.uint8)

# ────────────────────────────────────────────────────────────────────────────

def detect_objects(frame: np.ndarray) -> Dict[str, List[Tuple[int, int]]]:
    """Return a mapping {object_type: [(x, y), …]} for the current frame."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    detected: Dict[str, List[Tuple[int, int]]] = {k: [] for k in COLOR_RANGES}

    for obj, (lower, upper) in COLOR_RANGES.items():
        mask = cv2.inRange(
            hsv,
            np.array(lower, dtype=np.uint8),
            np.array(upper, dtype=np.uint8),
        )
        # morphological open to delete single‑pixel speckles
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _K3, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < MIN_AREA[obj]:
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            detected[obj].append((cx, cy))

    return detected

# ─── quick manual test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, pathlib

    if len(sys.argv) < 2:
        print("Usage: python object_detection.py <image_path>")
        raise SystemExit(1)

    img_path = pathlib.Path(sys.argv[1])
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[error] could not load {img_path}")
        raise SystemExit(1)

    dets = detect_objects(img)
    dbg = img.copy()
    COL = {
        "player": (0, 0, 255),
        "enemy": (255, 0, 0),
        "goal": (0, 255, 0),
        "coin": (0, 255, 255),
        "wall": (80, 80, 80),
    }
    for k, pts in dets.items():
        for x, y in pts:
            cv2.circle(dbg, (x, y), 5, COL[k], 1)
            cv2.putText(dbg, k[:2], (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL[k], 1)

    cv2.imshow("detection", dbg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
