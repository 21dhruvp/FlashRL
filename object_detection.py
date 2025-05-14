from __future__ import annotations
from typing import Dict, List, Tuple
import cv2, numpy as np

cv2.setUseOptimized(True)
cv2.setNumThreads(0)          # 0 = use all cores

# ─── BGR colour ranges (tweak as needed) ───────────────────────────
R_RED_LO,  R_RED_HI  = (0, 0, 240), (30, 30, 255)
B_ENEMY_LO,B_ENEMY_HI = (240, 0, 0), (255, 30, 30)
Y_COIN_LO, Y_COIN_HI = (0,255, 255), (0,255, 255)
G_GOAL_LO, G_GOAL_HI = (180,254, 181), (180,254, 181)

_MIN_AREA = {"player": 30, "enemy": 9, "coin": 10, "goal": 20}

def _centroids(mask: np.ndarray, min_area: int) -> List[Tuple[int, int]]:
    centroids = []
    # work on half res for speed
    small = cv2.resize(mask, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0: continue
        cx, cy = int(M["m10"] / M["m00"] * 2), int(M["m01"] / M["m00"] * 2)
        centroids.append((cx, cy))
    return centroids


def detect_objects(frame_bgr: np.ndarray) -> Dict[str, List[Tuple[int, int]]]:
    bgr = frame_bgr

    masks = {
        "player": cv2.inRange(bgr, R_RED_LO,  R_RED_HI),
        "enemy":  cv2.inRange(bgr, B_ENEMY_LO, B_ENEMY_HI),
        "coin":   cv2.inRange(bgr, Y_COIN_LO, Y_COIN_HI),
        "goal":   cv2.inRange(bgr, G_GOAL_LO, G_GOAL_HI),
    }

    objs: Dict[str, List[Tuple[int, int]]] = {}
    for k, m in masks.items():
        centroids = _centroids(m, _MIN_AREA[k])
        if centroids:
            objs[k] = centroids
    return objs


# quick CLI test: python object_detection.py img.png
if __name__ == "__main__":
    import sys
    img = cv2.imread(sys.argv[1])
    det = detect_objects(img)
    dbg = img.copy()
    colours = {"player": (0,0,255), "enemy": (255,0,0),
               "coin": (0,255,255), "goal": (0,255,0)}
    for k, pts in det.items():
        for x,y in pts:
            cv2.circle(dbg, (x,y), 4, colours[k], 1)
    cv2.imshow("debug", dbg); cv2.waitKey(0)
