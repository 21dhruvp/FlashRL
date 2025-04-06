import mss
import numpy as np
import cv2

GAME_REGION = {"top": 100, "left": 100, "width": 820, "height": 630}

def capture_screen(region=GAME_REGION):
    with mss.mss() as sct:
        screenshot = sct.grab(region)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (84, 84))
    return np.array(resized).flatten()

if __name__ == "__main__":
    frame = capture_screen()
    cv2.imshow("Captured Frame", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
