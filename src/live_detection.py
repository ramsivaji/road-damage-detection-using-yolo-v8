"""OpenCV webcam inference helper using the project-produced model."""

import time
import cv2
from pathlib import Path
import sys
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import BEST_MODEL_PATH, DETECTION_OUTPUTS_DIR


def run_camera(camera_index: int = 0, confidence: float = 0.25, max_frames: int = 0):
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Train the project model first: {BEST_MODEL_PATH}")
    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not camera.isOpened():
        raise RuntimeError("Unable to open webcam. Check permissions and camera index 0.")
    model = YOLO(str(BEST_MODEL_PATH))
    frame_count = 0
    last_time = time.perf_counter()
    try:
        while max_frames <= 0 or frame_count < max_frames:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("The webcam stopped returning frames.")
            result = model.predict(frame, conf=confidence, verbose=False)[0]
            annotated = result.plot()
            now = time.perf_counter()
            fps = 1.0 / max(now - last_time, 1e-6)
            last_time = now
            yield annotated, result, fps
            frame_count += 1
    finally:
        camera.release()


def save_frame(frame):
    DETECTION_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    path = DETECTION_OUTPUTS_DIR / f"webcam_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"Could not save webcam frame: {path}")
    return path
