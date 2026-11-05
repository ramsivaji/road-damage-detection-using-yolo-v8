"""Run project-model detection over an uploaded video."""

from pathlib import Path
import cv2
from ultralytics import YOLO

from config import BEST_MODEL_PATH, VIDEO_OUTPUTS_DIR


def detect_video(input_path: Path, confidence: float = 0.25) -> Path:
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Train the project model first: {BEST_MODEL_PATH}")
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    output = VIDEO_OUTPUTS_DIR / f"{input_path.stem}_detected.mp4"
    VIDEO_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    model = YOLO(str(BEST_MODEL_PATH))
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            result = model.predict(frame, conf=confidence, verbose=False)[0]
            writer.write(result.plot())
    finally:
        capture.release()
        writer.release()
    return output
