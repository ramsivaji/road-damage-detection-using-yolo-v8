"""Run detection on an uploaded image using the project-produced model."""

from pathlib import Path
import sys
from PIL import Image
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import BEST_MODEL_PATH, DETECTION_OUTPUTS_DIR


def detect_image(image: Image.Image, confidence: float = 0.25):
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Train the project model first: {BEST_MODEL_PATH}")
    model = YOLO(str(BEST_MODEL_PATH))
    result = model.predict(source=image, conf=confidence, verbose=False)[0]
    annotated = Image.fromarray(result.plot()[..., ::-1])
    DETECTION_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    return annotated, result
