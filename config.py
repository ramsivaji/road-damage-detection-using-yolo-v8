"""Central configuration for the road damage detection project."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

# Dataset locations. Downloaded archives and extracted source data are kept
# separate from the generated YOLO dataset.
DATASET_DIR = PROJECT_ROOT / "dataset"
DOWNLOADS_DIR = DATASET_DIR / "downloads"
RAW_IMAGES_DIR = DATASET_DIR / "raw" / "images"
RAW_ANNOTATIONS_DIR = DATASET_DIR / "raw" / "annotations"
ANNOTATIONS_DIR = DATASET_DIR / "annotations"
PREPARED_IMAGES_DIR = DATASET_DIR / "images"
PREPARED_LABELS_DIR = DATASET_DIR / "labels"
DATA_YAML_PATH = DATASET_DIR / "data.yaml"

# Model and output locations.
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_VERSIONS_DIR = MODELS_DIR / "versions"
BEST_MODEL_PATH = MODELS_DIR / "best.pt"
LAST_MODEL_PATH = MODELS_DIR / "last.pt"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DETECTION_OUTPUTS_DIR = OUTPUTS_DIR / "detections"
VIDEO_OUTPUTS_DIR = OUTPUTS_DIR / "videos"
TRAINING_OUTPUTS_DIR = OUTPUTS_DIR / "training"
EVALUATION_OUTPUTS_DIR = OUTPUTS_DIR / "evaluation"

# RDD2022 class identifiers are kept separately from display names because the
# XML source annotations use the identifiers.
CLASS_IDS = ["D00", "D10", "D20", "D40"]
CLASS_NAMES = {
    "D00": "Longitudinal Crack",
    "D10": "Transverse Crack",
    "D20": "Alligator Crack",
    "D40": "Pothole",
}

RDD2022_REPOSITORY_URL = "https://github.com/sekilab/RoadDamageDetector"
RDD2022_DOWNLOAD_URL = (
    "https://ndownloader.figshare.com/files/38030910"
)
DEFAULT_COUNTRY = "India"

# Reproducible default split ratios. Preparation code will validate these.
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.20
TEST_SPLIT = 0.10

# Training defaults. The training phase will explicitly choose either random
# initialization or transfer learning; no weights are bundled in Phase 1.
YOLO_MODEL_CONFIG = "yolov8n.yaml"
YOLO_PRETRAINED_WEIGHTS = "yolov8n.pt"
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 8
DEFAULT_IMAGE_SIZE = 640
DEFAULT_LEARNING_RATE = 0.01
MODEL_PATH = BEST_MODEL_PATH


def ensure_project_directories() -> None:
    """Create runtime directories if they are missing."""

    directories = (
        DOWNLOADS_DIR,
        RAW_IMAGES_DIR,
        RAW_ANNOTATIONS_DIR,
        ANNOTATIONS_DIR,
        PREPARED_IMAGES_DIR / "train",
        PREPARED_IMAGES_DIR / "val",
        PREPARED_IMAGES_DIR / "test",
        PREPARED_LABELS_DIR / "train",
        PREPARED_LABELS_DIR / "val",
        PREPARED_LABELS_DIR / "test",
        MODEL_VERSIONS_DIR,
        DETECTION_OUTPUTS_DIR,
        VIDEO_OUTPUTS_DIR,
        TRAINING_OUTPUTS_DIR,
        EVALUATION_OUTPUTS_DIR,
    )
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
