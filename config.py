"""
Discipline Monitoring System Configuration
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"
FACES_DB_DIR = DATA_DIR / "faces_database"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories
for dir_path in [DATA_DIR, MODELS_DIR, OUTPUT_DIR, FACES_DB_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# Roboflow settings
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "your_api_key_here")
ROBOFLOW_WORKSPACE = "systemmonitoring"
ROBOFLOW_PROJECT = "system_monitoring2"
ROBOFLOW_VERSION = 4
               
                
# Violation classes
VIOLATION_CLASSES = {
    0: "bottle",
    1: "food",
    2: "phone",
    3: "sleep"
}

# English labels for video display
VIOLATION_LABELS = {
    "bottle": "Bottle",
    "food": "Food",
    "phone": "Phone", 
    "sleep": "Sleep"
}

# YOLO settings
YOLO_MODEL = "yolo11s.pt"
YOLO_TRAINED_MODEL = MODELS_DIR / "best.pt"
YOLO_CONFIDENCE = 0.4
YOLO_IOU = 0.45
YOLO_IMG_SIZE = 640

# Training settings
TRAIN_EPOCHS = 35
TRAIN_BATCH = 24  # Increased for RTX 5070 (12GB)
TRAIN_IMG_SIZE = 640
TRAIN_PATIENCE = 10  # Early stopping
USE_GPU = True

# Violation detection settings
VIOLATION_DURATION_THRESHOLD = 1.5  # Seconds to consider a violation
BUFFER_AFTER_VIOLATION = 3  # Seconds to record after violation ends
FPS = 30

# Video processing settings
FRAME_SKIP = 1  # Process every N-th frame (1 = all frames, 2 = every other)
# FRAME_SKIP recommendations:
# 1 - maximum accuracy, slower
# 2 - good balance of speed and accuracy (recommended)
# 3-5 - faster, but may miss short violations

# Face recognition settings
# Порог распознавания (меньше = строже):
# 0.3-0.4 - очень строгий (много Unknown)
# 0.5-0.6 - оптимальный баланс
# 0.7-0.8 - мягкий (больше false positives)
FACE_RECOGNITION_TOLERANCE = 0.7
FACE_DETECTION_MODEL = "hog"  # "hog" быстрее, "cnn" точнее но требует GPU
FACE_RECOGNITION_FRAME_SKIP = 2  # Распознавать лица каждые N кадров

# Video settings
VIDEO_CODEC = "mp4v"
VIDEO_EXTENSION = ".mp4"

# Report settings
REPORT_FORMAT = "json"
REPORT_FILE = OUTPUT_DIR / f"violations_report.{REPORT_FORMAT}"

# Colors for drawing (BGR for OpenCV)
COLORS = {
    "bottle": (0, 0, 255),      # Red
    "food": (0, 165, 255),      # Orange
    "phone": (0, 255, 255),     # Yellow
    "sleep": (128, 0, 128),     # Purple       
    "face": (255, 0, 0),        # Blue
    "unknown": (128, 128, 128)  # Gray
}

# Text settings for video
FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2
TEXT_COLOR = (255, 255, 255)  # White
TEXT_BG_COLOR = (0, 0, 0)     # Black

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
