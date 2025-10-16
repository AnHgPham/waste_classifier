"""
Cấu hình tập trung cho dự án
Tất cả hyperparameters và settings ở đây
"""

import os

# ==================== PATHS ====================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODELS_DIR = os.path.join(OUTPUTS_DIR, "models")
LOGS_DIR = os.path.join(OUTPUTS_DIR, "logs")
REPORTS_DIR = os.path.join(OUTPUTS_DIR, "reports")
SCREENSHOTS_DIR = os.path.join(OUTPUTS_DIR, "screenshots")

# ==================== DATA ====================
DATASET_NAME = "sumn2u/garbage-classification-v2"
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# ==================== MODEL ====================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_BASELINE = 30
EPOCHS_TRANSFER_PHASE1 = 15
EPOCHS_TRANSFER_PHASE2 = 10
NUM_CLASSES = 10

CLASS_NAMES = [
    'battery', 'biological', 'cardboard', 'clothes', 'glass',
    'metal', 'paper', 'plastic', 'shoes', 'trash'
]

# ==================== TRAINING ====================
SEED = 42
LEARNING_RATE_BASELINE = 0.001
LEARNING_RATE_TRANSFER_PHASE1 = 0.001
LEARNING_RATE_TRANSFER_PHASE2 = 0.0001

# ==================== AUGMENTATION ====================
USE_AUGMENTATION = True
AUGMENTATION_CONFIG = {
    'horizontal_flip': True,
    'rotation_factor': 0.1,
    'zoom_factor': 0.1,
    'contrast_factor': 0.1
}

# ==================== CALLBACKS ====================
EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.5

