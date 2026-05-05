import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_OK_DIR = os.path.join(DATA_DIR, "ok_front")
TEST_DEFECT_DIR = os.path.join(DATA_DIR, "def_front")

# Model Parameters
IMAGE_SIZE = 256
FEATURE_MAP_SIZE = 32
BLUR_KERNEL = (21, 21)