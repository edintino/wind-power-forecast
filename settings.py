import torch
import random
import numpy as np
import logging

logger = logging.getLogger(__name__)

# -----------------------
# General Data Settings
# -----------------------
DATA_FILE = "./data/T1.csv"
FREQ = "10T"  # frequency for date range reindexing
LOCATION_NAME = "Yalova"#"Izmir"
REGION_NAME = "Turkey"
TIMEZONE = "Europe/Istanbul"
LAT, LON = 40.58545, 28.99035

# -----------------------
# Data Split Ratios
# -----------------------
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# -----------------------
# Model Hyperparameters
# -----------------------
SEQ_LENGTH = 45
FORECAST_HORIZON = 1
BATCH_SIZE = 64
NUM_EPOCHS = 250
LR = 1e-3

PATIENCE = 25

# Transformer
HIDDEN_SIZE = 64
OUTPUT_SIZE = 1
NUM_HEADS = 4
NUM_LAYERS = 3
DROPOUT = 0.15

# -----------------------
# Seed for Reproducibility
# -----------------------
SEED = 7843

def set_global_seed(seed=SEED):
    """
    Set global seeds for reproducibility in random, NumPy, and PyTorch.

    Args:
        seed (int, optional): Seed value to be set. Defaults to SEED.

    Returns:
        None
    """
    logger.info(f"Setting global seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False