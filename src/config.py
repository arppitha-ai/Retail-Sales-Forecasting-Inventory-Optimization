"""Central configuration for the retail forecasting project."""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SIMULATED_DATA_DIR = DATA_DIR / "simulated"

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FORECAST_DIR = OUTPUT_DIR / "forecasts"
INVENTORY_DIR = OUTPUT_DIR / "inventory"
METRICS_DIR = OUTPUT_DIR / "metrics"
FIGURES_DIR = OUTPUT_DIR / "figures"

# Additional project directories
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
IMAGES_DIR = PROJECT_ROOT / "images"

# Forecasting configuration
FORECAST_HORIZON = 14
TEST_SIZE_DAYS = 30

# Synthetic data configuration
RANDOM_SEED = 42
NUM_STORES = 3
NUM_PRODUCTS = 8
HISTORY_DAYS = 180

# Inventory planning defaults
DEFAULT_LEAD_TIME_DAYS = 7
DEFAULT_REVIEW_PERIOD_DAYS = 7
DEFAULT_SERVICE_LEVEL_Z = 1.65

# Create required directories if they do not exist
REQUIRED_DIRECTORIES = [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SIMULATED_DATA_DIR,
    FORECAST_DIR,
    INVENTORY_DIR,
    METRICS_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    IMAGES_DIR,
]

for directory in REQUIRED_DIRECTORIES:
    directory.mkdir(parents=True, exist_ok=True)