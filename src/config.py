from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", ".")).resolve()
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

CRITEO_URL = "https://go.criteo.net/criteo-research-uplift-v2.1.csv.gz"
CRITEO_RAW_FILE = DATA_RAW_DIR / "criteo-research-uplift-v2.1.csv.gz"
PROCESSED_FILE = DATA_PROCESSED_DIR / "criteo_uplift_sample.csv"
MODEL_BUNDLE_FILE = MODELS_DIR / "campaignlift_model_bundle.joblib"
PREDICTIONS_FILE = REPORTS_DIR / "uplift_predictions.csv"
METRICS_FILE = REPORTS_DIR / "model_metrics.json"

RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
DEFAULT_SAMPLE_ROWS = int(os.getenv("CRITEO_SAMPLE_ROWS", "200000"))
TARGET_COL = os.getenv("TARGET_COL", "visit")
TREATMENT_COL = os.getenv("TREATMENT_COL", "treatment")
OUTCOME_VALUE = float(os.getenv("OUTCOME_VALUE", "25.0"))
CONTACT_COST = float(os.getenv("CONTACT_COST", "0.25"))

FEATURE_COLUMNS = [f"f{i}" for i in range(12)]
REQUIRED_COLUMNS = FEATURE_COLUMNS + ["treatment", "conversion", "visit", "exposure"]

for directory in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
