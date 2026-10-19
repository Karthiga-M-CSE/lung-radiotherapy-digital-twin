from pathlib import Path
import os
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_default_external_dataset = PROJECT_ROOT.parent / "DATASET1"
DATASET_DIR = Path(os.environ.get("LUNG_TWIN_DATA_DIR", _default_external_dataset)).resolve()

DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

for directory in (DATA_DIR, OUTPUTS_DIR, MODELS_DIR, REPORTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

FILES = {
    "tumor_segmentation": "tumor_segmentation_results.csv",
    "tumor_trajectory": "tumor_trajectory.csv",
    "rpm_signal": "rpm_signal.txt",
    "tumor_motion_with_rpm": "tumor_motion_with_rpm.csv",
    "cycle_boundaries": "respiratory_cycle_boundaries.csv",
    "respiratory_prediction": "respiratory_prediction_results.csv",
    "respiratory_heldout": "respiratory_heldout_test_results.csv",
    "kalman_prediction": "kalman_prediction_results.csv",
    "kalman_acceleration": "kalman_acceleration_results.csv",
    "model_comparison": "heldout_model_comparison.csv",
    "uncertainty_safety_gate": "uncertainty_safety_gate_results.csv",
    "digital_twin_state": "digital_twin_state.csv",
    "replanning_trigger": "replanning_trigger_results.csv",
    "ridge_model": "respiratory_ridge_model.joblib",
    "ridge_scaler": "respiratory_ridge_scaler.joblib",
}

RTSTRUCT_FILE = Path(os.environ.get("RTSTRUCT_PATH", PROJECT_ROOT / "sample_data" / "rtstruct.dcm"))
RTDOSE_FILE = Path(os.environ.get("RTDOSE_PATH", PROJECT_ROOT / "sample_data" / "rtdose.dcm"))

N_LAGS = 5
ALPHA = 1.0
MIN_TRAINING_SAMPLES = 50

TRAIN_START_FRAME = 8
TRAIN_END_FRAME = 138
TEST_START_FRAME = 139
TEST_END_FRAME = 179

KALMAN_DT = 1.0
KALMAN_PROCESS_NOISE_VAR = 0.01
KALMAN_MEASUREMENT_NOISE_VAR = 0.5

PERSISTENCE_COUNT = 3
RESEARCH_P95_THRESHOLD_MM = 3.4304

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"


def resolve_file(filename: str, preferred_dir: Optional[Path] = None) -> Path:
    candidates = []
    if preferred_dir:
        candidates.append(preferred_dir / filename)
    candidates.extend([
        PROJECT_ROOT / filename,
        DATASET_DIR / filename,
        DATA_DIR / filename,
        OUTPUTS_DIR / filename,
    ])

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return (preferred_dir / filename) if preferred_dir else (PROJECT_ROOT / filename)
