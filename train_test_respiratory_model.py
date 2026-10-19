import os
import numpy as np
import pandas as pd

from src.config import (
    resolve_file,
    PROJECT_ROOT,
    MODELS_DIR,
    OUTPUTS_DIR,
    N_LAGS,
    ALPHA,
    TRAIN_START_FRAME,
    TRAIN_END_FRAME,
    TEST_START_FRAME,
    TEST_END_FRAME,
)
from src.models.ridge_predictor import RespiratoryRidgePredictor
from src.models.evaluation import evaluate_model_predictions
from src.utils.logging import get_logger

logger = get_logger("train_test_respiratory_model")

INPUT_FILE = resolve_file("tumor_motion_with_rpm.csv")
CYCLE_FILE = resolve_file("respiratory_cycle_boundaries.csv")

MODEL_FILE = MODELS_DIR / "respiratory_ridge_model.joblib"
SCALER_FILE = MODELS_DIR / "respiratory_ridge_scaler.joblib"
OUTPUT_FILE = resolve_file("respiratory_heldout_test_results.csv", preferred_dir=PROJECT_ROOT)


def main():
    print("=" * 70)
    print("RESPIRATORY MODEL - CYCLE-BASED TRAIN / TEST")
    print("=" * 70)

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"\nMotion dataset not found:\n{INPUT_FILE}")

    if not os.path.exists(CYCLE_FILE):
        raise FileNotFoundError(f"\nRespiratory cycle file not found:\n{CYCLE_FILE}")

    df = pd.read_csv(INPUT_FILE)
    print(f"\nLoaded motion data:\nRows: {len(df)}")
    print(f"\nColumns:\n{df.columns.tolist()}")

    predictor = RespiratoryRidgePredictor(alpha=ALPHA, n_lags=N_LAGS)
    X, y, frames, lag_columns = predictor.create_features(
        df=df,
        feature_cols=["rpm", "x_mm", "y_mm", "z_mm"],
        target_cols=["x_mm", "y_mm", "z_mm"],
    )

    print(f"\nRows after lag-feature preparation:\n{len(X)}")

    train_mask = (frames >= TRAIN_START_FRAME) & (frames <= TRAIN_END_FRAME)
    test_mask = (frames >= TEST_START_FRAME) & (frames <= TEST_END_FRAME)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    train_frames, test_frames = frames[train_mask], frames[test_mask]

    print("\n" + "=" * 70)
    print("TEMPORAL SPLIT")
    print("=" * 70)
    print(f"Training frames : {train_frames[0]} -> {train_frames[-1]}")
    print(f"Training rows   : {len(X_train)}")
    print(f"\nTesting frames  : {test_frames[0]} -> {test_frames[-1]}")
    print(f"Testing rows    : {len(X_test)}")

    if len(X_train) == 0:
        raise RuntimeError("Training set is empty.")
    if len(X_test) == 0:
        raise RuntimeError("Testing set is empty.")

    train_frame_set = set(train_frames)
    test_frame_set = set(test_frames)

    if train_frame_set.intersection(test_frame_set):
        raise RuntimeError("ERROR: Training and testing frames overlap!")
    if train_frames.max() >= test_frames.min():
        raise RuntimeError("ERROR: Training data extends into the test period!")

    print("\n[OK] No train/test frame overlap")
    print("[OK] Training occurs before testing")
    print("[OK] Temporal split is locked")

    print("\n" + "=" * 70)
    print("FEATURE STANDARDIZATION & TRAINING")
    print("=" * 70)

    predictor.fit(X_train, y_train)
    print("[OK] Scaler fitted using training data only")
    print("[OK] Model trained")

    predictor.save(MODEL_FILE, SCALER_FILE)
    predictor.save(PROJECT_ROOT / "respiratory_ridge_model.joblib", PROJECT_ROOT / "respiratory_ridge_scaler.joblib")

    print(f"\nSaved model:\n{MODEL_FILE}")
    print(f"\nSaved scaler:\n{SCALER_FILE}")

    print("\n" + "=" * 70)
    print("HELD-OUT TEST PREDICTION")
    print("=" * 70)

    results = predictor.predict_sequence(X_test, y_test, frames=test_frames)
    print(f"Generated predictions: {len(results)}")

    results.to_csv(OUTPUT_FILE, index=False)
    results.to_csv(OUTPUTS_DIR / "respiratory_heldout_test_results.csv", index=False)

    _, metrics = evaluate_model_predictions(y_test, predictor.predict(X_test), frames=test_frames, model_name="Ridge")

    print("\n" + "=" * 70)
    print("HELD-OUT TEST RESULTS")
    print("=" * 70)
    print(f"\nNumber of test predictions: {len(results)}")
    print("\n3D ERROR")
    print(f"Mean   : {metrics.mean_3d:.4f} mm")
    print(f"Median : {metrics.median_3d:.4f} mm")
    print(f"P90    : {metrics.p90_3d:.4f} mm")
    print(f"P95    : {metrics.p95_3d:.4f} mm")
    print(f"P99    : {metrics.p99_3d:.4f} mm")
    print(f"Maximum: {metrics.max_3d:.4f} mm")

    print("\nAXIS-WISE MEAN ERROR")
    print(f"X: {metrics.mean_x:.4f} mm")
    print(f"Y: {metrics.mean_y:.4f} mm")
    print(f"Z: {metrics.mean_z:.4f} mm")

    print("\nAXIS-WISE P95 ERROR")
    print(f"X: {metrics.p95_x:.4f} mm")
    print(f"Y: {metrics.p95_y:.4f} mm")
    print(f"Z: {metrics.p95_z:.4f} mm")

    print("\n" + "=" * 70)
    print("OUTPUT")
    print("=" * 70)
    print(f"\nResults saved to:\n{OUTPUT_FILE}")
    print(f"\nModel saved to:\n{MODEL_FILE}")
    print(f"\nScaler saved to:\n{SCALER_FILE}")
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
