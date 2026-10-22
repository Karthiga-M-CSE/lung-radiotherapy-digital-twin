import os
import numpy as np
import pandas as pd

from src.config import (
    resolve_file,
    PROJECT_ROOT,
    OUTPUTS_DIR,
    N_LAGS,
    ALPHA,
    TRAIN_START_FRAME,
    TRAIN_END_FRAME,
    CALIB_START_FRAME,
    CALIB_END_FRAME,
    TEST_START_FRAME,
    TEST_END_FRAME,
)
from src.data.splits import chronological_frame_split
from src.models.conformal import SplitConformalPredictor
from src.models.ridge_predictor import RespiratoryRidgePredictor
from src.utils.logging import get_logger
from src.utils.metrics import compute_prediction_errors

logger = get_logger("respiratory_conformal_prediction")

INPUT_FILE = resolve_file("tumor_motion_with_rpm.csv")
CYCLE_FILE = resolve_file("respiratory_cycle_boundaries.csv")
OUTPUT_FILE = resolve_file("conformal_prediction_results.csv", preferred_dir=PROJECT_ROOT)


def main():
    print("=" * 70)
    print("CHRONOLOGICAL SPLIT CONFORMAL PREDICTION & EVALUATION")
    print("=" * 70)

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Motion dataset not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    print(f"\nLoaded motion data: {len(df)} rows")

    predictor = RespiratoryRidgePredictor(alpha=ALPHA, n_lags=N_LAGS)
    X, y, frames, _ = predictor.create_features(
        df=df,
        feature_cols=["rpm", "x_mm", "y_mm", "z_mm"],
        target_cols=["x_mm", "y_mm", "z_mm"],
    )

    # 1. Proper chronological 3-way split: Train < Calibration < Test
    split = chronological_frame_split(
        X, y, frames,
        train_range=(TRAIN_START_FRAME, TRAIN_END_FRAME),
        calib_range=(CALIB_START_FRAME, CALIB_END_FRAME),
        test_range=(TEST_START_FRAME, TEST_END_FRAME),
    )
    print("\n" + split.summary())

    # 2. Train Ridge model on chronological Training set
    print("\n" + "=" * 70)
    print("1. MODEL TRAINING (CHRONOLOGICAL TRAIN SET)")
    print("=" * 70)
    predictor.fit(split.X_train, split.y_train)
    print(f"[OK] Trained on frames {int(split.train_frames.min())} -> {int(split.train_frames.max())} ({len(split.X_train)} samples)")

    # 3. Predict on chronological Calibration set & Calibrate Conformal Predictor
    print("\n" + "=" * 70)
    print("2. CONFORMAL CALIBRATION (HELD-OUT CALIBRATION SET)")
    print("=" * 70)
    calib_preds = predictor.predict(split.X_calib)

    conformal = SplitConformalPredictor(alpha=0.05)
    conformal.calibrate(split.y_calib, calib_preds)
    print(f"[OK] Calibrated on frames {int(split.calib_frames.min())} -> {int(split.calib_frames.max())} ({len(split.X_calib)} samples)")
    print(f"     Calibrated 3D Radius (95% target) : {conformal.radius_3d_mm:.4f} mm")
    print(f"     Axis Margins (X, Y, Z)           : [+/-{conformal.axis_margins_mm[0]:.4f}, +/-{conformal.axis_margins_mm[1]:.4f}, +/-{conformal.axis_margins_mm[2]:.4f}] mm")

    # 4. Predict on held-out Test set
    print("\n" + "=" * 70)
    print("3. HELD-OUT TEST PREDICTION & EVALUATION")
    print("=" * 70)
    test_preds = predictor.predict(split.X_test)
    test_intervals = conformal.predict_intervals(test_preds)

    # Metrics evaluation
    metrics = compute_prediction_errors(split.y_test, test_preds)

    print("\n" + "-" * 70)
    print("ACCURACY METRICS (HELD-OUT TEST SET)")
    print("-" * 70)
    print(f"MAE 3D (Mean Euclidean)  : {metrics['mae_3d']:.4f} mm")
    print(f"RMSE 3D                  : {metrics['rmse_3d']:.4f} mm")
    print(f"Median 3D Error (P50)    : {metrics['median_3d']:.4f} mm")
    print(f"P90 3D Error             : {metrics['p90_3d']:.4f} mm")
    print(f"P95 3D Error             : {metrics['p95_3d']:.4f} mm")
    print(f"P99 3D Error             : {metrics['p99_3d']:.4f} mm")
    print(f"Maximum 3D Error         : {metrics['max_3d']:.4f} mm")
    print("\nAXIS-WISE BREAKDOWN:")
    print(f"X -- MAE: {metrics['mae_x']:.4f} mm | RMSE: {metrics['rmse_x']:.4f} mm | P95: {metrics['p95_x']:.4f} mm")
    print(f"Y -- MAE: {metrics['mae_y']:.4f} mm | RMSE: {metrics['rmse_y']:.4f} mm | P95: {metrics['p95_y']:.4f} mm")
    print(f"Z -- MAE: {metrics['mae_z']:.4f} mm | RMSE: {metrics['rmse_z']:.4f} mm | P95: {metrics['p95_z']:.4f} mm")

    # 5. Coverage evaluation
    cov_report = conformal.evaluate_coverage(split.y_test, test_preds)
    print("\n" + cov_report.summary())

    # Build results DataFrame
    results_df = test_intervals.to_dataframe(frames=split.test_frames)
    results_df["actual_x_mm"] = split.y_test[:, 0]
    results_df["actual_y_mm"] = split.y_test[:, 1]
    results_df["actual_z_mm"] = split.y_test[:, 2]
    results_df["error_3d_mm"] = metrics["error_3d"]
    results_df["within_conformal_3d"] = metrics["error_3d"] <= conformal.radius_3d_mm

    results_df.to_csv(OUTPUT_FILE, index=False)
    results_df.to_csv(OUTPUTS_DIR / "conformal_prediction_results.csv", index=False)

    print(f"\n[OK] Results saved to:\n{OUTPUT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
