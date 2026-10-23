import os
import numpy as np
import pandas as pd

from src.config import (
    resolve_file,
    PROJECT_ROOT,
    OUTPUTS_DIR,
    KALMAN_DT,
    KALMAN_PROCESS_NOISE_VAR,
    KALMAN_MEASUREMENT_NOISE_VAR,
)
from src.models.kalman_filter import ConstantVelocityKalmanFilter
from src.models.evaluation import evaluate_model_predictions
from src.utils.logging import get_logger

logger = get_logger("kalman_motion_predictor")

INPUT_FILE = resolve_file("tumor_motion_with_rpm.csv")
OUTPUT_FILE = resolve_file("kalman_prediction_results.csv", preferred_dir=PROJECT_ROOT)


def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file not found:\n{INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    positions = df[["x_mm", "y_mm", "z_mm"]].values.astype(float)
    frames = df["frame"].values

    kf = ConstantVelocityKalmanFilter(
        dt=KALMAN_DT,
        q_var=KALMAN_PROCESS_NOISE_VAR,
        r_var=KALMAN_MEASUREMENT_NOISE_VAR,
    )

    results = kf.predict_sequence(positions, frames=frames)

    # Save to project root and outputs directory for consistency
    results.to_csv(OUTPUT_FILE, index=False)
    results.to_csv(OUTPUTS_DIR / "kalman_prediction_results.csv", index=False)

    actual = results[["actual_x_mm", "actual_y_mm", "actual_z_mm"]].values
    predicted = results[["predicted_x_mm", "predicted_y_mm", "predicted_z_mm"]].values
    _, metrics = evaluate_model_predictions(actual, predicted, frames=results["frame"].values, model_name="Kalman")

    print("\n" + "=" * 60)
    print("CONSTANT-VELOCITY KALMAN MOTION PREDICTION")
    print("=" * 60)
    print(f"\nFrames processed: {len(positions)}")
    print("\nONE-STEP-AHEAD 3D PREDICTION ERROR")
    print(f"Mean error:    {metrics.mean_3d:.4f} mm")
    print(f"Median error:  {metrics.median_3d:.4f} mm")
    print(f"P90 error:     {metrics.p90_3d:.4f} mm")
    print(f"P95 error:     {metrics.p95_3d:.4f} mm")
    print(f"P99 error:     {metrics.p99_3d:.4f} mm")
    print(f"Maximum error: {metrics.max_3d:.4f} mm")

    print("\n" + "=" * 60)
    print("RESULTS SAVED TO:")
    print(OUTPUT_FILE)
    print("=" * 60)


if __name__ == "__main__":
    main()
