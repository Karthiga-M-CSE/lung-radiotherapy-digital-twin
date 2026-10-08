from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATASET_DIR = Path(
    r"C:\Users\smk28\Downloads\DATASET1\data"
)

INPUT_FILE = DATASET_DIR / "tumor_motion_with_rpm.csv"

OUTPUT_FILE = DATASET_DIR / "respiratory_prediction_results.csv"

N_LAGS = 5

MIN_TRAINING_SAMPLES = 50

ALPHA = 1.0

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"\nInput file not found:\n{INPUT_FILE}\n"
    )

data = pd.read_csv(INPUT_FILE)

required_columns = [
    "frame",
    "x_mm",
    "y_mm",
    "z_mm",
    "rpm"
]

for column in required_columns:

    if column not in data.columns:

        raise ValueError(
            f"\nRequired column '{column}' "
            f"was not found in:\n{INPUT_FILE}\n\n"
            f"Available columns:\n"
            f"{list(data.columns)}"
        )

data = data.dropna(
    subset=required_columns
).reset_index(drop=True)

if len(data) < MIN_TRAINING_SAMPLES + N_LAGS + 2:

    raise ValueError(
        "\nNot enough samples for prediction.\n"
        f"Samples available: {len(data)}\n"
        f"Minimum required: "
        f"{MIN_TRAINING_SAMPLES + N_LAGS + 2}"
    )

data = data.sort_values(
    "frame"
).reset_index(drop=True)

frames = data[
    "frame"
].to_numpy(dtype=float)

rpm = data[
    "rpm"
].to_numpy(dtype=float)

positions = data[
    ["x_mm", "y_mm", "z_mm"]
].to_numpy(dtype=float)

n = len(data)

def create_features(index):

    if index < N_LAGS:

        raise ValueError(
            f"Cannot create features for index {index}. "
            f"Need at least {N_LAGS} previous samples."
        )

    start = index - N_LAGS
    end = index + 1

    feature_vector = []

    feature_vector.extend(
        rpm[start:end]
    )

    feature_vector.extend(
        positions[start:end, 0]
    )

    feature_vector.extend(
        positions[start:end, 1]
    )

    feature_vector.extend(
        positions[start:end, 2]
    )

    return np.asarray(
        feature_vector,
        dtype=float
    )

predicted_positions = []

actual_positions = []

prediction_frames = []

for target_index in range(
    MIN_TRAINING_SAMPLES,
    n
):

    latest_input_index = (
        target_index - 1
    )

    X_train = []

    y_train = []

    first_training_target = (
        N_LAGS + 1
    )

    for train_target in range(
        first_training_target,
        target_index
    ):

        feature_index = (
            train_target - 1
        )

        feature_vector = create_features(
            feature_index
        )

        X_train.append(
            feature_vector
        )

        y_train.append(
            positions[train_target]
        )

    X_train = np.vstack(
        X_train
    )

    y_train = np.vstack(
        y_train
    )

    current_features = create_features(
        latest_input_index
    )

    current_features = current_features.reshape(
        1,
        -1
    )

    model = make_pipeline(
        StandardScaler(),
        Ridge(
            alpha=ALPHA
        )
    )

    model.fit(
        X_train,
        y_train
    )

    predicted_position = model.predict(
        current_features
    )[0]

    actual_position = positions[
        target_index
    ]

    predicted_positions.append(
        predicted_position
    )

    actual_positions.append(
        actual_position
    )

    prediction_frames.append(
        frames[target_index]
    )

predicted_positions = np.asarray(
    predicted_positions
)

actual_positions = np.asarray(
    actual_positions
)

prediction_frames = np.asarray(
    prediction_frames
)

errors = (
    actual_positions
    - predicted_positions
)

absolute_errors = np.abs(
    errors
)

error_3d = np.linalg.norm(
    errors,
    axis=1
)

results = pd.DataFrame({

    "frame":
        prediction_frames,

    "actual_x_mm":
        actual_positions[:, 0],

    "actual_y_mm":
        actual_positions[:, 1],

    "actual_z_mm":
        actual_positions[:, 2],

    "pred_x_mm":
        predicted_positions[:, 0],

    "pred_y_mm":
        predicted_positions[:, 1],

    "pred_z_mm":
        predicted_positions[:, 2],

    "error_x_mm":
        errors[:, 0],

    "error_y_mm":
        errors[:, 1],

    "error_z_mm":
        errors[:, 2],

    "absolute_error_x_mm":
        absolute_errors[:, 0],

    "absolute_error_y_mm":
        absolute_errors[:, 1],

    "absolute_error_z_mm":
        absolute_errors[:, 2],

    "prediction_error_3d_mm":
        error_3d
})

mean_error = np.mean(
    error_3d
)

median_error = np.median(
    error_3d
)

p90_error = np.percentile(
    error_3d,
    90
)

p95_error = np.percentile(
    error_3d,
    95
)

p99_error = np.percentile(
    error_3d,
    99
)

max_error = np.max(
    error_3d
)

mean_x_error = np.mean(
    absolute_errors[:, 0]
)

mean_y_error = np.mean(
    absolute_errors[:, 1]
)

mean_z_error = np.mean(
    absolute_errors[:, 2]
)

p95_x_error = np.percentile(
    absolute_errors[:, 0],
    95
)

p95_y_error = np.percentile(
    absolute_errors[:, 1],
    95
)

p95_z_error = np.percentile(
    absolute_errors[:, 2],
    95
)

results.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("=" * 60)
print("RESPIRATORY-INFORMED MOTION PREDICTION")
print("=" * 60)

print()
print(f"Total samples: {n}")

print(
    f"Prediction samples: {len(results)}"
)

print(
    f"History window: {N_LAGS + 1} samples"
)

print(
    f"Training begins at frame: "
    f"{MIN_TRAINING_SAMPLES}"
)

print()
print("=" * 60)
print("3D PREDICTION ERROR")
print("=" * 60)

print(
    f"\nMean error:    "
    f"{mean_error:.4f} mm"
)

print(
    f"Median error:  "
    f"{median_error:.4f} mm"
)

print(
    f"P90 error:     "
    f"{p90_error:.4f} mm"
)

print(
    f"P95 error:     "
    f"{p95_error:.4f} mm"
)

print(
    f"P99 error:     "
    f"{p99_error:.4f} mm"
)

print(
    f"Maximum error: "
    f"{max_error:.4f} mm"
)

print()
print("=" * 60)
print("AXIS-SPECIFIC ERROR")
print("=" * 60)

print(
    f"\nMean X error: "
    f"{mean_x_error:.4f} mm"
)

print(
    f"Mean Y error: "
    f"{mean_y_error:.4f} mm"
)

print(
    f"Mean Z error: "
    f"{mean_z_error:.4f} mm"
)

print(
    f"\nP95 X error: "
    f"{p95_x_error:.4f} mm"
)

print(
    f"P95 Y error: "
    f"{p95_y_error:.4f} mm"
)

print(
    f"P95 Z error: "
    f"{p95_z_error:.4f} mm"
)

plt.figure(
    figsize=(12, 5)
)

plt.plot(
    results["frame"],
    results["actual_z_mm"],
    label="Actual tumor Z"
)

plt.plot(
    results["frame"],
    results["pred_z_mm"],
    label="Respiratory-informed prediction"
)

plt.xlabel(
    "Frame"
)

plt.ylabel(
    "Tumor Z Position (mm)"
)

plt.title(
    "Actual vs Respiratory-Informed Predicted Tumor Z"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()

plt.figure(
    figsize=(12, 5)
)

plt.plot(
    results["frame"],
    results["prediction_error_3d_mm"]
)

plt.axhline(
    p95_error,
    linestyle="--",
    label=f"P95 error = {p95_error:.2f} mm"
)

plt.xlabel(
    "Frame"
)

plt.ylabel(
    "3D Prediction Error (mm)"
)

plt.title(
    "Respiratory-Informed Tumor Motion Prediction Error"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()

plt.figure(
    figsize=(7, 7)
)

plt.scatter(
    results["actual_z_mm"],
    results["pred_z_mm"]
)

minimum = min(
    results["actual_z_mm"].min(),
    results["pred_z_mm"].min()
)

maximum = max(
    results["actual_z_mm"].max(),
    results["pred_z_mm"].max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--",
    label="Perfect prediction"
)

plt.xlabel(
    "Actual Tumor Z (mm)"
)

plt.ylabel(
    "Predicted Tumor Z (mm)"
)

plt.title(
    "Actual vs Predicted Tumor Z"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()

print()
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

KALMAN_P95 = 8.81

print(
    f"\nConstant-velocity Kalman P95: "
    f"{KALMAN_P95:.2f} mm"
)

print(
    f"Respiratory-informed P95:      "
    f"{p95_error:.4f} mm"
)

if p95_error < KALMAN_P95:

    improvement = (
        (KALMAN_P95 - p95_error)
        / KALMAN_P95
        * 100
    )

    print(
        f"\nRESULT: Respiratory-informed "
        f"prediction is BETTER."
    )

    print(
        f"Improvement: {improvement:.2f}%"
    )

else:

    difference = (
        p95_error - KALMAN_P95
    )

    print(
        f"\nRESULT: Kalman baseline is "
        f"BETTER by {difference:.4f} mm."
    )

print()
print("=" * 60)

print(
    "RESULTS SAVED TO:"
)

print(
    OUTPUT_FILE
)

print("=" * 60)
