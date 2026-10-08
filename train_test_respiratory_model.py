import os
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

DATA_DIR = r"C:\Users\smk28\Downloads\DATASET1\data"
PROJECT_DIR = r"C:\Users\smk28\Downloads\RADIOTHERAPY SHIT"

INPUT_FILE = os.path.join(
    DATA_DIR,
    "tumor_motion_with_rpm.csv"
)

CYCLE_FILE = os.path.join(
    DATA_DIR,
    "respiratory_cycle_boundaries.csv"
)

MODEL_FILE = os.path.join(
    PROJECT_DIR,
    "respiratory_ridge_model.joblib"
)

SCALER_FILE = os.path.join(
    PROJECT_DIR,
    "respiratory_ridge_scaler.joblib"
)

OUTPUT_FILE = os.path.join(
    DATA_DIR,
    "respiratory_heldout_test_results.csv"
)

N_LAGS = 5
ALPHA = 1.0

TRAIN_START_FRAME = 8
TRAIN_END_FRAME = 138

TEST_START_FRAME = 139
TEST_END_FRAME = 179

print("=" * 70)
print("RESPIRATORY MODEL — CYCLE-BASED TRAIN / TEST")
print("=" * 70)

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"\nMotion dataset not found:\n{INPUT_FILE}"
    )

if not os.path.exists(CYCLE_FILE):
    raise FileNotFoundError(
        f"\nRespiratory cycle file not found:\n{CYCLE_FILE}"
    )

df = pd.read_csv(INPUT_FILE)
cycles = pd.read_csv(CYCLE_FILE)

print("\nLoaded motion data:")
print(f"Rows: {len(df)}")

print("\nColumns:")
print(df.columns.tolist())

required_columns = [
    "frame",
    "x_mm",
    "y_mm",
    "z_mm",
    "rpm"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"\nMissing required columns: {missing_columns}"
    )

df = df.sort_values(
    "frame"
).reset_index(drop=True)

feature_columns = [
    "rpm",
    "x_mm",
    "y_mm",
    "z_mm"
]

for lag in range(1, N_LAGS + 1):

    for column in feature_columns:

        df[f"{column}_lag{lag}"] = (
            df[column].shift(lag)
        )

df_model = (
    df
    .dropna()
    .reset_index(drop=True)
)

print("\nRows after lag-feature preparation:")
print(len(df_model))

lag_columns = []

for lag in range(1, N_LAGS + 1):

    for column in feature_columns:

        lag_columns.append(
            f"{column}_lag{lag}"
        )

X = df_model[
    lag_columns
].values

y = df_model[
    [
        "x_mm",
        "y_mm",
        "z_mm"
    ]
].values

train_mask = (
    (df_model["frame"] >= TRAIN_START_FRAME)
    &
    (df_model["frame"] <= TRAIN_END_FRAME)
)

test_mask = (
    (df_model["frame"] >= TEST_START_FRAME)
    &
    (df_model["frame"] <= TEST_END_FRAME)
)

X_train = X[train_mask]
y_train = y[train_mask]

X_test = X[test_mask]
y_test = y[test_mask]

train_frames = (
    df_model
    .loc[train_mask, "frame"]
    .values
)

test_frames = (
    df_model
    .loc[test_mask, "frame"]
    .values
)

print("\n" + "=" * 70)
print("TEMPORAL SPLIT")
print("=" * 70)

print(
    f"Training frames : "
    f"{train_frames[0]} → {train_frames[-1]}"
)

print(
    f"Training rows   : "
    f"{len(X_train)}"
)

print(
    f"\nTesting frames  : "
    f"{test_frames[0]} → {test_frames[-1]}"
)

print(
    f"Testing rows    : "
    f"{len(X_test)}"
)

if len(X_train) == 0:
    raise RuntimeError(
        "Training set is empty."
    )

if len(X_test) == 0:
    raise RuntimeError(
        "Testing set is empty."
    )

train_frame_set = set(train_frames)
test_frame_set = set(test_frames)

if train_frame_set.intersection(test_frame_set):

    raise RuntimeError(
        "ERROR: Training and testing frames overlap!"
    )

if train_frames.max() >= test_frames.min():

    raise RuntimeError(
        "ERROR: Training data extends into the test period!"
    )

print("\n✓ No train/test frame overlap")
print("✓ Training occurs before testing")
print("✓ Temporal split is locked")

print("\n" + "=" * 70)
print("FEATURE STANDARDIZATION")
print("=" * 70)

scaler = StandardScaler()

X_train_scaled = (
    scaler.fit_transform(X_train)
)

X_test_scaled = (
    scaler.transform(X_test)
)

print("✓ Scaler fitted using training data only")

print("\n" + "=" * 70)
print("TRAINING RIDGE MODEL")
print("=" * 70)

model = Ridge(
    alpha=ALPHA
)

model.fit(
    X_train_scaled,
    y_train
)

print("✓ Model trained")

joblib.dump(
    model,
    MODEL_FILE
)

joblib.dump(
    scaler,
    SCALER_FILE
)

print("\nSaved model:")
print(MODEL_FILE)

print("\nSaved scaler:")
print(SCALER_FILE)

print("\n" + "=" * 70)
print("HELD-OUT TEST PREDICTION")
print("=" * 70)

y_pred = model.predict(
    X_test_scaled
)

print(
    f"Generated predictions: "
    f"{len(y_pred)}"
)

error_x = np.abs(
    y_test[:, 0]
    -
    y_pred[:, 0]
)

error_y = np.abs(
    y_test[:, 1]
    -
    y_pred[:, 1]
)

error_z = np.abs(
    y_test[:, 2]
    -
    y_pred[:, 2]
)

error_3d = np.sqrt(

    (y_test[:, 0] - y_pred[:, 0]) ** 2

    +

    (y_test[:, 1] - y_pred[:, 1]) ** 2

    +

    (y_test[:, 2] - y_pred[:, 2]) ** 2
)

results = pd.DataFrame({

    "frame": test_frames,

    "actual_x_mm":
        y_test[:, 0],

    "actual_y_mm":
        y_test[:, 1],

    "actual_z_mm":
        y_test[:, 2],

    "predicted_x_mm":
        y_pred[:, 0],

    "predicted_y_mm":
        y_pred[:, 1],

    "predicted_z_mm":
        y_pred[:, 2],

    "error_x_mm":
        error_x,

    "error_y_mm":
        error_y,

    "error_z_mm":
        error_z,

    "error_3d_mm":
        error_3d
})

results.to_csv(
    OUTPUT_FILE,
    index=False
)

mean_3d = np.mean(
    error_3d
)

median_3d = np.median(
    error_3d
)

p90_3d = np.percentile(
    error_3d,
    90
)

p95_3d = np.percentile(
    error_3d,
    95
)

p99_3d = np.percentile(
    error_3d,
    99
)

max_3d = np.max(
    error_3d
)

mean_x = np.mean(
    error_x
)

mean_y = np.mean(
    error_y
)

mean_z = np.mean(
    error_z
)

p95_x = np.percentile(
    error_x,
    95
)

p95_y = np.percentile(
    error_y,
    95
)

p95_z = np.percentile(
    error_z,
    95
)

print("\n" + "=" * 70)
print("HELD-OUT TEST RESULTS")
print("=" * 70)

print(
    f"\nNumber of test predictions: "
    f"{len(results)}"
)

print("\n3D ERROR")

print(
    f"Mean   : "
    f"{mean_3d:.4f} mm"
)

print(
    f"Median : "
    f"{median_3d:.4f} mm"
)

print(
    f"P90    : "
    f"{p90_3d:.4f} mm"
)

print(
    f"P95    : "
    f"{p95_3d:.4f} mm"
)

print(
    f"P99    : "
    f"{p99_3d:.4f} mm"
)

print(
    f"Maximum: "
    f"{max_3d:.4f} mm"
)

print("\nAXIS-WISE MEAN ERROR")

print(
    f"X: {mean_x:.4f} mm"
)

print(
    f"Y: {mean_y:.4f} mm"
)

print(
    f"Z: {mean_z:.4f} mm"
)

print("\nAXIS-WISE P95 ERROR")

print(
    f"X: {p95_x:.4f} mm"
)

print(
    f"Y: {p95_y:.4f} mm"
)

print(
    f"Z: {p95_z:.4f} mm"
)

print("\n" + "=" * 70)
print("OUTPUT")
print("=" * 70)

print("\nResults saved to:")
print(
    OUTPUT_FILE
)

print("\nModel saved to:")
print(
    MODEL_FILE
)

print("\nScaler saved to:")
print(
    SCALER_FILE
)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
