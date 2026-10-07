import os
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge


# ============================================================
# RESPIRATORY MODEL — CYCLE-BASED TRAIN / TEST
# ============================================================


# ============================================================
# 1. PATHS
# ============================================================

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


# ============================================================
# 2. MODEL SETTINGS
# ============================================================

N_LAGS = 5
ALPHA = 1.0


# ============================================================
# 3. LOCKED RESPIRATORY-CYCLE SPLIT
# ============================================================
#
# Respiratory cycle inspection identified:
#
# Training cycles: 1–13
# Testing cycles : 14–17
#
# Complete-cycle frame ranges:
#
# TRAIN = 8–138
# TEST  = 139–179
#
# Frames 0–7 and 180–181 are outside the complete
# peak-to-peak cycles and are excluded.
#
# IMPORTANT:
# The test cycles are NEVER used to fit the model
# or the scaler.
# ============================================================

TRAIN_START_FRAME = 8
TRAIN_END_FRAME = 138

TEST_START_FRAME = 139
TEST_END_FRAME = 179


# ============================================================
# 4. START
# ============================================================

print("=" * 70)
print("RESPIRATORY MODEL — CYCLE-BASED TRAIN / TEST")
print("=" * 70)


# ============================================================
# 5. LOAD MOTION DATA
# ============================================================

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


# ============================================================
# 6. CHECK REQUIRED COLUMNS
# ============================================================

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


# ============================================================
# 7. SORT BY FRAME
# ============================================================

df = df.sort_values(
    "frame"
).reset_index(drop=True)


# ============================================================
# 8. CREATE LAG FEATURES
# ============================================================
#
# The model uses the previous 5 observations of:
#
# RPM
# X position
# Y position
# Z position
#
# Example:
#
# rpm_lag1
# rpm_lag2
# ...
# rpm_lag5
#
# x_mm_lag1
# ...
#
# ============================================================

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


# ============================================================
# 9. REMOVE ROWS WITHOUT COMPLETE HISTORY
# ============================================================

df_model = (
    df
    .dropna()
    .reset_index(drop=True)
)


print("\nRows after lag-feature preparation:")
print(len(df_model))


# ============================================================
# 10. DEFINE FEATURE COLUMNS
# ============================================================

lag_columns = []

for lag in range(1, N_LAGS + 1):

    for column in feature_columns:

        lag_columns.append(
            f"{column}_lag{lag}"
        )


# ============================================================
# 11. CREATE X AND y
# ============================================================

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


# ============================================================
# 12. STRICT TEMPORAL TRAIN / TEST SPLIT
# ============================================================
#
# TRAIN:
# frames 8–138
#
# TEST:
# frames 139–179
#
# No random splitting.
# No shuffling.
# No future test data used during training.
# ============================================================

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


# ============================================================
# 13. VERIFY SPLIT
# ============================================================

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


# ============================================================
# 14. SAFETY CHECKS
# ============================================================

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


# ============================================================
# 15. STANDARDIZE FEATURES
# ============================================================
#
# IMPORTANT:
#
# The scaler is fitted ONLY on training data.
#
# The test data is transformed using the already-fitted
# training scaler.
#
# This prevents information leakage.
# ============================================================

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


# ============================================================
# 16. TRAIN RIDGE MODEL
# ============================================================

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


# ============================================================
# 17. SAVE TRAINED MODEL
# ============================================================

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


# ============================================================
# 18. HELD-OUT TEST PREDICTION
# ============================================================

print("\n" + "=" * 70)
print("HELD-OUT TEST PREDICTION")
print("=" * 70)


# IMPORTANT:
# Test data is ONLY passed through the already-trained model.

y_pred = model.predict(
    X_test_scaled
)


print(
    f"Generated predictions: "
    f"{len(y_pred)}"
)


# ============================================================
# 19. AXIS-WISE ABSOLUTE ERROR
# ============================================================

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


# ============================================================
# 20. 3D EUCLIDEAN ERROR
# ============================================================

error_3d = np.sqrt(

    (y_test[:, 0] - y_pred[:, 0]) ** 2

    +

    (y_test[:, 1] - y_pred[:, 1]) ** 2

    +

    (y_test[:, 2] - y_pred[:, 2]) ** 2
)


# ============================================================
# 21. CREATE RESULTS TABLE
# ============================================================

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


# ============================================================
# 22. SAVE TEST RESULTS
# ============================================================

results.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 23. CALCULATE 3D METRICS
# ============================================================

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


# ============================================================
# 24. AXIS-WISE METRICS
# ============================================================

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


# ============================================================
# 25. PRINT FINAL RESULTS
# ============================================================

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


# ============================================================
# 26. OUTPUT INFORMATION
# ============================================================

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


# ============================================================
# 27. FINAL STATUS
# ============================================================

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)