import os
import numpy as np
import pandas as pd

DATA_DIR = r"C:\Users\gayu0\OneDrive\Desktop\projects\DATASET1"

MOTION_FILE = os.path.join(
    DATA_DIR,
    "tumor_motion_with_rpm.csv"
)

RIDGE_FILE = os.path.join(
    DATA_DIR,
    "respiratory_heldout_test_results.csv"
)

KALMAN_FILE = os.path.join(
    DATA_DIR,
    "kalman_prediction_results.csv"
)

OUTPUT_FILE = os.path.join(
    DATA_DIR,
    "heldout_model_comparison.csv"
)

TEST_START_FRAME = 139
TEST_END_FRAME = 179

print("=" * 70)
print("HELD-OUT MODEL COMPARISON")
print("KALMAN vs RESPIRATORY-INFORMED RIDGE")
print("=" * 70)

for file_path in [
    MOTION_FILE,
    RIDGE_FILE,
    KALMAN_FILE
]:

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"\nRequired file not found:\n{file_path}"
        )

motion = pd.read_csv(MOTION_FILE)
ridge = pd.read_csv(RIDGE_FILE)
kalman = pd.read_csv(KALMAN_FILE)

print("\nLoaded files:")

print(f"Motion data : {len(motion)} rows")
print(f"Ridge test  : {len(ridge)} rows")
print(f"Kalman data : {len(kalman)} rows")

required_kalman_columns = [
    "frame",
    "actual_x_mm",
    "actual_y_mm",
    "actual_z_mm",
    "predicted_x_mm",
    "predicted_y_mm",
    "predicted_z_mm"
]

missing = [
    col
    for col in required_kalman_columns
    if col not in kalman.columns
]

if missing:

    raise ValueError(
        f"\nMissing Kalman columns: {missing}"
    )

required_ridge_columns = [
    "frame",
    "actual_x_mm",
    "actual_y_mm",
    "actual_z_mm",
    "predicted_x_mm",
    "predicted_y_mm",
    "predicted_z_mm"
]

missing = [
    col
    for col in required_ridge_columns
    if col not in ridge.columns
]

if missing:

    raise ValueError(
        f"\nMissing Ridge columns: {missing}"
    )

ridge_test = ridge[
    (ridge["frame"] >= TEST_START_FRAME)
    &
    (ridge["frame"] <= TEST_END_FRAME)
].copy()

kalman_test = kalman[
    (kalman["frame"] >= TEST_START_FRAME)
    &
    (kalman["frame"] <= TEST_END_FRAME)
].copy()

ridge_test = ridge_test[
    [
        "frame",
        "actual_x_mm",
        "actual_y_mm",
        "actual_z_mm",
        "predicted_x_mm",
        "predicted_y_mm",
        "predicted_z_mm"
    ]
].copy()

ridge_test = ridge_test.rename(
    columns={
        "predicted_x_mm": "ridge_predicted_x_mm",
        "predicted_y_mm": "ridge_predicted_y_mm",
        "predicted_z_mm": "ridge_predicted_z_mm"
    }
)

kalman_test = kalman_test[
    [
        "frame",
        "actual_x_mm",
        "actual_y_mm",
        "actual_z_mm",
        "predicted_x_mm",
        "predicted_y_mm",
        "predicted_z_mm"
    ]
].copy()

kalman_test = kalman_test.rename(
    columns={
        "actual_x_mm": "kalman_actual_x_mm",
        "actual_y_mm": "kalman_actual_y_mm",
        "actual_z_mm": "kalman_actual_z_mm",

        "predicted_x_mm": "kalman_predicted_x_mm",
        "predicted_y_mm": "kalman_predicted_y_mm",
        "predicted_z_mm": "kalman_predicted_z_mm"
    }
)

comparison = pd.merge(
    ridge_test,
    kalman_test,
    on="frame",
    how="inner"
)

print("\n" + "=" * 70)
print("HELD-OUT TEST COVERAGE")
print("=" * 70)

print(
    f"Requested frames : "
    f"{TEST_START_FRAME} → {TEST_END_FRAME}"
)

print(
    f"Ridge frames     : "
    f"{len(ridge_test)}"
)

print(
    f"Kalman frames    : "
    f"{len(kalman_test)}"
)

print(
    f"Common frames    : "
    f"{len(comparison)}"
)

if len(comparison) == 0:

    raise RuntimeError(
        "\nNo common held-out frames found."
    )

ridge_frames = set(
    ridge_test["frame"]
)

kalman_frames = set(
    kalman_test["frame"]
)

common_frames = set(
    comparison["frame"]
)

print(
    f"\nCommon frame range: "
    f"{comparison['frame'].min()} → "
    f"{comparison['frame'].max()}"
)

if common_frames != ridge_frames.intersection(
    kalman_frames
):

    raise RuntimeError(
        "\nFrame matching error."
    )

print("✓ Both models evaluated on identical frames")

kalman_error_3d = np.sqrt(

    (
        comparison["kalman_actual_x_mm"]
        -
        comparison["kalman_predicted_x_mm"]
    ) ** 2

    +

    (
        comparison["kalman_actual_y_mm"]
        -
        comparison["kalman_predicted_y_mm"]
    ) ** 2

    +

    (
        comparison["kalman_actual_z_mm"]
        -
        comparison["kalman_predicted_z_mm"]
    ) ** 2
)

ridge_error_3d = np.sqrt(

    (
        comparison["actual_x_mm"]
        -
        comparison["ridge_predicted_x_mm"]
    ) ** 2

    +

    (
        comparison["actual_y_mm"]
        -
        comparison["ridge_predicted_y_mm"]
    ) ** 2

    +

    (
        comparison["actual_z_mm"]
        -
        comparison["ridge_predicted_z_mm"]
    ) ** 2
)

comparison["kalman_error_3d_mm"] = (
    kalman_error_3d
)

comparison["ridge_error_3d_mm"] = (
    ridge_error_3d
)

def metrics(errors):

    return {
        "mean": np.mean(errors),
        "median": np.median(errors),
        "p90": np.percentile(errors, 90),
        "p95": np.percentile(errors, 95),
        "p99": np.percentile(errors, 99),
        "maximum": np.max(errors)
    }

kalman_metrics = metrics(
    kalman_error_3d
)

ridge_metrics = metrics(
    ridge_error_3d
)

kalman_p95 = kalman_metrics["p95"]
ridge_p95 = ridge_metrics["p95"]

p95_reduction = (
    (kalman_p95 - ridge_p95)
    /
    kalman_p95
) * 100

print("\n" + "=" * 70)
print("HELD-OUT PERFORMANCE COMPARISON")
print("=" * 70)

print("\nMetric                  Kalman        Ridge")

print(
    f"Mean 3D error          "
    f"{kalman_metrics['mean']:.4f}       "
    f"{ridge_metrics['mean']:.4f} mm"
)

print(
    f"Median                 "
    f"{kalman_metrics['median']:.4f}       "
    f"{ridge_metrics['median']:.4f} mm"
)

print(
    f"P90                    "
    f"{kalman_metrics['p90']:.4f}       "
    f"{ridge_metrics['p90']:.4f} mm"
)

print(
    f"P95                    "
    f"{kalman_metrics['p95']:.4f}       "
    f"{ridge_metrics['p95']:.4f} mm"
)

print(
    f"P99                    "
    f"{kalman_metrics['p99']:.4f}       "
    f"{ridge_metrics['p99']:.4f} mm"
)

print(
    f"Maximum                "
    f"{kalman_metrics['maximum']:.4f}       "
    f"{ridge_metrics['maximum']:.4f} mm"
)

print("\n" + "=" * 70)
print("P95 IMPROVEMENT")
print("=" * 70)

print(
    f"\nKalman P95 : "
    f"{kalman_p95:.4f} mm"
)

print(
    f"Ridge P95  : "
    f"{ridge_p95:.4f} mm"
)

print(
    f"\nP95 reduction: "
    f"{p95_reduction:.2f}%"
)

if ridge_p95 < kalman_p95:

    print(
        "\nRESULT: "
        "Respiratory-informed Ridge performs "
        "BETTER on the held-out respiratory cycles."
    )

else:

    print(
        "\nRESULT: "
        "Kalman performs better on the held-out "
        "respiratory cycles."
    )

comparison.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("OUTPUT")
print("=" * 70)

print(
    "\nSaved comparison to:"
)

print(
    OUTPUT_FILE
)

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
