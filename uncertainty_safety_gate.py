import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

# FINAL HELD-OUT RIDGE PREDICTIONS
INPUT_FILE = Path(
    r"C:\Users\smk28\Downloads\DATASET1\data\respiratory_heldout_test_results.csv"
)

OUTPUT_FILE = INPUT_FILE.parent / "uncertainty_safety_gate_results.csv"


# ============================================================
# LOAD RESULTS
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("\nLoaded FINAL held-out prediction results")
print("-" * 60)
print("Rows:", len(df))
print("Columns:")
print(df.columns.tolist())


# ============================================================
# IDENTIFY 3D ERROR
# ============================================================

possible_3d_columns = [
    "prediction_error_3d_mm",
    "error_3d_mm",
    "3d_error_mm",
    "error_3d",
]

error_col = None

for col in possible_3d_columns:
    if col in df.columns:
        error_col = col
        break


# If no 3D error column exists, calculate it from axis errors.
if error_col is None:

    possible_axis_sets = [
        ("error_x_mm", "error_y_mm", "error_z_mm"),
        ("x_error_mm", "y_error_mm", "z_error_mm"),
        ("abs_error_x_mm", "abs_error_y_mm", "abs_error_z_mm"),
    ]

    found = False

    for x_col, y_col, z_col in possible_axis_sets:
        if all(c in df.columns for c in [x_col, y_col, z_col]):

            df["prediction_error_3d_mm"] = np.sqrt(
                df[x_col] ** 2 +
                df[y_col] ** 2 +
                df[z_col] ** 2
            )

            error_col = "prediction_error_3d_mm"
            found = True
            break

    if not found:
        raise ValueError(
            "\nCould not find a 3D error column or X/Y/Z error columns.\n"
            "Please check the printed column names."
        )


# ============================================================
# EMPIRICAL UNCERTAINTY
# ============================================================

errors = pd.to_numeric(
    df[error_col],
    errors="coerce"
).dropna()

p50 = np.percentile(errors, 50)
p90 = np.percentile(errors, 90)
p95 = np.percentile(errors, 95)
p99 = np.percentile(errors, 99)
maximum = np.max(errors)
mean_error = np.mean(errors)


print("\n")
print("=" * 60)
print("FINAL HELD-OUT MOTION-PREDICTION UNCERTAINTY")
print("=" * 60)

print(f"Mean error : {mean_error:.4f} mm")
print(f"P50 error  : {p50:.4f} mm")
print(f"P90 error  : {p90:.4f} mm")
print(f"P95 error  : {p95:.4f} mm")
print(f"P99 error  : {p99:.4f} mm")
print(f"Maximum    : {maximum:.4f} mm")


# ============================================================
# RESEARCH SAFETY GATE
# ============================================================

# Use the empirical P95 from the FINAL held-out test set.
# IMPORTANT:
# This is NOT a clinically validated beam-hold threshold.

SAFETY_THRESHOLD_MM = p95

df["uncertainty_bound_mm"] = SAFETY_THRESHOLD_MM

df["safety_status"] = np.where(
    df[error_col] <= SAFETY_THRESHOLD_MM,
    "SAFE / CONTINUE",
    "FLAG / REVIEW"
)

df["safety_threshold_mm"] = SAFETY_THRESHOLD_MM

df["threshold_exceeded"] = (
    df[error_col] > SAFETY_THRESHOLD_MM
)


# ============================================================
# SUMMARY
# ============================================================

safe_count = (
    df["safety_status"] == "SAFE / CONTINUE"
).sum()

flag_count = (
    df["safety_status"] == "FLAG / REVIEW"
).sum()

safe_percentage = safe_count / len(df) * 100
flag_percentage = flag_count / len(df) * 100


print("\n")
print("=" * 60)
print("FINAL SAFETY GATE RESULT")
print("=" * 60)

print(
    f"Research P95 threshold : "
    f"{SAFETY_THRESHOLD_MM:.4f} mm"
)

print(
    f"SAFE / CONTINUE        : "
    f"{safe_count} frames ({safe_percentage:.2f}%)"
)

print(
    f"FLAG / REVIEW          : "
    f"{flag_count} frames ({flag_percentage:.2f}%)"
)


# ============================================================
# SAVE RESULTS
# ============================================================

df.to_csv(OUTPUT_FILE, index=False)

print("\nSaved:")
print(OUTPUT_FILE)


# ============================================================
# PLOT 1 — PREDICTION ERROR + SAFETY THRESHOLD
# ============================================================

plt.figure(figsize=(12, 5))

plt.plot(
    df["frame"] if "frame" in df.columns else df.index,
    df[error_col],
    linewidth=1.5,
    label="3D prediction error"
)

plt.axhline(
    SAFETY_THRESHOLD_MM,
    linestyle="--",
    linewidth=2,
    label=f"Research P95 threshold ({SAFETY_THRESHOLD_MM:.2f} mm)"
)

plt.xlabel("Frame")
plt.ylabel("3D prediction error (mm)")
plt.title(
    "Respiratory-Informed Tumor Motion Prediction Error"
)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plot1 = (
    INPUT_FILE.parent /
    "prediction_error_safety_gate.png"
)

plt.savefig(plot1, dpi=300)
plt.show()


# ============================================================
# PLOT 2 — ERROR DISTRIBUTION
# ============================================================

plt.figure(figsize=(9, 5))

plt.hist(
    errors,
    bins=30,
    edgecolor="black"
)

plt.axvline(
    p95,
    linestyle="--",
    linewidth=2,
    label=f"P95 = {p95:.2f} mm"
)

plt.xlabel("3D prediction error (mm)")
plt.ylabel("Number of frames")
plt.title(
    "Distribution of Tumor Motion Prediction Error"
)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plot2 = (
    INPUT_FILE.parent /
    "prediction_error_distribution.png"
)

plt.savefig(plot2, dpi=300)
plt.show()


print("\nPlots saved:")
print(plot1)
print(plot2)

print("\nDONE.")