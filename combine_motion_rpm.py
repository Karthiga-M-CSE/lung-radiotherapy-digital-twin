from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

DATASET_DIR = Path(
    r"C:\Users\smk28\Downloads\DATASET1\data"
)

TRAJECTORY_FILE = DATASET_DIR / "tumor_trajectory.csv"

RPM_FILE = DATASET_DIR / "rpm_signal.txt"

OUTPUT_FILE = DATASET_DIR / "tumor_motion_with_rpm.csv"


# ============================================================
# CHECK FILES
# ============================================================

if not TRAJECTORY_FILE.exists():
    raise FileNotFoundError(
        f"Trajectory file not found:\n{TRAJECTORY_FILE}"
    )

if not RPM_FILE.exists():
    raise FileNotFoundError(
        f"RPM signal file not found:\n{RPM_FILE}"
    )


# ============================================================
# LOAD TRAJECTORY
# ============================================================

trajectory = pd.read_csv(TRAJECTORY_FILE)

print("=" * 60)
print("LOADING XCAT MOTION DATA")
print("=" * 60)

print(f"\nTrajectory rows: {len(trajectory)}")


# ============================================================
# LOAD RESPIRATORY SIGNAL
# ============================================================

rpm = np.loadtxt(RPM_FILE)

print(f"RPM samples: {len(rpm)}")


# ============================================================
# VERIFY LENGTHS
# ============================================================

if len(trajectory) != len(rpm):

    raise ValueError(
        "Trajectory and RPM signal have different lengths!\n"
        f"Trajectory: {len(trajectory)}\n"
        f"RPM: {len(rpm)}"
    )


# ============================================================
# ADD RPM SIGNAL
# ============================================================

trajectory["rpm"] = rpm


# ============================================================
# SAVE COMBINED DATASET
# ============================================================

trajectory.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# BASIC STATISTICS
# ============================================================

print("\n")
print("=" * 60)
print("RESPIRATORY SIGNAL")
print("=" * 60)

print(
    f"\nRPM minimum: {rpm.min():.3f}"
)

print(
    f"RPM maximum: {rpm.max():.3f}"
)

print(
    f"RPM mean: {rpm.mean():.3f}"
)

print(
    f"RPM standard deviation: {rpm.std():.3f}"
)


# ============================================================
# DISPLAY FIRST ROWS
# ============================================================

print("\n")
print("=" * 60)
print("COMBINED DATA")
print("=" * 60)

print(
    trajectory[
        [
            "frame",
            "x_mm",
            "y_mm",
            "z_mm",
            "displacement_3d_mm",
            "rpm"
        ]
    ].head(10).to_string(index=False)
)


# ============================================================
# PLOT 1 — TUMOR Z POSITION
# ============================================================

plt.figure(figsize=(12, 5))

plt.plot(
    trajectory["frame"],
    trajectory["z_mm"]
)

plt.xlabel("Frame")
plt.ylabel("Tumor Z Position (mm)")
plt.title("XCAT Tumor Z-Position Across Respiratory Samples")

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# PLOT 2 — RESPIRATORY SIGNAL
# ============================================================

plt.figure(figsize=(12, 5))

plt.plot(
    trajectory["frame"],
    trajectory["rpm"]
)

plt.xlabel("Frame")
plt.ylabel("Respiratory Signal")
plt.title("XCAT Respiratory Signal")

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# PLOT 3 — Z POSITION VS RESPIRATORY SIGNAL
# ============================================================

plt.figure(figsize=(7, 6))

plt.scatter(
    trajectory["rpm"],
    trajectory["z_mm"]
)

plt.xlabel("Respiratory Signal")
plt.ylabel("Tumor Z Position (mm)")
plt.title("Tumor Z Position vs Respiratory Signal")

plt.grid(True)

plt.tight_layout()

plt.show()


# ============================================================
# CORRELATION
# ============================================================

correlation = trajectory["rpm"].corr(
    trajectory["z_mm"]
)

print("\n")
print("=" * 60)
print("RESPIRATION–MOTION RELATIONSHIP")
print("=" * 60)

print(
    f"\nPearson correlation between "
    f"RPM signal and tumor Z position: "
    f"{correlation:.4f}"
)


# ============================================================
# OUTPUT
# ============================================================

print("\n")
print("=" * 60)

print("COMBINED CSV SAVED:")

print(OUTPUT_FILE)

print("=" * 60)