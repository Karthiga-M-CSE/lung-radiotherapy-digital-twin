from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import resolve_file, PROJECT_ROOT, OUTPUTS_DIR
from src.data.trajectory import combine_trajectory_with_respiratory, validate_trajectory_against_respiratory
from src.utils.logging import get_logger

logger = get_logger("combine_motion_rpm")

TRAJECTORY_FILE = resolve_file("tumor_trajectory.csv")
RPM_FILE = resolve_file("rpm_signal.txt")
OUTPUT_FILE = resolve_file("tumor_motion_with_rpm.csv", preferred_dir=PROJECT_ROOT)


def main():
    if not TRAJECTORY_FILE.exists():
        raise FileNotFoundError(f"Trajectory file not found:\n{TRAJECTORY_FILE}")

    if not RPM_FILE.exists():
        raise FileNotFoundError(f"RPM signal file not found:\n{RPM_FILE}")

    trajectory = pd.read_csv(TRAJECTORY_FILE)
    print("=" * 60)
    print("LOADING XCAT MOTION DATA")
    print("=" * 60)
    print(f"\nTrajectory rows: {len(trajectory)}")

    rpm = np.loadtxt(RPM_FILE)
    print(f"RPM samples: {len(rpm)}")

    if len(trajectory) != len(rpm):
        raise ValueError(
            "Trajectory and RPM signal have different lengths!\n"
            f"Trajectory: {len(trajectory)}\n"
            f"RPM: {len(rpm)}"
        )

    trajectory = combine_trajectory_with_respiratory(trajectory, rpm, rpm_col="rpm", validate=True)
    validation = validate_trajectory_against_respiratory(trajectory, rpm)

    trajectory.to_csv(OUTPUT_FILE, index=False)
    trajectory.to_csv(OUTPUTS_DIR / "tumor_motion_with_rpm.csv", index=False)

    print("\n" + "=" * 60)
    print("RESPIRATORY SIGNAL")
    print("=" * 60)
    print(f"\nRPM minimum: {rpm.min():.3f}")
    print(f"RPM maximum: {rpm.max():.3f}")
    print(f"RPM mean: {rpm.mean():.3f}")
    print(f"RPM standard deviation: {rpm.std():.3f}")

    print("\n" + "=" * 60)
    print("COMBINED DATA")
    print("=" * 60)
    cols_to_show = [c for c in ["frame", "x_mm", "y_mm", "z_mm", "displacement_3d_mm", "rpm"] if c in trajectory.columns]
    print(trajectory[cols_to_show].head(10).to_string(index=False))

    correlation = trajectory["rpm"].corr(trajectory["z_mm"])
    print("\n" + "=" * 60)
    print("RESPIRATION-MOTION RELATIONSHIP")
    print("=" * 60)
    print(f"\nPearson correlation between RPM signal and tumor Z position: {correlation:.4f}")
    print(f"Validation Summary:\n{validation.summary()}")
    print("\n" + "=" * 60)
    print(f"COMBINED CSV SAVED:\n{OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
