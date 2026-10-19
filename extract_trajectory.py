from pathlib import Path
import pandas as pd
import numpy as np

from src.config import DATASET_DIR, PROJECT_ROOT, OUTPUTS_DIR
from src.data.trajectory import (
    extract_trajectory_from_masks,
    generate_physical_trajectory,
    validate_tumor_trajectory,
)
from src.utils.logging import get_logger

logger = get_logger("extract_trajectory")

MASK_DIR = DATASET_DIR / "tumor_masks"
OUTPUT_FILE = PROJECT_ROOT / "tumor_trajectory.csv"


def main():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset folder not found:\n{DATASET_DIR}")

    if not MASK_DIR.exists():
        raise FileNotFoundError(f"Tumor mask folder not found:\n{MASK_DIR}")

    print("=" * 60)
    print("XCAT TUMOR TRAJECTORY EXTRACTION")
    print("=" * 60)
    print(f"\nDataset directory:\n{DATASET_DIR}")
    print(f"\nTumor mask directory:\n{MASK_DIR}")

    trajectory = extract_trajectory_from_masks(
        mask_source=MASK_DIR,
        pattern="tumormask_*.nii.gz",
        reference_frame=0,
        validate_spatial_consistency=True,
    )

    trajectory = generate_physical_trajectory(trajectory)

    # Validate against physiological bounds
    validation = validate_tumor_trajectory(trajectory)
    logger.info(f"Trajectory validation status: {'PASSED' if validation.passed else 'FAILED'}")

    # Save to outputs and project root for backward compatibility
    trajectory.to_csv(OUTPUT_FILE, index=False)
    trajectory.to_csv(OUTPUTS_DIR / "tumor_trajectory.csv", index=False)

    print("\n" + "=" * 60)
    print("TRAJECTORY EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"\nValid frames processed: {len(trajectory)}")
    print(f"Frame range: {trajectory['frame'].min()} to {trajectory['frame'].max()}")

    display_cols = [
        col for col in ["frame", "x_mm", "y_mm", "z_mm", "displacement_3d_mm", "tumor_voxels"]
        if col in trajectory.columns
    ]

    print("\n===== FIRST 10 FRAMES =====")
    print(trajectory[display_cols].head(10).to_string(index=False))

    print("\n===== LAST 5 FRAMES =====")
    print(trajectory[display_cols].tail(5).to_string(index=False))

    print("\n" + "=" * 60)
    print("MOTION RANGE")
    print("=" * 60)
    print(f"\nX position range: {trajectory['x_mm'].min():.3f} to {trajectory['x_mm'].max():.3f} mm")
    print(f"Y position range: {trajectory['y_mm'].min():.3f} to {trajectory['y_mm'].max():.3f} mm")
    print(f"Z position range: {trajectory['z_mm'].min():.3f} to {trajectory['z_mm'].max():.3f} mm")

    print(f"\nMaximum X displacement: {trajectory['dx_mm'].abs().max():.3f} mm")
    print(f"Maximum Y displacement: {trajectory['dy_mm'].abs().max():.3f} mm")
    print(f"Maximum Z displacement: {trajectory['dz_mm'].abs().max():.3f} mm")
    print(f"\nMaximum 3D displacement from frame 0: {trajectory['displacement_3d_mm'].max():.3f} mm")

    print("\n===== TUMOR MASK SIZE =====")
    print(f"Minimum tumor voxels: {trajectory['tumor_voxels'].min()}")
    print(f"Maximum tumor voxels: {trajectory['tumor_voxels'].max()}")
    print(f"Mean tumor voxels: {trajectory['tumor_voxels'].mean():.2f}")

    print("\n" + "=" * 60)
    print("CSV SAVED SUCCESSFULLY:")
    print(OUTPUT_FILE)
    print("=" * 60)


if __name__ == "__main__":
    main()
