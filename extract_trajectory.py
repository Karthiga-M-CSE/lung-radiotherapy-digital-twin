from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

DATASET_DIR = Path(r"C:\Users\smk28\Downloads\DATASET1\data")
MASK_DIR = DATASET_DIR / "tumor_masks"

OUTPUT_FILE = DATASET_DIR / "tumor_trajectory.csv"


# ============================================================
# CHECK DATASET PATHS
# ============================================================

if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"Dataset folder not found:\n{DATASET_DIR}"
    )

if not MASK_DIR.exists():
    raise FileNotFoundError(
        f"Tumor mask folder not found:\n{MASK_DIR}"
    )


# ============================================================
# GET FRAME NUMBER
# ============================================================

def get_frame_id(mask_path):
    """
    Extract frame number from filenames such as:

        tumormask_0.nii.gz
        tumormask_1.nii.gz
        tumormask_181.nii.gz

    Returns:
        integer frame number
    """

    filename = mask_path.name

    prefix = "tumormask_"
    suffix = ".nii.gz"

    if not filename.startswith(prefix):
        raise ValueError(
            f"Unexpected mask filename: {filename}"
        )

    if not filename.endswith(suffix):
        raise ValueError(
            f"Unexpected mask filename: {filename}"
        )

    frame_string = filename[len(prefix):-len(suffix)]

    return int(frame_string)


# ============================================================
# GET TUMOR CENTROID
# ============================================================

def get_tumor_centroid(mask_path):
    """
    Load a tumor mask and calculate its centroid.

    Returns:
        voxel_centroid
        physical_centroid_mm
        tumor_voxel_count
        mask_shape
        mask_affine
    """

    mask_img = nib.load(str(mask_path))

    mask = mask_img.get_fdata()

    # Tumor = every voxel with value > 0
    tumor_voxels = np.argwhere(mask > 0)

    if tumor_voxels.size == 0:
        return None

    # Centroid in voxel coordinates
    voxel_centroid = tumor_voxels.mean(axis=0)

    # Convert voxel coordinates to physical coordinates (mm)
    physical_centroid = nib.affines.apply_affine(
        mask_img.affine,
        voxel_centroid
    )

    tumor_voxel_count = len(tumor_voxels)

    return (
        voxel_centroid,
        physical_centroid,
        tumor_voxel_count,
        mask.shape,
        mask_img.affine
    )


# ============================================================
# FIND ALL TUMOR MASKS
# ============================================================

mask_files = list(
    MASK_DIR.glob("tumormask_*.nii.gz")
)

if len(mask_files) == 0:
    raise FileNotFoundError(
        f"No tumor mask files found in:\n{MASK_DIR}"
    )


# Sort using the actual numeric frame number
mask_files = sorted(
    mask_files,
    key=get_frame_id
)


print("=" * 60)
print("XCAT TUMOR TRAJECTORY EXTRACTION")
print("=" * 60)

print(f"\nDataset directory:")
print(DATASET_DIR)

print(f"\nTumor mask directory:")
print(MASK_DIR)

print(f"\nFound {len(mask_files)} tumor masks.")


# ============================================================
# PROCESS ALL MASKS
# ============================================================

results = []

reference_shape = None
reference_affine = None

for index, mask_path in enumerate(mask_files):

    frame_id = get_frame_id(mask_path)

    try:

        result = get_tumor_centroid(mask_path)

        if result is None:

            print(
                f"WARNING: No tumor voxels found in "
                f"{mask_path.name}"
            )

            continue

        (
            voxel_centroid,
            physical_centroid,
            tumor_voxel_count,
            mask_shape,
            mask_affine
        ) = result


        # ----------------------------------------------------
        # Check that all masks have the same geometry
        # ----------------------------------------------------

        if reference_shape is None:

            reference_shape = mask_shape
            reference_affine = mask_affine

        else:

            if mask_shape != reference_shape:

                raise ValueError(
                    f"Mask shape mismatch in {mask_path.name}\n"
                    f"Expected: {reference_shape}\n"
                    f"Found: {mask_shape}"
                )

            if not np.allclose(
                mask_affine,
                reference_affine
            ):

                raise ValueError(
                    f"Mask affine mismatch in {mask_path.name}"
                )


        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        results.append(
            {
                "frame": frame_id,

                "x_mm": float(physical_centroid[0]),
                "y_mm": float(physical_centroid[1]),
                "z_mm": float(physical_centroid[2]),

                "x_voxel": float(voxel_centroid[0]),
                "y_voxel": float(voxel_centroid[1]),
                "z_voxel": float(voxel_centroid[2]),

                "tumor_voxels": int(tumor_voxel_count)
            }
        )


    except Exception as e:

        print(
            f"ERROR processing {mask_path.name}: {e}"
        )

        raise


    # Progress indicator
    if (index + 1) % 20 == 0 or index == 0:

        print(
            f"Processed {index + 1}/{len(mask_files)} masks..."
        )


# ============================================================
# CHECK RESULTS
# ============================================================

if len(results) == 0:

    raise RuntimeError(
        "No valid tumor masks were processed."
    )


trajectory = pd.DataFrame(results)


# Sort by frame number
trajectory = trajectory.sort_values(
    "frame"
).reset_index(drop=True)


# ============================================================
# CALCULATE DISPLACEMENT FROM FIRST FRAME
# ============================================================

reference_x = trajectory.loc[0, "x_mm"]
reference_y = trajectory.loc[0, "y_mm"]
reference_z = trajectory.loc[0, "z_mm"]


trajectory["dx_mm"] = (
    trajectory["x_mm"] - reference_x
)

trajectory["dy_mm"] = (
    trajectory["y_mm"] - reference_y
)

trajectory["dz_mm"] = (
    trajectory["z_mm"] - reference_z
)


trajectory["displacement_3d_mm"] = np.sqrt(
    trajectory["dx_mm"] ** 2
    + trajectory["dy_mm"] ** 2
    + trajectory["dz_mm"] ** 2
)


# ============================================================
# CALCULATE MOTION MAGNITUDE
# ============================================================

trajectory["motion_magnitude_mm"] = np.sqrt(
    trajectory["dx_mm"] ** 2
    + trajectory["dy_mm"] ** 2
    + trajectory["dz_mm"] ** 2
)


# ============================================================
# SAVE CSV
# ============================================================

trajectory.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("TRAJECTORY EXTRACTION COMPLETE")
print("=" * 60)

print(
    f"\nValid frames processed: "
    f"{len(trajectory)}"
)

print(
    f"Frame range: "
    f"{trajectory['frame'].min()} "
    f"to "
    f"{trajectory['frame'].max()}"
)


print("\n===== FIRST 10 FRAMES =====")

print(
    trajectory[
        [
            "frame",
            "x_mm",
            "y_mm",
            "z_mm",
            "displacement_3d_mm",
            "tumor_voxels"
        ]
    ].head(10).to_string(index=False)
)


print("\n===== LAST 5 FRAMES =====")

print(
    trajectory[
        [
            "frame",
            "x_mm",
            "y_mm",
            "z_mm",
            "displacement_3d_mm",
            "tumor_voxels"
        ]
    ].tail(5).to_string(index=False)
)


# ============================================================
# MOTION RANGE
# ============================================================

print("\n")
print("=" * 60)
print("MOTION RANGE")
print("=" * 60)

print(
    f"\nX position range: "
    f"{trajectory['x_mm'].min():.3f} "
    f"to "
    f"{trajectory['x_mm'].max():.3f} mm"
)

print(
    f"Y position range: "
    f"{trajectory['y_mm'].min():.3f} "
    f"to "
    f"{trajectory['y_mm'].max():.3f} mm"
)

print(
    f"Z position range: "
    f"{trajectory['z_mm'].min():.3f} "
    f"to "
    f"{trajectory['z_mm'].max():.3f} mm"
)

print(
    f"\nMaximum X displacement: "
    f"{trajectory['dx_mm'].abs().max():.3f} mm"
)

print(
    f"Maximum Y displacement: "
    f"{trajectory['dy_mm'].abs().max():.3f} mm"
)

print(
    f"Maximum Z displacement: "
    f"{trajectory['dz_mm'].abs().max():.3f} mm"
)

print(
    f"\nMaximum 3D displacement "
    f"from frame 0: "
    f"{trajectory['displacement_3d_mm'].max():.3f} mm"
)


# ============================================================
# TUMOR VOLUME PROXY
# ============================================================

print("\n===== TUMOR MASK SIZE =====")

print(
    f"Minimum tumor voxels: "
    f"{trajectory['tumor_voxels'].min()}"
)

print(
    f"Maximum tumor voxels: "
    f"{trajectory['tumor_voxels'].max()}"
)

print(
    f"Mean tumor voxels: "
    f"{trajectory['tumor_voxels'].mean():.2f}"
)


# ============================================================
# OUTPUT
# ============================================================

print("\n")
print("=" * 60)

print("CSV SAVED SUCCESSFULLY:")

print(OUTPUT_FILE)

print("=" * 60)