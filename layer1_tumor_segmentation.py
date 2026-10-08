import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib

DATA_DIR = r"C:\Users\gayu0\OneDrive\Desktop\projects\DATASET1"

VOLUME_DIR = os.path.join(DATA_DIR, "volumes")
MASK_DIR = os.path.join(DATA_DIR, "tumor_masks")

OUTPUT_FILE = "tumor_segmentation_results.csv"

volume_files = sorted(
    glob.glob(os.path.join(VOLUME_DIR, "volume_*.nii.gz")),
    key=lambda x: int(
        os.path.basename(x).split("_")[1].split(".")[0]
    )
)

mask_files = sorted(
    glob.glob(os.path.join(MASK_DIR, "tumormask_*.nii.gz")),
    key=lambda x: int(
        os.path.basename(x).split("_")[1].split(".")[0]
    )
)

print("==========================================")
print("      LAYER 1 - TUMOR SEGMENTATION")
print("==========================================")

print(f"Volumes found : {len(volume_files)}")
print(f"Masks found   : {len(mask_files)}")

if len(volume_files) != len(mask_files):
    raise ValueError("Number of volumes and masks do not match.")

results = []

for volume_file, mask_file in zip(volume_files, mask_files):

    volume_id = int(
        os.path.basename(volume_file)
        .split("_")[1]
        .split(".")[0]
    )

    volume_img = nib.load(volume_file)
    volume = volume_img.get_fdata()

    mask_img = nib.load(mask_file)
    mask = mask_img.get_fdata()

    mask_binary = mask > 0

    tumor_indices = np.argwhere(mask_binary)

    voxel_count = len(tumor_indices)

    if voxel_count == 0:
        print(f"Warning: frame {volume_id} has empty mask.")
        continue

    voxel_dimensions = mask_img.header.get_zooms()[:3]

    voxel_volume_mm3 = np.prod(voxel_dimensions)

    tumor_volume_mm3 = voxel_count * voxel_volume_mm3

    tumor_volume_cm3 = tumor_volume_mm3 / 1000.0

    centroid_voxel = tumor_indices.mean(axis=0)

    centroid_world = nib.affines.apply_affine(
        mask_img.affine,
        centroid_voxel
    )

    x_mm = centroid_world[0]
    y_mm = centroid_world[1]
    z_mm = centroid_world[2]

    min_coords = tumor_indices.min(axis=0)
    max_coords = tumor_indices.max(axis=0)

    bbox_voxels = max_coords - min_coords + 1

    bbox_x_mm = bbox_voxels[0] * voxel_dimensions[0]
    bbox_y_mm = bbox_voxels[1] * voxel_dimensions[1]
    bbox_z_mm = bbox_voxels[2] * voxel_dimensions[2]

    results.append({
        "frame": volume_id,

        "tumor_voxels": voxel_count,

        "tumor_volume_mm3": tumor_volume_mm3,
        "tumor_volume_cm3": tumor_volume_cm3,

        "centroid_x_mm": x_mm,
        "centroid_y_mm": y_mm,
        "centroid_z_mm": z_mm,

        "bbox_x_mm": bbox_x_mm,
        "bbox_y_mm": bbox_y_mm,
        "bbox_z_mm": bbox_z_mm
    })

df = pd.DataFrame(results)

df.to_csv(OUTPUT_FILE, index=False)

print("\n==========================================")
print("LAYER 1 COMPLETE")
print("==========================================")

print(f"Frames processed : {len(df)}")

print(
    f"Tumor volume mean : "
    f"{df['tumor_volume_cm3'].mean():.4f} cm³"
)

print(
    f"Tumor volume min  : "
    f"{df['tumor_volume_cm3'].min():.4f} cm³"
)

print(
    f"Tumor volume max  : "
    f"{df['tumor_volume_cm3'].max():.4f} cm³"
)

print("\nCentroid range:")

print(
    f"X: {df['centroid_x_mm'].min():.3f} "
    f"to {df['centroid_x_mm'].max():.3f} mm"
)

print(
    f"Y: {df['centroid_y_mm'].min():.3f} "
    f"to {df['centroid_y_mm'].max():.3f} mm"
)

print(
    f"Z: {df['centroid_z_mm'].min():.3f} "
    f"to {df['centroid_z_mm'].max():.3f} mm"
)

print("\nOutput:")
print(OUTPUT_FILE)

print("==========================================")
