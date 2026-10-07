from pathlib import Path
import nibabel as nib
import numpy as np

# XCAT dataset
DATASET_DIR = Path(r"C:\Users\smk28\Downloads\DATASET1\data")

# Load tumor mask
mask_path = DATASET_DIR / "tumor_masks" / "tumormask_0.nii.gz"
mask_img = nib.load(str(mask_path))

mask = mask_img.get_fdata()

# Find tumor voxels
tumor_voxels = np.argwhere(mask > 0)

print("Number of tumor voxels:", len(tumor_voxels))

# Calculate centroid in voxel coordinates
centroid_voxel = tumor_voxels.mean(axis=0)

print("\nCentroid in voxel coordinates:")
print("X:", centroid_voxel[0])
print("Y:", centroid_voxel[1])
print("Z:", centroid_voxel[2])

# Convert voxel coordinates → physical coordinates (mm)
centroid_mm = nib.affines.apply_affine(
    mask_img.affine,
    centroid_voxel
)

print("\nCentroid in physical coordinates (mm):")
print("X:", centroid_mm[0])
print("Y:", centroid_mm[1])
print("Z:", centroid_mm[2])