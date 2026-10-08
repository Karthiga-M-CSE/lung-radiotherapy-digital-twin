from pathlib import Path
import nibabel as nib
import numpy as np

DATASET_DIR = Path(r"C:\Users\gayu0\OneDrive\Desktop\projects\DATASET1")

volume_path = DATASET_DIR / "volumes" / "volume_0.nii.gz"
mask_path = DATASET_DIR / "tumor_masks" / "tumormask_0.nii.gz"

print("Loading volume...")
volume_img = nib.load(str(volume_path))

print("Loading tumor mask...")
mask_img = nib.load(str(mask_path))

print("\n===== VOLUME =====")
print("Shape:", volume_img.shape)
print("Data type:", volume_img.get_data_dtype())
print("Affine:")
print(volume_img.affine)

print("\n===== TUMOR MASK =====")
print("Shape:", mask_img.shape)
print("Data type:", mask_img.get_data_dtype())
print("Affine:")
print(mask_img.affine)

volume = volume_img.get_fdata()
mask = mask_img.get_fdata()

print("\n===== DATA CHECK =====")
print("Volume minimum:", np.min(volume))
print("Volume maximum:", np.max(volume))

print("Mask minimum:", np.min(mask))
print("Mask maximum:", np.max(mask))

print("Tumor voxels:", np.sum(mask > 0))

print("\n===== ALIGNMENT CHECK =====")
print("Same shape:", volume.shape == mask.shape)
print(
    "Same affine:",
    np.allclose(volume_img.affine, mask_img.affine)
)
