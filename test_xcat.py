from pathlib import Path

DATASET_DIR = Path(r"C:\Users\smk28\Downloads\DATASET1\data")

print("Dataset exists:", DATASET_DIR.exists())

if DATASET_DIR.exists():
    print("\nContents:")
    for item in DATASET_DIR.iterdir():
        print(" -", item.name)

    volumes = list((DATASET_DIR / "volumes").glob("*.nii.gz"))
    masks = list((DATASET_DIR / "tumor_masks").glob("*.nii.gz"))

    print("\nNumber of volume files:", len(volumes))
    print("Number of tumor-mask files:", len(masks))

    print("\nFirst 5 volumes:")
    for f in volumes[:5]:
        print(" -", f.name)

    print("\nFirst 5 masks:")
    for f in masks[:5]:
        print(" -", f.name)
else:
    print("\n❌ Python cannot find the dataset.")
