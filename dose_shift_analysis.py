import pydicom
import numpy as np

# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

sfile = r"C:\Users\smk28\Downloads\Center2\Center 2\structureSetFiles\RTSTRUCT_V15_QA_FH_20240708\9999.144874522247533734992787435849309847299"

dfile = r"C:\Users\smk28\Downloads\Center2\Center 2\doseFiles\RTDOSE_EclipseDosen_20240708\9999.74820833721338163423465608981710622482"

# --------------------------------------------------
# READ DICOM
# --------------------------------------------------

S = pydicom.dcmread(sfile)
D = pydicom.dcmread(dfile)

dose = D.pixel_array.astype(float) * float(D.DoseGridScaling)

print("Dose array shape:", dose.shape)
print("Dose maximum:", dose.max(), "Gy")

# --------------------------------------------------
# CTV_1 ROI
# --------------------------------------------------

roi = next(
    x for x in S.ROIContourSequence
    if x.ReferencedROINumber == 3
)

origin = np.array(D.ImagePositionPatient, dtype=float)

sx = float(D.PixelSpacing[0])
sy = float(D.PixelSpacing[1])

# --------------------------------------------------
# POLYGON RASTERIZATION
# --------------------------------------------------

def polygon_inside(rows, cols, shape):

    rmin = max(0, int(np.floor(rows.min())))
    rmax = min(shape[0] - 1, int(np.ceil(rows.max())))

    cmin = max(0, int(np.floor(cols.min())))
    cmax = min(shape[1] - 1, int(np.ceil(cols.max())))

    rr, cc = np.meshgrid(
        np.arange(rmin, rmax + 1),
        np.arange(cmin, cmax + 1),
        indexing="ij"
    )

    inside = np.zeros(rr.shape, dtype=bool)

    j = len(rows) - 1

    for i in range(len(rows)):

        condition = (
            ((rows[i] > rr) != (rows[j] > rr))
            &
            (
                cc
                <
                (cols[j] - cols[i])
                * (rr - rows[i])
                / (rows[j] - rows[i] + 1e-12)
                + cols[i]
            )
        )

        inside ^= condition

        j = i

    return rr, cc, inside


# --------------------------------------------------
# CREATE CTV MASK
# --------------------------------------------------

mask = np.zeros(dose.shape, dtype=bool)

for contour in roi.ContourSequence:

    points = np.array(
        contour.ContourData,
        dtype=float
    ).reshape(-1, 3)

    z = points[0, 2]

    z_index = int(round(z - origin[2]))

    if z_index < 0 or z_index >= dose.shape[0]:
        continue

    cols = (points[:, 0] - origin[0]) / sx
    rows = (points[:, 1] - origin[1]) / sy

    rr, cc, inside = polygon_inside(
        rows,
        cols,
        dose.shape[1:]
    )

    mask[z_index, rr, cc] |= inside


# --------------------------------------------------
# DOSE STATISTICS
# --------------------------------------------------

def dose_stats(test_mask):

    values = dose[test_mask]

    if len(values) == 0:
        return None

    return {
        "voxels": len(values),
        "mean": values.mean(),
        "D95": np.percentile(values, 5),
        "max": values.max()
    }


# --------------------------------------------------
# BASELINE
# --------------------------------------------------

baseline = dose_stats(mask)

print()
print("========================================")
print("BASELINE CTV_1")
print("========================================")

print("CTV voxels :", baseline["voxels"])
print("Dmean      :", round(baseline["mean"], 4), "Gy")
print("D95        :", round(baseline["D95"], 4), "Gy")
print("Dmax       :", round(baseline["max"], 4), "Gy")


# --------------------------------------------------
# SHIFT FUNCTION
# --------------------------------------------------

def shift_mask(mask, dx_mm, dy_mm, dz_mm):

    dx = int(round(dx_mm / sx))
    dy = int(round(dy_mm / sy))
    dz = int(round(dz_mm))

    shifted = np.zeros_like(mask)

    z_src_start = max(0, -dz)
    z_src_end = min(mask.shape[0], mask.shape[0] - dz)

    y_src_start = max(0, -dy)
    y_src_end = min(mask.shape[1], mask.shape[1] - dy)

    x_src_start = max(0, -dx)
    x_src_end = min(mask.shape[2], mask.shape[2] - dx)

    z_dst_start = max(0, dz)
    z_dst_end = min(mask.shape[0], mask.shape[0] + dz)

    y_dst_start = max(0, dy)
    y_dst_end = min(mask.shape[1], mask.shape[1] + dy)

    x_dst_start = max(0, dx)
    x_dst_end = min(mask.shape[2], mask.shape[2] + dx)

    if (
        z_src_end > z_src_start
        and y_src_end > y_src_start
        and x_src_end > x_src_start
    ):

        shifted[
            z_dst_start:z_dst_end,
            y_dst_start:y_dst_end,
            x_dst_start:x_dst_end
        ] = mask[
            z_src_start:z_src_end,
            y_src_start:y_src_end,
            x_src_start:x_src_end
        ]

    return shifted


# --------------------------------------------------
# EXPERIMENT
# --------------------------------------------------

shifts = [0, 1, 2, 3, 3.68, 4]

print()
print("========================================")
print("POSITIVE X SHIFT")
print("========================================")

print("Shift(mm)   Dmean(Gy)   D95(Gy)   Dmax(Gy)")

for s in shifts:

    shifted = shift_mask(mask, s, 0, 0)

    stats = dose_stats(shifted)

    print(
        f"{s:8.2f}   "
        f"{stats['mean']:10.4f}   "
        f"{stats['D95']:8.4f}   "
        f"{stats['max']:8.4f}"
    )


print()
print("========================================")
print("POSITIVE Y SHIFT")
print("========================================")

print("Shift(mm)   Dmean(Gy)   D95(Gy)   Dmax(Gy)")

for s in shifts:

    shifted = shift_mask(mask, 0, s, 0)

    stats = dose_stats(shifted)

    print(
        f"{s:8.2f}   "
        f"{stats['mean']:10.4f}   "
        f"{stats['D95']:8.4f}   "
        f"{stats['max']:8.4f}"
    )


print()
print("========================================")
print("POSITIVE Z SHIFT")
print("========================================")

print("Shift(mm)   Dmean(Gy)   D95(Gy)   Dmax(Gy)")

for s in shifts:

    shifted = shift_mask(mask, 0, 0, s)

    stats = dose_stats(shifted)

    print(
        f"{s:8.2f}   "
        f"{stats['mean']:10.4f}   "
        f"{stats['D95']:8.4f}   "
        f"{stats['max']:8.4f}"
    )


print()
print("========================================")
print("EXPERIMENT COMPLETE")
print("========================================")