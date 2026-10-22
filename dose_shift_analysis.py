"""Motion-conditioned dose assessment for DICOM RTDOSE and RTSTRUCT data."""

import argparse
import csv
from pathlib import Path

import numpy as np
import pydicom
from scipy.ndimage import map_coordinates


def _dose_geometry(dose_dataset):
    dose = dose_dataset.pixel_array.astype(float)
    dose *= float(dose_dataset.DoseGridScaling)
    if dose.ndim == 2:
        dose = dose[np.newaxis, ...]
    if dose.ndim != 3:
        raise ValueError(f"Expected a 3D RTDOSE grid, got array shape {dose.shape}.")
    if str(getattr(dose_dataset, "DoseUnits", "GY")).upper() != "GY":
        raise ValueError("RTDOSE must use Gy units for the reported metrics.")

    origin = np.asarray(dose_dataset.ImagePositionPatient, dtype=float)
    orientation = np.asarray(dose_dataset.ImageOrientationPatient, dtype=float)
    if origin.shape != (3,) or orientation.shape != (6,):
        raise ValueError("RTDOSE patient position/orientation metadata is malformed.")
    row_direction = orientation[:3]
    column_direction = orientation[3:]
    row_norm = np.linalg.norm(row_direction)
    column_norm = np.linalg.norm(column_direction)
    if (
        not np.all(np.isfinite(orientation))
        or row_norm == 0
        or column_norm == 0
    ):
        raise ValueError("RTDOSE ImageOrientationPatient must contain valid directions.")
    row_direction = row_direction / row_norm
    column_direction = column_direction / column_norm
    if not np.isclose(np.dot(row_direction, column_direction), 0.0, atol=1e-4):
        raise ValueError("RTDOSE image direction vectors must be perpendicular.")
    normal = np.cross(row_direction, column_direction)
    normal /= np.linalg.norm(normal)
    spacing = np.asarray(dose_dataset.PixelSpacing, dtype=float)
    offsets = np.asarray(
        getattr(dose_dataset, "GridFrameOffsetVector", [0.0]), dtype=float
    )
    if offsets.size != dose.shape[0]:
        raise ValueError(
            "GridFrameOffsetVector length does not match the RTDOSE frame count."
        )
    if not np.all(np.isfinite(offsets)):
        raise ValueError("RTDOSE frame offsets must be finite.")
    if offsets.size > 1 and np.any(np.diff(offsets) == 0):
        raise ValueError("RTDOSE frame offsets must be distinct.")
    if spacing.size != 2 or np.any(spacing <= 0):
        raise ValueError("RTDOSE PixelSpacing must contain two positive values.")
    single_frame_spacing = abs(
        float(
            getattr(
                dose_dataset,
                "SpacingBetweenSlices",
                getattr(dose_dataset, "SliceThickness", 1.0),
            )
        )
    )
    if not np.isfinite(single_frame_spacing) or single_frame_spacing <= 0:
        raise ValueError("RTDOSE single-frame slice spacing must be positive.")

    return {
        "dose": dose,
        "origin": origin,
        "row_direction": row_direction,
        "column_direction": column_direction,
        "normal": normal,
        "spacing": spacing,
        "offsets": offsets,
        "single_frame_spacing": single_frame_spacing,
    }


def patient_to_grid(points, geometry):
    """Convert DICOM patient coordinates to fractional (frame, row, column)."""
    relative = np.asarray(points, dtype=float) - geometry["origin"]
    column_mm = relative @ geometry["row_direction"]
    row_mm = relative @ geometry["column_direction"]
    frame_mm = relative @ geometry["normal"]

    offsets = geometry["offsets"]
    sorted_indices = np.argsort(offsets)
    sorted_offsets = offsets[sorted_indices]
    frame_indices = np.interp(frame_mm, sorted_offsets, sorted_indices)
    outside_frames = (frame_mm < sorted_offsets[0]) | (frame_mm > sorted_offsets[-1])
    frame_indices = np.where(outside_frames, -1.0, frame_indices)
    return np.stack(
        (
            frame_indices,
            row_mm / geometry["spacing"][0],
            column_mm / geometry["spacing"][1],
        ),
        axis=0,
    )


def sample_dose(dose, coordinates):
    """Trilinearly sample dose at fractional frame/row/column coordinates."""
    return map_coordinates(
        dose,
        coordinates,
        order=1,
        mode="constant",
        cval=np.nan,
        prefilter=False,
    )


def polygon_inside(rows, columns, shape):
    """Return integer grid coordinates and a mask for a polygon."""
    row_min = max(0, int(np.floor(rows.min())))
    row_max = min(shape[0] - 1, int(np.ceil(rows.max())))
    column_min = max(0, int(np.floor(columns.min())))
    column_max = min(shape[1] - 1, int(np.ceil(columns.max())))
    if row_max < row_min or column_max < column_min:
        return None

    rr, cc = np.meshgrid(
        np.arange(row_min, row_max + 1),
        np.arange(column_min, column_max + 1),
        indexing="ij",
    )
    inside = np.zeros(rr.shape, dtype=bool)
    previous = len(rows) - 1
    for current in range(len(rows)):
        crosses = (rows[current] > rr) != (rows[previous] > rr)
        crossing_column = (
            (columns[previous] - columns[current])
            * (rr - rows[current])
            / (rows[previous] - rows[current] + 1e-12)
            + columns[current]
        )
        inside ^= crosses & (cc < crossing_column)
        previous = current
    return rr, cc, inside


def build_roi_masks(structure_dataset, geometry):
    """Rasterize each RTSTRUCT contour onto the RTDOSE grid."""
    names = {
        int(item.ROINumber): str(item.ROIName)
        for item in structure_dataset.StructureSetROISequence
    }
    masks = {}
    for roi_contour in getattr(structure_dataset, "ROIContourSequence", []):
        roi_number = int(roi_contour.ReferencedROINumber)
        roi_name = names.get(roi_number, f"ROI_{roi_number}")
        mask = np.zeros(geometry["dose"].shape, dtype=bool)
        for contour in getattr(roi_contour, "ContourSequence", []):
            points = np.asarray(contour.ContourData, dtype=float).reshape(-1, 3)
            if len(points) < 3:
                continue
            grid = patient_to_grid(points, geometry)
            frame = int(np.rint(np.mean(grid[0])))
            if frame < 0 or frame >= mask.shape[0]:
                continue
            polygon = polygon_inside(grid[1], grid[2], mask.shape[1:])
            if polygon is None:
                continue
            rr, cc, inside = polygon
            mask[frame, rr, cc] |= inside
        if np.any(mask):
            masks[roi_name] = mask
    if not masks:
        raise ValueError("No RTSTRUCT contours overlap the RTDOSE grid.")
    return masks


def shifted_grid_coordinates(geometry, displacement_mm):
    """Map each dose-grid point, shifted in patient space, to dose indices."""
    dose = geometry["dose"]
    _, row_count, column_count = dose.shape
    offsets = geometry["offsets"]
    displacement_mm = np.asarray(displacement_mm, dtype=float)
    row_shift = np.dot(displacement_mm, geometry["column_direction"]) / geometry[
        "spacing"
    ][0]
    column_shift = np.dot(displacement_mm, geometry["row_direction"]) / geometry[
        "spacing"
    ][1]
    frame_mm = offsets + np.dot(displacement_mm, geometry["normal"])
    sorted_indices = np.argsort(offsets)
    sorted_offsets = offsets[sorted_indices]
    frame_indices = np.interp(frame_mm, sorted_offsets, sorted_indices)
    outside_frames = (frame_mm < sorted_offsets[0]) | (frame_mm > sorted_offsets[-1])
    frame_indices = np.where(outside_frames, -1.0, frame_indices)

    coordinates = np.empty((3,) + dose.shape, dtype=float)
    coordinates[0] = np.broadcast_to(
        frame_indices[:, np.newaxis, np.newaxis], dose.shape
    )
    coordinates[1] = np.arange(row_count)[np.newaxis, :, np.newaxis] + row_shift
    coordinates[2] = np.arange(column_count)[np.newaxis, np.newaxis, :] + column_shift
    return coordinates


def dose_metrics(values, voxel_volume_cc):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None

    descending = np.sort(values)[::-1]

    def dose_to_volume(cc):
        if values.size * voxel_volume_cc < cc:
            return None
        voxel_count = max(1, int(np.ceil(cc / voxel_volume_cc)))
        return float(descending[min(voxel_count, len(descending)) - 1])

    return {
        "sampled_voxels": int(values.size),
        "D95_Gy": float(np.percentile(values, 5)),
        "D98_Gy": float(np.percentile(values, 2)),
        "Dmean_Gy": float(np.mean(values)),
        "Dmax_Gy": float(np.max(values)),
        "D0.1cc_Gy": dose_to_volume(0.1),
        "D2cc_Gy": dose_to_volume(2.0),
    }


def _voxel_volume_cc(geometry):
    if len(geometry["offsets"]) > 1:
        frame_spacing = float(np.median(np.abs(np.diff(geometry["offsets"]))))
    else:
        frame_spacing = geometry["single_frame_spacing"]
    return (
        geometry["spacing"][0] * geometry["spacing"][1] * frame_spacing / 1000.0
    )


def _dvh_rows(roi_name, condition, values, bin_width):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return []
    maximum = float(values.max())
    doses = np.arange(int(np.ceil(maximum / bin_width)) + 2) * bin_width
    return [
        {
            "roi_name": roi_name,
            "condition": condition,
            "dose_Gy": float(dose),
            "volume_percent": float(100.0 * np.count_nonzero(values >= dose) / values.size),
        }
        for dose in doses
    ]


def _write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_assessment(
    rtstruct_path,
    rtdose_path,
    output_dir,
    displacement_mm=(0.0, 0.0, 0.0),
    target_roi=None,
    dvh_bin_width=0.1,
):
    displacement_mm = np.asarray(displacement_mm, dtype=float)
    if displacement_mm.shape != (3,) or not np.all(np.isfinite(displacement_mm)):
        raise ValueError("Displacement must contain three finite values in mm.")
    if dvh_bin_width <= 0:
        raise ValueError("DVH bin width must be positive.")

    dose_dataset = pydicom.dcmread(str(rtdose_path))
    structure_dataset = pydicom.dcmread(str(rtstruct_path))
    geometry = _dose_geometry(dose_dataset)
    masks = build_roi_masks(structure_dataset, geometry)

    target_names = [name for name in masks if "CTV" in name.upper()]
    if target_roi:
        if target_roi not in masks:
            raise ValueError(f"Target ROI {target_roi!r} was not found among contours.")
        target_names = [target_roi]
    elif not target_names:
        raise ValueError(
            "No CTV ROI was found; specify --target-roi with an exact contour name."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    voxel_volume_cc = _voxel_volume_cc(geometry)
    shifted_coordinates = shifted_grid_coordinates(geometry, displacement_mm)
    shifted_dose = sample_dose(geometry["dose"], shifted_coordinates)
    difference = shifted_dose - geometry["dose"]
    difference_path = output_dir / "motion_conditioned_dose_difference.npz"
    np.savez_compressed(
        difference_path,
        difference_Gy=difference,
        ImagePositionPatient=geometry["origin"],
        ImageOrientationPatient=np.concatenate(
            (geometry["row_direction"], geometry["column_direction"])
        ),
        PixelSpacing=geometry["spacing"],
        GridFrameOffsetVector=geometry["offsets"],
        displacement_mm=displacement_mm,
        difference_definition="Dose(p + displacement) - Dose(p), in Gy",
    )

    metric_rows = []
    dvh_rows = []
    roi_numbers = {
        str(item.ROIName): int(item.ROINumber)
        for item in structure_dataset.StructureSetROISequence
    }
    for roi_name, mask in masks.items():
        for condition, sampled_dose in (
            ("reference", geometry["dose"]),
            ("motion-conditioned", shifted_dose),
        ):
            values = sampled_dose[mask]
            metrics = dose_metrics(values, voxel_volume_cc)
            if metrics is None:
                continue
            metric_rows.append(
                {
                    "roi_number": roi_numbers.get(roi_name, ""),
                    "roi_name": roi_name,
                    "role": "CTV" if roi_name in target_names else "OAR/other",
                    "condition": condition,
                    "displacement_x_mm": float(displacement_mm[0])
                    if condition == "motion-conditioned"
                    else 0.0,
                    "displacement_y_mm": float(displacement_mm[1])
                    if condition == "motion-conditioned"
                    else 0.0,
                    "displacement_z_mm": float(displacement_mm[2])
                    if condition == "motion-conditioned"
                    else 0.0,
                    "contour_voxels": int(np.count_nonzero(mask)),
                    "coverage_percent": float(
                        100.0 * metrics["sampled_voxels"] / np.count_nonzero(mask)
                    ),
                    **metrics,
                }
            )
            dvh_rows.extend(_dvh_rows(roi_name, condition, values, dvh_bin_width))

    metric_fields = [
        "roi_number",
        "roi_name",
        "role",
        "condition",
        "displacement_x_mm",
        "displacement_y_mm",
        "displacement_z_mm",
        "contour_voxels",
        "sampled_voxels",
        "coverage_percent",
        "D95_Gy",
        "D98_Gy",
        "Dmean_Gy",
        "Dmax_Gy",
        "D0.1cc_Gy",
        "D2cc_Gy",
    ]
    _write_csv(output_dir / "motion_conditioned_dose_metrics.csv", metric_rows, metric_fields)
    _write_csv(
        output_dir / "motion_conditioned_dvhs.csv",
        dvh_rows,
        ["roi_name", "condition", "dose_Gy", "volume_percent"],
    )
    return difference_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Research-only motion-conditioned dose assessment using RTDOSE and RTSTRUCT."
        )
    )
    parser.add_argument("--rtstruct", required=True, help="Path to the RTSTRUCT DICOM.")
    parser.add_argument("--rtdose", required=True, help="Path to the RTDOSE DICOM.")
    parser.add_argument(
        "--output-dir", default="dose_assessment", help="Directory for CSV and NPZ outputs."
    )
    parser.add_argument(
        "--displacement-mm",
        nargs=3,
        type=float,
        metavar=("DX", "DY", "DZ"),
        default=(0.0, 0.0, 0.0),
        help="Motion displacement in DICOM patient x/y/z axes; fractional mm is supported.",
    )
    parser.add_argument(
        "--target-roi",
        help="Exact target contour name; when omitted, all CTV-named contours are targets.",
    )
    parser.add_argument(
        "--dvh-bin-width",
        type=float,
        default=0.1,
        help="Dose histogram bin width in Gy (default: 0.1).",
    )
    args = parser.parse_args()

    try:
        difference_path = run_assessment(
            args.rtstruct,
            args.rtdose,
            args.output_dir,
            args.displacement_mm,
            args.target_roi,
            args.dvh_bin_width,
        )
    except (
        OSError,
        ValueError,
        AttributeError,
        KeyError,
        pydicom.errors.InvalidDicomError,
    ) as error:
        parser.error(str(error))

    print("MOTION-CONDITIONED DOSE ASSESSMENT (research only; not for clinical use)")
    print(f"Results written to: {Path(args.output_dir).resolve()}")
    print(f"Dose-difference map: {difference_path.resolve()}")


if __name__ == "__main__":
    main()
