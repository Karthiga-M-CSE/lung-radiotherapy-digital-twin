"""Geometry validation for 4D-CT, RTDOSE, and RTSTRUCT DICOM data.

Validates spatial consistency (spacing, origin, direction cosines, orientation,
grid dimensions) across imaging modalities to ensure physical-coordinate
alignment before dosimetric or motion analysis.

Uses SimpleITK for image geometry extraction and PyDICOM for DICOM-specific
metadata (RTSTRUCT contour coordinates, RTDOSE grid scaling).
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


class Severity(Enum):
    """Validation issue severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class ValidationIssue:
    """A single geometry validation finding."""
    check: str
    severity: Severity
    message: str
    details: Optional[Dict] = None


@dataclass
class ImageGeometry:
    """Extracted spatial geometry from a medical image."""
    source: str
    origin: np.ndarray
    spacing: np.ndarray
    direction: np.ndarray
    size: Tuple[int, ...]
    dimension: int
    modality: str = ""

    @property
    def direction_matrix(self) -> np.ndarray:
        n = self.dimension if self.dimension <= 3 else 3
        return self.direction[:n * n].reshape(n, n)

    @property
    def physical_extent(self) -> np.ndarray:
        n = min(self.dimension, 3)
        return np.array([self.size[i] * self.spacing[i] for i in range(n)])


@dataclass
class GeometryReport:
    """Aggregated geometry validation report."""
    issues: List[ValidationIssue] = field(default_factory=list)
    images_validated: List[str] = field(default_factory=list)
    passed: bool = True

    def add(self, issue: ValidationIssue) -> None:
        self.issues.append(issue)
        if issue.severity in (Severity.ERROR, Severity.CRITICAL):
            self.passed = False

    @property
    def error_count(self) -> int:
        return sum(
            1 for i in self.issues
            if i.severity in (Severity.ERROR, Severity.CRITICAL)
        )

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.WARNING)

    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        lines = [
            f"Geometry Validation: {status}",
            f"  Images validated : {len(self.images_validated)}",
            f"  Errors           : {self.error_count}",
            f"  Warnings         : {self.warning_count}",
            f"  Total issues     : {len(self.issues)}",
        ]
        return "\n".join(lines)



def extract_geometry_sitk(image_path: Union[str, Path]) -> ImageGeometry:
    """Extract spatial geometry from a SimpleITK-readable image file.

    Supports NIfTI (.nii, .nii.gz), NRRD, MetaImage (.mha/.mhd), and any
    format SimpleITK can read.

    Args:
        image_path: Path to the image file.

    Returns:
        ImageGeometry with origin, spacing, direction, size, and dimension.

    Raises:
        FileNotFoundError: If the image file does not exist.
        RuntimeError: If SimpleITK cannot read the file.
    """
    import SimpleITK as sitk

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    img = sitk.ReadImage(str(path))
    dim = img.GetDimension()

    return ImageGeometry(
        source=str(path),
        origin=np.array(img.GetOrigin(), dtype=float),
        spacing=np.array(img.GetSpacing(), dtype=float),
        direction=np.array(img.GetDirection(), dtype=float),
        size=img.GetSize(),
        dimension=dim,
    )


def extract_geometry_dicom(dcm_path: Union[str, Path], modality: str = "") -> ImageGeometry:
    """Extract spatial geometry from a DICOM file using PyDICOM.

    Handles RTDOSE (3D dose grid) and CT/MR image slices. For RTSTRUCT,
    use `extract_rtstruct_contour_bounds` instead.

    Args:
        dcm_path: Path to the DICOM file.
        modality: Override modality string (auto-detected if empty).

    Returns:
        ImageGeometry populated from DICOM header fields.

    Raises:
        FileNotFoundError: If the file does not exist.
        AttributeError: If required DICOM tags are missing.
    """
    import pydicom

    path = Path(dcm_path)
    if not path.exists():
        raise FileNotFoundError(f"DICOM file not found: {path}")

    ds = pydicom.dcmread(str(path))
    detected_modality = modality or getattr(ds, "Modality", "UNKNOWN")

    origin = np.array(getattr(ds, "ImagePositionPatient", [0.0, 0.0, 0.0]), dtype=float)

    pixel_spacing = getattr(ds, "PixelSpacing", None)
    grid_frame_offset = getattr(ds, "GridFrameOffsetVector", None)

    if pixel_spacing is not None:
        sx, sy = float(pixel_spacing[0]), float(pixel_spacing[1])
    else:
        sx, sy = 1.0, 1.0
        logger.warning(f"PixelSpacing missing in {path.name}, defaulting to (1.0, 1.0)")

    if grid_frame_offset is not None and len(grid_frame_offset) > 1:
        sz = abs(float(grid_frame_offset[1]) - float(grid_frame_offset[0]))
    elif hasattr(ds, "SliceThickness"):
        sz = float(ds.SliceThickness)
    else:
        sz = 1.0
        logger.warning(f"Slice spacing unavailable in {path.name}, defaulting to 1.0")

    spacing = np.array([sx, sy, sz], dtype=float)

    iop = getattr(ds, "ImageOrientationPatient", None)
    if iop is not None:
        iop = np.array(iop, dtype=float)
        row_dir = iop[:3]
        col_dir = iop[3:6]
        slice_dir = np.cross(row_dir, col_dir)
        direction = np.eye(3)
        direction[:, 0] = row_dir
        direction[:, 1] = col_dir
        direction[:, 2] = slice_dir
    else:
        direction = np.eye(3)

    rows = int(getattr(ds, "Rows", 0))
    cols = int(getattr(ds, "Columns", 0))
    n_frames = int(getattr(ds, "NumberOfFrames", 1))

    if detected_modality == "RTDOSE" and hasattr(ds, "pixel_array"):
        arr = ds.pixel_array
        if arr.ndim == 3:
            n_frames = arr.shape[0]

    size = (cols, rows, n_frames)

    return ImageGeometry(
        source=str(path),
        origin=origin,
        spacing=spacing,
        direction=direction.flatten(),
        size=size,
        dimension=3,
        modality=detected_modality,
    )


def extract_rtstruct_contour_bounds(
    rtstruct_path: Union[str, Path],
    roi_number: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """Extract physical-coordinate bounding box from RTSTRUCT contours.

    Args:
        rtstruct_path: Path to the RTSTRUCT DICOM file.
        roi_number: Specific ROI number to extract. If None, uses all ROIs.

    Returns:
        Dictionary with keys 'min_bound', 'max_bound', 'center',
        and 'all_points' (Nx3 array of all contour points).

    Raises:
        FileNotFoundError: If the RTSTRUCT file does not exist.
        ValueError: If no contour data is found.
    """
    import pydicom

    path = Path(rtstruct_path)
    if not path.exists():
        raise FileNotFoundError(f"RTSTRUCT file not found: {path}")

    ds = pydicom.dcmread(str(path))
    all_points = []

    if not hasattr(ds, "ROIContourSequence"):
        raise ValueError(f"No ROIContourSequence found in {path.name}")

    for roi_contour in ds.ROIContourSequence:
        if roi_number is not None:
            if int(roi_contour.ReferencedROINumber) != roi_number:
                continue

        if not hasattr(roi_contour, "ContourSequence"):
            continue

        for contour in roi_contour.ContourSequence:
            pts = np.array(contour.ContourData, dtype=float).reshape(-1, 3)
            all_points.append(pts)

    if not all_points:
        raise ValueError(
            f"No contour points found in {path.name}"
            + (f" for ROI #{roi_number}" if roi_number else "")
        )

    all_pts = np.vstack(all_points)

    return {
        "min_bound": all_pts.min(axis=0),
        "max_bound": all_pts.max(axis=0),
        "center": all_pts.mean(axis=0),
        "all_points": all_pts,
        "num_contours": len(all_points),
        "num_points": len(all_pts),
    }



def _check_spacing(
    geom: ImageGeometry,
    report: GeometryReport,
    min_spacing_mm: float = 0.1,
    max_spacing_mm: float = 20.0,
) -> None:
    """Validate that voxel spacing falls within clinically plausible bounds."""
    for axis, label in enumerate(["X", "Y", "Z"]):
        if axis >= len(geom.spacing):
            break
        s = geom.spacing[axis]
        if s <= 0:
            report.add(ValidationIssue(
                check="spacing_positive",
                severity=Severity.CRITICAL,
                message=f"{geom.source}: {label}-spacing is non-positive ({s:.6f} mm)",
                details={"axis": label, "value": float(s)},
            ))
        elif s < min_spacing_mm:
            report.add(ValidationIssue(
                check="spacing_range",
                severity=Severity.WARNING,
                message=f"{geom.source}: {label}-spacing unusually small ({s:.6f} mm)",
                details={"axis": label, "value": float(s), "min": min_spacing_mm},
            ))
        elif s > max_spacing_mm:
            report.add(ValidationIssue(
                check="spacing_range",
                severity=Severity.WARNING,
                message=f"{geom.source}: {label}-spacing unusually large ({s:.6f} mm)",
                details={"axis": label, "value": float(s), "max": max_spacing_mm},
            ))


def _check_direction_orthogonality(
    geom: ImageGeometry,
    report: GeometryReport,
    tolerance: float = 1e-4,
) -> None:
    """Validate that direction cosine matrix is orthogonal (proper rotation)."""
    mat = geom.direction_matrix
    product = mat.T @ mat
    identity = np.eye(mat.shape[0])
    deviation = np.max(np.abs(product - identity))

    if deviation > tolerance:
        report.add(ValidationIssue(
            check="direction_orthogonality",
            severity=Severity.ERROR,
            message=(
                f"{geom.source}: Direction cosines are not orthogonal "
                f"(max deviation from identity: {deviation:.6f})"
            ),
            details={"max_deviation": float(deviation), "tolerance": tolerance},
        ))


def _check_direction_determinant(
    geom: ImageGeometry,
    report: GeometryReport,
    tolerance: float = 1e-4,
) -> None:
    """Validate that direction matrix determinant is +1 (right-handed)."""
    mat = geom.direction_matrix
    det = np.linalg.det(mat)

    if abs(det - 1.0) > tolerance:
        severity = Severity.WARNING if abs(det + 1.0) < tolerance else Severity.ERROR
        report.add(ValidationIssue(
            check="direction_determinant",
            severity=severity,
            message=(
                f"{geom.source}: Direction determinant is {det:.6f} "
                f"(expected +1.0 for right-handed coordinate system)"
            ),
            details={"determinant": float(det)},
        ))


def _check_size(geom: ImageGeometry, report: GeometryReport) -> None:
    """Validate that image dimensions are positive."""
    for axis, label in enumerate(["X", "Y", "Z"]):
        if axis >= len(geom.size):
            break
        if geom.size[axis] <= 0:
            report.add(ValidationIssue(
                check="size_positive",
                severity=Severity.CRITICAL,
                message=f"{geom.source}: {label}-dimension is non-positive ({geom.size[axis]})",
                details={"axis": label, "value": geom.size[axis]},
            ))


def _check_origin_finite(geom: ImageGeometry, report: GeometryReport) -> None:
    """Validate that origin coordinates are finite."""
    if not np.all(np.isfinite(geom.origin)):
        report.add(ValidationIssue(
            check="origin_finite",
            severity=Severity.CRITICAL,
            message=f"{geom.source}: Origin contains non-finite values: {geom.origin}",
            details={"origin": geom.origin.tolist()},
        ))


def validate_single_image(geom: ImageGeometry, report: Optional[GeometryReport] = None) -> GeometryReport:
    """Run all single-image geometry checks.

    Args:
        geom: Extracted image geometry.
        report: Existing report to append to (creates new if None).

    Returns:
        GeometryReport with all findings.
    """
    if report is None:
        report = GeometryReport()

    report.images_validated.append(geom.source)

    _check_origin_finite(geom, report)
    _check_spacing(geom, report)
    _check_size(geom, report)
    _check_direction_orthogonality(geom, report)
    _check_direction_determinant(geom, report)

    logger.debug(f"Validated geometry for {geom.source}: {report.error_count} errors")
    return report



def _compare_spacing(
    geom_a: ImageGeometry,
    geom_b: ImageGeometry,
    report: GeometryReport,
    tolerance_mm: float = 0.01,
) -> None:
    """Compare voxel spacing between two images."""
    n = min(len(geom_a.spacing), len(geom_b.spacing), 3)
    for i, label in enumerate(["X", "Y", "Z"][:n]):
        diff = abs(geom_a.spacing[i] - geom_b.spacing[i])
        if diff > tolerance_mm:
            report.add(ValidationIssue(
                check="cross_spacing_match",
                severity=Severity.WARNING,
                message=(
                    f"{label}-spacing mismatch: "
                    f"{geom_a.source} ({geom_a.spacing[i]:.4f}) vs "
                    f"{geom_b.source} ({geom_b.spacing[i]:.4f}), "
                    f"diff={diff:.4f} mm"
                ),
                details={
                    "axis": label,
                    "spacing_a": float(geom_a.spacing[i]),
                    "spacing_b": float(geom_b.spacing[i]),
                    "diff_mm": float(diff),
                },
            ))


def _compare_origin(
    geom_a: ImageGeometry,
    geom_b: ImageGeometry,
    report: GeometryReport,
    tolerance_mm: float = 0.5,
) -> None:
    """Compare image origins for spatial alignment."""
    n = min(len(geom_a.origin), len(geom_b.origin), 3)
    diff = np.abs(geom_a.origin[:n] - geom_b.origin[:n])
    max_diff = float(np.max(diff))

    if max_diff > tolerance_mm:
        severity = Severity.ERROR if max_diff > 5.0 else Severity.WARNING
        report.add(ValidationIssue(
            check="cross_origin_match",
            severity=severity,
            message=(
                f"Origin mismatch: max diff={max_diff:.4f} mm between "
                f"{geom_a.source} and {geom_b.source}"
            ),
            details={
                "origin_a": geom_a.origin[:n].tolist(),
                "origin_b": geom_b.origin[:n].tolist(),
                "diff_mm": diff.tolist(),
            },
        ))


def _compare_direction(
    geom_a: ImageGeometry,
    geom_b: ImageGeometry,
    report: GeometryReport,
    tolerance: float = 1e-3,
) -> None:
    """Compare direction cosine matrices for orientation consistency."""
    mat_a = geom_a.direction_matrix
    mat_b = geom_b.direction_matrix

    if mat_a.shape != mat_b.shape:
        report.add(ValidationIssue(
            check="cross_direction_shape",
            severity=Severity.ERROR,
            message=(
                f"Direction matrix shape mismatch: "
                f"{geom_a.source} {mat_a.shape} vs {geom_b.source} {mat_b.shape}"
            ),
        ))
        return

    max_diff = float(np.max(np.abs(mat_a - mat_b)))
    if max_diff > tolerance:
        report.add(ValidationIssue(
            check="cross_direction_match",
            severity=Severity.ERROR,
            message=(
                f"Direction cosine mismatch (max diff={max_diff:.6f}) between "
                f"{geom_a.source} and {geom_b.source}. "
                f"Images may have different orientations."
            ),
            details={"max_diff": max_diff, "tolerance": tolerance},
        ))


def _check_contours_within_dose_grid(
    contour_bounds: Dict[str, np.ndarray],
    dose_geom: ImageGeometry,
    report: GeometryReport,
    margin_mm: float = 1.0,
) -> None:
    """Verify that RTSTRUCT contour points fall within the RTDOSE physical grid."""
    extent = dose_geom.physical_extent
    dose_min = dose_geom.origin[:3]
    dose_max = dose_min + extent

    contour_min = contour_bounds["min_bound"]
    contour_max = contour_bounds["max_bound"]

    for i, label in enumerate(["X", "Y", "Z"]):
        if contour_min[i] < dose_min[i] - margin_mm:
            report.add(ValidationIssue(
                check="contour_within_dose",
                severity=Severity.ERROR,
                message=(
                    f"Contour extends below dose grid in {label}: "
                    f"contour min={contour_min[i]:.2f} mm < "
                    f"dose min={dose_min[i]:.2f} mm"
                ),
                details={
                    "axis": label,
                    "contour_min": float(contour_min[i]),
                    "dose_min": float(dose_min[i]),
                },
            ))
        if contour_max[i] > dose_max[i] + margin_mm:
            report.add(ValidationIssue(
                check="contour_within_dose",
                severity=Severity.ERROR,
                message=(
                    f"Contour extends above dose grid in {label}: "
                    f"contour max={contour_max[i]:.2f} mm > "
                    f"dose max={dose_max[i]:.2f} mm"
                ),
                details={
                    "axis": label,
                    "contour_max": float(contour_max[i]),
                    "dose_max": float(dose_max[i]),
                },
            ))


def validate_cross_modality(
    geom_a: ImageGeometry,
    geom_b: ImageGeometry,
    report: Optional[GeometryReport] = None,
    spacing_tolerance_mm: float = 0.01,
    origin_tolerance_mm: float = 0.5,
    direction_tolerance: float = 1e-3,
) -> GeometryReport:
    """Validate geometric consistency between two images.

    Args:
        geom_a: First image geometry.
        geom_b: Second image geometry.
        report: Existing report to append to.
        spacing_tolerance_mm: Max acceptable spacing difference per axis.
        origin_tolerance_mm: Max acceptable origin difference per axis.
        direction_tolerance: Max acceptable direction cosine difference.

    Returns:
        GeometryReport with cross-modality findings appended.
    """
    if report is None:
        report = GeometryReport()

    _compare_spacing(geom_a, geom_b, report, tolerance_mm=spacing_tolerance_mm)
    _compare_origin(geom_a, geom_b, report, tolerance_mm=origin_tolerance_mm)
    _compare_direction(geom_a, geom_b, report, tolerance=direction_tolerance)

    logger.info(
        f"Cross-modality validation: {geom_a.source} vs {geom_b.source} — "
        f"{report.error_count} errors, {report.warning_count} warnings"
    )
    return report


def validate_rtstruct_dose_alignment(
    rtstruct_path: Union[str, Path],
    dose_geom: ImageGeometry,
    roi_number: Optional[int] = None,
    report: Optional[GeometryReport] = None,
    margin_mm: float = 1.0,
) -> GeometryReport:
    """Validate that RTSTRUCT contours are spatially consistent with RTDOSE grid.

    Args:
        rtstruct_path: Path to RTSTRUCT DICOM file.
        dose_geom: Extracted geometry of the RTDOSE image.
        roi_number: Specific ROI to validate (all ROIs if None).
        report: Existing report to append to.
        margin_mm: Allowable margin outside dose grid boundary.

    Returns:
        GeometryReport with contour-vs-dose alignment findings.
    """
    if report is None:
        report = GeometryReport()

    contour_bounds = extract_rtstruct_contour_bounds(rtstruct_path, roi_number=roi_number)

    report.add(ValidationIssue(
        check="rtstruct_contour_stats",
        severity=Severity.INFO,
        message=(
            f"RTSTRUCT contours: {contour_bounds['num_contours']} contour slices, "
            f"{contour_bounds['num_points']} total points"
        ),
        details={
            "num_contours": contour_bounds["num_contours"],
            "num_points": contour_bounds["num_points"],
            "center": contour_bounds["center"].tolist(),
        },
    ))

    _check_contours_within_dose_grid(contour_bounds, dose_geom, report, margin_mm=margin_mm)

    return report



def validate_4dct_phase_consistency(
    phase_geometries: List[ImageGeometry],
    report: Optional[GeometryReport] = None,
    spacing_tolerance_mm: float = 0.001,
    origin_tolerance_mm: float = 0.1,
) -> GeometryReport:
    """Validate that all phases of a 4D-CT share consistent geometry.

    Each phase should have identical spacing, origin, direction, and
    grid dimensions. Small deviations may indicate reconstruction issues.

    Args:
        phase_geometries: List of ImageGeometry, one per respiratory phase.
        report: Existing report to append to.
        spacing_tolerance_mm: Per-axis spacing tolerance.
        origin_tolerance_mm: Per-axis origin tolerance.

    Returns:
        GeometryReport with 4D-CT consistency findings.
    """
    if report is None:
        report = GeometryReport()

    if len(phase_geometries) < 2:
        report.add(ValidationIssue(
            check="4dct_phase_count",
            severity=Severity.WARNING,
            message=f"Only {len(phase_geometries)} phase(s) provided for 4D-CT validation.",
        ))
        return report

    ref = phase_geometries[0]

    for idx, phase in enumerate(phase_geometries[1:], start=1):
        if phase.size != ref.size:
            report.add(ValidationIssue(
                check="4dct_size_consistency",
                severity=Severity.ERROR,
                message=(
                    f"Phase {idx} grid size {phase.size} differs from "
                    f"reference phase 0 size {ref.size}"
                ),
                details={"phase": idx, "size": phase.size, "ref_size": ref.size},
            ))

        spacing_diff = np.abs(phase.spacing[:3] - ref.spacing[:3])
        if np.any(spacing_diff > spacing_tolerance_mm):
            report.add(ValidationIssue(
                check="4dct_spacing_consistency",
                severity=Severity.ERROR,
                message=(
                    f"Phase {idx} spacing differs from reference: "
                    f"max diff={spacing_diff.max():.6f} mm"
                ),
                details={
                    "phase": idx,
                    "spacing": phase.spacing[:3].tolist(),
                    "ref_spacing": ref.spacing[:3].tolist(),
                },
            ))

        origin_diff = np.abs(phase.origin[:3] - ref.origin[:3])
        if np.any(origin_diff > origin_tolerance_mm):
            report.add(ValidationIssue(
                check="4dct_origin_consistency",
                severity=Severity.WARNING,
                message=(
                    f"Phase {idx} origin differs from reference: "
                    f"max diff={origin_diff.max():.4f} mm"
                ),
                details={
                    "phase": idx,
                    "origin": phase.origin[:3].tolist(),
                    "ref_origin": ref.origin[:3].tolist(),
                },
            ))

    report.add(ValidationIssue(
        check="4dct_phase_count",
        severity=Severity.INFO,
        message=f"Validated {len(phase_geometries)} 4D-CT phases for geometric consistency.",
    ))

    return report



def physical_to_voxel(
    physical_coords: np.ndarray,
    origin: np.ndarray,
    spacing: np.ndarray,
    direction: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Convert physical (mm) coordinates to continuous voxel indices.

    Args:
        physical_coords: (N, 3) array of physical coordinates.
        origin: (3,) image origin in mm.
        spacing: (3,) voxel spacing in mm.
        direction: (3, 3) direction cosine matrix (identity if None).

    Returns:
        (N, 3) array of continuous voxel indices.
    """
    coords = np.atleast_2d(physical_coords)
    offset = coords - origin[:3]

    if direction is not None:
        dir_mat = direction.reshape(3, 3) if direction.ndim == 1 else direction
        inv_dir = np.linalg.inv(dir_mat)
        offset = (inv_dir @ offset.T).T

    voxel_coords = offset / spacing[:3]
    return voxel_coords


def voxel_to_physical(
    voxel_coords: np.ndarray,
    origin: np.ndarray,
    spacing: np.ndarray,
    direction: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Convert continuous voxel indices to physical (mm) coordinates.

    Args:
        voxel_coords: (N, 3) array of voxel indices.
        origin: (3,) image origin in mm.
        spacing: (3,) voxel spacing in mm.
        direction: (3, 3) direction cosine matrix (identity if None).

    Returns:
        (N, 3) array of physical coordinates in mm.
    """
    coords = np.atleast_2d(voxel_coords)
    scaled = coords * spacing[:3]

    if direction is not None:
        dir_mat = direction.reshape(3, 3) if direction.ndim == 1 else direction
        scaled = (dir_mat @ scaled.T).T

    physical = scaled + origin[:3]
    return physical



def run_full_validation(
    ct_paths: Optional[List[Union[str, Path]]] = None,
    dose_path: Optional[Union[str, Path]] = None,
    rtstruct_path: Optional[Union[str, Path]] = None,
    roi_number: Optional[int] = None,
) -> GeometryReport:
    """Run comprehensive geometry validation across all available inputs.

    Args:
        ct_paths: List of 4D-CT phase image paths (NIfTI or DICOM).
        dose_path: Path to RTDOSE DICOM file.
        rtstruct_path: Path to RTSTRUCT DICOM file.
        roi_number: ROI number for RTSTRUCT validation.

    Returns:
        Complete GeometryReport with all findings.
    """
    report = GeometryReport()

    ct_geometries = []
    if ct_paths:
        for ct_path in ct_paths:
            p = Path(ct_path)
            if p.suffix.lower() in (".dcm", ""):
                geom = extract_geometry_dicom(p, modality="CT")
            else:
                geom = extract_geometry_sitk(p)
            validate_single_image(geom, report)
            ct_geometries.append(geom)

        if len(ct_geometries) > 1:
            validate_4dct_phase_consistency(ct_geometries, report)

    dose_geom = None
    if dose_path:
        dose_geom = extract_geometry_dicom(dose_path, modality="RTDOSE")
        validate_single_image(dose_geom, report)

    if dose_geom and ct_geometries:
        validate_cross_modality(ct_geometries[0], dose_geom, report)

    if rtstruct_path and dose_geom:
        validate_rtstruct_dose_alignment(rtstruct_path, dose_geom, roi_number=roi_number, report=report)

    logger.info(f"Full geometry validation complete:\n{report.summary()}")
    return report
