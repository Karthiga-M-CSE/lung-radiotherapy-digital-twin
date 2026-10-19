from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import signal, stats

from src.utils.geometry import compute_bounding_box
from src.utils.geometry_validation import ImageGeometry, Severity, ValidationIssue
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TumorCentroid:
    frame: int
    voxel_centroid: Tuple[float, float, float]
    physical_centroid_mm: Tuple[float, float, float]
    tumor_voxels: int
    tumor_volume_mm3: float
    bbox_voxels: Tuple[float, float, float]
    bbox_mm: Tuple[float, float, float]
    mask_shape: Tuple[int, ...]
    affine: Optional[np.ndarray] = None


@dataclass
class TrajectoryValidationResult:
    passed: bool = True
    geometry_passed: bool = True
    respiratory_passed: bool = True
    issues: List[ValidationIssue] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def add_issue(self, issue: ValidationIssue) -> None:
        self.issues.append(issue)
        if issue.severity in (Severity.ERROR, Severity.CRITICAL):
            self.passed = False
            if issue.check.startswith("geom_") or issue.check.startswith("boundary_"):
                self.geometry_passed = False
            elif issue.check.startswith("resp_") or issue.check.startswith("correlation_"):
                self.respiratory_passed = False

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity in (Severity.ERROR, Severity.CRITICAL))

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.WARNING)

    def summary(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        lines = [
            f"Trajectory Validation: {status}",
            f"  Geometry checks    : {'PASSED' if self.geometry_passed else 'FAILED'}",
            f"  Respiratory checks : {'PASSED' if self.respiratory_passed else 'FAILED'}",
            f"  Errors             : {self.error_count}",
            f"  Warnings           : {self.warning_count}",
            f"  Total findings     : {len(self.issues)}",
        ]
        if self.metrics:
            lines.append("  Key Metrics:")
            for k, v in self.metrics.items():
                if isinstance(v, float):
                    lines.append(f"    - {k}: {v:.4f}")
                else:
                    lines.append(f"    - {k}: {v}")
        return "\n".join(lines)


@dataclass
class TumorTrajectoryResult:
    trajectory: pd.DataFrame
    centroids: List[TumorCentroid] = field(default_factory=list)
    validation: Optional[TrajectoryValidationResult] = None
    motion_range_mm: Dict[str, float] = field(default_factory=dict)


def get_frame_id(mask_path_or_str: Union[str, Path]) -> int:
    filename = Path(mask_path_or_str).name
    patterns = [
        r"^tumormask_(\d+)\.nii(?:\.gz)?$",
        r"^frame_?(\d+)\.nii(?:\.gz)?$",
        r"^mask_?(\d+)\.nii(?:\.gz)?$",
        r"(\d+)",
    ]
    for pat in patterns:
        m = re.search(pat, filename, re.IGNORECASE)
        if m:
            return int(m.group(1))
    raise ValueError(f"Could not extract frame ID from filename: '{filename}'")


def compute_tumor_centroid_from_mask(
    mask: np.ndarray,
    affine: Optional[np.ndarray] = None,
    spacing: Optional[Sequence[float]] = None,
    origin: Optional[Sequence[float]] = None,
    frame: int = 0,
) -> Optional[TumorCentroid]:
    tumor_indices = np.argwhere(mask > 0)
    if tumor_indices.size == 0:
        return None

    voxel_centroid = tumor_indices.mean(axis=0)

    if affine is not None:
        physical_centroid = nib.affines.apply_affine(affine, voxel_centroid)
        voxel_vol = abs(float(np.linalg.det(affine[:3, :3])))
        sp_x = float(np.linalg.norm(affine[:3, 0]))
        sp_y = float(np.linalg.norm(affine[:3, 1]))
        sp_z = float(np.linalg.norm(affine[:3, 2]))
        voxel_dims = (sp_x, sp_y, sp_z)
    else:
        sp = tuple(spacing) if spacing is not None else (1.0, 1.0, 1.0)
        orig = np.array(origin if origin is not None else (0.0, 0.0, 0.0), dtype=float)
        physical_centroid = orig + (voxel_centroid * np.array(sp))
        voxel_vol = float(sp[0] * sp[1] * sp[2])
        voxel_dims = (float(sp[0]), float(sp[1]), float(sp[2]))

    tumor_voxel_count = int(len(tumor_indices))
    tumor_volume_mm3 = float(tumor_voxel_count * voxel_vol)

    min_coords = tumor_indices.min(axis=0)
    max_coords = tumor_indices.max(axis=0)
    bbox_voxels = tuple(float(c) for c in (max_coords - min_coords + 1))
    bbox_mm = compute_bounding_box(tumor_indices, voxel_dims)

    return TumorCentroid(
        frame=frame,
        voxel_centroid=(float(voxel_centroid[0]), float(voxel_centroid[1]), float(voxel_centroid[2])),
        physical_centroid_mm=(float(physical_centroid[0]), float(physical_centroid[1]), float(physical_centroid[2])),
        tumor_voxels=tumor_voxel_count,
        tumor_volume_mm3=tumor_volume_mm3,
        bbox_voxels=(bbox_voxels[0], bbox_voxels[1], bbox_voxels[2]),
        bbox_mm=bbox_mm,
        mask_shape=mask.shape,
        affine=affine.copy() if affine is not None else None,
    )


def extract_tumor_centroid_from_file(mask_path: Union[str, Path]) -> Optional[TumorCentroid]:
    path = Path(mask_path)
    if not path.exists():
        raise FileNotFoundError(f"Mask file not found: {path}")

    frame_id = get_frame_id(path)
    img = nib.load(str(path))
    mask = img.get_fdata()
    affine = img.affine

    return compute_tumor_centroid_from_mask(
        mask=mask,
        affine=affine,
        frame=frame_id,
    )


def extract_trajectory_from_masks(
    mask_source: Union[str, Path, Sequence[Union[str, Path]]],
    pattern: str = "tumormask_*.nii.gz",
    reference_frame: Optional[int] = None,
    dt: float = 1.0,
    validate_spatial_consistency: bool = True,
) -> pd.DataFrame:
    if isinstance(mask_source, (str, Path)):
        source_dir = Path(mask_source)
        if not source_dir.exists():
            raise FileNotFoundError(f"Mask directory does not exist: {source_dir}")
        mask_files = list(source_dir.glob(pattern))
    else:
        mask_files = [Path(p) for p in mask_source]

    if not mask_files:
        raise FileNotFoundError(f"No mask files found matching '{pattern}'")

    mask_files = sorted(mask_files, key=get_frame_id)
    logger.info(f"Extracting trajectory from {len(mask_files)} mask files...")

    centroids: List[TumorCentroid] = []
    ref_shape = None
    ref_affine = None

    for mask_path in mask_files:
        centroid = extract_tumor_centroid_from_file(mask_path)
        if centroid is None:
            logger.warning(f"No tumor voxels found in {mask_path.name}")
            continue

        if validate_spatial_consistency:
            if ref_shape is None:
                ref_shape = centroid.mask_shape
                ref_affine = centroid.affine
            else:
                if centroid.mask_shape != ref_shape:
                    raise ValueError(
                        f"Mask shape mismatch in {mask_path.name}: "
                        f"expected {ref_shape}, found {centroid.mask_shape}"
                    )
                if ref_affine is not None and centroid.affine is not None:
                    if not np.allclose(centroid.affine, ref_affine, atol=1e-3):
                        raise ValueError(f"Mask affine mismatch in {mask_path.name}")

        centroids.append(centroid)

    if not centroids:
        raise RuntimeError("No valid tumor masks could be processed.")

    df = _centroids_to_dataframe(centroids, reference_frame=reference_frame, dt=dt)
    return df


def _centroids_to_dataframe(
    centroids: List[TumorCentroid],
    reference_frame: Optional[int] = None,
    dt: float = 1.0,
) -> pd.DataFrame:
    records = []
    for c in centroids:
        records.append({
            "frame": c.frame,
            "x_mm": c.physical_centroid_mm[0],
            "y_mm": c.physical_centroid_mm[1],
            "z_mm": c.physical_centroid_mm[2],
            "x_voxel": c.voxel_centroid[0],
            "y_voxel": c.voxel_centroid[1],
            "z_voxel": c.voxel_centroid[2],
            "tumor_voxels": c.tumor_voxels,
            "tumor_volume_mm3": c.tumor_volume_mm3,
            "bbox_x_mm": c.bbox_mm[0],
            "bbox_y_mm": c.bbox_mm[1],
            "bbox_z_mm": c.bbox_mm[2],
        })

    df = pd.DataFrame(records).sort_values("frame").reset_index(drop=True)

    if reference_frame is not None and reference_frame in df["frame"].values:
        ref_row = df[df["frame"] == reference_frame].iloc[0]
    else:
        ref_row = df.iloc[0]

    ref_x = ref_row["x_mm"]
    ref_y = ref_row["y_mm"]
    ref_z = ref_row["z_mm"]

    df["dx_mm"] = df["x_mm"] - ref_x
    df["dy_mm"] = df["y_mm"] - ref_y
    df["dz_mm"] = df["z_mm"] - ref_z

    df["displacement_3d_mm"] = np.sqrt(
        df["dx_mm"] ** 2 + df["dy_mm"] ** 2 + df["dz_mm"] ** 2
    )
    df["motion_magnitude_mm"] = df["displacement_3d_mm"]

    df = compute_kinematics(df, dt=dt)
    return df


def compute_kinematics(trajectory_df: pd.DataFrame, dt: float = 1.0) -> pd.DataFrame:
    df = trajectory_df.copy()
    dt = max(dt, 1e-6)

    for axis in ["x", "y", "z"]:
        pos = df[f"{axis}_mm"].values
        if len(pos) >= 2:
            vel = np.gradient(pos, dt)
            acc = np.gradient(vel, dt)
        else:
            vel = np.zeros_like(pos)
            acc = np.zeros_like(pos)

        df[f"v{axis}_mm_s"] = vel
        df[f"a{axis}_mm_s2"] = acc

    df["speed_mm_s"] = np.sqrt(
        df["vx_mm_s"] ** 2 + df["vy_mm_s"] ** 2 + df["vz_mm_s"] ** 2
    )
    df["accel_3d_mm_s2"] = np.sqrt(
        df["ax_mm_s2"] ** 2 + df["ay_mm_s2"] ** 2 + df["az_mm_s2"] ** 2
    )
    return df


def compute_principal_motion_axis(trajectory_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, float]:
    coords = trajectory_df[["dx_mm", "dy_mm", "dz_mm"]].values
    centered = coords - coords.mean(axis=0)

    if len(coords) < 3:
        default_axis = np.array([0.0, 0.0, 1.0])
        proj = coords @ default_axis
        return default_axis, proj, 1.0

    _, s, vh = np.linalg.svd(centered, full_matrices=False)
    principal_vector = vh[0]

    if principal_vector[2] < 0:
        principal_vector = -principal_vector

    projected_motion = coords @ principal_vector
    total_var = np.sum(s ** 2)
    var_ratio = float((s[0] ** 2) / total_var) if total_var > 0 else 1.0

    return principal_vector, projected_motion, var_ratio


def interpolate_missing_frames(
    trajectory_df: pd.DataFrame,
    full_frame_range: Optional[Tuple[int, int]] = None,
    method: str = "linear",
    dt: float = 1.0,
) -> pd.DataFrame:
    df = trajectory_df.sort_values("frame").copy()
    min_f = full_frame_range[0] if full_frame_range else int(df["frame"].min())
    max_f = full_frame_range[1] if full_frame_range else int(df["frame"].max())

    all_frames = pd.DataFrame({"frame": np.arange(min_f, max_f + 1, dtype=int)})
    merged = pd.merge(all_frames, df, on="frame", how="left")

    numeric_cols = [c for c in df.columns if c != "frame"]
    for col in numeric_cols:
        merged[col] = merged[col].interpolate(method=method).bfill().ffill()

    merged = compute_kinematics(merged, dt=dt)
    return merged


def generate_physical_trajectory(
    raw_trajectory: pd.DataFrame,
    dt: float = 1.0,
    interpolate_missing: bool = True,
    compute_principal_axis: bool = True,
) -> pd.DataFrame:
    df = raw_trajectory.copy()

    if interpolate_missing and len(df) > 1:
        expected_frames = int(df["frame"].max() - df["frame"].min() + 1)
        if len(df) < expected_frames:
            logger.info(f"Interpolating {expected_frames - len(df)} missing frames in trajectory...")
            df = interpolate_missing_frames(df, dt=dt)

    df = compute_kinematics(df, dt=dt)

    if compute_principal_axis and len(df) >= 3:
        axis_vec, proj, var_exp = compute_principal_motion_axis(df)
        df["principal_motion_mm"] = proj
        logger.info(
            f"Principal motion axis: [{axis_vec[0]:.3f}, {axis_vec[1]:.3f}, {axis_vec[2]:.3f}], "
            f"Variance explained: {var_exp * 100:.1f}%"
        )

    return df


def validate_trajectory_against_geometry(
    trajectory: pd.DataFrame,
    image_geometry: Optional[ImageGeometry] = None,
    max_plausible_displacement_mm: float = 35.0,
    max_velocity_jump_mm_s: float = 80.0,
    volume_variation_tolerance: float = 0.5,
) -> TrajectoryValidationResult:
    res = TrajectoryValidationResult()
    n_frames = len(trajectory)
    if n_frames == 0:
        res.add_issue(ValidationIssue(
            check="geom_empty_trajectory",
            severity=Severity.ERROR,
            message="Trajectory contains zero frames.",
        ))
        return res

    if image_geometry is not None:
        origin = image_geometry.origin[:3]
        extent = image_geometry.physical_extent
        upper_bound = origin + extent

        x_pts = trajectory["x_mm"].values
        y_pts = trajectory["y_mm"].values
        z_pts = trajectory["z_mm"].values

        axes = [("X", x_pts, 0), ("Y", y_pts, 1), ("Z", z_pts, 2)]
        for name, pts, ax_idx in axes:
            min_pt, max_pt = float(pts.min()), float(pts.max())
            o_min, o_max = float(min(origin[ax_idx], upper_bound[ax_idx])), float(max(origin[ax_idx], upper_bound[ax_idx]))

            if min_pt < o_min - 2.0 or max_pt > o_max + 2.0:
                res.add_issue(ValidationIssue(
                    check="geom_fov_exceeded",
                    severity=Severity.ERROR,
                    message=(
                        f"Tumor trajectory in {name} [{min_pt:.2f}, {max_pt:.2f}] mm "
                        f"extends outside image FOV [{o_min:.2f}, {o_max:.2f}] mm"
                    ),
                    details={"axis": name, "min": min_pt, "max": max_pt, "fov_min": o_min, "fov_max": o_max},
                ))

        if "x_voxel" in trajectory.columns and len(image_geometry.size) >= 3:
            size = image_geometry.size
            for name, col, sz in [("X", "x_voxel", size[0]), ("Y", "y_voxel", size[1]), ("Z", "z_voxel", size[2])]:
                v_min = float(trajectory[col].min())
                v_max = float(trajectory[col].max())
                if v_min < 0 or v_max >= sz:
                    res.add_issue(ValidationIssue(
                        check="geom_voxel_out_of_bounds",
                        severity=Severity.ERROR,
                        message=f"Voxel coordinates {name} [{v_min:.1f}, {v_max:.1f}] exceed grid size {sz}",
                        details={"axis": name, "min": v_min, "max": v_max, "grid_size": sz},
                    ))

    disp = trajectory.get("displacement_3d_mm", None)
    if disp is None:
        dx = trajectory["x_mm"] - trajectory["x_mm"].iloc[0]
        dy = trajectory["y_mm"] - trajectory["y_mm"].iloc[0]
        dz = trajectory["z_mm"] - trajectory["z_mm"].iloc[0]
        disp = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    max_disp = float(disp.max())
    res.metrics["max_displacement_3d_mm"] = max_disp

    if max_disp > max_plausible_displacement_mm * 1.5:
        res.add_issue(ValidationIssue(
            check="geom_displacement_unphysical",
            severity=Severity.ERROR,
            message=f"Maximum 3D displacement ({max_disp:.2f} mm) is unphysiologically large (> {max_plausible_displacement_mm * 1.5:.1f} mm)",
            details={"max_displacement_mm": max_disp},
        ))
    elif max_disp > max_plausible_displacement_mm:
        res.add_issue(ValidationIssue(
            check="geom_displacement_high",
            severity=Severity.WARNING,
            message=f"Maximum 3D displacement ({max_disp:.2f} mm) is unusually high (> {max_plausible_displacement_mm:.1f} mm)",
            details={"max_displacement_mm": max_disp},
        ))

    if "speed_mm_s" in trajectory.columns:
        max_speed = float(trajectory["speed_mm_s"].max())
        res.metrics["max_speed_mm_s"] = max_speed
        if max_speed > max_velocity_jump_mm_s:
            res.add_issue(ValidationIssue(
                check="geom_velocity_jump",
                severity=Severity.WARNING,
                message=f"Maximum instantaneous speed ({max_speed:.2f} mm/s) exceeds threshold ({max_velocity_jump_mm_s:.1f} mm/s)",
                details={"max_speed_mm_s": max_speed},
            ))

    if "tumor_volume_mm3" in trajectory.columns or "tumor_voxels" in trajectory.columns:
        vol_col = "tumor_volume_mm3" if "tumor_volume_mm3" in trajectory.columns else "tumor_voxels"
        vols = trajectory[vol_col].values
        mean_vol = float(np.mean(vols))
        std_vol = float(np.std(vols))
        cv = (std_vol / mean_vol) if mean_vol > 0 else 0.0
        res.metrics["volume_mean"] = mean_vol
        res.metrics["volume_cv"] = cv

        if cv > volume_variation_tolerance:
            res.add_issue(ValidationIssue(
                check="geom_volume_instability",
                severity=Severity.WARNING,
                message=f"Tumor volume shows high variability across frames (CV = {cv:.2f} > {volume_variation_tolerance:.2f})",
                details={"cv": cv, "mean_volume": mean_vol},
            ))

    res.metrics["frames_validated"] = n_frames
    for ax in ["x", "y", "z"]:
        if f"{ax}_mm" in trajectory.columns:
            res.metrics[f"range_{ax}_mm"] = float(trajectory[f"{ax}_mm"].max() - trajectory[f"{ax}_mm"].min())

    return res


def validate_trajectory_against_respiratory(
    trajectory: pd.DataFrame,
    respiratory_signal_data: Union[np.ndarray, pd.Series, pd.DataFrame],
    rpm_col: Optional[str] = None,
    min_correlation: float = 0.5,
) -> TrajectoryValidationResult:
    res = TrajectoryValidationResult()

    if isinstance(respiratory_signal_data, pd.DataFrame):
        col = rpm_col or ("rpm" if "rpm" in respiratory_signal_data.columns else respiratory_signal_data.columns[0])
        resp_vals = respiratory_signal_data[col].values
    elif isinstance(respiratory_signal_data, pd.Series):
        resp_vals = respiratory_signal_data.values
    else:
        resp_vals = np.asarray(respiratory_signal_data)

    n_traj = len(trajectory)
    n_resp = len(resp_vals)

    if n_resp == 0 or n_traj == 0:
        res.add_issue(ValidationIssue(
            check="resp_empty_signal",
            severity=Severity.ERROR,
            message="Respiratory signal or trajectory is empty.",
        ))
        return res

    if n_resp != n_traj:
        res.add_issue(ValidationIssue(
            check="resp_length_mismatch",
            severity=Severity.INFO,
            message=f"Resampling respiratory surrogate from {n_resp} to {n_traj} samples to match trajectory.",
            details={"resp_len": n_resp, "traj_len": n_traj},
        ))
        t_orig = np.linspace(0, 1, n_resp)
        t_new = np.linspace(0, 1, n_traj)
        resp_vals = np.interp(t_new, t_orig, resp_vals)

    if np.std(resp_vals) < 1e-6:
        res.add_issue(ValidationIssue(
            check="resp_flat_signal",
            severity=Severity.ERROR,
            message="Respiratory signal has near-zero variance (flatline).",
        ))
        return res

    motion_signals = {}
    if "displacement_3d_mm" in trajectory.columns:
        motion_signals["displacement_3d"] = trajectory["displacement_3d_mm"].values
    if "dz_mm" in trajectory.columns:
        motion_signals["dz"] = trajectory["dz_mm"].values
    if "principal_motion_mm" in trajectory.columns:
        motion_signals["principal_motion"] = trajectory["principal_motion_mm"].values

    correlations = {}
    best_abs_corr = 0.0

    for name, sig in motion_signals.items():
        if np.std(sig) < 1e-6:
            continue
        r, _ = stats.pearsonr(resp_vals, sig)
        rho, _ = stats.spearmanr(resp_vals, sig)
        correlations[f"pearson_{name}"] = float(r)
        correlations[f"spearman_{name}"] = float(rho)
        if abs(r) > best_abs_corr:
            best_abs_corr = abs(r)

    res.metrics.update(correlations)
    res.metrics["best_abs_correlation"] = best_abs_corr

    sig_norm = (resp_vals - np.mean(resp_vals)) / (np.std(resp_vals) + 1e-9)
    primary_sig = motion_signals.get("dz", motion_signals.get("displacement_3d", None))

    if primary_sig is not None and np.std(primary_sig) > 1e-6:
        mot_norm = (primary_sig - np.mean(primary_sig)) / (np.std(primary_sig) + 1e-9)
        xcorr = signal.correlate(mot_norm, sig_norm, mode="full")
        lags = signal.correlation_lags(len(mot_norm), len(sig_norm), mode="full")
        best_lag = int(lags[np.argmax(np.abs(xcorr))])
        res.metrics["estimated_phase_lag_frames"] = best_lag

    if best_abs_corr < min_correlation:
        res.add_issue(ValidationIssue(
            check="resp_low_correlation",
            severity=Severity.WARNING,
            message=(
                f"Peak correlation between tumor trajectory and respiratory signal is low "
                f"(|r| = {best_abs_corr:.3f} < {min_correlation:.2f})."
            ),
            details={"best_abs_r": best_abs_corr, "threshold": min_correlation},
        ))
    else:
        res.add_issue(ValidationIssue(
            check="resp_correlation_ok",
            severity=Severity.INFO,
            message=f"Good correlation with respiratory signal (|r| = {best_abs_corr:.3f} >= {min_correlation:.2f})",
            details={"best_abs_r": best_abs_corr},
        ))

    return res


def combine_trajectory_with_respiratory(
    trajectory: pd.DataFrame,
    respiratory_data: Union[np.ndarray, pd.Series, pd.DataFrame],
    rpm_col: str = "rpm",
    validate: bool = True,
) -> pd.DataFrame:
    df = trajectory.copy()

    if isinstance(respiratory_data, pd.DataFrame):
        col = rpm_col if rpm_col in respiratory_data.columns else respiratory_data.columns[0]
        rpm_vals = respiratory_data[col].values
    elif isinstance(respiratory_data, pd.Series):
        rpm_vals = respiratory_data.values
    else:
        rpm_vals = np.asarray(respiratory_data)

    if len(rpm_vals) != len(df):
        logger.warning(
            f"Length mismatch: trajectory has {len(df)} rows, RPM has {len(rpm_vals)}. "
            f"Resampling RPM to match trajectory frames."
        )
        t_orig = np.linspace(0, 1, len(rpm_vals))
        t_new = np.linspace(0, 1, len(df))
        rpm_vals = np.interp(t_new, t_orig, rpm_vals)

    df[rpm_col] = rpm_vals

    if validate:
        val_res = validate_trajectory_against_respiratory(df, df[rpm_col])
        logger.info(f"Combined trajectory + respiratory: {val_res.summary()}")

    return df


def validate_tumor_trajectory(
    trajectory: pd.DataFrame,
    image_geometry: Optional[ImageGeometry] = None,
    respiratory_signal: Optional[Union[np.ndarray, pd.Series, pd.DataFrame]] = None,
    max_plausible_displacement_mm: float = 35.0,
    min_respiratory_correlation: float = 0.5,
) -> TrajectoryValidationResult:
    report = TrajectoryValidationResult()

    geom_res = validate_trajectory_against_geometry(
        trajectory,
        image_geometry=image_geometry,
        max_plausible_displacement_mm=max_plausible_displacement_mm,
    )
    for issue in geom_res.issues:
        report.add_issue(issue)
    report.metrics.update(geom_res.metrics)
    report.geometry_passed = geom_res.geometry_passed

    if respiratory_signal is not None:
        resp_res = validate_trajectory_against_respiratory(
            trajectory,
            respiratory_signal,
            min_correlation=min_respiratory_correlation,
        )
        for issue in resp_res.issues:
            report.add_issue(issue)
        report.metrics.update(resp_res.metrics)
        report.respiratory_passed = resp_res.respiratory_passed

    logger.info(f"Complete trajectory validation:\n{report.summary()}")
    return report
