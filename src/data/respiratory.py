from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, find_peaks

from src.utils.logging import get_logger

logger = get_logger(__name__)


def butterworth_lowpass(
    signal: np.ndarray,
    cutoff_hz: float,
    fs_hz: float,
    order: int = 4,
) -> np.ndarray:
    nyquist = fs_hz / 2.0
    if cutoff_hz >= nyquist:
        raise ValueError(f"Cutoff ({cutoff_hz} Hz) must be below Nyquist ({nyquist} Hz)")
    min_length = 3 * max(order, 1) + 1
    if len(signal) < min_length:
        raise ValueError(
            f"Signal length ({len(signal)}) too short for filter order {order}. "
            f"Need at least {min_length} samples."
        )

    b, a = butter(order, cutoff_hz / nyquist, btype="low")
    filtered = filtfilt(b, a, signal)
    logger.debug(f"Lowpass filter applied: cutoff={cutoff_hz} Hz, fs={fs_hz} Hz, order={order}")
    return filtered


def butterworth_bandpass(
    signal: np.ndarray,
    low_hz: float,
    high_hz: float,
    fs_hz: float,
    order: int = 4,
) -> np.ndarray:
    nyquist = fs_hz / 2.0
    if low_hz <= 0:
        raise ValueError(f"Lower cutoff ({low_hz} Hz) must be positive")
    if high_hz >= nyquist:
        raise ValueError(f"Upper cutoff ({high_hz} Hz) must be below Nyquist ({nyquist} Hz)")
    if low_hz >= high_hz:
        raise ValueError(f"Lower cutoff ({low_hz}) must be less than upper cutoff ({high_hz})")
    min_length = 3 * max(order, 1) + 1
    if len(signal) < min_length:
        raise ValueError(
            f"Signal length ({len(signal)}) too short for filter order {order}. "
            f"Need at least {min_length} samples."
        )

    b, a = butter(order, [low_hz / nyquist, high_hz / nyquist], btype="band")
    filtered = filtfilt(b, a, signal)
    logger.debug(f"Bandpass filter applied: {low_hz}-{high_hz} Hz, fs={fs_hz} Hz, order={order}")
    return filtered


def synchronize_timestamps(
    rpm_timestamps: np.ndarray,
    rpm_values: np.ndarray,
    ct_timestamps: np.ndarray,
    method: str = "linear",
) -> np.ndarray:
    if len(rpm_timestamps) != len(rpm_values):
        raise ValueError(
            f"RPM timestamps ({len(rpm_timestamps)}) and values ({len(rpm_values)}) "
            f"must have the same length"
        )

    if not np.all(np.diff(rpm_timestamps) > 0):
        raise ValueError("RPM timestamps must be monotonically increasing")

    ct_min, ct_max = ct_timestamps.min(), ct_timestamps.max()
    rpm_min, rpm_max = rpm_timestamps.min(), rpm_timestamps.max()

    if ct_min < rpm_min or ct_max > rpm_max:
        logger.warning(
            f"CT timestamps [{ct_min:.3f}, {ct_max:.3f}] extend beyond "
            f"RPM range [{rpm_min:.3f}, {rpm_max:.3f}]. Extrapolating."
        )

    interpolator = interp1d(
        rpm_timestamps,
        rpm_values,
        kind=method,
        fill_value="extrapolate",
        assume_sorted=True,
    )
    synced = interpolator(ct_timestamps)
    logger.info(
        f"Synchronized {len(ct_timestamps)} CT frames to RPM signal "
        f"({len(rpm_timestamps)} samples) via {method} interpolation"
    )
    return synced


def resample_uniform(
    timestamps: np.ndarray,
    values: np.ndarray,
    target_fs_hz: Optional[float] = None,
    n_samples: Optional[int] = None,
    method: str = "cubic",
) -> Tuple[np.ndarray, np.ndarray]:
    if target_fs_hz is None and n_samples is None:
        raise ValueError("Specify either target_fs_hz or n_samples")

    t_start, t_end = timestamps[0], timestamps[-1]
    duration = t_end - t_start

    if n_samples is not None:
        uniform_t = np.linspace(t_start, t_end, n_samples)
    else:
        dt = 1.0 / target_fs_hz
        uniform_t = np.arange(t_start, t_end + dt * 0.5, dt)

    values_2d = np.atleast_2d(values)
    if values_2d.shape[0] == len(timestamps):
        pass
    elif values_2d.shape[1] == len(timestamps):
        values_2d = values_2d.T
    else:
        raise ValueError(
            f"values shape {values.shape} incompatible with {len(timestamps)} timestamps"
        )

    resampled_cols = []
    for col in range(values_2d.shape[1]):
        interpolator = interp1d(
            timestamps,
            values_2d[:, col],
            kind=method,
            fill_value="extrapolate",
            assume_sorted=True,
        )
        resampled_cols.append(interpolator(uniform_t))

    resampled = np.column_stack(resampled_cols)
    if values.ndim == 1:
        resampled = resampled.ravel()

    actual_fs = (len(uniform_t) - 1) / duration if duration > 0 else 0
    logger.info(
        f"Resampled {len(timestamps)} -> {len(uniform_t)} samples "
        f"(effective fs={actual_fs:.2f} Hz)"
    )
    return uniform_t, resampled


def detect_respiratory_cycles(
    rpm: np.ndarray,
    distance: int = 6,
    prominence_ratio: float = 0.10,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    amplitude_range = rpm.max() - rpm.min()
    prominence = amplitude_range * prominence_ratio

    peaks, _ = find_peaks(rpm, distance=distance, prominence=prominence)
    valleys, _ = find_peaks(-rpm, distance=distance, prominence=prominence)

    cycles = []
    for i in range(len(peaks) - 1):
        start = int(peaks[i])
        end = int(peaks[i + 1])
        mid_valleys = valleys[(valleys > start) & (valleys < end)]
        valley_amp = float(rpm[mid_valleys].min()) if len(mid_valleys) > 0 else float("nan")

        cycles.append({
            "cycle": i + 1,
            "start_frame": start,
            "end_frame": end,
            "num_frames": end - start + 1,
            "peak_amplitude": float(rpm[start]),
            "valley_amplitude": valley_amp,
            "cycle_range": float(rpm[start]) - valley_amp if not np.isnan(valley_amp) else float("nan"),
        })

    cycle_df = pd.DataFrame(cycles)
    logger.info(f"Detected {len(peaks)} peaks, {len(valleys)} valleys, {len(cycle_df)} cycles.")
    return peaks, valleys, cycle_df


@dataclass
class RespiratoryPhaseInfo:
    phase_label: np.ndarray
    phase_fraction: np.ndarray
    cycle_id: np.ndarray


def extract_respiratory_phases(
    rpm: np.ndarray,
    peaks: np.ndarray,
    valleys: np.ndarray,
) -> RespiratoryPhaseInfo:
    n = len(rpm)
    phase_label = np.full(n, "unassigned", dtype=object)
    phase_fraction = np.full(n, np.nan)
    cycle_id = np.full(n, -1, dtype=int)

    landmarks = [(int(p), "peak") for p in peaks] + [(int(v), "valley") for v in valleys]
    landmarks.sort(key=lambda x: x[0])

    current_cycle = 0
    for i in range(len(landmarks) - 1):
        idx_start, type_start = landmarks[i]
        idx_end, type_end = landmarks[i + 1]

        if type_start == "peak" and type_end == "valley":
            label = "exhale"
        elif type_start == "valley" and type_end == "peak":
            label = "inhale"
        else:
            label = "exhale" if type_start == "peak" else "inhale"

        if type_start == "peak":
            current_cycle += 1

        span = idx_end - idx_start
        if span <= 0:
            continue

        for j in range(idx_start, min(idx_end + 1, n)):
            phase_label[j] = label
            phase_fraction[j] = (j - idx_start) / span
            cycle_id[j] = current_cycle

    logger.info(
        f"Phase extraction: {int(np.sum(phase_label == 'inhale'))} inhale, "
        f"{int(np.sum(phase_label == 'exhale'))} exhale, "
        f"{int(np.sum(phase_label == 'unassigned'))} unassigned samples"
    )
    return RespiratoryPhaseInfo(
        phase_label=phase_label,
        phase_fraction=phase_fraction,
        cycle_id=cycle_id,
    )


def compute_kinematic_features(
    positions: np.ndarray,
    dt: float = 1.0,
    smooth_window: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    pos = np.atleast_2d(positions)
    if pos.shape[0] == 1 and pos.shape[1] > 1 and positions.ndim == 1:
        pos = pos.T
    was_1d = positions.ndim == 1
    n = pos.shape[0]

    def _gradient(arr: np.ndarray) -> np.ndarray:
        return np.gradient(arr, dt, axis=0)

    def _smooth(arr: np.ndarray, window: int) -> np.ndarray:
        if window < 2 or n < window:
            return arr
        kernel = np.ones(window) / window
        smoothed = np.empty_like(arr)
        for col in range(arr.shape[1]):
            smoothed[:, col] = np.convolve(arr[:, col], kernel, mode="same")
        return smoothed

    velocity = _gradient(pos)
    acceleration = _gradient(velocity)
    jerk = _gradient(acceleration)

    if smooth_window is not None:
        velocity = _smooth(velocity, smooth_window)
        acceleration = _smooth(acceleration, smooth_window)
        jerk = _smooth(jerk, smooth_window)

    speed = np.linalg.norm(velocity, axis=1)
    abs_accel = np.linalg.norm(acceleration, axis=1)
    displacement = pos - pos[0:1, :]
    amplitude = np.linalg.norm(displacement, axis=1)

    result = {
        "velocity": velocity.ravel() if was_1d else velocity,
        "acceleration": acceleration.ravel() if was_1d else acceleration,
        "jerk": jerk.ravel() if was_1d else jerk,
        "speed": speed,
        "abs_accel": abs_accel,
        "amplitude": amplitude,
    }
    logger.debug(f"Kinematic features computed for {n} samples, {pos.shape[1]} axes, dt={dt}")
    return result


def compute_motion_amplitude_per_cycle(
    positions: np.ndarray,
    cycle_ids: np.ndarray,
) -> Dict[int, Dict[str, float]]:
    pos = np.atleast_2d(positions)
    if pos.shape[0] == 1 and pos.ndim != 1:
        pos = pos.T

    unique_cycles = sorted(set(cycle_ids) - {-1})
    result = {}

    for cid in unique_cycles:
        mask = cycle_ids == cid
        cycle_pos = pos[mask]
        if len(cycle_pos) < 2:
            continue

        magnitudes = np.linalg.norm(cycle_pos - cycle_pos.mean(axis=0), axis=1)
        result[cid] = {
            "range": float(magnitudes.max() - magnitudes.min()),
            "peak_to_peak": float(cycle_pos.max() - cycle_pos.min()) if pos.shape[1] == 1
                else float(np.linalg.norm(cycle_pos.max(axis=0) - cycle_pos.min(axis=0))),
            "std": float(magnitudes.std()),
            "num_samples": int(mask.sum()),
        }

    logger.debug(f"Motion amplitude computed for {len(result)} cycles")
    return result


def create_lagged_dataframe(
    df: pd.DataFrame,
    feature_columns: List[str],
    n_lags: int = 5,
) -> Tuple[pd.DataFrame, List[str]]:
    df_out = df.sort_values("frame").reset_index(drop=True).copy()
    lag_columns = []

    for lag in range(1, n_lags + 1):
        for col in feature_columns:
            col_name = f"{col}_lag{lag}"
            df_out[col_name] = df_out[col].shift(lag)
            lag_columns.append(col_name)

    clean_df = df_out.dropna().reset_index(drop=True)
    logger.debug(f"Created {len(lag_columns)} lag features. Valid rows: {len(clean_df)}")
    return clean_df, lag_columns


@dataclass
class RespiratoryPreprocessResult:
    df: pd.DataFrame
    peaks: np.ndarray
    valleys: np.ndarray
    cycles: pd.DataFrame
    phases: RespiratoryPhaseInfo
    kinematics: Dict[str, np.ndarray]
    cycle_amplitudes: Dict[int, Dict[str, float]]
    metadata: Dict = field(default_factory=dict)


def run_preprocessing_pipeline(
    positions: np.ndarray,
    rpm: np.ndarray,
    frames: Optional[np.ndarray] = None,
    dt: float = 1.0,
    filter_cutoff_hz: Optional[float] = None,
    filter_fs_hz: Optional[float] = None,
    filter_order: int = 4,
    peak_distance: int = 6,
    prominence_ratio: float = 0.10,
    n_lags: int = 5,
    smooth_window: Optional[int] = None,
) -> RespiratoryPreprocessResult:
    n = len(rpm)
    if frames is None:
        frames = np.arange(n)

    positions = np.atleast_2d(positions)
    if positions.shape[0] != n:
        if positions.shape[1] == n:
            positions = positions.T
        else:
            raise ValueError(f"positions shape {positions.shape} incompatible with {n} RPM samples")

    rpm_processed = rpm.copy()
    if filter_cutoff_hz is not None and filter_fs_hz is not None:
        rpm_processed = butterworth_lowpass(rpm_processed, filter_cutoff_hz, filter_fs_hz, filter_order)

    peaks, valleys, cycles = detect_respiratory_cycles(
        rpm_processed, distance=peak_distance, prominence_ratio=prominence_ratio
    )
    phases = extract_respiratory_phases(rpm_processed, peaks, valleys)
    kinematics = compute_kinematic_features(positions, dt=dt, smooth_window=smooth_window)
    cycle_amplitudes = compute_motion_amplitude_per_cycle(positions, phases.cycle_id)

    n_axes = positions.shape[1]
    axis_labels = ["x", "y", "z"][:n_axes]

    df = pd.DataFrame({"frame": frames, "rpm_raw": rpm, "rpm_filtered": rpm_processed})
    for i, label in enumerate(axis_labels):
        df[f"{label}_mm"] = positions[:, i]
    for i, label in enumerate(axis_labels):
        df[f"velocity_{label}"] = kinematics["velocity"][:, i] if positions.ndim > 1 and n_axes > 1 else kinematics["velocity"]
    df["speed"] = kinematics["speed"]
    for i, label in enumerate(axis_labels):
        df[f"accel_{label}"] = kinematics["acceleration"][:, i] if positions.ndim > 1 and n_axes > 1 else kinematics["acceleration"]
    df["abs_accel"] = kinematics["abs_accel"]
    df["amplitude"] = kinematics["amplitude"]
    df["phase"] = phases.phase_label
    df["phase_fraction"] = phases.phase_fraction
    df["cycle_id"] = phases.cycle_id

    metadata = {
        "n_samples": n,
        "n_peaks": len(peaks),
        "n_valleys": len(valleys),
        "n_cycles": len(cycles),
        "dt": dt,
        "filter_applied": filter_cutoff_hz is not None,
    }
    logger.info(
        f"Preprocessing pipeline complete: {n} samples, {len(cycles)} cycles, "
        f"{n_axes} axes, filter={'yes' if metadata['filter_applied'] else 'no'}"
    )

    return RespiratoryPreprocessResult(
        df=df,
        peaks=peaks,
        valleys=valleys,
        cycles=cycles,
        phases=phases,
        kinematics=kinematics,
        cycle_amplitudes=cycle_amplitudes,
        metadata=metadata,
    )
