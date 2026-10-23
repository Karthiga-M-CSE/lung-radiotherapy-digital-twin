"""Chronological train/calibration/test splitting for respiratory time-series.

Guarantees strict temporal ordering without data leakage:
    Train set < Calibration set < Test set
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd

from src.config import (
    TRAIN_START_FRAME,
    TRAIN_END_FRAME,
    CALIB_START_FRAME,
    CALIB_END_FRAME,
    TEST_START_FRAME,
    TEST_END_FRAME,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ChronologicalSplit:
    """Holds data partitions resulting from a chronological split."""
    X_train: np.ndarray
    y_train: np.ndarray
    train_frames: np.ndarray
    X_calib: np.ndarray
    y_calib: np.ndarray
    calib_frames: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    test_frames: np.ndarray
    train_indices: np.ndarray
    calib_indices: np.ndarray
    test_indices: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        """Verify strict chronological ordering and non-overlapping partitions."""
        n_tr = len(self.X_train)
        n_ca = len(self.X_calib)
        n_te = len(self.X_test)

        if n_tr == 0:
            raise ValueError("Train partition is empty.")
        if n_ca == 0:
            raise ValueError("Calibration partition is empty.")
        if n_te == 0:
            raise ValueError("Test partition is empty.")

        s_tr = set(self.train_frames)
        s_ca = set(self.calib_frames)
        s_te = set(self.test_frames)

        if s_tr.intersection(s_ca):
            raise ValueError("Train and Calibration frame sets overlap.")
        if s_ca.intersection(s_te):
            raise ValueError("Calibration and Test frame sets overlap.")
        if s_tr.intersection(s_te):
            raise ValueError("Train and Test frame sets overlap.")

        max_tr = float(np.max(self.train_frames))
        min_ca = float(np.min(self.calib_frames))
        max_ca = float(np.max(self.calib_frames))
        min_te = float(np.min(self.test_frames))

        if max_tr >= min_ca:
            raise ValueError(
                f"Temporal leakage: max train frame ({max_tr}) >= min calib frame ({min_ca})"
            )
        if max_ca >= min_te:
            raise ValueError(
                f"Temporal leakage: max calib frame ({max_ca}) >= min test frame ({min_te})"
            )

    def summary(self) -> str:
        total = len(self.X_train) + len(self.X_calib) + len(self.X_test)
        lines = [
            "=" * 60,
            "CHRONOLOGICAL TRAIN / CALIBRATION / TEST SPLIT",
            "=" * 60,
            f"{'Partition':<15} {'Samples':<10} {'Fraction':<10} {'Frame Range':<15}",
            "-" * 60,
            f"{'Train':<15} {len(self.X_train):<10} {len(self.X_train)/total*100:>5.1f}%     {int(np.min(self.train_frames))} -> {int(np.max(self.train_frames))}",
            f"{'Calibration':<15} {len(self.X_calib):<10} {len(self.X_calib)/total*100:>5.1f}%     {int(np.min(self.calib_frames))} -> {int(np.max(self.calib_frames))}",
            f"{'Test':<15} {len(self.X_test):<10} {len(self.X_test)/total*100:>5.1f}%     {int(np.min(self.test_frames))} -> {int(np.max(self.test_frames))}",
            "=" * 60,
        ]
        return "\n".join(lines)


def chronological_split(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    frames: Optional[np.ndarray] = None,
    train_frac: float = 0.60,
    calib_frac: float = 0.20,
    test_frac: float = 0.20,
) -> ChronologicalSplit:
    """Split data into chronological train, calibration, and test sets by fractions.

    Args:
        X: Feature matrix of shape (N, D).
        y: Optional target array of shape (N, ...).
        frames: Optional frame numbers of shape (N,). If None, 0..N-1 used.
        train_frac: Fraction of samples for training.
        calib_frac: Fraction of samples for calibration.
        test_frac: Fraction of samples for testing.

    Returns:
        ChronologicalSplit object.
    """
    total_frac = train_frac + calib_frac + test_frac
    if not np.isclose(total_frac, 1.0, atol=1e-3):
        raise ValueError(f"Fractions must sum to 1.0 (got {total_frac:.4f})")

    X_arr = np.asarray(X)
    n = len(X_arr)
    if n < 3:
        raise ValueError("Need at least 3 samples to perform a 3-way split.")

    y_arr = np.asarray(y) if y is not None else np.zeros((n, 1))
    f_arr = np.asarray(frames) if frames is not None else np.arange(n)

    idx_train_end = max(1, int(round(n * train_frac)))
    idx_calib_end = max(idx_train_end + 1, int(round(n * (train_frac + calib_frac))))
    idx_calib_end = min(idx_calib_end, n - 1)

    train_idx = np.arange(0, idx_train_end)
    calib_idx = np.arange(idx_train_end, idx_calib_end)
    test_idx = np.arange(idx_calib_end, n)

    return ChronologicalSplit(
        X_train=X_arr[train_idx],
        y_train=y_arr[train_idx],
        train_frames=f_arr[train_idx],
        X_calib=X_arr[calib_idx],
        y_calib=y_arr[calib_idx],
        calib_frames=f_arr[calib_idx],
        X_test=X_arr[test_idx],
        y_test=y_arr[test_idx],
        test_frames=f_arr[test_idx],
        train_indices=train_idx,
        calib_indices=calib_idx,
        test_indices=test_idx,
        metadata={"method": "fraction", "train_frac": train_frac, "calib_frac": calib_frac, "test_frac": test_frac},
    )


def chronological_frame_split(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    frames: Optional[np.ndarray] = None,
    train_range: Optional[Tuple[int, int]] = None,
    calib_range: Optional[Tuple[int, int]] = None,
    test_range: Optional[Tuple[int, int]] = None,
) -> ChronologicalSplit:
    """Split data using explicit frame boundary intervals.

    Defaults to config settings:
        train_range: (TRAIN_START_FRAME, TRAIN_END_FRAME)
        calib_range: (CALIB_START_FRAME, CALIB_END_FRAME)
        test_range:  (TEST_START_FRAME, TEST_END_FRAME)
    """
    tr_r = train_range or (TRAIN_START_FRAME, TRAIN_END_FRAME)
    ca_r = calib_range or (CALIB_START_FRAME, CALIB_END_FRAME)
    te_r = test_range or (TEST_START_FRAME, TEST_END_FRAME)

    X_arr = np.asarray(X)
    n = len(X_arr)
    y_arr = np.asarray(y) if y is not None else np.zeros((n, 1))
    f_arr = np.asarray(frames) if frames is not None else np.arange(n)

    train_mask = (f_arr >= tr_r[0]) & (f_arr <= tr_r[1])
    calib_mask = (f_arr >= ca_r[0]) & (f_arr <= ca_r[1])
    test_mask = (f_arr >= te_r[0]) & (f_arr <= te_r[1])

    train_idx = np.where(train_mask)[0]
    calib_idx = np.where(calib_mask)[0]
    test_idx = np.where(test_mask)[0]

    return ChronologicalSplit(
        X_train=X_arr[train_idx],
        y_train=y_arr[train_idx],
        train_frames=f_arr[train_idx],
        X_calib=X_arr[calib_idx],
        y_calib=y_arr[calib_idx],
        calib_frames=f_arr[calib_idx],
        X_test=X_arr[test_idx],
        y_test=y_arr[test_idx],
        test_frames=f_arr[test_idx],
        train_indices=train_idx,
        calib_indices=calib_idx,
        test_indices=test_idx,
        metadata={"method": "frame_range", "train_range": tr_r, "calib_range": ca_r, "test_range": te_r},
    )


def chronological_cycle_split(
    df: pd.DataFrame,
    cycle_col: str = "cycle_id",
    train_cycles: Sequence[int] = range(1, 11),
    calib_cycles: Sequence[int] = range(11, 14),
    test_cycles: Sequence[int] = range(14, 18),
    feature_cols: Optional[Sequence[str]] = None,
    target_cols: Sequence[str] = ("x_mm", "y_mm", "z_mm"),
    frame_col: str = "frame",
) -> ChronologicalSplit:
    """Split a DataFrame containing respiratory cycles into chronological cycle blocks."""
    if cycle_col not in df.columns:
        raise ValueError(f"Cycle column '{cycle_col}' not found in DataFrame.")

    tr_set = set(train_cycles)
    ca_set = set(calib_cycles)
    te_set = set(test_cycles)

    if tr_set.intersection(ca_set) or ca_set.intersection(te_set) or tr_set.intersection(te_set):
        raise ValueError("Cycle IDs between train, calibration, and test must be disjoint.")

    if max(tr_set) >= min(ca_set):
        raise ValueError("Train cycles must precede calibration cycles.")
    if max(ca_set) >= min(te_set):
        raise ValueError("Calibration cycles must precede test cycles.")

    df_clean = df.sort_values(frame_col).reset_index(drop=True)

    train_mask = df_clean[cycle_col].isin(tr_set)
    calib_mask = df_clean[cycle_col].isin(ca_set)
    test_mask = df_clean[cycle_col].isin(te_set)

    f_cols = list(feature_cols) if feature_cols is not None else [c for c in df_clean.columns if c not in list(target_cols) + [frame_col, cycle_col]]
    t_cols = list(target_cols)

    X = df_clean[f_cols].values.astype(float)
    y = df_clean[t_cols].values.astype(float)
    frames = df_clean[frame_col].values

    train_idx = np.where(train_mask)[0]
    calib_idx = np.where(calib_mask)[0]
    test_idx = np.where(test_mask)[0]

    return ChronologicalSplit(
        X_train=X[train_idx],
        y_train=y[train_idx],
        train_frames=frames[train_idx],
        X_calib=X[calib_idx],
        y_calib=y[calib_idx],
        calib_frames=frames[calib_idx],
        X_test=X[test_idx],
        y_test=y[test_idx],
        test_frames=frames[test_idx],
        train_indices=train_idx,
        calib_indices=calib_idx,
        test_indices=test_idx,
        metadata={"method": "cycles", "train_cycles": list(tr_set), "calib_cycles": list(ca_set), "test_cycles": list(te_set)},
    )
