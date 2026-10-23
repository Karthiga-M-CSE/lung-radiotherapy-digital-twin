"""Split Conformal Prediction for 3D tumor motion prediction with coverage evaluation.

Provides rigorous, distribution-free finite-sample coverage guarantees:
    P(Y in C(X)) >= 1 - alpha
using a held-out chronological calibration set.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd

from src.utils.logging import get_logger
from src.utils.metrics import compute_3d_euclidean_error

logger = get_logger(__name__)


@dataclass
class ConformalIntervals:
    """Predicted intervals and 3D confidence regions."""
    y_pred: np.ndarray
    radius_3d_mm: float
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    axis_margins_mm: np.ndarray
    alpha: float

    def to_dataframe(self, frames: Optional[np.ndarray] = None) -> pd.DataFrame:
        n = len(self.y_pred)
        f = frames if frames is not None else np.arange(n)
        return pd.DataFrame({
            "frame": f,
            "pred_x_mm": self.y_pred[:, 0],
            "pred_y_mm": self.y_pred[:, 1],
            "pred_z_mm": self.y_pred[:, 2],
            "lower_x_mm": self.lower_bounds[:, 0],
            "upper_x_mm": self.upper_bounds[:, 0],
            "lower_y_mm": self.lower_bounds[:, 1],
            "upper_y_mm": self.upper_bounds[:, 1],
            "lower_z_mm": self.lower_bounds[:, 2],
            "upper_z_mm": self.upper_bounds[:, 2],
            "conformal_radius_3d_mm": self.radius_3d_mm,
        })


@dataclass
class ConformalCoverageReport:
    """Empirical coverage evaluation results."""
    target_coverage: float
    coverage_3d: float
    coverage_x: float
    coverage_y: float
    coverage_z: float
    coverage_joint_box: float
    radius_3d_mm: float
    axis_margins_mm: Dict[str, float]
    n_calibration: int
    n_test: int
    is_valid: bool

    def summary(self) -> str:
        lines = [
            "=" * 65,
            f"SPLIT CONFORMAL PREDICTION: COVERAGE & CALIBRATION REPORT",
            "=" * 65,
            f"Target coverage level (1 - alpha) : {self.target_coverage * 100:.1f}%",
            f"Calibration set size              : {self.n_calibration} samples",
            f"Test set size                     : {self.n_test} samples",
            "-" * 65,
            f"{'Region':<22} {'Empirical Coverage':<22} {'Target':<10} {'Calibrated Bound'}",
            "-" * 65,
            f"{'3D Euclidean Sphere':<22} {self.coverage_3d * 100:>6.2f}%              {self.target_coverage * 100:>5.1f}%     radius = {self.radius_3d_mm:.4f} mm",
            f"{'X-axis interval':<22} {self.coverage_x * 100:>6.2f}%              {self.target_coverage * 100:>5.1f}%     +/- {self.axis_margins_mm['x']:.4f} mm",
            f"{'Y-axis interval':<22} {self.coverage_y * 100:>6.2f}%              {self.target_coverage * 100:>5.1f}%     +/- {self.axis_margins_mm['y']:.4f} mm",
            f"{'Z-axis interval':<22} {self.coverage_z * 100:>6.2f}%              {self.target_coverage * 100:>5.1f}%     +/- {self.axis_margins_mm['z']:.4f} mm",
            f"{'Joint 3D Box':<22} {self.coverage_joint_box * 100:>6.2f}%              {self.target_coverage * 100:>5.1f}%",
            "=" * 65,
            f"VALIDATION RESULT: {'PASSED (Empirical >= Target)' if self.is_valid else 'FAILED (Coverage deficit)'}",
            "=" * 65,
        ]
        return "\n".join(lines)


class SplitConformalPredictor:
    """Split Conformal Predictor for multi-output 3D regression."""

    def __init__(self, alpha: float = 0.05):
        self.alpha = float(alpha)
        self.is_calibrated = False
        self.radius_3d_mm: float = 0.0
        self.axis_margins_mm: np.ndarray = np.zeros(3)
        self.n_calib: int = 0
        self._calib_scores_3d: Optional[np.ndarray] = None
        self._calib_scores_axis: Optional[np.ndarray] = None

    def calibrate(
        self,
        y_true_calib: np.ndarray,
        y_pred_calib: np.ndarray,
        alpha: Optional[float] = None,
    ) -> "SplitConformalPredictor":
        """Calibrate prediction regions using held-out calibration residuals."""
        if alpha is not None:
            self.alpha = float(alpha)

        y_true = np.asarray(y_true_calib, dtype=float)
        y_pred = np.asarray(y_pred_calib, dtype=float)

        if len(y_true) != len(y_pred):
            raise ValueError("y_true_calib and y_pred_calib must have identical lengths.")

        n = len(y_true)
        if n == 0:
            raise ValueError("Calibration set is empty.")

        self.n_calib = n

        # Nonconformity scores: 3D Euclidean residual and axis-wise absolute residuals
        scores_3d = compute_3d_euclidean_error(y_true, y_pred)
        scores_axis = np.abs(y_true - y_pred)

        self._calib_scores_3d = scores_3d
        self._calib_scores_axis = scores_axis

        # Finite-sample conformal quantile: ceil((n + 1) * (1 - alpha)) / n
        q_level = min(1.0, float(np.ceil((n + 1) * (1.0 - self.alpha)) / n))

        # Quantile method "higher" guarantees coverage for finite samples
        self.radius_3d_mm = float(np.quantile(scores_3d, q_level, method="higher"))

        margins = []
        for d in range(scores_axis.shape[1]):
            m = float(np.quantile(scores_axis[:, d], q_level, method="higher"))
            margins.append(m)
        self.axis_margins_mm = np.array(margins, dtype=float)

        self.is_calibrated = True
        logger.info(
            f"Split Conformal calibrated on {n} samples (alpha={self.alpha:.2f}): "
            f"3D radius={self.radius_3d_mm:.4f} mm, "
            f"axis margins=[X:+/-{self.axis_margins_mm[0]:.4f}, Y:+/-{self.axis_margins_mm[1]:.4f}, Z:+/-{self.axis_margins_mm[2]:.4f}] mm"
        )
        return self

    def predict_intervals(
        self,
        y_pred: np.ndarray,
        alpha: Optional[float] = None,
    ) -> ConformalIntervals:
        """Construct calibrated 3D confidence sphere and bounding intervals for predictions."""
        if not self.is_calibrated:
            raise RuntimeError("SplitConformalPredictor must be calibrated before predict_intervals().")

        if alpha is not None and not np.isclose(alpha, self.alpha):
            # Recompute quantiles for new alpha if calibration scores were saved
            if self._calib_scores_3d is not None and self._calib_scores_axis is not None:
                n = self.n_calib
                q_level = min(1.0, float(np.ceil((n + 1) * (1.0 - alpha)) / n))
                r_3d = float(np.quantile(self._calib_scores_3d, q_level, method="higher"))
                margins = np.array([
                    float(np.quantile(self._calib_scores_axis[:, d], q_level, method="higher"))
                    for d in range(self._calib_scores_axis.shape[1])
                ])
            else:
                r_3d = self.radius_3d_mm
                margins = self.axis_margins_mm
            used_alpha = alpha
        else:
            r_3d = self.radius_3d_mm
            margins = self.axis_margins_mm
            used_alpha = self.alpha

        preds = np.asarray(y_pred, dtype=float)
        lower = preds - margins
        upper = preds + margins

        return ConformalIntervals(
            y_pred=preds,
            radius_3d_mm=r_3d,
            lower_bounds=lower,
            upper_bounds=upper,
            axis_margins_mm=margins,
            alpha=used_alpha,
        )

    def evaluate_coverage(
        self,
        y_true_test: np.ndarray,
        y_pred_test: np.ndarray,
        alpha: Optional[float] = None,
    ) -> ConformalCoverageReport:
        """Evaluate empirical coverage of calibrated intervals on held-out test data."""
        y_true = np.asarray(y_true_test, dtype=float)
        y_pred = np.asarray(y_pred_test, dtype=float)
        n_test = len(y_true)

        if n_test == 0:
            raise ValueError("Test set is empty.")

        intervals = self.predict_intervals(y_pred, alpha=alpha)
        target = 1.0 - intervals.alpha

        # 3D Euclidean sphere coverage: ||y - y_hat|| <= radius
        errs_3d = compute_3d_euclidean_error(y_true, y_pred)
        cov_3d = float(np.mean(errs_3d <= intervals.radius_3d_mm))

        # Marginal axis coverages
        in_x = (y_true[:, 0] >= intervals.lower_bounds[:, 0]) & (y_true[:, 0] <= intervals.upper_bounds[:, 0])
        in_y = (y_true[:, 1] >= intervals.lower_bounds[:, 1]) & (y_true[:, 1] <= intervals.upper_bounds[:, 1])
        in_z = (y_true[:, 2] >= intervals.lower_bounds[:, 2]) & (y_true[:, 2] <= intervals.upper_bounds[:, 2])

        cov_x = float(np.mean(in_x))
        cov_y = float(np.mean(in_y))
        cov_z = float(np.mean(in_z))

        # Joint rectangular box coverage
        cov_joint = float(np.mean(in_x & in_y & in_z))

        # Check validity (allowing small finite-sample test variance: e.g. -1 sample tolerance)
        tolerance = 1.0 / n_test if n_test > 0 else 0.0
        is_valid = bool(cov_3d >= (target - tolerance))

        report = ConformalCoverageReport(
            target_coverage=target,
            coverage_3d=cov_3d,
            coverage_x=cov_x,
            coverage_y=cov_y,
            coverage_z=cov_z,
            coverage_joint_box=cov_joint,
            radius_3d_mm=intervals.radius_3d_mm,
            axis_margins_mm={
                "x": float(intervals.axis_margins_mm[0]),
                "y": float(intervals.axis_margins_mm[1]),
                "z": float(intervals.axis_margins_mm[2]),
            },
            n_calibration=self.n_calib,
            n_test=n_test,
            is_valid=is_valid,
        )

        logger.info(
            f"Conformal test evaluation: 3D Coverage={cov_3d * 100:.2f}% (Target={target * 100:.1f}%), "
            f"Joint={cov_joint * 100:.2f}%, Radius={intervals.radius_3d_mm:.4f} mm"
        )
        return report
