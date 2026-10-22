from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

from src.models.conformal import (
    ConformalCoverageReport,
    ConformalIntervals,
    SplitConformalPredictor,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


def evaluate_safety_gate(
    errors_3d: np.ndarray,
    threshold_percentile: float = 95.0,
    preset_threshold: Optional[float] = None,
) -> Tuple[float, np.ndarray, Dict[str, Any]]:
    threshold_mm = (
        preset_threshold
        if preset_threshold is not None
        else float(np.percentile(errors_3d, threshold_percentile))
    )

    status = np.where(errors_3d <= threshold_mm, "SAFE / CONTINUE", "FLAG / REVIEW")
    safe_count = int(np.sum(status == "SAFE / CONTINUE"))
    flag_count = int(np.sum(status == "FLAG / REVIEW"))
    n = len(errors_3d)

    summary = {
        "threshold_mm": threshold_mm,
        "mean_error_mm": float(np.mean(errors_3d)),
        "p50_error_mm": float(np.percentile(errors_3d, 50)),
        "p90_error_mm": float(np.percentile(errors_3d, 90)),
        "p95_error_mm": float(np.percentile(errors_3d, 95)),
        "p99_error_mm": float(np.percentile(errors_3d, 99)),
        "max_error_mm": float(np.max(errors_3d)),
        "safe_count": safe_count,
        "flag_count": flag_count,
        "safe_pct": (safe_count / n * 100) if n > 0 else 0.0,
        "flag_pct": (flag_count / n * 100) if n > 0 else 0.0,
    }

    logger.info(
        f"Safety Gate evaluated: Threshold={threshold_mm:.4f} mm | "
        f"SAFE={safe_count} ({summary['safe_pct']:.1f}%) | "
        f"REVIEW={flag_count} ({summary['flag_pct']:.1f}%)"
    )
    return threshold_mm, status, summary


def evaluate_conformal_safety_gate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    conformal_predictor: SplitConformalPredictor,
    alpha: float = 0.05,
) -> Tuple[ConformalCoverageReport, np.ndarray, Dict[str, Any]]:
    report = conformal_predictor.evaluate_coverage(y_true, y_pred, alpha=alpha)
    diff = np.linalg.norm(np.asarray(y_true) - np.asarray(y_pred), axis=-1)
    status = np.where(diff <= report.radius_3d_mm, "SAFE / CONTINUE", "FLAG / REVIEW")

    summary = {
        "conformal_radius_3d_mm": report.radius_3d_mm,
        "target_coverage": report.target_coverage,
        "empirical_coverage_3d": report.coverage_3d,
        "is_valid_coverage": report.is_valid,
        "safe_count": int(np.sum(status == "SAFE / CONTINUE")),
        "flag_count": int(np.sum(status == "FLAG / REVIEW")),
    }
    return report, status, summary
