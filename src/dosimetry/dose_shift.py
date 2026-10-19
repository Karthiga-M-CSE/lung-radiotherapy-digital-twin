from typing import Dict, Any, List, Optional
import numpy as np

from src.utils.geometry import shift_mask
from src.utils.logging import get_logger

logger = get_logger(__name__)


def compute_mask_dose_stats(dose_grid: np.ndarray, binary_mask: np.ndarray) -> Optional[Dict[str, float]]:
    values = dose_grid[binary_mask]
    if len(values) == 0:
        return None
    return {
        "voxels": int(len(values)),
        "mean": float(values.mean()),
        "D95": float(np.percentile(values, 5)),
        "max": float(values.max()),
    }


def evaluate_motion_shifts(
    dose_grid: np.ndarray,
    baseline_mask: np.ndarray,
    shifts_mm: List[float],
    sx: float = 1.0,
    sy: float = 1.0
) -> Dict[str, List[Dict[str, Any]]]:
    results = {"X": [], "Y": [], "Z": []}

    for s in shifts_mm:
        mask_x = shift_mask(baseline_mask, dx_mm=s, dy_mm=0.0, dz_mm=0.0, sx=sx, sy=sy)
        stats_x = compute_mask_dose_stats(dose_grid, mask_x)
        if stats_x:
            results["X"].append({"shift_mm": s, **stats_x})

        mask_y = shift_mask(baseline_mask, dx_mm=0.0, dy_mm=s, dz_mm=0.0, sx=sx, sy=sy)
        stats_y = compute_mask_dose_stats(dose_grid, mask_y)
        if stats_y:
            results["Y"].append({"shift_mm": s, **stats_y})

        mask_z = shift_mask(baseline_mask, dx_mm=0.0, dy_mm=0.0, dz_mm=s, sx=sx, sy=sy)
        stats_z = compute_mask_dose_stats(dose_grid, mask_z)
        if stats_z:
            results["Z"].append({"shift_mm": s, **stats_z})

    return results
