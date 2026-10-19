from typing import Dict, Union
import numpy as np


def compute_prediction_errors(
    actual: np.ndarray,
    predicted: np.ndarray
) -> Dict[str, Union[np.ndarray, float]]:
    diff = actual - predicted
    abs_diff = np.abs(diff)
    error_3d = np.linalg.norm(diff, axis=1)

    rmse_3d = float(np.sqrt(np.mean(error_3d ** 2)))
    mae_3d = float(np.mean(error_3d))

    return {
        "diff": diff,
        "abs_diff": abs_diff,
        "error_3d": error_3d,
        "mean_3d": float(np.mean(error_3d)),
        "median_3d": float(np.median(error_3d)),
        "p90_3d": float(np.percentile(error_3d, 90)),
        "p95_3d": float(np.percentile(error_3d, 95)),
        "p99_3d": float(np.percentile(error_3d, 99)),
        "max_3d": float(np.max(error_3d)),
        "rmse_3d": rmse_3d,
        "mae_3d": mae_3d,
        "mean_x": float(np.mean(abs_diff[:, 0])),
        "mean_y": float(np.mean(abs_diff[:, 1])),
        "mean_z": float(np.mean(abs_diff[:, 2])),
        "p95_x": float(np.percentile(abs_diff[:, 0], 95)),
        "p95_y": float(np.percentile(abs_diff[:, 1], 95)),
        "p95_z": float(np.percentile(abs_diff[:, 2], 95)),
    }
