from typing import Dict, Optional, Sequence, Union
import numpy as np


def compute_3d_euclidean_error(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> np.ndarray:
    diff = np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)
    return np.linalg.norm(diff, axis=-1)


def compute_mae(
    actual: np.ndarray,
    predicted: np.ndarray,
    axis: Optional[int] = None,
) -> Union[float, np.ndarray]:
    diff = np.abs(np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float))
    return float(np.mean(diff)) if axis is None else np.mean(diff, axis=axis)


def compute_rmse(
    actual: np.ndarray,
    predicted: np.ndarray,
    axis: Optional[int] = None,
) -> Union[float, np.ndarray]:
    diff = np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)
    sq_err = diff ** 2
    return float(np.sqrt(np.mean(sq_err))) if axis is None else np.sqrt(np.mean(sq_err, axis=axis))


def compute_percentile_errors(
    errors: np.ndarray,
    percentiles: Sequence[float] = (50.0, 90.0, 95.0, 99.0),
) -> Dict[str, float]:
    arr = np.asarray(errors, dtype=float).ravel()
    result = {}
    for p in percentiles:
        key = f"p{int(p) if p.is_integer() else p}"
        result[key] = float(np.percentile(arr, p))
    return result


def compute_prediction_errors(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> Dict[str, Union[np.ndarray, float]]:
    act = np.asarray(actual, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    diff = act - pred
    abs_diff = np.abs(diff)
    error_3d = np.linalg.norm(diff, axis=1)

    rmse_3d = float(np.sqrt(np.mean(error_3d ** 2)))
    mae_3d = float(np.mean(error_3d))
    p_dict = compute_percentile_errors(error_3d, (50, 90, 95, 99))

    res: Dict[str, Union[np.ndarray, float]] = {
        "diff": diff,
        "abs_diff": abs_diff,
        "error_3d": error_3d,
        "mean_3d": mae_3d,
        "mae_3d": mae_3d,
        "rmse_3d": rmse_3d,
        "median_3d": p_dict["p50"],
        "p50_3d": p_dict["p50"],
        "p90_3d": p_dict["p90"],
        "p95_3d": p_dict["p95"],
        "p99_3d": p_dict["p99"],
        "max_3d": float(np.max(error_3d)),
        "mean_x": float(np.mean(abs_diff[:, 0])),
        "mean_y": float(np.mean(abs_diff[:, 1])),
        "mean_z": float(np.mean(abs_diff[:, 2])),
        "mae_x": float(np.mean(abs_diff[:, 0])),
        "mae_y": float(np.mean(abs_diff[:, 1])),
        "mae_z": float(np.mean(abs_diff[:, 2])),
        "rmse_x": float(np.sqrt(np.mean(diff[:, 0] ** 2))),
        "rmse_y": float(np.sqrt(np.mean(diff[:, 1] ** 2))),
        "rmse_z": float(np.sqrt(np.mean(diff[:, 2] ** 2))),
        "p90_x": float(np.percentile(abs_diff[:, 0], 90)),
        "p90_y": float(np.percentile(abs_diff[:, 1], 90)),
        "p90_z": float(np.percentile(abs_diff[:, 2], 90)),
        "p95_x": float(np.percentile(abs_diff[:, 0], 95)),
        "p95_y": float(np.percentile(abs_diff[:, 1], 95)),
        "p95_z": float(np.percentile(abs_diff[:, 2], 95)),
        "p99_x": float(np.percentile(abs_diff[:, 0], 99)),
        "p99_y": float(np.percentile(abs_diff[:, 1], 99)),
        "p99_z": float(np.percentile(abs_diff[:, 2], 99)),
    }
    return res
