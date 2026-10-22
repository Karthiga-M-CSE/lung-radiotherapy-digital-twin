from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.utils.logging import get_logger
from src.utils.metrics import compute_prediction_errors

logger = get_logger(__name__)


@dataclass
class ModelEvaluationMetrics:
    model_name: str
    n_samples: int
    mean_3d: float
    median_3d: float
    p90_3d: float
    p95_3d: float
    p99_3d: float
    max_3d: float
    rmse_3d: float
    mae_3d: float
    mean_x: float
    mean_y: float
    mean_z: float
    p95_x: float
    p95_y: float
    p95_z: float
    p90_x: float = 0.0
    p90_y: float = 0.0
    p90_z: float = 0.0
    p99_x: float = 0.0
    p99_y: float = 0.0
    p99_z: float = 0.0
    rmse_x: float = 0.0
    rmse_y: float = 0.0
    rmse_z: float = 0.0
    mae_x: float = 0.0
    mae_y: float = 0.0
    mae_z: float = 0.0

    def to_dict(self) -> Dict[str, Union[str, int, float]]:
        return {
            "model_name": self.model_name,
            "n_samples": self.n_samples,
            "mean_3d_mm": self.mean_3d,
            "median_3d_mm": self.median_3d,
            "p90_3d_mm": self.p90_3d,
            "p95_3d_mm": self.p95_3d,
            "p99_3d_mm": self.p99_3d,
            "max_3d_mm": self.max_3d,
            "rmse_3d_mm": self.rmse_3d,
            "mae_3d_mm": self.mae_3d,
            "mean_x_mm": self.mean_x,
            "mean_y_mm": self.mean_y,
            "mean_z_mm": self.mean_z,
            "mae_x_mm": self.mae_x or self.mean_x,
            "mae_y_mm": self.mae_y or self.mean_y,
            "mae_z_mm": self.mae_z or self.mean_z,
            "rmse_x_mm": self.rmse_x,
            "rmse_y_mm": self.rmse_y,
            "rmse_z_mm": self.rmse_z,
            "p90_x_mm": self.p90_x,
            "p90_y_mm": self.p90_y,
            "p90_z_mm": self.p90_z,
            "p95_x_mm": self.p95_x,
            "p95_y_mm": self.p95_y,
            "p95_z_mm": self.p95_z,
            "p99_x_mm": self.p99_x,
            "p99_y_mm": self.p99_y,
            "p99_z_mm": self.p99_z,
        }


@dataclass
class ModelComparisonResult:
    comparison_df: pd.DataFrame
    baseline_metrics: ModelEvaluationMetrics
    candidate_metrics: ModelEvaluationMetrics
    p95_reduction_pct: float
    winner: str
    common_frames: int
    frame_range: Tuple[int, int]

    def summary(self) -> str:
        b = self.baseline_metrics
        c = self.candidate_metrics

        lines = [
            "=" * 70,
            f"HELD-OUT PERFORMANCE COMPARISON: {b.model_name} vs {c.model_name}",
            "=" * 70,
            f"Common frames evaluated: {self.common_frames} (Frames {self.frame_range[0]} -> {self.frame_range[1]})",
            "",
            f"{'Metric':<24} {b.model_name:<14} {c.model_name:<14}",
            "-" * 55,
            f"{'Mean 3D error':<24} {b.mean_3d:.4f} mm     {c.mean_3d:.4f} mm",
            f"{'Median':<24} {b.median_3d:.4f} mm     {c.median_3d:.4f} mm",
            f"{'P90':<24} {b.p90_3d:.4f} mm     {c.p90_3d:.4f} mm",
            f"{'P95':<24} {b.p95_3d:.4f} mm     {c.p95_3d:.4f} mm",
            f"{'P99':<24} {b.p99_3d:.4f} mm     {c.p99_3d:.4f} mm",
            f"{'Maximum':<24} {b.max_3d:.4f} mm     {c.max_3d:.4f} mm",
            f"{'RMSE 3D':<24} {b.rmse_3d:.4f} mm     {c.rmse_3d:.4f} mm",
            "",
            "=" * 70,
            "P95 IMPROVEMENT",
            "=" * 70,
            f"{b.model_name} P95 : {b.p95_3d:.4f} mm",
            f"{c.model_name} P95  : {c.p95_3d:.4f} mm",
            f"P95 reduction : {self.p95_reduction_pct:.2f}%",
            f"WINNER        : {self.winner}",
            "=" * 70,
        ]
        return "\n".join(lines)


def evaluate_model_predictions(
    actual: np.ndarray,
    predicted: np.ndarray,
    frames: Optional[np.ndarray] = None,
    model_name: str = "Model",
) -> Tuple[pd.DataFrame, ModelEvaluationMetrics]:
    act = np.asarray(actual, dtype=float)
    pred = np.asarray(predicted, dtype=float)

    if len(act) != len(pred):
        raise ValueError(f"Actual ({len(act)}) and Predicted ({len(pred)}) length mismatch.")

    err_dict = compute_prediction_errors(act, pred)
    n = len(act)
    frame_indices = np.asarray(frames) if frames is not None else np.arange(n)

    df_results = pd.DataFrame({
        "frame": frame_indices,
        "actual_x_mm": act[:, 0],
        "actual_y_mm": act[:, 1],
        "actual_z_mm": act[:, 2],
        "predicted_x_mm": pred[:, 0],
        "predicted_y_mm": pred[:, 1],
        "predicted_z_mm": pred[:, 2],
        "error_x_mm": err_dict["abs_diff"][:, 0],
        "error_y_mm": err_dict["abs_diff"][:, 1],
        "error_z_mm": err_dict["abs_diff"][:, 2],
        "error_3d_mm": err_dict["error_3d"],
    })

    metrics = ModelEvaluationMetrics(
        model_name=model_name,
        n_samples=n,
        mean_3d=float(err_dict["mean_3d"]),
        median_3d=float(err_dict["median_3d"]),
        p90_3d=float(err_dict["p90_3d"]),
        p95_3d=float(err_dict["p95_3d"]),
        p99_3d=float(err_dict["p99_3d"]),
        max_3d=float(err_dict["max_3d"]),
        rmse_3d=float(err_dict["rmse_3d"]),
        mae_3d=float(err_dict["mae_3d"]),
        mean_x=float(err_dict["mean_x"]),
        mean_y=float(err_dict["mean_y"]),
        mean_z=float(err_dict["mean_z"]),
        p95_x=float(err_dict["p95_x"]),
        p95_y=float(err_dict["p95_y"]),
        p95_z=float(err_dict["p95_z"]),
        p90_x=float(err_dict.get("p90_x", 0.0)),
        p90_y=float(err_dict.get("p90_y", 0.0)),
        p90_z=float(err_dict.get("p90_z", 0.0)),
        p99_x=float(err_dict.get("p99_x", 0.0)),
        p99_y=float(err_dict.get("p99_y", 0.0)),
        p99_z=float(err_dict.get("p99_z", 0.0)),
        rmse_x=float(err_dict.get("rmse_x", 0.0)),
        rmse_y=float(err_dict.get("rmse_y", 0.0)),
        rmse_z=float(err_dict.get("rmse_z", 0.0)),
        mae_x=float(err_dict.get("mae_x", err_dict["mean_x"])),
        mae_y=float(err_dict.get("mae_y", err_dict["mean_y"])),
        mae_z=float(err_dict.get("mae_z", err_dict["mean_z"])),
    )

    return df_results, metrics


def compare_motion_models(
    baseline_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    baseline_name: str = "Kalman",
    candidate_name: str = "Ridge",
    frame_range: Optional[Tuple[int, int]] = None,
) -> ModelComparisonResult:
    required_cols = [
        "frame",
        "actual_x_mm", "actual_y_mm", "actual_z_mm",
        "predicted_x_mm", "predicted_y_mm", "predicted_z_mm",
    ]

    for name, df_in in [(baseline_name, baseline_df), (candidate_name, candidate_df)]:
        missing = [c for c in required_cols if c not in df_in.columns]
        if missing:
            raise ValueError(f"Missing columns in {name} results: {missing}")

    b_df = baseline_df.copy()
    c_df = candidate_df.copy()

    if frame_range is not None:
        start_f, end_f = frame_range
        b_df = b_df[(b_df["frame"] >= start_f) & (b_df["frame"] <= end_f)]
        c_df = c_df[(c_df["frame"] >= start_f) & (c_df["frame"] <= end_f)]

    b_sub = b_df[required_cols].rename(columns={
        "actual_x_mm": f"{baseline_name.lower()}_actual_x_mm",
        "actual_y_mm": f"{baseline_name.lower()}_actual_y_mm",
        "actual_z_mm": f"{baseline_name.lower()}_actual_z_mm",
        "predicted_x_mm": f"{baseline_name.lower()}_predicted_x_mm",
        "predicted_y_mm": f"{baseline_name.lower()}_predicted_y_mm",
        "predicted_z_mm": f"{baseline_name.lower()}_predicted_z_mm",
    })

    c_sub = c_df[required_cols].rename(columns={
        "predicted_x_mm": f"{candidate_name.lower()}_predicted_x_mm",
        "predicted_y_mm": f"{candidate_name.lower()}_predicted_y_mm",
        "predicted_z_mm": f"{candidate_name.lower()}_predicted_z_mm",
    })

    comparison = pd.merge(c_sub, b_sub, on="frame", how="inner")
    if len(comparison) == 0:
        raise RuntimeError("No common frames found between candidate and baseline models.")

    b_frames = set(b_df["frame"])
    c_frames = set(c_df["frame"])
    common_frames = set(comparison["frame"])

    if common_frames != b_frames.intersection(c_frames):
        raise RuntimeError("Frame matching error: merged frames do not match set intersection.")

    act_pos = comparison[["actual_x_mm", "actual_y_mm", "actual_z_mm"]].values

    b_pred = comparison[[
        f"{baseline_name.lower()}_predicted_x_mm",
        f"{baseline_name.lower()}_predicted_y_mm",
        f"{baseline_name.lower()}_predicted_z_mm",
    ]].values
    _, b_metrics = evaluate_model_predictions(act_pos, b_pred, frames=comparison["frame"].values, model_name=baseline_name)

    c_pred = comparison[[
        f"{candidate_name.lower()}_predicted_x_mm",
        f"{candidate_name.lower()}_predicted_y_mm",
        f"{candidate_name.lower()}_predicted_z_mm",
    ]].values
    _, c_metrics = evaluate_model_predictions(act_pos, c_pred, frames=comparison["frame"].values, model_name=candidate_name)

    b_err_3d = np.sqrt(np.sum((act_pos - b_pred) ** 2, axis=1))
    c_err_3d = np.sqrt(np.sum((act_pos - c_pred) ** 2, axis=1))

    comparison[f"{baseline_name.lower()}_error_3d_mm"] = b_err_3d
    comparison[f"{candidate_name.lower()}_error_3d_mm"] = c_err_3d

    b_p95 = b_metrics.p95_3d
    c_p95 = c_metrics.p95_3d
    p95_reduction = ((b_p95 - c_p95) / b_p95) * 100.0 if b_p95 > 0 else 0.0

    winner = candidate_name if c_p95 < b_p95 else baseline_name
    min_frame = int(comparison["frame"].min())
    max_frame = int(comparison["frame"].max())

    result = ModelComparisonResult(
        comparison_df=comparison,
        baseline_metrics=b_metrics,
        candidate_metrics=c_metrics,
        p95_reduction_pct=p95_reduction,
        winner=winner,
        common_frames=len(comparison),
        frame_range=(min_frame, max_frame),
    )

    logger.info(
        f"Compared {baseline_name} (P95={b_p95:.4f} mm) vs {candidate_name} (P95={c_p95:.4f} mm): "
        f"Winner={winner} ({abs(p95_reduction):.2f}% difference)"
    )
    return result
