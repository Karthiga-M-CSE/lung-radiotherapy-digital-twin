import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


def build_digital_twin_dataframe(prediction_df: pd.DataFrame, safety_df: pd.DataFrame) -> pd.DataFrame:
    if not prediction_df["frame"].equals(safety_df["frame"]):
        raise ValueError("Prediction and safety-gate frames do not match.")

    uncertainty_bound = float(safety_df["uncertainty_bound_mm"].iloc[0])

    twin = pd.DataFrame()
    twin["frame"] = prediction_df["frame"]

    twin["actual_x_mm"] = prediction_df["actual_x_mm"]
    twin["actual_y_mm"] = prediction_df["actual_y_mm"]
    twin["actual_z_mm"] = prediction_df["actual_z_mm"]

    twin["predicted_x_mm"] = prediction_df["predicted_x_mm"]
    twin["predicted_y_mm"] = prediction_df["predicted_y_mm"]
    twin["predicted_z_mm"] = prediction_df["predicted_z_mm"]

    if "error_3d_mm" in prediction_df.columns:
        twin["prediction_error_mm"] = prediction_df["error_3d_mm"]
    elif "prediction_error_3d_mm" in prediction_df.columns:
        twin["prediction_error_mm"] = prediction_df["prediction_error_3d_mm"]
    else:
        raise ValueError("Cannot find 3D error column in prediction DataFrame.")

    twin["uncertainty_bound_mm"] = uncertainty_bound

    twin["safety_status"] = safety_df["safety_status"].map({
        "SAFE / CONTINUE": "CONTINUE",
        "FLAG / REVIEW": "REVIEW"
    }).fillna("REVIEW")

    twin["predicted_motion_magnitude_mm"] = np.sqrt(
        twin["predicted_x_mm"] ** 2
        + twin["predicted_y_mm"] ** 2
        + twin["predicted_z_mm"] ** 2
    )

    logger.info(
        f"Built Digital Twin State: {len(twin)} frames | "
        f"Mean Error={twin['prediction_error_mm'].mean():.4f} mm | "
        f"Uncertainty Bound={uncertainty_bound:.4f} mm"
    )
    return twin
