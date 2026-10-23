from typing import Tuple, Dict, Any
import pandas as pd

from src.config import PERSISTENCE_COUNT
from src.utils.logging import get_logger

logger = get_logger(__name__)


def evaluate_replanning_persistence(
    twin_df: pd.DataFrame,
    persistence: int = PERSISTENCE_COUNT
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = twin_df.copy()
    threshold_mm = float(df["uncertainty_bound_mm"].iloc[0])

    df["state"] = df["prediction_error_mm"].apply(
        lambda x: "DEVIATION" if x > threshold_mm else "STABLE"
    )

    df["consecutive_deviations"] = 0
    count = 0

    for i in range(len(df)):
        if df.loc[i, "state"] == "DEVIATION":
            count += 1
        else:
            count = 0
        df.loc[i, "consecutive_deviations"] = count

    df["replanning_review"] = df["consecutive_deviations"] >= persistence

    stable_count = int((df["state"] == "STABLE").sum())
    deviation_count = int((df["state"] == "DEVIATION").sum())
    review_count = int(df["replanning_review"].sum())

    summary = {
        "total_frames": len(df),
        "stable_count": stable_count,
        "deviation_count": deviation_count,
        "replanning_review_count": review_count,
        "persistence_requirement": persistence,
        "threshold_mm": threshold_mm,
    }

    logger.info(
        f"Replanning Trigger evaluated: Total={len(df)} | Stable={stable_count} | "
        f"Deviations={deviation_count} | Reviews={review_count} (Persistence={persistence})"
    )
    return df, summary
