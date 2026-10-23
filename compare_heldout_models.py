import os
import numpy as np
import pandas as pd

from src.config import (
    resolve_file,
    PROJECT_ROOT,
    OUTPUTS_DIR,
    TEST_START_FRAME,
    TEST_END_FRAME,
)
from src.models.evaluation import compare_motion_models
from src.utils.logging import get_logger

logger = get_logger("compare_heldout_models")

MOTION_FILE = resolve_file("tumor_motion_with_rpm.csv")
RIDGE_FILE = resolve_file("respiratory_heldout_test_results.csv")
KALMAN_FILE = resolve_file("kalman_prediction_results.csv")
OUTPUT_FILE = resolve_file("heldout_model_comparison.csv", preferred_dir=PROJECT_ROOT)


def main():
    print("=" * 70)
    print("HELD-OUT MODEL COMPARISON")
    print("KALMAN vs RESPIRATORY-INFORMED RIDGE")
    print("=" * 70)

    for file_path in [MOTION_FILE, RIDGE_FILE, KALMAN_FILE]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"\nRequired file not found:\n{file_path}")

    motion = pd.read_csv(MOTION_FILE)
    ridge = pd.read_csv(RIDGE_FILE)
    kalman = pd.read_csv(KALMAN_FILE)

    print("\nLoaded files:")
    print(f"Motion data : {len(motion)} rows")
    print(f"Ridge test  : {len(ridge)} rows")
    print(f"Kalman data : {len(kalman)} rows")

    result = compare_motion_models(
        baseline_df=kalman,
        candidate_df=ridge,
        baseline_name="Kalman",
        candidate_name="Ridge",
        frame_range=(TEST_START_FRAME, TEST_END_FRAME),
    )

    print("\n" + "=" * 70)
    print("HELD-OUT TEST COVERAGE")
    print("=" * 70)
    print(f"Requested frames : {TEST_START_FRAME} -> {TEST_END_FRAME}")
    print(f"Common frames    : {result.common_frames}")
    print(f"\nCommon frame range: {result.frame_range[0]} -> {result.frame_range[1]}")
    print("[OK] Both models evaluated on identical frames")

    print(f"\n{result.summary()}")

    if result.winner == "Ridge":
        print("\nRESULT: Respiratory-informed Ridge performs BETTER on the held-out respiratory cycles.")
    else:
        print("\nRESULT: Kalman performs better on the held-out respiratory cycles.")

    result.comparison_df.to_csv(OUTPUT_FILE, index=False)
    result.comparison_df.to_csv(OUTPUTS_DIR / "heldout_model_comparison.csv", index=False)

    print("\n" + "=" * 70)
    print("OUTPUT")
    print("=" * 70)
    print(f"\nSaved comparison to:\n{OUTPUT_FILE}")
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
