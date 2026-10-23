import pandas as pd
import numpy as np

from src.config import resolve_file, PROJECT_ROOT, OUTPUTS_DIR
from src.twin.state import build_digital_twin_dataframe
from src.utils.logging import get_logger

logger = get_logger("digital_twin_state")

prediction_file = resolve_file("respiratory_heldout_test_results.csv")
safety_gate_file = resolve_file("uncertainty_safety_gate_results.csv")
output_file = resolve_file("digital_twin_state.csv", preferred_dir=PROJECT_ROOT)

df = pd.read_csv(prediction_file)

print("Prediction rows:", len(df))
print("Prediction columns:")
print(df.columns.tolist())

safety = pd.read_csv(safety_gate_file)

print()
print("Safety-gate rows:", len(safety))

twin = build_digital_twin_dataframe(df, safety)
P95_UNCERTAINTY_MM = twin["uncertainty_bound_mm"].iloc[0]

twin.to_csv(output_file, index=False)
twin.to_csv(OUTPUTS_DIR / "digital_twin_state.csv", index=False)

print()
print("========================================")
print("FINAL DIGITAL TWIN STATE")
print("========================================")

print("Frames:", len(twin))

print(
    "Mean prediction error:",
    round(twin["prediction_error_mm"].mean(), 4),
    "mm"
)

print(
    "P95 uncertainty:",
    round(P95_UNCERTAINTY_MM, 4),
    "mm"
)

print()
print("Safety status:")
print(twin["safety_status"].value_counts())

print()
print("Frame range:")
print(
    int(twin["frame"].min()),
    "->",
    int(twin["frame"].max())
)

print()
print("Saved:")
print(output_file)
print("========================================\n")
