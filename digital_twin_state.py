import pandas as pd
import numpy as np

prediction_file = r"C:\Users\gayu0\OneDrive\Desktop\projects\DATASET1\respiratory_heldout_test_results.csv"

safety_gate_file = r"C:\Users\gayu0\OneDrive\Desktop\projects\DATASET1\uncertainty_safety_gate_results.csv"

output_file = r"C:\Users\smk28\Downloads\RADIOTHERAPY SHIT\digital_twin_state.csv"

df = pd.read_csv(prediction_file)

print("Prediction rows:", len(df))
print("Prediction columns:")
print(df.columns.tolist())

safety = pd.read_csv(safety_gate_file)

print()
print("Safety-gate rows:", len(safety))

if not df["frame"].equals(safety["frame"]):
    raise ValueError(
        "Prediction and safety-gate frames do not match."
    )

P95_UNCERTAINTY_MM = safety["uncertainty_bound_mm"].iloc[0]

twin = pd.DataFrame()

twin["frame"] = df["frame"]

twin["actual_x_mm"] = df["actual_x_mm"]
twin["actual_y_mm"] = df["actual_y_mm"]
twin["actual_z_mm"] = df["actual_z_mm"]

twin["predicted_x_mm"] = df["predicted_x_mm"]
twin["predicted_y_mm"] = df["predicted_y_mm"]
twin["predicted_z_mm"] = df["predicted_z_mm"]

twin["prediction_error_mm"] = df["error_3d_mm"]

twin["uncertainty_bound_mm"] = P95_UNCERTAINTY_MM

twin["safety_status"] = safety["safety_status"].map({
    "SAFE / CONTINUE": "CONTINUE",
    "FLAG / REVIEW": "REVIEW"
})

twin["predicted_motion_magnitude_mm"] = np.sqrt(
    twin["predicted_x_mm"] ** 2
    + twin["predicted_y_mm"] ** 2
    + twin["predicted_z_mm"] ** 2
)

twin.to_csv(output_file, index=False)

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
    "→",
    int(twin["frame"].max())
)

print()
print("Saved:")
print(output_file)
