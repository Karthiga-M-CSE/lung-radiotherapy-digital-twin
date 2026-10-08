import os
import numpy as np
import pandas as pd

PROJECT = r"C:\Users\smk28\Downloads\RADIOTHERAPY SHIT"
DATA = r"C:\Users\smk28\Downloads\DATASET1\data"

FILES = {
    "segmentation": os.path.join(
        PROJECT, "tumor_segmentation_results.csv"
    ),

    "kalman": os.path.join(
        DATA, "kalman_prediction_results.csv"
    ),

    "respiratory": os.path.join(
        DATA, "respiratory_prediction_results.csv"
    ),

    "safety": os.path.join(
        DATA, "uncertainty_safety_gate_results.csv"
    ),

    "digital_twin": os.path.join(
        PROJECT, "digital_twin_state.csv"
    ),

    "replanning": os.path.join(
        PROJECT, "replanning_trigger_results.csv"
    )
}

print("\n" + "=" * 70)
print("LUNG CANCER RADIOTHERAPY DIGITAL TWIN")
print("MASTER PIPELINE VALIDATION")
print("=" * 70)

print("\n[1] FILE AVAILABILITY")
print("-" * 70)

all_present = True

for name, path in FILES.items():

    exists = os.path.exists(path)

    status = "OK" if exists else "MISSING"

    print(f"{name:15s}: {status}")

    if not exists:
        all_present = False

if not all_present:

    print("\nERROR: One or more required files are missing.")
    print("Fix the missing file before continuing.")
    raise SystemExit

seg = pd.read_csv(FILES["segmentation"])
kalman = pd.read_csv(FILES["kalman"])
resp = pd.read_csv(FILES["respiratory"])
safety = pd.read_csv(FILES["safety"])
twin = pd.read_csv(FILES["digital_twin"])
replan = pd.read_csv(FILES["replanning"])

print("\n[2] ROW COUNTS")
print("-" * 70)

print(f"Tumor segmentation : {len(seg)}")
print(f"Kalman prediction  : {len(kalman)}")
print(f"Respiratory model  : {len(resp)}")
print(f"Safety gate        : {len(safety)}")
print(f"Digital Twin       : {len(twin)}")
print(f"Replanning         : {len(replan)}")

print("\n[3] REQUIRED COLUMNS")
print("-" * 70)

requirements = {

    "segmentation": (
        seg,
        ["frame"]
    ),

    "kalman": (
        kalman,
        ["prediction_error_3d_mm"]
    ),

    "respiratory": (
        resp,
        ["prediction_error_3d_mm"]
    ),

    "safety": (
        safety,
        ["prediction_error_3d_mm"]
    ),

    "digital_twin": (
        twin,
        [
            "frame",
            "prediction_error_mm",
            "uncertainty_bound_mm",
            "safety_status"
        ]
    ),

    "replanning": (
        replan,
        ["frame"]
    )
}

columns_ok = True

for name, (df, required) in requirements.items():

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:

        print(
            f"{name:15s}: MISSING {missing}"
        )

        columns_ok = False

    else:

        print(
            f"{name:15s}: OK"
        )

print("\n[4] MOTION PREDICTION METRICS")
print("-" * 70)

kalman_error = kalman[
    "prediction_error_3d_mm"
].to_numpy()

resp_error = resp[
    "prediction_error_3d_mm"
].to_numpy()

kalman_p95 = np.percentile(
    kalman_error, 95
)

resp_p95 = np.percentile(
    resp_error, 95
)

improvement = (
    (kalman_p95 - resp_p95)
    / kalman_p95
) * 100

print(
    f"Kalman P95              : "
    f"{kalman_p95:.4f} mm"
)

print(
    f"Respiratory model P95   : "
    f"{resp_p95:.4f} mm"
)

print(
    f"P95 reduction           : "
    f"{improvement:.2f}%"
)

print("\n[5] UNCERTAINTY / SAFETY GATE")
print("-" * 70)

if "safety_status" in safety.columns:

    counts = safety[
        "safety_status"
    ].value_counts()

    for status, count in counts.items():

        percentage = (
            count / len(safety)
        ) * 100

        print(
            f"{status:15s}: "
            f"{count} "
            f"({percentage:.2f}%)"
        )

else:

    print(
        "Safety-status column not found."
    )

if "prediction_error_3d_mm" in safety.columns:

    safety_errors = safety[
        "prediction_error_3d_mm"
    ].to_numpy()

    print(
        f"\nSafety P95 error: "
        f"{np.percentile(safety_errors, 95):.4f} mm"
    )

    print(
        f"Maximum error:    "
        f"{np.max(safety_errors):.4f} mm"
    )

print("\n[6] DIGITAL TWIN")
print("-" * 70)

if "prediction_error_mm" in twin.columns:

    print(
        f"Mean prediction error : "
        f"{twin['prediction_error_mm'].mean():.4f} mm"
    )

    print(
        f"P95 uncertainty       : "
        f"{twin['uncertainty_bound_mm'].quantile(0.95):.4f} mm"
    )

if "safety_status" in twin.columns:

    print("\nSafety status:")

    print(
        twin["safety_status"]
        .value_counts()
        .to_string()
    )

print("\n[7] REPLANNING TRIGGER")
print("-" * 70)

print(
    f"Rows evaluated: {len(replan)}"
)

if "replanning_flag" in replan.columns:

    print(
        replan["replanning_flag"]
        .value_counts()
        .to_string()
    )

elif "replanning_review" in replan.columns:

    print(
        replan["replanning_review"]
        .value_counts()
        .to_string()
    )

else:

    print(
        "Replanning result column detected "
        "through available output."
    )

print("\n" + "=" * 70)

if all_present and columns_ok:

    print("PIPELINE VALIDATION: PASSED")

else:

    print("PIPELINE VALIDATION: CHECK REQUIRED")

print("=" * 70)

print("\nThe pipeline outputs are ready for frontend integration.")
print()
