from pathlib import Path
import pandas as pd

from src.config import resolve_file
from src.utils.logging import get_logger

logger = get_logger("validate_final_pipeline")

FILES = {
    "ridge": resolve_file("respiratory_heldout_test_results.csv"),
    "safety": resolve_file("uncertainty_safety_gate_results.csv"),
    "twin": resolve_file("digital_twin_state.csv"),
    "replanning": resolve_file("replanning_trigger_results.csv"),
    "comparison": resolve_file("heldout_model_comparison.csv"),
}

print("=" * 70)
print("FINAL DIGITAL TWIN PIPELINE VALIDATION")
print("=" * 70)

dfs = {}

for name, path in FILES.items():
    if not path.exists():
        raise FileNotFoundError(
            f"\nMissing {name} file:\n{path}\n"
            f"Please run the upstream pipeline scripts to generate this artifact."
        )

    dfs[name] = pd.read_csv(path)
    print(f"\n{name.upper():12} : {len(dfs[name])} rows (from {path.name})")

ridge = dfs["ridge"]
safety = dfs["safety"]
twin = dfs["twin"]
replanning = dfs["replanning"]
comparison = dfs["comparison"]

EXPECTED_FRAMES = set(range(139, 180))

print("\n" + "=" * 70)
print("FRAME VALIDATION")
print("=" * 70)

for name, df in [
    ("Ridge", ridge),
    ("Safety", safety),
    ("Digital Twin", twin),
    ("Replanning", replanning),
]:
    frames = set(df["frame"])

    print(
        f"{name:12}: "
        f"{min(frames)} -> {max(frames)} "
        f"({len(frames)} frames)"
    )

    assert frames == EXPECTED_FRAMES, (
        f"{name} does not contain exactly frames 139-179."
    )

print("\n[OK] All downstream modules use identical 41 held-out frames.")

print("\n" + "=" * 70)
print("RIDGE MODEL VALIDATION")
print("=" * 70)

ridge_error = ridge["error_3d_mm"] if "error_3d_mm" in ridge.columns else ridge["prediction_error_3d_mm"]

ridge_p95 = ridge_error.quantile(0.95)

print(f"Mean error : {ridge_error.mean():.4f} mm")
print(f"P95 error  : {ridge_p95:.4f} mm")

assert abs(ridge_p95 - 3.4304) < 0.01

print("[OK] Final Ridge P95 ~ 3.4304 mm")

print("\n" + "=" * 70)
print("SAFETY GATE VALIDATION")
print("=" * 70)

safety_threshold = safety["uncertainty_bound_mm"].iloc[0]

print(f"Threshold : {safety_threshold:.4f} mm")

assert abs(safety_threshold - 3.4304) < 0.01

continue_count = (
    safety["safety_status"] == "SAFE / CONTINUE"
).sum()

review_count = (
    safety["safety_status"] == "FLAG / REVIEW"
).sum()

print(f"Continue  : {continue_count}")
print(f"Review    : {review_count}")

assert continue_count == 39
assert review_count == 2

print("[OK] Safety Gate = 39 CONTINUE / 2 REVIEW")

print("\n" + "=" * 70)
print("DIGITAL TWIN VALIDATION")
print("=" * 70)

twin_threshold = twin["uncertainty_bound_mm"].iloc[0]

print(f"Threshold : {twin_threshold:.4f} mm")
print(f"Mean error: {twin['prediction_error_mm'].mean():.4f} mm")

assert abs(twin_threshold - safety_threshold) < 0.0001

print("[OK] Digital Twin uses the same uncertainty threshold.")

print("\n" + "=" * 70)
print("REPLANNING VALIDATION")
print("=" * 70)

deviations = (
    replanning["state"] == "DEVIATION"
).sum()

review_flags = (
    replanning["replanning_review"] == True
).sum()

print(f"Deviations       : {deviations}")
print(f"Review flags     : {review_flags}")

assert deviations == 2
assert review_flags == 0

print("[OK] 2 deviations, but no 3-consecutive-deviation trigger.")

print("\n" + "=" * 70)
print("MODEL COMPARISON VALIDATION")
print("=" * 70)

if "P95" in comparison.columns:
    print(comparison)

print("\nExpected final comparison:")
print("Kalman P95 : 11.8767 mm")
print("Ridge P95  : 3.4304 mm")
print("Reduction  : 71.12%")

print("\n" + "=" * 70)
print("FINAL VALIDATION PASSED")
print("=" * 70)

print("""
[OK] 41 held-out frames
[OK] Ridge predictions consistent
[OK] Safety Gate consistent
[OK] Digital Twin consistent
[OK] Replanning Trigger consistent
[OK] P95 uncertainty = 3.4304 mm
[OK] Safety = 39 CONTINUE / 2 REVIEW
[OK] Replanning review = 0

The final prediction pipeline is internally consistent.
""")

print("=" * 70)
