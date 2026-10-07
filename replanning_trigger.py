import pandas as pd

# -----------------------------
# Load FINAL Digital Twin state
# -----------------------------
input_file = "digital_twin_state.csv"
output_file = "replanning_trigger_results.csv"

df = pd.read_csv(input_file)

# -----------------------------
# Research prototype settings
# -----------------------------
# Use the uncertainty threshold generated
# from the final held-out Ridge predictions.
#
# IMPORTANT:
# This is NOT a clinically validated
# replanning or treatment threshold.

THRESHOLD_MM = df["uncertainty_bound_mm"].iloc[0]

# Number of consecutive deviations required
PERSISTENCE = 3


# -----------------------------
# Classify prediction state
# -----------------------------
df["state"] = df["prediction_error_mm"].apply(
    lambda x: "DEVIATION"
    if x > THRESHOLD_MM
    else "STABLE"
)


# -----------------------------
# Detect persistent deviation
# -----------------------------
df["consecutive_deviations"] = 0

count = 0

for i in range(len(df)):

    if df.loc[i, "state"] == "DEVIATION":
        count += 1
    else:
        count = 0

    df.loc[i, "consecutive_deviations"] = count


# -----------------------------
# Replanning review trigger
# -----------------------------
df["replanning_review"] = (
    df["consecutive_deviations"] >= PERSISTENCE
)


# -----------------------------
# Save results
# -----------------------------
df.to_csv(output_file, index=False)


# -----------------------------
# Summary
# -----------------------------
stable = (
    df["state"] == "STABLE"
).sum()

deviation = (
    df["state"] == "DEVIATION"
).sum()

reviews = (
    df["replanning_review"]
).sum()


print("\n===================================")
print("      FINAL REPLANNING TRIGGER")
print("===================================")

print(
    f"Total predictions       : {len(df)}"
)

print(
    f"Stable predictions      : {stable}"
)

print(
    f"Deviation predictions   : {deviation}"
)

print(
    f"Replanning review flags : {reviews}"
)

print(
    f"\nResearch threshold      : "
    f"{THRESHOLD_MM:.4f} mm"
)

print(
    f"Persistence requirement : "
    f"{PERSISTENCE} consecutive deviations"
)

print("\nOutput saved to:")
print(output_file)

print("===================================\n")