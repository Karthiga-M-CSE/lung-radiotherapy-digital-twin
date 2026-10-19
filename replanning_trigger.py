import pandas as pd

from src.config import resolve_file, PROJECT_ROOT, OUTPUTS_DIR, PERSISTENCE_COUNT
from src.twin.replanning import evaluate_replanning_persistence
from src.utils.logging import get_logger

logger = get_logger("replanning_trigger")

input_file = resolve_file("digital_twin_state.csv")
output_file = resolve_file("replanning_trigger_results.csv", preferred_dir=PROJECT_ROOT)

df = pd.read_csv(input_file)

df_out, summary = evaluate_replanning_persistence(df, persistence=PERSISTENCE_COUNT)

df_out.to_csv(output_file, index=False)
df_out.to_csv(OUTPUTS_DIR / "replanning_trigger_results.csv", index=False)

print("\n===================================")
print("      FINAL REPLANNING TRIGGER")
print("===================================")

print(
    f"Total predictions       : {summary['total_frames']}"
)

print(
    f"Stable predictions      : {summary['stable_count']}"
)

print(
    f"Deviation predictions   : {summary['deviation_count']}"
)

print(
    f"Replanning review flags : {summary['replanning_review_count']}"
)

print(
    f"\nResearch threshold      : "
    f"{summary['threshold_mm']:.4f} mm"
)

print(
    f"Persistence requirement : "
    f"{summary['persistence_requirement']} consecutive deviations"
)

print("\nOutput saved to:")
print(output_file)

print("===================================\n")
