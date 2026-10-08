import numpy as np
import pandas as pd
from scipy.signal import find_peaks

RPM_FILE = r"C:\Users\gayu0\OneDrive\Desktop\projects\DATASET1\rpm_signal.txt"

rpm = np.loadtxt(RPM_FILE)

print("=" * 60)
print("XCAT RESPIRATORY CYCLE INSPECTION")
print("=" * 60)

print(f"\nTotal RPM samples: {len(rpm)}")
print(f"Minimum RPM value: {rpm.min():.4f}")
print(f"Maximum RPM value: {rpm.max():.4f}")

peaks, properties = find_peaks(
    rpm,
    distance=6,
    prominence=(rpm.max() - rpm.min()) * 0.10
)

print(f"\nDetected respiratory peaks: {len(peaks)}")
print("Peak frame indices:")
print(peaks.tolist())

valleys, _ = find_peaks(
    -rpm,
    distance=6,
    prominence=(rpm.max() - rpm.min()) * 0.10
)

print(f"\nDetected respiratory valleys: {len(valleys)}")
print("Valley frame indices:")
print(valleys.tolist())

cycles = []

for i in range(len(peaks) - 1):
    start = int(peaks[i])
    end = int(peaks[i + 1])

    cycles.append({
        "cycle": i + 1,
        "start_frame": start,
        "end_frame": end,
        "num_frames": end - start + 1
    })

cycle_df = pd.DataFrame(cycles)

print("\n" + "=" * 60)
print("RESPIRATORY CYCLES")
print("=" * 60)

print(cycle_df.to_string(index=False))

if len(cycle_df) >= 10:

    test_cycles = max(4, int(round(len(cycle_df) * 0.25)))
    train_cycles = len(cycle_df) - test_cycles

    print("\n" + "=" * 60)
    print("PROPOSED TEMPORAL SPLIT")
    print("=" * 60)

    print(f"\nTraining cycles: 1–{train_cycles}")
    print(f"Testing cycles : {train_cycles + 1}–{len(cycle_df)}")

    train_end = cycle_df.iloc[train_cycles - 1]["end_frame"]
    test_start = cycle_df.iloc[train_cycles]["start_frame"]

    print(f"\nTraining ends at frame: {train_end}")
    print(f"Testing begins at frame: {test_start}")

    print("\nIMPORTANT:")
    print("The model must NEVER be fitted using the test-cycle data.")

OUTPUT = r"C:\Users\gayu0\OneDrive\Desktop\projects\DATASET1\respiratory_cycle_boundaries.csv"

cycle_df.to_csv(OUTPUT, index=False)

print(f"\nSaved cycle information to:")
print(OUTPUT)

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
