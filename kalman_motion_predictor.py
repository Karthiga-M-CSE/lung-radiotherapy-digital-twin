import os
import numpy as np
import pandas as pd

INPUT_FILE = r"C:\Users\gayu0\OneDrive\Desktop\projects\DATASET1\tumor_motion_with_rpm.csv"
OUTPUT_FILE = r"C:\Users\gayu0\OneDrive\Desktop\projects\DATASET1\kalman_prediction_results.csv"

df = pd.read_csv(INPUT_FILE)

positions = df[["x_mm", "y_mm", "z_mm"]].values.astype(float)

dt = 1.0
state = np.zeros(6)
state[:3] = positions[0]

P = np.eye(6)
Q = np.eye(6) * 0.01
R = np.eye(3) * 0.5

H = np.zeros((3, 6))
H[0, 0] = 1
H[1, 1] = 1
H[2, 2] = 1

F = np.eye(6)
F[0, 3] = dt
F[1, 4] = dt
F[2, 5] = dt

predictions = []
errors = []

for i in range(1, len(positions)):
    predicted_state = F @ state
    predicted_covariance = F @ P @ F.T + Q

    predicted_position = predicted_state[:3]
    actual_position = positions[i]

    error_vector = actual_position - predicted_position
    error_3d = np.linalg.norm(error_vector)

    predictions.append({
        "frame": df.iloc[i]["frame"],
        "actual_x_mm": actual_position[0],
        "actual_y_mm": actual_position[1],
        "actual_z_mm": actual_position[2],
        "predicted_x_mm": predicted_position[0],
        "predicted_y_mm": predicted_position[1],
        "predicted_z_mm": predicted_position[2],
        "error_x_mm": error_vector[0],
        "error_y_mm": error_vector[1],
        "error_z_mm": error_vector[2],
        "prediction_error_3d_mm": error_3d
    })

    errors.append(error_3d)

    measurement = actual_position

    innovation = measurement - H @ predicted_state
    innovation_covariance = H @ predicted_covariance @ H.T + R
    K = predicted_covariance @ H.T @ np.linalg.inv(innovation_covariance)

    state = predicted_state + K @ innovation
    P = (np.eye(6) - K @ H) @ predicted_covariance

results = pd.DataFrame(predictions)
results.to_csv(OUTPUT_FILE, index=False)

errors = np.array(errors)

print("\n============================================================")
print("CONSTANT-VELOCITY KALMAN MOTION PREDICTION")
print("============================================================")
print(f"\nFrames processed: {len(positions)}")
print("\nONE-STEP-AHEAD 3D PREDICTION ERROR")
print(f"Mean error:    {errors.mean():.4f} mm")
print(f"Median error:  {np.median(errors):.4f} mm")
print(f"P90 error:     {np.percentile(errors, 90):.4f} mm")
print(f"P95 error:     {np.percentile(errors, 95):.4f} mm")
print(f"P99 error:     {np.percentile(errors, 99):.4f} mm")
print(f"Maximum error: {errors.max():.4f} mm")

print("\n============================================================")
print("RESULTS SAVED TO:")
print(OUTPUT_FILE)
print("============================================================")
