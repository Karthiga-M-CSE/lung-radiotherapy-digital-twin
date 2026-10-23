import unittest
import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, resolve_file
from src.utils.geometry import compute_voxel_centroid, compute_bounding_box, shift_mask
from src.utils.metrics import compute_prediction_errors
from src.models.kalman_filter import ConstantVelocityKalmanFilter
from src.models.uncertainty import evaluate_safety_gate
from src.twin.replanning import evaluate_replanning_persistence


class TestPipelineModules(unittest.TestCase):

    def test_config_resolution(self):
        twin_file = resolve_file("digital_twin_state.csv")
        self.assertTrue(twin_file.exists(), f"Expected {twin_file} to exist.")
        self.assertTrue(PROJECT_ROOT.exists())

    def test_geometry_utilities(self):
        mask = np.zeros((10, 10, 10), dtype=bool)
        mask[2:4, 2:4, 2:4] = True

        centroid = compute_voxel_centroid(mask)
        self.assertIsNotNone(centroid)
        np.testing.assert_allclose(centroid, [2.5, 2.5, 2.5])

        indices = np.argwhere(mask)
        bbox = compute_bounding_box(indices, (2.0, 2.0, 3.0))
        self.assertEqual(bbox, (4.0, 4.0, 6.0))

        shifted = shift_mask(mask, dx_mm=2.0, dy_mm=0.0, dz_mm=0.0, sx=1.0, sy=1.0)
        self.assertEqual(shifted.shape, mask.shape)
        self.assertEqual(int(np.sum(shifted)), int(np.sum(mask)))

    def test_metrics_calculation(self):
        actual = np.array([
            [0.0, 0.0, 0.0],
            [3.0, 4.0, 0.0]
        ])
        predicted = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ])

        results = compute_prediction_errors(actual, predicted)
        np.testing.assert_allclose(results["error_3d"], [0.0, 5.0])
        self.assertEqual(results["max_3d"], 5.0)
        self.assertEqual(results["mean_3d"], 2.5)

    def test_kalman_filter(self):
        kf = ConstantVelocityKalmanFilter(dt=1.0)
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0]
        ])
        pred_positions, errors = kf.filter_sequence(positions)
        self.assertEqual(len(pred_positions), 2)
        self.assertEqual(len(errors), 2)
        self.assertTrue(np.all(errors >= 0.0))

    def test_uncertainty_safety_gate(self):
        errors = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
        threshold, status, summary = evaluate_safety_gate(errors, preset_threshold=3.5)
        self.assertEqual(threshold, 3.5)
        self.assertEqual(int((status == "SAFE / CONTINUE").sum()), 3)
        self.assertEqual(int((status == "FLAG / REVIEW").sum()), 2)

    def test_replanning_trigger_persistence(self):
        twin_df = pd.DataFrame({
            "frame": [1, 2, 3, 4, 5, 6],
            "prediction_error_mm": [1.0, 4.0, 4.0, 1.0, 4.0, 4.0],
            "uncertainty_bound_mm": [3.43] * 6
        })
        df_out, summary = evaluate_replanning_persistence(twin_df, persistence=3)
        self.assertEqual(summary["replanning_review_count"], 0)
        self.assertEqual(summary["deviation_count"], 4)

        twin_df_alert = pd.DataFrame({
            "frame": [1, 2, 3, 4],
            "prediction_error_mm": [4.0, 4.0, 4.0, 1.0],
            "uncertainty_bound_mm": [3.43] * 4
        })
        df_alert_out, alert_summary = evaluate_replanning_persistence(twin_df_alert, persistence=3)
        self.assertEqual(alert_summary["replanning_review_count"], 1)


if __name__ == "__main__":
    unittest.main()
