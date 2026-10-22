"""Unit tests for metrics, chronological splits, and split conformal prediction."""

import unittest
import numpy as np
import pandas as pd

from src.utils.metrics import (
    compute_3d_euclidean_error,
    compute_mae,
    compute_rmse,
    compute_percentile_errors,
    compute_prediction_errors,
)
from src.data.splits import (
    ChronologicalSplit,
    chronological_split,
    chronological_frame_split,
    chronological_cycle_split,
)
from src.models.conformal import (
    SplitConformalPredictor,
    ConformalIntervals,
    ConformalCoverageReport,
)
from src.models.ridge_predictor import RespiratoryRidgePredictor
from src.models.uncertainty import evaluate_conformal_safety_gate


class TestMetricsCalculation(unittest.TestCase):
    """Test MAE, RMSE, 3D Euclidean error, and P90/P95/P99 percentile functions."""

    def setUp(self):
        self.actual = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 2.0],
            [3.0, 0.0, 4.0],
            [0.0, 0.0, 10.0],
        ])
        self.predicted = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 2.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ])

    def test_3d_euclidean_error(self):
        errs = compute_3d_euclidean_error(self.actual, self.predicted)
        # Expected distances: 0.0, 0.0, sqrt(3^2 + 4^2)=5.0, 10.0
        np.testing.assert_allclose(errs, [0.0, 0.0, 5.0, 10.0])

    def test_mae_and_rmse(self):
        mae = compute_mae(self.actual, self.predicted)
        rmse = compute_rmse(self.actual, self.predicted)
        # diffs: [0,0,0], [0,0,0], [3,0,4], [0,0,10] -> sum = 17, 12 elements -> MAE = 17/12 ~ 1.4167
        self.assertAlmostEqual(mae, 17.0 / 12.0)
        # sq diffs: 9 + 16 + 100 = 125 -> RMSE = sqrt(125/12) ~ 3.227486
        self.assertAlmostEqual(rmse, np.sqrt(125.0 / 12.0))

    def test_percentile_errors(self):
        errs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        p_dict = compute_percentile_errors(errs, (50, 90, 95, 99))
        self.assertEqual(p_dict["p50"], 5.5)
        self.assertEqual(p_dict["p90"], 9.1)
        self.assertAlmostEqual(p_dict["p95"], 9.55)
        self.assertAlmostEqual(p_dict["p99"], 9.91)

    def test_comprehensive_prediction_errors(self):
        res = compute_prediction_errors(self.actual, self.predicted)
        self.assertIn("error_3d", res)
        self.assertIn("mean_3d", res)
        self.assertIn("rmse_3d", res)
        self.assertIn("p90_3d", res)
        self.assertIn("p95_3d", res)
        self.assertIn("p99_3d", res)
        self.assertIn("p95_x", res)
        self.assertIn("rmse_z", res)
        self.assertAlmostEqual(res["mean_3d"], (0.0 + 0.0 + 5.0 + 10.0) / 4.0)


class TestChronologicalSplits(unittest.TestCase):
    """Test proper chronological train/calibration/test split without data leakage."""

    def setUp(self):
        self.n = 100
        self.X = np.arange(self.n).reshape(-1, 1)
        self.y = np.arange(self.n).reshape(-1, 1) * 2.0
        self.frames = np.arange(10, 10 + self.n)

    def test_fraction_based_split(self):
        split = chronological_split(self.X, self.y, self.frames, train_frac=0.6, calib_frac=0.2, test_frac=0.2)
        self.assertEqual(len(split.X_train), 60)
        self.assertEqual(len(split.X_calib), 20)
        self.assertEqual(len(split.X_test), 20)

        # Strict chronological ordering
        self.assertLess(np.max(split.train_frames), np.min(split.calib_frames))
        self.assertLess(np.max(split.calib_frames), np.min(split.test_frames))

        # Disjoint frame sets
        s_tr = set(split.train_frames)
        s_ca = set(split.calib_frames)
        s_te = set(split.test_frames)
        self.assertEqual(len(s_tr & s_ca), 0)
        self.assertEqual(len(s_ca & s_te), 0)
        self.assertEqual(len(s_tr & s_te), 0)

    def test_frame_interval_split(self):
        split = chronological_frame_split(
            self.X, self.y, self.frames,
            train_range=(10, 59),
            calib_range=(60, 79),
            test_range=(80, 109),
        )
        self.assertEqual(len(split.X_train), 50)
        self.assertEqual(len(split.X_calib), 20)
        self.assertEqual(len(split.X_test), 30)
        self.assertEqual(int(np.max(split.train_frames)), 59)
        self.assertEqual(int(np.min(split.calib_frames)), 60)
        self.assertEqual(int(np.max(split.calib_frames)), 79)
        self.assertEqual(int(np.min(split.test_frames)), 80)

    def test_temporal_leakage_detection(self):
        # Mismatched order should raise ValueError on validation
        with self.assertRaises(ValueError):
            ChronologicalSplit(
                X_train=np.array([[1]]), y_train=np.array([[1]]), train_frames=np.array([50]),
                X_calib=np.array([[2]]), y_calib=np.array([[2]]), calib_frames=np.array([40]),  # calib < train!
                X_test=np.array([[3]]), y_test=np.array([[3]]), test_frames=np.array([60]),
                train_indices=np.array([0]), calib_indices=np.array([1]), test_indices=np.array([2]),
            )

    def test_overlapping_frame_detection(self):
        with self.assertRaises(ValueError):
            ChronologicalSplit(
                X_train=np.array([[1]]), y_train=np.array([[1]]), train_frames=np.array([10]),
                X_calib=np.array([[2]]), y_calib=np.array([[2]]), calib_frames=np.array([10]),  # overlap
                X_test=np.array([[3]]), y_test=np.array([[3]]), test_frames=np.array([20]),
                train_indices=np.array([0]), calib_indices=np.array([1]), test_indices=np.array([2]),
            )

    def test_cycle_based_split(self):
        df = pd.DataFrame({
            "frame": np.arange(30),
            "cycle_id": [1] * 10 + [2] * 10 + [3] * 10,
            "x_mm": np.zeros(30),
            "y_mm": np.zeros(30),
            "z_mm": np.arange(30) * 0.5,
            "rpm": np.sin(np.linspace(0, 2 * np.pi, 30)),
        })
        split = chronological_cycle_split(
            df,
            cycle_col="cycle_id",
            train_cycles=[1],
            calib_cycles=[2],
            test_cycles=[3],
            target_cols=["x_mm", "y_mm", "z_mm"],
        )
        self.assertEqual(len(split.X_train), 10)
        self.assertEqual(len(split.X_calib), 10)
        self.assertEqual(len(split.X_test), 10)
        self.assertLess(np.max(split.train_frames), np.min(split.calib_frames))
        self.assertLess(np.max(split.calib_frames), np.min(split.test_frames))


class TestSplitConformalPrediction(unittest.TestCase):
    """Test Split Conformal Prediction, interval prediction, and coverage evaluation."""

    def setUp(self):
        # Synthetic 3D calibration residuals with known distribution
        rng = np.random.RandomState(42)
        n_calib = 200
        self.y_true_calib = rng.normal(0, 1.0, size=(n_calib, 3))
        # Simulated model predictions with zero-mean Gaussian error (std=0.5)
        self.y_pred_calib = self.y_true_calib + rng.normal(0, 0.5, size=(n_calib, 3))

        # Held-out test set from the same distribution
        n_test = 200
        self.y_true_test = rng.normal(0, 1.0, size=(n_test, 3))
        self.y_pred_test = self.y_true_test + rng.normal(0, 0.5, size=(n_test, 3))

    def test_calibration_and_intervals(self):
        cp = SplitConformalPredictor(alpha=0.05)
        cp.calibrate(self.y_true_calib, self.y_pred_calib)
        self.assertTrue(cp.is_calibrated)
        self.assertGreater(cp.radius_3d_mm, 0.0)
        self.assertEqual(len(cp.axis_margins_mm), 3)

        intervals = cp.predict_intervals(self.y_pred_test)
        self.assertEqual(len(intervals.lower_bounds), len(self.y_pred_test))
        self.assertEqual(len(intervals.upper_bounds), len(self.y_pred_test))
        self.assertTrue(np.all(intervals.upper_bounds >= intervals.lower_bounds))

    def test_coverage_guarantee_on_test_data(self):
        cp = SplitConformalPredictor(alpha=0.10)  # 90% target coverage
        cp.calibrate(self.y_true_calib, self.y_pred_calib)

        report = cp.evaluate_coverage(self.y_true_test, self.y_pred_test)
        self.assertAlmostEqual(report.target_coverage, 0.90)

        # Empirical 3D coverage should be near or above target 90%
        self.assertGreaterEqual(report.coverage_3d, 0.85)
        self.assertTrue(report.is_valid)
        self.assertIn("SPLIT CONFORMAL PREDICTION", report.summary())

    def test_axis_margins_coverage(self):
        cp = SplitConformalPredictor(alpha=0.05)  # 95% target
        cp.calibrate(self.y_true_calib, self.y_pred_calib)

        report = cp.evaluate_coverage(self.y_true_test, self.y_pred_test)
        # Coordinate-wise coverages should satisfy target ~95%
        self.assertGreaterEqual(report.coverage_x, 0.90)
        self.assertGreaterEqual(report.coverage_y, 0.90)
        self.assertGreaterEqual(report.coverage_z, 0.90)

    def test_conformal_safety_gate_integration(self):
        cp = SplitConformalPredictor(alpha=0.05)
        cp.calibrate(self.y_true_calib, self.y_pred_calib)

        report, status, summary = evaluate_conformal_safety_gate(
            self.y_true_test, self.y_pred_test, cp, alpha=0.05
        )
        self.assertIn("SAFE / CONTINUE", status)
        self.assertGreater(summary["safe_count"], 0)
        self.assertEqual(len(status), len(self.y_true_test))


class TestFullChronologicalConformalPipeline(unittest.TestCase):
    """End-to-end integration test with Ridge model on chronological respiratory data."""

    def test_pipeline_with_ridge(self):
        # 120 frames of smooth synthetic motion
        t = np.linspace(0, 6 * np.pi, 120)
        df = pd.DataFrame({
            "frame": np.arange(120),
            "rpm": np.sin(t),
            "x_mm": 1.0 + 0.5 * np.cos(t),
            "y_mm": 2.0 + 0.8 * np.sin(t),
            "z_mm": 10.0 + 5.0 * np.sin(t),
        })

        # 1. Feature extraction
        ridge = RespiratoryRidgePredictor(alpha=0.5, n_lags=3)
        X, y, frames, _ = ridge.create_features(df)

        # 2. Chronological 3-way split: Train (60%), Calib (20%), Test (20%)
        split = chronological_split(X, y, frames, train_frac=0.6, calib_frac=0.2, test_frac=0.2)
        self.assertTrue(split.train_frames.max() < split.calib_frames.min())
        self.assertTrue(split.calib_frames.max() < split.test_frames.min())

        # 3. Fit Ridge on Train
        ridge.fit(split.X_train, split.y_train)

        # 4. Predict on Calibration set
        calib_preds = ridge.predict(split.X_calib)

        # 5. Calibrate Conformal Predictor on held-out Calibration set
        cp = SplitConformalPredictor(alpha=0.05)
        cp.calibrate(split.y_calib, calib_preds)

        # 6. Predict on held-out Test set
        test_preds = ridge.predict(split.X_test)

        # 7. Evaluate calibrated coverage and interval bounds on Test
        report = cp.evaluate_coverage(split.y_test, test_preds)
        self.assertGreaterEqual(report.coverage_3d, 0.85)

        # 8. Check test metrics
        errors = compute_prediction_errors(split.y_test, test_preds)
        self.assertIn("mae_3d", errors)
        self.assertIn("rmse_3d", errors)
        self.assertIn("p95_3d", errors)


if __name__ == "__main__":
    unittest.main()
