"""Unit tests for Kalman filtering, Ridge regression, and unified model evaluation.

Tests cover:
    1. ConstantVelocityKalmanFilter: initialization, fit, predict, update, step, and sequence prediction
    2. RespiratoryRidgePredictor: lag feature generation, fit, predict, update, save/load, and sequence prediction
    3. Unified evaluation pipeline: compute_prediction_errors, evaluate_model_predictions
    4. compare_motion_models: identical frame alignment, P95 reduction, and winner determination
    5. Head-to-head model comparison on identical synthetic respiratory motion sequences
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.kalman_filter import ConstantVelocityKalmanFilter
from src.models.ridge_predictor import RespiratoryRidgePredictor
from src.models.evaluation import (
    ModelEvaluationMetrics,
    ModelComparisonResult,
    evaluate_model_predictions,
    compare_motion_models,
)


class TestConstantVelocityKalmanFilter(unittest.TestCase):
    """Test 3D constant-velocity Kalman filter architecture."""

    def setUp(self):
        self.dt = 0.5
        self.kf = ConstantVelocityKalmanFilter(dt=self.dt, q_var=0.01, r_var=0.05)

    def test_uninitialized_predict_raises(self):
        with self.assertRaises(RuntimeError):
            self.kf.predict()

    def test_initialization(self):
        init_pos = [10.0, 20.0, 30.0]
        init_vel = [1.0, 2.0, 3.0]
        self.kf.initialize(init_pos, initial_velocity=init_vel)
        self.assertTrue(self.kf.is_initialized)
        np.testing.assert_allclose(self.kf.state[:3], init_pos)
        np.testing.assert_allclose(self.kf.state[3:], init_vel)

    def test_predict_state_propagation(self):
        self.kf.initialize([10.0, 20.0, 30.0], initial_velocity=[2.0, 0.0, -4.0])
        pred_state, pred_cov = self.kf.predict()
        # Next position: pos + vel * dt = 10 + 2*0.5 = 11, 20 + 0 = 20, 30 - 4*0.5 = 28
        expected_pos = [11.0, 20.0, 28.0]
        np.testing.assert_allclose(pred_state[:3], expected_pos)
        self.assertTrue(np.all(np.diag(pred_cov) > 0))

    def test_predict_position_convenience(self):
        self.kf.initialize([5.0, 5.0, 5.0])
        pos, uncertainty = self.kf.predict_position()
        self.assertEqual(len(pos), 3)
        self.assertGreater(uncertainty, 0.0)

    def test_update_reduces_uncertainty(self):
        self.kf.initialize([0.0, 0.0, 0.0])
        pred_state, pred_cov = self.kf.predict()
        prior_trace = np.trace(pred_cov)

        measurement = [0.1, -0.1, 0.05]
        up_state, up_cov = self.kf.update(measurement, pred_state, pred_cov)
        post_trace = np.trace(up_cov)

        # Measurement update must reduce covariance uncertainty
        self.assertLess(post_trace, prior_trace)

    def test_step_cycle(self):
        self.kf.initialize([0.0, 0.0, 0.0])
        pred_pos, updated_pos, uncert = self.kf.step([1.0, 1.0, 1.0])
        self.assertEqual(len(pred_pos), 3)
        self.assertEqual(len(updated_pos), 3)
        self.assertGreater(uncert, 0.0)

    def test_fit_from_positions(self):
        positions = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
        ])
        self.kf.fit(positions, dt=1.0)
        self.assertTrue(self.kf.is_initialized)
        np.testing.assert_allclose(self.kf.state[:3], [0.0, 0.0, 0.0])
        # Estimated initial velocity is (pos[1] - pos[0])/1.0 = [1.0, 2.0, 3.0]
        np.testing.assert_allclose(self.kf.state[3:], [1.0, 2.0, 3.0])

    def test_filter_sequence_compatibility(self):
        # 10 linear constant-velocity steps
        t = np.arange(10)
        positions = np.column_stack([t, 2 * t, 3 * t])
        pred_positions, errors = self.kf.filter_sequence(positions)
        self.assertEqual(len(pred_positions), 9)
        self.assertEqual(len(errors), 9)
        self.assertTrue(np.all(errors >= 0.0))

    def test_predict_sequence_dataframe(self):
        positions = np.array([
            [10.0, 20.0, 30.0],
            [11.0, 21.0, 31.0],
            [12.0, 22.0, 32.0],
            [13.0, 23.0, 33.0],
        ])
        frames = np.array([101, 102, 103, 104])
        df = self.kf.predict_sequence(positions, frames=frames)
        self.assertEqual(len(df), 3)
        self.assertEqual(list(df["frame"]), [102, 103, 104])
        self.assertIn("error_3d_mm", df.columns)
        self.assertIn("uncertainty_3d_mm", df.columns)


class TestRespiratoryRidgePredictor(unittest.TestCase):
    """Test RespiratoryRidgePredictor fit/predict/update architecture."""

    def setUp(self):
        self.predictor = RespiratoryRidgePredictor(alpha=0.1, n_lags=2)

    def test_unfitted_predict_raises(self):
        with self.assertRaises(RuntimeError):
            self.predictor.predict(np.zeros((5, 8)))

    def test_feature_creation_from_dataframe(self):
        df = pd.DataFrame({
            "frame": np.arange(10),
            "rpm": np.linspace(0, 1, 10),
            "x_mm": np.arange(10) * 1.0,
            "y_mm": np.arange(10) * 2.0,
            "z_mm": np.arange(10) * 3.0,
        })
        X, y, frames, feat_names = self.predictor.create_features(
            df=df,
            feature_cols=["rpm", "x_mm", "y_mm", "z_mm"],
            target_cols=["x_mm", "y_mm", "z_mm"],
        )
        # With n_lags=2, first 2 rows have NaNs and are dropped -> 8 rows remaining
        self.assertEqual(len(X), 8)
        self.assertEqual(len(y), 8)
        self.assertEqual(len(frames), 8)
        self.assertEqual(frames[0], 2)
        # 4 features x 2 lags = 8 feature columns
        self.assertEqual(len(feat_names), 8)

    def test_fit_and_predict(self):
        rng = np.random.RandomState(42)
        X = rng.randn(50, 6)
        # Linear relationship
        weights = rng.randn(6, 3)
        y = X @ weights

        self.predictor.fit(X, y)
        self.assertTrue(self.predictor.is_fitted)

        preds = self.predictor.predict(X[:5])
        self.assertEqual(preds.shape, (5, 3))
        # Well-conditioned linear fit should have low RMSE
        rmse = np.sqrt(np.mean((preds - y[:5]) ** 2))
        self.assertLess(rmse, 0.5)

    def test_update_incremental(self):
        rng = np.random.RandomState(42)
        X_init = rng.randn(30, 4)
        y_init = X_init @ np.ones((4, 3))
        self.predictor.fit(X_init, y_init)

        X_new = rng.randn(10, 4)
        y_new = X_new @ np.ones((4, 3))
        self.predictor.update(X_new, y_new, refit=True)

        preds = self.predictor.predict(X_new)
        self.assertEqual(preds.shape, (10, 3))

    def test_predict_sequence_dataframe(self):
        X = np.ones((10, 4))
        y = np.ones((10, 3)) * 5.0
        self.predictor.fit(X, y)
        frames = np.arange(10) + 100
        df = self.predictor.predict_sequence(X, y, frames=frames)
        self.assertEqual(len(df), 10)
        self.assertIn("error_3d_mm", df.columns)
        self.assertIn("actual_z_mm", df.columns)
        self.assertIn("predicted_z_mm", df.columns)

    def test_save_and_load(self):
        X = np.ones((10, 4))
        y = np.ones((10, 3)) * 2.5
        self.predictor.fit(X, y)

        with tempfile.TemporaryDirectory() as tmp_dir:
            m_path = Path(tmp_dir) / "model.joblib"
            s_path = Path(tmp_dir) / "scaler.joblib"
            self.predictor.save(m_path, s_path)

            loaded = RespiratoryRidgePredictor()
            loaded.load(m_path, s_path)
            self.assertTrue(loaded.is_fitted)
            preds = loaded.predict(X[:2])
            np.testing.assert_allclose(preds, [[2.5, 2.5, 2.5], [2.5, 2.5, 2.5]], atol=1e-2)


class TestUnifiedModelEvaluation(unittest.TestCase):
    """Test unified evaluation metrics and model comparison pipeline."""

    def test_evaluate_model_predictions_zero_error(self):
        act = np.array([[10.0, 20.0, 30.0], [11.0, 21.0, 31.0]])
        pred = act.copy()
        df, metrics = evaluate_model_predictions(act, pred, model_name="PerfectModel")
        self.assertEqual(metrics.mean_3d, 0.0)
        self.assertEqual(metrics.p95_3d, 0.0)
        self.assertEqual(metrics.rmse_3d, 0.0)

    def test_evaluate_model_predictions_known_deviation(self):
        act = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        pred = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 10.0]])
        df, metrics = evaluate_model_predictions(act, pred, model_name="Test")
        # 3D errors are 5.0 and 10.0 -> mean 7.5
        self.assertAlmostEqual(metrics.mean_3d, 7.5)
        self.assertAlmostEqual(metrics.max_3d, 10.0)

    def test_compare_motion_models(self):
        frames = np.arange(20)
        actual = np.zeros((20, 3))

        # Baseline (Kalman): errors ~ 2.0 mm
        baseline_df = pd.DataFrame({
            "frame": frames,
            "actual_x_mm": actual[:, 0],
            "actual_y_mm": actual[:, 1],
            "actual_z_mm": actual[:, 2],
            "predicted_x_mm": np.full(20, 2.0),
            "predicted_y_mm": np.zeros(20),
            "predicted_z_mm": np.zeros(20),
        })

        # Candidate (Ridge): errors ~ 0.5 mm
        candidate_df = pd.DataFrame({
            "frame": frames,
            "actual_x_mm": actual[:, 0],
            "actual_y_mm": actual[:, 1],
            "actual_z_mm": actual[:, 2],
            "predicted_x_mm": np.full(20, 0.5),
            "predicted_y_mm": np.zeros(20),
            "predicted_z_mm": np.zeros(20),
        })

        res = compare_motion_models(
            baseline_df=baseline_df,
            candidate_df=candidate_df,
            baseline_name="Kalman",
            candidate_name="Ridge",
            frame_range=(5, 15),
        )

        self.assertEqual(res.common_frames, 11)  # frames 5 through 15 inclusive
        self.assertEqual(res.winner, "Ridge")
        self.assertGreater(res.p95_reduction_pct, 60.0)
        self.assertIn("HELD-OUT PERFORMANCE COMPARISON", res.summary())

    def test_compare_motion_models_frame_mismatch_raises(self):
        b_df = pd.DataFrame({
            "frame": [1, 2, 3],
            "actual_x_mm": [0, 0, 0], "actual_y_mm": [0, 0, 0], "actual_z_mm": [0, 0, 0],
            "predicted_x_mm": [1, 1, 1], "predicted_y_mm": [1, 1, 1], "predicted_z_mm": [1, 1, 1],
        })
        c_df = pd.DataFrame({
            "frame": [4, 5, 6],  # No overlap
            "actual_x_mm": [0, 0, 0], "actual_y_mm": [0, 0, 0], "actual_z_mm": [0, 0, 0],
            "predicted_x_mm": [1, 1, 1], "predicted_y_mm": [1, 1, 1], "predicted_z_mm": [1, 1, 1],
        })
        with self.assertRaises(RuntimeError):
            compare_motion_models(b_df, c_df)


class TestHeadToHeadModelComparison(unittest.TestCase):
    """End-to-end evaluation of Kalman vs Ridge on a realistic synthetic respiratory motion signal."""

    def test_head_to_head_pipeline(self):
        # 100 frames of sinusoidal motion (tumor moves with breathing)
        t = np.linspace(0, 4 * np.pi, 100)
        rpm = np.sin(t)
        z_pos = 10.0 * np.sin(t)  # 10mm motion
        x_pos = 2.0 * np.cos(t)
        y_pos = 1.0 * np.sin(t)

        df = pd.DataFrame({
            "frame": np.arange(100),
            "rpm": rpm,
            "x_mm": x_pos,
            "y_mm": y_pos,
            "z_mm": z_pos,
        })

        # 1. Train Ridge on frames 0-60
        ridge = RespiratoryRidgePredictor(alpha=0.1, n_lags=3)
        X, y, frames, _ = ridge.create_features(df)
        train_idx = frames <= 60
        test_idx = frames > 60

        ridge.fit(X[train_idx], y[train_idx])
        ridge_results = ridge.predict_sequence(X[test_idx], y[test_idx], frames=frames[test_idx])

        # 2. Run Kalman filter across the sequence
        kf = ConstantVelocityKalmanFilter(dt=0.1)
        positions = df[["x_mm", "y_mm", "z_mm"]].values
        kalman_results = kf.predict_sequence(positions, frames=df["frame"].values)

        # 3. Compare both models using the unified evaluation pipeline on test frames
        comparison_res = compare_motion_models(
            baseline_df=kalman_results,
            candidate_df=ridge_results,
            baseline_name="Kalman",
            candidate_name="Ridge",
            frame_range=(int(frames[test_idx].min()), int(frames[test_idx].max())),
        )

        self.assertGreater(comparison_res.common_frames, 30)
        self.assertIsNotNone(comparison_res.winner)
        self.assertTrue(np.all(comparison_res.comparison_df["kalman_error_3d_mm"] >= 0))
        self.assertTrue(np.all(comparison_res.comparison_df["ridge_error_3d_mm"] >= 0))


if __name__ == "__main__":
    unittest.main()
