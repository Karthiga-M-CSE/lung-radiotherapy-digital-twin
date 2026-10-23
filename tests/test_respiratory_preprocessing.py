"""Unit tests for the respiratory preprocessing module.

Tests cover all six preprocessing stages:
    1. Butterworth lowpass and bandpass filtering
    2. Timestamp synchronization via interpolation
    3. Uniform resampling of non-uniform signals
    4. Respiratory cycle detection (peaks, valleys, cycles)
    5. Inhale / exhale phase extraction and labeling
    6. Kinematic features (velocity, acceleration, jerk, amplitude)
    7. Lag feature construction
    8. Full pipeline integration
"""

import unittest
import numpy as np
import pandas as pd

from src.data.respiratory import (
    butterworth_lowpass,
    butterworth_bandpass,
    synchronize_timestamps,
    resample_uniform,
    detect_respiratory_cycles,
    extract_respiratory_phases,
    compute_kinematic_features,
    compute_motion_amplitude_per_cycle,
    create_lagged_dataframe,
    run_preprocessing_pipeline,
)


def _make_sine_signal(n=200, freq_hz=0.25, fs_hz=10.0, noise_std=0.0):
    """Generate a sine-wave respiratory surrogate with optional noise."""
    t = np.arange(n) / fs_hz
    signal = np.sin(2 * np.pi * freq_hz * t)
    if noise_std > 0:
        rng = np.random.default_rng(42)
        signal += rng.normal(0, noise_std, n)
    return t, signal


def _make_positions(n=200, freq_hz=0.25, fs_hz=10.0):
    """Generate synthetic 3-axis tumor positions following sinusoidal motion."""
    t = np.arange(n) / fs_hz
    x = 0.5 * np.sin(2 * np.pi * freq_hz * t)
    y = 0.3 * np.cos(2 * np.pi * freq_hz * t)
    z = 5.0 * np.sin(2 * np.pi * freq_hz * t)
    return np.column_stack([x, y, z])



class TestButterworthLowpass(unittest.TestCase):

    def test_preserves_length(self):
        _, sig = _make_sine_signal(200)
        filtered = butterworth_lowpass(sig, cutoff_hz=2.0, fs_hz=10.0)
        self.assertEqual(len(filtered), len(sig))

    def test_removes_high_frequency_noise(self):
        _, sig = _make_sine_signal(500, freq_hz=0.25, fs_hz=10.0, noise_std=0.3)
        filtered = butterworth_lowpass(sig, cutoff_hz=1.0, fs_hz=10.0)
        self.assertLess(np.std(np.diff(filtered)), np.std(np.diff(sig)))

    def test_cutoff_above_nyquist_raises(self):
        _, sig = _make_sine_signal(200)
        with self.assertRaises(ValueError):
            butterworth_lowpass(sig, cutoff_hz=6.0, fs_hz=10.0)

    def test_short_signal_raises(self):
        sig = np.array([1.0, 2.0])
        with self.assertRaises(ValueError):
            butterworth_lowpass(sig, cutoff_hz=1.0, fs_hz=10.0)


class TestButterworthBandpass(unittest.TestCase):

    def test_preserves_length(self):
        _, sig = _make_sine_signal(300)
        filtered = butterworth_bandpass(sig, 0.1, 0.5, fs_hz=10.0)
        self.assertEqual(len(filtered), len(sig))

    def test_invalid_bounds_raise(self):
        _, sig = _make_sine_signal(300)
        with self.assertRaises(ValueError):
            butterworth_bandpass(sig, 0.5, 0.1, fs_hz=10.0)
        with self.assertRaises(ValueError):
            butterworth_bandpass(sig, -0.1, 0.5, fs_hz=10.0)
        with self.assertRaises(ValueError):
            butterworth_bandpass(sig, 0.1, 5.5, fs_hz=10.0)



class TestSynchronizeTimestamps(unittest.TestCase):

    def test_identity_when_same_timestamps(self):
        t = np.linspace(0, 10, 100)
        vals = np.sin(t)
        synced = synchronize_timestamps(t, vals, t)
        np.testing.assert_allclose(synced, vals, atol=1e-10)

    def test_interpolates_to_coarser_grid(self):
        t_rpm = np.linspace(0, 10, 1000)
        vals = np.sin(t_rpm)
        t_ct = np.linspace(0, 10, 50)
        synced = synchronize_timestamps(t_rpm, vals, t_ct)
        self.assertEqual(len(synced), 50)
        np.testing.assert_allclose(synced, np.sin(t_ct), atol=0.01)

    def test_mismatched_lengths_raise(self):
        with self.assertRaises(ValueError):
            synchronize_timestamps(np.arange(10), np.arange(5), np.arange(10))

    def test_non_monotonic_raises(self):
        t = np.array([0, 2, 1, 3])
        with self.assertRaises(ValueError):
            synchronize_timestamps(t, np.arange(4), np.arange(4))



class TestResampleUniform(unittest.TestCase):

    def test_resample_to_n_samples(self):
        t = np.sort(np.random.default_rng(0).uniform(0, 10, 50))
        v = np.sin(t)
        t_new, v_new = resample_uniform(t, v, n_samples=100)
        self.assertEqual(len(t_new), 100)
        self.assertEqual(len(v_new), 100)
        self.assertAlmostEqual(t_new[0], t[0])
        self.assertAlmostEqual(t_new[-1], t[-1])

    def test_resample_to_target_fs(self):
        t = np.linspace(0, 5, 50)
        v = np.sin(2 * np.pi * 0.5 * t)
        t_new, v_new = resample_uniform(t, v, target_fs_hz=20.0)
        actual_fs = (len(t_new) - 1) / (t_new[-1] - t_new[0])
        self.assertAlmostEqual(actual_fs, 20.0, places=0)

    def test_multidim_resample(self):
        t = np.linspace(0, 5, 50)
        v = np.column_stack([np.sin(t), np.cos(t)])
        t_new, v_new = resample_uniform(t, v, n_samples=100)
        self.assertEqual(v_new.shape, (100, 2))

    def test_no_params_raises(self):
        with self.assertRaises(ValueError):
            resample_uniform(np.arange(10), np.arange(10))



class TestDetectRespiratoryCycles(unittest.TestCase):

    def test_finds_peaks_in_sine(self):
        _, sig = _make_sine_signal(400, freq_hz=0.25, fs_hz=10.0)
        peaks, valleys, cycles = detect_respiratory_cycles(sig, distance=15)
        self.assertGreater(len(peaks), 2)
        self.assertGreater(len(valleys), 2)
        self.assertGreater(len(cycles), 1)

    def test_cycle_columns(self):
        _, sig = _make_sine_signal(400, freq_hz=0.25, fs_hz=10.0)
        _, _, cycles = detect_respiratory_cycles(sig, distance=15)
        required_cols = {"cycle", "start_frame", "end_frame", "num_frames",
                         "peak_amplitude", "valley_amplitude", "cycle_range"}
        self.assertTrue(required_cols.issubset(set(cycles.columns)))

    def test_cycles_are_non_overlapping(self):
        _, sig = _make_sine_signal(400, freq_hz=0.25, fs_hz=10.0)
        _, _, cycles = detect_respiratory_cycles(sig, distance=15)
        for i in range(len(cycles) - 1):
            self.assertGreaterEqual(
                cycles.iloc[i + 1]["start_frame"],
                cycles.iloc[i]["end_frame"]
            )



class TestExtractRespiratoryPhases(unittest.TestCase):

    def test_phase_labels_assigned(self):
        _, sig = _make_sine_signal(400, freq_hz=0.25, fs_hz=10.0)
        peaks, valleys, _ = detect_respiratory_cycles(sig, distance=15)
        phases = extract_respiratory_phases(sig, peaks, valleys)

        self.assertEqual(len(phases.phase_label), len(sig))
        assigned = np.sum(phases.phase_label != "unassigned")
        self.assertGreater(assigned, len(sig) * 0.5)

    def test_has_both_inhale_and_exhale(self):
        _, sig = _make_sine_signal(400, freq_hz=0.25, fs_hz=10.0)
        peaks, valleys, _ = detect_respiratory_cycles(sig, distance=15)
        phases = extract_respiratory_phases(sig, peaks, valleys)

        unique_labels = set(phases.phase_label)
        self.assertIn("inhale", unique_labels)
        self.assertIn("exhale", unique_labels)

    def test_phase_fraction_bounds(self):
        _, sig = _make_sine_signal(400, freq_hz=0.25, fs_hz=10.0)
        peaks, valleys, _ = detect_respiratory_cycles(sig, distance=15)
        phases = extract_respiratory_phases(sig, peaks, valleys)

        valid = ~np.isnan(phases.phase_fraction)
        self.assertTrue(np.all(phases.phase_fraction[valid] >= 0.0))
        self.assertTrue(np.all(phases.phase_fraction[valid] <= 1.0 + 1e-10))

    def test_cycle_ids_positive_when_assigned(self):
        _, sig = _make_sine_signal(400, freq_hz=0.25, fs_hz=10.0)
        peaks, valleys, _ = detect_respiratory_cycles(sig, distance=15)
        phases = extract_respiratory_phases(sig, peaks, valleys)

        assigned = phases.cycle_id[phases.phase_label != "unassigned"]
        self.assertTrue(np.all(assigned >= 0))



class TestKinematicFeatures(unittest.TestCase):

    def test_output_keys(self):
        positions = _make_positions(100)
        kin = compute_kinematic_features(positions, dt=0.1)
        expected_keys = {"velocity", "acceleration", "jerk", "speed", "abs_accel", "amplitude"}
        self.assertEqual(set(kin.keys()), expected_keys)

    def test_output_shapes_3d(self):
        positions = _make_positions(100)
        kin = compute_kinematic_features(positions, dt=0.1)
        self.assertEqual(kin["velocity"].shape, (100, 3))
        self.assertEqual(kin["acceleration"].shape, (100, 3))
        self.assertEqual(kin["jerk"].shape, (100, 3))
        self.assertEqual(kin["speed"].shape, (100,))
        self.assertEqual(kin["abs_accel"].shape, (100,))
        self.assertEqual(kin["amplitude"].shape, (100,))

    def test_output_shapes_1d(self):
        positions = np.sin(np.linspace(0, 4 * np.pi, 100))
        kin = compute_kinematic_features(positions, dt=0.1)
        self.assertEqual(kin["velocity"].shape, (100,))
        self.assertEqual(kin["acceleration"].shape, (100,))

    def test_stationary_has_zero_velocity(self):
        positions = np.ones((50, 3)) * 10.0
        kin = compute_kinematic_features(positions, dt=1.0)
        np.testing.assert_allclose(kin["velocity"], 0.0, atol=1e-10)
        np.testing.assert_allclose(kin["acceleration"], 0.0, atol=1e-10)
        np.testing.assert_allclose(kin["speed"], 0.0, atol=1e-10)

    def test_linear_motion_constant_velocity(self):
        positions = np.column_stack([
            np.linspace(0, 10, 50),
            np.zeros(50),
            np.zeros(50),
        ])
        kin = compute_kinematic_features(positions, dt=1.0)
        v_x = kin["velocity"][5:-5, 0]
        np.testing.assert_allclose(v_x, v_x[0], atol=1e-10)

    def test_amplitude_starts_at_zero(self):
        positions = _make_positions(100)
        kin = compute_kinematic_features(positions, dt=0.1)
        self.assertAlmostEqual(kin["amplitude"][0], 0.0, places=10)

    def test_smooth_window_reduces_noise(self):
        rng = np.random.default_rng(42)
        positions = np.sin(np.linspace(0, 4 * np.pi, 200)) + rng.normal(0, 0.1, 200)
        kin_raw = compute_kinematic_features(positions, dt=0.1)
        kin_smooth = compute_kinematic_features(positions, dt=0.1, smooth_window=5)
        self.assertLess(
            np.std(kin_smooth["velocity"]),
            np.std(kin_raw["velocity"]) * 1.1
        )


class TestMotionAmplitudePerCycle(unittest.TestCase):

    def test_per_cycle_output(self):
        positions = np.sin(np.linspace(0, 4 * np.pi, 100)).reshape(-1, 1)
        cycle_ids = np.zeros(100, dtype=int)
        cycle_ids[:50] = 1
        cycle_ids[50:] = 2

        result = compute_motion_amplitude_per_cycle(positions, cycle_ids)
        self.assertIn(1, result)
        self.assertIn(2, result)
        self.assertIn("range", result[1])
        self.assertIn("peak_to_peak", result[1])
        self.assertIn("std", result[1])
        self.assertIn("num_samples", result[1])

    def test_ignores_unassigned(self):
        positions = np.ones((20, 3))
        cycle_ids = np.full(20, -1, dtype=int)
        result = compute_motion_amplitude_per_cycle(positions, cycle_ids)
        self.assertEqual(len(result), 0)



class TestLaggedDataframe(unittest.TestCase):

    def test_correct_lag_count(self):
        df = pd.DataFrame({
            "frame": range(20),
            "x": np.random.default_rng(0).random(20),
            "y": np.random.default_rng(1).random(20),
        })
        result, lag_cols = create_lagged_dataframe(df, ["x", "y"], n_lags=3)
        self.assertEqual(len(lag_cols), 6)
        self.assertEqual(len(result), 20 - 3)

    def test_lag_columns_named_correctly(self):
        df = pd.DataFrame({"frame": range(10), "z": np.arange(10, dtype=float)})
        _, lag_cols = create_lagged_dataframe(df, ["z"], n_lags=2)
        self.assertIn("z_lag1", lag_cols)
        self.assertIn("z_lag2", lag_cols)

    def test_lag_values_correct(self):
        df = pd.DataFrame({"frame": range(10), "v": np.arange(10, dtype=float)})
        result, _ = create_lagged_dataframe(df, ["v"], n_lags=1)
        expected_lag1 = result["v"] - 1
        np.testing.assert_array_equal(result["v_lag1"].values, expected_lag1.values)



class TestFullPipeline(unittest.TestCase):

    def test_pipeline_runs_without_filter(self):
        positions = _make_positions(200)
        _, rpm = _make_sine_signal(200)
        result = run_preprocessing_pipeline(positions, rpm)

        self.assertEqual(len(result.df), 200)
        self.assertGreater(len(result.peaks), 0)
        self.assertGreater(len(result.cycles), 0)
        self.assertIn("rpm_raw", result.df.columns)
        self.assertIn("velocity_x", result.df.columns)
        self.assertIn("accel_z", result.df.columns)
        self.assertIn("speed", result.df.columns)
        self.assertIn("phase", result.df.columns)
        self.assertIn("cycle_id", result.df.columns)
        self.assertIn("amplitude", result.df.columns)

    def test_pipeline_runs_with_filter(self):
        positions = _make_positions(200, fs_hz=10.0)
        _, rpm = _make_sine_signal(200, fs_hz=10.0, noise_std=0.2)
        result = run_preprocessing_pipeline(
            positions, rpm,
            filter_cutoff_hz=1.0, filter_fs_hz=10.0,
        )
        self.assertTrue(result.metadata["filter_applied"])
        self.assertEqual(len(result.df), 200)

    def test_pipeline_metadata(self):
        positions = _make_positions(200)
        _, rpm = _make_sine_signal(200)
        result = run_preprocessing_pipeline(positions, rpm)
        self.assertEqual(result.metadata["n_samples"], 200)
        self.assertIn("n_cycles", result.metadata)
        self.assertIn("dt", result.metadata)

    def test_pipeline_with_custom_frames(self):
        positions = _make_positions(100)
        _, rpm = _make_sine_signal(100)
        frames = np.arange(10, 110)
        result = run_preprocessing_pipeline(positions, rpm, frames=frames)
        np.testing.assert_array_equal(result.df["frame"].values, frames)


if __name__ == "__main__":
    unittest.main()
