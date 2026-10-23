"""Unit tests for the tumour trajectory extraction and validation module.

Tests cover:
    1. Frame ID parsing from filenames and path patterns
    2. 3D tumor centroid and bounding box computation from binary masks
    3. Coordinate transformation (voxel to physical mm using affine matrix)
    4. Kinematic feature computation (velocities, accelerations, 3D speed)
    5. Principal motion axis extraction via PCA
    6. Missing frame interpolation
    7. Full trajectory generation pipeline
    8. Trajectory validation against image geometry (FOV bounds, unphysical displacement)
    9. Trajectory validation against respiratory surrogate (correlation, phase lag)
    10. Trajectory and respiratory data merging
    11. File-based sequence extraction using synthetic NIfTI masks
"""

import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from src.data.trajectory import (
    TumorCentroid,
    TrajectoryValidationResult,
    TumorTrajectoryResult,
    get_frame_id,
    compute_tumor_centroid_from_mask,
    extract_tumor_centroid_from_file,
    extract_trajectory_from_masks,
    compute_kinematics,
    compute_principal_motion_axis,
    interpolate_missing_frames,
    generate_physical_trajectory,
    validate_trajectory_against_geometry,
    validate_trajectory_against_respiratory,
    combine_trajectory_with_respiratory,
    validate_tumor_trajectory,
)
from src.utils.geometry_validation import ImageGeometry, Severity


class TestFrameIdParsing(unittest.TestCase):
    """Test extracting frame numbers from various filename conventions."""

    def test_standard_tumormask_patterns(self):
        self.assertEqual(get_frame_id("tumormask_0.nii.gz"), 0)
        self.assertEqual(get_frame_id("tumormask_181.nii.gz"), 181)
        self.assertEqual(get_frame_id(Path("path/to/tumormask_42.nii")), 42)

    def test_alternative_patterns(self):
        self.assertEqual(get_frame_id("frame_05.nii.gz"), 5)
        self.assertEqual(get_frame_id("mask_12.nii"), 12)
        self.assertEqual(get_frame_id("patient1_slice_99.nii"), 1)  # first group of digits

    def test_invalid_filename(self):
        with self.assertRaises(ValueError):
            get_frame_id("no_numbers_here.nii.gz")


class TestCentroidComputation(unittest.TestCase):
    """Test 3D centroid calculation from binary masks and affines."""

    def setUp(self):
        self.mask_shape = (20, 20, 20)
        self.mask = np.zeros(self.mask_shape, dtype=np.uint8)
        # Create a 4x4x4 cube of tumor voxels centered at (10, 10, 10)
        self.mask[9:13, 9:13, 9:13] = 1

        # Affine with 2.0 mm voxel spacing and origin at (10.0, 20.0, 30.0)
        self.affine = np.array([
            [2.0, 0.0, 0.0, 10.0],
            [0.0, 2.0, 0.0, 20.0],
            [0.0, 0.0, 2.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

    def test_empty_mask_returns_none(self):
        empty_mask = np.zeros((10, 10, 10), dtype=np.uint8)
        centroid = compute_tumor_centroid_from_mask(empty_mask, affine=self.affine)
        self.assertIsNone(centroid)

    def test_voxel_centroid_accuracy(self):
        centroid = compute_tumor_centroid_from_mask(self.mask, affine=self.affine, frame=5)
        self.assertIsNotNone(centroid)
        self.assertEqual(centroid.frame, 5)
        # Indices 9, 10, 11, 12 mean is 10.5
        np.testing.assert_allclose(centroid.voxel_centroid, [10.5, 10.5, 10.5])
        self.assertEqual(centroid.tumor_voxels, 4 * 4 * 4)

    def test_physical_centroid_accuracy(self):
        centroid = compute_tumor_centroid_from_mask(self.mask, affine=self.affine)
        # Expected physical: origin + voxel * spacing = 10.0 + 10.5 * 2.0 = 31.0
        expected_physical = [31.0, 41.0, 51.0]
        np.testing.assert_allclose(centroid.physical_centroid_mm, expected_physical)

    def test_volume_and_bounding_box(self):
        centroid = compute_tumor_centroid_from_mask(self.mask, affine=self.affine)
        # Voxel volume is 2 * 2 * 2 = 8.0 mm3; 64 voxels * 8 = 512.0 mm3
        self.assertAlmostEqual(centroid.tumor_volume_mm3, 512.0)
        # Bbox in voxels: 4 x 4 x 4; in mm: 8 x 8 x 8
        self.assertEqual(centroid.bbox_voxels, (4.0, 4.0, 4.0))
        self.assertEqual(centroid.bbox_mm, (8.0, 8.0, 8.0))

    def test_fallback_spacing_and_origin(self):
        centroid = compute_tumor_centroid_from_mask(
            self.mask,
            affine=None,
            spacing=(1.5, 1.5, 3.0),
            origin=(5.0, 5.0, 5.0),
        )
        self.assertIsNotNone(centroid)
        np.testing.assert_allclose(centroid.voxel_centroid, [10.5, 10.5, 10.5])
        expected_phys = [5.0 + 10.5 * 1.5, 5.0 + 10.5 * 1.5, 5.0 + 10.5 * 3.0]
        np.testing.assert_allclose(centroid.physical_centroid_mm, expected_phys)


class TestKinematicsAndPrincipalAxis(unittest.TestCase):
    """Test velocity, acceleration, and PCA principal motion axis extraction."""

    def setUp(self):
        # 100 frames of sinusoidal motion predominantly along Z (Superior-Inferior)
        t = np.linspace(0, 4 * np.pi, 100)
        self.df = pd.DataFrame({
            "frame": np.arange(100),
            "x_mm": 10.0 + 1.0 * np.sin(t),
            "y_mm": 20.0 + 2.0 * np.sin(t),
            "z_mm": 50.0 + 15.0 * np.sin(t),  # 15 mm amplitude in Z
            "x_voxel": 5.0 + 0.5 * np.sin(t),
            "y_voxel": 10.0 + 1.0 * np.sin(t),
            "z_voxel": 25.0 + 7.5 * np.sin(t),
            "tumor_voxels": 200,
            "tumor_volume_mm3": 1600.0,
        })
        self.df["dx_mm"] = self.df["x_mm"] - self.df["x_mm"].iloc[0]
        self.df["dy_mm"] = self.df["y_mm"] - self.df["y_mm"].iloc[0]
        self.df["dz_mm"] = self.df["z_mm"] - self.df["z_mm"].iloc[0]
        self.df["displacement_3d_mm"] = np.sqrt(
            self.df["dx_mm"] ** 2 + self.df["dy_mm"] ** 2 + self.df["dz_mm"] ** 2
        )

    def test_compute_kinematics(self):
        kin = compute_kinematics(self.df, dt=0.1)
        self.assertIn("vx_mm_s", kin.columns)
        self.assertIn("vz_mm_s", kin.columns)
        self.assertIn("speed_mm_s", kin.columns)
        self.assertIn("accel_3d_mm_s2", kin.columns)
        # Z velocity should have the highest peak velocity
        self.assertGreater(kin["vz_mm_s"].abs().max(), kin["vx_mm_s"].abs().max())
        self.assertTrue(np.all(kin["speed_mm_s"] >= 0))
        self.assertTrue(np.all(kin["accel_3d_mm_s2"] >= 0))

    def test_compute_principal_motion_axis(self):
        axis_vec, proj, var_exp = compute_principal_motion_axis(self.df)
        # Primary motion is along Z, so Z component of axis vector should be close to 1.0
        self.assertGreater(abs(axis_vec[2]), 0.95)
        # Dominant axis should explain > 95% of total motion variance
        self.assertGreater(var_exp, 0.95)
        self.assertEqual(len(proj), len(self.df))

    def test_interpolate_missing_frames(self):
        # Drop every 3rd frame
        incomplete = self.df[self.df["frame"] % 3 != 0].copy()
        self.assertLess(len(incomplete), 100)
        interpolated = interpolate_missing_frames(incomplete, full_frame_range=(1, 98))
        self.assertEqual(len(interpolated), 98)
        self.assertEqual(list(interpolated["frame"]), list(range(1, 99)))

    def test_generate_physical_trajectory_pipeline(self):
        processed = generate_physical_trajectory(self.df, dt=0.1)
        self.assertIn("principal_motion_mm", processed.columns)
        self.assertIn("speed_mm_s", processed.columns)
        self.assertEqual(len(processed), len(self.df))


class TestTrajectoryGeometryValidation(unittest.TestCase):
    """Test validation of trajectories against ImageGeometry and physiological limits."""

    def setUp(self):
        self.geom = ImageGeometry(
            source="test_ct.nii",
            origin=np.array([0.0, 0.0, 0.0]),
            spacing=np.array([1.0, 1.0, 2.0]),
            direction=np.eye(3).flatten(),
            size=(256, 256, 100),
            dimension=3,
        )

    def test_valid_trajectory_passes(self):
        df = pd.DataFrame({
            "frame": np.arange(20),
            "x_mm": np.full(20, 120.0),
            "y_mm": np.full(20, 120.0),
            "z_mm": 50.0 + 5.0 * np.sin(np.linspace(0, 2 * np.pi, 20)),
            "x_voxel": np.full(20, 120.0),
            "y_voxel": np.full(20, 120.0),
            "z_voxel": 25.0 + 2.5 * np.sin(np.linspace(0, 2 * np.pi, 20)),
            "tumor_voxels": np.full(20, 100),
            "tumor_volume_mm3": np.full(20, 200.0),
            "displacement_3d_mm": 5.0 * np.abs(np.sin(np.linspace(0, 2 * np.pi, 20))),
            "speed_mm_s": np.full(20, 2.0),
        })
        res = validate_trajectory_against_geometry(df, image_geometry=self.geom)
        self.assertTrue(res.passed)
        self.assertTrue(res.geometry_passed)
        self.assertEqual(res.error_count, 0)

    def test_trajectory_exceeding_fov_fails(self):
        # Coordinates outside [0, 256] mm
        df = pd.DataFrame({
            "frame": np.arange(5),
            "x_mm": np.array([100.0, 200.0, 300.0, 400.0, 500.0]),  # > 256 mm FOV
            "y_mm": np.full(5, 50.0),
            "z_mm": np.full(5, 50.0),
            "displacement_3d_mm": np.arange(5) * 5.0,
        })
        res = validate_trajectory_against_geometry(df, image_geometry=self.geom)
        self.assertFalse(res.passed)
        self.assertFalse(res.geometry_passed)
        self.assertTrue(any(i.check == "geom_fov_exceeded" for i in res.issues))

    def test_unphysiological_displacement_triggers_error(self):
        # 80 mm displacement is unphysiologically large for lung tumors
        df = pd.DataFrame({
            "frame": np.arange(10),
            "x_mm": np.full(10, 50.0),
            "y_mm": np.full(10, 50.0),
            "z_mm": 50.0 + np.linspace(0, 80.0, 10),
            "displacement_3d_mm": np.linspace(0, 80.0, 10),
        })
        res = validate_trajectory_against_geometry(df, image_geometry=self.geom)
        self.assertFalse(res.passed)
        self.assertTrue(any(i.check == "geom_displacement_unphysical" for i in res.issues))

    def test_volume_instability_warning(self):
        # Tumor volume wildly oscillating
        df = pd.DataFrame({
            "frame": np.arange(10),
            "x_mm": np.full(10, 50.0),
            "y_mm": np.full(10, 50.0),
            "z_mm": np.full(10, 50.0),
            "displacement_3d_mm": np.full(10, 0.0),
            "tumor_voxels": np.array([10, 500, 15, 600, 12, 550, 10, 650, 11, 600]),
        })
        res = validate_trajectory_against_geometry(df, image_geometry=self.geom)
        self.assertTrue(any(i.check == "geom_volume_instability" for i in res.issues))


class TestTrajectoryRespiratoryValidation(unittest.TestCase):
    """Test validation and synchronization against respiratory surrogate signals."""

    def setUp(self):
        t = np.linspace(0, 4 * np.pi, 50)
        # Respiratory signal is sinusoidal
        self.rpm = np.sin(t)
        # Tumor motion strongly correlated with respiration
        self.traj = pd.DataFrame({
            "frame": np.arange(50),
            "x_mm": np.zeros(50),
            "y_mm": np.zeros(50),
            "z_mm": 10.0 * np.sin(t),
            "dz_mm": 10.0 * np.sin(t),
            "displacement_3d_mm": 10.0 * np.abs(np.sin(t)),
            "principal_motion_mm": 10.0 * np.sin(t),
        })

    def test_high_correlation_passes(self):
        res = validate_trajectory_against_respiratory(self.traj, self.rpm, min_correlation=0.6)
        self.assertTrue(res.passed)
        self.assertTrue(res.respiratory_passed)
        self.assertAlmostEqual(res.metrics["pearson_dz"], 1.0, places=3)
        self.assertTrue(any(i.check == "resp_correlation_ok" for i in res.issues))

    def test_low_correlation_warns(self):
        # Random noise surrogate with no correlation to tumor motion
        rng = np.random.default_state = np.random.RandomState(42)
        noise_rpm = rng.randn(50)
        res = validate_trajectory_against_respiratory(self.traj, noise_rpm, min_correlation=0.7)
        self.assertTrue(any(i.check == "resp_low_correlation" for i in res.issues))

    def test_flatline_signal_fails(self):
        flat_rpm = np.ones(50)
        res = validate_trajectory_against_respiratory(self.traj, flat_rpm)
        self.assertFalse(res.passed)
        self.assertFalse(res.respiratory_passed)
        self.assertTrue(any(i.check == "resp_flat_signal" for i in res.issues))

    def test_combine_trajectory_with_respiratory(self):
        combined = combine_trajectory_with_respiratory(self.traj, self.rpm, rpm_col="rpm")
        self.assertIn("rpm", combined.columns)
        self.assertEqual(len(combined), len(self.traj))
        np.testing.assert_allclose(combined["rpm"], self.rpm)

    def test_resampling_mismatched_length(self):
        # Respiratory has 100 samples, trajectory has 50
        rpm_100 = np.sin(np.linspace(0, 4 * np.pi, 100))
        combined = combine_trajectory_with_respiratory(self.traj, rpm_100, rpm_col="rpm")
        self.assertEqual(len(combined), 50)
        # Preserves correlation after interpolation
        corr = combined["rpm"].corr(combined["dz_mm"])
        self.assertGreater(corr, 0.95)


class TestFileExtractionIntegration(unittest.TestCase):
    """Integration test extracting a multi-frame trajectory from synthetic NIfTI files."""

    def test_extract_trajectory_from_synthetic_nifti_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            affine = np.eye(4)
            affine[0, 0] = 2.0
            affine[1, 1] = 2.0
            affine[2, 2] = 2.5

            # Create 5 synthetic frames with translating tumor
            for frame_idx in range(5):
                mask = np.zeros((30, 30, 30), dtype=np.uint8)
                # Shift tumor along Z: center moves from 10 to 14
                z_start = 10 + frame_idx
                mask[10:15, 10:15, z_start:z_start + 5] = 1
                img = nib.Nifti1Image(mask, affine)
                nib.save(img, str(tmp_path / f"tumormask_{frame_idx}.nii.gz"))

            traj = extract_trajectory_from_masks(
                mask_source=tmp_path,
                pattern="tumormask_*.nii.gz",
                reference_frame=0,
            )

            self.assertEqual(len(traj), 5)
            self.assertEqual(list(traj["frame"]), [0, 1, 2, 3, 4])
            # Each frame tumor moves 1 voxel = 2.5 mm along Z
            expected_dz = np.array([0.0, 2.5, 5.0, 7.5, 10.0])
            np.testing.assert_allclose(traj["dz_mm"].values, expected_dz, atol=1e-3)
            np.testing.assert_allclose(traj["displacement_3d_mm"].values, expected_dz, atol=1e-3)

            # Full multi-modal validation
            full_val = validate_tumor_trajectory(traj)
            self.assertTrue(full_val.passed)


if __name__ == "__main__":
    unittest.main()
