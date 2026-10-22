import unittest

import numpy as np

from dose_shift_analysis import (
    dose_metrics,
    _dvh_rows,
    patient_to_grid,
    sample_dose,
    shifted_grid_coordinates,
)


class DoseInterpolationTests(unittest.TestCase):
    def setUp(self):
        self.geometry = {
            "origin": np.array([0.0, 0.0, 0.0]),
            "row_direction": np.array([1.0, 0.0, 0.0]),
            "column_direction": np.array([0.0, 1.0, 0.0]),
            "normal": np.array([0.0, 0.0, 1.0]),
            "spacing": np.array([2.0, 1.0]),
            "offsets": np.array([0.0, 2.0, 4.0]),
            "dose": None,
        }
        frames, rows, columns = np.indices((3, 3, 3), dtype=float)
        self.dose = frames * 100.0 + rows * 10.0 + columns
        self.geometry["dose"] = self.dose

    def test_patient_coordinates_map_to_fractional_grid_indices(self):
        points = np.array([[0.5, 1.0, 1.0]])
        coordinates = patient_to_grid(points, self.geometry)
        np.testing.assert_allclose(coordinates[:, 0], [0.5, 0.5, 0.5])

    def test_dose_is_trilinearly_interpolated_for_subvoxel_coordinates(self):
        coordinates = np.array([[0.5], [0.5], [0.5]])
        self.assertAlmostEqual(float(sample_dose(self.dose, coordinates)[0]), 55.5)

    def test_patient_space_displacement_shifts_all_fractional_grid_axes(self):
        coordinates = shifted_grid_coordinates(self.geometry, [0.5, 1.0, 1.0])
        sampled = sample_dose(self.dose, coordinates)
        self.assertAlmostEqual(float(sampled[1, 1, 1]), 166.5)

    def test_metrics_include_target_and_oar_dose_values(self):
        metrics = dose_metrics(np.array([10.0, 20.0, 30.0, 40.0]), 0.01)
        self.assertEqual(metrics["sampled_voxels"], 4)
        self.assertAlmostEqual(metrics["Dmean_Gy"], 25.0)
        self.assertAlmostEqual(metrics["Dmax_Gy"], 40.0)
        self.assertAlmostEqual(metrics["D95_Gy"], 11.5)
        self.assertAlmostEqual(metrics["D98_Gy"], 10.6)
        self.assertIsNone(metrics["D0.1cc_Gy"])
        self.assertIsNone(metrics["D2cc_Gy"])

    def test_dose_to_volume_metrics_are_reported_when_roi_is_large_enough(self):
        metrics = dose_metrics(np.arange(1.0, 21.0), 0.01)
        self.assertEqual(metrics["D0.1cc_Gy"], 11.0)
        self.assertIsNone(metrics["D2cc_Gy"])

    def test_dvh_rows_identify_condition_and_end_at_zero_volume(self):
        rows = _dvh_rows("CTV", "motion-conditioned", [1.0, 2.0], 1.0)
        self.assertEqual(rows[0]["condition"], "motion-conditioned")
        self.assertEqual(rows[-1]["dose_Gy"], 3.0)
        self.assertEqual(rows[-1]["volume_percent"], 0.0)


if __name__ == "__main__":
    unittest.main()
