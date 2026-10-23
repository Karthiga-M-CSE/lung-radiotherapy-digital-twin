"""Unit tests for the geometry validation module.

Tests cover:
- ImageGeometry dataclass properties
- Single-image validation checks (spacing, origin, direction, size)
- Cross-modality comparison (spacing, origin, direction mismatch)
- 4D-CT phase consistency validation
- Physical-to-voxel and voxel-to-physical coordinate transforms
- Contour-within-dose-grid boundary checks
- GeometryReport aggregation and severity tracking
"""

import unittest
import numpy as np

from src.utils.geometry_validation import (
    ImageGeometry,
    GeometryReport,
    ValidationIssue,
    Severity,
    validate_single_image,
    validate_cross_modality,
    validate_4dct_phase_consistency,
    physical_to_voxel,
    voxel_to_physical,
    _check_spacing,
    _check_direction_orthogonality,
    _check_direction_determinant,
    _check_size,
    _check_origin_finite,
    _compare_spacing,
    _compare_origin,
    _compare_direction,
    _check_contours_within_dose_grid,
)


def _make_geometry(
    origin=(0.0, 0.0, 0.0),
    spacing=(1.0, 1.0, 1.0),
    direction=None,
    size=(256, 256, 100),
    source="test_image",
    modality="CT",
):
    if direction is None:
        direction = np.eye(3).flatten()
    return ImageGeometry(
        source=source,
        origin=np.array(origin, dtype=float),
        spacing=np.array(spacing, dtype=float),
        direction=np.array(direction, dtype=float),
        size=size,
        dimension=3,
        modality=modality,
    )


class TestImageGeometry(unittest.TestCase):

    def test_direction_matrix_identity(self):
        geom = _make_geometry()
        mat = geom.direction_matrix
        np.testing.assert_array_almost_equal(mat, np.eye(3))

    def test_direction_matrix_rotation(self):
        angle = np.pi / 4
        rot = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0,              0,             1],
        ])
        geom = _make_geometry(direction=rot.flatten())
        np.testing.assert_array_almost_equal(geom.direction_matrix, rot)

    def test_physical_extent(self):
        geom = _make_geometry(spacing=(0.5, 0.5, 2.0), size=(100, 100, 50))
        extent = geom.physical_extent
        np.testing.assert_array_almost_equal(extent, [50.0, 50.0, 100.0])


class TestSingleImageValidation(unittest.TestCase):

    def test_valid_geometry_passes(self):
        geom = _make_geometry(
            origin=(-100.0, -150.0, 0.0),
            spacing=(0.9766, 0.9766, 2.5),
            size=(512, 512, 120),
        )
        report = validate_single_image(geom)
        self.assertTrue(report.passed)
        self.assertEqual(report.error_count, 0)

    def test_zero_spacing_critical(self):
        geom = _make_geometry(spacing=(0.0, 1.0, 1.0))
        report = GeometryReport()
        _check_spacing(geom, report)
        self.assertFalse(report.passed)
        self.assertTrue(any(
            i.severity == Severity.CRITICAL and "non-positive" in i.message
            for i in report.issues
        ))

    def test_negative_spacing_critical(self):
        geom = _make_geometry(spacing=(-1.0, 1.0, 1.0))
        report = GeometryReport()
        _check_spacing(geom, report)
        self.assertFalse(report.passed)

    def test_unusually_small_spacing_warning(self):
        geom = _make_geometry(spacing=(0.01, 1.0, 1.0))
        report = GeometryReport()
        _check_spacing(geom, report)
        self.assertTrue(report.passed)
        self.assertTrue(any(i.severity == Severity.WARNING for i in report.issues))

    def test_unusually_large_spacing_warning(self):
        geom = _make_geometry(spacing=(1.0, 1.0, 25.0))
        report = GeometryReport()
        _check_spacing(geom, report)
        self.assertTrue(any(
            i.severity == Severity.WARNING and "large" in i.message
            for i in report.issues
        ))

    def test_non_orthogonal_direction_error(self):
        bad_dir = np.array([
            [1.0, 0.5, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        geom = _make_geometry(direction=bad_dir.flatten())
        report = GeometryReport()
        _check_direction_orthogonality(geom, report)
        self.assertFalse(report.passed)

    def test_left_handed_direction_warning(self):
        left_handed = np.diag([1.0, 1.0, -1.0])
        geom = _make_geometry(direction=left_handed.flatten())
        report = GeometryReport()
        _check_direction_determinant(geom, report)
        self.assertTrue(any(i.severity == Severity.WARNING for i in report.issues))

    def test_zero_size_critical(self):
        geom = _make_geometry(size=(0, 256, 100))
        report = GeometryReport()
        _check_size(geom, report)
        self.assertFalse(report.passed)

    def test_non_finite_origin_critical(self):
        geom = _make_geometry(origin=(float("nan"), 0.0, 0.0))
        report = GeometryReport()
        _check_origin_finite(geom, report)
        self.assertFalse(report.passed)

    def test_infinite_origin_critical(self):
        geom = _make_geometry(origin=(float("inf"), 0.0, 0.0))
        report = GeometryReport()
        _check_origin_finite(geom, report)
        self.assertFalse(report.passed)


class TestCrossModalityValidation(unittest.TestCase):

    def test_matching_geometries_pass(self):
        geom_a = _make_geometry(spacing=(1.0, 1.0, 2.5), origin=(-128.0, -128.0, 0.0))
        geom_b = _make_geometry(spacing=(1.0, 1.0, 2.5), origin=(-128.0, -128.0, 0.0))
        report = validate_cross_modality(geom_a, geom_b)
        self.assertTrue(report.passed)

    def test_spacing_mismatch_warning(self):
        geom_a = _make_geometry(spacing=(1.0, 1.0, 2.5))
        geom_b = _make_geometry(spacing=(1.0, 1.0, 3.0))
        report = GeometryReport()
        _compare_spacing(geom_a, geom_b, report, tolerance_mm=0.01)
        self.assertTrue(any(
            i.check == "cross_spacing_match" for i in report.issues
        ))

    def test_origin_small_mismatch_warning(self):
        geom_a = _make_geometry(origin=(0.0, 0.0, 0.0))
        geom_b = _make_geometry(origin=(1.0, 0.0, 0.0))
        report = GeometryReport()
        _compare_origin(geom_a, geom_b, report, tolerance_mm=0.5)
        self.assertTrue(any(
            i.check == "cross_origin_match" and i.severity == Severity.WARNING
            for i in report.issues
        ))

    def test_origin_large_mismatch_error(self):
        geom_a = _make_geometry(origin=(0.0, 0.0, 0.0))
        geom_b = _make_geometry(origin=(10.0, 0.0, 0.0))
        report = GeometryReport()
        _compare_origin(geom_a, geom_b, report, tolerance_mm=0.5)
        self.assertTrue(any(
            i.check == "cross_origin_match" and i.severity == Severity.ERROR
            for i in report.issues
        ))

    def test_direction_mismatch_error(self):
        dir_a = np.eye(3).flatten()
        angle = np.pi / 6
        rot = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0,              0,             1],
        ])
        geom_a = _make_geometry(direction=dir_a)
        geom_b = _make_geometry(direction=rot.flatten())
        report = GeometryReport()
        _compare_direction(geom_a, geom_b, report, tolerance=1e-3)
        self.assertFalse(report.passed)


class TestFourDCTValidation(unittest.TestCase):

    def test_consistent_phases_pass(self):
        phases = [
            _make_geometry(source=f"phase_{i}", spacing=(0.9766, 0.9766, 2.5), size=(512, 512, 120))
            for i in range(10)
        ]
        report = validate_4dct_phase_consistency(phases)
        self.assertTrue(report.passed)

    def test_size_mismatch_error(self):
        phases = [
            _make_geometry(source="phase_0", size=(512, 512, 120)),
            _make_geometry(source="phase_1", size=(512, 512, 110)),
        ]
        report = validate_4dct_phase_consistency(phases)
        self.assertFalse(report.passed)
        self.assertTrue(any("size" in i.check for i in report.issues))

    def test_spacing_mismatch_error(self):
        phases = [
            _make_geometry(source="phase_0", spacing=(1.0, 1.0, 2.5)),
            _make_geometry(source="phase_1", spacing=(1.0, 1.0, 2.6)),
        ]
        report = validate_4dct_phase_consistency(phases, spacing_tolerance_mm=0.001)
        self.assertFalse(report.passed)

    def test_single_phase_warning(self):
        phases = [_make_geometry(source="phase_0")]
        report = validate_4dct_phase_consistency(phases)
        self.assertTrue(any(i.severity == Severity.WARNING for i in report.issues))


class TestContourWithinDoseGrid(unittest.TestCase):

    def test_contour_inside_grid(self):
        dose_geom = _make_geometry(
            origin=(0.0, 0.0, 0.0),
            spacing=(1.0, 1.0, 2.5),
            size=(100, 100, 50),
        )
        contour_bounds = {
            "min_bound": np.array([10.0, 10.0, 5.0]),
            "max_bound": np.array([80.0, 80.0, 100.0]),
        }
        report = GeometryReport()
        _check_contours_within_dose_grid(contour_bounds, dose_geom, report)
        self.assertTrue(report.passed)

    def test_contour_outside_grid_error(self):
        dose_geom = _make_geometry(
            origin=(0.0, 0.0, 0.0),
            spacing=(1.0, 1.0, 2.5),
            size=(100, 100, 50),
        )
        contour_bounds = {
            "min_bound": np.array([-20.0, 10.0, 5.0]),
            "max_bound": np.array([80.0, 80.0, 200.0]),
        }
        report = GeometryReport()
        _check_contours_within_dose_grid(contour_bounds, dose_geom, report)
        self.assertFalse(report.passed)
        contour_issues = [i for i in report.issues if i.check == "contour_within_dose"]
        self.assertGreaterEqual(len(contour_issues), 2)


class TestCoordinateTransforms(unittest.TestCase):

    def test_physical_to_voxel_identity(self):
        origin = np.array([0.0, 0.0, 0.0])
        spacing = np.array([1.0, 1.0, 1.0])
        physical = np.array([[5.0, 10.0, 15.0]])

        voxel = physical_to_voxel(physical, origin, spacing)
        np.testing.assert_array_almost_equal(voxel, [[5.0, 10.0, 15.0]])

    def test_physical_to_voxel_with_spacing(self):
        origin = np.array([0.0, 0.0, 0.0])
        spacing = np.array([2.0, 2.0, 5.0])
        physical = np.array([[10.0, 20.0, 50.0]])

        voxel = physical_to_voxel(physical, origin, spacing)
        np.testing.assert_array_almost_equal(voxel, [[5.0, 10.0, 10.0]])

    def test_physical_to_voxel_with_origin(self):
        origin = np.array([10.0, 20.0, 30.0])
        spacing = np.array([1.0, 1.0, 1.0])
        physical = np.array([[15.0, 25.0, 35.0]])

        voxel = physical_to_voxel(physical, origin, spacing)
        np.testing.assert_array_almost_equal(voxel, [[5.0, 5.0, 5.0]])

    def test_roundtrip_identity(self):
        origin = np.array([-100.0, -150.0, 0.0])
        spacing = np.array([0.9766, 0.9766, 2.5])
        direction = np.eye(3)
        physical_orig = np.array([[50.0, -30.0, 100.0], [-10.0, 20.0, 50.0]])

        voxel = physical_to_voxel(physical_orig, origin, spacing, direction)
        physical_back = voxel_to_physical(voxel, origin, spacing, direction)
        np.testing.assert_array_almost_equal(physical_back, physical_orig)

    def test_roundtrip_with_rotation(self):
        origin = np.array([0.0, 0.0, 0.0])
        spacing = np.array([1.0, 1.0, 2.0])
        angle = np.pi / 4
        direction = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0,              0,             1],
        ])
        physical_orig = np.array([[10.0, 5.0, 8.0]])

        voxel = physical_to_voxel(physical_orig, origin, spacing, direction)
        physical_back = voxel_to_physical(voxel, origin, spacing, direction)
        np.testing.assert_array_almost_equal(physical_back, physical_orig)


class TestGeometryReport(unittest.TestCase):

    def test_empty_report_passes(self):
        report = GeometryReport()
        self.assertTrue(report.passed)
        self.assertEqual(report.error_count, 0)
        self.assertEqual(report.warning_count, 0)

    def test_info_does_not_fail(self):
        report = GeometryReport()
        report.add(ValidationIssue(check="test", severity=Severity.INFO, message="ok"))
        self.assertTrue(report.passed)

    def test_warning_does_not_fail(self):
        report = GeometryReport()
        report.add(ValidationIssue(check="test", severity=Severity.WARNING, message="warn"))
        self.assertTrue(report.passed)
        self.assertEqual(report.warning_count, 1)

    def test_error_fails(self):
        report = GeometryReport()
        report.add(ValidationIssue(check="test", severity=Severity.ERROR, message="err"))
        self.assertFalse(report.passed)
        self.assertEqual(report.error_count, 1)

    def test_critical_fails(self):
        report = GeometryReport()
        report.add(ValidationIssue(check="test", severity=Severity.CRITICAL, message="crit"))
        self.assertFalse(report.passed)
        self.assertEqual(report.error_count, 1)

    def test_summary_format(self):
        report = GeometryReport()
        report.images_validated.append("test.nii.gz")
        report.add(ValidationIssue(check="a", severity=Severity.WARNING, message="w"))
        report.add(ValidationIssue(check="b", severity=Severity.ERROR, message="e"))
        summary = report.summary()
        self.assertIn("FAILED", summary)
        self.assertIn("Errors", summary)
        self.assertIn("Warnings", summary)


if __name__ == "__main__":
    unittest.main()
