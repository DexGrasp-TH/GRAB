import unittest

import numpy as np

from grab.tabletop_grasps import (
    ContactInterval,
    calibrated_table_distance_threshold,
    compute_tabletop_metrics,
    detect_contact_intervals,
    finger_group_contacts,
    select_candidate_frame,
    tabletop_rejection_reasons,
)


def box_vertices(minimum, maximum):
    """Return the eight vertices of an axis-aligned box."""

    minimum = np.asarray(minimum, dtype=np.float64)
    maximum = np.asarray(maximum, dtype=np.float64)
    return np.asarray(
        [
            [x, y, z]
            for x in (minimum[0], maximum[0])
            for y in (minimum[1], maximum[1])
            for z in (minimum[2], maximum[2])
        ]
    )


class ContactSegmentationTest(unittest.TestCase):
    def test_detects_all_half_open_intervals(self):
        labels = np.zeros((9, 3), dtype=np.int64)
        labels[1:3, 0] = 21
        labels[5:8, 1] = 41

        intervals, counts = detect_contact_intervals(labels, [21, 41])

        self.assertEqual(
            intervals,
            [ContactInterval(0, 1, 3), ContactInterval(1, 5, 8)],
        )
        np.testing.assert_array_equal(counts, [0, 1, 1, 0, 0, 1, 1, 1, 0])

    def test_selects_frame_before_translation_threshold(self):
        translations = np.zeros((5, 3), dtype=np.float64)
        translations[:, 0] = [0.0, 0.004, 0.009, 0.011, 0.020]
        orientations = np.zeros((5, 3), dtype=np.float64)
        counts = np.asarray([1, 2, 4, 5, 1])

        candidate = select_candidate_frame(
            translations,
            orientations,
            counts,
            ContactInterval(0, 0, 5),
            0.01,
        )

        self.assertEqual(candidate.frame_index, 2)
        self.assertEqual(candidate.selection_reason, "before_translation_threshold")
        self.assertAlmostEqual(candidate.translation_from_interval_start_m, 0.009)

    def test_falls_back_to_maximum_contact(self):
        translations = np.zeros((4, 3), dtype=np.float64)
        orientations = np.zeros((4, 3), dtype=np.float64)
        counts = np.asarray([1, 4, 2, 3])

        candidate = select_candidate_frame(
            translations,
            orientations,
            counts,
            ContactInterval(0, 0, 4),
            0.01,
        )

        self.assertEqual(candidate.frame_index, 1)
        self.assertEqual(candidate.selection_reason, "max_contact_fallback")

    def test_groups_contact_ids_by_finger(self):
        id_to_name = {
            21: "L_Hand",
            26: "L_Index1",
            40: "L_Thumb3",
            44: "R_Middle1",
        }

        contacts = finger_group_contacts(np.asarray([21, 26, 40, 44, 0]), id_to_name)

        self.assertEqual(contacts["left"], [True, True, False, False, False])
        self.assertEqual(contacts["right"], [False, False, True, False, False])


class TabletopGeometryTest(unittest.TestCase):
    def setUp(self):
        self.table_vertices = box_vertices((-1.0, -0.01, -1.0), (1.0, 0.01, 1.0))
        self.table_transl = np.zeros(3)
        self.object_vertices = box_vertices((-0.1, -0.1, 0.01), (0.1, 0.1, 0.21))

    def metrics(self, object_vertices=None, object_transl=(0.0, 0.0, 0.0), table_rotation=(-np.pi / 2, 0, 0)):
        return compute_tabletop_metrics(
            self.object_vertices if object_vertices is None else object_vertices,
            (0.0, 0.0, 0.0),
            object_transl,
            self.table_vertices,
            table_rotation,
            self.table_transl,
            footprint_margin_m=0.0,
            near_surface_band_m=0.002,
        )

    def test_supported_object_has_zero_surface_distance(self):
        metrics = self.metrics()

        self.assertAlmostEqual(metrics.surface_distance_m, 0.0, places=7)
        self.assertGreater(metrics.footprint_overlap_fraction, 0.9)
        self.assertEqual(tabletop_rejection_reasons(metrics, 0.01, 0.01), [])

    def test_lifted_object_is_rejected(self):
        metrics = self.metrics(object_transl=(0.0, 0.0, 0.05))

        self.assertAlmostEqual(metrics.surface_distance_m, 0.05, places=7)
        self.assertIn("table_surface_too_far", tabletop_rejection_reasons(metrics, 0.01, 0.01))

    def test_table_normal_sign_flip_keeps_same_upper_surface(self):
        positive_normal = self.metrics(table_rotation=(-np.pi / 2, 0, 0))
        negative_normal = self.metrics(table_rotation=(np.pi / 2, 0, 0))

        self.assertEqual(positive_normal.table_normal_sign, 1.0)
        self.assertEqual(negative_normal.table_normal_sign, -1.0)
        self.assertAlmostEqual(positive_normal.surface_distance_m, negative_normal.surface_distance_m, places=7)

    def test_through_plane_mesh_is_measured_as_crossing(self):
        object_vertices = box_vertices((-0.1, -0.1, -0.05), (0.1, 0.1, 0.05))
        metrics = self.metrics(object_vertices=object_vertices)

        self.assertTrue(metrics.surface_crossing)
        self.assertEqual(metrics.surface_distance_m, 0.0)

    def test_object_outside_table_footprint_is_rejected(self):
        metrics = self.metrics(object_transl=(2.0, 0.0, 0.0))
        reasons = tabletop_rejection_reasons(metrics, 0.01, 0.01)

        self.assertIn("no_object_vertices_in_table_footprint", reasons)
        self.assertIn("insufficient_table_footprint_overlap", reasons)
        self.assertIsNone(metrics.to_dict()["surface_distance_m"])

    def test_anchor_calibration_is_bounded(self):
        self.assertAlmostEqual(calibrated_table_distance_threshold(0.01, 0.014, 0.002, 0.02), 0.016)
        self.assertAlmostEqual(calibrated_table_distance_threshold(0.01, 0.1, 0.002, 0.02), 0.02)


if __name__ == "__main__":
    unittest.main()
