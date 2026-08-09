"""Core utilities for extracting table-relative grasp candidates from GRAB."""

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class ContactInterval:
    """A half-open interval containing hand-object contact."""

    segment_index: int
    start_frame: int
    end_frame: int

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame


@dataclass(frozen=True)
class CandidateFrame:
    """The representative frame selected from a contact interval."""

    frame_index: int
    selection_reason: str
    translation_from_interval_start_m: float
    rotation_from_interval_start_rad: float
    contact_vertex_count: int


@dataclass(frozen=True)
class TabletopMetrics:
    """Geometry measurements in the posed table coordinate frame."""

    table_thickness_axis: int
    table_normal_sign: float
    surface_distance_m: float
    signed_bottom_gap_m: float
    signed_top_height_m: float
    footprint_overlap_fraction: float
    near_surface_vertex_count: int
    considered_vertex_count: int
    surface_crossing: bool

    def to_dict(self) -> Dict[str, object]:
        """Return JSON-compatible metric values."""

        values = asdict(self)
        return {
            key: None if isinstance(value, float) and not np.isfinite(value) else value for key, value in values.items()
        }


def detect_contact_intervals(
    contact_labels: np.ndarray,
    hand_contact_values: Iterable[int],
) -> Tuple[List[ContactInterval], np.ndarray]:
    """Find continuous intervals where sampled object vertices contact a hand.

    Args:
        contact_labels: Per-frame contact labels with shape ``(T, V)``.
        hand_contact_values: GRAB contact IDs belonging to either hand.

    Returns:
        The ordered half-open intervals and per-frame contacted-vertex counts.
    """

    labels = np.asarray(contact_labels)
    if labels.ndim != 2:
        raise ValueError(f"contact_labels must have shape (T, V), got {labels.shape}")

    hand_values = np.asarray(tuple(hand_contact_values), dtype=labels.dtype)
    hand_mask = np.isin(labels, hand_values)
    contact_counts = hand_mask.sum(axis=1, dtype=np.int64)
    contacting = contact_counts > 0
    changes = np.diff(contacting.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    intervals = [
        ContactInterval(segment_index=index, start_frame=int(start), end_frame=int(end))
        for index, (start, end) in enumerate(zip(starts, ends))
    ]
    return intervals, contact_counts


def select_candidate_frame(
    translations: np.ndarray,
    global_orientations: np.ndarray,
    contact_counts: np.ndarray,
    interval: ContactInterval,
    translation_threshold_m: float,
) -> CandidateFrame:
    """Apply the ``segment_grasps_2.py`` pre-motion selection rule."""

    start, end = interval.start_frame, interval.end_frame
    segment_translations = np.asarray(translations)[start:end]
    segment_orientations = np.asarray(global_orientations)[start:end]
    segment_counts = np.asarray(contact_counts)[start:end]
    if segment_translations.shape[0] == 0:
        raise ValueError("Cannot select a candidate from an empty interval")

    displacement = np.linalg.norm(segment_translations - segment_translations[0], axis=1)
    moving = np.flatnonzero(displacement > translation_threshold_m)
    if moving.size:
        local_index = max(0, int(moving[0]) - 1)
        reason = "before_translation_threshold"
    else:
        local_index = int(np.argmax(segment_counts))
        reason = "max_contact_fallback"

    start_rotation = Rotation.from_rotvec(segment_orientations[0]).as_matrix()
    candidate_rotation = Rotation.from_rotvec(segment_orientations[local_index]).as_matrix()
    relative_rotation = start_rotation.T @ candidate_rotation
    rotation_displacement = float(Rotation.from_matrix(relative_rotation).magnitude())

    return CandidateFrame(
        frame_index=start + local_index,
        selection_reason=reason,
        translation_from_interval_start_m=float(displacement[local_index]),
        rotation_from_interval_start_rad=rotation_displacement,
        contact_vertex_count=int(segment_counts[local_index]),
    )


def finger_group_contacts(
    contact_labels: np.ndarray,
    contact_id_to_name: Mapping[int, str],
) -> Dict[str, List[bool]]:
    """Convert GRAB contact labels to the legacy five-finger group flags."""

    names = [contact_id_to_name[int(value)] for value in np.asarray(contact_labels) if int(value) > 0]
    finger_names = ("Thumb", "Index", "Middle", "Ring", "Pinky")
    return {
        side: [any(f"{prefix}_{finger}" in name for name in names) for finger in finger_names]
        for side, prefix in (("left", "L"), ("right", "R"))
    }


def pose_matrix(global_orient: Sequence[float], transl: Sequence[float]) -> np.ndarray:
    """Build the column-vector pose convention used by formatted GRAB records."""

    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = Rotation.from_rotvec(np.asarray(global_orient)).as_matrix().T
    pose[:3, 3] = np.asarray(transl)
    return pose


def transform_vertices(
    vertices: np.ndarray,
    global_orient: Sequence[float],
    transl: Sequence[float],
) -> np.ndarray:
    """Pose row-vector mesh vertices using the convention in ``ObjectModel``."""

    rotation = Rotation.from_rotvec(np.asarray(global_orient)).as_matrix()
    return np.asarray(vertices) @ rotation + np.asarray(transl)


def compute_tabletop_metrics(
    object_vertices: np.ndarray,
    object_global_orient: Sequence[float],
    object_transl: Sequence[float],
    table_vertices: np.ndarray,
    table_global_orient: Sequence[float],
    table_transl: Sequence[float],
    footprint_margin_m: float,
    near_surface_band_m: float,
    world_up: Sequence[float] = (0.0, 0.0, 1.0),
) -> TabletopMetrics:
    """Measure object proximity to the upper surface of a posed table mesh.

    The table's thinnest local axis is treated as its surface normal. Its sign is
    chosen so that the normal points toward GRAB world up. A mesh crossing the
    surface has zero surface distance even when no discrete vertex lies exactly
    on the plane, which avoids rejecting through-plane meshes solely because of
    their lowest vertex.
    """

    object_vertices = np.asarray(object_vertices, dtype=np.float64)
    table_vertices = np.asarray(table_vertices, dtype=np.float64)
    if object_vertices.ndim != 2 or object_vertices.shape[1] != 3:
        raise ValueError(f"object_vertices must have shape (V, 3), got {object_vertices.shape}")
    if table_vertices.ndim != 2 or table_vertices.shape[1] != 3:
        raise ValueError(f"table_vertices must have shape (V, 3), got {table_vertices.shape}")

    table_extent = np.ptp(table_vertices, axis=0)
    thickness_axis = int(np.argmin(table_extent))
    footprint_axes = [axis for axis in range(3) if axis != thickness_axis]

    table_rotation = Rotation.from_rotvec(np.asarray(table_global_orient)).as_matrix()
    local_normal_world = table_rotation[thickness_axis]
    normal_dot_up = float(np.dot(local_normal_world, np.asarray(world_up, dtype=np.float64)))
    if abs(normal_dot_up) < 0.5:
        raise ValueError(
            f"The thinnest table axis is not sufficiently aligned with GRAB world up: dot={normal_dot_up:.6f}"
        )
    normal_sign = 1.0 if normal_dot_up > 0 else -1.0
    surface_coordinate = float(np.max(normal_sign * table_vertices[:, thickness_axis]))

    object_world = transform_vertices(object_vertices, object_global_orient, object_transl)
    table_translation = np.asarray(table_transl, dtype=np.float64)
    object_table_local = (object_world - table_translation) @ table_rotation.T

    inside_footprint = np.ones(object_table_local.shape[0], dtype=bool)
    for axis in footprint_axes:
        lower = float(np.min(table_vertices[:, axis]) - footprint_margin_m)
        upper = float(np.max(table_vertices[:, axis]) + footprint_margin_m)
        inside_footprint &= (object_table_local[:, axis] >= lower) & (object_table_local[:, axis] <= upper)

    considered = object_table_local[inside_footprint]
    overlap_fraction = float(np.mean(inside_footprint))
    if considered.shape[0] == 0:
        return TabletopMetrics(
            table_thickness_axis=thickness_axis,
            table_normal_sign=normal_sign,
            surface_distance_m=float("inf"),
            signed_bottom_gap_m=float("inf"),
            signed_top_height_m=float("inf"),
            footprint_overlap_fraction=overlap_fraction,
            near_surface_vertex_count=0,
            considered_vertex_count=0,
            surface_crossing=False,
        )

    signed_height = normal_sign * considered[:, thickness_axis] - surface_coordinate
    signed_bottom = float(np.min(signed_height))
    signed_top = float(np.max(signed_height))
    crossing = signed_bottom <= 0.0 <= signed_top
    surface_distance = 0.0 if crossing else float(np.min(np.abs(signed_height)))

    return TabletopMetrics(
        table_thickness_axis=thickness_axis,
        table_normal_sign=normal_sign,
        surface_distance_m=surface_distance,
        signed_bottom_gap_m=signed_bottom,
        signed_top_height_m=signed_top,
        footprint_overlap_fraction=overlap_fraction,
        near_surface_vertex_count=int(np.sum(np.abs(signed_height) <= near_surface_band_m)),
        considered_vertex_count=int(considered.shape[0]),
        surface_crossing=crossing,
    )


def calibrated_table_distance_threshold(
    base_threshold_m: float,
    start_anchor_distance_m: float,
    anchor_margin_m: float,
    max_calibrated_threshold_m: float,
) -> float:
    """Calibrate the distance threshold with GRAB's supported first frame."""

    if not np.isfinite(start_anchor_distance_m):
        return float(base_threshold_m)
    calibrated = min(start_anchor_distance_m + anchor_margin_m, max_calibrated_threshold_m)
    return float(max(base_threshold_m, calibrated))


def tabletop_rejection_reasons(
    metrics: TabletopMetrics,
    max_surface_distance_m: float,
    min_footprint_overlap_fraction: float,
) -> List[str]:
    """Return machine-readable reasons why a candidate is not tabletop-like."""

    reasons = []
    if not np.isfinite(metrics.surface_distance_m):
        reasons.append("no_object_vertices_in_table_footprint")
    elif metrics.surface_distance_m > max_surface_distance_m:
        reasons.append("table_surface_too_far")
    if metrics.footprint_overlap_fraction < min_footprint_overlap_fraction:
        reasons.append("insufficient_table_footprint_overlap")
    if np.isfinite(metrics.signed_top_height_m) and metrics.signed_top_height_m < -max_surface_distance_m:
        reasons.append("object_below_table_surface")
    return reasons
