from __future__ import annotations

from typing import Any

import numpy as np


def build_section_reference_frame(
    left_fit: dict[str, float],
    right_fit: dict[str, float],
    left_point_xy: np.ndarray,
    right_point_xy: np.ndarray,
) -> dict[str, Any]:
    _, left_normal = reference_basis_from_fit(left_fit)
    _, right_normal = reference_basis_from_fit(right_fit)
    if left_normal is None or right_normal is None:
        return _empty_reference_frame()

    left_normal = np.asarray(left_normal, dtype=np.float32)
    right_normal = np.asarray(right_normal, dtype=np.float32)
    if float(np.dot(left_normal, right_normal)) < 0.0:
        right_normal = -right_normal

    normal = left_normal + right_normal
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 1e-6:
        return _empty_reference_frame()
    normal = normal / normal_norm
    tangent = np.asarray([normal[1], -normal[0]], dtype=np.float32)
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= 1e-6:
        return _empty_reference_frame()
    tangent = tangent / tangent_norm

    origin_xy = 0.5 * (np.asarray(left_point_xy, dtype=np.float32) + np.asarray(right_point_xy, dtype=np.float32))
    return {
        "valid": True,
        "origin_xy": origin_xy.astype(np.float32),
        "tangent": tangent.astype(np.float32),
        "normal": normal.astype(np.float32),
    }


def reference_basis_from_fit(fit: dict[str, float]) -> tuple[np.ndarray | None, np.ndarray | None]:
    slope = float(fit.get("slope", 0.0))
    intercept = float(fit.get("intercept", np.nan))
    if not np.isfinite(intercept):
        return None, None
    tangent = np.asarray([1.0, slope], dtype=np.float32)
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= 1e-6:
        return None, None
    tangent = tangent / tangent_norm
    normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float32)
    return tangent, normal


def transform_xy_to_reference(points_xy: np.ndarray, origin_xy: np.ndarray, fit: dict[str, float]) -> np.ndarray:
    tangent, normal = reference_basis_from_fit(fit)
    if tangent is None or normal is None:
        return np.full_like(points_xy, np.nan, dtype=np.float32)
    deltas = np.asarray(points_xy, dtype=np.float32) - np.asarray(origin_xy, dtype=np.float32)
    tangent_coords = deltas @ tangent
    normal_coords = deltas @ normal
    return np.stack([tangent_coords, normal_coords], axis=1).astype(np.float32)


def transform_plot_points_to_reference(points: dict[str, np.ndarray], origin_xy: np.ndarray, fit: dict[str, float]) -> np.ndarray:
    if points is None or len(points.get("u", [])) == 0:
        return np.empty((0, 2), dtype=np.float32)
    xy = np.stack(
        [
            np.asarray(points["u"], dtype=np.float32).reshape(-1),
            np.asarray(points["z"], dtype=np.float32).reshape(-1),
        ],
        axis=1,
    )
    finite_mask = np.all(np.isfinite(xy), axis=1)
    if not np.any(finite_mask):
        return np.empty((0, 2), dtype=np.float32)
    return transform_xy_to_reference(xy[finite_mask], origin_xy, fit)


def transform_xy_to_frame(points_xy: np.ndarray, frame: dict[str, Any]) -> np.ndarray:
    if not bool(frame.get("valid", False)):
        return np.full_like(points_xy, np.nan, dtype=np.float32)
    deltas = np.asarray(points_xy, dtype=np.float32) - np.asarray(frame["origin_xy"], dtype=np.float32)
    tangent_coords = deltas @ np.asarray(frame["tangent"], dtype=np.float32)
    normal_coords = deltas @ np.asarray(frame["normal"], dtype=np.float32)
    return np.stack([tangent_coords, normal_coords], axis=1).astype(np.float32)


def transform_plot_points_to_frame(points: dict[str, np.ndarray], frame: dict[str, Any]) -> np.ndarray:
    if points is None or len(points.get("u", [])) == 0 or not bool(frame.get("valid", False)):
        return np.empty((0, 2), dtype=np.float32)
    xy = np.stack(
        [
            np.asarray(points["u"], dtype=np.float32).reshape(-1),
            np.asarray(points["z"], dtype=np.float32).reshape(-1),
        ],
        axis=1,
    )
    finite_mask = np.all(np.isfinite(xy), axis=1)
    if not np.any(finite_mask):
        return np.empty((0, 2), dtype=np.float32)
    return transform_xy_to_frame(xy[finite_mask], frame)


def measure_point_line_gap_flush(
    source_point: np.ndarray,
    other_edge_point: np.ndarray,
    fit: dict[str, float],
) -> dict[str, Any]:
    tangent, _ = reference_basis_from_fit(fit)
    if tangent is None:
        return {"gap": np.nan, "flush": np.nan, "foot_point": np.asarray([np.nan, np.nan], dtype=np.float32)}

    intercept = float(fit.get("intercept", np.nan))
    line_anchor = np.asarray([0.0, intercept], dtype=np.float32)
    source_point = np.asarray(source_point, dtype=np.float32)
    other_edge_point = np.asarray(other_edge_point, dtype=np.float32)
    source_delta = source_point - line_anchor
    foot_point = line_anchor + float(np.dot(source_delta, tangent)) * tangent
    flush = float(np.linalg.norm(source_point - foot_point))
    gap = float(np.linalg.norm(other_edge_point - foot_point))
    return {"gap": gap, "flush": flush, "foot_point": foot_point}


def measure_gap_in_reference_frame(
    left_point_xy: np.ndarray,
    right_point_xy: np.ndarray,
    frame: dict[str, Any],
) -> dict[str, Any]:
    if not bool(frame.get("valid", False)):
        return {
            "gap": np.nan,
            "left_point_ref": np.asarray([np.nan, np.nan], dtype=np.float32),
            "right_point_ref": np.asarray([np.nan, np.nan], dtype=np.float32),
            "left_foot_ref": np.asarray([np.nan, np.nan], dtype=np.float32),
            "right_foot_ref": np.asarray([np.nan, np.nan], dtype=np.float32),
        }
    left_point_ref = transform_xy_to_frame(np.asarray([left_point_xy], dtype=np.float32), frame)[0]
    right_point_ref = transform_xy_to_frame(np.asarray([right_point_xy], dtype=np.float32), frame)[0]
    left_foot_ref = np.asarray([left_point_ref[0], 0.0], dtype=np.float32)
    right_foot_ref = np.asarray([right_point_ref[0], 0.0], dtype=np.float32)
    return {
        "gap": float(abs(right_foot_ref[0] - left_foot_ref[0])),
        "left_point_ref": left_point_ref,
        "right_point_ref": right_point_ref,
        "left_foot_ref": left_foot_ref,
        "right_foot_ref": right_foot_ref,
    }


def _empty_reference_frame() -> dict[str, Any]:
    return {
        "valid": False,
        "origin_xy": np.asarray([np.nan, np.nan], dtype=np.float32),
        "tangent": np.asarray([np.nan, np.nan], dtype=np.float32),
        "normal": np.asarray([np.nan, np.nan], dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# 3D reference-plane helpers (scheme B: full-3D gap / flush)
#
# These work in the sensor's world frame (millimetres) and make no
# assumption that the skin is aligned with the sensor xy plane.  The
# seam is decomposed into
#   - gap   : in-surface component of the edge-to-edge vector
#             ( = distance along the averaged skin tangent plane )
#   - flush : out-of-surface component
#             ( = step along the averaged skin normal )
# by projecting onto the average normal of two side-specific plane fits.
# ---------------------------------------------------------------------------


def fit_plane_3d(points_xyz: np.ndarray, min_points: int = 6) -> dict[str, np.ndarray] | None:
    """Fit a plane to ``points_xyz`` (shape ``(N, 3)``) via SVD.

    Returns ``None`` if the input has too few finite points or is too
    degenerate (all points nearly collinear) to define a plane reliably.
    On success, returns a dict with:

    - ``centroid``: ``(3,)`` float32 mean of the input points.
    - ``normal``:   ``(3,)`` float32 unit normal vector. Oriented so the
      +z component is non-negative (sensor looks downward at the skin),
      which keeps the sign of ``flush`` consistent across sections.
    - ``tangents``: ``(2, 3)`` float32 in-plane orthonormal basis whose
      first row is the direction of largest in-plane spread (typically
      the cross-seam direction for seam-edge point sets).
    - ``singular_values``: ``(3,)`` float32 singular values of the
      centred input — useful for plane-quality diagnostics.
    """
    pts = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    finite = np.all(np.isfinite(pts), axis=1)
    pts = pts[finite]
    if len(pts) < max(3, int(min_points)):
        return None
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    try:
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if singular_values.size < 3:
        return None
    # Reject near-isotropic point clouds: a genuine skin patch is much
    # flatter (smallest SV) than it is wide (largest SV). Threshold is
    # permissive — rejects only obviously ambiguous cases.
    if float(singular_values[0]) <= 1e-9:
        return None
    # Reject near-isotropic blobs: a genuine plane has SV[2] much smaller
    # than SV[0] (flat along the normal).
    if float(singular_values[2]) / float(singular_values[0]) > 0.5:
        return None
    # Reject near-collinear sets: SV[1] collapses to zero in that case
    # and the plane orientation becomes ambiguous (any rotation about
    # the dominant axis is equally valid).
    if float(singular_values[1]) / float(singular_values[0]) < 1e-3:
        return None
    normal = vt[2].astype(np.float32)
    norm_len = float(np.linalg.norm(normal))
    if norm_len <= 1e-9:
        return None
    normal = normal / norm_len
    # Orient toward +z so per-section normals are in the same hemisphere.
    if float(normal[2]) < 0.0:
        normal = -normal
    tangents = vt[:2].astype(np.float32)
    return {
        "centroid": centroid.astype(np.float32),
        "normal": normal,
        "tangents": tangents,
        "singular_values": singular_values.astype(np.float32),
    }


_DEFAULT_PLANE_AGREEMENT_MIN = 0.7    # cos(~45°) - normals must agree within 45°
_DEFAULT_AVG_NORMAL_UP_MIN = 0.7      # cos(~45°) - averaged normal within 45° of +z


def measure_gap_flush_3d(
    left_edge_xyz: np.ndarray,
    right_edge_xyz: np.ndarray,
    left_plane: dict[str, np.ndarray],
    right_plane: dict[str, np.ndarray],
    *,
    seam_tangent_3d: np.ndarray | None = None,
    plane_agreement_min: float = _DEFAULT_PLANE_AGREEMENT_MIN,
    avg_normal_up_min: float = _DEFAULT_AVG_NORMAL_UP_MIN,
) -> dict[str, Any]:
    """Decompose the left→right edge vector into an in-surface gap and a
    surface-normal flush, in millimetres.

    Two side-specific planes are fit upstream (``left_plane`` /
    ``right_plane``) and their normals are averaged (with consistent
    sign) to form the reference normal. ``gap`` is the length of the
    component orthogonal to that normal; ``flush`` is the absolute
    value of the parallel component. ``gap² + flush² == ||Δ||²`` by
    Pythagoras, which is checked by ``full_3d_mm`` in the output.

    Two quality guards reject sections whose plane fits are obviously
    unreliable:

    - ``plane_agreement_min`` (default ``0.7`` ≈ ``cos(45°)``): the two
      sides' normals must agree within this cosine, i.e. not tilt away
      from each other by more than 45°. A larger disagreement usually
      means one side's surface-point selection caught a seam-edge
      artefact and tilted the fit.
    - ``avg_normal_up_min`` (default ``0.7`` ≈ ``cos(45°)``): the
      averaged normal must sit within 45° of the sensor +z axis.  For
      typical skin geometry the real normal is at most ~20° off +z;
      anything larger is a fit gone wrong (e.g. PCA axes permuted when
      the point cloud happens to be narrower across than along).

    Guarded failures return ``valid=False`` with a descriptive reason so
    downstream aggregates skip them instead of being poisoned.
    """
    left_xyz = np.asarray(left_edge_xyz, dtype=np.float32).reshape(3)
    right_xyz = np.asarray(right_edge_xyz, dtype=np.float32).reshape(3)
    n_left = np.asarray(left_plane["normal"], dtype=np.float32).reshape(3)
    n_right = np.asarray(right_plane["normal"], dtype=np.float32).reshape(3)
    # Guarantee the two normals sit in the same hemisphere before averaging
    # so the average is not cancelled out by opposite signs.
    if float(np.dot(n_left, n_right)) < 0.0:
        n_right = -n_right
    agreement = float(np.dot(n_left, n_right))
    if agreement < float(plane_agreement_min):
        return {
            "valid": False,
            "reason": f"plane_disagreement:{agreement:.3f}",
            "gap": float("nan"),
            "flush": float("nan"),
        }
    n_sum = n_left + n_right
    norm_sum = float(np.linalg.norm(n_sum))
    if norm_sum <= 1e-6:
        return {
            "valid": False,
            "reason": "degenerate_average_normal",
            "gap": float("nan"),
            "flush": float("nan"),
        }
    n_avg = (n_sum / norm_sum).astype(np.float32)
    # Enforce that the averaged normal points roughly toward the sensor
    # (+z). ``fit_plane_3d`` already flips individual normals to have
    # non-negative z, but the average can still tip if both normals are
    # tilted far off.
    if float(n_avg[2]) < float(avg_normal_up_min):
        return {
            "valid": False,
            "reason": f"avg_normal_not_upright:{float(n_avg[2]):.3f}",
            "gap": float("nan"),
            "flush": float("nan"),
        }

    delta = right_xyz - left_xyz
    parallel = float(np.dot(delta, n_avg))  # signed flush component
    perpendicular = delta - parallel * n_avg  # in-plane (tangent-plane) component
    flush = float(abs(parallel))
    full_3d = float(np.linalg.norm(delta))
    perp_magnitude = float(np.linalg.norm(perpendicular))

    # Decompose ``perpendicular`` (which lies in the tangent plane) into
    # along-seam and cross-seam components when a local seam tangent is
    # supplied. The "gap" we ultimately want to report is the
    # cross-seam component: the along-seam component captures the
    # discretisation wobble introduced by nearest-pixel edge selection
    # (left / right edges are not guaranteed to sit at the same v).
    t_along_plane: np.ndarray | None = None
    t_cross_plane: np.ndarray | None = None
    gap_cross = perp_magnitude
    gap_along_residual = float("nan")
    if seam_tangent_3d is not None:
        t_seam = np.asarray(seam_tangent_3d, dtype=np.float32).reshape(3)
        # Project the requested seam tangent onto the tangent plane to
        # strip any stray normal component, then renormalise.
        t_along_plane = t_seam - float(np.dot(t_seam, n_avg)) * n_avg
        norm_along = float(np.linalg.norm(t_along_plane))
        if norm_along > 1e-6:
            t_along_plane = (t_along_plane / norm_along).astype(np.float32)
            t_cross_plane = np.cross(n_avg, t_along_plane).astype(np.float32)
            along_comp = float(np.dot(perpendicular, t_along_plane))
            cross_comp = float(np.dot(perpendicular, t_cross_plane))
            gap_cross = float(abs(cross_comp))
            gap_along_residual = float(abs(along_comp))

    gap = gap_cross  # authoritative cross-seam opening in mm
    return {
        "valid": bool(np.isfinite(gap) and np.isfinite(flush)),
        "reason": "ok",
        "gap": gap,
        "gap_along_residual": gap_along_residual,
        "gap_perp_full": perp_magnitude,
        "flush": flush,
        "full_3d_mm": full_3d,
        "n_avg": n_avg,
        "t_along_plane": t_along_plane if t_along_plane is not None else np.asarray([np.nan] * 3, dtype=np.float32),
        "t_cross_plane": t_cross_plane if t_cross_plane is not None else np.asarray([np.nan] * 3, dtype=np.float32),
        "delta": delta.astype(np.float32),
        "parallel_component_signed": parallel,
        "plane_agreement_cos": agreement,
        "left_xyz": left_xyz.astype(np.float32),
        "right_xyz": right_xyz.astype(np.float32),
    }
