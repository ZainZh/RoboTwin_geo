"""Camera-only task-frame estimation for the geometry-marker target.

The task deliberately renders a yellow T-shaped raised marker.  This module
uses its known primitive geometry, not simulator pose, to recover a proper
world SE(3) frame from the segmented XYZRGB cloud.  The implementation is a
task-specific perception baseline; a learned keypoint detector can replace it
without changing the downstream flow model.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize


def yellow_marker_mask(xyzrgb: np.ndarray) -> np.ndarray:
    value = np.asarray(xyzrgb, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] < 6:
        raise ValueError(f"expected (N,C>=6) XYZRGB cloud, got {value.shape}")
    color = value[:, 3:6]
    return color[:, 0] + color[:, 1] - 2.0 * color[:, 2] > 0.65


def _rectangle_distance_squared(
    points: np.ndarray, half_x: float, half_y: float, center_x: float
) -> np.ndarray:
    dx = np.maximum(np.abs(points[:, 0] - float(center_x)) - float(half_x), 0.0)
    dy = np.maximum(np.abs(points[:, 1]) - float(half_y), 0.0)
    return dx * dx + dy * dy


def _t_distance_squared(points_local: np.ndarray) -> np.ndarray:
    longitudinal = _rectangle_distance_squared(points_local, 0.050, 0.007, 0.0)
    crossbar = _rectangle_distance_squared(points_local, 0.007, 0.035, 0.043)
    return np.minimum(longitudinal, crossbar)


def _t_template() -> np.ndarray:
    long_x = np.linspace(-0.050, 0.050, 21)
    long_y = np.linspace(-0.007, 0.007, 5)
    cross_x = np.linspace(0.036, 0.050, 5)
    cross_y = np.linspace(-0.035, 0.035, 15)
    first = np.stack(np.meshgrid(long_x, long_y, indexing="ij"), axis=-1).reshape(-1, 2)
    second = np.stack(np.meshgrid(cross_x, cross_y, indexing="ij"), axis=-1).reshape(-1, 2)
    return np.unique(np.round(np.concatenate((first, second), axis=0), 6), axis=0)


T_TEMPLATE = _t_template()


def _fit_t_pose(points_2d: np.ndarray) -> tuple[np.ndarray, float, float]:
    points = np.asarray(points_2d, dtype=np.float64)
    if len(points) < 8:
        raise ValueError(f"at least 8 marker points are required, got {len(points)}")

    def transform_to_local(parameters: np.ndarray) -> np.ndarray:
        center = parameters[:2]
        angle = float(parameters[2])
        cosine, sine = math.cos(angle), math.sin(angle)
        rotation = np.asarray(((cosine, -sine), (sine, cosine)))
        return (points - center[None]) @ rotation

    def objective(parameters: np.ndarray) -> float:
        local = transform_to_local(parameters)
        observed_cost = float(np.mean(_t_distance_squared(local)))
        angle = float(parameters[2])
        cosine, sine = math.cos(angle), math.sin(angle)
        rotation = np.asarray(((cosine, -sine), (sine, cosine)))
        template_world = T_TEMPLATE @ rotation.T + parameters[:2]
        nearest = np.sum(
            (template_world[:, None] - points[None]) ** 2, axis=-1
        ).min(axis=1)
        # The sparse camera sample need not cover the complete template, hence
        # the asymmetric but nonzero coverage term.
        return observed_cost + 0.15 * float(np.mean(nearest))

    # The area centroid of the union of both marker rectangles is about 15 mm
    # along +X.  Starting from all yaw hypotheses makes the T sign observable.
    initial_candidates = []
    for angle in np.linspace(-math.pi, math.pi, 48, endpoint=False):
        offset = np.asarray((0.0154 * math.cos(angle), 0.0154 * math.sin(angle)))
        initial = np.asarray((*((points.mean(axis=0) - offset).tolist()), angle))
        initial_candidates.append((objective(initial), initial))
    candidates = []
    for _, initial in sorted(initial_candidates, key=lambda item: item[0])[:6]:
        fitted = minimize(
            objective,
            initial,
            method="Powell",
            options={"maxiter": 80, "xtol": 1e-5, "ftol": 1e-8},
        )
        candidates.append((float(fitted.fun), fitted.x))
    score, parameters = min(candidates, key=lambda item: item[0])
    angle = float((parameters[2] + math.pi) % (2.0 * math.pi) - math.pi)
    return parameters[:2], angle, score


def estimate_geometry_marker_frame(xyzrgb: np.ndarray) -> tuple[np.ndarray, dict]:
    """Return marker-local-to-world transform and camera-only diagnostics."""
    cloud = np.asarray(xyzrgb, dtype=np.float64)
    mask = yellow_marker_mask(cloud)
    marker = cloud[mask, :3]
    if len(marker) < 8:
        raise ValueError(f"insufficient yellow marker observations: {len(marker)}")

    xyz = cloud[:, :3]
    centered = xyz - xyz.mean(axis=0)
    _, vectors = np.linalg.eigh(centered.T @ centered / max(len(xyz), 1))
    normal = vectors[:, 0]
    if normal[2] < 0.0:
        normal = -normal
    world_x = np.asarray((1.0, 0.0, 0.0))
    plane_x = world_x - normal * float(np.dot(world_x, normal))
    plane_x /= np.linalg.norm(plane_x)
    plane_y = np.cross(normal, plane_x)
    plane_y /= np.linalg.norm(plane_y)
    plane_origin = marker.mean(axis=0)
    marker_2d = np.stack(
        (
            (marker - plane_origin) @ plane_x,
            (marker - plane_origin) @ plane_y,
        ),
        axis=-1,
    )
    center_2d, angle, score = _fit_t_pose(marker_2d)
    cosine, sine = math.cos(angle), math.sin(angle)
    axis_x = cosine * plane_x + sine * plane_y
    axis_y = -sine * plane_x + cosine * plane_y
    center = plane_origin + center_2d[0] * plane_x + center_2d[1] * plane_y
    # Move the fit from visible top surface to the functional plane.  This
    # constant affects only the frame origin, not the recovered orientation.
    center = center - 0.006 * normal
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.stack((axis_x, axis_y, normal), axis=1)
    transform[:3, 3] = center
    return transform.astype(np.float32), {
        "marker_points": int(len(marker)),
        "fit_score_m2": float(score),
        "orthogonality_error": float(
            np.max(np.abs(transform[:3, :3].T @ transform[:3, :3] - np.eye(3)))
        ),
        "determinant": float(np.linalg.det(transform[:3, :3])),
    }
