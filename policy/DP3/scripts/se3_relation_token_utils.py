from __future__ import annotations

from collections.abc import Mapping

import numpy as np


RELATION_TOKEN_KEY = "se3_relation_token_A_to_B"
RELATION_TOKEN_DIM = 11
TABLE_RELATION_ROUTES = ("oracle", "ndf_no_direction", "ndf_direction")
OBSERVATION_RELATION_ROUTES = ("ndf_observation_goal",)
SUPPORTED_RELATION_ROUTES = TABLE_RELATION_ROUTES + OBSERVATION_RELATION_ROUTES


def functional_goal_a_from_b(functional_a: np.ndarray, functional_b: np.ndarray) -> np.ndarray:
    functional_a = np.asarray(functional_a, dtype=np.float64).reshape(4, 4)
    functional_b = np.asarray(functional_b, dtype=np.float64).reshape(4, 4)
    return functional_a @ np.linalg.inv(functional_b)


def infer_placement_phase(
    initial_object_z: float,
    current_object_z: float,
    *,
    gripper_closed: bool,
    lift_threshold_m: float = 0.03,
) -> float:
    lifted = float(current_object_z) - float(initial_object_z) >= float(lift_threshold_m)
    return float(bool(gripper_closed) and lifted)


def pose7_to_matrix(pose: np.ndarray) -> np.ndarray:
    value = np.asarray(pose, dtype=np.float64)
    if value.shape == (4, 4):
        return value.copy()
    value = value.reshape(-1)
    if value.size != 7:
        raise ValueError(f"pose must contain xyz + quaternion(wxyz), got shape {np.asarray(pose).shape}")

    translation = value[:3]
    quaternion = value[3:]
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("pose quaternion must be non-zero")
    w, x, y, z = quaternion / norm
    rotation = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def rotation_to_6d(rotation: np.ndarray) -> np.ndarray:
    value = np.asarray(rotation, dtype=np.float64)
    if value.shape != (3, 3):
        raise ValueError(f"rotation must be 3x3, got {value.shape}")
    return value[:, :2].T.reshape(6)


def current_a_from_b(object_pose_a: np.ndarray, object_pose_b: np.ndarray) -> np.ndarray:
    world_from_a = pose7_to_matrix(object_pose_a)
    world_from_b = pose7_to_matrix(object_pose_b)
    return np.linalg.inv(world_from_a) @ world_from_b


def relation_correction_in_b(
    object_pose_a: np.ndarray,
    object_pose_b: np.ndarray,
    goal_a_from_b: np.ndarray,
) -> np.ndarray:
    current_b_from_a = np.linalg.inv(current_a_from_b(object_pose_a, object_pose_b))
    goal_b_from_a = np.linalg.inv(np.asarray(goal_a_from_b, dtype=np.float64).reshape(4, 4))
    return goal_b_from_a @ np.linalg.inv(current_b_from_a)


def build_se3_relation_token(
    *,
    object_pose_a: np.ndarray,
    object_pose_b: np.ndarray,
    goal_a_from_b: np.ndarray | None,
    phase_gate: float,
    solver_energy: float,
    confidence: float,
) -> np.ndarray:
    token = np.zeros((RELATION_TOKEN_DIM,), dtype=np.float32)
    gate = float(np.clip(float(phase_gate), 0.0, 1.0)) * float(
        np.clip(float(confidence), 0.0, 1.0)
    )
    if goal_a_from_b is None or gate <= 0.0 or not np.isfinite(float(solver_energy)):
        return token

    correction = relation_correction_in_b(object_pose_a, object_pose_b, goal_a_from_b)
    if not np.all(np.isfinite(correction)):
        return token
    token[:3] = correction[:3, 3].astype(np.float32) * gate
    token[3:9] = rotation_to_6d(correction[:3, :3]).astype(np.float32) * gate
    token[9] = np.float32(float(solver_energy) * gate)
    token[10] = np.float32(gate)
    return token


def _scalar(value, *, cast):
    array = np.asarray(value).reshape(-1)
    if array.size == 0:
        raise ValueError("expected a scalar value")
    return cast(array[0])


def _goal_entries(route: str, goal_table: Mapping) -> Mapping:
    routes = goal_table.get("routes") if isinstance(goal_table, Mapping) else None
    if isinstance(routes, Mapping) and route in routes:
        route_data = routes[route]
        if isinstance(route_data, Mapping):
            return route_data.get("goals", route_data)
    if isinstance(goal_table, Mapping) and "goals" in goal_table:
        return goal_table["goals"]
    return goal_table


def resolve_relation_goal(
    *,
    route: str,
    task_state: Mapping,
    goal_table: Mapping | None,
) -> tuple[np.ndarray | None, float, float]:
    route = str(route)
    if route not in SUPPORTED_RELATION_ROUTES:
        raise ValueError(f"unsupported relation route {route!r}; expected one of {SUPPORTED_RELATION_ROUTES}")
    if route in OBSERVATION_RELATION_ROUTES:
        raise ValueError(
            f"route={route} must be resolved from object point clouds by "
            "GeometryRelationEstimator, not from task_state or a goal table"
        )
    if route == "oracle":
        value = task_state.get("goal_T_A_from_B_oracle")
        if value is None:
            return None, float("inf"), 0.0
        return np.asarray(value, dtype=np.float64).reshape(4, 4), 0.0, 1.0

    if goal_table is None:
        return None, float("inf"), 0.0
    shoe_id_value = task_state.get("shoe_id")
    if shoe_id_value is None:
        return None, float("inf"), 0.0
    shoe_id = _scalar(shoe_id_value, cast=int)
    entry = _goal_entries(route, goal_table).get(str(shoe_id))
    if not isinstance(entry, Mapping):
        return None, float("inf"), 0.0
    transform = entry.get("goal_T_A_from_B")
    if transform is None:
        return None, float("inf"), 0.0
    return (
        np.asarray(transform, dtype=np.float64).reshape(4, 4),
        float(entry.get("solver_energy", 0.0)),
        float(entry.get("confidence", 1.0)),
    )


def build_relation_token_from_task_state(
    *,
    route: str,
    task_state: Mapping,
    goal_table: Mapping | None,
) -> np.ndarray:
    goal, solver_energy, confidence = resolve_relation_goal(
        route=route,
        task_state=task_state,
        goal_table=goal_table,
    )
    phase = float(np.asarray(task_state["relation_phase"]).reshape(-1)[0])
    return build_se3_relation_token(
        object_pose_a=task_state["object_pose_A"],
        object_pose_b=task_state["object_pose_B"],
        goal_a_from_b=goal,
        phase_gate=phase,
        solver_energy=solver_energy,
        confidence=confidence,
    )
