"""Deployment-only SE(3) token ablations kept outside policy architecture."""

from __future__ import annotations

import numpy as np

from se3_relation_token_utils import (
    OBSERVATION_RELATION_ROUTES,
    RELATION_TOKEN_DIM,
    build_relation_token_from_task_state,
    build_se3_relation_token,
)


SUPPORTED_TOKEN_ABLATIONS = ("none", "zero", "constant_goal")


def build_deploy_relation_token(*, observation: dict, object_pointcloud: dict, model):
    ablation = str(getattr(model, "se3_relation_token_ablation", "none"))
    if ablation not in SUPPORTED_TOKEN_ABLATIONS:
        raise ValueError(
            f"unsupported SE(3) token ablation {ablation!r}; "
            f"expected one of {SUPPORTED_TOKEN_ABLATIONS}"
        )
    if ablation == "zero":
        return np.zeros((RELATION_TOKEN_DIM,), dtype=np.float32)

    task_state = observation.get("task_state")
    if not isinstance(task_state, dict):
        raise RuntimeError(
            "SE(3) relation eval requires task_state pose metadata from "
            "place_shoe_rotating_block.get_obs()."
        )
    route = str(getattr(model, "se3_relation_route", "oracle"))
    if route in OBSERVATION_RELATION_ROUTES:
        pointcloud_a = object_pointcloud.get("{A}")
        pointcloud_b = object_pointcloud.get("{B}")
        if pointcloud_a is None or pointcloud_b is None:
            raise RuntimeError(
                "Estimator-backed SE(3) goal requires separated "
                "object_pointcloud/{A} and object_pointcloud/{B}."
            )
        estimator = getattr(model, "se3_geometry_estimator", None)
        if estimator is None:
            raise RuntimeError("SE(3) geometry estimator is not initialized")
        prediction = estimator.estimate_goal(pointcloud_a, pointcloud_b)
        return build_se3_relation_token(
            object_pose_a=task_state["object_pose_A"],
            object_pose_b=task_state["object_pose_B"],
            goal_a_from_b=prediction.goal_t_a_from_b,
            phase_gate=float(
                np.asarray(task_state["relation_phase"]).reshape(-1)[0]
            ),
            solver_energy=prediction.solver_energy,
            confidence=prediction.confidence,
        ).astype(np.float32)

    if ablation == "constant_goal":
        raise ValueError(
            "constant_goal token ablation requires an estimator-backed policy route"
        )
    return build_relation_token_from_task_state(
        route=route,
        task_state=task_state,
        goal_table=getattr(model, "se3_relation_goal_table", None),
    ).astype(np.float32)


__all__ = ("SUPPORTED_TOKEN_ABLATIONS", "build_deploy_relation_token")
