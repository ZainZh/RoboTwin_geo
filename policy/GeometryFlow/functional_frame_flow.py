"""Camera object-frame calibration for task-flow endpoint construction.

The simulator object poses are training labels only.  At deployment a PCA
frame is fitted to the operated-object cloud, then a training-time calibrated
PCA-to-functional-frame transform and a target-marker goal offset construct
the desired rigid endpoint.  The retrieved demonstration still supplies the
motion schedule seen by the action policy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .build_grasp_dataset import deterministic_farthest_points, valid_xyz
from .canonicalize_task_flow_dataset import (
    matrix_to_pose7,
    pose7_to_matrix,
    sixd_to_rotation,
)
from .evaluate_flow_transport import pca_frame, symmetric_chamfer
from .train_flow_generator import rigid_rotation


GRIPPER_X_FLIP = np.eye(4, dtype=np.float64)
GRIPPER_X_FLIP[:3, :3] = np.diag([1.0, -1.0, -1.0])


def pose9_matrix(value: np.ndarray) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64).reshape(9)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = sixd_to_rotation(pose[3:])
    result[:3, 3] = pose[:3]
    return result


def pca_pose(points: np.ndarray) -> np.ndarray:
    xyz = valid_xyz(points)
    if len(xyz) > 256:
        xyz = deterministic_farthest_points(xyz, 256)
    center, basis, _ = pca_frame(xyz)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = basis
    result[:3, 3] = center
    return result


def mean_transform(transforms: np.ndarray) -> np.ndarray:
    values = np.asarray(transforms, dtype=np.float64)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_matrix(values[:, :3, :3]).mean().as_matrix()
    result[:3, 3] = values[:, :3, 3].mean(axis=0)
    return result


def endpoint_correct_flow(
    initial_anchors: np.ndarray,
    predicted_flow: np.ndarray,
    goal_transform: np.ndarray,
) -> np.ndarray:
    """Smoothly compose a correction that makes the rigid endpoint exact."""
    anchors = np.asarray(initial_anchors, dtype=np.float64)
    flow = np.asarray(predicted_flow, dtype=np.float64)
    rotation = rigid_rotation(anchors, flow[:, -1])
    predicted_transform = np.eye(4, dtype=np.float64)
    predicted_transform[:3, :3] = rotation
    predicted_transform[:3, 3] = (
        flow[:, -1].mean(axis=0) - anchors.mean(axis=0) @ rotation.T
    )
    correction = np.asarray(goal_transform, dtype=np.float64) @ np.linalg.inv(
        predicted_transform
    )
    correction_vector = Rotation.from_matrix(correction[:3, :3]).as_rotvec()
    output = np.empty_like(flow)
    denominator = max(flow.shape[1] - 1, 1)
    for step in range(flow.shape[1]):
        fraction = float(step / denominator)
        step_rotation = Rotation.from_rotvec(correction_vector * fraction).as_matrix()
        output[:, step] = (
            flow[:, step] @ step_rotation.T + correction[:3, 3] * fraction
        )
    output[:, 0] = anchors
    return output.astype(np.float32)


class FunctionalPcaFlowEstimator:
    """Calibrate functional object and goal frames from training episodes."""

    def __init__(
        self,
        library: str | Path,
        oracle_task_dataset: str | Path,
        *,
        train_shoe_ids: tuple[int, ...] | list[int],
    ) -> None:
        with np.load(Path(library), allow_pickle=False) as archive:
            camera = {key: archive[key] for key in archive.files}
        with np.load(Path(oracle_task_dataset), allow_pickle=False) as archive:
            oracle = {key: archive[key] for key in archive.files}
        episode_pose = np.asarray(oracle["episode_pose"], dtype=np.float64)
        if len(episode_pose) != len(camera["episode_id"]):
            raise ValueError("camera library and oracle task dataset disagree")
        train_mask = np.isin(camera["shoe_id"], np.asarray(train_shoe_ids, dtype=np.int64))
        if not np.any(train_mask):
            raise ValueError("functional-frame calibration has no training episodes")
        self.eligible_episodes = np.flatnonzero(train_mask).astype(np.int64)
        self.pca_to_object = np.empty((len(episode_pose), 4, 4), dtype=np.float64)
        self.eef_to_object = np.empty((len(episode_pose), 4, 4), dtype=np.float64)
        self.source_points_object = np.empty(
            (len(episode_pose), camera["points_a_xyzrgb"].shape[1], 3),
            dtype=np.float64,
        )
        goal_offsets = []
        for episode in range(len(episode_pose)):
            object_initial = pose9_matrix(episode_pose[episode, 0])
            self.pca_to_object[episode] = np.linalg.inv(
                pca_pose(camera["points_a_xyzrgb"][episode])
            ) @ object_initial
            source_world = camera["points_a_xyzrgb"][episode, :, :3]
            self.source_points_object[episode] = (
                (source_world - object_initial[:3, 3]) @ object_initial[:3, :3]
            )
            rows = np.flatnonzero(
                (oracle["episode_index"] == episode)
                & (oracle["relation_phase"] >= 0.5)
            )
            active_right = bool(camera["active_arm_right"][episode] >= 0.5)
            gripper_column = 19 if active_right else 9
            rows = rows[oracle["state"][rows, gripper_column] < 0.5]
            if not len(rows):
                raise ValueError(f"episode {episode} has no closed-gripper relation frames")
            eef_index = 1 if active_right else 0
            grasp_relations = []
            for row in rows:
                eef_pose = pose7_to_matrix(oracle["current_eef_pose7"][row, eef_index])
                object_pose = pose9_matrix(oracle["current_object_pose9"][row])
                grasp_relations.append(np.linalg.inv(eef_pose) @ object_pose)
            self.eef_to_object[episode] = mean_transform(np.asarray(grasp_relations))
            if train_mask[episode]:
                goal_offsets.append(
                    np.linalg.inv(camera["target_frame_camera"][episode])
                    @ pose9_matrix(episode_pose[episode, -1])
                )
        self.goal_offset = mean_transform(np.asarray(goal_offsets))
        self.ndf_features = np.asarray(
            camera.get("ndf_features", np.empty((len(episode_pose), 0, 0))),
            dtype=np.float64,
        )[..., :256]

    def desired_object_pose(self, target_frame: np.ndarray) -> np.ndarray:
        """Return the calibrated world-frame object pose at task completion."""
        return np.asarray(target_frame, dtype=np.float64) @ self.goal_offset

    def desired_eef_pose7(
        self,
        target_frame: np.ndarray,
        *,
        source_episode: int,
        clearance_m: float = 0.0,
        gripper_symmetry_variant: int = 0,
    ) -> np.ndarray:
        """Convert a selected object-in-gripper relation into an EEF target.

        ``eef_to_object`` is calibrated only from training demonstrations.  At
        deployment this operation needs the camera-estimated target frame and
        the discrete grasp-relation hypothesis, but no simulator object pose.
        Positive clearance moves away from the target along its outward normal.
        """
        source = int(source_episode)
        variant = int(gripper_symmetry_variant)
        if variant not in (0, 1):
            raise ValueError(f"invalid gripper symmetry variant {variant}")
        symmetry = GRIPPER_X_FLIP if variant else np.eye(4, dtype=np.float64)
        relation = symmetry @ self.eef_to_object[source]
        desired_object = self.desired_object_pose(target_frame)
        desired_eef = desired_object @ np.linalg.inv(relation)
        target = np.asarray(target_frame, dtype=np.float64)
        outward = -target[:3, 2]
        outward /= max(float(np.linalg.norm(outward)), 1e-12)
        desired_eef[:3, 3] += outward * float(clearance_m)
        return matrix_to_pose7(desired_eef).astype(np.float32)

    def predict_flow(
        self,
        operated_point_cloud: np.ndarray,
        target_frame: np.ndarray,
        *,
        source_episode: int,
        initial_anchors: np.ndarray,
        base_flow: np.ndarray,
    ) -> np.ndarray:
        current_object = (
            pca_pose(operated_point_cloud) @ self.pca_to_object[int(source_episode)]
        )
        desired_object = np.asarray(target_frame, dtype=np.float64) @ self.goal_offset
        goal_transform = desired_object @ np.linalg.inv(current_object)
        return endpoint_correct_flow(initial_anchors, base_flow, goal_transform)

    def predict_hypotheses(
        self,
        operated_point_cloud: np.ndarray,
        target_frame: np.ndarray,
        *,
        initial_anchors: np.ndarray,
        base_flow: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return every training functional-frame hypothesis for ceiling tests."""
        current_pca = pca_pose(operated_point_cloud)
        desired_object = np.asarray(target_frame, dtype=np.float64) @ self.goal_offset
        flows = []
        for source_episode in self.eligible_episodes:
            current_object = current_pca @ self.pca_to_object[int(source_episode)]
            goal_transform = desired_object @ np.linalg.inv(current_object)
            flows.append(endpoint_correct_flow(initial_anchors, base_flow, goal_transform))
        return self.eligible_episodes.copy(), np.stack(flows)

    def predict_grasp_relation_flow(
        self,
        active_eef_pose7: np.ndarray,
        target_frame: np.ndarray,
        *,
        source_episode: int,
        initial_anchors: np.ndarray,
        base_flow: np.ndarray,
    ) -> np.ndarray:
        """Transfer a retrieved object-in-gripper relation at deployment."""
        current_object = (
            pose7_to_matrix(active_eef_pose7)
            @ self.eef_to_object[int(source_episode)]
        )
        desired_object = np.asarray(target_frame, dtype=np.float64) @ self.goal_offset
        goal_transform = desired_object @ np.linalg.inv(current_object)
        return endpoint_correct_flow(initial_anchors, base_flow, goal_transform)

    def predict_grasp_relation_hypotheses(
        self,
        active_eef_pose7: np.ndarray,
        target_frame: np.ndarray,
        *,
        initial_anchors: np.ndarray,
        base_flow: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        eef_pose = pose7_to_matrix(active_eef_pose7)
        desired_object = np.asarray(target_frame, dtype=np.float64) @ self.goal_offset
        flows = []
        for source_episode in self.eligible_episodes:
            current_object = eef_pose @ self.eef_to_object[int(source_episode)]
            goal_transform = desired_object @ np.linalg.inv(current_object)
            flows.append(endpoint_correct_flow(initial_anchors, base_flow, goal_transform))
        return self.eligible_episodes.copy(), np.stack(flows)

    def predict_geometry_selected_grasp_flow(
        self,
        operated_point_cloud: np.ndarray,
        active_eef_pose7: np.ndarray,
        target_frame: np.ndarray,
        *,
        initial_anchors: np.ndarray,
        base_flow: np.ndarray,
    ) -> tuple[np.ndarray, int, float, float]:
        """Select the grasp prototype whose transformed shape fits the camera cloud."""
        sources, candidates, costs = self.geometry_grasp_candidates(
            operated_point_cloud, active_eef_pose7
        )
        order = np.argsort(costs)
        row = int(order[0])
        source_episode = int(sources[row])
        desired_object = np.asarray(target_frame, dtype=np.float64) @ self.goal_offset
        goal_transform = desired_object @ np.linalg.inv(candidates[row])
        flow = endpoint_correct_flow(initial_anchors, base_flow, goal_transform)
        best = float(costs[row])
        second = float(costs[int(order[1])]) if len(order) > 1 else best
        return flow, source_episode, best, second - best

    def predict_symmetry_selected_grasp_flow(
        self,
        operated_point_cloud: np.ndarray,
        active_eef_pose7: np.ndarray,
        target_frame: np.ndarray,
        *,
        initial_anchors: np.ndarray,
        base_flow: np.ndarray,
        min_cost_improvement_m2: float = 0.0,
    ) -> tuple[np.ndarray, int, float, float, int]:
        """Select camera-fit grasp relation after expanding gripper symmetry.

        A parallel-jaw grasp is unchanged by a 180-degree rotation around the
        gripper's local x axis.  Demonstrations can therefore encode the same
        physical grasp in either wrist-roll mode.  Expanding both modes before
        camera scoring keeps the shoe pose exact instead of weakening the task
        rotation metric.
        """
        sources, candidates, costs, variants = (
            self.geometry_grasp_candidates_with_symmetry(
                operated_point_cloud, active_eef_pose7
            )
        )
        order = np.argsort(costs)
        row = int(order[0])
        baseline_rows = np.flatnonzero(variants == 0)
        baseline_row = int(baseline_rows[np.argmin(costs[baseline_rows])])
        improvement = float(costs[baseline_row] - costs[row])
        if (
            int(variants[row]) == 1
            and improvement < float(min_cost_improvement_m2)
        ):
            row = baseline_row
            order = baseline_rows[np.argsort(costs[baseline_rows])]
        source_episode = int(sources[row])
        desired_object = np.asarray(target_frame, dtype=np.float64) @ self.goal_offset
        goal_transform = desired_object @ np.linalg.inv(candidates[row])
        flow = endpoint_correct_flow(initial_anchors, base_flow, goal_transform)
        best = float(costs[row])
        second = float(costs[int(order[1])]) if len(order) > 1 else best
        return flow, source_episode, best, second - best, int(variants[row])

    def geometry_grasp_candidates(
        self,
        operated_point_cloud: np.ndarray,
        active_eef_pose7: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return source ids, implied object poses, and camera-fit costs."""
        query = valid_xyz(operated_point_cloud)
        if len(query) > 256:
            query = deterministic_farthest_points(query, 256)
        eef_pose = pose7_to_matrix(active_eef_pose7)
        candidates = []
        costs = []
        for source_episode in self.eligible_episodes:
            current_object = eef_pose @ self.eef_to_object[int(source_episode)]
            source_world = (
                self.source_points_object[int(source_episode)]
                @ current_object[:3, :3].T
                + current_object[:3, 3]
            )
            candidates.append(current_object)
            costs.append(symmetric_chamfer(query, source_world))
        return (
            self.eligible_episodes.copy(),
            np.asarray(candidates, dtype=np.float64),
            np.asarray(costs, dtype=np.float64),
        )

    def geometry_grasp_candidates_with_symmetry(
        self,
        operated_point_cloud: np.ndarray,
        active_eef_pose7: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return both parallel-gripper wrist-roll modes for every relation."""
        query = valid_xyz(operated_point_cloud)
        if len(query) > 256:
            query = deterministic_farthest_points(query, 256)
        eef_pose = pose7_to_matrix(active_eef_pose7)
        sources = []
        candidates = []
        costs = []
        variants = []
        for source_episode in self.eligible_episodes:
            source = int(source_episode)
            for variant, symmetry in enumerate(
                (np.eye(4, dtype=np.float64), GRIPPER_X_FLIP)
            ):
                current_object = eef_pose @ symmetry @ self.eef_to_object[source]
                source_world = (
                    self.source_points_object[source] @ current_object[:3, :3].T
                    + current_object[:3, 3]
                )
                sources.append(source)
                candidates.append(current_object)
                costs.append(symmetric_chamfer(query, source_world))
                variants.append(variant)
        return (
            np.asarray(sources, dtype=np.int64),
            np.asarray(candidates, dtype=np.float64),
            np.asarray(costs, dtype=np.float64),
            np.asarray(variants, dtype=np.int64),
        )

    def predict_gripper_symmetry_grasp_hypotheses(
        self,
        operated_point_cloud: np.ndarray,
        active_eef_pose7: np.ndarray,
        target_frame: np.ndarray,
        *,
        initial_anchors: np.ndarray,
        base_flow: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return all symmetry-expanded relation flows for an oracle ceiling."""
        sources, candidates, _, variants = (
            self.geometry_grasp_candidates_with_symmetry(
                operated_point_cloud, active_eef_pose7
            )
        )
        desired_object = np.asarray(target_frame, dtype=np.float64) @ self.goal_offset
        flows = [
            endpoint_correct_flow(
                initial_anchors,
                base_flow,
                desired_object @ np.linalg.inv(candidate),
            )
            for candidate in candidates
        ]
        return sources, variants, np.stack(flows)

    def predict_consensus_selected_grasp_flow(
        self,
        operated_point_cloud: np.ndarray,
        active_eef_pose7: np.ndarray,
        target_frame: np.ndarray,
        *,
        initial_anchors: np.ndarray,
        base_flow: np.ndarray,
        top_k: int = 8,
        translation_radius_m: float = 0.04,
        rotation_radius_deg: float = 35.0,
    ) -> tuple[np.ndarray, int, float, float, dict[str, float | int]]:
        """Select a low-cost relation supported by an SE(3) hypothesis cluster.

        Symmetric partial views can give one wrong orientation a marginally
        lower Chamfer cost.  This selector ranks the lowest-cost hypotheses by
        how many other low-cost object poses agree in both translation and
        rotation, then chooses the cheapest member of the strongest cluster.
        """
        sources, candidates, costs = self.geometry_grasp_candidates(
            operated_point_cloud, active_eef_pose7
        )
        count = min(max(int(top_k), 1), len(costs))
        order = np.argsort(costs)
        active = order[:count]
        transforms = candidates[active]
        translation = np.linalg.norm(
            transforms[:, None, :3, 3] - transforms[None, :, :3, 3], axis=-1
        )
        relative_rotation = (
            transforms[:, None, :3, :3]
            @ np.swapaxes(transforms[None, :, :3, :3], -1, -2)
        )
        rotation_deg = np.degrees(
            Rotation.from_matrix(relative_rotation.reshape(-1, 3, 3))
            .magnitude()
            .reshape(count, count)
        )
        neighbors = (translation <= float(translation_radius_m)) & (
            rotation_deg <= float(rotation_radius_deg)
        )
        support = neighbors.sum(axis=1)
        # Lexicographic order: largest support, then lowest original Chamfer.
        local = min(range(count), key=lambda row: (-int(support[row]), costs[active[row]]))
        row = int(active[local])
        source_episode = int(sources[row])
        desired_object = np.asarray(target_frame, dtype=np.float64) @ self.goal_offset
        goal_transform = desired_object @ np.linalg.inv(candidates[row])
        flow = endpoint_correct_flow(initial_anchors, base_flow, goal_transform)
        nearest = int(order[0])
        second = float(costs[int(order[1])]) if len(order) > 1 else float(costs[nearest])
        diagnostics: dict[str, float | int] = {
            "cluster_support": int(support[local]),
            "top_k": int(count),
            "nearest_source_episode": int(sources[nearest]),
            "nearest_cost_m2": float(costs[nearest]),
            "selected_cost_m2": float(costs[row]),
            "selected_minus_nearest_cost_m2": float(costs[row] - costs[nearest]),
            "top_k_translation_dispersion_m": float(
                np.median(translation[np.triu_indices(count, 1)]) if count > 1 else 0.0
            ),
            "top_k_rotation_dispersion_deg": float(
                np.median(rotation_deg[np.triu_indices(count, 1)]) if count > 1 else 0.0
            ),
        }
        return flow, source_episode, float(costs[row]), second - float(costs[nearest]), diagnostics

    def predict_descriptor_selected_grasp_flow(
        self,
        operated_point_cloud: np.ndarray,
        query_features: np.ndarray,
        active_eef_pose7: np.ndarray,
        target_frame: np.ndarray,
        *,
        initial_anchors: np.ndarray,
        base_flow: np.ndarray,
    ) -> tuple[np.ndarray, int, float, float]:
        """Use NDF only to score discrete grasp-pose hypotheses."""
        query = valid_xyz(operated_point_cloud)
        if len(query) > 256:
            query = deterministic_farthest_points(query, 256)
        query_descriptor = np.asarray(query_features, dtype=np.float64)[..., :256]
        if query_descriptor.shape[0] != query.shape[0]:
            raise ValueError("query NDF features must align with sampled operated points")
        query_descriptor /= np.maximum(
            np.linalg.norm(query_descriptor, axis=-1, keepdims=True), 1e-8
        )
        eef_pose = pose7_to_matrix(active_eef_pose7)
        candidates = []
        costs = []
        for source_episode in self.eligible_episodes:
            source = int(source_episode)
            current_object = eef_pose @ self.eef_to_object[source]
            source_world = (
                self.source_points_object[source] @ current_object[:3, :3].T
                + current_object[:3, 3]
            )
            source_descriptor = self.ndf_features[source]
            source_descriptor = source_descriptor / np.maximum(
                np.linalg.norm(source_descriptor, axis=-1, keepdims=True), 1e-8
            )
            similarity = source_descriptor @ query_descriptor.T
            source_to_query = np.argmax(similarity, axis=1)
            query_to_source = np.argmax(similarity, axis=0)
            source_ids = np.flatnonzero(
                query_to_source[source_to_query] == np.arange(len(source_descriptor))
            )
            if len(source_ids) < 8:
                source_ids = np.argsort(similarity.max(axis=1))[-32:]
            query_ids = source_to_query[source_ids]
            residual = np.linalg.norm(
                source_world[source_ids] - query[query_ids], axis=-1
            )
            descriptor_weight = np.clip(
                similarity[source_ids, query_ids], 0.0, 1.0
            )
            cost = float(
                np.average(residual, weights=descriptor_weight + 1e-3)
                + 0.01 * np.mean(1.0 - similarity[source_ids, query_ids])
            )
            candidates.append(current_object)
            costs.append(cost)
        order = np.argsort(np.asarray(costs))
        row = int(order[0])
        source_episode = int(self.eligible_episodes[row])
        desired_object = np.asarray(target_frame, dtype=np.float64) @ self.goal_offset
        goal_transform = desired_object @ np.linalg.inv(candidates[row])
        flow = endpoint_correct_flow(initial_anchors, base_flow, goal_transform)
        best = float(costs[row])
        second = float(costs[int(order[1])]) if len(order) > 1 else best
        return flow, source_episode, best, second - best
