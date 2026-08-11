"""Camera-only full-task deployment for the relation-preserving TAGRT v2 policy."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from policy.DP3.scripts.eef_delta_action_utils import decode_eef_delta_action16

from .build_task_flow_dataset import deterministic_sample
from .deploy_policy import _endpose_state, _object_clouds
from .fulltask_tagrt_v2 import FullTaskTAGRTPolicy, merge_action14, merge_action26
from .online_ndf_functional_frame import OnlineNdfFunctionalFrameProvider
from .prepare_fulltask_v2_archive import fixed_sample, local_relation_tokens
from .train_fulltask_tagrt_v2 import Statistics


def _path_list(value) -> list[str]:
    if value in {None, ""}:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item) for item in value]


def _goal_frame9(transform: np.ndarray) -> np.ndarray:
    value = np.asarray(transform, dtype=np.float32).reshape(4, 4)
    return np.concatenate((value[:3, 3], value[:3, 0], value[:3, 1])).astype(np.float32)


class FullTaskTAGRTV2Runtime:
    def __init__(
        self,
        checkpoint: str | Path,
        *,
        frame_provider: OnlineNdfFunctionalFrameProvider | None,
        device: str,
        flow_steps: int = 8,
        inference_seed: int = 0,
    ) -> None:
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        payload = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        self.condition = str(payload["condition"])
        self.action_representation = str(
            payload.get("action_representation", "eef_delta14")
        )
        self.statistics = Statistics(**payload["statistics"])
        self.device = torch.device(device)
        self.policy = FullTaskTAGRTPolicy(**payload["model_config"]).to(self.device)
        self.policy.load_state_dict(payload["model"])
        self.policy.eval()
        self.frame_provider = frame_provider
        if self.condition.startswith("tagrt") and frame_provider is None:
            raise ValueError("TAGRT checkpoint requires a camera NDF frame provider")
        self.history: deque[dict[str, np.ndarray]] = deque(
            maxlen=int(payload["model_config"]["history"])
        )
        self.flow_steps = int(flow_steps)
        self.inference_seed = int(inference_seed)
        self.observation_index = 0
        self.cached_target: np.ndarray | None = None
        self.cached_operated: np.ndarray | None = None
        self.last_diagnostic: dict = {}
        self.last_gripper_open_probability = np.ones(
            (self.policy.horizon, 2), dtype=np.float32
        )
        self.action_chunks = 0

    def reset(self) -> None:
        self.history.clear()
        self.observation_index = 0
        self.cached_target = None
        self.cached_operated = None
        self.last_diagnostic = {}
        self.last_gripper_open_probability.fill(1.0)
        self.action_chunks = 0
        if self.frame_provider is not None:
            self.frame_provider.reset()

    def _normalized_frame(self, observation: dict) -> dict[str, np.ndarray]:
        operated, target = _object_clouds(observation)
        if len(operated):
            self.cached_operated = operated.copy()
        elif self.cached_operated is None:
            raise ValueError("operated object A is not visible at task start")
        if self.cached_target is None:
            if not len(target):
                raise ValueError("target object B is not visible at task start")
            self.cached_target = target.copy()
        operated = np.asarray(self.cached_operated, dtype=np.float32)
        target = np.asarray(self.cached_target, dtype=np.float32)
        base_a = deterministic_sample(
            operated,
            128,
            seed=self.observation_index * 17 + 1,
            point_channels=6,
        )
        base_b = deterministic_sample(target, 128, seed=2, point_channels=6)
        scene = fixed_sample(np.concatenate((base_a, base_b), axis=0), 48)
        points_a = fixed_sample(base_a, 32)
        points_b = fixed_sample(base_b, 32, offset=1)
        state = _endpose_state(observation).astype(np.float32)
        confidence = 0.0
        source_disagreement = None
        relative = np.zeros(9, dtype=np.float32)
        if self.condition.startswith("tagrt"):
            assert self.frame_provider is not None
            estimate = self.frame_provider.estimate(
                current_point_cloud=operated,
                target_point_cloud=target,
                mode="clean",
            )
            if estimate.goal_transform is None:
                raise RuntimeError("clean camera estimate did not return a goal transform")
            relative = estimate.frame9_metric.astype(np.float32)
            goal = _goal_frame9(estimate.goal_transform)
            local = local_relation_tokens(base_a, state, relative, goal, count=16)
            confidence = float(estimate.combined_confidence)
            source_disagreement = estimate.source_disagreement_deg
            global_relation = np.concatenate((relative, [confidence])).astype(np.float32)
        else:
            local = np.zeros((16, 12), dtype=np.float32)
            global_relation = np.zeros(10, dtype=np.float32)
        stats = self.statistics
        point_mean = np.asarray(stats.point_mean, dtype=np.float32)
        point_std = np.asarray(stats.point_std, dtype=np.float32)
        frame = {
            "scene": (scene - point_mean) / point_std,
            "points_a": (points_a - point_mean) / point_std,
            "points_b": (points_b - point_mean) / point_std,
            "local": (
                local - np.asarray(stats.local_mean, dtype=np.float32)
            ) / np.asarray(stats.local_std, dtype=np.float32),
            "global": (
                global_relation - np.asarray(stats.global_mean, dtype=np.float32)
            ) / np.asarray(stats.global_std, dtype=np.float32),
            "state": (
                state - np.asarray(stats.state_mean, dtype=np.float32)
            ) / np.asarray(stats.state_std, dtype=np.float32),
        }
        # Raw policies were trained with exact zero geometry after normalization.
        if self.condition.startswith("raw"):
            frame["local"].fill(0.0)
            frame["global"].fill(0.0)
        self.last_diagnostic = {
            "condition": self.condition,
            "confidence": confidence,
            "source_disagreement_deg": source_disagreement,
            "remaining_translation_m": relative[:3].astype(float).tolist(),
            "left_eef_to_object_centroid_m": float(
                np.linalg.norm(state[:3] - operated[:, :3].mean(axis=0))
            ),
            "right_eef_to_object_centroid_m": float(
                np.linalg.norm(state[10:13] - operated[:, :3].mean(axis=0))
            ),
        }
        self.observation_index += 1
        return frame

    @torch.no_grad()
    def predict(self, observation: dict) -> np.ndarray:
        frame = self._normalized_frame(observation)
        if not self.history:
            for _ in range(self.history.maxlen):
                self.history.append({key: value.copy() for key, value in frame.items()})
        else:
            self.history.append(frame)
        batch = {
            key: torch.from_numpy(
                np.stack([item[key] for item in self.history], axis=0)[None]
            ).to(self.device)
            for key in frame
        }
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.inference_seed + self.action_chunks * 1009)
        initial = torch.randn(
            1,
            self.policy.horizon,
            self.policy.motion_dim,
            generator=generator,
            device=self.device,
        )
        output = self.policy.sample(
            batch,
            integration_steps=self.flow_steps,
            initial_noise=(initial if self.condition.endswith("flow") else None),
        )
        motion = output.motion[0].float().cpu()
        motion = (
            motion * torch.tensor(self.statistics.motion_std)
            + torch.tensor(self.statistics.motion_mean)
        )
        gripper_probability = output.gripper_logits[0].sigmoid().float().cpu()
        self.last_gripper_open_probability = gripper_probability.numpy().astype(
            np.float32
        )
        gripper = (gripper_probability >= 0.5).float()
        self.action_chunks += 1
        merged = (
            merge_action26(motion, gripper)
            if self.action_representation == "dense_joint26"
            else merge_action14(motion, gripper)
        )
        return merged.numpy().astype(np.float32)


def get_model(usr_args):
    device = str(usr_args.get("fulltask_tagrt_device", "cuda:0"))
    checkpoint = Path(usr_args["fulltask_tagrt_checkpoint"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    condition = str(payload["condition"])
    provider = None
    if condition.startswith("tagrt"):
        provider = OnlineNdfFunctionalFrameProvider.from_metadata(
            ndf_checkpoints=_path_list(usr_args["fulltask_tagrt_frame_checkpoints"]),
            camera_metadata=usr_args["fulltask_tagrt_camera_metadata"],
            ensemble_metadata=usr_args["fulltask_tagrt_ensemble_metadata"],
            device=device,
        )
    runtime = FullTaskTAGRTV2Runtime(
        checkpoint,
        frame_provider=provider,
        device=device,
        flow_steps=int(usr_args.get("fulltask_tagrt_flow_steps", 8)),
        inference_seed=int(usr_args.get("fulltask_tagrt_inference_seed", 0)),
    )
    model = SimpleNamespace(
        runtime=runtime,
        execute_steps=max(1, int(usr_args.get("fulltask_tagrt_execute_steps", 2))),
        max_policy_steps=max(1, int(usr_args.get("fulltask_tagrt_max_policy_steps", 220))),
        max_translation_delta_m=float(usr_args.get("max_eef_translation_delta_m", 0.08)),
        max_rotation_delta_rad=float(usr_args.get("max_eef_rotation_delta_rad", 0.5)),
        executed_actions=0,
        action_chunks=0,
        translation_sum=0.0,
        rotation_sum=0.0,
        gripper_closed_count=np.zeros(2, dtype=np.int64),
        min_gripper_open_probability=np.ones(2, dtype=np.float32),
        first_gripper_close_step=[None, None],
        last_gripper_command=np.ones(2, dtype=np.float32),
    )

    def metrics():
        return {
            "fulltask_tagrt_condition": runtime.condition,
            "fulltask_tagrt_action_representation": runtime.action_representation,
            "fulltask_tagrt_action_chunks": int(model.action_chunks),
            "fulltask_tagrt_executed_actions": int(model.executed_actions),
            "fulltask_tagrt_mean_translation_delta_m": float(
                model.translation_sum / max(model.executed_actions, 1)
            ),
            "fulltask_tagrt_mean_rotation_delta_deg": float(
                np.rad2deg(model.rotation_sum / max(model.executed_actions, 1))
            ),
            "fulltask_tagrt_gripper_closed_fraction": (
                model.gripper_closed_count / max(model.executed_actions, 1)
            ).astype(float).tolist(),
            "fulltask_tagrt_first_gripper_close_step": list(
                model.first_gripper_close_step
            ),
            "fulltask_tagrt_last_gripper_command": model.last_gripper_command.astype(
                float
            ).tolist(),
            "fulltask_tagrt_min_gripper_open_probability": (
                model.min_gripper_open_probability.astype(float).tolist()
            ),
            **{
                f"fulltask_tagrt_{key}": value
                for key, value in runtime.last_diagnostic.items()
            },
        }

    model.get_evaluation_metrics = metrics
    return model


def eval(TASK_ENV, model, observation):
    actions = model.runtime.predict(observation)
    model.action_chunks += 1
    if model.runtime.action_representation == "dense_joint26":
        chunk = np.asarray(actions[: model.execute_steps], dtype=np.float32)
        probability = model.runtime.last_gripper_open_probability[: len(chunk)]
        model.min_gripper_open_probability = np.minimum(
            model.min_gripper_open_probability,
            probability.min(axis=0),
        )
        gripper = np.stack((chunk[:, 12], chunk[:, 25]), axis=-1)
        model.last_gripper_command = gripper[-1].copy()
        closed = gripper < 0.5
        model.gripper_closed_count += closed.sum(axis=0).astype(np.int64)
        for arm in range(2):
            hits = np.flatnonzero(closed[:, arm])
            if len(hits) and model.first_gripper_close_step[arm] is None:
                model.first_gripper_close_step[arm] = int(
                    model.executed_actions + hits[0]
                )
        position = np.concatenate(
            (chunk[:, :6], chunk[:, 12:13], chunk[:, 13:19], chunk[:, 25:26]),
            axis=-1,
        )
        arm_velocity = np.concatenate(
            (chunk[:, 6:12], chunk[:, 19:25]), axis=-1
        )
        TASK_ENV.take_low_level_joint_action_chunk(position, arm_velocity)
        model.executed_actions += len(chunk)
        if model.executed_actions >= model.max_policy_steps:
            TASK_ENV.step_lim = TASK_ENV.take_action_cnt
        return
    endpose = {
        "left_endpose": np.asarray(observation["endpose"]["left_endpose"], dtype=np.float32),
        "right_endpose": np.asarray(observation["endpose"]["right_endpose"], dtype=np.float32),
    }
    for chunk_step, action in enumerate(actions[: model.execute_steps]):
        action = np.asarray(action, dtype=np.float32).copy()
        model.min_gripper_open_probability = np.minimum(
            model.min_gripper_open_probability,
            model.runtime.last_gripper_open_probability[chunk_step],
        )
        gripper = np.asarray((action[6], action[13]), dtype=np.float32)
        model.last_gripper_command = gripper.copy()
        closed = gripper < 0.5
        model.gripper_closed_count += closed.astype(np.int64)
        for arm in range(2):
            if closed[arm] and model.first_gripper_close_step[arm] is None:
                model.first_gripper_close_step[arm] = int(model.executed_actions)
        translation = max(np.linalg.norm(action[:3]), np.linalg.norm(action[7:10]))
        rotation = max(np.linalg.norm(action[3:6]), np.linalg.norm(action[10:13]))
        model.translation_sum += float(translation)
        model.rotation_sum += float(rotation)
        decoded = decode_eef_delta_action16(
            action,
            endpose["left_endpose"],
            endpose["right_endpose"],
            max_translation_delta_m=model.max_translation_delta_m,
            max_rotation_delta_rad=model.max_rotation_delta_rad,
        )
        TASK_ENV.take_action(decoded, action_type="ee")
        model.executed_actions += 1
        endpose = {
            "left_endpose": decoded[:7].copy(),
            "right_endpose": decoded[8:15].copy(),
        }
        if model.executed_actions >= model.max_policy_steps:
            TASK_ENV.step_lim = TASK_ENV.take_action_cnt
            break
        if TASK_ENV.eval_success or TASK_ENV.take_action_cnt >= TASK_ENV.step_lim:
            break


def reset_model(model):
    model.runtime.reset()
    model.executed_actions = 0
    model.action_chunks = 0
    model.translation_sum = 0.0
    model.rotation_sum = 0.0
    model.gripper_closed_count[:] = 0
    model.min_gripper_open_probability[:] = 1.0
    model.first_gripper_close_step = [None, None]
    model.last_gripper_command[:] = 1.0
