"""Privileged expert-action replay for validating the online action contract.

This diagnostic is not a policy baseline: it reads recorded expert actions and
therefore cannot be deployed.  It exists only to prove that the EEF-delta
decoder and simulator execution path can reproduce a successful demonstration.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from policy.DP3.scripts.eef_delta_action_utils import decode_eef_delta_action16


def get_model(usr_args):
    # A directory-backed dense replay is an ordered episode bank and does not
    # select a single episode here.  Keep zero as the harmless default for
    # that batch mode; single-episode NPZ replay can still override it.
    episode = int(usr_args.get("expert_replay_episode", 0))
    dense_data_dir = usr_args.get("expert_replay_dense_control_data_dir")
    dense_hdf5_path = usr_args.get("expert_replay_dense_control_hdf5")
    hdf5_path = usr_args.get("expert_replay_hdf5")
    use_dense_control = dense_hdf5_path is not None or dense_data_dir is not None
    use_qpos = hdf5_path is not None
    if use_dense_control and use_qpos:
        raise ValueError("dense-control and sparse-qpos replay are mutually exclusive")
    future_pose = None
    rows = None
    dense_arm_velocity = None
    replay_bank = None
    if use_dense_control:
        paths = (
            [Path(dense_hdf5_path)]
            if dense_data_dir is None
            else [
                Path(dense_data_dir) / f"episode{index}.hdf5"
                for index in range(
                    int(usr_args.get("evaluation_start_episode_index", 0)),
                    int(usr_args.get("evaluation_start_episode_index", 0))
                    + int(usr_args.get("test_num", 1)),
                )
            ]
        )
        replay_bank = []
        for path in paths:
            with h5py.File(path, "r") as archive:
                bank_position = np.asarray(
                    archive["dense_control/position"], dtype=np.float32
                )
                bank_velocity = np.asarray(
                    archive["dense_control/arm_velocity"], dtype=np.float32
                )
            if bank_velocity.shape != (bank_position.shape[0], 12):
                raise ValueError(
                    "dense-control velocity/position length mismatch: "
                    f"{bank_velocity.shape} vs {bank_position.shape} in {path}"
                )
            replay_bank.append((bank_position, bank_velocity))
        transitions, dense_arm_velocity = replay_bank[0]
    elif use_qpos:
        frame_stride = max(1, int(usr_args.get("expert_replay_frame_stride", 1)))
        with h5py.File(Path(hdf5_path), "r") as archive:
            qpos = np.asarray(archive["joint_action/vector"], dtype=np.float32)
        # Frame i records the arm drive target active when observation i was
        # captured.  The first policy transition therefore targets frame 1.
        transitions = qpos[1::frame_stride]
    else:
        dataset_path = Path(usr_args["expert_replay_dataset"])
        with np.load(dataset_path, allow_pickle=False) as archive:
            episode_id = np.asarray(archive["episode_id"], dtype=np.int64)
            frame_index = np.asarray(archive["frame_index"], dtype=np.int64)
            action = np.asarray(archive["action"], dtype=np.float32)
            future_pose = (
                np.asarray(archive["future_eef_pose7"], dtype=np.float32)
                if "future_eef_pose7" in archive
                else None
            )
        rows = np.flatnonzero(episode_id == episode)
        rows = rows[np.argsort(frame_index[rows])]
        # The first element of every chunk is the unique next transition.
        # Taking complete overlapping chunks would replay actions repeatedly.
        transitions = action[rows, 0]

    use_absolute_pose = bool(usr_args.get("expert_replay_absolute_pose", False))
    if (use_qpos or use_dense_control) and use_absolute_pose:
        raise ValueError("qpos and absolute EEF replay modes are mutually exclusive")
    if use_absolute_pose and future_pose is None:
        raise KeyError("absolute expert replay requires future_eef_pose7")
    absolute_targets = None
    if use_absolute_pose:
        pose = future_pose[rows, 0]
        absolute_targets = np.concatenate(
            (
                pose[:, 0],
                transitions[:, 6:7],
                pose[:, 1],
                transitions[:, 13:14],
            ),
            axis=-1,
        ).astype(np.float32)
    return SimpleNamespace(
        transitions=transitions,
        dense_arm_velocity=dense_arm_velocity,
        replay_bank=replay_bank,
        replay_bank_index=0,
        absolute_targets=absolute_targets,
        replay_mode=(
            "dense_joint_control"
            if use_dense_control
            else (
                "qpos_drive_target"
                if use_qpos
                else ("absolute_pose" if use_absolute_pose else "eef_delta")
            )
        ),
        cursor=0,
        execute_steps=max(1, int(usr_args.get("expert_replay_execute_steps", 2))),
        terminal_hold_steps=max(
            0, int(usr_args.get("expert_replay_terminal_hold_steps", 0))
        ),
        terminal_hold_done=False,
        max_translation_delta_m=float(usr_args.get("max_eef_translation_delta_m", 0.08)),
        max_rotation_delta_rad=float(usr_args.get("max_eef_rotation_delta_rad", 0.5)),
        closed_count=np.zeros(2, dtype=np.int64),
        executed=0,
        get_evaluation_metrics=lambda: {},
    )


def eval(TASK_ENV, model, observation):
    endpose = {
        "left_endpose": np.asarray(observation["endpose"]["left_endpose"]),
        "right_endpose": np.asarray(observation["endpose"]["right_endpose"]),
    }
    stop = min(model.cursor + model.execute_steps, len(model.transitions))
    if model.replay_mode == "dense_joint_control":
        chunk = model.transitions[model.cursor:stop]
        velocity = model.dense_arm_velocity[model.cursor:stop]
        model.closed_count += np.asarray(
            ((chunk[:, 6] < 0.5).sum(), (chunk[:, 13] < 0.5).sum()),
            dtype=np.int64,
        )
        TASK_ENV.take_low_level_joint_action_chunk(chunk, velocity)
        model.executed += len(chunk)
        model.cursor = stop
        # Collection tasks advance the simulator for a short stability check
        # after the final recorded motion primitive.  Reproduce that exact
        # evaluator-only post-roll without rewriting or inventing an action.
        if (
            model.cursor >= len(model.transitions)
            and model.terminal_hold_steps > 0
            and not model.terminal_hold_done
            and not TASK_ENV.eval_success
        ):
            TASK_ENV.take_low_level_settle_steps(model.terminal_hold_steps)
            model.terminal_hold_done = True
    else:
        for replay_index, action in enumerate(
            model.transitions[model.cursor : stop], start=model.cursor
        ):
            model.closed_count += np.asarray((action[6] < 0.5, action[13] < 0.5))
            if model.replay_mode == "qpos_drive_target":
                decoded = action
                TASK_ENV.take_action(decoded, action_type="qpos")
            else:
                decoded = (
                    model.absolute_targets[replay_index]
                    if model.absolute_targets is not None
                    else decode_eef_delta_action16(
                        action,
                        endpose["left_endpose"],
                        endpose["right_endpose"],
                        max_translation_delta_m=model.max_translation_delta_m,
                        max_rotation_delta_rad=model.max_rotation_delta_rad,
                    )
                )
                TASK_ENV.take_action(decoded, action_type="ee")
            model.executed += 1
            if model.replay_mode != "qpos_drive_target":
                endpose = {
                    "left_endpose": decoded[:7].copy(),
                    "right_endpose": decoded[8:15].copy(),
                }
            if TASK_ENV.eval_success or TASK_ENV.take_action_cnt >= TASK_ENV.step_lim:
                break
        model.cursor = stop
    if model.cursor >= len(model.transitions):
        TASK_ENV.step_lim = TASK_ENV.take_action_cnt

    def metrics():
        return {
            "expert_replay_executed_actions": int(model.executed),
            "expert_replay_total_actions": int(len(model.transitions)),
            "expert_replay_mode": model.replay_mode,
            "expert_replay_terminal_hold_steps": int(
                model.terminal_hold_steps if model.terminal_hold_done else 0
            ),
            "expert_replay_closed_fraction": (
                model.closed_count / max(model.executed, 1)
            ).astype(float).tolist(),
        }

    model.get_evaluation_metrics = metrics


def reset_model(model):
    if model.replay_bank is not None:
        if model.replay_bank_index >= len(model.replay_bank):
            raise IndexError("dense replay reset exceeds the loaded episode bank")
        model.transitions, model.dense_arm_velocity = model.replay_bank[
            model.replay_bank_index
        ]
        model.replay_bank_index += 1
    model.cursor = 0
    model.executed = 0
    model.terminal_hold_done = False
    model.closed_count[:] = 0
