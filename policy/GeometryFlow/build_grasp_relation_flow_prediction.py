#!/usr/bin/env python3
"""Build cross-fitted camera grasp-relation flows for policy training.

Each episode is reconstructed with the same deployable path used by
``deploy_policy``: segmented A/B clouds are passed through online PCA flow
transport, then the camera-fit object-in-gripper relation corrects the rigid
endpoint.  Training episodes exclude every demonstration of the query shoe
from both retrieval banks, which prevents same-object flow leakage.  Held-out
validation/test shoes naturally use the complete training-shoe bank.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from .functional_frame_flow import FunctionalPcaFlowEstimator
from .online_flow import OnlinePcaFlowTransporter
from .target_frame import estimate_geometry_marker_frame
from .train_task_flow_benchmark import FOLDS
from .train_flow_generator import rigid_rotation


def transfer_rigid_flow_to_anchors(
    source_anchors: np.ndarray,
    source_flow: np.ndarray,
    destination_anchors: np.ndarray,
) -> np.ndarray:
    """Apply a rigid flow schedule to another, identity-consistent anchor set.

    Online transport samples anchors from the query camera cloud, whereas the
    task dataset tracks a fixed set of object-local anchors through the whole
    episode.  The two sets can have the same shape without sharing point
    identities.  Passing one flow with the other current-anchor array silently
    corrupts every per-point displacement and SE(3) token.  Transfer the rigid
    transform at each step instead of pairing unrelated point indices.
    """
    source = np.asarray(source_anchors, dtype=np.float64)
    flow = np.asarray(source_flow, dtype=np.float64)
    destination = np.asarray(destination_anchors, dtype=np.float64)
    if flow.ndim != 3 or flow.shape[0] != len(source) or flow.shape[2] != 3:
        raise ValueError(
            f"source flow must be [anchors,steps,3], got {flow.shape}"
        )
    if source.shape != destination.shape or source.shape[1:] != (3,):
        raise ValueError(
            f"anchor sets must share [anchors,3], got {source.shape} and "
            f"{destination.shape}"
        )
    transferred = np.empty_like(flow)
    for step in range(flow.shape[1]):
        rotation = rigid_rotation(source, flow[:, step])
        translation = flow[:, step].mean(axis=0) - source.mean(axis=0) @ rotation.T
        transferred[:, step] = destination @ rotation.T + translation
    transferred[:, 0] = destination
    return transferred.astype(np.float32)


def episode_query_index(payload: dict[str, np.ndarray], episode: int) -> int:
    rows = np.flatnonzero(
        (payload["episode_index"] == int(episode))
        & (payload["relation_phase"] >= 0.5)
    )
    if not len(rows):
        raise ValueError(f"episode {episode} has no relation-phase samples")
    active_right = bool(payload["active_arm_right"][rows[0]] >= 0.5)
    gripper_column = 19 if active_right else 9
    closed = rows[payload["state"][rows, gripper_column] < 0.5]
    candidates = closed if len(closed) else rows
    return int(candidates[np.argmin(payload["frame_index"][candidates])])


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", type=Path, required=True)
    result.add_argument("--library", type=Path, required=True)
    result.add_argument("--oracle-task-dataset", type=Path, required=True)
    result.add_argument(
        "--recovery-raw",
        action="append",
        type=Path,
        default=[],
        help="Recovery directories in the same order used by the merged dataset.",
    )
    result.add_argument("--fold", type=int, default=0)
    result.add_argument("--fit-threshold-m2", type=float, default=2.5e-4)
    result.add_argument("--output", type=Path, required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(args.dataset, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    fold = FOLDS[int(args.fold)]
    train_shoes = tuple(int(value) for value in fold["train"])
    transporter = OnlinePcaFlowTransporter(
        args.library, train_shoe_ids=train_shoes
    )
    estimator = FunctionalPcaFlowEstimator(
        args.library,
        args.oracle_task_dataset,
        train_shoe_ids=train_shoes,
    )
    with np.load(args.library, allow_pickle=False) as archive:
        library_shoes = archive["shoe_id"].astype(np.int64)
        target_frames = [
            value.astype(np.float32) for value in archive["target_frame_camera"]
        ]

    for raw_dir in args.recovery_raw:
        local_episode = 0
        while (raw_dir / "data" / f"episode{local_episode}.hdf5").is_file():
            path = raw_dir / "data" / f"episode{local_episode}.hdf5"
            with h5py.File(path, "r") as root:
                frame, _ = estimate_geometry_marker_frame(
                    np.asarray(root["object_pointcloud/{B}"][0])
                )
            target_frames.append(frame.astype(np.float32))
            local_episode += 1
        if local_episode == 0:
            raise FileNotFoundError(f"no recovery HDF5 episodes in {raw_dir}")

    episode_count = int(len(payload["episode_flow"]))
    if len(target_frames) != episode_count:
        raise ValueError(
            f"target frames cover {len(target_frames)} episodes, expected {episode_count}"
        )
    predicted = []
    costs = []
    margins = []
    accepted = []
    sources = []
    query_indices = []
    bank_sizes = []
    records = []
    original_transport_eligible = transporter.eligible.copy()
    original_relation_eligible = estimator.eligible_episodes.copy()
    for episode in range(episode_count):
        query = episode_query_index(payload, episode)
        query_indices.append(query)
        shoe = int(payload["shoe_id"][query])
        # Cross-fit training objects at the object rather than episode level.
        # This matches the intended generalization claim and prevents another
        # view of the identical shoe from being used as a prototype.
        if shoe in train_shoes:
            transporter.eligible = original_transport_eligible[
                library_shoes[original_transport_eligible] != shoe
            ]
            estimator.eligible_episodes = original_relation_eligible[
                library_shoes[original_relation_eligible] != shoe
            ]
        else:
            transporter.eligible = original_transport_eligible
            estimator.eligible_episodes = original_relation_eligible
        if not len(transporter.eligible) or not len(estimator.eligible_episodes):
            raise RuntimeError(f"episode {episode} has an empty cross-fit bank")
        bank_sizes.append(int(len(estimator.eligible_episodes)))

        base = transporter.predict(
            payload["points_a"][query],
            payload["points_b"][query],
            target_frame=target_frames[episode],
        )
        dataset_anchors = np.asarray(
            payload["current_anchors"][query], dtype=np.float32
        )
        base_flow = transfer_rigid_flow_to_anchors(
            base.initial_anchors,
            base.flow,
            dataset_anchors,
        )
        active_right = bool(payload["active_arm_right"][query] >= 0.5)
        active_pose = payload["current_eef_pose7"][query, 1 if active_right else 0]
        relation_flow, source, cost, margin = (
            estimator.predict_geometry_selected_grasp_flow(
                payload["points_a"][query],
                active_pose,
                base.target_frame,
                initial_anchors=dataset_anchors,
                base_flow=base_flow,
            )
        )
        use_relation = bool(float(cost) <= float(args.fit_threshold_m2))
        flow = relation_flow if use_relation else base_flow
        predicted.append(np.asarray(flow, dtype=np.float32))
        costs.append(float(cost))
        margins.append(float(margin))
        accepted.append(use_relation)
        sources.append(int(source if use_relation else base.source_episode))
        records.append(
            {
                "episode": episode,
                "query_index": query,
                "shoe_id": shoe,
                "is_recovery_episode": bool(
                    np.any(
                        payload["is_recovery_sample"][
                            payload["episode_index"] == episode
                        ]
                        >= 0.5
                    )
                ),
                "relation_accepted": use_relation,
                "relation_cost_m2": float(cost),
                "selection_margin_m2": float(margin),
                "source_episode": int(source if use_relation else base.source_episode),
                "cross_fit_bank_size": int(len(estimator.eligible_episodes)),
            }
        )

    transporter.eligible = original_transport_eligible
    estimator.eligible_episodes = original_relation_eligible
    prediction = np.stack(predicted).astype(np.float32)
    if prediction.shape != payload["episode_flow"].shape:
        raise RuntimeError(
            f"predicted flow {prediction.shape} != dataset flow {payload['episode_flow'].shape}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        episode_indices=np.arange(episode_count, dtype=np.int64),
        predicted_flow=prediction,
        relation_cost_m2=np.asarray(costs, dtype=np.float32),
        selection_margin_m2=np.asarray(margins, dtype=np.float32),
        relation_accepted=np.asarray(accepted, dtype=np.float32),
        source_episode=np.asarray(sources, dtype=np.int64),
        query_index=np.asarray(query_indices, dtype=np.int64),
        cross_fit_bank_size=np.asarray(bank_sizes, dtype=np.int64),
    )
    summary = {
        "dataset": str(args.dataset.resolve()),
        "library": str(args.library.resolve()),
        "oracle_task_dataset": str(args.oracle_task_dataset.resolve()),
        "recovery_raw": [str(path.resolve()) for path in args.recovery_raw],
        "output": str(args.output.resolve()),
        "fold": int(args.fold),
        "train_shoes": list(train_shoes),
        "episodes": episode_count,
        "recovery_episodes": int(sum(row["is_recovery_episode"] for row in records)),
        "fit_threshold_m2": float(args.fit_threshold_m2),
        "relation_acceptance": float(np.mean(accepted)),
        "mean_relation_cost_m2": float(np.mean(costs)),
        "training_interface": (
            "camera A/B XYZRGB + EEF; same-shoe-excluded cross-fitted relation flow"
        ),
        "records": records,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
