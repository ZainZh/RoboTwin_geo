"""Online camera-only construction of a rigid object task flow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .build_grasp_dataset import deterministic_farthest_points, valid_xyz
from .evaluate_flow_transport import representation_cost, transport_flow
from .target_frame import estimate_geometry_marker_frame


class PointDescriptorEncoder(Protocol):
    def encode(
        self, support_points_world: np.ndarray, query_points_world: np.ndarray
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class OnlineFlowResult:
    flow: np.ndarray
    initial_anchors: np.ndarray
    source_episode: int
    source_shoe: int
    representation_cost: float
    target_frame: np.ndarray


class OnlinePcaFlowTransporter:
    """Retrieve a demonstration and transport its rigid flow online.

    PCA remains the zero-dependency default.  A frozen point-descriptor encoder
    can be supplied for the NDF condition while preserving exactly the same
    rigid-flow construction and downstream action policy.
    """

    def __init__(
        self,
        library: str | Path,
        *,
        train_shoe_ids: tuple[int, ...] | list[int],
        object_points: int = 256,
        target_points: int = 1024,
        representation: str = "pca",
        descriptor_encoder: PointDescriptorEncoder | None = None,
    ) -> None:
        with np.load(Path(library), allow_pickle=False) as archive:
            self.library = {key: archive[key] for key in archive.files}
        self.train_shoe_ids = tuple(int(value) for value in train_shoe_ids)
        self.object_points = int(object_points)
        self.target_points = int(target_points)
        self.representation = str(representation)
        self.descriptor_encoder = descriptor_encoder
        if self.representation not in {"pca", "ndf"}:
            raise ValueError(
                f"online representation must be 'pca' or 'ndf', got {self.representation!r}"
            )
        if self.representation == "ndf":
            if descriptor_encoder is None:
                raise ValueError("ndf online flow requires descriptor_encoder")
            if "ndf_features" not in self.library:
                raise KeyError("ndf online flow requires ndf_features in the library")
        self.episode_count = int(len(self.library["episode_flow"]))
        eligible = np.flatnonzero(
            np.isin(self.library["shoe_id"], self.train_shoe_ids)
        )
        if not len(eligible):
            raise ValueError("no library episodes match train_shoe_ids")
        self.eligible = eligible.astype(np.int64)

    @staticmethod
    def _valid_xyzrgb(point_cloud: np.ndarray) -> np.ndarray:
        value = np.asarray(point_cloud, dtype=np.float32)
        if value.ndim != 2 or value.shape[1] < 3:
            raise ValueError(f"point cloud must be [N,C>=3], got {value.shape}")
        keep = ~np.isclose(value[:, :3], 0.0).all(axis=1)
        value = value[keep]
        if not len(value):
            raise ValueError("point cloud is empty")
        if value.shape[1] < 6:
            value = np.concatenate(
                (value[:, :3], np.zeros((len(value), 3), dtype=np.float32)), axis=-1
            )
        return value[:, :6]

    def _sample_target(self, point_cloud: np.ndarray) -> np.ndarray:
        value = self._valid_xyzrgb(point_cloud)
        if len(value) >= self.target_points:
            return value[: self.target_points].copy()
        indices = np.arange(self.target_points) % len(value)
        return value[indices].copy()

    def predict(
        self,
        operated_point_cloud: np.ndarray,
        target_point_cloud: np.ndarray,
    ) -> OnlineFlowResult:
        operated_full = self._valid_xyzrgb(operated_point_cloud)
        operated_xyz = valid_xyz(operated_full)
        operated_indices_xyz = deterministic_farthest_points(
            operated_xyz, self.object_points
        )
        # Recover RGB by nearest neighbor. XYZ is the only input to PCA, but
        # retaining a valid XYZRGB schema makes the appended target compatible
        # with other representation backends.
        square = np.sum(
            (operated_indices_xyz[:, None] - operated_full[None, :, :3]) ** 2,
            axis=-1,
        )
        nearest = np.argmin(square, axis=1)
        operated = operated_full[nearest].copy()
        operated[:, :3] = operated_indices_xyz
        target = self._sample_target(target_point_cloud)
        anchors = operated[: self.library["initial_anchors"].shape[1], :3].copy()
        target_frame, _ = estimate_geometry_marker_frame(target)

        task_steps = int(self.library["episode_flow"].shape[2])
        dummy_flow = np.repeat(anchors[:, None, :], task_steps, axis=1)
        payload = dict(self.library)
        payload["points_a_xyzrgb"] = np.concatenate(
            (self.library["points_a_xyzrgb"], operated[None]), axis=0
        )
        payload["points_b_xyzrgb"] = np.concatenate(
            (self.library["points_b_xyzrgb"], target[None]), axis=0
        )
        payload["initial_anchors"] = np.concatenate(
            (self.library["initial_anchors"], anchors[None]), axis=0
        )
        payload["episode_flow"] = np.concatenate(
            (self.library["episode_flow"], dummy_flow[None]), axis=0
        )
        payload["target_frame_camera"] = np.concatenate(
            (self.library["target_frame_camera"], target_frame[None]), axis=0
        )
        if self.representation == "ndf":
            current_features = self.descriptor_encoder.encode(
                operated_full[:, :3], operated[:, :3]
            )
            expected_dim = int(self.library["ndf_features"].shape[-1])
            if current_features.shape[1] < expected_dim:
                raise ValueError(
                    f"online NDF descriptor has {current_features.shape[1]} channels, "
                    f"library requires {expected_dim}"
                )
            payload["ndf_features"] = np.concatenate(
                (
                    self.library["ndf_features"],
                    current_features[None, :, :expected_dim],
                ),
                axis=0,
            )
        target_index = self.episode_count
        target_right = float(operated[:, 0].mean()) >= 0.0
        compatible = self.eligible[
            np.asarray(
                [
                    float(self.library["points_a_xyzrgb"][source, :, 0].mean()) >= 0.0
                    for source in self.eligible
                ]
            )
            == target_right
        ]
        if not len(compatible):
            compatible = self.eligible
        costs = np.asarray(
            [
                representation_cost(
                    payload, int(source), target_index, self.representation
                )
                for source in compatible
            ]
        )
        source = int(compatible[int(np.argmin(costs))])
        flow = transport_flow(payload, source, target_index, self.representation)
        return OnlineFlowResult(
            flow=flow,
            initial_anchors=anchors.astype(np.float32),
            source_episode=source,
            source_shoe=int(self.library["shoe_id"][source]),
            representation_cost=float(np.min(costs)),
            target_frame=target_frame.astype(np.float32),
        )
