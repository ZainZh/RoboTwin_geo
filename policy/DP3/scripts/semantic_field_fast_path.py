"""Isolated semantic-only fast path for the released universal-field trainer.

The released dataset samples occupancy/probe points even in semantic mode.
Those points are unused by the semantic loss, but sampling calls
``mesh.contains``, assigns nearest-surface labels, collates the tensors, and
moves them to CUDA.  The released model also builds the frozen pose tri-plane
although no pose/occupancy query is executed.

This module installs an opt-in runtime patch.  It deliberately consumes the
same NumPy random draws as the released query sampler, preserving every later
support/query augmentation, while replacing only unused labels and tensors.
The external release checkout is never edited.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable

import numpy as np
import torch
from torch.nn.parameter import UninitializedParameter


IGNORE_LABEL = -1
SEMANTIC_SAMPLE_KEYS = (
    "utonia_input_view1",
    "support_points_view1",
    "query_surface_points_view1",
    "query_surface_points_canonical",
    "query_surface_correspondence",
    "query_surface_labels",
    "category",
    "model_id",
    "mesh_path",
)
SEMANTIC_VIEW2_KEYS = (
    "utonia_input_view2",
    "support_points_view2",
    "query_surface_points_view2",
)


def make_rng_equivalent_query_sampler(
    sample_surface: Callable[[Any, int], tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> Callable[..., tuple[np.ndarray, np.ndarray]]:
    """Return a query sampler with the release RNG stream but no containment.

    The point sampling, Gaussian perturbations, uniform samples, and final
    permutation intentionally match the release implementation statement for
    statement.  Occupancy labels are unused in semantic mode, so zero
    placeholders replace ``mesh.contains(query_points)``.
    """

    def sample_queries(
        mesh: Any,
        num_points: int,
        near_surface_ratio: float,
        near_surface_noise: float,
        uniform_padding: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        bbox_min, bbox_max = mesh.bounds
        extent = float(np.max(bbox_max - bbox_min))
        extent = max(extent, 1e-3)

        num_near = int(num_points * near_surface_ratio)
        num_uniform = num_points - num_near
        num_near_small = num_near // 2
        num_near_large = num_near - num_near_small

        near_small, _, _ = sample_surface(mesh, num_near_small)
        near_large, _, _ = sample_surface(mesh, num_near_large)
        near_small += np.random.normal(
            loc=0.0,
            scale=extent * near_surface_noise,
            size=near_small.shape,
        ).astype(np.float32)
        near_large += np.random.normal(
            loc=0.0,
            scale=extent * near_surface_noise * 4.0,
            size=near_large.shape,
        ).astype(np.float32)

        padding = extent * uniform_padding
        uniform = np.random.uniform(
            low=bbox_min - padding,
            high=bbox_max + padding,
            size=(num_uniform, 3),
        ).astype(np.float32)
        query_points = np.concatenate([near_small, near_large, uniform], axis=0)

        # Trimesh 4.4.3's ray containment retries broken bidirectional rays
        # with ``np.random.random(3)``.  The PartNext near-surface queries
        # trigger that retry in the matched Hammer recipe.  Consume those
        # three draws without running the ray test so later rotations,
        # jitters, and GridSample choices remain bitwise identical.
        np.random.random(3)

        # This placeholder is discarded by semantic_only_collate_fn.
        occupancy = np.zeros((len(query_points),), dtype=np.float32)
        permutation = np.random.permutation(len(query_points))
        return query_points[permutation], occupancy[permutation]

    return sample_queries


def semantic_only_probe_labels(
    probe_points: np.ndarray,
    surface_points: np.ndarray,
    surface_labels: np.ndarray,
    max_distance: float,
) -> np.ndarray:
    """Return discarded placeholders without the O(probe*surface) nearest pass."""

    del surface_points, surface_labels, max_distance
    return np.full((probe_points.shape[0],), IGNORE_LABEL, dtype=np.int64)


def semantic_only_collate_fn(
    batch: list[dict],
    *,
    utonia_collate_fn: Callable[[list[dict]], dict],
) -> dict:
    """Collate only tensors consumed by semantic forward/loss/metrics."""

    collated = {
        "utonia_input_view1": utonia_collate_fn(
            [item["utonia_input_view1"] for item in batch]
        ),
        "support_points_view1": torch.stack(
            [item["support_points_view1"] for item in batch], dim=0
        ),
        "query_surface_points_view1": torch.stack(
            [item["query_surface_points_view1"] for item in batch], dim=0
        ),
        "query_surface_points_canonical": torch.stack(
            [item["query_surface_points_canonical"] for item in batch], dim=0
        ),
        "query_surface_correspondence": torch.stack(
            [item["query_surface_correspondence"] for item in batch], dim=0
        ),
        "query_surface_labels": torch.stack(
            [item["query_surface_labels"] for item in batch], dim=0
        ),
        "category": [item["category"] for item in batch],
        "model_id": [item["model_id"] for item in batch],
        "mesh_path": [item["mesh_path"] for item in batch],
    }
    if "utonia_input_view2" in batch[0]:
        collated["utonia_input_view2"] = utonia_collate_fn(
            [item["utonia_input_view2"] for item in batch]
        )
        collated["support_points_view2"] = torch.stack(
            [item["support_points_view2"] for item in batch], dim=0
        )
        collated["query_surface_points_view2"] = torch.stack(
            [item["query_surface_points_view2"] for item in batch], dim=0
        )
    return collated


def semantic_only_encode_support(
    model: Any,
    utonia_input: dict,
    support_points: torch.Tensor,
) -> dict[str, Any]:
    """Build semantic cache; materialize a lazy pose adapter once for checkpoints."""

    num_support_points = int(support_points.shape[1])
    support_features = model.feature_extractor(
        utonia_input,
        num_support_points=num_support_points,
    )
    semantic_support = model.semantic_adapter(support_features)
    pose_parameters = getattr(model.pose_adapter, "parameters", lambda: ())()
    if any(
        isinstance(parameter, UninitializedParameter)
        for parameter in pose_parameters
    ):
        model.pose_adapter(support_features)

    semantic_cache = model.semantic_projector.build(
        support_points=support_points,
        support_features=semantic_support,
    )
    return {"semantic": semantic_cache}


def semantic_only_forward(
    model: Any,
    batch: dict,
    train_mode: str = "semantic",
) -> dict[str, dict[str, torch.Tensor]]:
    """Semantic forward that never indexes absent occupancy/probe tensors."""

    if train_mode != "semantic":
        raise ValueError(
            "semantic-only fast path requires --train-mode semantic; "
            f"received {train_mode!r}"
        )

    def encode(suffix: str) -> dict[str, torch.Tensor]:
        surface_points = batch[f"query_surface_points_{suffix}"]
        return model.encode_view(
            utonia_input=batch[f"utonia_input_{suffix}"],
            support_points=batch[f"support_points_{suffix}"],
            query_surface_points=surface_points,
            query_occ_points=surface_points[:, :0],
            compute_semantic=True,
            compute_pose=False,
            compute_occupancy=False,
        )

    outputs = {"view1": encode("view1")}
    if "utonia_input_view2" in batch:
        outputs["view2"] = encode("view2")
    return outputs




def make_zero_weight_aware_semantic_loss(trainer: Any) -> Callable[..., dict]:
    """Skip CE/SupCon kernels whose coefficient is exactly zero."""

    original = trainer._compute_loss_terms

    def compute(outputs: dict, batch: dict, args: Any) -> dict:
        if (
            args.train_mode != "semantic"
            or (
                float(args.sem_ce_weight) != 0.0
                and float(args.sem_contrastive_weight) != 0.0
            )
        ):
            return original(outputs, batch, args)

        labels = batch["query_surface_labels"]
        zero = batch["query_surface_points_view1"].new_zeros(())
        sem_ce = zero
        sem_contrastive = zero
        sem_consistency = zero
        views = [outputs["view1"]]
        if "view2" in outputs and "sem_embeddings" in outputs["view2"]:
            views.append(outputs["view2"])

        if float(args.sem_ce_weight) != 0.0:
            sem_ce = torch.stack(
                [
                    trainer.semantic_cross_entropy_loss(
                        view["sem_logits"],
                        labels,
                        label_smoothing=args.sem_label_smoothing,
                    )
                    for view in views
                ]
            ).mean()
        if float(args.sem_contrastive_weight) != 0.0:
            sem_contrastive = torch.stack(
                [
                    trainer.universal_supervised_contrastive_loss(
                        view["sem_embeddings"],
                        labels,
                        temperature=args.sem_contrastive_temperature,
                        max_points=args.sem_contrastive_max_points,
                    )
                    for view in views
                ]
            ).mean()
        else:
            # The release SupCon loss samples points with one randperm per
            # view before its zero coefficient is applied. Preserve those
            # draws so subsequent stochastic training follows the same RNG
            # trajectory while eliding the expensive similarity matrices.
            valid_labels = labels.reshape(-1)
            valid_labels = valid_labels[valid_labels != -1]
            if valid_labels.numel() > args.sem_contrastive_max_points:
                for _ in views:
                    torch.randperm(
                        valid_labels.numel(), device=valid_labels.device
                    )[: args.sem_contrastive_max_points]
        if len(views) == 2:
            sem_consistency = trainer.embedding_consistency_loss(
                views[0]["sem_embeddings"],
                views[1]["sem_embeddings"],
            )

        total = (
            args.sem_ce_weight * sem_ce
            + args.sem_contrastive_weight * sem_contrastive
            + args.sem_consistency_weight * sem_consistency
        )
        return {
            "loss": total,
            "sem_ce": sem_ce,
            "sem_contrastive": sem_contrastive,
            "pose_dense_correspondence": zero,
            "pose_metric": zero,
            "pose_consistency": zero,
            "geo_dense_correspondence": zero,
            "geo_metric": zero,
            "geo_consistency": zero,
            "sem_consistency": sem_consistency,
            "occ": zero,
        }

    return compute


def install_semantic_only_fast_path(trainer: Any) -> None:
    """Patch an imported release trainer for a fresh semantic-only process."""

    from my_datasets import partnext_universal_field as dataset_module
    from my_datasets import partnext_occupancy as occupancy_module
    from models.universal_field import utonia_universal_field as model_module

    original_loss = trainer._compute_loss_terms
    original_build_argparser = trainer.build_argparser

    def build_fast_argparser():
        parser = original_build_argparser()
        parser.set_defaults(compute_elided_zero_weight_losses=True)
        return parser

    dataset_module._sample_queries = make_rng_equivalent_query_sampler(
        occupancy_module._sample_surface
    )
    dataset_module._assign_nearest_surface_labels = semantic_only_probe_labels
    trainer.partnext_universal_field_collate_fn = partial(
        semantic_only_collate_fn,
        utonia_collate_fn=dataset_module.UTONIA.data.collate_fn,
    )
    model_module.UtoniaUniversalFieldNet.encode_support = semantic_only_encode_support
    model_module.UtoniaUniversalFieldNet.forward = semantic_only_forward
    trainer._compute_loss_terms = make_zero_weight_aware_semantic_loss(trainer)

    trainer.build_argparser = build_fast_argparser
    installed = (
        trainer.UtoniaUniversalFieldNet is model_module.UtoniaUniversalFieldNet
        and dataset_module._assign_nearest_surface_labels
        is semantic_only_probe_labels
        and model_module.UtoniaUniversalFieldNet.encode_support
        is semantic_only_encode_support
        and model_module.UtoniaUniversalFieldNet.forward is semantic_only_forward
        and isinstance(trainer.partnext_universal_field_collate_fn, partial)
        and trainer.partnext_universal_field_collate_fn.func
        is semantic_only_collate_fn
        and trainer._compute_loss_terms is not original_loss
        and trainer.build_argparser is build_fast_argparser
    )
    if not installed:
        raise RuntimeError("semantic fast path identity verification failed")
    trainer.SEMANTIC_FAST_PATH_ENABLED = True
    print(
        "[semantic-fast-path] enabled: RNG-equivalent no-contains queries, "
        "no nearest/occ-probe collate/pose cache, zero-weight CE/SupCon skipped",
        flush=True,
    )
