#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from ndf_feature_utils import load_ndf_model


REPO_ROOT = Path(__file__).resolve().parents[3]

RAMP_FUNCTIONAL = np.array(
    [
        [0.0, -1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
RAMP_UPHILL_AXIS = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
RAMP_NORMAL_AXIS = np.array([0.0, 0.0, 1.0], dtype=np.float64)
SHOE_SOLE_NORMAL = np.array([0.0, 1.0, 0.0], dtype=np.float64)


@dataclass(frozen=True)
class TrialResult:
    query_shoe_id: int
    trial: int
    direction_weight: float
    target_feature_energy: float
    flipped_feature_energy: float
    feature_margin: float
    predicted_feature_energy: float
    predicted_direction_energy: float
    total_energy: float
    predicted_transform_a_from_b: list[list[float]]
    translation_error_m: float
    rotation_error_deg: float
    toe_cosine: float
    flipped: bool
    success: bool


def functional_matrix_with_scale(matrix: np.ndarray, scale: Sequence[float]) -> np.ndarray:
    result = np.asarray(matrix, dtype=np.float64).copy()
    if result.shape != (4, 4):
        raise ValueError(f"functional matrix must be 4x4, got {result.shape}")
    scale_arr = np.asarray(scale, dtype=np.float64).reshape(3)
    result[:3, 3] *= scale_arr
    return result


def goal_shoe_from_ramp(shoe_functional: np.ndarray, ramp_functional: np.ndarray) -> np.ndarray:
    """Return the actor-local ramp pose that aligns the two functional frames."""
    return np.asarray(shoe_functional, dtype=np.float64) @ np.linalg.inv(
        np.asarray(ramp_functional, dtype=np.float64)
    )


def axis_alignment_cosine(
    transform_target_from_source: np.ndarray,
    *,
    source_axis: np.ndarray,
    target_axis: np.ndarray,
) -> float:
    transform = np.asarray(transform_target_from_source, dtype=np.float64)
    source = _unit_vector(source_axis)
    target = _unit_vector(target_axis)
    return float(np.clip(np.dot(transform[:3, :3] @ source, target), -1.0, 1.0))


def frame_alignment_loss(transform_shoe_from_ramp: np.ndarray) -> float:
    toe_loss = 1.0 - axis_alignment_cosine(
        transform_shoe_from_ramp,
        source_axis=RAMP_UPHILL_AXIS,
        target_axis=np.array([0.0, 0.0, 1.0]),
    )
    normal_loss = 1.0 - axis_alignment_cosine(
        transform_shoe_from_ramp,
        source_axis=RAMP_NORMAL_AXIS,
        target_axis=SHOE_SOLE_NORMAL,
    )
    return float(toe_loss + normal_loss)


def rotation_about_point(*, axis: np.ndarray, angle_rad: float, point: np.ndarray) -> np.ndarray:
    axis = _unit_vector(axis)
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    rotation = np.eye(3) + math.sin(angle_rad) * skew + (1.0 - math.cos(angle_rad)) * (skew @ skew)
    point = np.asarray(point, dtype=np.float64).reshape(3)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = point - rotation @ point
    return transform


def _unit_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("direction vector must be non-zero")
    return vector / norm


def configure_descriptor_acts(model: torch.nn.Module, descriptor_acts: str) -> None:
    descriptor_acts = str(descriptor_acts).lower()
    valid = {"all", "last"}
    if descriptor_acts not in valid:
        raise ValueError(f"descriptor_acts must be one of {sorted(valid)}, got {descriptor_acts!r}")
    model.decoder.acts = descriptor_acts


def _parse_csv_numbers(value: str, cast) -> list:
    return [cast(item.strip()) for item in str(value).split(",") if item.strip()]


def _load_shoe_metadata(asset_root: Path, shoe_id: int) -> dict:
    path = asset_root / f"model_data{shoe_id}.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _shoe_functional_and_toe(metadata: dict) -> tuple[np.ndarray, np.ndarray]:
    scale = np.asarray(metadata["scale"], dtype=np.float64)
    functional = functional_matrix_with_scale(np.asarray(metadata["functional_matrix"][0]), scale)
    orientation = np.asarray(metadata["orientation_point"], dtype=np.float64)
    toe_vector = (orientation[:3, 3] - np.asarray(metadata["functional_matrix"][0])[:3, 3]) * scale

    # Mesh-local +Y is the sole normal. Project the annotation onto the sole plane
    # so the constraint controls toe/heel direction without constraining pitch.
    sole_normal = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    toe_vector = toe_vector - np.dot(toe_vector, sole_normal) * sole_normal
    return functional, _unit_vector(toe_vector)


def _load_normalized_shoe_cloud(
    asset_root: Path,
    shoe_id: int,
    *,
    point_count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    import trimesh

    metadata = _load_shoe_metadata(asset_root, shoe_id)
    mesh = trimesh.load(asset_root / "visual" / f"base{shoe_id}.glb", force="mesh")
    sampled, _ = trimesh.sample.sample_surface(mesh, int(point_count), seed=int(seed))
    actor_points = sampled.astype(np.float32) * np.asarray(metadata["scale"], dtype=np.float32)[None, :]
    center = actor_points.mean(axis=0).astype(np.float32)
    extents = actor_points.max(axis=0) - actor_points.min(axis=0)
    scale = max(float(np.max(extents)), 1e-6)
    normalized = ((actor_points - center[None, :]) / scale).astype(np.float32)
    return normalized, center, scale


def _ramp_probe_grid(nx: int, ny: int, *, half_length: float = 0.13, half_width: float = 0.05) -> np.ndarray:
    xs = np.linspace(-float(half_length), float(half_length), int(nx), dtype=np.float32)
    ys = np.linspace(-float(half_width), float(half_width), int(ny), dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    zz = np.full_like(xx, 0.0005)
    return np.stack([xx.reshape(-1), yy.reshape(-1), zz.reshape(-1)], axis=-1)


def _so3_exp(omega: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    theta = torch.linalg.norm(omega, dim=-1, keepdim=True).clamp_min(eps)
    axis = omega / theta
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack(
        [
            torch.stack([zero, -z, y], dim=-1),
            torch.stack([z, zero, -x], dim=-1),
            torch.stack([-y, x, zero], dim=-1),
        ],
        dim=-2,
    )
    eye = torch.eye(3, dtype=omega.dtype, device=omega.device).expand(skew.shape)
    return eye + torch.sin(theta)[..., None] * skew + (1.0 - torch.cos(theta))[..., None] * (skew @ skew)


def _matrix_from_rt(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = np.asarray(rotation, dtype=np.float64)
    result[:3, 3] = np.asarray(translation, dtype=np.float64)
    return result


def _rotation_error_deg(predicted: np.ndarray, target: np.ndarray) -> float:
    delta = predicted[:3, :3] @ target[:3, :3].T
    cosine = float(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _feature_energy(
    model: torch.nn.Module,
    latent: torch.Tensor,
    reference_features: torch.Tensor,
    probes: torch.Tensor,
    center: torch.Tensor,
    scale: float,
    transform: np.ndarray,
) -> float:
    transform_t = torch.as_tensor(transform, dtype=probes.dtype, device=probes.device)
    query = probes @ transform_t[:3, :3].T + transform_t[:3, 3]
    query = (query - center) / float(scale)
    with torch.no_grad():
        features = model.forward_latent(latent, query[None, ...]).squeeze(0)
        energy = (features - reference_features).abs().sum(dim=-1).mean()
    return float(energy.item())


def _optimize_relation(
    *,
    model: torch.nn.Module,
    latent: torch.Tensor,
    reference_features: torch.Tensor,
    probes: torch.Tensor,
    center: torch.Tensor,
    scale: float,
    toe_axis: np.ndarray,
    direction_weight: float,
    normal_direction_ratio: float,
    restarts: int,
    iterations: int,
    learning_rate: float,
    init_translation_m: float,
    seed: int,
) -> tuple[np.ndarray, float, float, float]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    axes = torch.randn((int(restarts), 3), generator=generator)
    axes = axes / torch.linalg.norm(axes, dim=-1, keepdim=True).clamp_min(1e-8)
    angles = (torch.rand((int(restarts), 1), generator=generator) * 2.0 - 1.0) * math.pi
    translations = (torch.rand((int(restarts), 3), generator=generator) * 2.0 - 1.0) * float(
        init_translation_m
    )
    se3 = torch.cat([axes * angles, translations], dim=-1).to(probes.device).requires_grad_(True)
    optimizer = torch.optim.Adam([se3], lr=float(learning_rate))

    latent_batch = latent.repeat(int(restarts), *([1] * (latent.ndim - 1)))
    probe_batch = probes.unsqueeze(0).expand(int(restarts), -1, -1)
    reference_batch = reference_features.unsqueeze(0)
    center_batch = center.reshape(1, 1, 3)
    uphill = torch.as_tensor(RAMP_UPHILL_AXIS, dtype=probes.dtype, device=probes.device)
    toe = torch.as_tensor(toe_axis, dtype=probes.dtype, device=probes.device)
    ramp_normal = torch.as_tensor(RAMP_NORMAL_AXIS, dtype=probes.dtype, device=probes.device)
    sole_normal = torch.as_tensor(SHOE_SOLE_NORMAL, dtype=probes.dtype, device=probes.device)

    for _ in range(int(iterations)):
        optimizer.zero_grad(set_to_none=True)
        rotation = _so3_exp(se3[:, :3])
        translation = se3[:, 3:]
        query = probe_batch @ rotation.transpose(-1, -2) + translation[:, None, :]
        query = (query - center_batch) / float(scale)
        features = model.forward_latent(latent_batch, query)
        feature_energy = (features - reference_batch).abs().sum(dim=-1).mean(dim=-1)
        mapped_uphill = torch.einsum("bij,j->bi", rotation, uphill)
        mapped_normal = torch.einsum("bij,j->bi", rotation, ramp_normal)
        toe_energy = 1.0 - torch.sum(mapped_uphill * toe[None, :], dim=-1)
        normal_energy = 1.0 - torch.sum(mapped_normal * sole_normal[None, :], dim=-1)
        direction_energy = toe_energy + float(normal_direction_ratio) * normal_energy
        total = feature_energy + float(direction_weight) * direction_energy
        total.sum().backward()
        optimizer.step()

    with torch.no_grad():
        rotation = _so3_exp(se3[:, :3])
        translation = se3[:, 3:]
        query = probe_batch @ rotation.transpose(-1, -2) + translation[:, None, :]
        query = (query - center_batch) / float(scale)
        features = model.forward_latent(latent_batch, query)
        feature_energy = (features - reference_batch).abs().sum(dim=-1).mean(dim=-1)
        mapped_uphill = torch.einsum("bij,j->bi", rotation, uphill)
        mapped_normal = torch.einsum("bij,j->bi", rotation, ramp_normal)
        toe_energy = 1.0 - torch.sum(mapped_uphill * toe[None, :], dim=-1)
        normal_energy = 1.0 - torch.sum(mapped_normal * sole_normal[None, :], dim=-1)
        direction_energy = toe_energy + float(normal_direction_ratio) * normal_energy
        total = feature_energy + float(direction_weight) * direction_energy
        best = int(torch.argmin(total).item())

    transform = _matrix_from_rt(
        rotation[best].detach().cpu().numpy(),
        translation[best].detach().cpu().numpy(),
    )
    return (
        transform,
        float(feature_energy[best].item()),
        float(direction_energy[best].item()),
        float(total[best].item()),
    )


def _summarize(results: Iterable[TrialResult]) -> dict:
    rows = list(results)
    groups: dict[str, list[TrialResult]] = {}
    for row in rows:
        groups.setdefault(f"direction_weight={row.direction_weight:g}", []).append(row)

    summary = {}
    for name, group in groups.items():
        summary[name] = {
            "trials": len(group),
            "success_rate": float(np.mean([row.success for row in group])),
            "flip_rate": float(np.mean([row.flipped for row in group])),
            "median_translation_error_m": float(np.median([row.translation_error_m for row in group])),
            "median_rotation_error_deg": float(np.median([row.rotation_error_deg for row in group])),
            "median_toe_cosine": float(np.median([row.toe_cosine for row in group])),
            "target_beats_flip_rate": float(np.mean([row.feature_margin > 0.0 for row in group])),
        }
    return summary


def run_validation(args: argparse.Namespace) -> dict:

    device = torch.device(args.device)
    asset_root = Path(args.asset_root)
    model = load_ndf_model(str(args.checkpoint), dgcnn=False, device=device)
    configure_descriptor_acts(model, args.descriptor_acts)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    demo_cloud, demo_center, demo_scale = _load_normalized_shoe_cloud(
        asset_root,
        args.demo_shoe_id,
        point_count=args.point_count,
        seed=args.sample_seed,
    )
    demo_metadata = _load_shoe_metadata(asset_root, args.demo_shoe_id)
    demo_functional, _ = _shoe_functional_and_toe(demo_metadata)
    demo_goal = goal_shoe_from_ramp(demo_functional, RAMP_FUNCTIONAL)
    probes_np = _ramp_probe_grid(
        args.probe_grid_x,
        args.probe_grid_y,
        half_length=args.probe_half_length_m,
        half_width=args.probe_half_width_m,
    )

    demo_cloud_t = torch.from_numpy(demo_cloud[None, ...]).to(device)
    probes_t = torch.from_numpy(probes_np).to(device)
    demo_center_t = torch.from_numpy(demo_center).to(device)
    with torch.no_grad():
        demo_latent = model.extract_latent({"point_cloud": demo_cloud_t})
        demo_transform_t = torch.as_tensor(demo_goal, dtype=probes_t.dtype, device=device)
        demo_queries = probes_t @ demo_transform_t[:3, :3].T + demo_transform_t[:3, 3]
        demo_queries = (demo_queries - demo_center_t) / float(demo_scale)
        reference_features = model.forward_latent(demo_latent, demo_queries[None, ...]).squeeze(0).detach()

    rows: list[TrialResult] = []
    for shoe_id in args.query_shoe_ids:
        query_cloud, query_center, query_scale = _load_normalized_shoe_cloud(
            asset_root,
            shoe_id,
            point_count=args.point_count,
            seed=args.sample_seed + 1000 + shoe_id,
        )
        metadata = _load_shoe_metadata(asset_root, shoe_id)
        shoe_functional, toe_axis = _shoe_functional_and_toe(metadata)
        target = goal_shoe_from_ramp(shoe_functional, RAMP_FUNCTIONAL)
        flip = rotation_about_point(
            axis=np.array([0.0, 1.0, 0.0]),
            angle_rad=np.pi,
            point=shoe_functional[:3, 3],
        ) @ target

        query_cloud_t = torch.from_numpy(query_cloud[None, ...]).to(device)
        query_center_t = torch.from_numpy(query_center).to(device)
        with torch.no_grad():
            latent = model.extract_latent({"point_cloud": query_cloud_t}).detach()
        target_energy = _feature_energy(
            model, latent, reference_features, probes_t, query_center_t, query_scale, target
        )
        flip_energy = _feature_energy(
            model, latent, reference_features, probes_t, query_center_t, query_scale, flip
        )

        for direction_weight in args.direction_weights:
            for trial in range(args.trials):
                predicted, feature_energy, direction_energy, total_energy = _optimize_relation(
                    model=model,
                    latent=latent,
                    reference_features=reference_features,
                    probes=probes_t,
                    center=query_center_t,
                    scale=query_scale,
                    toe_axis=toe_axis,
                    direction_weight=direction_weight,
                    normal_direction_ratio=args.normal_direction_ratio,
                    restarts=args.restarts,
                    iterations=args.iterations,
                    learning_rate=args.learning_rate,
                    init_translation_m=args.init_translation_m,
                    seed=args.optimizer_seed + 10000 * shoe_id + trial,
                )
                translation_error = float(np.linalg.norm(predicted[:3, 3] - target[:3, 3]))
                rotation_error = _rotation_error_deg(predicted, target)
                toe_cosine = axis_alignment_cosine(
                    predicted,
                    source_axis=RAMP_UPHILL_AXIS,
                    target_axis=toe_axis,
                )
                flipped = toe_cosine < 0.0
                success = (
                    translation_error <= args.success_translation_m
                    and rotation_error <= args.success_rotation_deg
                    and toe_cosine >= args.success_toe_cosine
                )
                row = TrialResult(
                    query_shoe_id=shoe_id,
                    trial=trial,
                    direction_weight=direction_weight,
                    target_feature_energy=target_energy,
                    flipped_feature_energy=flip_energy,
                    feature_margin=flip_energy - target_energy,
                    predicted_feature_energy=feature_energy,
                    predicted_direction_energy=direction_energy,
                    total_energy=total_energy,
                    predicted_transform_a_from_b=predicted.tolist(),
                    translation_error_m=translation_error,
                    rotation_error_deg=rotation_error,
                    toe_cosine=toe_cosine,
                    flipped=flipped,
                    success=success,
                )
                rows.append(row)
                print(
                    f"shoe={shoe_id} trial={trial} dir_w={direction_weight:g} "
                    f"target/flip={target_energy:.4f}/{flip_energy:.4f} "
                    f"trans={translation_error:.4f}m rot={rotation_error:.2f}deg "
                    f"toe_cos={toe_cosine:.3f} flip={flipped} success={success}"
                )

    result = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "summary": _summarize(rows),
        "trials": [asdict(row) for row in rows],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(f"wrote {output}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate one-way NDF SE(3) shoe-ramp alignment with an explicit toe/uphill constraint."
    )
    parser.add_argument("--checkpoint", default="/home/zheng/model/ndf/shoe.pth")
    parser.add_argument("--asset_root", default=str(REPO_ROOT / "assets" / "objects" / "041_shoe"))
    parser.add_argument("--demo_shoe_id", type=int, default=0)
    parser.add_argument("--query_shoe_ids", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--descriptor_acts", choices=["all", "last"], default="last")
    parser.add_argument("--point_count", type=int, default=2048)
    parser.add_argument("--probe_grid_x", type=int, default=11)
    parser.add_argument("--probe_grid_y", type=int, default=5)
    parser.add_argument("--probe_half_length_m", type=float, default=0.09)
    parser.add_argument("--probe_half_width_m", type=float, default=0.035)
    parser.add_argument("--direction_weights", default="0,5")
    parser.add_argument("--normal_direction_ratio", type=float, default=1.0)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--restarts", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--learning_rate", type=float, default=0.03)
    parser.add_argument("--init_translation_m", type=float, default=0.15)
    parser.add_argument("--sample_seed", type=int, default=128)
    parser.add_argument("--optimizer_seed", type=int, default=2026)
    parser.add_argument("--success_translation_m", type=float, default=0.03)
    parser.add_argument("--success_rotation_deg", type=float, default=15.0)
    parser.add_argument("--success_toe_cosine", type=float, default=0.9)
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "outputs" / "ndf_shoe_ramp_se3" / "validation.json"),
    )
    args = parser.parse_args()
    args.query_shoe_ids = _parse_csv_numbers(args.query_shoe_ids, int)
    args.direction_weights = _parse_csv_numbers(args.direction_weights, float)
    return args


def main() -> None:
    run_validation(parse_args())


if __name__ == "__main__":
    main()
