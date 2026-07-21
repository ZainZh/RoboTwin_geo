from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


DP3_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = DP3_ROOT / "scripts"
THIRD_PARTY_ROOT = DP3_ROOT / "third_party"
for path in (SCRIPTS_ROOT, THIRD_PARTY_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


COMMON_MODULES = ("h5py", "numpy", "torch", "trimesh", "zarr")
TRAIN_MODULES = ("diffusers", "hydra", "omegaconf")


def check_modules(names: tuple[str, ...]) -> list[str]:
    missing = []
    for name in names:
        try:
            importlib.import_module(name)
        except Exception as exc:
            missing.append(f"{name}: {exc}")
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description="Check dependencies for the shoe SE(3) comparison.")
    parser.add_argument("--route", choices=("baseline", "oracle", "ndf"), default="baseline")
    parser.add_argument("--training", action="store_true", help="Also check DP3 training packages.")
    args = parser.parse_args()

    missing = check_modules(COMMON_MODULES)
    if not missing:
        import zarr

        if int(str(zarr.__version__).split(".", 1)[0]) >= 3:
            missing.append(f"zarr: version {zarr.__version__} is unsupported; install zarr<3")
    if args.training:
        missing.extend(check_modules(TRAIN_MODULES))
    if args.route == "ndf":
        try:
            import ndf_robot.model.vnn_occupancy_net_pointnet_dgcnn  # noqa: F401
        except Exception as exc:
            missing.append(f"vendored ndf_robot: {exc}")

    if missing:
        print("Missing or broken dependencies:")
        for item in missing:
            print(f"- {item}")
        print(f"Install common packages with: pip install -r {DP3_ROOT / 'requirements_shoe_se3.txt'}")
        print("Install PyTorch through RoboTwin's script/requirements.txt or an equivalent CUDA-matched wheel.")
        return 1

    print(f"shoe SE(3) dependency check passed: route={args.route}, training={args.training}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
