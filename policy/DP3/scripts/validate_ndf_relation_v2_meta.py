import argparse
import json
from pathlib import Path

from ndf_relation_schema import validate_relation_v2_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Reject stale or incompatible NDF relation-v2 caches.")
    parser.add_argument("meta_path", type=Path)
    args = parser.parse_args()

    if not args.meta_path.is_file():
        raise FileNotFoundError(f"NDF relation-v2 metadata does not exist: {args.meta_path}")
    with args.meta_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    validate_relation_v2_metadata(metadata)
    print(f"validated NDF relation-v2 metadata: {args.meta_path}")


if __name__ == "__main__":
    main()
