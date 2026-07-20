import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Validate an NDF relation-token v1 preprocessing cache.")
    parser.add_argument("meta_path")
    args = parser.parse_args()

    path = Path(args.meta_path)
    with path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    expected_dim = 9 + int(metadata.get("ndf_feat_dim", 256))
    checks = {
        "relation_token_schema_version": 1,
        "relation_token_dim": expected_dim,
        "relation_token_geometry_dim": 9,
        "relation_token_descriptor_pool": "valid_query_mean",
        "save_relation_tokens": True,
    }
    errors = [
        f"{key}: expected {expected!r}, got {metadata.get(key)!r}"
        for key, expected in checks.items()
        if metadata.get(key) != expected
    ]
    if errors:
        raise ValueError("Invalid NDF relation-token v1 metadata: " + "; ".join(errors))
    print(f"Validated NDF relation-token v1 metadata: {path}")


if __name__ == "__main__":
    main()
