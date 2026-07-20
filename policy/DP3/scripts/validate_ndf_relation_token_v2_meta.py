import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Validate an NDF relation-token v2 preprocessing cache.")
    parser.add_argument("meta_path")
    args = parser.parse_args()

    path = Path(args.meta_path)
    with path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    projection_dim = int(metadata.get("relation_token_projection_dim", -1))
    checks = {
        "relation_token_schema_version": 2,
        "relation_token_dim": 9 + projection_dim,
        "relation_token_geometry_dim": 9,
        "relation_token_descriptor_pool": "valid_query_mean_fixed_random_projection",
        "relation_token_gate_geometry": True,
        "save_relation_tokens": True,
    }
    errors = [
        f"{key}: expected {expected!r}, got {metadata.get(key)!r}"
        for key, expected in checks.items()
        if metadata.get(key) != expected
    ]
    if projection_dim <= 0:
        errors.append(f"relation_token_projection_dim must be positive, got {projection_dim}")
    if errors:
        raise ValueError("Invalid NDF relation-token v2 metadata: " + "; ".join(errors))
    print(f"Validated NDF relation-token v2 metadata: {path}")


if __name__ == "__main__":
    main()
