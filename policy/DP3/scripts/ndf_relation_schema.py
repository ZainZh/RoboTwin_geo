from collections.abc import Mapping


RELATION_V2_SCHEMA = {
    "relation_schema_version": 2,
    "relation_xyz_frame": "world",
    "relation_query_frame": "support_normalized",
}


def relation_v2_metadata() -> dict:
    return dict(RELATION_V2_SCHEMA)


def validate_relation_v2_metadata(metadata: Mapping) -> None:
    mismatches = []
    for key, expected in RELATION_V2_SCHEMA.items():
        actual = metadata.get(key)
        if actual != expected:
            mismatches.append(f"{key}={actual!r}, expected {expected!r}")
    if mismatches:
        raise ValueError("Invalid NDF relation-v2 metadata: " + "; ".join(mismatches))
