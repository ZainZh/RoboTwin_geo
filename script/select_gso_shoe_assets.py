"""Create a frozen Google Scanned Objects low-shoe benchmark manifest.

Only public Fuel metadata is used for selection.  Product families are
deduplicated by the normalized first description line, then ordered by a fixed
SHA-256 salt before meshes are downloaded or policy rollouts are observed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import quote


FUEL_ROOT = "https://fuel.gazebosim.org/1.0"
OWNER = "GoogleResearch"
SELECTION_SALT = "RoboTwin_GSO_low_shoe_blind_v2"
EXCLUDED_LOW_SHOE_TERMS = (
    "BOOT",
    "BOOTIE",
    "CLOG",
    "FLIP FLOP",
    "FLIPFLOP",
    "HIGH",
    "MID",
    "SANDAL",
    "SLIDE",
    "THONG",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dev-count", type=int, default=4)
    parser.add_argument("--blind-count", type=int, default=16)
    parser.add_argument("--reserve-count", type=int, default=8)
    return parser.parse_args()


def load_metadata_pages(metadata_dir: Path) -> list[dict]:
    paths = list(metadata_dir.glob("models_page*.json"))
    if not paths:
        raise FileNotFoundError(f"no models_page*.json under {metadata_dir}")

    def page_number(path: Path) -> int:
        match = re.search(r"models_page(\d+)$", path.stem)
        return int(match.group(1)) if match else 0

    rows = []
    for path in sorted(paths, key=page_number):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"expected a JSON list in {path}")
        rows.extend(payload)
    return rows


def first_description_line(row: dict) -> str:
    description = str(row.get("description", "")).strip()
    return description.splitlines()[0].strip() if description else str(row.get("name", ""))


def normalized_family(row: dict) -> str:
    text = first_description_line(row).upper()
    return re.sub(r"[^A-Z0-9]+", " ", text).strip()


def metadata_rejection(row: dict) -> str | None:
    if str(row.get("owner", "")) != OWNER:
        return "wrong_owner"
    if bool(row.get("private", False)):
        return "private"
    if "Shoe" not in row.get("categories", []):
        return "not_shoe_category"
    if "Creative Commons Attribution 4.0" not in str(row.get("license_name", "")):
        return "unexpected_license"
    text = re.sub(
        r"[^A-Z0-9]+",
        " ",
        f"{row.get('name', '')} {first_description_line(row)}".upper(),
    ).strip()
    padded_text = f" {text} "
    if any(f" {term} " in padded_text for term in EXCLUDED_LOW_SHOE_TERMS):
        return "outside_low_shoe_scope"
    if int(row.get("filesize", 0)) <= 0:
        return "empty_archive"
    return None


def _hash(value: str) -> str:
    return hashlib.sha256(f"{SELECTION_SALT}:{value}".encode("utf-8")).hexdigest()


def collect_unique_candidates(rows: Iterable[dict]) -> tuple[list[dict], dict]:
    rejection_counts: Counter[str] = Counter()
    accepted_rows = []
    for row in rows:
        rejection = metadata_rejection(row)
        if rejection is not None:
            rejection_counts[rejection] += 1
            continue
        accepted_rows.append(row)

    # Identical products can appear under several scan identifiers.  Choosing
    # the minimum fixed hash is independent of file order and policy behavior.
    by_family: dict[str, list[dict]] = {}
    for row in accepted_rows:
        by_family.setdefault(normalized_family(row), []).append(row)

    candidates = []
    duplicate_scans = 0
    for family, family_rows in by_family.items():
        ordered_family = sorted(family_rows, key=lambda row: _hash(str(row["name"])))
        row = ordered_family[0]
        duplicate_scans += len(family_rows) - 1
        name = str(row["name"])
        candidates.append(
            {
                "name": name,
                "family": family,
                "description": str(row.get("description", "")),
                "categories": list(row.get("categories", [])),
                "archive_size_bytes": int(row["filesize"]),
                "license": str(row["license_name"]),
                "license_url": str(row["license_url"]),
                "download_url": f"{FUEL_ROOT}/{OWNER}/models/{quote(name, safe='')}.zip",
                "thumbnail_url": f"https://fuel.gazebosim.org{row.get('thumbnail_url', '')}",
            }
        )
    audit = {
        "metadata_rows": sum(rejection_counts.values()) + len(accepted_rows),
        "accepted_low_shoe_rows": len(accepted_rows),
        "unique_product_families": len(candidates),
        "deduplicated_scans": duplicate_scans,
        "rejections": dict(sorted(rejection_counts.items())),
    }
    return candidates, audit


def build_manifest(
    candidates: list[dict], *, dev_count: int, blind_count: int, reserve_count: int
) -> dict:
    if min(dev_count, blind_count, reserve_count) < 0:
        raise ValueError("partition counts must be non-negative")
    required = dev_count + blind_count + reserve_count
    ordered = sorted(candidates, key=lambda row: _hash(row["family"]))
    if len(ordered) < required:
        raise ValueError(f"need {required} unique product families, found {len(ordered)}")
    selected = []
    for rank, source in enumerate(ordered[:required]):
        row = dict(source)
        if rank < dev_count:
            partition = "development"
        elif rank < dev_count + blind_count:
            partition = "blind_test"
        else:
            partition = "reserve"
        row.update(
            {
                "partition": partition,
                "selection_rank": rank,
                "selection_hash": _hash(row["family"]),
                "asset_gate_status": "pending",
            }
        )
        selected.append(row)
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "dataset": "Google Scanned Objects",
            "api": FUEL_ROOT,
            "owner": OWNER,
            "license": "Creative Commons Attribution 4.0 International",
        },
        "selection": {
            "salt": SELECTION_SALT,
            "ordering": "ascending SHA-256(salt + ':' + normalized product family)",
            "dev_count": dev_count,
            "blind_count": blind_count,
            "reserve_count": reserve_count,
            "rules": {
                "required_category": "Shoe",
                "required_owner": OWNER,
                "required_license": "CC BY 4.0",
                "excluded_low_shoe_terms": list(EXCLUDED_LOW_SHOE_TERMS),
                "deduplication": "normalized first description line",
                "policy_outcomes_used": False,
            },
        },
        "assets": selected,
    }


def main() -> None:
    args = parse_args()
    candidates, audit = collect_unique_candidates(load_metadata_pages(args.metadata_dir))
    manifest = build_manifest(
        candidates,
        dev_count=args.dev_count,
        blind_count=args.blind_count,
        reserve_count=args.reserve_count,
    )
    manifest["audit"] = audit
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "audit": audit,
                "partitions": dict(Counter(row["partition"] for row in manifest["assets"])),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
