import unittest

from script.select_gso_shoe_assets import (
    build_manifest,
    collect_unique_candidates,
    metadata_rejection,
    normalized_family,
)


def row(name: str, description: str, categories=None) -> dict:
    return {
        "name": name,
        "owner": "GoogleResearch",
        "description": description,
        "filesize": 1234,
        "license_name": "Creative Commons Attribution 4.0 International",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "private": False,
        "categories": ["Shoe"] if categories is None else categories,
        "thumbnail_url": "/thumbnail.jpg",
    }


class SelectGsoShoeAssetsTest(unittest.TestCase):
    def test_scope_filter(self) -> None:
        self.assertIsNone(metadata_rejection(row("low_shoe", "A low shoe")))
        self.assertEqual(
            metadata_rejection(row("winter_boot", "A winter boot")),
            "outside_low_shoe_scope",
        )
        self.assertEqual(
            metadata_rejection(row("court_mid", "A court mid shoe")),
            "outside_low_shoe_scope",
        )
        self.assertEqual(
            metadata_rejection(row("high_top", "A high top shoe")),
            "outside_low_shoe_scope",
        )
        self.assertIsNone(metadata_rejection(row("midnight_blue", "A midnight blue low shoe")))
        self.assertEqual(
            metadata_rejection(row("mug", "A mug", categories=["Consumer Goods"])),
            "not_shoe_category",
        )

    def test_family_normalization_and_deduplication(self) -> None:
        first = row("shoe_scan_a", "Women's Boat-Shoe\nFirst scan")
        second = row("shoe_scan_b", "Women's Boat Shoe\nSecond scan")
        self.assertEqual(normalized_family(first), normalized_family(second))
        candidates, audit = collect_unique_candidates([first, second])
        self.assertEqual(len(candidates), 1)
        self.assertEqual(audit["deduplicated_scans"], 1)

    def test_manifest_is_order_invariant(self) -> None:
        rows = [row(f"shoe_{index}", f"Distinct low shoe {index}") for index in range(8)]
        candidates, _ = collect_unique_candidates(rows)
        first = build_manifest(candidates, dev_count=2, blind_count=3, reserve_count=2)
        second = build_manifest(list(reversed(candidates)), dev_count=2, blind_count=3, reserve_count=2)
        self.assertEqual(
            [asset["name"] for asset in first["assets"]],
            [asset["name"] for asset in second["assets"]],
        )
        self.assertEqual(
            [asset["partition"] for asset in first["assets"]],
            [
                "development",
                "development",
                "blind_test",
                "blind_test",
                "blind_test",
                "reserve",
                "reserve",
            ],
        )


if __name__ == "__main__":
    unittest.main()
