from ast import ClassDef, FunctionDef, Module, fix_missing_locations, parse
import json
from pathlib import Path
import unittest

import yaml


ENV_SOURCE_PATH = Path("envs/pour_kettle_mug.py")
INSTRUCTION_PATH = Path("description/task_instruction/pour_kettle_mug.json")
POINTCLOUD_TARGETS_PATH = Path("envs/object_pointcloud_targets.py")
EVAL_STEP_LIMIT_PATH = Path("task_config/_eval_step_limit.yml")


def load_env_class_names():
    source = ENV_SOURCE_PATH.read_text(encoding="utf-8")
    tree = parse(source)
    return [node.name for node in tree.body if isinstance(node, ClassDef)]


class TestPourKettleMugTaskIntegration(unittest.TestCase):
    def test_env_source_exists_and_defines_expected_task_class(self):
        self.assertTrue(ENV_SOURCE_PATH.exists(), f"Missing task source: {ENV_SOURCE_PATH}")
        self.assertIn("pour_kettle_mug", load_env_class_names())

    def test_env_source_mentions_expected_assets_and_left_arm_usage(self):
        source = ENV_SOURCE_PATH.read_text(encoding="utf-8")

        self.assertIn('modelname="009_kettle"', source)
        self.assertIn('modelname="039_mug"', source)
        self.assertIn('ArmTag("left")', source)
        self.assertIn('"{A}"', source)
        self.assertIn('"{B}"', source)
        self.assertIn('"{a}"', source)

    def test_env_source_keeps_mug_static_and_upright(self):
        source = ENV_SOURCE_PATH.read_text(encoding="utf-8")

        self.assertIn("is_static=True", source)
        self.assertIn("qpos=[1, 0, 0, 0]", source)

    def test_env_source_limits_kettle_instances_to_reachable_subset(self):
        source = ENV_SOURCE_PATH.read_text(encoding="utf-8")

        self.assertIn("REACHABLE_KETTLE_IDS = [2]", source)

    def test_instruction_template_exists_and_uses_expected_placeholders(self):
        self.assertTrue(INSTRUCTION_PATH.exists(), f"Missing instruction template: {INSTRUCTION_PATH}")
        payload = json.loads(INSTRUCTION_PATH.read_text(encoding="utf-8"))

        self.assertIn("seen", payload)
        self.assertIn("unseen", payload)
        self.assertTrue(any("{A}" in entry for entry in payload["seen"]))
        self.assertTrue(any("{B}" in entry for entry in payload["seen"]))
        self.assertTrue(any("{a}" in entry for entry in payload["seen"]))

    def test_pointcloud_target_registry_contains_task_mapping(self):
        source = POINTCLOUD_TARGETS_PATH.read_text(encoding="utf-8")

        self.assertIn('"pour_kettle_mug"', source)
        self.assertIn('"{A}": "kettle"', source)
        self.assertIn('"{B}": "mug"', source)

    def test_eval_step_limit_contains_new_task(self):
        payload = yaml.safe_load(EVAL_STEP_LIMIT_PATH.read_text(encoding="utf-8"))

        self.assertIn("pour_kettle_mug", payload)
        self.assertIsInstance(payload["pour_kettle_mug"], int)


if __name__ == "__main__":
    unittest.main()
