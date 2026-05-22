from ast import FunctionDef, Module, fix_missing_locations, parse
from pathlib import Path
import types
import unittest


EVAL_POLICY_SOURCE_PATH = Path("script/eval_policy.py")


def load_helpers(*function_names):
    source = EVAL_POLICY_SOURCE_PATH.read_text(encoding="utf-8")
    tree = parse(source)
    func_nodes = [
        node for node in tree.body if isinstance(node, FunctionDef) and node.name in set(function_names)
    ]
    module = Module(body=func_nodes, type_ignores=[])
    fix_missing_locations(module)
    namespace = {}
    exec(compile(module, filename=str(EVAL_POLICY_SOURCE_PATH), mode="exec"), namespace)
    return [namespace[name] for name in function_names]


class _DummyBlock:
    def __init__(self, x_value):
        self._x_value = x_value

    def get_functional_point(self, idx, ret):
        return types.SimpleNamespace(p=[self._x_value, 0.0, 0.0])


class _DummyTaskEnv:
    def __init__(self, hammer_path, x_value):
        self.hammer_asset_config = {"info_asset_path": hammer_path}
        self.block = _DummyBlock(x_value)
        self.info = {
            "cluttered_table_info": [],
            "texture_info": {"wall_texture": None, "table_texture": None},
        }


class TestEvalPolicyHelpers(unittest.TestCase):
    def test_should_run_expert_check_skips_custom_hammer_eval(self):
        should_run_expert_check, = load_helpers("should_run_expert_check")

        self.assertTrue(should_run_expert_check({}))
        self.assertTrue(should_run_expert_check({"custom_hammer_eval": {"enabled": False}}))
        self.assertFalse(should_run_expert_check({"custom_hammer_eval": {"enabled": True}}))
        self.assertFalse(should_run_expert_check({"custom_mug_eval": {"enabled": True}}))

    def test_build_instruction_episode_info_for_beat_block_hammer_without_play_once(self):
        _, build_instruction_episode_info = load_helpers(
            "should_run_expert_check",
            "build_instruction_episode_info",
        )

        env = _DummyTaskEnv("partnext_hammer_eval/base7", x_value=-0.12)

        episode_info = build_instruction_episode_info("beat_block_hammer", env, episode_info=None)

        self.assertEqual(
            episode_info,
            {"info": {"{A}": "partnext_hammer_eval/base7", "{a}": "left"}},
        )

    def test_build_instruction_episode_info_ignores_empty_existing_info(self):
        _, build_instruction_episode_info = load_helpers(
            "should_run_expert_check",
            "build_instruction_episode_info",
        )

        env = _DummyTaskEnv("partnext_hammer_eval/base7", x_value=0.12)
        env.info["info"] = {}

        episode_info = build_instruction_episode_info("beat_block_hammer", env, episode_info=None)

        self.assertEqual(
            episode_info,
            {"info": {"{A}": "partnext_hammer_eval/base7", "{a}": "right"}},
        )

    def test_build_instruction_episode_info_prefers_existing_episode_info(self):
        _, build_instruction_episode_info = load_helpers(
            "should_run_expert_check",
            "build_instruction_episode_info",
        )

        env = _DummyTaskEnv("partnext_hammer_eval/base7", x_value=0.12)
        existing = {"info": {"{A}": "020_hammer/base0", "{a}": "right"}}

        episode_info = build_instruction_episode_info("beat_block_hammer", env, episode_info=existing)

        self.assertIs(episode_info, existing)

    def test_build_instruction_episode_info_for_hanging_mug_without_play_once(self):
        _, build_instruction_episode_info = load_helpers(
            "should_run_expert_check",
            "build_instruction_episode_info",
        )

        env = types.SimpleNamespace(
            mug_asset_config={"info_asset_path": "partnext_mug_eval/base3"},
            info={
                "cluttered_table_info": [],
                "texture_info": {"wall_texture": None, "table_texture": None},
            },
        )

        episode_info = build_instruction_episode_info("hanging_mug", env, episode_info=None)

        self.assertEqual(
            episode_info,
            {"info": {"{A}": "partnext_mug_eval/base3", "{B}": "040_rack/base0"}},
        )

    def test_resolve_episode_instruction_falls_back_for_dp3_when_template_missing(self):
        resolve_episode_instruction, = load_helpers("resolve_episode_instruction")

        def missing_template_generator(*_args, **_kwargs):
            raise FileNotFoundError("missing task instruction template")

        instruction = resolve_episode_instruction(
            task_name="open_microwave",
            episode_info_list=[{"{A}": "044_microwave/base0", "{a}": "left"}],
            instruction_type="unseen",
            test_num=100,
            policy_name="DP3",
            description_generator=missing_template_generator,
        )

        self.assertEqual(instruction, "open microwave")

    def test_resolve_episode_instruction_keeps_missing_template_fatal_for_non_dp3(self):
        resolve_episode_instruction, = load_helpers("resolve_episode_instruction")

        def missing_template_generator(*_args, **_kwargs):
            raise FileNotFoundError("missing task instruction template")

        with self.assertRaises(FileNotFoundError):
            resolve_episode_instruction(
                task_name="open_microwave",
                episode_info_list=[{"{A}": "044_microwave/base0", "{a}": "left"}],
                instruction_type="unseen",
                test_num=100,
                policy_name="SomeLanguagePolicy",
                description_generator=missing_template_generator,
            )


if __name__ == "__main__":
    unittest.main()
