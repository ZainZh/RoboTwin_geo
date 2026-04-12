from ast import FunctionDef, Module, fix_missing_locations, parse
from pathlib import Path


def load_resolver():
    source = Path("envs/beat_block_hammer.py").read_text()
    tree = parse(source)
    func_node = next(node for node in tree.body if isinstance(node, FunctionDef) and node.name == "resolve_hammer_asset_config")
    module = Module(body=[func_node], type_ignores=[])
    fix_missing_locations(module)
    namespace = {}
    exec(compile(module, filename="envs/beat_block_hammer.py", mode="exec"), namespace)
    return namespace["resolve_hammer_asset_config"]


def test_resolve_hammer_asset_config_defaults():
    resolve_hammer_asset_config = load_resolver()
    assert resolve_hammer_asset_config(None) == {
        "modelname": "020_hammer",
        "model_id": 0,
        "info_asset_path": "020_hammer/base0",
    }


def test_resolve_hammer_asset_config_custom_override():
    resolve_hammer_asset_config = load_resolver()
    assert resolve_hammer_asset_config(
        {
            "enabled": True,
            "modelname": "partnext_hammer_eval",
            "model_id": 0,
        }
    ) == {
        "modelname": "partnext_hammer_eval",
        "model_id": 0,
        "info_asset_path": "partnext_hammer_eval/base0",
    }
