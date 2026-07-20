from pathlib import Path


def test_collect_data_uses_script_local_data_path_import():
    source = (Path(__file__).resolve().parent / "collect_data.py").read_text()

    assert "from data_path_utils import resolve_data_paths" in source
    assert "from script.data_path_utils import resolve_data_paths" not in source
