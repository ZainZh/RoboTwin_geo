import json

from description.utils.generate_episode_instructions import load_scene_info


def test_load_scene_info_prefers_resolved_raw_data_root(tmp_path, monkeypatch):
    task_name = "path_test_task"
    setting = "path_test_setting"
    scene_path = tmp_path / task_name / setting / "scene_info.json"
    scene_path.parent.mkdir(parents=True)
    scene_path.write_text(json.dumps({"episode_0": {"info": {}}}))
    monkeypatch.setenv("ROBOTWIN_RAW_DATA_ROOT", str(tmp_path))

    scene_info = load_scene_info(task_name, setting, "./data")

    assert "episode_0" in scene_info
