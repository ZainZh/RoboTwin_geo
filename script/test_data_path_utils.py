from pathlib import Path

from script.data_path_utils import resolve_data_paths


def test_uses_project_paths_when_server_storage_is_missing(tmp_path):
    repo_root = tmp_path / "repo"
    server_root = tmp_path / "missing-shared2"

    paths = resolve_data_paths(
        repo_root=repo_root,
        server_storage_root=server_root,
        environ={},
    )

    assert paths.mode == "local"
    assert paths.raw_data_root == repo_root / "data"
    assert paths.dp3_data_root == repo_root / "policy" / "DP3" / "data"


def test_uses_shared2_paths_when_server_storage_exists(tmp_path):
    repo_root = tmp_path / "repo"
    server_root = tmp_path / "shared2" / "sz"
    server_root.mkdir(parents=True)

    paths = resolve_data_paths(
        repo_root=repo_root,
        server_storage_root=server_root,
        environ={},
    )

    assert paths.mode == "server"
    assert paths.raw_data_root == server_root / "RoboTwin_geo" / "data"
    assert paths.dp3_data_root == server_root / "RoboTwin_geo" / "policy" / "DP3" / "data"


def test_environment_can_override_resolved_data_roots(tmp_path):
    repo_root = tmp_path / "repo"
    raw_root = tmp_path / "custom-raw"
    dp3_root = tmp_path / "custom-dp3"

    paths = resolve_data_paths(
        repo_root=repo_root,
        server_storage_root=tmp_path / "missing-shared2",
        environ={
            "ROBOTWIN_RAW_DATA_ROOT": str(raw_root),
            "ROBOTWIN_DP3_DATA_ROOT": str(dp3_root),
        },
    )

    assert paths.raw_data_root == Path(raw_root)
    assert paths.dp3_data_root == Path(dp3_root)
