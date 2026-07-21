from script.data_path_utils import resolve_data_paths


def test_existing_shared2_without_robotwin_data_falls_back_to_project_data(tmp_path):
    repo_root = tmp_path / "repo"
    server_root = tmp_path / "shared2" / "sz"
    server_root.mkdir(parents=True)

    paths = resolve_data_paths(
        repo_root=repo_root,
        server_storage_root=server_root,
        environ={},
    )

    assert paths.mode == "local"
    assert paths.raw_data_root == repo_root / "data"
    assert paths.dp3_data_root == repo_root / "policy" / "DP3" / "data"
