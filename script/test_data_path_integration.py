from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DP3_ROOT = REPO_ROOT / "policy" / "DP3"

PROCESSORS = (
    "process_data.py",
    "process_data_ndf.py",
    "process_data_ndf_pointwise.py",
    "process_data_objpc.py",
    "process_data_objpc_actorseg.py",
    "process_data_semantic_pointwise.py",
    "process_data_semantic_pointwise_actorseg_hybrid.py",
    "process_data_ndf_pointwise_actorseg_hybrid.py",
    "process_data_se3_relation_hybrid.py",
    "process_data_shoe_se3_placement_comparison.py",
    "process_data_utonia_pointwise.py",
)


def test_collection_and_instruction_generation_use_resolved_raw_root():
    collect_source = (REPO_ROOT / "script" / "collect_data.py").read_text()
    instruction_source = (
        REPO_ROOT / "description" / "utils" / "generate_episode_instructions.py"
    ).read_text()

    assert "resolve_data_paths" in collect_source
    assert "os.environ.get" in instruction_source
    assert "ROBOTWIN_RAW_DATA_ROOT" in instruction_source


def test_dp3_processors_use_resolved_raw_and_output_roots():
    for name in PROCESSORS:
        source = (DP3_ROOT / "scripts" / name).read_text()
        assert "raw_task_data_dir" in source, name
        assert "dp3_data_path" in source, name
        assert '"../../data"' not in source, name
        assert '"./data/' not in source, name


def test_dp3_training_entrypoints_use_absolute_resolved_zarr_paths():
    checked = []
    for path in sorted(DP3_ROOT.glob("train*.sh")):
        source = path.read_text()
        if ".zarr" not in source:
            continue
        checked.append(path.name)
        assert "data_paths.sh" in source, path.name
        assert "ROBOTWIN_DP3_DATA_ROOT" in source, path.name
        assert '"./data/' not in source, path.name
        assert "../../../data/" not in source, path.name
    assert checked
