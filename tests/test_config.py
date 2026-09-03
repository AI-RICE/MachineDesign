import json

from machine_design.config import load_config


def test_load_config_defaults(tmp_path):
    config = load_config(local_path=tmp_path / "missing.json")
    assert config == {
        "aedt_version": "2024.1",
        "num_cores": 4,
        "n_workers": 22,
        "project_dir": "data",
    }


def test_load_config_merges_local_override(tmp_path):
    local_path = tmp_path / "config.json"
    local_path.write_text(json.dumps({"aedt_version": "2025.1"}))

    config = load_config(local_path=local_path)

    assert config["aedt_version"] == "2025.1"
    assert config["num_cores"] == 4
