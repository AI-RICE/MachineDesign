import json
import os

default_config_path = os.path.join(os.path.dirname(__file__), "config.default.json")
local_config_path = os.path.join(os.path.expanduser("~"), ".machine_design", "config.json")


def load_config(local_path=local_config_path):

    with open(default_config_path) as f:
        config = json.load(f)

    if os.path.exists(local_path):
        with open(local_path) as f:
            config.update(json.load(f))

    return config
