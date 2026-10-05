"""Replaceable file locations; relative paths are based on the JSON file."""

from dataclasses import dataclass
import json
import os
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    dataset_csv: str
    trajectories_file: str
    rollouts_jsonl: str
    checkpoint_dir: str
    final_model_dir: str
    export_copy_dir: str | None
    probe_input: str
    probe_output: str
    verifier_data_dir: str
    resume_from_checkpoint: str | None


def load_paths(filename=None):
    config_file = Path(filename).expanduser().resolve() if filename else Path(__file__).with_name("paths.json")
    values = json.loads(config_file.read_text(encoding="utf-8"))
    resolved = {}
    for key, value in values.items():
        if value is None:
            resolved[key] = None
            continue
        path = Path(os.path.expandvars(value)).expanduser()
        if not path.is_absolute():
            path = config_file.parent / path
        resolved[key] = str(path.resolve())
    return Paths(**resolved)


def prepare_outputs(paths):
    for filename in (paths.trajectories_file, paths.rollouts_jsonl, paths.probe_output):
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
    for directory in (paths.checkpoint_dir, paths.final_model_dir):
        Path(directory).mkdir(parents=True, exist_ok=True)


def mount_colab():
    from google.colab import drive
    drive.mount("/content/drive")


def disconnect_colab():
    from google.colab import runtime
    runtime.unassign()
