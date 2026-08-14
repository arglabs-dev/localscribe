from __future__ import annotations

from pathlib import Path
import os

import yaml

BASE_DIR = Path("/app")
DATA_DIR = BASE_DIR / "data"
CONFIG_PATH = Path(os.getenv("LOCALSCRIBE_CONFIG", "/app/config/config.yaml"))


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config.setdefault("watcher", {})
    config.setdefault("transcription", {})
    config.setdefault("diarization", {})
    config.setdefault("session", {})
    config.setdefault("cleanup", {})
    config.setdefault("output", {})
    config["runtime"] = {
        "device": os.getenv("DEVICE", "cpu"),
        "language": os.getenv("LANGUAGE", config["transcription"].get("language", "es")),
        "whisper_compute_type": os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
        "whisper_model": os.getenv("WHISPER_MODEL", "large-v3"),
        "whisper_model_path": os.getenv("WHISPER_MODEL_PATH", "/app/models/whisper"),
        "pyannote_model_path": os.getenv("PYANNOTE_MODEL_PATH", "/app/models/pyannote-community-1"),
        "hf_token": os.getenv("HF_TOKEN") or None,
        "allow_model_download": os.getenv("ALLOW_MODEL_DOWNLOAD", "0") == "1",
    }
    return config


def paths() -> dict[str, Path]:
    directories = {
        "input": DATA_DIR / "input",
        "processing": DATA_DIR / "processing",
        "output": DATA_DIR / "output",
        "completed": DATA_DIR / "completed",
        "failed": DATA_DIR / "failed",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories
