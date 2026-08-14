from __future__ import annotations

from pathlib import Path
import os
import sys

from faster_whisper.utils import download_model
from huggingface_hub import snapshot_download

MODELS = Path("/app/models")
WHISPER_DIR = MODELS / "whisper"
PYANNOTE_DIR = MODELS / "pyannote-community-1"


def main() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)

    whisper_name = os.getenv("WHISPER_MODEL", "large-v3")
    print(f"Downloading Faster-Whisper model '{whisper_name}' to {WHISPER_DIR}...")
    WHISPER_DIR.mkdir(parents=True, exist_ok=True)
    download_model(whisper_name, output_dir=str(WHISPER_DIR))

    token = os.getenv("HF_TOKEN")
    if not token:
        print(
            "HF_TOKEN is required only for the initial pyannote Community-1 download.\n"
            "Accept the model conditions on Hugging Face, set HF_TOKEN in .env, and rerun bootstrap.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    print(f"Downloading pyannote Community-1 to {PYANNOTE_DIR}...")
    PYANNOTE_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="pyannote/speaker-diarization-community-1",
        token=token,
        local_dir=str(PYANNOTE_DIR),
    )

    print("Models are ready. Normal LocalScribe runs can now be offline.")


if __name__ == "__main__":
    main()
