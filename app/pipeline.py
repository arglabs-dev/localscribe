from __future__ import annotations

from pathlib import Path
import logging

from .align import build_turns
from .diarization import Diarizer
from .metadata import extract_metadata
from .participants import map_participants
from .render import apply_names, write_outputs
from .text_utils import clean_fillers
from .transcription import Transcriber

log = logging.getLogger("localscribe.pipeline")


class LocalScribePipeline:
    """Orchestrates the deterministic local processing stages for one audio file."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.transcriber = Transcriber(cfg)
        self.diarizer = Diarizer(cfg)

    def process(self, audio_path: Path, out_dir: Path) -> dict:
        log.info("Transcribing %s", audio_path.name)
        segments, transcription_info, raw_whisper = self.transcriber.transcribe(audio_path)

        log.info("Diarizing %s", audio_path.name)
        intervals = self.diarizer.diarize(audio_path)

        cleanup_cfg = self.cfg["cleanup"]

        def clean(text: str) -> str:
            if not cleanup_cfg.get("enabled", True):
                return text.strip()
            return clean_fillers(
                text,
                cleanup_cfg.get("fillers", []),
                bool(cleanup_cfg.get("collapse_repetitions", True)),
            )

        turns = build_turns(segments, intervals, clean)
        intro_window = float(self.cfg["session"].get("intro_window_seconds", 180))
        participant_map = map_participants(turns, intro_window)
        turns = apply_names(turns, participant_map)

        runtime = self.cfg["runtime"]
        metadata = extract_metadata(
            audio_path=audio_path,
            turns=turns,
            participant_map=participant_map,
            intro_window_seconds=intro_window,
            timezone_name=self.cfg["session"].get("fallback_timezone", "America/Mexico_City"),
            language=transcription_info["language"],
            whisper_model=runtime["whisper_model"],
            diarization_model="pyannote/speaker-diarization-community-1",
        )

        write_outputs(out_dir, metadata, turns, raw_whisper, self.cfg["output"])
        return metadata
