from __future__ import annotations

from pathlib import Path
from typing import Any
import logging

from faster_whisper import WhisperModel

log = logging.getLogger("localscribe.transcription")


class Transcriber:
    """Faster-Whisper adapter with explicit offline/runtime behavior."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        runtime = cfg["runtime"]
        model_path = Path(runtime["whisper_model_path"])

        if model_path.exists() and any(model_path.iterdir()):
            model_ref = str(model_path)
            local_files_only = True
        elif runtime["allow_model_download"]:
            model_ref = runtime["whisper_model"]
            local_files_only = False
        else:
            raise RuntimeError(
                f"Whisper model not found at {model_path}. "
                "Run `docker compose --profile setup run --rm bootstrap-models` first."
            )

        log.info("Loading Faster-Whisper model from %s", model_ref)
        self.model = WhisperModel(
            model_ref,
            device=runtime["device"],
            compute_type=runtime["whisper_compute_type"],
            local_files_only=local_files_only,
        )

    def transcribe(self, audio_path: Path) -> tuple[list[Any], dict, list[dict]]:
        trans_cfg = self.cfg["transcription"]
        runtime = self.cfg["runtime"]

        segments_iter, info = self.model.transcribe(
            str(audio_path),
            language=runtime["language"],
            beam_size=int(trans_cfg.get("beam_size", 5)),
            vad_filter=bool(trans_cfg.get("vad_filter", True)),
            word_timestamps=bool(trans_cfg.get("word_timestamps", True)),
            log_progress=True,
        )
        segments = list(segments_iter)

        raw_segments: list[dict] = []
        for segment in segments:
            item = {
                "id": int(segment.id),
                "start": float(segment.start),
                "end": float(segment.end),
                "text": segment.text,
            }
            if segment.words:
                item["words"] = [
                    {
                        "start": float(word.start) if word.start is not None else None,
                        "end": float(word.end) if word.end is not None else None,
                        "word": word.word,
                        "probability": float(word.probability),
                    }
                    for word in segment.words
                ]
            raw_segments.append(item)

        metadata = {
            "language": getattr(info, "language", runtime["language"]),
            "language_probability": float(getattr(info, "language_probability", 0.0) or 0.0),
            "duration": float(getattr(info, "duration", 0.0) or 0.0),
        }
        return segments, metadata, raw_segments
