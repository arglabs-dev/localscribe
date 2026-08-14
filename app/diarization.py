from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging

import torch
from pyannote.audio import Pipeline

log = logging.getLogger("localscribe.diarization")


@dataclass(frozen=True)
class SpeakerInterval:
    start: float
    end: float
    speaker_id: str


def intervals_from_output(output) -> list[SpeakerInterval]:
    annotation = getattr(output, "exclusive_speaker_diarization", None)
    if annotation is None:
        annotation = getattr(output, "speaker_diarization", output)

    intervals: list[SpeakerInterval] = []
    if hasattr(annotation, "itertracks"):
        for segment, _, label in annotation.itertracks(yield_label=True):
            intervals.append(SpeakerInterval(float(segment.start), float(segment.end), str(label)))
    else:
        for item in annotation:
            if len(item) == 2:
                segment, label = item
            else:
                segment, _, label = item
            intervals.append(SpeakerInterval(float(segment.start), float(segment.end), str(label)))

    return sorted(intervals, key=lambda item: (item.start, item.end, item.speaker_id))


class Diarizer:
    """Local pyannote diarization adapter."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        runtime = cfg["runtime"]
        model_path = Path(runtime["pyannote_model_path"])
        if not model_path.exists() or not any(model_path.iterdir()):
            raise RuntimeError(
                f"pyannote model not found at {model_path}. "
                "Run `docker compose --profile setup run --rm bootstrap-models` first."
            )

        log.info("Loading pyannote pipeline from %s", model_path)
        self.pipeline = Pipeline.from_pretrained(str(model_path))

        if runtime["device"].startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("DEVICE=cuda was requested but CUDA is not available")
            self.pipeline.to(torch.device("cuda"))

    def diarize(self, audio_path: Path) -> list[SpeakerInterval]:
        diar_cfg = self.cfg["diarization"]
        kwargs: dict[str, int] = {}
        if diar_cfg.get("min_speakers") is not None:
            kwargs["min_speakers"] = int(diar_cfg["min_speakers"])
        if diar_cfg.get("max_speakers") is not None:
            kwargs["max_speakers"] = int(diar_cfg["max_speakers"])

        output = self.pipeline(str(audio_path), **kwargs)
        return intervals_from_output(output)
