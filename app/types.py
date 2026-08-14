from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpeakerInterval:
    start: float
    end: float
    speaker_id: str
