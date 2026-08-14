from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import re

import dateparser
from dateparser.search import search_dates

SESSION_RE = re.compile(
    r"\b(?:sesión|reunión)\s+(?:de|del|para)\s+(.+?)(?=,|\.|;|\byo\s+soy\b|\bmi\s+nombre\b|\bme\s+llamo\b|$)",
    re.IGNORECASE,
)


def _file_datetime(path: Path, timezone_name: str) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=ZoneInfo(timezone_name))


def _spoken_datetime(intro_text: str, fallback: datetime) -> datetime | None:
    date_match = re.search(r"(?i)\b(?:hoy\s+es|fecha|día)\b(.{0,100})", intro_text)
    time_match = re.search(r"(?i)\b(?:son\s+las|a\s+las)\b(.{0,40})", intro_text)
    candidates = [match.group(0) for match in (date_match, time_match) if match]
    if not candidates:
        return None

    phrase = ". ".join(candidates)
    settings = {
        "RELATIVE_BASE": fallback.replace(tzinfo=None),
        "RETURN_AS_TIMEZONE_AWARE": False,
        "PREFER_DATES_FROM": "current_period",
    }
    hits = search_dates(phrase, languages=["es"], settings=settings)
    parsed = hits[-1][1] if hits else dateparser.parse(phrase, languages=["es"], settings=settings)
    return parsed.replace(tzinfo=fallback.tzinfo) if parsed else None


def extract_metadata(
    audio_path: Path,
    turns: list[dict],
    participant_map: dict[str, dict],
    intro_window_seconds: float,
    timezone_name: str,
    language: str,
    whisper_model: str,
    diarization_model: str,
) -> dict:
    fallback_dt = _file_datetime(audio_path, timezone_name)
    intro_text = " ".join(turn["raw_text"] for turn in turns if turn["start"] <= intro_window_seconds)

    spoken_dt = _spoken_datetime(intro_text, fallback_dt)
    effective_dt = spoken_dt or fallback_dt

    session_match = SESSION_RE.search(intro_text)
    session_title = session_match.group(1).strip() if session_match else None
    duration = max((turn["end"] for turn in turns), default=0.0)

    participants: list[dict] = []
    seen: set[str] = set()
    for turn in turns:
        speaker_id = turn["speaker_id"]
        if speaker_id in seen:
            continue
        seen.add(speaker_id)
        resolution = participant_map.get(speaker_id)
        participants.append(
            {
                "speaker_id": speaker_id,
                "name": resolution["name"] if resolution else None,
                "label": resolution["name"] if resolution else speaker_id,
                "name_source": resolution["source"] if resolution else "unresolved",
                "confidence": resolution.get("confidence") if resolution else None,
            }
        )

    return {
        "source_audio": audio_path.name,
        "file_timestamp": fallback_dt.isoformat(),
        "session_datetime": effective_dt.isoformat(),
        "datetime_source": "spoken_header" if spoken_dt else "file_mtime",
        "session_title": session_title,
        "participants": participants,
        "language": language,
        "duration_seconds": round(duration, 3),
        "models": {
            "transcription": whisper_model,
            "diarization": diarization_model,
        },
    }
