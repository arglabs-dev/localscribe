from __future__ import annotations

from collections.abc import Callable, Iterable

from .types import SpeakerInterval


def speaker_for_range(start: float, end: float, intervals: list[SpeakerInterval]) -> str:
    if not intervals:
        return "SPEAKER_UNKNOWN"

    midpoint = (start + end) / 2.0
    for interval in intervals:
        if interval.start <= midpoint <= interval.end:
            return interval.speaker_id

    best_speaker = "SPEAKER_UNKNOWN"
    best_overlap = 0.0
    for interval in intervals:
        overlap = max(0.0, min(end, interval.end) - max(start, interval.start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = interval.speaker_id
    return best_speaker


def build_turns(
    whisper_segments: Iterable,
    intervals: list[SpeakerInterval],
    clean_fn: Callable[[str], str],
    max_gap: float = 1.3,
) -> list[dict]:
    words: list[dict] = []

    for segment in whisper_segments:
        added_word = False
        if segment.words:
            for word in segment.words:
                if word.start is None or word.end is None:
                    continue
                words.append({"start": float(word.start), "end": float(word.end), "text": word.word})
                added_word = True
        if not added_word:
            words.append({"start": float(segment.start), "end": float(segment.end), "text": segment.text})

    turns: list[dict] = []
    current: dict | None = None

    for word in words:
        speaker_id = speaker_for_range(word["start"], word["end"], intervals)
        if (
            current is not None
            and current["speaker_id"] == speaker_id
            and word["start"] - current["end"] <= max_gap
        ):
            current["end"] = word["end"]
            current["raw_text"] += word["text"]
        else:
            if current is not None:
                current["raw_text"] = current["raw_text"].strip()
                current["clean_text"] = clean_fn(current["raw_text"])
                turns.append(current)
            current = {
                "start": word["start"],
                "end": word["end"],
                "speaker_id": speaker_id,
                "raw_text": word["text"],
            }

    if current is not None:
        current["raw_text"] = current["raw_text"].strip()
        current["clean_text"] = clean_fn(current["raw_text"])
        turns.append(current)

    return turns
