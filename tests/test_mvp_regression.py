from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json
import os

from app.align import build_turns
from app.metadata import extract_metadata
from app.participants import map_participants
from app.render import apply_names, write_outputs
from app.text_utils import clean_fillers
from app.types import SpeakerInterval


def word(start: float, end: float, text: str):
    return SimpleNamespace(start=start, end=end, word=text)


def segment(start: float, end: float, text: str, words: list):
    return SimpleNamespace(start=start, end=end, text=text, words=words)


def test_post_model_flow_maps_names_and_preserves_raw(tmp_path: Path) -> None:
    segments = [
        segment(0.0, 2.0, " Yo soy Armando Reyes.", [
            word(0.0, 0.25, " Yo"), word(0.25, 0.5, " soy"),
            word(0.5, 1.1, " Armando"), word(1.1, 1.8, " Reyes."),
        ]),
        segment(2.1, 4.1, " Yo soy Jimena Hernández.", [
            word(2.1, 2.35, " Yo"), word(2.35, 2.6, " soy"),
            word(2.6, 3.2, " Jimena"), word(3.2, 4.0, " Hernández."),
        ]),
        segment(5.0, 7.0, " eh este proyecto avanza avanza.", [
            word(5.0, 5.2, " eh"), word(5.2, 5.55, " este"),
            word(5.55, 6.2, " proyecto"), word(6.2, 6.55, " avanza"),
            word(6.55, 6.95, " avanza."),
        ]),
    ]
    intervals = [
        SpeakerInterval(0.0, 2.0, "SPEAKER_00"),
        SpeakerInterval(2.0, 4.5, "SPEAKER_01"),
        SpeakerInterval(4.5, 8.0, "SPEAKER_00"),
    ]

    clean = lambda text: clean_fillers(text, ["eh"], collapse_repetitions=True)
    turns = build_turns(segments, intervals, clean)
    mapping = map_participants(turns, intro_window_seconds=10)
    named_turns = apply_names(turns, mapping)

    assert mapping["SPEAKER_00"]["name"] == "Armando Reyes"
    assert mapping["SPEAKER_01"]["name"] == "Jimena Hernández"
    assert named_turns[0]["speaker_label"] == "Armando Reyes"
    assert named_turns[1]["speaker_label"] == "Jimena Hernández"
    assert "eh" in named_turns[2]["raw_text"].lower()
    assert "este proyecto" in named_turns[2]["clean_text"].lower()
    assert "avanza avanza" not in named_turns[2]["clean_text"].lower()


def test_metadata_falls_back_to_file_mtime_and_outputs_are_auditable(tmp_path: Path) -> None:
    audio = tmp_path / "meeting.wav"
    audio.write_bytes(b"fixture")
    os.utime(audio, (1_700_000_000, 1_700_000_000))

    turns = [
        {
            "start": 0.0,
            "end": 3.0,
            "speaker_id": "SPEAKER_00",
            "raw_text": "Sesión de Proyecto Aurora. Yo soy Armando Reyes.",
            "clean_text": "Sesión de Proyecto Aurora. Yo soy Armando Reyes.",
            "speaker_name": "Armando Reyes",
            "speaker_label": "Armando Reyes",
            "identity_source": "self_introduction",
            "identity_confidence": 1.0,
        }
    ]
    mapping = {
        "SPEAKER_00": {
            "name": "Armando Reyes",
            "source": "self_introduction",
            "confidence": 1.0,
        }
    }

    metadata = extract_metadata(
        audio_path=audio,
        turns=turns,
        participant_map=mapping,
        intro_window_seconds=180,
        timezone_name="America/Mexico_City",
        language="es",
        whisper_model="large-v3",
        diarization_model="pyannote/speaker-diarization-community-1",
    )
    assert metadata["datetime_source"] == "file_mtime"
    assert metadata["session_title"] == "Proyecto Aurora"
    assert metadata["participants"][0]["name_source"] == "self_introduction"

    out = tmp_path / "output"
    raw_whisper = [{"start": 0.0, "end": 3.0, "text": turns[0]["raw_text"]}]
    write_outputs(out, metadata, turns, raw_whisper, {"markdown": True, "srt": True})

    assert (out / "metadata.json").exists()
    assert (out / "transcript.json").exists()
    assert (out / "transcript.md").exists()
    assert (out / "transcript.srt").exists()

    transcript = json.loads((out / "transcript.json").read_text(encoding="utf-8"))
    assert transcript["whisper_segments"][0]["text"] == turns[0]["raw_text"]
    assert transcript["turns"][0]["speaker_id"] == "SPEAKER_00"
