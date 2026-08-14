from __future__ import annotations

from pathlib import Path
import json

import yaml


def ts(seconds: float, srt: bool = False) -> str:
    milliseconds = int(round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    separator = "," if srt else "."
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def apply_names(turns: list[dict], participant_map: dict[str, dict]) -> list[dict]:
    enriched: list[dict] = []
    for turn in turns:
        item = dict(turn)
        resolution = participant_map.get(turn["speaker_id"])
        item["speaker_name"] = resolution["name"] if resolution else None
        item["speaker_label"] = item["speaker_name"] or turn["speaker_id"]
        item["identity_source"] = resolution["source"] if resolution else "unresolved"
        item["identity_confidence"] = resolution.get("confidence") if resolution else None
        enriched.append(item)
    return enriched


def write_outputs(
    out_dir: Path,
    metadata: dict,
    turns: list[dict],
    raw_whisper: list[dict],
    config: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    transcript_payload = {
        "metadata": metadata,
        "turns": turns,
        "whisper_segments": raw_whisper,
    }
    (out_dir / "transcript.json").write_text(
        json.dumps(transcript_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if config.get("markdown", True):
        frontmatter = {
            "source_audio": metadata["source_audio"],
            "session_datetime": metadata["session_datetime"],
            "datetime_source": metadata["datetime_source"],
            "session_title": metadata.get("session_title"),
            "participants": [participant["label"] for participant in metadata.get("participants", [])],
            "language": metadata.get("language"),
        }
        lines = [
            "---",
            yaml.safe_dump(frontmatter, allow_unicode=True, sort_keys=False).strip(),
            "---",
            "",
            f"# {metadata.get('session_title') or 'Transcripción de sesión'}",
            "",
        ]
        for turn in turns:
            lines.append(
                f"**[{ts(turn['start'])}–{ts(turn['end'])}] {turn['speaker_label']}:** "
                f"{turn['clean_text']}"
            )
            lines.append("")
        (out_dir / "transcript.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    if config.get("srt", True):
        blocks: list[str] = []
        for index, turn in enumerate(turns, 1):
            blocks.extend(
                [
                    str(index),
                    f"{ts(turn['start'], True)} --> {ts(turn['end'], True)}",
                    f"{turn['speaker_label']}: {turn['clean_text']}",
                    "",
                ]
            )
        (out_dir / "transcript.srt").write_text("\n".join(blocks), encoding="utf-8")
