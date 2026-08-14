from __future__ import annotations

import re

NAME_CUE = re.compile(
    r"\b(?:yo\s+soy|mi\s+nombre\s+es|me\s+llamo)\s+"
    r"([A-Za-zÁÉÍÓÚÜÑáéíóúüñ'’-]+(?:\s+[A-Za-zÁÉÍÓÚÜÑáéíóúüñ'’-]+){0,4})",
    re.IGNORECASE,
)

STOP_WORDS = {
    "y", "pero", "entonces", "vamos", "bueno", "buenas", "este", "eh",
    "de", "del", "para", "porque", "que", "quien", "una", "un", "la", "el",
}
BAD_FIRST = {"una", "un", "la", "el", "parte", "responsable", "quien", "de", "del"}


def _normalize_name(candidate: str) -> str | None:
    tokens = [token.strip(" ,.;:!?") for token in candidate.split()]
    kept: list[str] = []
    for token in tokens:
        low = token.lower()
        if kept and low in STOP_WORDS:
            break
        kept.append(token)

    if not kept or kept[0].lower() in BAD_FIRST:
        return None

    return " ".join(
        part if any(ch.isupper() for ch in part[1:]) else part[:1].upper() + part[1:].lower()
        for part in kept[:4]
    )


def map_participants(turns: list[dict], intro_window_seconds: float) -> dict[str, dict]:
    """Resolve speaker identities from explicit self-introductions in the intro window."""
    mapping: dict[str, dict] = {}

    for turn in turns:
        if turn["start"] > intro_window_seconds:
            break
        for match in NAME_CUE.finditer(turn["raw_text"]):
            name = _normalize_name(match.group(1))
            if name:
                mapping.setdefault(
                    turn["speaker_id"],
                    {"name": name, "source": "self_introduction", "confidence": 1.0},
                )
                break

    return mapping
