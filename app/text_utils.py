from __future__ import annotations

import re


def clean_fillers(text: str, fillers: list[str], collapse_repetitions: bool = True) -> str:
    result = text

    for filler in fillers:
        pattern = rf"(?i)(?<!\w){re.escape(filler)}(?!\w)"
        result = re.sub(pattern, " ", result)

    if collapse_repetitions:
        result = re.sub(
            r"(?i)\b([\wáéíóúüñ]+)(?:\s*[,;:]?\s+\1\b)+",
            r"\1",
            result,
        )

    result = re.sub(r"\s+([,.;:!?])", r"\1", result)
    result = re.sub(r"([,;:])\s*([,;:])+", r"\1", result)
    result = re.sub(r"\s{2,}", " ", result)
    return result.strip(" ,;")
