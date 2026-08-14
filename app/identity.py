from __future__ import annotations


def enforce_unique_voice_profiles(resolutions: dict[str, dict]) -> dict[str, dict]:
    """Ensure one saved voice profile is assigned to at most one diarized speaker."""
    ranked = sorted(
        resolutions.items(),
        key=lambda item: (
            float(item[1].get("cosine_distance", 1.0)),
            item[0],
        ),
    )
    accepted: dict[str, dict] = {}
    used_profiles: set[str] = set()
    for speaker_id, resolution in ranked:
        profile_id = resolution.get("profile_id")
        if not profile_id or profile_id in used_profiles:
            continue
        used_profiles.add(profile_id)
        accepted[speaker_id] = resolution
    return accepted
