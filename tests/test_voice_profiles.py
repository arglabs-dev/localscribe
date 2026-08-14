from __future__ import annotations

from pathlib import Path

from app.voice_profiles import VoiceProfileStore, cosine_distance


def test_profile_store_upsert_replace_list_and_delete(tmp_path: Path) -> None:
    store = VoiceProfileStore(tmp_path / "profiles.json")
    first = store.upsert("Armando Reyes", [1.0, 0.0, 0.0])
    replacement = store.upsert("  Armando   Reyes  ", [0.99, 0.01, 0.0])

    profiles = store.list_profiles()
    assert len(profiles) == 1
    assert first["id"] == replacement["id"]
    assert profiles[0]["name"] == "Armando Reyes"
    assert store.delete("armando reyes") is True
    assert store.list_profiles() == []
    assert store.delete("Armando Reyes") is False


def test_best_match_requires_threshold_and_unambiguous_margin(tmp_path: Path) -> None:
    store = VoiceProfileStore(tmp_path / "profiles.json")
    store.upsert("Armando Reyes", [1.0, 0.0, 0.0])
    store.upsert("Jimena Hernández", [0.0, 1.0, 0.0])

    match = store.best_match([0.99, 0.01, 0.0], max_distance=0.25, min_margin=0.05)
    assert match is not None
    assert match["name"] == "Armando Reyes"
    assert match["source"] == "voice_profile"

    assert store.best_match([0.0, 0.0, 1.0], max_distance=0.25, min_margin=0.05) is None

    ambiguous_store = VoiceProfileStore(tmp_path / "ambiguous.json")
    ambiguous_store.upsert("Persona Uno", [1.0, 0.0])
    ambiguous_store.upsert("Persona Dos", [0.999, 0.045])
    assert ambiguous_store.best_match([1.0, 0.01], max_distance=0.25, min_margin=0.05) is None


def test_cosine_distance_is_scale_invariant() -> None:
    assert cosine_distance([1.0, 0.0], [10.0, 0.0]) < 1e-9
