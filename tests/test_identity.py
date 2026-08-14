from app.identity import enforce_unique_voice_profiles


def test_one_profile_cannot_be_assigned_to_two_speakers() -> None:
    resolutions = {
        "SPEAKER_00": {
            "name": "Armando Reyes",
            "profile_id": "profile-armando",
            "source": "voice_profile",
            "cosine_distance": 0.08,
        },
        "SPEAKER_01": {
            "name": "Armando Reyes",
            "profile_id": "profile-armando",
            "source": "voice_profile",
            "cosine_distance": 0.17,
        },
    }

    accepted = enforce_unique_voice_profiles(resolutions)
    assert set(accepted) == {"SPEAKER_00"}
