from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import math
import os

from .config import load_config
from .types import SpeakerInterval


def normalize_embedding(values) -> list[float]:
    vector = [float(value) for value in values]
    norm = math.sqrt(sum(value * value for value in vector))
    if not vector or norm == 0.0 or not math.isfinite(norm):
        raise ValueError("Speaker embedding is empty or invalid")
    return [value / norm for value in vector]


def cosine_distance(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Embeddings must have the same non-zero dimension")
    a = normalize_embedding(left)
    b = normalize_embedding(right)
    similarity = max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b))))
    return 1.0 - similarity


class VoiceProfileStore:
    def __init__(self, path: Path):
        self.path = path

    @staticmethod
    def _key(name: str) -> str:
        normalized = " ".join(name.split()).casefold()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    def _load(self) -> dict:
        if not self.path.exists():
            return {"version": 1, "profiles": {}}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if data.get("version") != 1 or not isinstance(data.get("profiles"), dict):
            raise ValueError(f"Unsupported voice profile store: {self.path}")
        return data

    def _save(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self.path)

    def list_profiles(self) -> list[dict]:
        profiles = list(self._load()["profiles"].values())
        return sorted(profiles, key=lambda item: item["name"].casefold())

    def upsert(self, name: str, embedding: list[float]) -> dict:
        display_name = " ".join(name.split())
        if not display_name:
            raise ValueError("Participant name cannot be empty")
        data = self._load()
        key = self._key(display_name)
        profile = {
            "id": key,
            "name": display_name,
            "embedding": normalize_embedding(embedding),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        data["profiles"][key] = profile
        self._save(data)
        return profile

    def delete(self, name: str) -> bool:
        data = self._load()
        removed = data["profiles"].pop(self._key(name), None) is not None
        if removed:
            self._save(data)
        return removed

    def best_match(
        self,
        embedding: list[float],
        max_distance: float,
        min_margin: float,
    ) -> dict | None:
        ranked = []
        for profile in self.list_profiles():
            distance = cosine_distance(embedding, profile["embedding"])
            ranked.append((distance, profile))
        ranked.sort(key=lambda item: item[0])
        if not ranked or ranked[0][0] > max_distance:
            return None
        if len(ranked) > 1 and ranked[1][0] - ranked[0][0] < min_margin:
            return None
        distance, profile = ranked[0]
        return {
            "name": profile["name"],
            "source": "voice_profile",
            "confidence": round(max(0.0, 1.0 - distance), 6),
            "cosine_distance": round(distance, 6),
            "profile_id": profile["id"],
        }


class SpeakerEmbedder:
    def __init__(self, model_path: Path, device: str):
        checkpoint = model_path / "pytorch_model.bin"
        if not checkpoint.is_file():
            raise RuntimeError(
                f"Voice embedding checkpoint not found at {checkpoint}. "
                "Accept the pyannote/embedding conditions and rerun model bootstrap."
            )
        import torch
        from pyannote.audio import Inference, Model
        from pyannote.core import Segment

        self.Segment = Segment
        model = Model.from_pretrained(str(checkpoint))
        self.inference = Inference(model, window="whole")
        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("DEVICE=cuda was requested but CUDA is not available")
            self.inference.to(torch.device("cuda"))

    @staticmethod
    def _flatten(embedding) -> list[float]:
        return normalize_embedding(embedding.reshape(-1).tolist())

    def whole_file(self, audio_path: Path) -> list[float]:
        return self._flatten(self.inference(str(audio_path)))

    def crop(self, audio_path: Path, start: float, end: float) -> list[float]:
        return self._flatten(self.inference.crop(str(audio_path), self.Segment(start, end)))


def resolve_voice_profiles(
    audio_path: Path,
    intervals: list[SpeakerInterval],
    store: VoiceProfileStore,
    embedder: SpeakerEmbedder,
    cfg: dict,
) -> dict[str, dict]:
    profiles = store.list_profiles()
    if not profiles:
        return {}

    grouped: dict[str, list[SpeakerInterval]] = defaultdict(list)
    min_seconds = float(cfg.get("min_segment_seconds", 1.5))
    max_seconds = float(cfg.get("max_seconds_per_speaker", 30))
    for interval in intervals:
        if interval.end - interval.start >= min_seconds:
            grouped[interval.speaker_id].append(interval)

    resolved: dict[str, dict] = {}
    for speaker_id, speaker_intervals in grouped.items():
        vectors: list[list[float]] = []
        durations: list[float] = []
        used = 0.0
        for interval in sorted(speaker_intervals, key=lambda item: item.end - item.start, reverse=True):
            if used >= max_seconds:
                break
            end = min(interval.end, interval.start + (max_seconds - used))
            duration = end - interval.start
            if duration < min_seconds:
                continue
            vectors.append(embedder.crop(audio_path, interval.start, end))
            durations.append(duration)
            used += duration

        if not vectors:
            continue
        dimension = len(vectors[0])
        total_duration = sum(durations)
        average = [
            sum(vector[index] * duration for vector, duration in zip(vectors, durations)) / total_duration
            for index in range(dimension)
        ]
        match = store.best_match(
            average,
            max_distance=float(cfg.get("max_cosine_distance", 0.25)),
            min_margin=float(cfg.get("min_distance_margin", 0.05)),
        )
        if match:
            resolved[speaker_id] = match
    return resolved


def _store_from_config(cfg: dict) -> VoiceProfileStore:
    return VoiceProfileStore(Path(cfg["voice_profiles"].get("store_path", "/app/data/profiles/profiles.json")))


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage LocalScribe local voice profiles")
    subparsers = parser.add_subparsers(dest="command", required=True)

    enroll = subparsers.add_parser("enroll", help="Create or replace a voice profile")
    enroll.add_argument("--name", required=True)
    enroll.add_argument("--audio", required=True, type=Path)
    subparsers.add_parser("list", help="List registered profiles")
    delete = subparsers.add_parser("delete", help="Delete a registered profile")
    delete.add_argument("--name", required=True)

    args = parser.parse_args()
    cfg = load_config()
    store = _store_from_config(cfg)

    if args.command == "list":
        for profile in store.list_profiles():
            print(f"{profile['name']}\t{profile['id']}\t{profile['updated_at']}")
        return
    if args.command == "delete":
        if not store.delete(args.name):
            raise SystemExit(f"No voice profile found for {args.name}")
        print(f"Deleted voice profile: {args.name}")
        return

    if not args.audio.is_file():
        raise SystemExit(f"Enrollment audio not found: {args.audio}")
    runtime = cfg["runtime"]
    embedder = SpeakerEmbedder(Path(runtime["voice_embedding_model_path"]), runtime["device"])
    profile = store.upsert(args.name, embedder.whole_file(args.audio))
    print(f"Registered voice profile: {profile['name']} ({profile['id']})")


if __name__ == "__main__":
    main()
