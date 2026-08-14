from __future__ import annotations

from pathlib import Path
import logging
import os
import signal
import time
import traceback

from watchdog.events import FileSystemEventHandler
from watchdog.observers.polling import PollingObserver

from .config import load_config, paths
from .pipeline import LocalScribePipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
log = logging.getLogger("localscribe")


def wait_until_stable(path: Path, stable_seconds: float, poll_seconds: float = 1.0) -> bool:
    signature: tuple[int, int] | None = None
    stable_since: float | None = None

    while path.exists():
        stat = path.stat()
        current = (stat.st_size, stat.st_mtime_ns)
        now = time.monotonic()
        if current == signature:
            stable_since = stable_since or now
            if now - stable_since >= stable_seconds:
                return True
        else:
            signature = current
            stable_since = None
        time.sleep(poll_seconds)
    return False


def unique_target(directory: Path, name: str) -> Path:
    target = directory / name
    if not target.exists():
        return target

    source = Path(name)
    for index in range(2, 100_000):
        candidate = directory / f"{source.stem}-{index}{source.suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not allocate unique target for {name}")


class AudioHandler(FileSystemEventHandler):
    def __init__(self, cfg: dict, pipeline: LocalScribePipeline, directories: dict[str, Path]):
        self.cfg = cfg
        self.pipeline = pipeline
        self.directories = directories
        self.extensions = {ext.lower() for ext in cfg["watcher"].get("extensions", [])}
        self.processing_now: set[str] = set()

    def on_created(self, event) -> None:
        if not event.is_directory:
            self._maybe_process(Path(event.src_path))

    def on_moved(self, event) -> None:
        if not event.is_directory:
            self._maybe_process(Path(event.dest_path))

    def _maybe_process(self, source: Path) -> None:
        if source.suffix.lower() not in self.extensions or str(source) in self.processing_now:
            return
        self.processing_now.add(str(source))
        try:
            self._process(source)
        finally:
            self.processing_now.discard(str(source))

    def _process(self, source: Path) -> None:
        stable_seconds = float(self.cfg["watcher"].get("stable_seconds", 4))
        if not wait_until_stable(source, stable_seconds):
            return

        processing_path = unique_target(self.directories["processing"], source.name)
        try:
            os.replace(source, processing_path)
        except FileNotFoundError:
            # Another worker/process already claimed the same source path.
            return

        output_dir = unique_target(self.directories["output"], processing_path.stem)
        try:
            output_dir.mkdir(parents=True, exist_ok=False)
            metadata = self.pipeline.process(processing_path, output_dir)
            completed_path = unique_target(self.directories["completed"], processing_path.name)
            os.replace(processing_path, completed_path)
            log.info(
                "Completed %s -> %s (%s)",
                source.name,
                output_dir,
                metadata.get("session_title") or "sin título",
            )
        except Exception as exc:
            log.exception("Failed to process %s", processing_path.name)
            failed_path = unique_target(self.directories["failed"], processing_path.name)
            if processing_path.exists():
                os.replace(processing_path, failed_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "ERROR.txt").write_text(
                f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}",
                encoding="utf-8",
            )


def main() -> None:
    cfg = load_config()
    directories = paths()
    pipeline = LocalScribePipeline(cfg)
    handler = AudioHandler(cfg, pipeline, directories)

    observer = PollingObserver(timeout=float(cfg["watcher"].get("poll_seconds", 2)))
    observer.schedule(handler, str(directories["input"]), recursive=False)
    observer.start()
    log.info("Watching %s", directories["input"])

    for existing in sorted(directories["input"].iterdir()):
        if existing.is_file():
            handler._maybe_process(existing)

    stopping = False

    def shutdown(*_) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    try:
        while not stopping:
            time.sleep(1)
    finally:
        observer.stop()
        observer.join()


if __name__ == "__main__":
    main()
