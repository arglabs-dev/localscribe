from __future__ import annotations

from dataclasses import dataclass, field


def parse_progress_step(value: object, default: int = 5) -> int:
    """Return a validated integer percentage step in the inclusive range 1..100."""
    candidate = default if value is None else value
    if isinstance(candidate, bool):
        raise ValueError("progress_log_interval_percent must be an integer between 1 and 100")

    try:
        numeric = float(candidate)
    except (TypeError, ValueError) as exc:
        raise ValueError("progress_log_interval_percent must be an integer between 1 and 100") from exc

    if not numeric.is_integer():
        raise ValueError("progress_log_interval_percent must be an integer between 1 and 100")

    step = int(numeric)
    if not 1 <= step <= 100:
        raise ValueError("progress_log_interval_percent must be an integer between 1 and 100")
    return step


@dataclass
class PercentageProgress:
    """Track percentage thresholds without emitting duplicate milestones."""

    step: int
    next_percent: int = field(init=False)
    complete_logged: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.step = parse_progress_step(self.step)
        self.next_percent = self.step

    def advance(self, current: float, total: float) -> list[int]:
        if total <= 0 or self.complete_logged:
            return []

        current = max(0.0, current)
        percent = min(99, int((current / total) * 100))
        reached: list[int] = []

        while self.next_percent <= percent:
            reached.append(self.next_percent)
            self.next_percent += self.step

        return reached

    def complete(self) -> list[int]:
        if self.complete_logged:
            return []
        self.complete_logged = True
        return [100]
