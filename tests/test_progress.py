import pytest

from app.progress import PercentageProgress, parse_progress_step


def test_progress_emits_each_configured_threshold_once() -> None:
    progress = PercentageProgress(5)

    assert progress.advance(4, 100) == []
    assert progress.advance(5, 100) == [5]
    assert progress.advance(17, 100) == [10, 15]
    assert progress.advance(17, 100) == []
    assert progress.advance(99, 100) == list(range(20, 100, 5))
    assert progress.complete() == [100]
    assert progress.complete() == []


def test_progress_supports_different_interval() -> None:
    progress = PercentageProgress(10)

    assert progress.advance(35, 100) == [10, 20, 30]
    assert progress.advance(81, 100) == [40, 50, 60, 70, 80]
    assert progress.complete() == [100]


def test_progress_handles_missing_duration_until_completion() -> None:
    progress = PercentageProgress(5)

    assert progress.advance(10, 0) == []
    assert progress.complete() == [100]


@pytest.mark.parametrize("value", [0, 101, 2.5, True, "bad"])
def test_progress_step_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValueError):
        parse_progress_step(value)


def test_progress_step_uses_default_and_accepts_numeric_string() -> None:
    assert parse_progress_step(None) == 5
    assert parse_progress_step("10") == 10
