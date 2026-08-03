from pathlib import Path

import pytest
from purerl.rl import find_checkpoint


def test_find_checkpoint_selects_latest_matching_run_and_file(tmp_path: Path):
    older = tmp_path / "2026-01-01_00-00-00"
    newer = tmp_path / "2026-01-02_00-00-00"
    older.mkdir()
    newer.mkdir()
    (older / "model_100.pt").touch()
    (newer / "model_100.pt").touch()
    expected = newer / "model_200.pt"
    expected.touch()

    assert find_checkpoint(tmp_path) == expected


def test_find_checkpoint_reports_missing_match(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="No run"):
        find_checkpoint(tmp_path)
