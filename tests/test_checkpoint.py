from pathlib import Path

import pytest
import torch
from purerl.rl import (
    find_checkpoint,
    read_checkpoint_mean_noise_std,
    validate_checkpoint_noise_std,
)


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


def test_find_checkpoint_sorts_iteration_numbers_naturally(tmp_path: Path):
    run = tmp_path / "run_10"
    run.mkdir()
    (run / "model_99.pt").touch()
    expected = run / "model_100.pt"
    expected.touch()

    assert find_checkpoint(tmp_path, load_run="run_.*") == expected


def test_checkpoint_noise_guard_rejects_degenerate_policy(tmp_path: Path):
    checkpoint = tmp_path / "model.pt"
    torch.save({"model_state_dict": {"std": torch.full((20,), 12.0)}}, checkpoint)

    assert read_checkpoint_mean_noise_std(checkpoint) == 12.0
    with pytest.raises(ValueError, match="above the safe limit"):
        validate_checkpoint_noise_std(checkpoint, maximum=3.0)


def test_checkpoint_noise_guard_accepts_log_standard_deviation(tmp_path: Path):
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {"model_state_dict": {"log_std": torch.full((20,), torch.log(torch.tensor(0.5)))}},
        checkpoint,
    )

    assert read_checkpoint_mean_noise_std(checkpoint) == pytest.approx(0.5)
