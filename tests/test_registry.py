import pytest
from purerl.contracts import TASK_IDS
from purerl.registry import get_task_spec, list_task_ids


def test_registry_preserves_public_task_ids_and_variants():
    assert list_task_ids() == TASK_IDS

    for task_id in TASK_IDS:
        spec = get_task_spec(task_id)
        cfg = spec.make_env_cfg()
        assert cfg.play == ("-Play-" in task_id)
        assert cfg.task_kind == ("flat" if "-Flat-" in task_id else "rough")


def test_unknown_task_reports_available_ids():
    with pytest.raises(KeyError, match="Available tasks"):
        get_task_spec("missing-task")
