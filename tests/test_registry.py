import pytest
import yaml
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


def test_registry_accepts_custom_environment_and_runner_configs(tmp_path):
    spec = get_task_spec(TASK_IDS[0])
    env_data = spec.make_env_cfg().to_dict()
    runner_data = spec.make_runner_cfg().to_dict()
    env_data["scene"]["num_envs"] = 7
    runner_data["max_iterations"] = 3
    env_path = tmp_path / "env.yaml"
    runner_path = tmp_path / "runner.yaml"

    env_path.write_text(yaml.safe_dump(env_data, sort_keys=False), encoding="utf-8")
    runner_path.write_text(yaml.safe_dump(runner_data, sort_keys=False), encoding="utf-8")

    assert spec.make_env_cfg(env_path).scene.num_envs == 7
    assert spec.make_runner_cfg(runner_path).max_iterations == 3
