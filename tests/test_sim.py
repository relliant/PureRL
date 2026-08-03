import importlib.util

import numpy as np
import pytest
from purerl.app import AppLauncherCfg, IsaacSimLauncher
from purerl.config import JOINT_NAMES, make_tienkung_robot_cfg
from purerl.sim import ArticulationIndexMap, make_grid_origins


def test_grid_origins_are_centered_and_stable():
    origins = make_grid_origins(4, 2.0)
    assert origins.shape == (4, 3)
    assert np.allclose(origins.mean(axis=0), 0.0)
    assert len(np.unique(origins, axis=0)) == 4


def test_articulation_index_map_restores_policy_joint_order():
    cfg = make_tienkung_robot_cfg()
    simulator_joints = tuple(reversed(JOINT_NAMES))
    simulator_bodies = ("unrelated", "ankle_roll_r_link", "pelvis", "ankle_roll_l_link")
    mapping = ArticulationIndexMap.resolve(cfg, simulator_joints, simulator_bodies)

    assert tuple(simulator_joints[index] for index in mapping.joint_indices) == JOINT_NAMES
    assert mapping.root_body_index == 2
    assert mapping.foot_body_indices == (3, 1)


def test_articulation_index_map_rejects_missing_schema():
    with pytest.raises(ValueError, match="missing bodies"):
        ArticulationIndexMap.resolve(make_tienkung_robot_cfg(), JOINT_NAMES, ("pelvis",))


def test_launcher_reports_missing_isaac_sim_without_isaac_lab_fallback():
    if importlib.util.find_spec("isaacsim") is not None:
        pytest.skip("Isaac Sim is installed; application startup is covered by simulator integration tests")
    launcher = IsaacSimLauncher(AppLauncherCfg())
    with pytest.raises(RuntimeError, match="Isaac Sim 5.1"):
        launcher.launch()
