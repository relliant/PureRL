import json
from pathlib import Path

from purerl.config import JOINT_NAMES, make_flat_env_cfg, make_rough_env_cfg
from purerl.contracts import (
    ACTION_DIM,
    DECIMATION,
    FOOT_BODY_NAMES,
    OBSERVATION_DIM,
    PHYSICS_DT,
    ROOT_BODY_NAME,
    TASK_IDS,
    observation_slices,
)

FIXTURE = Path(__file__).parent / "fixtures" / "locomotion_contract.json"


def test_static_contract_fixture_matches_local_configuration():
    contract = json.loads(FIXTURE.read_text(encoding="utf-8"))
    rough = make_rough_env_cfg()
    flat = make_flat_env_cfg()

    assert contract["simulator_reference_available"] is False
    assert contract["action"] == {"dimension": ACTION_DIM, "scale": rough.actions.scale}
    assert contract["simulation"] == {
        "physics_dt": PHYSICS_DT,
        "decimation": DECIMATION,
        "step_dt": rough.sim.step_dt,
        "episode_length_s": rough.episode_length_s,
    }
    assert contract["observation"]["dimension"] == OBSERVATION_DIM
    assert contract["root_body"] == ROOT_BODY_NAME
    assert tuple(contract["foot_bodies"]) == FOOT_BODY_NAMES
    assert tuple(contract["task_ids"]) == TASK_IDS
    assert rough.observations.dimension == flat.observations.dimension == OBSERVATION_DIM
    assert len(JOINT_NAMES) == ACTION_DIM


def test_observation_layout_matches_checkpoint_contract():
    contract = json.loads(FIXTURE.read_text(encoding="utf-8"))
    layout = observation_slices()

    actual = [[name, term_slice.start, term_slice.stop] for name, term_slice in layout.items()]
    assert actual == contract["observation"]["terms"]
