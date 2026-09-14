"""Behavioral regressions for the short-hop policy diagnosed on 2026-09-14."""

import numpy as np
import pytest
from purerl.config import make_flat_env_cfg
from purerl.mdp.rewards import base_height, feet_air_time_on_contact, feet_clearance, feet_contact_number
from purerl.sensors import BipedContactHistory


@pytest.fixture(params=["numpy", "torch"])
def array(request):
    if request.param == "torch":
        return pytest.importorskip("torch").as_tensor
    return np.asarray


def test_full_support_pattern_rejects_hops_wrong_phase_and_brief_unloading(array):
    contact = array([[True, False], [True, True], [False, False], [False, True], [True, False]])
    stance = array([[True, False]] * 5)
    air = array([[0.0, 0.15]] * 4 + [[0.0, 0.025]])
    support = array([[0.2, 0.0]] * 5)
    reward = feet_contact_number(contact, stance, current_air_time=air, current_contact_time=support)
    assert reward.tolist() == [1.0, -1.0, -1.0, -1.0, 0.0]
    # A real double-support transition remains valid.
    reward = feet_contact_number(
        array([[True, True]]),
        array([[True, True]]),
        current_air_time=array([[0.0, 0.0]]),
        current_contact_time=array([[0.1, 0.2]]),
    )
    assert reward.tolist() == [1.0]


def test_clearance_needs_lift_and_sustained_opposite_support(array):
    heights = array([[0.0, 0.04], [0.04, 0.04], [0.0, 0.04], [0.0, 0.002], [0.0, 0.0]])
    target = array([[0.0, 0.04]] * 5)
    contact = array([[True, False], [False, False], [True, False], [True, False], [True, True]])
    times = array([[0.2, 0.0], [0.0, 0.0], [0.01, 0.0], [0.2, 0.0], [0.2, 0.2]])
    reward = feet_clearance(heights, target, contact=contact, current_contact_time=times)
    assert float(reward[0]) == pytest.approx(1.0)
    assert float(reward[1]) == float(reward[2]) == float(reward[4]) == 0.0
    assert 0.0 < float(reward[3]) < 0.01


def test_ground_reference_penalizes_body_hop_and_translates_with_terrain(array):
    root = array([0.89, 0.95, 1.39])
    ground = array([0.0, 0.0, 0.5])
    reward = base_height(root, ground, target=0.89)
    assert float(reward[0]) == pytest.approx(1.0)
    assert float(reward[1]) < 0.25
    assert float(reward[2]) == pytest.approx(float(reward[0]))


def advance(history, array, contact, steps):
    forces = np.zeros((1, 2, 3), dtype=np.float32)
    forces[0, :, 2] = np.asarray(contact) * 100.0
    for _ in range(steps):
        history.update(array(forces), 0.005)


def landing_reward(history, array):
    return feet_air_time_on_contact(
        history.valid_landing_air_time,
        history.contact_events,
        array([[0.5, 0.0, 0.0]]),
        supported_landing=history.valid_landing_events,
        threshold=0.12,
    )


def test_landing_requires_alternation_and_support_throughout_swing(array):
    history = BipedContactHistory(array(np.zeros((1, 2), dtype=np.float32)))
    advance(history, array, [1, 1], 10)
    history.clear_events()
    advance(history, array, [0, 1], 40)
    advance(history, array, [1, 1], 1)
    assert float(landing_reward(history, array)[0]) > 0.07
    # A brief contact bounce in the same policy interval cannot overwrite the
    # completed valid swing with a near-zero last_air_time.
    advance(history, array, [0, 1], 2)
    advance(history, array, [1, 1], 1)
    assert float(landing_reward(history, array)[0]) > 0.07
    history.clear_events()
    assert float(landing_reward(history, array)[0]) == 0.0
    advance(history, array, [0, 1], 40)
    advance(history, array, [1, 1], 1)
    assert float(landing_reward(history, array)[0]) == 0.0  # same foot again
    history.clear_events()
    advance(history, array, [1, 0], 40)
    advance(history, array, [1, 1], 1)
    assert float(landing_reward(history, array)[0]) > 0.07


@pytest.mark.parametrize("stagger", [0, 1, 10])
def test_hop_landing_is_not_a_step_even_when_feet_land_at_different_times(array, stagger):
    history = BipedContactHistory(array(np.zeros((1, 2), dtype=np.float32)))
    advance(history, array, [1, 1], 10)
    history.clear_events()
    advance(history, array, [0, 0], 40)
    advance(history, array, [1, 0], stagger)
    history.clear_events()  # still invalid if touchdown straddles policy steps
    advance(history, array, [1, 1], 1)
    assert float(landing_reward(history, array)[0]) == 0.0


def test_dropout_filter_and_flight_penalty_keep_physics_substeps(array):
    history = BipedContactHistory(array(np.zeros((1, 2), dtype=np.float32)))
    advance(history, array, [1, 1], 10)
    history.clear_events()
    advance(history, array, [0, 1], 1)
    assert history.in_contact.tolist() == [[True, True]]
    advance(history, array, [1, 1], 1)
    assert history.contact_events.tolist() == [[False, False]]
    assert float(landing_reward(history, array)[0]) == 0.0
    history.clear_events()
    advance(history, array, [0, 0], 3)
    advance(history, array, [1, 1], 1)  # sampled policy-end contact alone misses the hop
    assert history.in_contact.tolist() == [[True, True]]
    assert float(history.step_flight_time[0]) == pytest.approx(0.015)
    assert float(history.step_unsupported_time[0]) == pytest.approx(0.005)
    history.clear_events()
    assert float(history.step_unsupported_time[0]) == 0.0


def test_selective_reset_clears_gait_events_timers_and_alternation(array):
    history = BipedContactHistory(array(np.zeros((2, 2), dtype=np.float32)))
    history.update(array(np.zeros((2, 2, 3), dtype=np.float32)), 0.2)
    history.last_landed_foot[:, 0] = True
    history.valid_landing_events[:, 0] = True
    history.valid_landing_air_time[:] = 0.2
    history.reset([0])
    assert history.last_landed_foot.tolist() == [[False, False], [True, False]]
    assert history.valid_landing_events.tolist() == [[False, False], [True, False]]
    assert float(history.flight_time[0]) == float(history.valid_landing_air_time[0, 0]) == 0.0
    assert float(history.flight_time[1]) > 0.0


@pytest.mark.parametrize(
    "changes",
    [
        {"double_support_fraction": 1.0},
        {"min_phase_time": 0.0},
        {"contact_release_time": 0.1},
        {"air_time_threshold": 0.3},
        {"min_clearance": 0.0},
        {"flight_grace_time": -0.1},
        {"cycle_time": float("nan")},
    ],
)
def test_invalid_gait_timing_is_rejected(changes):
    cfg = make_flat_env_cfg()
    with pytest.raises(ValueError):
        cfg.replace(gait=cfg.gait.replace(**changes)).validate()
