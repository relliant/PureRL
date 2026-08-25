import numpy as np
import torch
from purerl.sensors import ContactHistory, HeightFieldSampler, height_observation, make_grid_pattern


def test_height_scan_pattern_has_stable_187_point_order():
    pattern = make_grid_pattern()
    assert pattern.shape == (187, 3)
    assert np.allclose(pattern[0], [-0.8, -0.5, 0.0])
    assert np.allclose(pattern[-1], [0.8, 0.5, 0.0])


def test_height_observation_uses_root_relative_clipped_values():
    root_height = np.asarray([0.9, 2.0])
    hits = np.stack((np.zeros(187), np.zeros(187)))
    observation = height_observation(root_height, hits)
    assert observation.shape == (2, 187)
    assert np.allclose(observation[0], 0.4)
    assert np.allclose(observation[1], 1.0)


def test_height_field_sampler_matches_triangle_mesh_interpolation():
    sampler = HeightFieldSampler(
        np.asarray([[[0.0, 1.0], [2.0, 4.0]]], dtype=np.float32),
        np.asarray([[1.0, 1.0, 0.0]], dtype=np.float32),
        np.asarray([0]),
        size=(2.0, 2.0),
        horizontal_scale=1.0,
        device="cpu",
    )
    points = torch.tensor([[[0.75, 0.25], [0.25, 0.75]]])

    heights = sampler.sample(points)

    assert torch.allclose(heights, torch.tensor([[2.0, 1.5]]))

    sampler.update_env_tiles(torch.tensor([0]), torch.tensor([0]))
    assert torch.equal(sampler.env_tile_indices, torch.tensor([0]))


def test_contact_history_tracks_first_contact_and_air_time():
    history = ContactHistory(np.zeros((1, 2)), force_threshold=1.0)
    no_contact = np.zeros((1, 2, 3))
    left_contact = no_contact.copy()
    left_contact[0, 0, 2] = 10.0

    history.update(no_contact, 0.1)
    history.update(left_contact, 0.1)
    assert history.first_contact.tolist() == [[True, False]]
    assert np.allclose(history.last_air_time, [[0.2, 0.0]])
    assert np.allclose(history.current_air_time, [[0.0, 0.2]])
    assert np.allclose(history.current_contact_time, [[0.1, 0.0]])

    history.update(left_contact, 0.1)
    assert history.first_contact.tolist() == [[False, False]]
    assert history.contact_events.tolist() == [[True, False]]
    assert np.allclose(history.current_contact_time, [[0.2, 0.0]])
    history.clear_events()
    assert history.contact_events.tolist() == [[False, False]]
    history.reset(np.asarray([0]))
    assert np.allclose(history.current_air_time, 0.0)
    assert np.allclose(history.current_contact_time, 0.0)
