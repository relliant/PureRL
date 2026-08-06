from types import SimpleNamespace

import pytest
import torch
from purerl.rl import clamp_policy_noise, install_policy_noise_bounds


def test_clamp_policy_noise_supports_scalar_standard_deviation():
    policy = SimpleNamespace(std=torch.nn.Parameter(torch.tensor([-1.0, 1.0, 8.0])))

    clamp_policy_noise(policy, minimum=0.05, maximum=3.0)

    assert policy.std.tolist() == pytest.approx([0.05, 1.0, 3.0])


def test_optimizer_hook_keeps_log_standard_deviation_bounded():
    policy = SimpleNamespace(log_std=torch.nn.Parameter(torch.zeros(2)))
    optimizer = torch.optim.SGD([policy.log_std], lr=10.0)
    hook = install_policy_noise_bounds(policy, optimizer, minimum=0.1, maximum=2.0)

    optimizer.zero_grad()
    (-policy.log_std.sum()).backward()
    optimizer.step()

    assert policy.log_std.exp().tolist() == pytest.approx([2.0, 2.0])
    hook.remove()
