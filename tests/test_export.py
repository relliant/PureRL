from types import SimpleNamespace

import onnx
import torch
from purerl.rl import export_feedforward_policy


def test_feedforward_policy_exports_jit_and_onnx(tmp_path):
    policy = SimpleNamespace(
        actor=torch.nn.Linear(4, 2),
        actor_obs_normalizer=torch.nn.Identity(),
        state_dependent_std=False,
        is_recurrent=False,
    )
    exported = export_feedforward_policy(policy, tmp_path, observation_dim=4)

    observation = torch.ones((3, 4))
    jit_policy = torch.jit.load(str(exported["jit"]))
    assert torch.allclose(jit_policy(observation), policy.actor(observation))
    onnx.checker.check_model(onnx.load(exported["onnx"]))
