"""Policy export helpers independent of Isaac Lab."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any


def export_feedforward_policy(
    policy: Any,
    output_dir: str | Path,
    *,
    observation_dim: int,
    export_jit: bool = True,
    export_onnx: bool = True,
) -> dict[str, Path]:
    """Export an RSL-RL feed-forward actor with its observation normalizer."""

    import torch

    if getattr(policy, "is_recurrent", False):
        raise ValueError("Recurrent policy export is not implemented")
    if observation_dim <= 0:
        raise ValueError("observation_dim must be positive")
    if not export_jit and not export_onnx:
        raise ValueError("At least one export format must be enabled")

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    module = _FeedForwardPolicy(policy).cpu().eval()
    example = torch.zeros((1, observation_dim), dtype=torch.float32)
    exported: dict[str, Path] = {}

    if export_jit:
        jit_path = destination / "policy.pt"
        torch.jit.trace(module, example).save(str(jit_path))
        exported["jit"] = jit_path
    if export_onnx:
        onnx_path = destination / "policy.onnx"
        torch.onnx.export(
            module,
            example,
            str(onnx_path),
            input_names=["observation"],
            output_names=["action"],
            dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )
        exported["onnx"] = onnx_path
    return exported


class _FeedForwardPolicy:
    def __new__(cls, policy: Any):
        import torch

        class Module(torch.nn.Module):
            def __init__(self, source: Any):
                super().__init__()
                self.actor = copy.deepcopy(source.actor)
                self.normalizer = copy.deepcopy(source.actor_obs_normalizer)
                self.state_dependent_std = bool(source.state_dependent_std)

            def forward(self, observation):
                action = self.actor(self.normalizer(observation))
                return action[..., 0, :] if self.state_dependent_std else action

        return Module(policy)
