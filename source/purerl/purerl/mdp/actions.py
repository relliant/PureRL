"""Joint-position action processing."""

from __future__ import annotations

from typing import Any

from purerl.contracts import ACTION_DIM

from ._array import clip


class JointPositionActionManager:
    """Convert normalized policy actions into articulation position targets."""

    def __init__(
        self,
        default_joint_positions: Any,
        *,
        scale: float = 0.5,
        action_clip: float | None = None,
    ):
        if default_joint_positions.shape[-1] != ACTION_DIM:
            raise ValueError(f"Expected {ACTION_DIM} default joint positions")
        self.default_joint_positions = default_joint_positions
        self.scale = scale
        self.action_clip = action_clip
        self.action = default_joint_positions * 0.0
        self.previous_action = default_joint_positions * 0.0

    def process(self, action: Any) -> Any:
        if action.shape != self.default_joint_positions.shape:
            raise ValueError(
                f"Action shape {action.shape} does not match expected shape {self.default_joint_positions.shape}"
            )
        self.previous_action = self.action
        self.action = (
            action
            if self.action_clip is None
            else clip(action, -self.action_clip, self.action_clip)
        )
        return self.default_joint_positions + self.scale * self.action

    def reset(self, env_ids: Any) -> None:
        self.action[env_ids] = 0.0
        self.previous_action[env_ids] = 0.0
