"""Stable mapping from simulator articulation names to policy order."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from purerl.config.robot import RobotCfg


@dataclass(frozen=True)
class ArticulationIndexMap:
    joint_indices: tuple[int, ...]
    root_body_index: int
    foot_body_indices: tuple[int, ...]

    @classmethod
    def resolve(
        cls,
        cfg: RobotCfg,
        simulator_joint_names: Sequence[str],
        simulator_body_names: Sequence[str],
    ) -> "ArticulationIndexMap":
        _require_unique("simulator joint", simulator_joint_names)
        _require_unique("simulator body", simulator_body_names)
        joint_lookup = {name: index for index, name in enumerate(simulator_joint_names)}
        body_lookup = {name: index for index, name in enumerate(simulator_body_names)}

        missing_joints = [name for name in cfg.joint_names if name not in joint_lookup]
        required_bodies = (cfg.root_body_name, *cfg.foot_body_names)
        missing_bodies = [name for name in required_bodies if name not in body_lookup]
        if missing_joints or missing_bodies:
            details = []
            if missing_joints:
                details.append(f"missing joints: {', '.join(missing_joints)}")
            if missing_bodies:
                details.append(f"missing bodies: {', '.join(missing_bodies)}")
            raise ValueError("Articulation schema mismatch (" + "; ".join(details) + ")")

        return cls(
            joint_indices=tuple(joint_lookup[name] for name in cfg.joint_names),
            root_body_index=body_lookup[cfg.root_body_name],
            foot_body_indices=tuple(body_lookup[name] for name in cfg.foot_body_names),
        )


def _require_unique(kind: str, names: Sequence[str]) -> None:
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate {kind} names are not supported")
