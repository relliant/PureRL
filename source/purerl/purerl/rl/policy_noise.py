"""Runtime bounds for RSL-RL Gaussian policy exploration noise."""

from __future__ import annotations

import math
from typing import Any


def clamp_policy_noise(policy: Any, *, minimum: float, maximum: float) -> None:
    """Clamp scalar or log-standard-deviation policy parameters in place."""

    if not 0.0 < minimum <= maximum:
        raise ValueError("Policy noise bounds must satisfy 0 < minimum <= maximum")

    import torch

    with torch.no_grad():
        if hasattr(policy, "std"):
            policy.std.clamp_(min=minimum, max=maximum)
        elif hasattr(policy, "log_std"):
            policy.log_std.clamp_(min=math.log(minimum), max=math.log(maximum))
        else:
            raise TypeError("Policy does not expose std or log_std parameters")


def install_policy_noise_bounds(
    policy: Any,
    optimizer: Any,
    *,
    minimum: float,
    maximum: float,
) -> Any:
    """Clamp now and after every optimizer update, returning the hook handle."""

    clamp_policy_noise(policy, minimum=minimum, maximum=maximum)

    def clamp_after_step(_optimizer, _args, _kwargs) -> None:
        clamp_policy_noise(policy, minimum=minimum, maximum=maximum)

    return optimizer.register_step_post_hook(clamp_after_step)
