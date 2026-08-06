"""Direct Isaac Sim application launcher with delayed simulator imports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AppLauncherCfg:
    headless: bool = True
    enable_cameras: bool = False
    raytracing_motion: bool = False
    livestream: int = 0
    experience: str | None = None


class IsaacSimLauncher:
    """Own a single Isaac Sim ``SimulationApp`` instance."""

    def __init__(self, cfg: AppLauncherCfg):
        self.cfg = cfg
        self._app: Any | None = None

    @property
    def app(self) -> Any:
        if self._app is None:
            raise RuntimeError("Isaac Sim has not been launched")
        return self._app

    def launch(self) -> Any:
        if self._app is not None:
            raise RuntimeError("Isaac Sim is already running")
        try:
            from isaacsim import SimulationApp
        except ImportError as exc:
            raise RuntimeError(
                "Isaac Sim 5.1 is required. Install the PureRL simulation dependencies; "
                "an Isaac Lab checkout is not required."
            ) from exc

        settings = {
            "headless": self.cfg.headless,
            "enable_cameras": self.cfg.enable_cameras,
            "livestream": self.cfg.livestream,
            "extra_args": (
                ["--/renderer/raytracingMotion/enabled=true"]
                if self.cfg.raytracing_motion
                else []
            ),
        }
        kwargs = {"experience": self.cfg.experience} if self.cfg.experience else {}
        self._app = SimulationApp(settings, **kwargs)
        return self._app

    def is_running(self) -> bool:
        return self._app is not None and bool(self._app.is_running())

    def close(self) -> None:
        if self._app is not None:
            self._app.close()
            self._app = None

    def __enter__(self) -> "IsaacSimLauncher":
        self.launch()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()
