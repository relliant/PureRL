"""Shared helpers for immutable PureRL configurations."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass, replace
from pathlib import Path
from typing import Any, TypeVar

ConfigT = TypeVar("ConfigT", bound="ConfigMixin")


def _serialize(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _serialize(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    return value


class ConfigMixin:
    """Serialization and immutable update operations for dataclass configs."""

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    def replace(self: ConfigT, **changes: Any) -> ConfigT:
        return replace(self, **changes)

    def to_yaml(self, path: str | Path) -> None:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise RuntimeError("PyYAML is required to write configuration files") from exc

        with Path(path).open("w", encoding="utf-8") as stream:
            yaml.safe_dump(self.to_dict(), stream, sort_keys=False)
