"""Shared helpers for immutable PureRL configurations."""

from __future__ import annotations

import types
from dataclasses import asdict, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

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


def _type_name(annotation: Any) -> str:
    return getattr(annotation, "__name__", str(annotation))


def _deserialize(
    value: Any,
    annotation: Any,
    *,
    base_dir: Path,
    field_path: str,
    require_complete: bool,
) -> Any:
    if annotation is Any:
        return value

    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in (types.UnionType, Union):
        if value is None and type(None) in args:
            return None
        errors = []
        for candidate in (item for item in args if item is not type(None)):
            try:
                return _deserialize(
                    value,
                    candidate,
                    base_dir=base_dir,
                    field_path=field_path,
                    require_complete=require_complete,
                )
            except (TypeError, ValueError) as exc:
                errors.append(str(exc))
        raise TypeError(
            f"{field_path} does not match {_type_name(annotation)}: " + "; ".join(errors)
        )

    if isinstance(annotation, type) and is_dataclass(annotation):
        if not isinstance(value, dict):
            raise TypeError(f"{field_path} must be a mapping")
        return _dataclass_from_dict(
            annotation,
            value,
            base_dir=base_dir,
            field_path=field_path,
            require_complete=require_complete,
        )

    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"{field_path} must be a sequence")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(
                _deserialize(
                    item,
                    args[0],
                    base_dir=base_dir,
                    field_path=f"{field_path}[{index}]",
                    require_complete=require_complete,
                )
                for index, item in enumerate(value)
            )
        if len(value) != len(args):
            raise ValueError(f"{field_path} must contain exactly {len(args)} values")
        return tuple(
            _deserialize(
                item,
                item_type,
                base_dir=base_dir,
                field_path=f"{field_path}[{index}]",
                require_complete=require_complete,
            )
            for index, (item, item_type) in enumerate(zip(value, args, strict=True))
        )

    if origin is list:
        if not isinstance(value, list):
            raise TypeError(f"{field_path} must be a list")
        item_type = args[0] if args else Any
        return [
            _deserialize(
                item,
                item_type,
                base_dir=base_dir,
                field_path=f"{field_path}[{index}]",
                require_complete=require_complete,
            )
            for index, item in enumerate(value)
        ]

    if origin is dict:
        if not isinstance(value, dict):
            raise TypeError(f"{field_path} must be a mapping")
        key_type, item_type = args or (Any, Any)
        return {
            _deserialize(
                key,
                key_type,
                base_dir=base_dir,
                field_path=f"{field_path}.<key>",
                require_complete=require_complete,
            ): _deserialize(
                item,
                item_type,
                base_dir=base_dir,
                field_path=f"{field_path}.{key}",
                require_complete=require_complete,
            )
            for key, item in value.items()
        }

    if annotation is Path:
        if not isinstance(value, (str, Path)):
            raise TypeError(f"{field_path} must be a filesystem path")
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (base_dir / path).resolve()

    if annotation is bool:
        if not isinstance(value, bool):
            raise TypeError(f"{field_path} must be a bool")
        return value
    if annotation is int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{field_path} must be an int")
        return value
    if annotation is float:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"{field_path} must be a float")
        return float(value)
    if annotation is str:
        if not isinstance(value, str):
            raise TypeError(f"{field_path} must be a string")
        return value
    if annotation is type(None):
        if value is not None:
            raise TypeError(f"{field_path} must be null")
        return None
    if isinstance(annotation, type) and isinstance(value, annotation):
        return value
    raise TypeError(f"Unsupported configuration type {_type_name(annotation)} at {field_path}")


def _dataclass_from_dict(
    config_type: type[ConfigT],
    data: dict[str, Any],
    *,
    base_dir: Path,
    field_path: str,
    require_complete: bool,
) -> ConfigT:
    config_fields = {item.name: item for item in fields(config_type)}
    unknown = sorted(set(data) - set(config_fields))
    if unknown:
        raise ValueError(f"Unknown field(s) in {field_path}: {', '.join(unknown)}")
    missing = sorted(set(config_fields) - set(data))
    if require_complete and missing:
        raise ValueError(f"Missing field(s) in {field_path}: {', '.join(missing)}")
    hints = get_type_hints(config_type)
    values = {
        name: _deserialize(
            value,
            hints[name],
            base_dir=base_dir,
            field_path=f"{field_path}.{name}",
            require_complete=require_complete,
        )
        for name, value in data.items()
    }
    try:
        return config_type(**values)
    except TypeError as exc:
        raise TypeError(f"Invalid {field_path}: {exc}") from exc


class ConfigMixin:
    """Serialization and immutable update operations for dataclass configs."""

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)

    @classmethod
    def from_dict(
        cls: type[ConfigT],
        data: dict[str, Any],
        *,
        base_dir: str | Path | None = None,
        require_complete: bool = True,
    ) -> ConfigT:
        if not isinstance(data, dict):
            raise TypeError(f"{cls.__name__} configuration must be a mapping")
        resolved_base = Path.cwd() if base_dir is None else Path(base_dir).expanduser().resolve()
        return _dataclass_from_dict(
            cls,
            data,
            base_dir=resolved_base,
            field_path=cls.__name__,
            require_complete=require_complete,
        )

    @classmethod
    def from_yaml(
        cls: type[ConfigT], path: str | Path, *, require_complete: bool = True
    ) -> ConfigT:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise RuntimeError("PyYAML is required to read configuration files") from exc

        config_path = Path(path).expanduser().resolve()
        with config_path.open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
        if not isinstance(data, dict):
            raise TypeError(f"{config_path} must contain a top-level mapping")
        return cls.from_dict(
            data,
            base_dir=config_path.parent,
            require_complete=require_complete,
        )

    def replace(self: ConfigT, **changes: Any) -> ConfigT:
        return replace(self, **changes)

    def to_yaml(self, path: str | Path) -> None:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise RuntimeError("PyYAML is required to write configuration files") from exc

        with Path(path).open("w", encoding="utf-8") as stream:
            yaml.safe_dump(self.to_dict(), stream, sort_keys=False)
