from __future__ import annotations

import importlib
from typing import Any


def load_symbol(path: str) -> Any:
    value = (path or "").strip()
    if not value:
        raise ValueError("Empty symbol path")

    module_name, sep, attr = value.rpartition(".")
    if not sep:
        raise ValueError(f"Invalid symbol path '{path}'. Expected 'module.symbol'.")

    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise AttributeError(f"Symbol '{attr}' not found in module '{module_name}'") from exc


def clean_config_dict(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {k: v for k, v in value.items() if not str(k).startswith("_")}


def resolve_ref(config: dict[str, Any], ref: Any) -> Any:
    if not isinstance(ref, str) or not ref.startswith("$"):
        return ref

    cursor: Any = config
    for key in ref[1:].split("."):
        if not isinstance(cursor, dict) or key not in cursor:
            raise KeyError(f"Reference '{ref}' not found in config")
        cursor = cursor[key]
    return cursor


def resolve_mapping(config: dict[str, Any], mapping: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(mapping, dict):
        return {}
    return {key: resolve_ref(config, value) for key, value in mapping.items()}
