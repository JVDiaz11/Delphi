"""
core/registry.py
----------------
Manages a persistent ``models_index.json`` catalogue of every training run
produced by any project built on Delphi.

Schema for each entry
~~~~~~~~~~~~~~~~~~~~~
{
    "name":            str            – human-readable label
    "run_dir":         str            – absolute path to the run directory
    "checkpoint_best": str | null     – absolute path to checkpoint_best.pt
    "checkpoint_last": str | null     – absolute path to checkpoint_last.pt
    "config_path":     str            – absolute path to the config JSON used
    "model_type":      str            – consumer-defined capability family
    "trained_at":      str            – ISO-8601 UTC timestamp
    "epochs":          int
    "best_val_loss":   float | null
    "architecture":    dict           – model.architecture_summary() or {}
    "metadata":        dict           – project-specific free-form bag
                                        (replaces the old hash-specific top-level
                                        fields: token_mode, inference_mode,
                                        difficulty_target, …)
}

Migration note
~~~~~~~~~~~~~~
Delphi's ``register_model()`` calls should pass a ``metadata`` dict
containing all hash-specific fields::

    register_model(make_entry(
        ...
        metadata={
            "token_mode": "hex",
            "inference_mode": "guided",
            "difficulty_target": "00000",
        },
    ))
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

# Default registry path: <project root>/runs/models_index.json
_DEFAULT_REGISTRY = Path.cwd() / "runs" / "models_index.json"


# ---------------------------------------------------------------------------
# Low-level I/O
# ---------------------------------------------------------------------------

def registry_path() -> Path:
    """Return the default path to models_index.json."""
    return _DEFAULT_REGISTRY


def load_registry(path: Path | None = None) -> list[dict[str, Any]]:
    """Load and return all entries.  Returns ``[]`` if the file does not exist."""
    target = Path(path) if path is not None else _DEFAULT_REGISTRY
    if not target.exists():
        return []
    with target.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, list) else []


def save_registry(entries: list[dict[str, Any]], path: Path | None = None) -> None:
    """Write *entries* back to the index file (pretty-printed)."""
    target = Path(path) if path is not None else _DEFAULT_REGISTRY
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        json.dump(entries, fh, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Entry helpers
# ---------------------------------------------------------------------------

def register_model(entry: dict[str, Any], path: Path | None = None) -> None:
    """Upsert *entry* in the registry, keyed by ``run_dir``.

    An existing entry with the same ``run_dir`` is replaced; otherwise the
    new entry is appended.
    """
    entries = load_registry(path)
    run_dir = str(entry.get("run_dir", ""))
    entries = [e for e in entries if str(e.get("run_dir", "")) != run_dir]
    entries.append(entry)
    save_registry(entries, path)


def make_entry(
    *,
    name: str,
    run_dir: Path,
    config_path: Path,
    model_type: str = "default",
    epochs: int,
    best_val_loss: float | None,
    architecture: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    trained_at: str | None = None,
) -> dict[str, Any]:
    """Build a registry entry dict.

    ``metadata`` is a free-form dict for any project-specific attributes
    (e.g. inference_mode, difficulty_target, token_mode for Delphi).
    """
    run_dir = Path(run_dir).resolve()
    checkpoint_best = run_dir / "checkpoint_best.pt"
    checkpoint_last = run_dir / "checkpoint_last.pt"
    return {
        "name": name,
        "run_dir": str(run_dir),
        "checkpoint_best": str(checkpoint_best) if checkpoint_best.exists() else None,
        "checkpoint_last": str(checkpoint_last) if checkpoint_last.exists() else None,
        "config_path": str(Path(config_path).resolve()),
        "model_type": str(model_type).lower(),
        "trained_at": trained_at or datetime.now(timezone.utc).isoformat(),
        "epochs": int(epochs),
        "best_val_loss": best_val_loss,
        "architecture": architecture or {},
        "metadata": metadata or {},
    }


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_runs(
    runs_root: Path,
    path: Path | None = None,
    *,
    type_fn: Callable[[dict[str, Any]], str] | None = None,
) -> list[dict[str, Any]]:
    """Scan *runs_root* for checkpoint files not yet in the registry.

    For each ``checkpoint_best.pt`` or ``checkpoint_last.pt`` found whose
    parent directory is not already registered, a minimal stub entry is
    created and registered.  Returns the list of newly discovered entries.

    Parameters
    ----------
    type_fn:
        Optional callable ``(config: dict) -> str`` that infers
        ``model_type`` from the run's ``param_used.json``.  If not
        provided, ``model_type`` defaults to ``"unknown"``.
    """
    runs_root = Path(runs_root).resolve()
    if not runs_root.exists():
        return []

    existing = load_registry(path)
    indexed_dirs = {str(Path(e["run_dir"]).resolve()) for e in existing}
    new_entries: list[dict[str, Any]] = []

    for ckpt in sorted(runs_root.rglob("checkpoint_best.pt")):
        run_dir = ckpt.parent.resolve()
        if str(run_dir) in indexed_dirs:
            continue

        param_json = run_dir / "param_used.json"
        config: dict[str, Any] = {}
        if param_json.exists():
            try:
                with param_json.open("r", encoding="utf-8") as fh:
                    config = json.load(fh)
            except Exception:
                pass

        model_type = type_fn(config) if type_fn is not None else "unknown"
        entry = make_entry(
            name=f"{run_dir.parent.name}/{run_dir.name}",
            run_dir=run_dir,
            config_path=param_json if param_json.exists() else run_dir / "config.json",
            model_type=model_type,
            epochs=int(config.get("trainer", {}).get("epochs", 0)),
            best_val_loss=None,
            metadata=config,
        )
        register_model(entry, path)
        indexed_dirs.add(str(run_dir))
        new_entries.append(entry)

    return new_entries


__all__ = [
    "registry_path",
    "load_registry",
    "save_registry",
    "register_model",
    "make_entry",
    "discover_runs",
]
